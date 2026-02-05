from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Sequence
from typing import cast

import torch
from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import GuardBuilder
from torch._dynamo.variables.builder import VariableBuilder
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.lists import ListVariable
from torch._dynamo.variables.lists import TupleVariable
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from helion.runtime.kernel import Kernel


_UNSUPPORTED_INPUT_TYPES: dict[type[VariableTracker], str] = {
    TupleVariable: "tuple",
    ListVariable: "list",
    ConstDictVariable: "dict",
}


def _find_alias_name(
    tensor: torch.Tensor, param_tensors: dict[str, torch.Tensor]
) -> tuple[str | None, bool]:
    """Find if tensor aliases any param. Returns (name, is_direct_alias)."""
    for name, pt in param_tensors.items():
        if tensor.untyped_storage() is pt.untyped_storage():
            is_direct = (
                tensor.storage_offset() == pt.storage_offset()
                and list(tensor.shape) == list(pt.shape)
                and list(tensor.stride()) == list(pt.stride())
            )
            return name, is_direct
    return None, False


def _detect_mutated_inputs(body: list[ast.stmt], param_names: set[str]) -> list[str]:
    """Find params mutated via subscript assignment (e.g. x[tile] = ...)."""
    mutated: list[str] = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        target = None
        if isinstance(node, ast.Assign):
            target = node.targets[0] if node.targets else None
        elif isinstance(node, ast.AugAssign):
            target = node.target
        if isinstance(target, ast.Subscript):
            base = target.value
            while isinstance(base, ast.Subscript):
                base = base.value
            if (
                isinstance(base, ast.Name)
                and base.id in param_names
                and base.id not in mutated
            ):
                mutated.append(base.id)
    return mutated


def _build_output_info(
    node: ast.expr, param_tensors: dict[str, torch.Tensor]
) -> tuple[dict[str, object] | None, str | None, bool]:
    """Build output spec for a return expression. Returns (spec, alias_name, is_direct)."""
    type_info = getattr(node, "_type_info", None)
    if type_info is None:
        raise RuntimeError(f"Expression `{ast.unparse(node)}` has no type info")
    t = type_info.proxy()
    if isinstance(t, torch.Tensor):
        alias, is_direct = _find_alias_name(t, param_tensors)
        if alias is None and isinstance(node, ast.Name) and node.id in param_tensors:
            alias, pt = node.id, param_tensors[node.id]
            is_direct = list(t.shape) == list(pt.shape) and list(t.stride()) == list(
                pt.stride()
            )
        return (
            {
                "shape": list(t.shape),
                "stride": list(t.stride()),
                "dtype": t.dtype,
                "device": str(t.device),
            },
            alias,
            is_direct,
        )
    if isinstance(t, (int, float, bool)):
        return {"scalar_value": t}, None, False
    raise RuntimeError(
        f"Unsupported return type {type(t).__name__} for `{ast.unparse(node)}`"
    )


def _infer_output_spec(
    kernel: Kernel, args: Sequence[VariableTracker]
) -> dict[str, object]:
    """Infer output specification by binding kernel with fake args."""
    # Check for unsupported container parameter types
    names = list(kernel.signature.parameters.keys())
    for name, arg in zip(names, args, strict=False):
        if (arg_type := type(arg)) in _UNSUPPORTED_INPUT_TYPES:
            type_name = _UNSUPPORTED_INPUT_TYPES[arg_type]
            raise RuntimeError(
                f"Helion kernels with {type_name.title()} input arguments are not supported with torch.compile fusion. "
                f"Input argument '{name}' is a {type_name}."
            )

    # Bind kernel with arg values to get type info
    fake_args = [
        a.as_python_constant()
        if a.is_python_constant()
        else a.as_proxy().node.meta.get("example_value")
        for a in args
    ]
    param_tensors = {
        n: v
        for n, v in zip(names, fake_args, strict=False)
        if isinstance(v, torch.Tensor)
    }
    bound = kernel.bind(tuple(fake_args))
    assert bound.host_function, "kernel.bind() succeeded but host_function is None"

    # Find return statement and build output spec
    num_outputs, return_nodes = -1, []
    for stmt in reversed(bound.host_function.body):
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            if isinstance(stmt.value, ast.List):
                raise RuntimeError("List returns not supported; use tuple instead")
            if isinstance(stmt.value, ast.Tuple):
                for elt in stmt.value.elts:
                    if isinstance(elt, (ast.Tuple, ast.List)):
                        raise RuntimeError("Nested tuple/list returns not supported")
                num_outputs, return_nodes = len(stmt.value.elts), list(stmt.value.elts)
            else:
                num_outputs, return_nodes = 1, [stmt.value]
            break

    specs, aliases, alias_is_direct = [], [], []
    for node in return_nodes:
        spec, alias, is_direct = _build_output_info(node, param_tensors)
        specs.append(spec)
        aliases.append(alias)
        alias_is_direct.append(is_direct)

    mutated = _detect_mutated_inputs(
        bound.host_function.body, set(param_tensors.keys())
    )
    for a in aliases:
        if a and a not in mutated:
            mutated.append(a)

    if num_outputs > 0 and all(s is None or "scalar_value" in s for s in specs):
        raise ValueError("Must return at least one tensor, not just scalars")

    return {
        "num_outputs": num_outputs,
        "output_specs": specs,
        "output_aliases": aliases,
        "output_alias_is_direct": alias_is_direct,
        "mutated_inputs": mutated,
    }


def _build_return_value(
    tx: InstructionTranslator,
    result: VariableTracker,
    output_spec: dict[str, object],
    param_vars: dict[str, VariableTracker],
) -> VariableTracker:
    """Build return value from HOP result, handling aliasing and scalars."""
    num = cast("int", output_spec["num_outputs"])
    specs = cast("list[dict[str, object] | None]", output_spec["output_specs"])
    aliases = cast("list[str | None]", output_spec["output_aliases"])
    direct = cast("list[bool]", output_spec["output_alias_is_direct"])

    def get(i: int) -> VariableTracker:
        s, a, d = (
            specs[i] if i < len(specs) else None,
            aliases[i] if i < len(aliases) else None,
            direct[i] if i < len(direct) else False,
        )
        if s and "scalar_value" in s:
            return variables.ConstantVariable.create(s["scalar_value"])
        if a and a in param_vars and d:
            return param_vars[a]
        return result.call_method(
            tx, "__getitem__", [variables.ConstantVariable.create(i)], {}
        )

    if num <= 0:
        return variables.ConstantVariable.create(None)
    return TupleVariable([get(i) for i in range(num)]) if num > 1 else get(0)


class HelionKernelVariable(VariableTracker):
    """Variable tracker for Helion kernel objects."""

    def __init__(
        self, kernel: Kernel, kernel_idx: int | None, **kwargs: object
    ) -> None:  # pyrefly: ignore[bad-argument-type]
        super().__init__(**kwargs)  # pyrefly: ignore[bad-argument-type]
        self._kernel = kernel
        self._kernel_idx = (
            kernel_idx
            if kernel_idx is not None
            else kernel_side_table.add_kernel(
                kernel  # pyrefly: ignore[bad-argument-type]
            )
        )

    def call_function(
        self,
        tx: InstructionTranslator,
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """Handle a call to a Helion kernel during Dynamo tracing."""
        # Lazy import: higher_order_ops requires PyTorch >= 2.11 (checked in wrap_helion_kernel)
        from helion._compiler._dynamo.higher_order_ops import (
            helion_kernel_wrapper_mutation,
        )

        sig_params = self._kernel.signature.parameters

        # Map positional args and kwargs to parameter names, partition into constants vs tensors
        param_vars = dict(zip(sig_params.keys(), args, strict=False))
        param_vars.update(kwargs)
        constant_args: dict[str, object] = {}
        tensor_args: dict[VariableTracker, VariableTracker] = {}
        for name, var in param_vars.items():
            if var.is_python_constant():
                constant_args[name] = var.as_python_constant()
            else:
                tensor_args[variables.ConstantVariable.create(name)] = var

        # Build ordered args in signature order (with defaults) for output inference
        ordered_args = [
            param_vars[name]
            if name in param_vars
            else variables.ConstantVariable.create(p.default)
            for name, p in sig_params.items()
            if name in param_vars or p.default is not p.empty
        ]

        output_spec = _infer_output_spec(self._kernel, ordered_args)
        hop_proxy = tx.output.create_proxy(
            "call_function",
            helion_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": self._kernel_idx,
                "constant_args": constant_args,
                "tensor_args": ConstDictVariable(tensor_args, dict).as_proxy(),
                "output_spec": output_spec,
            },
        )
        return _build_return_value(
            tx, wrap_fx_proxy(tx, hop_proxy), output_spec, param_vars
        )


def register_dynamo_variable() -> None:
    """Register HelionKernelVariable with Dynamo's VariableBuilder."""
    from helion._compat import requires_torch_version
    from helion.runtime.kernel import Kernel

    def wrap_helion_kernel(self: VariableBuilder, value: Kernel) -> VariableTracker:
        if value.settings._wip_experimental_allow_torch_compile_fusion:
            if not requires_torch_version("2.11"):
                raise RuntimeError(
                    "Helion kernel torch.compile fusion requires "
                    "PyTorch >= 2.11. Please upgrade PyTorch or set "
                    "_wip_experimental_allow_torch_compile_fusion=False in settings."
                )
            # Import template_buffer to register the HOP's Inductor lowering
            from helion._compiler._inductor import template_buffer  # noqa: F401

            self.install_guards(GuardBuilder.ID_MATCH)
            return HelionKernelVariable(value, None, source=self.source)
        return self.wrap_user_defined(value)

    VariableBuilder._type_dispatch()[Kernel] = wrap_helion_kernel
