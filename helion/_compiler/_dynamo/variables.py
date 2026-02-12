from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING
from typing import Sequence
from typing import cast

import torch
from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import GuardBuilder
from torch._dynamo.variables.builder import VariableBuilder
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.higher_order_ops import OutputSpec as _HopOutputSpec
from torch._dynamo.variables.higher_order_ops import _call_function_and_unflatten_output
from torch._dynamo.variables.lists import ListVariable
from torch._dynamo.variables.lists import TupleVariable
import torch.utils._pytree as pytree

from helion._compat import requires_torch_version
from helion._compiler.ast_read_writes import ReadWrites
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


_UNSUPPORTED_INPUT_TYPES: dict[type[VariableTracker], str] = {
    TupleVariable: "tuple",
    ListVariable: "list",
    ConstDictVariable: "dict",
}


def _detect_mutated_inputs(body: list[ast.stmt], param_names: set[str]) -> list[str]:
    """Find params mutated via subscript assignment (e.g. x[tile] = ...)."""
    rw = ReadWrites.from_list(body)
    return [name for name in rw.writes if name in param_names]


def _validate_return(
    body: list[ast.stmt], return_value: ast.expr, flat_leaves: list[object]
) -> None:
    """Validate return statement for torch.compile fusion compatibility."""
    # Check return not in control flow
    for stmt in body:
        for node in ast.walk(stmt):
            if node is not stmt and isinstance(node, ast.Return):
                raise RuntimeError(
                    "Return statements inside control flow (if/else/for/while) "
                    "are not supported with torch.compile fusion. "
                    "Please restructure the kernel to have a single return at the end."
                )

    # Check shared storage among distinct output tensors
    seen_objs: set[int] = set()
    seen_storages: set[int] = set()
    for leaf in flat_leaves:
        if not isinstance(leaf, torch.Tensor) or id(leaf) in seen_objs:
            continue
        seen_objs.add(id(leaf))
        sid = id(leaf.untyped_storage())
        if sid in seen_storages:
            raise RuntimeError(
                "Returning multiple outputs that share storage is not yet supported "
                "with torch.compile fusion. Please return independent tensors."
            )
        seen_storages.add(sid)


def _detect_output_aliases(
    flat_leaves: list[object],
    bind_param_tensors: dict[str, torch.Tensor],
) -> dict[int, tuple[str, bool]]:
    """Detect output-to-input aliases via tensor identity and storage checks.

    Uses the bound kernel's internal FakeTensors (from host_function.params)
    which share identity/storage with pass-through and view outputs.

    Returns:
        {flat_leaf_index: (input_name, is_direct_alias)} where is_direct_alias
        is True when the output is exactly the input (same tensor object).
    """
    id_to_name = {id(v): n for n, v in bind_param_tensors.items()}
    storage_to_name = {
        id(v.untyped_storage()): n for n, v in bind_param_tensors.items()
    }
    aliases: dict[int, tuple[str, bool]] = {}
    for i, leaf in enumerate(flat_leaves):
        if not isinstance(leaf, torch.Tensor):
            continue
        name = id_to_name.get(id(leaf))
        if name is not None:
            aliases[i] = (name, True)
        else:
            name = storage_to_name.get(id(leaf.untyped_storage()))
            if name is not None:
                aliases[i] = (name, False)
    return aliases


def _get_flat_output(
    host_function: object,
) -> tuple[list[object], pytree.TreeSpec | None, ast.expr | None]:
    """Get flattened output leaves, tree spec, and return AST from a host function."""
    body = getattr(host_function, "body", None)
    if body is None:
        return [], None, None
    for stmt in reversed(body):
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return [], None, None
            type_info = getattr(stmt.value, "_type_info", None)
            if type_info is not None:
                proxy_result = type_info.proxy()
                flat, spec = pytree.tree_flatten(proxy_result)
                return flat, spec, stmt.value
            break
    return [], None, None


def _infer_output_spec(
    kernel: Kernel, args: Sequence[VariableTracker]
) -> dict[str, object]:
    """Infer output specification by binding kernel with fake args."""
    # Check for unsupported container parameter types
    names = list(kernel.signature.parameters.keys())
    for name, arg in zip(names, args, strict=True):
        if (arg_type := type(arg)) in _UNSUPPORTED_INPUT_TYPES:
            type_name = _UNSUPPORTED_INPUT_TYPES[arg_type]
            raise RuntimeError(
                f"{type_name.title()} parameters are not supported with torch.compile fusion. "
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
        for n, v in zip(names, fake_args, strict=True)
        if isinstance(v, torch.Tensor)
    }
    bound = kernel.bind(tuple(fake_args))
    assert bound.host_function, "kernel.bind() succeeded but host_function is None"

    flat_leaves, tree_spec, return_value = _get_flat_output(bound.host_function)

    if tree_spec is None:
        # No return statement: pure mutation kernel (returns None)
        return {
            "leaf_specs": [],
            "tree_spec_str": None,
            "mutated_inputs": _detect_mutated_inputs(
                bound.host_function.body, set(param_tensors.keys())
            ),
        }

    # Validate return structure and shared storage
    assert return_value is not None
    _validate_return(bound.host_function.body, return_value, flat_leaves)

    # Detect aliases using bound kernel's internal FakeTensors
    bind_param_tensors = {
        n: v
        for n, v in bound.host_function.params.arguments.items()
        if isinstance(v, torch.Tensor)
    }
    output_aliases = _detect_output_aliases(flat_leaves, bind_param_tensors)
    direct_aliases = {i: name for i, (name, direct) in output_aliases.items() if direct}

    # Build leaf specs
    leaf_specs: list[dict[str, object]] = []
    for leaf in flat_leaves:
        if isinstance(leaf, torch.Tensor):
            leaf_specs.append(
                {
                    "type": "tensor",
                    "shape": list(leaf.shape),
                    "dtype": leaf.dtype,
                    "device": str(leaf.device),
                }
            )
        elif isinstance(leaf, (int, float, bool)):
            leaf_specs.append({"type": "scalar", "scalar_value": leaf})
        elif isinstance(leaf, torch.SymInt):
            # Only SymInt is supported: SymFloat/SymBool from float/bool
            # parameters are unbacked symbols that size_hint() cannot evaluate.
            # We call shape_env.size_hint() directly (not CompileEnvironment
            # .size_hint()) because it performs full sympy expression evaluation,
            # correctly handling both backed symbols (tensor shapes) and
            # expressions over unbacked symbols (e.g. param * 2) by substituting
            # values registered during kernel.bind().
            # Correctness: Dynamo guards on the inputs, so the evaluated value
            # here matches the runtime value for this compilation.
            hint = bound.env.shape_env.size_hint(leaf.node.expr)
            assert hint is not None
            scalar_value = int(hint)  # pyrefly: ignore[no-matching-overload]
            leaf_specs.append({"type": "scalar", "scalar_value": scalar_value})
        else:
            leaf_name = "None" if leaf is None else type(leaf).__name__
            raise RuntimeError(
                f"Returning {leaf_name} values from a Helion kernel "
                f"is not supported with torch.compile fusion."
            )

    # Detect mutated inputs (subscript writes + aliased outputs)
    mutated = _detect_mutated_inputs(
        bound.host_function.body, set(param_tensors.keys())
    )
    for alias_name in {n for n, _ in output_aliases.values()}:
        if alias_name not in mutated:
            mutated.append(alias_name)

    # Check for aliased/overlapping mutated input tensors
    if len(mutated) > 1:
        storage_to_names: dict[int, list[str]] = {}
        for mut_name in mutated:
            if mut_name in param_tensors:
                sid = id(param_tensors[mut_name].untyped_storage())
                storage_to_names.setdefault(sid, []).append(mut_name)
        for shared_names in storage_to_names.values():
            if len(shared_names) > 1:
                raise RuntimeError(
                    f"torch.compile fusion does not support multiple mutated arguments "
                    f"that share storage ({', '.join(shared_names)}) in a Helion kernel"
                )

    return {
        "leaf_specs": leaf_specs,
        "tree_spec_str": pytree.treespec_dumps(tree_spec),
        "mutated_inputs": mutated,
        "direct_aliases": direct_aliases,
    }


def _replace_direct_aliases(
    result: VariableTracker,
    output_spec: dict[str, object],
    param_vars: dict[str, VariableTracker],
) -> VariableTracker:
    """Replace direct-alias outputs with the original input variables."""
    direct_aliases = cast("dict[int, str]", output_spec.get("direct_aliases", {}))
    replacements = {
        i: param_vars[name] for i, name in direct_aliases.items() if name in param_vars
    }
    if not replacements:
        return result

    # Walk the variable tree, replacing leaves at aliased positions
    counter = [0]

    def walk(vt: VariableTracker) -> VariableTracker:
        if isinstance(vt, (TupleVariable, ListVariable)):
            new_items = [walk(item) for item in vt.items]
            return type(vt)(new_items)
        idx = counter[0]
        counter[0] += 1
        return replacements.get(idx, vt)

    return walk(result)


class HelionKernelVariable(VariableTracker):
    """Variable tracker for Helion kernel objects."""

    def __init__(
        self, kernel: Kernel, kernel_idx: int | None, **kwargs: object
    ) -> None:  # pyrefly: ignore[bad-argument-type]
        from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table

        super().__init__(**kwargs)  # pyrefly: ignore[bad-argument-type]
        self._kernel = kernel
        self._kernel_idx = (
            kernel_idx
            if kernel_idx is not None
            else helion_kernel_side_table.add_kernel(kernel)
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

        # Emit HOP node into FX graph and unflatten output
        output_spec = _infer_output_spec(self._kernel, ordered_args)
        hop_kwargs = {
            "kernel_idx": self._kernel_idx,
            "constant_args": constant_args,
            "tensor_args": ConstDictVariable(tensor_args, dict).as_proxy(),
            "output_spec": output_spec,
        }

        tree_spec_str = cast("str | None", output_spec.get("tree_spec_str"))
        if tree_spec_str is None:
            # Pure mutation kernel: emit HOP for side effects, return None
            tx.output.create_proxy(
                "call_function",
                helion_kernel_wrapper_mutation,
                (),
                hop_kwargs,
            )
            return variables.ConstantVariable.create(None)

        tree_spec = pytree.treespec_loads(tree_spec_str)
        leaf_specs = cast("list[dict[str, object]]", output_spec["leaf_specs"])
        masks = [s["type"] == "scalar" for s in leaf_specs]
        ret_spec = _HopOutputSpec(
            treespec=tree_spec,
            masks_to_filter_const_values=masks if any(masks) else None,
            const_values=[s.get("scalar_value") for s in leaf_specs]
            if any(masks)
            else None,
        )
        result = _call_function_and_unflatten_output(
            tx,
            helion_kernel_wrapper_mutation,
            (),
            hop_kwargs,
            None,
            ret_spec,
            None,
        )
        return _replace_direct_aliases(result, output_spec, param_vars)


def register_dynamo_variable() -> None:
    """Register HelionKernelVariable with Dynamo's VariableBuilder."""

    def wrap_helion_kernel(self: VariableBuilder, value: Kernel) -> VariableTracker:
        if os.environ.get("_WIP_DEV_ONLY_HELION_TORCH_COMPILE_FUSION", "0") == "1":
            if not requires_torch_version("2.11"):
                raise RuntimeError(
                    "Helion kernel torch.compile fusion requires "
                    "PyTorch >= 2.11. Please upgrade PyTorch or unset "
                    "_WIP_DEV_ONLY_HELION_TORCH_COMPILE_FUSION environment variable."
                )
            # Import template_buffer to register the HOP's Inductor lowering
            from helion._compiler._inductor import template_buffer  # noqa: F401

            self.install_guards(GuardBuilder.ID_MATCH)
            return HelionKernelVariable(value, None, source=self.source)
        return self.wrap_user_defined(value)

    VariableBuilder._type_dispatch()[Kernel] = wrap_helion_kernel
