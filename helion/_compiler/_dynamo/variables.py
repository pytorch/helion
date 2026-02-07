from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING
from typing import Sequence
from typing import cast

from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import GuardBuilder
from torch._dynamo.variables.builder import VariableBuilder
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.lists import ListVariable
from torch._dynamo.variables.lists import TupleVariable

from helion._compat import requires_torch_version
from helion._compiler.type_propagation import TensorType
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


_UNSUPPORTED_INPUT_TYPES: dict[type[VariableTracker], str] = {
    TupleVariable: "tuple",
    ListVariable: "list",
    ConstDictVariable: "dict",
}


def _infer_output_spec(
    kernel: Kernel, args: Sequence[VariableTracker]
) -> dict[str, object]:
    """Infer output specification by binding kernel with fake args."""
    # Check for unsupported container parameter types
    for name, arg in zip(kernel.signature.parameters.keys(), args, strict=False):
        if (arg_type := type(arg)) in _UNSUPPORTED_INPUT_TYPES:
            type_name = _UNSUPPORTED_INPUT_TYPES[arg_type]
            raise RuntimeError(
                f"Helion kernels with {type_name.title()} input arguments are not supported with torch.compile fusion. "
                f"Input argument '{name}' is a {type_name}."
            )

    # Bind kernel with arg values to get type info
    bound = kernel.bind(
        tuple(
            v.as_python_constant()
            if v.is_python_constant()
            else v.as_proxy().node.meta.get("example_value")
            for v in args
        )
    )
    assert bound.host_function, "kernel.bind() succeeded but host_function is None"

    # Find return statement and build output spec
    for stmt in reversed(bound.host_function.body):
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            if isinstance(stmt.value, ast.List):
                raise RuntimeError(
                    "Helion kernels with list returns are not supported with "
                    "torch.compile fusion. Use a tuple return instead."
                )
            elts = (
                stmt.value.elts if isinstance(stmt.value, ast.Tuple) else [stmt.value]
            )
            if isinstance(stmt.value, ast.Tuple) and len(elts) == 1:
                raise RuntimeError(
                    "Helion kernels returning a single-element tuple are not supported "
                    "with torch.compile fusion. Use `return x` instead of `return (x,)`."
                )
            for elt in elts:
                if isinstance(elt, (ast.Tuple, ast.List)):
                    raise RuntimeError(
                        "Helion kernels with nested tuple/list returns are not supported "
                        "with torch.compile fusion."
                    )
            output_specs: list[dict[str, object]] = []
            for node in elts:
                type_info = getattr(node, "_type_info", None)
                if type_info is None:
                    raise RuntimeError(
                        f"Expression `{ast.unparse(node)}` has no type info"
                    )
                if isinstance(type_info, TensorType):
                    t = type_info.proxy()
                    output_specs.append(
                        {
                            "type": "tensor",
                            "shape": list(t.shape),
                            "stride": list(t.stride()),
                            "dtype": t.dtype,
                            "device": str(t.device),
                        }
                    )
                    continue
                t = type_info.proxy()
                if isinstance(t, (int, float, bool)):
                    output_specs.append({"type": "scalar", "scalar_value": t})
                else:
                    raise RuntimeError(
                        f"Helion kernels with {type(t).__name__} return type are not supported "
                        f"with torch.compile fusion. "
                        f"Expression `{ast.unparse(node)}` does not evaluate to a tensor or scalar."
                    )

            if output_specs and all(s["type"] == "scalar" for s in output_specs):
                raise RuntimeError(
                    "Helion kernels must return at least one tensor with "
                    "torch.compile fusion. Returning only scalars is not supported."
                )

            return {"output_specs": output_specs}

    raise NotImplementedError(
        "Helion kernels with no return value are not yet supported with torch.compile fusion."
    )


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

        # Emit HOP node into FX graph
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
        # Wrap proxy and extract outputs (tensors and scalars)
        result = wrap_fx_proxy(tx, hop_proxy)
        specs = cast("list[dict[str, object]]", output_spec["output_specs"])

        def _extract_output(i: int) -> VariableTracker:
            if specs[i]["type"] == "scalar":
                return variables.ConstantVariable.create(specs[i]["scalar_value"])
            return result.call_method(
                tx, "__getitem__", [variables.ConstantVariable.create(i)], {}
            )

        outputs = [_extract_output(i) for i in range(len(specs))]
        return TupleVariable(outputs) if len(specs) > 1 else outputs[0]


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
