from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Sequence

from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import GuardBuilder
from torch._dynamo.variables.builder import VariableBuilder
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.lists import ListVariable
from torch._dynamo.variables.lists import TupleVariable
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

from helion._compiler.type_propagation import TensorType

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from helion.runtime.kernel import Kernel


_UNSUPPORTED_INPUT_TYPES: dict[type[VariableTracker], str] = {
    TupleVariable: "tuple",
    ListVariable: "list",
    ConstDictVariable: "dict",
}


def _infer_output_spec(
    kernel: Kernel, args: Sequence[VariableTracker]
) -> dict[str, object]:
    """Infer output specification by binding kernel with fake args (single tensor output only)."""
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
            if isinstance(stmt.value, ast.Tuple):
                raise NotImplementedError(
                    f"Helion kernels with {len(stmt.value.elts)} outputs are not yet supported with "
                    "torch.compile fusion. Only single tensor output is supported."
                )
            # Build spec from return value's type info (TensorType.proxy() always returns Tensor)
            type_info = stmt.value._type_info  # type: ignore[attr-defined]
            if not isinstance(type_info, TensorType):
                raise RuntimeError(
                    f"Helion kernels with non-tensor outputs are not supported with torch.compile fusion. "
                    f"Expression `{ast.unparse(stmt.value)}` does not evaluate to a tensor."
                )
            t = type_info.proxy()
            return {
                "shape": list(t.shape),
                "stride": list(t.stride()),
                "dtype": t.dtype,
                "device": str(t.device),
            }

    raise NotImplementedError(
        "Helion kernels with no return value are not yet supported with torch.compile fusion."
    )


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

        # Emit HOP node into FX graph
        hop_proxy = tx.output.create_proxy(
            "call_function",
            helion_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": self._kernel_idx,
                "constant_args": constant_args,
                "tensor_args": ConstDictVariable(tensor_args, dict).as_proxy(),
                "output_spec": _infer_output_spec(self._kernel, ordered_args),
            },
        )

        # Wrap proxy and extract first element (single tensor output)
        result = wrap_fx_proxy(tx, hop_proxy)
        return result.call_method(
            tx, "__getitem__", [variables.ConstantVariable.create(0)], {}
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
