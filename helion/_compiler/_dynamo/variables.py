from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Sequence
from typing import TypedDict

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

from torch._dynamo import exc as _dynamo_exc

from helion._compiler._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from helion._compiler.type_propagation import LiteralType
from helion._compiler.type_propagation import NumericType
from helion._compiler.type_propagation import TensorType
from helion._compiler.type_propagation import TypeInfo

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from helion.runtime.kernel import Kernel


class OutputSpec(TypedDict):
    """Output specification for a Helion kernel call."""

    num_outputs: int
    output_specs: list[dict[str, object] | None]


# =============================================================================
# TypeInfo and VariableTracker Value Extraction
# =============================================================================


def _get_value_from_type_info(
    node: ast.expr,
    local_types: dict[str, TypeInfo],
    param_values: dict[str, object] | None = None,
) -> object:
    """Extract value from AST node's _type_info (set by type propagation)."""
    type_info = getattr(node, "_type_info", None)
    if isinstance(type_info, (TensorType, LiteralType)):
        return type_info.proxy()
    if isinstance(type_info, NumericType):
        # NumericType has symbolic values - evaluate with concrete param_values
        eval_globals: dict[str, object] = {"torch": torch}
        for name, vtype in local_types.items():
            if (fake := vtype.as_tensor()) is not None:
                eval_globals[name] = fake
            elif hasattr(vtype, "value"):
                eval_globals[name] = vtype.value
        if param_values:
            eval_globals.update(
                {
                    k: v
                    for k, v in param_values.items()
                    if not isinstance(v, torch.Tensor)
                }
            )
        try:
            return eval(ast.unparse(node), eval_globals)
        except Exception:
            return None
    return None


def _get_var_value(var: VariableTracker) -> object:
    """Extract value from VariableTracker (constant or proxy).

    Returns the Python constant value, fake tensor, or example_value stored
    in the proxy's FX node metadata, or None if not available.
    """
    if var.is_python_constant():
        return var.as_python_constant()
    if not hasattr(var, "as_proxy"):
        return None
    proxy = var.as_proxy()
    if proxy is not None and hasattr(proxy, "node"):
        return proxy.node.meta.get("example_value")
    return None


# =============================================================================
# AST Parsing Utilities
# =============================================================================


def _parse_return_statement(body: list[ast.stmt]) -> tuple[int, list[ast.expr]]:
    """Parse return statement from kernel body.

    Returns:
        (num_outputs, return_nodes) where num_outputs=-1 if no return found.

    Raises:
        RuntimeError: If return uses list or nested tuples (unsupported).
    """
    for stmt in reversed(body):
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            if isinstance(stmt.value, ast.List):
                raise RuntimeError(
                    "Returning a list from a Helion kernel is not supported with "
                    "torch.compile fusion. Please use a tuple instead: "
                    "`return (a, b)` instead of `return [a, b]`."
                )
            if isinstance(stmt.value, ast.Tuple):
                for elt in stmt.value.elts:
                    if isinstance(elt, (ast.Tuple, ast.List)):
                        raise RuntimeError(
                            "Returning nested tuples or lists from a Helion kernel is not "
                            "supported with torch.compile fusion. Please flatten the return "
                            "value: `return (a, b, c)` instead of `return (a, (b, c))`."
                        )
                return len(stmt.value.elts), list(stmt.value.elts)
            return 1, [stmt.value]
    return -1, []


def _check_unsupported_param_types(
    args: Sequence[VariableTracker],
    param_names: list[str],
) -> None:
    """Check for unsupported parameter types (tuple, list, dict)."""
    unsupported = {TupleVariable: "tuple", ListVariable: "list", ConstDictVariable: "dict"}
    for i, arg in enumerate(args):
        for var_type, type_name in unsupported.items():
            if isinstance(arg, var_type):
                param_name = param_names[i] if i < len(param_names) else f"arg{i}"
                raise RuntimeError(
                    f"{type_name.title()} parameters are not supported with torch.compile "
                    f"fusion. Parameter '{param_name}' is a {type_name}. "
                    f"Please pass tensors as individual parameters instead."
                )


# =============================================================================
# Output Specification Inference
# =============================================================================


def _build_return_value(
    tx: InstructionTranslator,
    result: VariableTracker,
    output_spec: OutputSpec,
) -> VariableTracker:
    """Build the appropriate return value from HOP result."""
    num_outputs = output_spec["num_outputs"]
    output_specs = output_spec["output_specs"]

    def get_output(i: int) -> VariableTracker:
        # Check if this output is a scalar (None, int, float, bool)
        # Scalars are known at compile time and should be returned as constants
        spec = output_specs[i] if i < len(output_specs) else None
        if spec is not None and "scalar_value" in spec:
            return variables.ConstantVariable.create(spec["scalar_value"])
        return result.call_method(
            tx, "__getitem__", [variables.ConstantVariable.create(i)], {}
        )

    if num_outputs <= 0:
        # Kernel has no return statement (returns None implicitly)
        return variables.ConstantVariable.create(None)
    if num_outputs > 1:
        return TupleVariable([get_output(i) for i in range(num_outputs)])
    return get_output(0)


def _build_output_info(
    node: ast.expr,
    local_types: dict[str, TypeInfo],
    param_values: dict[str, object],
) -> dict[str, object] | None:
    """Build output spec for a single return expression."""
    t = _get_value_from_type_info(node, local_types, param_values)

    if isinstance(t, torch.Tensor):
        return {
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "storage_offset": t.storage_offset(),
            "dtype": t.dtype,
            "device": str(t.device),
        }
    elif t is None:
        raise _dynamo_exc.InternalTorchDynamoError(
            f"None return values are not supported with torch.compile fusion. "
            f"Expression `{ast.unparse(node)}` evaluates to None."
        )
    elif isinstance(t, (int, float, bool)):
        return {"scalar_value": t}
    else:
        raise RuntimeError(
            f"Returning {type(t).__name__} values from a Helion kernel is not supported "
            f"with torch.compile fusion. Expression `{ast.unparse(node)}` evaluates to "
            f"{type(t).__name__}. Supported return types: tensor, int, float, bool."
        )


def _infer_output_spec(
    kernel: Kernel,
    args: Sequence[VariableTracker],
) -> OutputSpec:
    """Infer output specification by binding kernel with fake args."""
    param_names = list(kernel.signature.parameters.keys())
    _check_unsupported_param_types(args, param_names)

    fake_args = [_get_var_value(a) for a in args]

    # Build a mapping of parameter names to their values (for evaluating return expressions)
    param_values = dict(zip(param_names, fake_args, strict=False))

    bound = kernel.bind(tuple(fake_args))
    if not bound.host_function:
        raise AssertionError("kernel.bind() succeeded but host_function is None")

    local_types = bound.host_function.local_types or {}

    # Parse return statement to extract return expressions
    num_outputs, return_nodes = _parse_return_statement(bound.host_function.body)

    # Detect return statements inside control flow - not supported with torch.compile fusion.
    if num_outputs == -1:
        has_nested_return = any(
            isinstance(node, ast.Return) and node.value is not None
            for node in ast.walk(
                ast.Module(body=bound.host_function.body, type_ignores=[])
            )
        )
        if has_nested_return:
            raise RuntimeError(
                "Return statements inside control flow (if/else, for, while) are not "
                "supported with torch.compile fusion. Please use a single return statement "
                "at the end of the kernel. Example: `if cond: result = a; else: result = b; "
                "return result` instead of `if cond: return a; else: return b`."
            )

    # Build output specs for each return expression
    output_specs = [
        _build_output_info(node, local_types, param_values)
        for node in return_nodes
    ]

    # Kernel must return at least one tensor (not just scalars)
    if num_outputs > 0 and not any(
        spec is not None and "scalar_value" not in spec for spec in output_specs
    ):
        raise ValueError(
            "torch.compile with Helion kernels does not support kernels that "
            "return only scalars (no tensors). Please return at least one tensor, "
            "or call the kernel outside of torch.compile."
        )

    return OutputSpec(
        num_outputs=num_outputs,
        output_specs=output_specs,
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

        # Infer output specification
        output_spec = _infer_output_spec(self._kernel, ordered_args)

        # Emit HOP node into FX graph
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

        # Wrap proxy and build return value
        result = wrap_fx_proxy(tx, hop_proxy)
        return _build_return_value(tx, result, output_spec)


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
