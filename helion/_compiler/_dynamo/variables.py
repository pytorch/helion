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
    output_aliases: list[str | None]
    output_alias_is_direct: list[bool]
    mutated_inputs: list[str]


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
# Aliasing and Mutation Detection
# =============================================================================


def _tensors_share_storage(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """Check if two tensors share the same underlying storage.

    This works for both real tensors (using data_ptr) and FakeTensors
    (using object identity of storage).
    """
    # Use storage identity comparison which works for both real and fake tensors
    # For real tensors, same storage object means same data_ptr
    # For FakeTensors, same storage object means same meta storage
    return t1.untyped_storage() is t2.untyped_storage()


def _is_direct_alias(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """Check if t1 is a direct alias of t2 (same storage, offset, shape, stride)."""
    if not _tensors_share_storage(t1, t2):
        return False
    return (
        t1.storage_offset() == t2.storage_offset()
        and list(t1.shape) == list(t2.shape)
        and list(t1.stride()) == list(t2.stride())
    )


def _find_alias_name(
    tensor: torch.Tensor,
    param_tensors: dict[str, torch.Tensor],
) -> tuple[str | None, bool]:
    """Find if tensor aliases any parameter tensor.

    Returns:
        (param_name, is_direct) where param_name is the aliased parameter name
        or None if no alias found. is_direct indicates if it's a direct alias
        (same shape/stride/offset) vs indirect (view/slice).
    """
    for name, param_tensor in param_tensors.items():
        if _tensors_share_storage(tensor, param_tensor):
            is_direct = _is_direct_alias(tensor, param_tensor)
            return name, is_direct
    return None, False


def _detect_mutated_inputs_from_ast(
    kernel_body: list[ast.stmt],
    param_names: set[str],
) -> list[str]:
    """Detect mutated inputs by analyzing kernel AST for subscript assignments.

    A parameter is mutated if it appears on the LHS of a subscript assignment:
    - x[tile] = expr
    - x[i, j] = expr

    Returns:
        List of parameter names that are mutated.
    """
    mutated_names: list[str] = []

    class MutationVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                self._check_subscript_target(target)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self._check_subscript_target(node.target)
            self.generic_visit(node)

        def _check_subscript_target(self, target: ast.expr) -> None:
            if isinstance(target, ast.Subscript):
                # Get the base name (handle chained subscripts like x[i][j])
                base = target.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name):
                    if base.id in param_names and base.id not in mutated_names:
                        mutated_names.append(base.id)

    visitor = MutationVisitor()
    for stmt in kernel_body:
        visitor.visit(stmt)

    return mutated_names


def _detect_mutated_inputs(
    kernel: "Kernel",
    param_tensors: dict[str, torch.Tensor],
    local_types: dict[str, TypeInfo],
    kernel_body: list[ast.stmt] | None = None,
) -> list[str]:
    """Detect which input tensors are mutated by the kernel.

    Uses AST analysis to detect subscript assignments to input parameters.

    Returns:
        List of parameter names that are mutated.
    """
    if kernel_body is None:
        return []

    param_names = set(param_tensors.keys())
    return _detect_mutated_inputs_from_ast(kernel_body, param_names)


# =============================================================================
# Output Specification Inference
# =============================================================================


def _build_return_value(
    tx: InstructionTranslator,
    result: VariableTracker,
    output_spec: OutputSpec,
    param_vars: dict[str, VariableTracker],
) -> VariableTracker:
    """Build the appropriate return value from HOP result.

    For aliased outputs, return the original input tensor variable instead of
    extracting from the HOP result. This maintains proper aliasing semantics
    in the traced graph.
    """
    num_outputs = output_spec["num_outputs"]
    output_specs = output_spec["output_specs"]
    output_aliases = output_spec["output_aliases"]
    output_alias_is_direct = output_spec["output_alias_is_direct"]
    mutated_inputs = output_spec["mutated_inputs"]

    def get_output(i: int) -> VariableTracker:
        # Check if this output is a scalar (None, int, float, bool)
        # Scalars are known at compile time and should be returned as constants
        spec = output_specs[i] if i < len(output_specs) else None
        if spec is not None and "scalar_value" in spec:
            return variables.ConstantVariable.create(spec["scalar_value"])

        # Check if this output aliases an input
        alias = output_aliases[i] if i < len(output_aliases) else None
        is_direct = output_alias_is_direct[i] if i < len(output_alias_is_direct) else False

        if alias is not None and alias in param_vars:
            # For aliased outputs, return the input variable
            # This maintains proper aliasing in the traced graph
            input_var = param_vars[alias]
            if is_direct:
                # Direct alias: return input as-is
                return input_var
            else:
                # Indirect alias (view): need to extract from HOP result
                # since the view operation may have different shape/stride
                pass

        # Non-aliased or indirect alias: extract from HOP result
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
    param_tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, object] | None, str | None, bool, torch.Tensor | None]:
    """Build output spec for a single return expression.

    Returns:
        (spec_dict, alias_name, is_direct_alias, tensor)
        - spec_dict: Output specification (shape, dtype, etc.) or None for scalars
        - alias_name: Name of aliased input parameter, or None
        - is_direct_alias: Whether the alias is direct (same shape/stride)
        - tensor: The tensor value if available, for mutation detection
    """
    t = _get_value_from_type_info(node, local_types, param_values)

    if isinstance(t, torch.Tensor):
        # Check if output aliases an input via storage comparison
        alias_name, is_direct = _find_alias_name(t, param_tensors)

        # Fallback: if return expression is simply a parameter name, it's an alias.
        # This handles cases where FakeTensors from type propagation don't share
        # storage with the example_value FakeTensors from Dynamo (e.g., `return x`
        # where x is a mutated input parameter).
        if alias_name is None and isinstance(node, ast.Name):
            if node.id in param_tensors:
                alias_name = node.id
                # Check if shapes/strides match for direct alias determination
                param_t = param_tensors[alias_name]
                is_direct = (
                    list(t.shape) == list(param_t.shape)
                    and list(t.stride()) == list(param_t.stride())
                )

        spec = {
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "storage_offset": t.storage_offset(),
            "dtype": t.dtype,
            "device": str(t.device),
        }
        return spec, alias_name, is_direct, t
    elif t is None:
        raise _dynamo_exc.InternalTorchDynamoError(
            f"None return values are not supported with torch.compile fusion. "
            f"Expression `{ast.unparse(node)}` evaluates to None."
        )
    elif isinstance(t, (int, float, bool)):
        return {"scalar_value": t}, None, False, None
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

    # Build mapping of parameter names to tensor values (for aliasing detection)
    param_tensors = {
        name: val
        for name, val in param_values.items()
        if isinstance(val, torch.Tensor)
    }

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
    output_specs: list[dict[str, object] | None] = []
    output_aliases: list[str | None] = []
    output_alias_is_direct: list[bool] = []
    output_tensors: list[torch.Tensor | None] = []

    for node in return_nodes:
        spec, alias, is_direct, tensor = _build_output_info(
            node, local_types, param_values, param_tensors
        )
        output_specs.append(spec)
        output_aliases.append(alias)
        output_alias_is_direct.append(is_direct)
        output_tensors.append(tensor)

    # Detect mutated inputs via AST analysis
    mutated_inputs = _detect_mutated_inputs(
        kernel, param_tensors, local_types, bound.host_function.body
    )

    # Also mark inputs as mutated if they are returned (aliased) and modified
    for alias in output_aliases:
        if alias is not None and alias not in mutated_inputs:
            # Check if the returned tensor differs from input
            # (would indicate mutation happened during kernel execution)
            mutated_inputs.append(alias)

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
        output_aliases=output_aliases,
        output_alias_is_direct=output_alias_is_direct,
        mutated_inputs=mutated_inputs,
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
        return _build_return_value(tx, result, output_spec, param_vars)


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
