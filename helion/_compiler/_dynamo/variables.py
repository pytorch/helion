from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING
from typing import Callable
from typing import Sequence
from typing import TypedDict

import torch
from torch._dynamo import exc as _dynamo_exc
from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import GuardBuilder
from torch._dynamo.variables.builder import VariableBuilder
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.lists import ListVariable
from torch._dynamo.variables.lists import TupleVariable
from torch._higher_order_ops.auto_functionalize import get_base
from torch._higher_order_ops.auto_functionalize import is_alias
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch._higher_order_ops.utils import _tensor_storage
from torch.multiprocessing.reductions import StorageWeakRef

from helion._compiler._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from helion._compiler.type_propagation import LiteralType
from helion._compiler.type_propagation import NumericType
from helion._compiler.type_propagation import TensorType
from helion._compiler.type_propagation import TypeInfo

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from helion._compiler.device_ir import DeviceIR
    from helion._compiler.host_function import HostFunction
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


def _get_var_tensor(var: VariableTracker | None) -> torch.Tensor | None:
    """Extract tensor from VariableTracker, or None if not a tensor."""
    if var is None:
        return None
    val = _get_var_value(var)
    return val if isinstance(val, torch.Tensor) else None


def _find_aliasing_input(
    tensor: torch.Tensor | None,
    input_names: set[str],
    local_types: dict[str, TypeInfo],
) -> tuple[str | None, bool]:
    """Find which input parameter a tensor aliases by checking storage sharing.

    Uses StorageWeakRef for O(1) storage-based aliasing detection, which
    correctly handles all view operations (.view(), .reshape(), .T, etc.).

    Returns:
        (alias_name, is_direct_alias) where is_direct_alias means same shape/stride
        (i.e., the tensor has the same physical layout as the input).
    """
    if tensor is None:
        return None, False

    # Build storage -> input name map for O(1) lookup
    storage_to_input: dict[StorageWeakRef, tuple[str, torch.Tensor]] = {}
    for inp in input_names:
        inp_fake = local_types[inp].as_tensor() if inp in local_types else None
        if inp_fake is not None:
            storage_to_input[_tensor_storage(inp_fake)] = (inp, inp_fake)

    # Check if tensor's storage matches any input
    tensor_storage = _tensor_storage(tensor)
    if tensor_storage in storage_to_input:
        inp_name, inp_fake = storage_to_input[tensor_storage]
        is_direct = is_alias(inp_fake, tensor)
        return inp_name, is_direct

    return None, False


# =============================================================================
# AST Parsing Utilities
# =============================================================================


def _check_param_reassignment(body: list[ast.stmt], input_names: set[str]) -> None:
    """Check for parameter reassignment (not supported with torch.compile).

    Note: We cannot use ReadWrites.from_list() here because it treats subscript
    stores (x[i] = val) as writes to x, but we only want to detect direct
    parameter reassignment (x = something).
    """
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in input_names:
                    raise RuntimeError(
                        f"Reassigning parameter '{target.id}' is not supported "
                        f"with torch.compile fusion. Use a different variable name."
                    )


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
# Mutation Analysis
# =============================================================================


def _trace_to_host_tensor(
    node: torch.fx.Node, host_tensor_target: Callable[..., object]
) -> str | None:
    """Trace through FX graph to find host tensor reference."""
    stack = [node]
    visited: set[torch.fx.Node] = set()
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        if n.op == "call_function" and n.target is host_tensor_target:
            return n.args[0] if n.args and isinstance(n.args[0], str) else None
        stack.extend(arg for arg in n.args if isinstance(arg, torch.fx.Node))
    return None


@functools.lru_cache(maxsize=1)
def _get_mutation_registry() -> tuple[
    set[Callable[..., object]],
    set[Callable[..., object]],
    dict[Callable[..., object], str],
]:
    """Get the mutation behavior registry for Helion ops (cached).

    Returns:
        (mutates_first_arg, mutates_unknown, mutates_conditional) where:
        - mutates_first_arg: ops that always mutate their first argument
        - mutates_unknown: ops with unknown mutation patterns (assume all inputs mutated)
        - mutates_conditional: ops that conditionally mutate first arg based on a param
    """
    from helion.language import atomic_ops
    from helion.language import memory_ops
    from helion.language.inline_triton_ops import inline_triton
    from helion.language.inline_triton_ops import triton_kernel
    from helion.language.signal_wait import signal
    from helion.language.signal_wait import wait

    mutates_first_arg = {memory_ops.store, signal} | {
        getattr(atomic_ops, n)
        for n in atomic_ops.__all__
        if callable(getattr(atomic_ops, n, None))
    }
    mutates_unknown = {inline_triton, triton_kernel}
    # Maps op -> param name that controls whether mutation happens
    mutates_conditional: dict[Callable[..., object], str] = {wait: "update"}

    return mutates_first_arg, mutates_unknown, mutates_conditional


def _find_mutated_inputs(
    host_function: HostFunction,
    device_ir: DeviceIR,
    input_names: set[str],
) -> list[str]:
    """Find input parameters that are mutated (store/atomic ops) in the kernel."""
    from helion.language._tracing_ops import _host_tensor

    mutates_first_arg, mutates_unknown, mutates_conditional = _get_mutation_registry()

    mutated: set[str] = set()
    local_types = host_function.local_types or {}

    def add_mutated(param_name: str | None) -> None:
        """Add param_name to mutated set, or its aliasing input if local."""
        if param_name is None:
            return
        if param_name in input_names:
            mutated.add(param_name)
        elif (fake := local_types[param_name].as_tensor() if param_name in local_types else None) is not None:
            alias, _ = _find_aliasing_input(fake, input_names, local_types)
            if alias:
                mutated.add(alias)

    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target

            # Unknown mutation ops - conservatively assume all inputs mutated
            if target in mutates_unknown and input_names:
                return list(input_names)

            # Conditional mutation (e.g., wait with update parameter)
            cond_param = mutates_conditional.get(target)
            if cond_param is not None and input_names:
                # Check if the conditional param is provided and not None
                # For wait(), update is arg[3] or kwargs["update"]
                update = node.kwargs.get(cond_param) or (
                    node.args[3] if len(node.args) > 3 else None
                )
                if update is None:
                    continue
                target_node = node.args[0] if node.args else None
                if isinstance(target_node, torch.fx.Node):
                    param_name = _trace_to_host_tensor(target_node, _host_tensor)
                    if param_name is None:
                        return list(input_names)
                    add_mutated(param_name)

            # Standard first-arg mutation
            elif target in mutates_first_arg:
                if node.args and isinstance(node.args[0], torch.fx.Node):
                    param_name = _trace_to_host_tensor(node.args[0], _host_tensor)
                    add_mutated(param_name)

    return list(mutated)


# =============================================================================
# Output Specification Inference
# =============================================================================


def _build_return_value(
    tx: InstructionTranslator,
    result: VariableTracker,
    output_spec: OutputSpec,
    param_vars: dict[str, VariableTracker],
) -> VariableTracker:
    """Build the appropriate return value from HOP result."""
    num_outputs = output_spec["num_outputs"]
    output_aliases = output_spec["output_aliases"]
    output_alias_is_direct = output_spec["output_alias_is_direct"]
    output_specs = output_spec["output_specs"]
    mutated_inputs = set(output_spec["mutated_inputs"])

    def get_output(i: int) -> VariableTracker:
        # Check if this output is a scalar (None, int, float, bool)
        # Scalars are known at compile time and should be returned as constants
        spec = output_specs[i] if i < len(output_specs) else None
        if spec is not None and "scalar_value" in spec:
            return variables.ConstantVariable.create(spec["scalar_value"])

        # Check if output i is a direct alias of an input parameter
        alias = output_aliases[i] if i < len(output_aliases) else None
        is_direct = i < len(output_alias_is_direct) and output_alias_is_direct[i]
        # For direct aliases of NON-mutated inputs, return the original variable
        # to preserve tensor identity. For mutated inputs, return the HOP output
        # since functionalization confines mutation to a clone and does not call
        # ctx.replace_update(), so the original variable retains its pre-mutation value.
        if (
            is_direct
            and alias is not None
            and alias in param_vars
            and alias not in mutated_inputs
        ):
            return param_vars[alias]
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
    input_names: set[str],
) -> tuple[dict[str, object] | None, str | None, bool, torch.Tensor | None]:
    """Build output spec and aliasing info for a single return expression.

    Returns:
        (output_spec, alias_name, is_direct_alias, tensor)
    """
    t = _get_value_from_type_info(node, local_types, param_values)
    ret_tensor: torch.Tensor | None = None

    if isinstance(t, torch.Tensor):
        ret_tensor = t
        spec: dict[str, object] | None = {
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
        spec = {"scalar_value": t}
    else:
        raise RuntimeError(
            f"Returning {type(t).__name__} values from a Helion kernel is not supported "
            f"with torch.compile fusion. Expression `{ast.unparse(node)}` evaluates to "
            f"{type(t).__name__}. Supported return types: tensor, int, float, bool."
        )

    alias_name, is_direct = _find_aliasing_input(ret_tensor, input_names, local_types)
    return spec, alias_name, is_direct, ret_tensor


def _check_output_aliasing(
    return_nodes: list[ast.expr], output_tensors: list[torch.Tensor | None]
) -> None:
    """Check for output-to-output aliasing (not supported)."""
    # AST check: same variable returned multiple times (e.g., return x, x)
    seen_names: dict[str, int] = {}
    for i, node in enumerate(return_nodes):
        if isinstance(node, ast.Name):
            if node.id in seen_names:
                raise RuntimeError(
                    f"Returning the same variable multiple times is not supported "
                    f"with torch.compile fusion. Variable '{node.id}' is returned "
                    f"at positions {seen_names[node.id]} and {i}."
                )
            seen_names[node.id] = i

    # Storage check: tensors sharing storage (e.g., return x, x.T)
    seen_storages: dict[StorageWeakRef, int] = {}
    for i, t in enumerate(output_tensors):
        if t is not None:
            storage = _tensor_storage(t)
            if storage in seen_storages:
                raise RuntimeError(
                    f"Returning aliased tensors is not supported with torch.compile "
                    f"fusion. Outputs at positions {seen_storages[storage]} and {i} "
                    f"share the same storage."
                )
            seen_storages[storage] = i


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

    input_names = set(param_names)
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

    _check_param_reassignment(bound.host_function.body, input_names)

    # Build output specs and track aliases for each return expression
    output_infos = [
        _build_output_info(node, local_types, param_values, input_names)
        for node in return_nodes
    ]
    output_specs = [info[0] for info in output_infos]
    output_aliases = [info[1] for info in output_infos]
    output_alias_is_direct = [info[2] for info in output_infos]
    output_tensors = [info[3] for info in output_infos]

    # Check for output-to-output aliasing (not supported)
    _check_output_aliasing(return_nodes, output_tensors)

    # Find mutated inputs for functionalization
    mutated_inputs = _find_mutated_inputs(
        bound.host_function, bound.host_function.device_ir, input_names
    )

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
        mutated_inputs=mutated_inputs,
        output_aliases=output_aliases,
        output_alias_is_direct=output_alias_is_direct,
    )


# =============================================================================
# View Pattern Validation
# =============================================================================


def _check_same_tensor_multiple_params(
    mutated_inputs: list[str],
    get_proxy_id: Callable[[str], int | None],
) -> None:
    """Check if the same tensor is passed to multiple mutated parameters."""
    proxy_id_to_mutated_args: dict[int, list[str]] = {}
    for name in mutated_inputs:
        proxy_id = get_proxy_id(name)
        if proxy_id is not None:
            proxy_id_to_mutated_args.setdefault(proxy_id, []).append(name)
            if len(proxy_id_to_mutated_args[proxy_id]) > 1:
                raise ValueError(
                    f"torch.compile with Helion kernels does not support passing the "
                    f"same tensor as multiple mutated arguments. Arguments "
                    f"{proxy_id_to_mutated_args[proxy_id]} refer to the same tensor. "
                    f"Please use separate tensors or call the kernel outside of torch.compile."
                )


def _check_view_base_conflicts(
    mutated_inputs: list[str],
    get_fake_tensor: Callable[[str], torch.Tensor | None],
    tensor_arg_names: list[str],
) -> None:
    """Check for view/base aliasing conflicts in mutated inputs."""
    # Build storage -> arg_name map for ALL tensor args
    storage_to_arg: dict[StorageWeakRef, str] = {}
    for name in tensor_arg_names:
        fake = get_fake_tensor(name)
        if fake is not None:
            storage_to_arg[_tensor_storage(fake)] = name

    # Track base storages for mutated args
    base_storage_to_mutated: dict[StorageWeakRef, list[str]] = {}
    base_storage_is_view: dict[StorageWeakRef, bool] = {}

    for name in mutated_inputs:
        fake = get_fake_tensor(name)
        if fake is None:
            continue

        base = get_base(fake)
        is_view = base is not None
        if base is None:
            base = fake

        base_storage = _tensor_storage(base)
        base_storage_to_mutated.setdefault(base_storage, []).append(name)

        # Multiple views of same base
        if is_view and len(base_storage_to_mutated[base_storage]) > 1:
            raise ValueError(
                f"torch.compile with Helion kernels does not support multiple mutated "
                f"views of the same base tensor. Arguments {base_storage_to_mutated[base_storage]} "
                f"are views of the same underlying tensor. Please use separate base tensors "
                f"or call the kernel outside of torch.compile."
            )

        # View + base passed separately
        if is_view:
            base_arg_name = storage_to_arg.get(base_storage)
            if base_arg_name is not None and base_arg_name != name:
                raise ValueError(
                    f"torch.compile with Helion kernels does not support passing both "
                    f"a view and its base tensor as separate arguments. Argument '{name}' "
                    f"is a view of argument '{base_arg_name}'. Please pass only one or "
                    f"call the kernel outside of torch.compile."
                )
            base_storage_is_view[base_storage] = True

    # Check non-mutated args that are bases of mutated views
    for name in tensor_arg_names:
        if name in mutated_inputs:
            continue
        fake = get_fake_tensor(name)
        if fake is None:
            continue
        fake_storage = _tensor_storage(fake)
        if fake_storage in base_storage_is_view:
            mutated_views = base_storage_to_mutated.get(fake_storage, [])
            raise ValueError(
                f"torch.compile with Helion kernels does not support passing both "
                f"a view and its base tensor as separate arguments. Argument '{name}' "
                f"is the base of mutated view argument(s) {mutated_views}. Please pass only "
                f"one or call the kernel outside of torch.compile."
            )


def _check_unsupported_view_patterns(
    mutated_inputs: list[str],
    get_fake_tensor: Callable[[str], torch.Tensor | None],
    tensor_arg_names: list[str],
    get_proxy_id: Callable[[str], int | None] | None = None,
) -> None:
    """Check for unsupported view patterns and raise errors."""
    if not mutated_inputs:
        return
    if get_proxy_id is not None:
        _check_same_tensor_multiple_params(mutated_inputs, get_proxy_id)
    _check_view_base_conflicts(mutated_inputs, get_fake_tensor, tensor_arg_names)


# =============================================================================
# HelionKernelVariable Class
# =============================================================================


class HelionKernelVariable(VariableTracker):
    """Variable tracker for Helion kernel objects."""

    _kernel: Kernel
    _kernel_idx: int

    def __init__(
        self,
        kernel: Kernel,
        kernel_idx: int | None,
        # pyrefly: ignore[bad-argument-type]
        **kwargs: object,
    ) -> None:
        # pyrefly: ignore[bad-argument-type]
        super().__init__(**kwargs)
        self._kernel = kernel
        self._kernel_idx = (
            kernel_idx
            if kernel_idx is not None
            # pyrefly: ignore[bad-argument-type]
            else kernel_side_table.add_kernel(kernel)
        )

    def call_function(
        self,
        tx: InstructionTranslator,
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """Handle a call to a Helion kernel during Dynamo tracing."""
        param_names = list(self._kernel.signature.parameters.keys())
        sig_params = self._kernel.signature.parameters

        # Step 1: Map positional args and kwargs to parameter names
        param_vars: dict[str, VariableTracker] = {}
        for i, arg in enumerate(args):
            if i < len(param_names):
                param_vars[param_names[i]] = arg
        param_vars.update(kwargs)

        # Step 2: Partition arguments into constants vs tensors
        # - constant_args: Python values (int, float, etc.) passed directly to kernel
        # - tensor_args: Tensor proxies that become graph inputs to the HOP
        # Also track proxy identity to detect kernel(x, x) vs kernel(x.clone(), x.clone())
        constant_args: dict[str, object] = {}
        tensor_args: dict[VariableTracker, VariableTracker] = {}
        tensor_arg_proxy_ids: dict[str, int] = {}
        for name, var in param_vars.items():
            key = variables.ConstantVariable.create(name)
            if var.is_python_constant():
                constant_args[name] = var.as_python_constant()
            else:
                tensor_args[key] = var
                tensor_arg_proxy_ids[name] = id(var.as_proxy())

        # Build ordered args in signature order (with defaults) for output inference
        ordered_args = [
            param_vars.get(name)
            or variables.ConstantVariable.create(sig_params[name].default)
            for name in param_names
            if name in param_vars
            or sig_params[name].default is not sig_params[name].empty
        ]

        # Step 3: Infer output specification by binding kernel with fake tensors
        # This determines: number of outputs, their shapes/dtypes, which inputs are mutated,
        # and which outputs alias which inputs
        output_spec = _infer_output_spec(self._kernel, ordered_args)

        # Step 4: Get mutated inputs for view pattern checking
        mutated_inputs = output_spec["mutated_inputs"]

        # Check for unsupported view patterns and raise errors early
        _check_unsupported_view_patterns(
            mutated_inputs,
            lambda name: _get_var_tensor(param_vars.get(name)),
            list(tensor_arg_proxy_ids.keys()),
            lambda name: tensor_arg_proxy_ids.get(name),
        )

        # Step 5: Emit a Higher-Order Op (HOP) node into the FX graph
        # The HOP encapsulates the entire Helion kernel call as a single graph node
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

        # Step 6: Wrap the proxy and build the return value
        # For direct input aliases (e.g., return x where x is an input), return the
        # original VariableTracker to preserve identity; otherwise extract from HOP result
        result = wrap_fx_proxy(tx, hop_proxy)
        return _build_return_value(tx, result, output_spec, param_vars)


def register_dynamo_variable() -> None:
    """Register HelionKernelVariable with Dynamo's VariableBuilder."""
    from helion.runtime.kernel import Kernel

    def wrap_helion_kernel(self: VariableBuilder, value: Kernel) -> VariableTracker:
        if value.settings._wip_experimental_allow_torch_compile_fusion:
            self.install_guards(GuardBuilder.ID_MATCH)
            return HelionKernelVariable(value, None, source=self.source)
        return self.wrap_user_defined(value)

    VariableBuilder._type_dispatch()[Kernel] = wrap_helion_kernel
