from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import torch
from torch._dynamo import variables
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builder import wrap_fx_proxy
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.lists import TupleVariable
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

from helion._compiler._dynamo.higher_order_ops import helion_kernel_wrapper_mutation
from helion._compiler.type_propagation import TensorType

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from helion._compiler.device_ir import DeviceIR
    from helion._compiler.host_function import HostFunction


def _get_fake_tensor(type_info: Any) -> torch.Tensor | None:  # noqa: ANN401
    """Extract fake tensor from TensorType or types with .value attribute."""
    if isinstance(type_info, TensorType):
        return type_info.fake_value
    return None


def _check_tensor_alias(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """Check if two tensors are the same object or aliases."""
    # pyrefly: ignore[missing-attribute]
    return t1 is t2 or torch._C._is_alias_of(t1, t2)


def _eval_return_expr(
    return_node: ast.expr,
    local_types: dict[str, Any],
) -> Any:
    """Evaluate a return expression using fake tensors from local_types.

    Uses ast.unparse() + eval() instead of manual AST walking.
    This handles ALL expressions correctly since FakeTensor supports
    standard tensor operations like .T, [0:2], .permute(), .view(), etc.
    """
    # Build evaluation namespace with fake tensors
    eval_globals: dict[str, Any] = {"torch": torch}
    for name, vtype in local_types.items():
        fake = _get_fake_tensor(vtype)
        if fake is not None:
            eval_globals[name] = fake
        elif hasattr(vtype, "value"):
            eval_globals[name] = vtype.value

    # Convert AST to string and evaluate
    expr_str = ast.unparse(return_node)
    try:
        return eval(expr_str, eval_globals)  # noqa: S307
    except Exception:
        return None  # Fall back to None if evaluation fails


def _find_aliasing_input(
    tensor: torch.Tensor,
    input_names: set[str],
    local_types: dict[str, Any],
) -> tuple[str | None, bool]:
    """Find which input parameter a tensor aliases.

    Returns:
        (alias_name, has_same_layout) where has_same_layout means same shape/stride.
    """
    for inp in input_names:
        inp_fake = _get_fake_tensor(local_types.get(inp))
        if inp_fake is not None and _check_tensor_alias(tensor, inp_fake):
            has_same_layout = tensor.shape == inp_fake.shape and tensor.stride() == inp_fake.stride()
            return inp, has_same_layout
    return None, False


def _add_mutated_param_or_alias(
    param_name: str | None,
    input_names: set[str],
    local_types: dict[str, Any],
    mutated: set[str],
) -> None:
    """Add param_name to mutated set, or its aliasing input if it's a local variable."""
    if param_name is None:
        return
    if param_name in input_names:
        mutated.add(param_name)
        return
    fake = _get_fake_tensor(local_types.get(param_name))
    if fake is not None:
        alias, _ = _find_aliasing_input(fake, input_names, local_types)
        if alias:
            mutated.add(alias)


def _trace_to_host_tensor(node: torch.fx.Node, host_tensor_target: Any) -> str | None:  # noqa: ANN401
    """Trace through FX graph to find host tensor reference."""
    visited: set[torch.fx.Node] = set()

    def _trace(n: torch.fx.Node) -> str | None:
        if n in visited:
            return None
        visited.add(n)
        if n.op == "call_function" and n.target is host_tensor_target:
            return n.args[0] if n.args and isinstance(n.args[0], str) else None
        for arg in n.args:
            if isinstance(arg, torch.fx.Node) and (r := _trace(arg)):
                return r
        return None

    return _trace(node)


def _find_mutated_inputs(
    host_function: HostFunction,
    device_ir: DeviceIR,
    input_names: set[str],
) -> list[str]:
    """Find input parameters that are mutated (store/atomic ops) in the kernel."""
    from helion.language import atomic_ops
    from helion.language import memory_ops
    from helion.language._tracing_ops import _host_tensor
    from helion.language.inline_triton_ops import inline_triton
    from helion.language.inline_triton_ops import triton_kernel
    from helion.language.signal_wait import signal
    from helion.language.signal_wait import wait

    mutating_ops = {memory_ops.store, signal} | {
        getattr(atomic_ops, n)
        for n in atomic_ops.__all__
        if callable(getattr(atomic_ops, n, None))
    }
    unknown_mutation_ops = {inline_triton, triton_kernel}
    mutated: set[str] = set()
    local_types = host_function.local_types or {}

    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if target in unknown_mutation_ops and input_names:
                return list(input_names)
            if target is wait and input_names:
                update = node.args[3] if len(node.args) > 3 else node.kwargs.get("update")
                if update is None:
                    continue
                target_node = node.args[0] if node.args else None
                if isinstance(target_node, torch.fx.Node):
                    param_name = _trace_to_host_tensor(target_node, _host_tensor)
                    if param_name is None:
                        return list(input_names)
                    _add_mutated_param_or_alias(
                        param_name, input_names, local_types, mutated
                    )
            elif target in mutating_ops and node.args and isinstance(node.args[0], torch.fx.Node):
                param_name = _trace_to_host_tensor(node.args[0], _host_tensor)
                if param_name is not None:
                    _add_mutated_param_or_alias(
                        param_name, input_names, local_types, mutated
                    )
    return list(mutated)


def _expand_mutated_inputs_with_aliases(
    mutated_inputs: Sequence[str],
    param_names: Sequence[str],
    example_args: Sequence[object],
) -> list[str]:
    """Expand mutated inputs to include any aliased tensor arguments."""
    if not mutated_inputs:
        return []
    tensors = {
        n: a
        for n, a in zip(param_names, example_args, strict=True)
        if isinstance(a, torch.Tensor)
    }
    if len(tensors) < 2:
        return list(mutated_inputs)
    mutated = set(mutated_inputs)
    for name in list(mutated):
        if name not in tensors:
            continue
        for other, t in tensors.items():
            if other != name and _check_tensor_alias(t, tensors[name]):
                mutated.add(other)
    return [n for n in param_names if n in mutated]


def _build_return_value(
    tx: InstructionTranslator,
    result: VariableTracker,
    output_spec: dict[str, Any],
    param_vars: dict[str, VariableTracker],
) -> VariableTracker:
    """Build the appropriate return value from HOP result."""
    num_outputs = output_spec["num_outputs"]
    output_aliases = output_spec.get("output_aliases", [])
    output_alias_is_direct = output_spec.get("output_alias_is_direct", [])

    def get_output(i: int) -> VariableTracker:
        # Check if output i is a direct alias of an input parameter
        alias = output_aliases[i] if i < len(output_aliases) else None
        is_direct = i < len(output_alias_is_direct) and output_alias_is_direct[i]
        if is_direct and alias in param_vars:
            return param_vars[alias]
        return result.call_method(
            tx, "__getitem__", [variables.ConstantVariable.create(i)], {}
        )

    if num_outputs > 1:
        return TupleVariable([get_output(i) for i in range(num_outputs)])
    return get_output(0)


def _infer_output_spec(
    kernel: Any,  # noqa: ANN401
    args: Sequence[VariableTracker],
) -> dict[str, Any]:
    """Infer output specification by binding kernel with fake args."""

    def get_example_value(arg: VariableTracker) -> Any:  # noqa: ANN401
        if arg.is_python_constant():
            return arg.as_python_constant()
        proxy = arg.as_proxy()
        if proxy is not None and hasattr(proxy, "node"):
            return proxy.node.meta.get("example_value")
        return None

    fake_args = [get_example_value(a) for a in args]
    param_names = list(kernel.signature.parameters.keys())

    bound = kernel.bind(tuple(fake_args))
    if not bound.host_function:
        raise AssertionError("kernel.bind() succeeded but host_function is None")

    num_tiled = sum(1 for bs in bound.env.block_sizes if not bs.reduction)
    input_names = set(param_names)
    local_types = bound.host_function.local_types or {}

    # Parse return statement to extract return expressions
    num_outputs, return_nodes = -1, []
    for stmt in reversed(bound.host_function.body):
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            if isinstance(stmt.value, ast.Tuple):
                num_outputs, return_nodes = len(stmt.value.elts), list(stmt.value.elts)
            else:
                num_outputs, return_nodes = 1, [stmt.value]
            break

    # Evaluate return expressions, build output specs, and track aliases.
    #
    # ALIAS TRACKING OVERVIEW:
    # We need to track whether each output aliases an input parameter. This matters
    # because torch.compile's FX graph needs to maintain tensor identity: if a kernel
    # returns an input tensor unchanged (e.g., `return x` where x is an input), the
    # output graph node must be the same as the input graph node, not a new tensor.
    #
    # Two types of aliases:
    # 1. Direct alias: `return x` where x is an input parameter name
    #    - output_alias_is_direct=True: return the original VariableTracker directly
    # 2. Indirect alias: `return x.T` or `return some_local` where some_local aliases x
    #    - output_alias_is_direct=False: the output shares storage but has different
    #      shape/strides, so we return the HOP result (which creates a view)
    #
    output_specs: list[dict[str, Any] | None] = []
    output_aliases: list[str | None] = []  # Which input each output aliases (or None)
    output_alias_is_direct: list[bool] = []  # True if output IS the input (same shape/stride)

    for node in return_nodes:
        t = _eval_return_expr(node, local_types)
        ret_tensor: torch.Tensor | None = None

        if isinstance(t, torch.Tensor):
            ret_tensor = t
            output_specs.append({
                "shape": list(t.shape),
                "stride": list(t.stride()),
                "storage_offset": t.storage_offset(),
                "dtype": t.dtype,
                "device": str(t.device),
            })
        elif t is None or isinstance(t, (int, float, bool)):
            output_specs.append({"scalar_value": t})
        else:
            output_specs.append(None)

        # Determine if this output aliases an input parameter.
        # Case 1: Direct reference to an input (e.g., `return x` where x is a param)
        if isinstance(node, ast.Name) and node.id in input_names:
            alias_name, is_direct = node.id, True
        else:
            # Case 2: Check if the evaluated tensor shares storage with any input.
            # This catches cases like:
            #   - `return x.T` (view of input x)
            #   - `return local_var` where local_var was assigned from an input
            tensor = (
                _get_fake_tensor(local_types.get(node.id))
                if isinstance(node, ast.Name)
                else ret_tensor
            )
            alias_name, is_direct = (
                _find_aliasing_input(tensor, input_names, local_types)
                if tensor is not None
                else (None, False)
            )
        output_aliases.append(alias_name)
        output_alias_is_direct.append(is_direct)

    # MUTATION TRACKING:
    # Find which input tensors are mutated by the kernel (via store/atomic ops).
    # This is crucial for correctness: torch.compile's functionalization pass needs
    # to know which inputs are modified so it can properly handle:
    # 1. Copy-on-write semantics (clone mutated inputs before the kernel)
    # 2. Propagating mutations back to the original tensors after the kernel
    mutated_inputs = _find_mutated_inputs(
        bound.host_function, bound.host_function.device_ir, input_names
    )
    # Expand to include aliased tensors: if input A is mutated and input B aliases A,
    # then B is effectively mutated too (they share the same underlying storage)
    mutated_inputs = _expand_mutated_inputs_with_aliases(
        mutated_inputs, param_names, fake_args
    )

    return {
        "num_outputs": num_outputs,
        "num_tiled_dims": num_tiled,
        "output_specs": output_specs,
        "mutated_inputs": mutated_inputs,
        "output_aliases": output_aliases,
        "output_alias_is_direct": output_alias_is_direct,
    }


class HelionKernelVariable(VariableTracker):
    """Variable tracker for Helion kernel objects."""

    _kernel: Any
    _kernel_idx: int

    def __init__(
        self,
        kernel: Any,  # noqa: ANN401
        kernel_idx: int | None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self._kernel = kernel
        self._kernel_idx = (
            kernel_idx
            if kernel_idx is not None
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
        for k, v in kwargs.items():
            param_vars[k] = v

        # Step 2: Partition arguments into constants vs tensors
        # - constant_args: Python values (int, float, etc.) passed directly to kernel
        # - tensor_args: Tensor proxies that become graph inputs to the HOP
        constant_args: dict[str, Any] = {}
        tensor_args: dict[VariableTracker, VariableTracker] = {}
        for name, var in param_vars.items():
            key = variables.ConstantVariable.create(name)
            if var.is_python_constant():
                constant_args[name] = var.as_python_constant()
            else:
                tensor_args[key] = var

        # Build ordered args in signature order (with defaults) for output inference
        ordered_args = [
            param_vars.get(name)
            or variables.ConstantVariable.create(sig_params[name].default)
            for name in param_names
            if name in param_vars or sig_params[name].default is not sig_params[name].empty
        ]

        # Step 3: Infer output specification by binding kernel with fake tensors
        # This determines: number of outputs, their shapes/dtypes, which inputs are mutated,
        # and which outputs alias which inputs
        output_spec = _infer_output_spec(self._kernel, ordered_args)

        # Step 4: Emit a Higher-Order Op (HOP) node into the FX graph
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

        # Step 5: Wrap the proxy and build the return value
        # For direct input aliases (e.g., return x where x is an input), return the
        # original VariableTracker to preserve identity; otherwise extract from HOP result
        result = wrap_fx_proxy(tx, hop_proxy)
        return _build_return_value(tx, result, output_spec, param_vars)
