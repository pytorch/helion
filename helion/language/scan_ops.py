from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING
from typing import cast
from typing import overload

import torch
import torch._higher_order_ops as higher_order_ops
from torch.fx.experimental import proxy_tensor

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.device_ir import HelperFunctionGraphInfo
    from .._compiler.helper_function import CombineFunction
    from .._compiler.helper_function import CombineFunctionBasic
    from .._compiler.helper_function import CombineFunctionTuple
    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.type_info import Origin
    from .._compiler.type_info import TypeInfo


__all__ = ["associative_scan", "cumprod", "cumsum"]


@overload
@_decorators.device_func_replacement(higher_order_ops.associative_scan)
@_decorators.api(is_device_only=True)
def associative_scan(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor,
    dim: int,
    reverse: bool = False,
) -> torch.Tensor: ...


@overload
@_decorators.device_func_replacement(higher_order_ops.associative_scan)
@_decorators.api(is_device_only=True)
def associative_scan(
    combine_fn: CombineFunction,
    input_tensor: tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> tuple[torch.Tensor, ...]: ...


@_decorators.device_func_replacement(higher_order_ops.associative_scan)
@_decorators.api(is_device_only=True)
def associative_scan(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Applies an associative scan operation along a specified dimension.

    Computes the prefix scan (cumulative operation) along a dimension using
    a custom combine function. Unlike :func:`~helion.language.reduce`, this
    preserves the input shape.

    Args:
        combine_fn: A binary function that combines two elements element-wise.
                   Must be associative for correct results.
                   Can be tensor->tensor or tuple->tuple function.
        input_tensor: Input tensor or tuple of tensors to scan
        dim: The dimension along which to scan
        reverse: If True, performs the scan in reverse order

    Returns:
        torch.Tensor or tuple[torch.Tensor, ...]: Tensor(s) with same shape as input
                                                  containing the scan result

    See Also:
        - :func:`~helion.language.reduce`: For dimension-reducing operations
        - :func:`~helion.language.cumsum`: For cumulative sum
        - :func:`~helion.language.cumprod`: For cumulative product

    Note:
        - combine_fn must be associative (not necessarily commutative)
        - Output has same shape as input (unlike reduce)
        - For standard scans, use :func:`~helion.language.cumsum` or :func:`~helion.language.cumprod` (faster)
        - Reverse scan applies the operation from right to left
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(associative_scan)
def _(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Fake implementation that returns fake tensors with the same shape as input."""
    if isinstance(input_tensor, (tuple, list)):
        return tuple(torch.empty_like(t) for t in input_tensor)
    return torch.empty_like(input_tensor)


@_decorators.ref(associative_scan)
def _(
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    # Eager inclusive scan. Ref mode must not call torch's associative_scan HOP: it
    # runs Dynamo, which breaks under Helion's active RefMode TorchFunctionMode.
    from .._compiler.helper_function import create_combine_function_wrapper

    is_tuple = isinstance(input_tensor, (tuple, list))
    leaves = tuple(input_tensor) if is_tuple else (input_tensor,)
    combine = create_combine_function_wrapper(
        combine_fn, is_tuple_input=is_tuple, target_format="tuple"
    )

    scan_dim = dim % leaves[0].ndim
    length = leaves[0].size(scan_dim)
    order = reversed(range(length)) if reverse else range(length)

    acc: tuple[torch.Tensor, ...] | None = None
    out: list[tuple[torch.Tensor, ...]] = [()] * length
    for k in order:
        cur = tuple(leaf.select(scan_dim, k) for leaf in leaves)
        if acc is None:
            acc = cur
        elif is_tuple:
            acc = tuple(cast("CombineFunctionTuple", combine)(acc, cur))
        else:
            acc = (cast("CombineFunctionBasic", combine)(acc[0], cur[0]),)
        out[k] = acc

    result = tuple(torch.stack(col, dim=scan_dim) for col in zip(*out, strict=True))
    return result if is_tuple else result[0]


@_decorators.register_to_device_ir(associative_scan)
def _(
    tracer: proxy_tensor.PythonKeyTracer,
    combine_fn: CombineFunction,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    Device IR implementation that handles tracing for associative_scan.  We map
    associative_scan to _associative_scan, with a pre-traced graph for the combine
    function.
    """
    from .._compiler.device_ir import DeviceIR
    from .._compiler.device_ir import HelperFunctionGraphInfo
    from .._compiler.device_ir import args_to_proxies
    from .._compiler.device_ir import select_decomp_table
    from .._compiler.helper_function import create_combine_function_wrapper
    from .._compiler.helper_function import extract_helper_function_name

    is_tuple_input = isinstance(input_tensor, (tuple, list))
    if is_tuple_input:
        assert all(isinstance(t, torch.Tensor) for t in input_tensor), (
            "associative_scan input must be a tuple of tensors"
        )
    else:
        assert isinstance(input_tensor, torch.Tensor), (
            "associative_scan input must be a tensor"
        )
    assert isinstance(dim, int), "associative_scan dim must be an integer"

    assert callable(combine_fn), "combine_fn must be callable"
    # Extract the function name before wrapping
    original_function_name = extract_helper_function_name(combine_fn)
    combine_fn = create_combine_function_wrapper(
        combine_fn, is_tuple_input=is_tuple_input, target_format="unpacked"
    )

    # Create fake inputs for the combine function
    fake_inputs = []
    for tensor in input_tensor if is_tuple_input else [input_tensor]:
        fake_inputs.extend(
            [
                torch.empty([1], dtype=tensor.dtype, device=tensor.device),
                torch.empty([1], dtype=tensor.dtype, device=tensor.device),
            ]
        )

    combine_graph = proxy_tensor.make_fx(
        combine_fn, decomposition_table=select_decomp_table()
    )(*fake_inputs).graph
    combine_graph_id = DeviceIR.current().add_graph(
        combine_graph,
        HelperFunctionGraphInfo,
        node_args=[],
        original_function_name=original_function_name,
    )

    # Create the associative_scan tracing operation
    scan_args = (combine_graph_id, input_tensor, dim, reverse, is_tuple_input)
    proxy_args, proxy_kwargs = args_to_proxies(tracer, scan_args)
    proxy_out = tracer.create_proxy(
        "call_function",
        _associative_scan,
        proxy_args,
        proxy_kwargs,
    )

    # Create new output tensors to avoid aliasing input with output.
    # tl.associative_scan modifies its input in-place, so we must track
    # distinct output tensors to ensure the input remains usable after the scan.
    if is_tuple_input:
        output_tensors = []
        assert isinstance(input_tensor, (tuple, list))
        for i, tensor in enumerate(input_tensor):
            output_tensor = torch.empty_like(tensor)
            element_proxy = tracer.create_proxy(
                "call_function",
                operator.getitem,
                (proxy_out, i),
                {},
            )
            proxy_tensor.track_tensor_tree(
                output_tensor, element_proxy, constant=None, tracer=tracer
            )
            output_tensors.append(output_tensor)
        return tuple(output_tensors)

    output_tensor = torch.empty_like(input_tensor)
    proxy_tensor.track_tensor_tree(
        output_tensor, proxy_out, constant=None, tracer=tracer
    )
    return output_tensor


@_decorators.type_propagation(associative_scan)
def _(
    combine_fn: TypeInfo,
    input_tensor: TypeInfo,
    dim: TypeInfo,
    reverse: TypeInfo | None = None,
    *,
    origin: Origin,
) -> TypeInfo:
    """Type propagation for associative_scan - output has same type as input."""
    from .._compiler.type_info import CallableType
    from .._compiler.type_info import SequenceType
    from .._compiler.type_info import TensorType

    # Validate that combine_fn is callable
    if not isinstance(combine_fn, CallableType):
        raise exc.TypeInferenceError(f"combine_fn must be callable, got {combine_fn}")

    # Validate that input_tensor is a tensor or tuple of tensors
    if isinstance(input_tensor, TensorType):
        # Single tensor case
        return input_tensor
    if isinstance(input_tensor, SequenceType):
        # Tuple of tensors case - validate all elements are tensors
        for elem_type in input_tensor.unpack():
            if not isinstance(elem_type, TensorType):
                raise exc.TypeInferenceError(
                    f"All elements in tuple must be tensors, got {elem_type}"
                )
        # Return the same tuple type
        return input_tensor
    raise exc.TypeInferenceError(
        f"input_tensor must be a tensor or tuple of tensors, got {input_tensor}"
    )


@_decorators.device_func_replacement(torch.cumsum)
def cumsum(input_tensor: torch.Tensor, dim: int, reverse: bool = False) -> torch.Tensor:
    """
    Compute the cumulative sum along a specified dimension.

    Equivalent to ``hl.associative_scan(torch.add, input_tensor, dim, reverse)``.

    Args:
        input_tensor: Input tensor to compute cumulative sum
        dim: The dimension along which to compute cumulative sum
        reverse: If True, performs the cumsum in reverse order

    Returns:
        torch.Tensor: Tensor with same shape as input containing cumulative sum

    See Also:
        - :func:`~helion.language.associative_scan`: For custom scan operations
        - :func:`~helion.language.cumprod`: For cumulative product
        - :func:`~helion.language.reduce`: For dimension-reducing operations

    Note:
        - Output has same shape as input
        - Reverse=True computes cumsum from right to left
        - Equivalent to torch.cumsum
    """
    return associative_scan(torch.add, input_tensor, dim, reverse)


@_decorators.device_func_replacement(torch.cumprod)
def cumprod(
    input_tensor: torch.Tensor, dim: int, reverse: bool = False
) -> torch.Tensor:
    """
    Compute the cumulative product along a specified dimension.

    Equivalent to ``hl.associative_scan(torch.mul, input_tensor, dim, reverse)``.

    Args:
        input_tensor: Input tensor to compute cumulative product
        dim: The dimension along which to compute cumulative product
        reverse: If True, performs the cumprod in reverse order

    Returns:
        torch.Tensor: Tensor with same shape as input containing cumulative product

    See Also:
        - :func:`~helion.language.associative_scan`: For custom scan operations
        - :func:`~helion.language.cumsum`: For cumulative sum
        - :func:`~helion.language.reduce`: For dimension-reducing operations

    Note:
        - Output has same shape as input
        - Reverse=True computes cumprod from right to left
        - Equivalent to torch.cumprod
    """
    return associative_scan(torch.mul, input_tensor, dim, reverse)


@_decorators.api()
def _associative_scan(
    combine_graph_id: int,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
    is_tuple_input: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Device IR implementation of associative scan, not meant to be called directly."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_associative_scan)
def _(
    combine_graph_id: int,
    input_tensor: torch.Tensor | tuple[torch.Tensor, ...],
    dim: int,
    reverse: bool = False,
    is_tuple_input: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Fake implementation that returns a tensor/tuple with the same shape as input."""
    if is_tuple_input:
        assert isinstance(input_tensor, (tuple, list)), input_tensor
        return tuple(torch.empty_like(t) for t in input_tensor)
    assert isinstance(input_tensor, torch.Tensor), input_tensor
    return torch.empty_like(input_tensor)


@_decorators.codegen(_associative_scan, "triton")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    """Generate code for associative scan with combine function."""

    combine_graph_id = state.proxy_arg(0)
    dim = state.proxy_arg(2)
    reverse = state.proxy_arg(3)
    is_tuple_input = state.proxy_arg(4)

    input_tensor = _get_input_tensor_ast(state, bool(is_tuple_input))
    helper_func_name = _register_helper_function(state, cast("int", combine_graph_id))
    scan_expr = _create_scan_expression(
        input_tensor, cast("int", dim), helper_func_name, bool(reverse)
    )

    if is_tuple_input:
        return _create_tuple_result_expressions(state, scan_expr)
    return scan_expr


@_decorators.codegen(_associative_scan, "cute")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    from torch.fx.node import Node

    from .._compiler.ast_extension import expr_from_string
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.cute.indexing import CuteSortableLoad
    from .._compiler.device_ir import HelperFunctionGraphInfo

    combine_graph_id = cast("int", state.proxy_arg(0))
    dim = cast("int", state.proxy_arg(2))
    reverse = bool(state.proxy_arg(3))
    is_tuple_input = bool(state.proxy_arg(4))
    if is_tuple_input:
        return _cute_codegen_tuple_scan(state, combine_graph_id, dim, reverse)

    helper_graph_info = state.get_graph(combine_graph_id)
    assert isinstance(helper_graph_info, HelperFunctionGraphInfo)
    op = _scan_combine_operator(helper_graph_info)
    if op not in ("add", "max", "min", "mul"):
        raise exc.BackendUnsupported("cute", "associative_scan combine function")

    fx_node = state.fx_node
    if fx_node is None:
        raise exc.BackendUnsupported("cute", "associative_scan without FX node")
    input_node = fx_node.args[1]
    input_tensor = fx_node.meta["val"]
    if dim < 0:
        dim += input_tensor.ndim
    if dim != input_tensor.ndim - 1:
        # A non-last-dim scalar scan is a single-stream serial scan; reuse the
        # dim-agnostic machinery the tuple path already relies on (it folds the
        # combine graph over global rows along an arbitrary scan dimension).
        (result,) = _cute_codegen_serial_scan(
            state, helper_graph_info, [input_node], dim, reverse
        )
        return result

    scan_source = state.ast_args[1]
    sorted_source: tuple[CuteSortableLoad, bool] | None = None
    if (
        isinstance(input_node, Node)
        and input_node.target is operator.getitem
        and isinstance(input_node.args[0], Node)
        and input_node.args[0].target is torch.ops.aten.sort.default
    ):
        sort_node = input_node.args[0]
        load = sort_node.meta.get("cute_sort_load")
        descending = sort_node.meta.get("cute_sort_descending")
        if isinstance(load, CuteSortableLoad) and isinstance(descending, bool):
            sorted_source = (load, descending)

    if sorted_source is None:
        if not isinstance(scan_source, CuteSortableLoad):
            if isinstance(input_node, Node):
                scan_source = input_node.meta.get("cute_sortable_load")
        if not isinstance(scan_source, CuteSortableLoad):
            raise exc.BackendUnsupported("cute", "associative_scan input")
        load = scan_source
    else:
        load = sorted_source[0]

    env = CompileEnvironment.current()
    n = input_tensor.shape[-1]
    n_hint = env.size_hint(n) if isinstance(n, torch.SymInt) else n
    if not isinstance(n_hint, int):
        raise exc.BackendUnsupported("cute", "dynamic associative_scan extent")

    dtype_str = env.backend.dtype_str(input_tensor.dtype)
    index_dtype = env.backend.dtype_str(env.index_dtype)
    out_pos = state.device_function.new_var("scan_out_pos")
    scan_i = state.device_function.new_var("scan_i")
    acc = state.device_function.new_var("scan_acc")
    initialized = state.device_function.new_var("scan_initialized")
    include = state.device_function.new_var("scan_include")
    value = state.device_function.new_var("scan_value")

    state.codegen.add_statement(
        statement_from_string(
            f"{out_pos} = {index_dtype}({load.index_exprs[load.sort_index_pos]})"
        )
    )
    identity = "1" if op == "mul" else "0"
    state.codegen.add_statement(
        statement_from_string(f"{acc} = {dtype_str}({identity})")
    )
    state.codegen.add_statement(statement_from_string(f"{initialized} = False"))

    if op == "add":
        combine_expr = f"{acc} + {value}"
    elif op == "mul":
        combine_expr = f"{acc} * {value}"
    elif op == "max":
        combine_expr = f"{acc} if {acc} > {value} else {value}"
    elif op == "min":
        combine_expr = f"{acc} if {acc} < {value} else {value}"
    else:
        raise AssertionError(op)
    if sorted_source is not None:
        value_lines = _cute_sorted_value_lines(
            state, load, sorted_source[1], scan_i, value, n_hint
        )
    else:
        value_lines = [f"    {value} = {_cute_scan_load_expr(load, scan_i)}"]
    include_expr = f"{scan_i} >= {out_pos}" if reverse else f"{scan_i} <= {out_pos}"
    state.codegen.add_statement(
        statement_from_string(
            "\n".join(
                [
                    f"for {scan_i} in range(cutlass.Int32(0), cutlass.Int32({n_hint}), cutlass.Int32(1)):",
                    f"    {include} = {include_expr}",
                    *value_lines,
                    f"    {acc} = ({combine_expr}) if ({include} and {initialized}) else ({value} if {include} else {acc})",
                    f"    {initialized} = True if {include} else {initialized}",
                ]
            )
        )
    )
    return expr_from_string(acc)


def _cute_recover_scan_load(node: object) -> tuple[object, object] | None:
    """Walk back from a scan tuple-element node to its ``CuteSortableLoad``.

    The value stream is usually a direct ``load`` that already carries
    ``cute_sortable_load`` in its meta.  The index stream typically flows
    through dtype-cast / shape ops (``float().unsqueeze(1).expand_as(...)``);
    those are pass-throughs for a per-lane scalar in CuTe, so we follow them
    back to the underlying scalar load.

    Returns ``(CuteSortableLoad, load_node)`` or ``None`` if no load is found.
    """
    from torch.fx.node import Node

    from .._compiler.cute.indexing import CuteSortableLoad
    from .._compiler.cute.indexing import is_cute_shape_chain_target

    passthrough_targets = (torch.ops.prims.convert_element_type.default,)
    current = node
    seen: set[Node] = set()
    while isinstance(current, Node) and current not in seen:
        seen.add(current)
        load = current.meta.get("cute_sortable_load")
        if isinstance(load, CuteSortableLoad):
            return load, current
        target = current.target
        if is_cute_shape_chain_target(target) or target in passthrough_targets:
            if current.args and isinstance(current.args[0], Node):
                current = current.args[0]
                continue
        break
    return None


def _cute_strip_mask_term(mask_expr: str, scan_mask_var: str | None) -> str | None:
    """Drop the scan-dimension term from a combined ``and``-mask expression.

    The recovered load's mask is built per-lane and combines one boolean per
    indexed dimension (e.g. ``(mask_0) and (mask_1)``).  When re-loading at a
    different scan position the scan-dim term is replaced by an explicit
    ``scan_row < size`` check, so here we remove the original scan-dim mask
    var and keep only the dimension-constant terms.  Returns the remaining
    expression, or ``None`` if nothing is left.
    """
    if scan_mask_var is None:
        return mask_expr
    parts = [part.strip() for part in mask_expr.split(" and ")]
    kept = [part for part in parts if part.strip("()") != scan_mask_var]
    if not kept:
        return None
    return " and ".join(kept)


def _cute_scan_sort_index_pos(load: object, scan_index_var: str) -> int:
    """Index position in ``load.index_exprs`` matching the scan dimension.

    ``scan_index_var`` is the per-lane index variable for the dimension being
    scanned (e.g. ``indices_0``).  The position whose index expression equals
    that variable is the one we sweep over during the serial scan.
    """
    from .._compiler.cute.indexing import CuteSortableLoad

    assert isinstance(load, CuteSortableLoad)
    for pos, expr in enumerate(load.index_exprs):
        if expr == scan_index_var:
            return pos
    # Fall back to the load's own recorded sort position (e.g. a 1D load whose
    # sole index is the scan dimension but was renamed by an upstream cast).
    return load.sort_index_pos


def _cute_inline_combine_graph(
    state: CodegenState,
    helper_graph_info: object,
    left_vars: list[str],
    right_vars: list[str],
) -> tuple[list[str], list[str]]:
    """Inline a tuple combine graph as CuTe scalar expressions.

    ``left_vars``/``right_vars`` are the per-lane scalar variable names holding
    the left (accumulator) and right (incoming) tuple elements.  Returns a
    tuple ``(body_lines, out_exprs)`` where ``body_lines`` are 4-space-indented
    assignment statements (to be spliced inside the scan ``for`` loop) and
    ``out_exprs`` are the output expression strings, one per tuple element.
    """
    import operator as operator_mod

    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.device_ir import HelperFunctionGraphInfo

    assert isinstance(helper_graph_info, HelperFunctionGraphInfo)
    env = CompileEnvironment.current()
    graph = helper_graph_info.graph

    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    num_inputs = len(placeholders)
    assert num_inputs == len(left_vars) + len(right_vars), (
        "combine graph arity does not match scan tuple width"
    )
    # Unpacked layout: (left_e0, left_e1, ..., right_e0, right_e1, ...)
    arg_vars = [*left_vars, *right_vars]

    binary_ops: dict[object, str] = {
        operator_mod.add: "+",
        torch.add: "+",
        torch.ops.aten.add.Tensor: "+",
        torch.ops.aten.add.Scalar: "+",
        operator_mod.sub: "-",
        torch.sub: "-",
        torch.ops.aten.sub.Tensor: "-",
        torch.ops.aten.sub.Scalar: "-",
        operator_mod.mul: "*",
        torch.mul: "*",
        torch.ops.aten.mul.Tensor: "*",
        torch.ops.aten.mul.Scalar: "*",
        torch.ops.aten.eq.Tensor: "==",
        torch.ops.aten.eq.Scalar: "==",
        torch.ops.aten.ne.Tensor: "!=",
        torch.ops.aten.ne.Scalar: "!=",
        torch.ops.aten.lt.Tensor: "<",
        torch.ops.aten.lt.Scalar: "<",
        torch.ops.aten.gt.Tensor: ">",
        torch.ops.aten.gt.Scalar: ">",
        torch.ops.aten.le.Tensor: "<=",
        torch.ops.aten.le.Scalar: "<=",
        torch.ops.aten.ge.Tensor: ">=",
        torch.ops.aten.ge.Scalar: ">=",
    }
    min_ops = (torch.minimum, torch.ops.aten.minimum.default)
    max_ops = (torch.maximum, torch.ops.aten.maximum.default)
    where_ops = (torch.where, torch.ops.aten.where.self)

    env_map: dict[object, str] = dict(zip(placeholders, arg_vars, strict=True))

    def operand(value: object) -> str:
        if isinstance(value, torch.fx.Node):
            assert value in env_map, f"unresolved combine node: {value}"
            return env_map[value]
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, float)):
            return repr(value)
        raise exc.BackendUnsupported(
            "cute", f"associative_scan combine operand {value!r}"
        )

    lines: list[str] = []

    def emit(node: torch.fx.Node, expr: str) -> None:
        var = state.device_function.new_var("scan_combine")
        lines.append(f"    {var} = {expr}")
        env_map[node] = var

    for node in graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op != "call_function":
            raise exc.BackendUnsupported(
                "cute", f"associative_scan combine op {node.op}"
            )
        target = node.target
        if target in binary_ops:
            lhs = operand(node.args[0])
            rhs = operand(node.args[1])
            emit(node, f"({lhs}) {binary_ops[target]} ({rhs})")
        elif target in where_ops:
            cond = operand(node.args[0])
            tval = operand(node.args[1])
            fval = operand(node.args[2])
            emit(node, env.backend.where_expr(cond, tval, fval))
        elif target in min_ops:
            lhs = operand(node.args[0])
            rhs = operand(node.args[1])
            emit(node, f"({lhs}) if ({lhs}) < ({rhs}) else ({rhs})")
        elif target in max_ops:
            lhs = operand(node.args[0])
            rhs = operand(node.args[1])
            emit(node, f"({lhs}) if ({lhs}) > ({rhs}) else ({rhs})")
        elif target is torch.ops.prims.convert_element_type.default:
            # Per-lane scalars are already in their storage dtype; treat the
            # cast as a pass-through (matches the load/store scalar pipeline).
            env_map[node] = operand(node.args[0])
        else:
            raise exc.BackendUnsupported(
                "cute", f"associative_scan combine function: {target}"
            )

    output_nodes = next(n for n in graph.nodes if n.op == "output")
    outputs = output_nodes.args[0]
    # A scalar (single-tensor) combine graph returns one value directly; a tuple
    # combine graph returns a tuple/list.  Normalize to a list of outputs.
    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]
    out_exprs = [operand(o) for o in outputs]
    return lines, out_exprs


def _cute_codegen_tuple_scan(
    state: CodegenState,
    combine_graph_id: int,
    dim: int,
    reverse: bool,
) -> list[ast.AST]:
    """CuTe codegen for ``hl.associative_scan`` over a tuple of streams.

    Implements a serial per-lane inclusive scan: for each output position
    ``out_pos`` along the scan dimension, fold the user's combine graph over
    elements ``0..out_pos`` (inclusive), carrying one accumulator per tuple
    element.  This mirrors the scalar CuTe scan but supports an arbitrary
    (non-monoid) combine and multiple parallel streams (e.g. value + index for
    segmented reduction).  Correctness, not performance, is the goal: the loop
    is O(n) per lane.
    """
    from .._compiler.device_ir import HelperFunctionGraphInfo

    helper_graph_info = state.get_graph(combine_graph_id)
    assert isinstance(helper_graph_info, HelperFunctionGraphInfo)

    fx_node = state.fx_node
    if fx_node is None:
        raise exc.BackendUnsupported("cute", "associative_scan without FX node")
    input_nodes = fx_node.args[1]
    if not isinstance(input_nodes, (tuple, list)):
        raise exc.BackendUnsupported("cute", "tuple associative_scan input")

    return _cute_codegen_serial_scan(
        state, helper_graph_info, list(input_nodes), dim, reverse
    )


def _cute_codegen_serial_scan(
    state: CodegenState,
    helper_graph_info: object,
    input_nodes: list[object],
    dim: int,
    reverse: bool,
) -> list[ast.AST]:
    """Dim-agnostic serial per-lane inclusive scan shared by the scalar and
    tuple CuTe scan paths.

    ``input_nodes`` is one FX node per scanned stream (a single element for a
    scalar scan, multiple for a tuple scan).  For each output position along the
    scan dimension this folds the user's combine graph over the global rows
    ``0..out_pos`` (inclusive), carrying one accumulator per stream.  Returns one
    output expression per stream.
    """
    from torch.fx.node import Node

    from .._compiler.ast_extension import expr_from_string
    from .._compiler.ast_extension import statement_from_string
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.cute.indexing import CuteSortableLoad

    # Fake-tensor metadata lives on the per-stream input nodes (the scan node
    # itself produces a tuple and may carry no ``val``).
    vals = [cast("torch.fx.Node", n).meta["val"] for n in input_nodes]
    first_val = vals[0]
    ndim = first_val.ndim
    if dim < 0:
        dim += ndim

    env = CompileEnvironment.current()
    from .._compiler.cute.cute_reshape import _get_dim_local_coord
    from .._compiler.cute.cute_reshape import _resolve_dim_block_id
    from .memory_ops import _cute_active_index_var
    from .memory_ops import _cute_tensor_dim_size_expr

    scan_block_id = _resolve_dim_block_id(state.codegen, first_val, dim)
    if scan_block_id is None:
        raise exc.BackendUnsupported("cute", "associative_scan scan-dim block id")

    # The scan loops over the *block-local* positions of the scan dimension; the
    # loop bound is the scan dim's block (tile) size.  Prefer the concrete block
    # size from config (the fake tensor's scan dim may be a fresh symbol that
    # ``size_hint`` cannot resolve to the tile size), falling back to the fake
    # extent's size hint.
    block_size = env.block_sizes[scan_block_id].from_config(
        state.device_function.config
    )
    if isinstance(block_size, int):
        n_hint = block_size
    else:
        extent = first_val.shape[dim]
        n_hint = env.size_hint(extent) if isinstance(extent, torch.SymInt) else extent
    if not isinstance(n_hint, int):
        raise exc.BackendUnsupported("cute", "dynamic associative_scan extent")

    # The current lane's block-local position along the scan dim.
    out_pos_expr = _get_dim_local_coord(state.codegen, first_val, dim)
    # The tile's global base along the scan dim, so a local ``scan_i`` maps to the
    # global row ``base + scan_i``.  ``offset_var`` aliases the per-thread global
    # index when the scan dim is split across threads (it equals
    # ``global_index``), so derive the base as ``global_index - local_coord``
    # which is the genuine tile base in both the looped and thread-split cases.
    scan_global_index_var = _cute_active_index_var(state, scan_block_id)
    if scan_global_index_var is not None:
        offset_expr = f"({scan_global_index_var}) - ({out_pos_expr})"
    else:
        offset_expr = state.codegen.offset_var(scan_block_id)

    # Recover a scalar load per tuple element and the index position that
    # corresponds to the scan dimension within that load.
    loads: list[CuteSortableLoad] = []
    load_nodes: list[object] = []
    sort_positions: list[int] = []
    for node in input_nodes:
        recovered = _cute_recover_scan_load(node)
        if recovered is None or not isinstance(recovered[0], CuteSortableLoad):
            raise exc.BackendUnsupported("cute", "tuple associative_scan input load")
        load, load_node = recovered
        assert isinstance(load, CuteSortableLoad)
        loads.append(load)
        load_nodes.append(load_node)
        if scan_global_index_var is not None:
            sort_positions.append(
                _cute_scan_sort_index_pos(load, scan_global_index_var)
            )
        else:
            sort_positions.append(load.sort_index_pos)

    index_dtype = env.backend.dtype_str(env.index_dtype)
    out_pos = state.device_function.new_var("scan_out_pos")
    scan_i = state.device_function.new_var("scan_i")
    scan_row = state.device_function.new_var("scan_row")
    include = state.device_function.new_var("scan_include")
    initialized = state.device_function.new_var("scan_initialized")

    acc_vars = [state.device_function.new_var("scan_acc") for _ in loads]
    value_vars = [state.device_function.new_var("scan_value") for _ in loads]

    state.codegen.add_statement(
        statement_from_string(f"{out_pos} = {index_dtype}({out_pos_expr})")
    )
    for acc_var, val in zip(acc_vars, vals, strict=True):
        dtype_str = env.backend.dtype_str(val.dtype)
        state.codegen.add_statement(
            statement_from_string(f"{acc_var} = {dtype_str}(0)")
        )
    state.codegen.add_statement(statement_from_string(f"{initialized} = False"))

    # Global scan row for this iteration: ``offset + scan_i``.
    row_line = f"    {scan_row} = cutlass.Int32({offset_expr}) + {scan_i}"

    # Per-iteration value loads (re-load each stream at the scanned global row).
    # Cast each loaded scalar to the *scanned* element dtype (``vals[i].dtype``):
    # the recovered load may have a different storage dtype than the value
    # entering the scan (e.g. the index stream is ``indices`` int64 but scanned
    # as float32 after ``idxs.float()``).  Each load is guarded so an
    # out-of-range scanned row (partial final tile) reads 0 instead of faulting.
    value_lines: list[str] = []
    for val_var, load, load_node, pos, val in zip(
        value_vars, loads, load_nodes, sort_positions, vals, strict=True
    ):
        scan_dtype_str = env.backend.dtype_str(val.dtype)
        load_dtype_str = env.backend.dtype_str(load.dtype)
        index_exprs = list(load.index_exprs)
        index_exprs[pos] = scan_row
        load_expr = f"{load.tensor_name}[{', '.join(index_exprs)}]"
        # Rebuild the mask for the scanned row: the scan-dim bound becomes
        # ``scan_row < tensor_size``; any non-scan-dim masks (e.g. the feature
        # column bound) are constant w.r.t. ``scan_i`` and reused as-is.
        load_tensor = (
            load_node.args[0].meta["val"]
            if isinstance(load_node, Node)
            and load_node.args
            and isinstance(load_node.args[0], Node)
            else None
        )
        scan_dim_mask: str | None = None
        if isinstance(load_tensor, torch.Tensor):
            size_expr = _cute_tensor_dim_size_expr(state, load_tensor, pos)
            scan_dim_mask = f"({scan_row}) < cutlass.Int32({size_expr})"
        mask_terms: list[str] = []
        if scan_dim_mask is not None:
            mask_terms.append(scan_dim_mask)
        if load.mask_expr is not None and scan_global_index_var is not None:
            scan_mask_var = state.codegen.mask_var(scan_block_id)
            non_scan_mask = _cute_strip_mask_term(load.mask_expr, scan_mask_var)
            if non_scan_mask is not None:
                mask_terms.append(non_scan_mask)
        if mask_terms:
            mask_expr = " and ".join(f"({term})" for term in mask_terms)
            load_expr = f"({load_expr} if {mask_expr} else {load_dtype_str}(0))"
        value_lines.append(f"    {val_var} = {scan_dtype_str}({load_expr})")

    include_expr = f"{scan_i} >= {out_pos}" if reverse else f"{scan_i} <= {out_pos}"

    # Inline the user's combine graph (one statement per node, 4-space
    # indented to sit inside the scan ``for`` loop).
    combine_lines, combine_exprs = _cute_inline_combine_graph(
        state, helper_graph_info, acc_vars, value_vars
    )
    # Fold: when this position is included and we already have a running prefix,
    # combine; on the first included element seed the accumulator with the
    # value.  Stage combined results so reads of the old accumulator inside the
    # combine graph see the prefix from before this step.
    next_vars = [state.device_function.new_var("scan_next") for _ in acc_vars]
    fold_lines: list[str] = []
    for next_var, acc_var, val_var, combine_expr in zip(
        next_vars, acc_vars, value_vars, combine_exprs, strict=True
    ):
        fold_lines.append(
            f"    {next_var} = ({combine_expr}) if ({include} and {initialized}) "
            f"else ({val_var} if {include} else {acc_var})"
        )
    for next_var, acc_var in zip(next_vars, acc_vars, strict=True):
        fold_lines.append(f"    {acc_var} = {next_var}")
    fold_lines.append(f"    {initialized} = True if {include} else {initialized}")

    state.codegen.add_statement(
        statement_from_string(
            "\n".join(
                [
                    f"for {scan_i} in range(cutlass.Int32(0), cutlass.Int32({n_hint}), cutlass.Int32(1)):",
                    f"    {include} = {include_expr}",
                    row_line,
                    *value_lines,
                    *combine_lines,
                    *fold_lines,
                ]
            )
        )
    )

    return [expr_from_string(acc_var) for acc_var in acc_vars]


def _get_input_tensor_ast(state: CodegenState, is_tuple_input: bool) -> ast.AST:
    """Get the input tensor AST, handling tuple inputs specially."""
    if not is_tuple_input:
        return state.ast_arg(1)

    raw_input = state.ast_args[1]
    if isinstance(raw_input, tuple):
        from .._compiler.ast_extension import create

        tuple_elts = [
            elt if isinstance(elt, ast.AST) else ast.Constant(value=elt)
            for elt in raw_input
        ]
        return create(ast.Tuple, elts=tuple_elts, ctx=ast.Load())
    return state.ast_arg(1)


def _scan_combine_operator(helper_graph_info: HelperFunctionGraphInfo) -> str:
    import operator as operator_mod

    graph = helper_graph_info.graph
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in (
            operator_mod.add,
            torch.add,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add.Scalar,
        ):
            return "add"
        if node.target in (
            operator_mod.mul,
            torch.mul,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul.Scalar,
        ):
            return "mul"
        if node.target in (
            torch.maximum,
            torch.ops.aten.maximum.default,
        ):
            return "max"
        if node.target in (
            torch.minimum,
            torch.ops.aten.minimum.default,
        ):
            return "min"
    raise exc.BackendUnsupported("cute", "associative_scan combine graph")


def _cute_scan_load_expr(load: object, index: str) -> str:
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.cute.indexing import CuteSortableLoad

    assert isinstance(load, CuteSortableLoad)
    index_exprs = list(load.index_exprs)
    index_exprs[load.sort_index_pos] = index
    expr = f"{load.tensor_name}[{', '.join(index_exprs)}]"
    if load.mask_expr is not None:
        dtype_str = CompileEnvironment.current().backend.dtype_str(load.dtype)
        return f"({expr} if {load.mask_expr} else {dtype_str}(0))"
    return expr


def _cute_sorted_value_lines(
    state: CodegenState,
    load: object,
    descending: bool,
    out_pos: str,
    output_var: str,
    n_hint: int,
) -> list[str]:
    from .._compiler.compile_environment import CompileEnvironment
    from .._compiler.cute.indexing import CuteSortableLoad

    assert isinstance(load, CuteSortableLoad)
    env = CompileEnvironment.current()
    dtype_str = env.backend.dtype_str(load.dtype)
    index_dtype = env.backend.dtype_str(env.index_dtype)
    sorted_value = state.device_function.new_var("scan_sorted_value")
    candidate = state.device_function.new_var("scan_sort_k")
    probe = state.device_function.new_var("scan_sort_j")
    candidate_rank = state.device_function.new_var("scan_sort_rank")
    candidate_value = state.device_function.new_var("scan_sort_candidate")
    probe_value = state.device_function.new_var("scan_sort_probe")
    before = state.device_function.new_var("scan_sort_before")
    selected = state.device_function.new_var("scan_sort_selected")
    cmp_op = ">" if descending else "<"
    return [
        f"    {sorted_value} = {dtype_str}(0)",
        f"    for {candidate} in range(cutlass.Int32(0), cutlass.Int32({n_hint}), cutlass.Int32(1)):",
        f"        {candidate_value} = {_cute_scan_load_expr(load, candidate)}",
        f"        {candidate_rank} = {index_dtype}(0)",
        f"        for {probe} in range(cutlass.Int32(0), cutlass.Int32({n_hint}), cutlass.Int32(1)):",
        f"            {probe_value} = {_cute_scan_load_expr(load, probe)}",
        f"            {before} = ({probe_value} {cmp_op} {candidate_value}) or (({probe_value} == {candidate_value}) and ({probe} < {candidate}))",
        f"            {candidate_rank} = {candidate_rank} + ({index_dtype}(1) if {before} else {index_dtype}(0))",
        f"        {selected} = {candidate_rank} == {out_pos}",
        f"        {sorted_value} = {candidate_value} if {selected} else {sorted_value}",
        f"    {output_var} = {sorted_value}",
    ]


def _register_helper_function(state: CodegenState, combine_graph_id: int) -> str:
    """Register the helper function and return its final name."""
    from .._compiler.device_ir import HelperFunctionGraphInfo

    helper_graph_info = state.get_graph(combine_graph_id)
    assert isinstance(helper_graph_info, HelperFunctionGraphInfo)
    state.codegen.device_function.register_helper_function(helper_graph_info)
    # Get the final name from the helper manager (which uses the namespace)
    return state.codegen.device_function.helper_manager.get_final_name(
        helper_graph_info
    )


def _create_scan_expression(
    input_tensor: ast.AST, dim: int, helper_func_name: str, reverse: bool
) -> ast.AST:
    """Create the tl.associative_scan expression."""
    from .._compiler.ast_extension import expr_from_string

    template = (
        f"tl.associative_scan({{input_tensor}}, {{dim_value}}, {helper_func_name}, reverse=True)"
        if reverse
        else f"tl.associative_scan({{input_tensor}}, {{dim_value}}, {helper_func_name})"
    )
    return expr_from_string(
        template,
        input_tensor=input_tensor,
        dim_value=ast.Constant(value=dim),
    )


def _create_tuple_result_expressions(
    state: CodegenState, scan_expr: ast.AST
) -> list[ast.AST]:
    """Create getitem expressions for tuple results."""
    from .._compiler.ast_extension import expr_from_string

    raw_input = state.ast_args[1]
    num_elements = len(raw_input) if isinstance(raw_input, tuple) else 2

    return [
        expr_from_string(
            "{scan_result}[{index}]", scan_result=scan_expr, index=ast.Constant(value=i)
        )
        for i in range(num_elements)
    ]
