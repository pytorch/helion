"""CuTe-backend codegen for ops defined in ``helion.language.scan_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``scan_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _decorators
from ...language.scan_ops import _associative_scan

if TYPE_CHECKING:
    import ast

    from ..device_ir import HelperFunctionGraphInfo
    from ..inductor_lowering import CodegenState


@_decorators.codegen(_associative_scan, "cute")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    from torch.fx.node import Node

    from ..ast_extension import expr_from_string
    from ..ast_extension import statement_from_string
    from ..compile_environment import CompileEnvironment
    from ..device_ir import HelperFunctionGraphInfo
    from .indexing import CuteSortableLoad

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

    from .indexing import CuteSortableLoad
    from .indexing import is_cute_shape_chain_target

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
    from .indexing import CuteSortableLoad

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

    from ..compile_environment import CompileEnvironment
    from ..device_ir import HelperFunctionGraphInfo

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
    from ..device_ir import HelperFunctionGraphInfo

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

    from ..ast_extension import expr_from_string
    from ..ast_extension import statement_from_string
    from ..compile_environment import CompileEnvironment
    from .indexing import CuteSortableLoad

    # Fake-tensor metadata lives on the per-stream input nodes (the scan node
    # itself produces a tuple and may carry no ``val``).
    vals = [cast("torch.fx.Node", n).meta["val"] for n in input_nodes]
    first_val = vals[0]
    ndim = first_val.ndim
    if dim < 0:
        dim += ndim

    env = CompileEnvironment.current()
    from ...language.memory_ops import _cute_active_index_var
    from ...language.memory_ops import _cute_tensor_dim_size_expr
    from .cute_reshape import _get_dim_local_coord
    from .cute_reshape import _resolve_dim_block_id

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
    from ..compile_environment import CompileEnvironment
    from .indexing import CuteSortableLoad

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
    from ..compile_environment import CompileEnvironment
    from .indexing import CuteSortableLoad

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
