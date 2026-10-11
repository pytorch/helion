"""CuTe-backend codegen for ops defined in ``helion.language.scan_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``scan_ops`` imports it at the bottom so registration keeps the
same eager timing as before.

Lowering strategy
-----------------

``hl.associative_scan`` first tries the register / warp-shuffle lowering in
:func:`_cute_try_parallel_scan`, which reuses the per-lane values the tile body
already loaded and never re-reads global memory:

* **Lane-looped scan axis** (one thread walks the whole axis in a lane loop):
  a loop-carried in-thread prefix per stream, ``acc = combine(acc, value)``,
  seeded on the first lane.  When lane / vector loops over *other* axes are
  nested inside the scan lane loop the prefix lives in a small register
  fragment indexed by those inner lane coordinates (one slot per column).
* **Thread-split scan axis** (``T`` threads on CUDA thread axis 0): each lane
  step is a chunk of ``T`` consecutive elements, scanned with a Kogge-Stone
  warp scan (``log2(T)`` ``shfl.up`` / ``shfl.down`` steps applying the inlined
  combine).  A blocked layout with more than one element per thread is *not*
  handled here (it needs a second pass) and keeps the serial fallback; the
  strided layout visits chunks in order and carries the previous chunk's total
  across lane steps.  When the axis spans more than one warp the warp totals
  are exchanged through shared memory (two ``sync_threads`` per chunk).
* **Reverse scans** mirror the shuffles (``shfl.down``) and, for lane-looped
  axes, ask the tile strategy to visit the lanes in descending order.  A
  lane-looped scan axis that is also *vectorised* (one thread, ``V``-wide
  vectors) keeps the serial fallback for reverse scans: the vector store
  protocol collects the ``V`` results of the constexpr vector loop in
  iteration order, so that loop cannot run backwards without permuting every
  stored vector.

Any shape the geometry cannot prove (dynamic extent, unresolvable block id,
blocked multi-element thread split, vectorised thread-split axis, scan axis on
a thread axis other than 0, nested device loops inside the scan lane loop, ...)
falls back to :func:`_cute_codegen_serial_scan`, the dim-agnostic O(n) per
element rescan.  So does a scan emitted inside a lane loop that also hosts a
reduction over a lane-looped block (``hl.cumsum(row) / row.sum()``): the
two-pass lane-reduction split cannot carry the scan prefix across its
accumulate / consume passes (see :func:`_cute_scan_lane_hosts_reduction`).
The serial fallback visits the scanned positions in scan order too, i.e.
descending for a reverse scan, so both lowerings apply a non-commutative
combine with the same operand order.

Numerics: the parallel lowering only reassociates the combine (tree order
inside a warp, chunk order across lane steps); it never introduces
approximations or assumes ``0`` is an identity.  Padded / out-of-range rows of
a partial final tile are excluded through the block mask: for a forward scan
the mask is a suffix of the axis, so those rows can only feed other masked
rows and need no bookkeeping; for a reverse scan they precede every valid row
and are skipped through an explicit per-element validity flag carried next to
the values.  The combine is always applied as ``combine(earlier, current)``
in scan order, i.e. for a reverse scan ``left`` holds the suffix of the
higher-index elements, matching ``torch.associative_scan(reverse=True)``.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import operator
import re
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _decorators
from ...language.scan_ops import _associative_scan

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..device_ir import HelperFunctionGraphInfo
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState
    from .indexing import CuteSortableLoad


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
    input_value = state.proxy_arg(1)
    inputs = input_value if is_tuple_input else (input_value,)
    assert isinstance(inputs, (tuple, list))
    if all(
        CompileEnvironment.current().known_equal(
            cast("torch.Tensor", value).size(dim), 1
        )
        for value in inputs
    ):
        # A singleton inclusive scan seeds from its only element and never
        # invokes the combine. Preserve the already-lowered value, including
        # its source masks, dtype and payload; no scan coordinate is needed.
        if is_tuple_input:
            expressions = state.ast_args[1]
            assert isinstance(expressions, (tuple, list))
            return [cast("ast.AST", expression) for expression in expressions]
        return state.ast_arg(1)

    helper_graph_info = state.get_graph(combine_graph_id)
    assert isinstance(helper_graph_info, HelperFunctionGraphInfo)
    fx_node = state.fx_node
    if fx_node is None:
        raise exc.BackendUnsupported("cute", "associative_scan without FX node")
    raw_inputs = fx_node.args[1]
    input_nodes: list[object] = (
        list(raw_inputs) if isinstance(raw_inputs, (tuple, list)) else [raw_inputs]
    )
    parallel = _cute_try_parallel_scan(
        state, helper_graph_info, input_nodes, dim, reverse
    )
    if not isinstance(parallel, str):
        return parallel if is_tuple_input else parallel[0]

    if is_tuple_input:
        return _cute_codegen_tuple_scan(
            state, combine_graph_id, dim, reverse, parallel_declined=parallel
        )

    input_node = fx_node.args[1]
    input_tensor = fx_node.meta["val"]
    if dim < 0:
        dim += input_tensor.ndim
    sorted_source: tuple[CuteSortableLoad, bool, CuteSerialAxis] | None = None
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
            sorted_source = (load, descending, sort_node.meta["cute_sort_axis"])
    if sorted_source is None or dim != input_tensor.ndim - 1:
        # The dim-agnostic serial scan folds the combine graph over the tile's
        # block-local rows and handles tile offsets and partial-tile masks, so
        # it serves scalar scans too (a last-dim rescan indexed by the global
        # position would read the first tile's rows for every other tile).
        # Only a scan over ``torch.sort`` output keeps the dedicated rescan
        # below, whose values come from the rank machinery rather than a load.
        (result,) = _cute_codegen_serial_scan(
            state,
            helper_graph_info,
            [input_node],
            dim,
            reverse,
            parallel_declined=parallel,
        )
        return result

    op = _scan_combine_operator(helper_graph_info)
    if op not in ("add", "max", "min", "mul"):
        raise exc.BackendUnsupported("cute", "associative_scan combine function")
    load, descending, axis = sorted_source
    n_hint = axis.extent

    env = CompileEnvironment.current()

    dtype_str = env.backend.dtype_str(input_tensor.dtype)
    index_dtype = env.backend.dtype_str(env.index_dtype)
    out_pos = state.device_function.new_var("scan_out_pos")
    scan_i = state.device_function.new_var("scan_i")
    acc = state.device_function.new_var("scan_acc")
    initialized = state.device_function.new_var("scan_initialized")
    include = state.device_function.new_var("scan_include")
    value = state.device_function.new_var("scan_value")

    state.codegen.add_statement(
        statement_from_string(f"{out_pos} = {index_dtype}({axis.position})")
    )
    identity = "1" if op == "mul" else "0"
    state.codegen.add_statement(
        statement_from_string(f"{acc} = {dtype_str}({identity})")
    )
    state.codegen.add_statement(statement_from_string(f"{initialized} = False"))
    position, position_lines = _cute_serial_scan_position(
        state, scan_i, n_hint, reverse
    )

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
    value_lines = _cute_sorted_value_lines(
        state, load, axis, descending, position, value
    )
    include_expr = _cute_scan_include_expr(axis, position, out_pos, reverse)
    state.codegen.add_statement(
        statement_from_string(
            "\n".join(
                [
                    f"for {scan_i} in range(cutlass.Int32(0), cutlass.Int32({n_hint}), cutlass.Int32(1)):",
                    *position_lines,
                    f"    {include} = {include_expr}",
                    *value_lines,
                    f"    {acc} = ({combine_expr}) if ({include} and {initialized}) else ({value} if {include} else {acc})",
                    f"    {initialized} = True if {include} else {initialized}",
                ]
            )
        )
    )
    return expr_from_string(acc)


def _cute_scan_include_expr(
    axis: CuteSerialAxis, position: str, out_pos: str, reverse: bool
) -> str:
    """Whether the scan at ``out_pos`` folds in ``position``.

    A reverse scan folds the positions after its own, which run past the
    tile's last element on a masked block; those hold no element (a padded
    zero would still fold into a product or a max).
    """
    include = f"{position} >= {out_pos}" if reverse else f"{position} <= {out_pos}"
    if reverse and (valid := axis.valid(position)) is not None:
        return f"({include}) and ({valid})"
    return include


def _cute_serial_scan_position(
    state: CodegenState, step: str, extent: int | str, reverse: bool
) -> tuple[str, list[str]]:
    """Block-local position scanned by serial step ``step`` of the fallback.

    The serial fallbacks fold ``acc = combine(acc, value)`` while walking the
    axis, so the walk order is the operand order of a non-commutative combine.
    A forward scan walks ascending (``position == step``); a reverse scan must
    walk descending so ``acc`` holds the higher-index suffix that the reference
    passes as the left operand.  Returns the position variable and the loop
    body lines (4-space indented) that define it.
    """
    if not reverse:
        return step, []
    position = state.device_function.new_var("scan_index")
    last = extent - 1 if isinstance(extent, int) else f"({extent}) - 1"
    return position, [f"    {position} = cutlass.Int32({last}) - {step}"]


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
    different scan position the scan-dim term is replaced by bounds on the
    re-read index (``cute_reread_index``: ``index >= 0``, ``index < size``)
    and on the block's end (``CuteSerialAxis.valid``), so here we remove the
    original scan-dim mask var and keep only the dimension-constant terms.
    Returns the remaining expression, or ``None`` if nothing is left.
    """
    if scan_mask_var is None:
        return mask_expr
    parts = [part.strip() for part in mask_expr.split(" and ")]
    kept = [part for part in parts if part.strip("()") != scan_mask_var]
    if not kept:
        return None
    return " and ".join(kept)


@dataclasses.dataclass(frozen=True)
class CuteSerialAxis:
    """The walk of a serial re-read along one dim of a tile value.

    The serial scan and the rank sort re-read a load at every position of the
    dim: position ``p`` is the element whose block index variable is
    ``base + p``, and this lane holds position ``position``.  The walk runs
    over ``extent`` positions; with a masked block only those whose index is
    below ``end`` hold an element (a partial tile, a slice shorter than its
    block's lanes, a tile range shorter than the tensor).
    """

    block_id: int
    extent: int | str
    position: str
    base: str
    index_var: str
    end: str | None

    def valid(self, position: str) -> str | None:
        """Whether ``position`` holds an element, or None if every one does."""
        if self.end is None:
            return None
        return f"(({self.base}) + ({position})) < ({self.end})"


def cute_serial_axis(
    state: CodegenState, value: torch.Tensor, dim: int, what: str
) -> CuteSerialAxis:
    """The serial walk of dim ``dim`` of ``value`` in the current scope."""
    from ...language.memory_ops import _cute_active_index_var
    from ..compile_environment import CompileEnvironment
    from ..tile_strategy import PerThreadFlattenedTileStrategy
    from .cute_reshape import _get_dim_local_coord
    from .cute_reshape import _resolve_dim_block_id
    from .tile_ops import cute_masked_block_end

    env = CompileEnvironment.current()
    block_id = _resolve_dim_block_id(state.codegen, value, dim)
    if block_id is None:
        raise exc.BackendUnsupported("cute", f"{what} dim block id")
    index_var = _cute_active_index_var(state, block_id)
    if index_var is None:
        raise exc.BackendUnsupported("cute", f"{what} dim is not active here")

    # The walk covers the dim's block (tile): prefer the block size from
    # config, keeping a symbolic host constexpr one symbolic (the fake
    # tensor's dim may be a fresh symbol whose hint is not the tile size),
    # falling back to the fake extent's size hint.
    block_size = env.block_sizes[block_id].from_config(state.device_function.config)
    extent: int | str
    if isinstance(block_size, int):
        extent = block_size
    elif isinstance(block_size, torch.SymInt):
        extent = state.device_function.literal_expr(block_size)
    else:
        size = value.shape[dim]
        hint = env.size_hint(size) if isinstance(size, torch.SymInt) else size
        if not isinstance(hint, int):
            raise exc.BackendUnsupported("cute", f"dynamic {what} extent")
        extent = hint
    block = env.block_sizes[block_id]
    if block.reduction and isinstance(block.size, int) and isinstance(extent, int):
        # A reduction dim (a slice's) can launch more lanes than it has
        # elements; the walk covers its elements only.
        extent = min(extent, block.size)

    # Prefer the strategy's uniform tile base: the generic local-coordinate
    # helper reports the *thread* coordinate for a lane-looped flattened tile
    # (and zero for its vectorised form), which is not the element's position.
    # Without a known base fall back to the helper and derive the base as
    # ``index - local_coord``.  A flattened multi-dim tile has neither: it is
    # a run of the flat iteration space, whose ``offsets % BLOCK // ...``
    # coordinate in a permuted loop order is no tile position at all (a
    # re-read at it would address negative columns).
    tile_base = _cute_scan_tile_base(state, block_id)
    if tile_base is not None:
        position = f"({index_var}) - ({tile_base})"
    else:
        loops = state.codegen.active_device_loops.get(block_id)
        strategy = loops[-1].strategy if loops else None
        if (
            isinstance(strategy, PerThreadFlattenedTileStrategy)
            and len(strategy.block_ids) > 1
        ):
            raise exc.BackendUnsupported(
                "cute",
                f"{what} along a dim of a flattened multi-dim tile, which has "
                "no per-dim tile base for the serial walk",
            )
        position = _get_dim_local_coord(state.codegen, value, dim)
        tile_base = f"({index_var}) - ({position})"
    return CuteSerialAxis(
        block_id,
        extent,
        position,
        tile_base,
        index_var,
        cute_masked_block_end(state.codegen, block_id, what),
    )


def cute_reread_index(
    state: CodegenState,
    load: CuteSortableLoad,
    axis: CuteSerialAxis,
    position: str,
) -> tuple[list[str], list[str]]:
    """``load``'s index and mask terms at ``position`` of ``axis``.

    The dim's index is a function of the axis' block index variable
    (``x[t, 1::2]`` reads ``1 + 2 * indices_1``): the variable takes
    ``base + position``, rather than the whole index being replaced, so a
    slice's start and step are kept.  The mask bounds the position by the
    block's own end (``CuteSerialAxis.valid``) and the address by the tensor
    dim, and keeps the load's other terms.  Those are evaluated at this
    lane's own element, which is right for the other dims' masks and the
    lane-validity bounds, but not for an ``extra_mask`` that varies along
    the walked dim: such a load is refused.
    """
    from ...language.memory_ops import _cute_tensor_dim_size_expr
    from ..compile_environment import CompileEnvironment
    from .cute_reshape import _resolve_dim_block_id

    pattern = re.compile(rf"\b{re.escape(axis.index_var)}\b")
    positions = [
        pos for pos, expr in enumerate(load.index_exprs) if pattern.search(expr)
    ]
    if len(positions) != 1:
        raise exc.BackendUnsupported(
            "cute",
            "a serial re-read of a load whose index does not follow the walked "
            "dim through exactly one subscript",
        )
    if load.extra_mask is not None:
        env = CompileEnvironment.current()
        value = load.value
        walked = [
            dim
            for dim in range(value.ndim)
            if (block_id := _resolve_dim_block_id(state.codegen, value, dim))
            is not None
            and env.canonical_block_id(block_id)
            == env.canonical_block_id(axis.block_id)
        ]
        mask = load.extra_mask
        if len(walked) != 1 or (
            (mask_dim := mask.ndim - value.ndim + walked[0]) >= 0
            and not env.known_equal(mask.shape[mask_dim], 1)
        ):
            raise exc.BackendUnsupported(
                "cute",
                "a serial re-read of a load whose extra_mask varies along the "
                "walked dim; the mask is evaluated at this lane's own element",
            )
    (pos,) = positions
    index_exprs = list(load.index_exprs)
    index = pattern.sub(f"(({axis.base}) + ({position}))", index_exprs[pos])
    index_exprs[pos] = index
    size_expr = _cute_tensor_dim_size_expr(state, load.tensor, pos)
    mask_terms = [f"({index}) >= 0", f"({index}) < cutlass.Int32({size_expr})"]
    if (valid := axis.valid(position)) is not None:
        mask_terms.append(valid)
    if load.mask_expr is not None:
        non_axis_mask = _cute_strip_mask_term(
            load.mask_expr, state.codegen.mask_var(axis.block_id)
        )
        if non_axis_mask is not None:
            mask_terms.append(non_axis_mask)
    return index_exprs, mask_terms


def _cute_inline_combine_graph(
    state: CodegenState,
    helper_graph_info: object,
    left_vars: list[str],
    right_vars: list[str],
    indent: str = "    ",
) -> tuple[list[str], list[str]]:
    """Inline a tuple combine graph as CuTe scalar expressions.

    ``left_vars``/``right_vars`` are the per-lane scalar variable names holding
    the left (accumulator) and right (incoming) tuple elements.  Returns a
    tuple ``(body_lines, out_exprs)`` where ``body_lines`` are assignment
    statements prefixed with ``indent`` (4 spaces by default, to be spliced
    inside the serial scan ``for`` loop) and ``out_exprs`` are the output
    expression strings, one per tuple element.
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
        lines.append(f"{indent}{var} = {expr}")
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
    *,
    parallel_declined: str,
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
        state,
        helper_graph_info,
        list(input_nodes),
        dim,
        reverse,
        parallel_declined=parallel_declined,
    )


def _cute_codegen_serial_scan(
    state: CodegenState,
    helper_graph_info: object,
    input_nodes: list[object],
    dim: int,
    reverse: bool,
    *,
    parallel_declined: str,
) -> list[ast.AST]:
    """Dim-agnostic serial per-lane inclusive scan shared by the scalar and
    tuple CuTe scan paths.

    ``input_nodes`` is one FX node per scanned stream (a single element for a
    scalar scan, multiple for a tuple scan).  For each output position along the
    scan dimension this folds the user's combine graph over the global rows
    ``0..out_pos`` (inclusive), carrying one accumulator per stream, re-reading
    each stream's load at every row.  Returns one output expression per stream.
    ``parallel_declined`` is why ``_cute_try_parallel_scan`` passed on the scan,
    for the refusal of a stream that is not a load.
    """

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
    axis = cute_serial_axis(state, first_val, dim, "associative_scan")

    # Recover a scalar load per tuple element.
    loads: list[CuteSortableLoad] = []
    for node in input_nodes:
        recovered = _cute_recover_scan_load(node)
        if recovered is None or not isinstance(recovered[0], CuteSortableLoad):
            raise exc.BackendUnsupported(
                "cute",
                "hl.associative_scan / hl.cumsum of a computed tile (not a "
                "load): the serial fallback re-reads loads only, and the "
                f"register scan does not apply because {parallel_declined}",
            )
        load, _load_node = recovered
        assert isinstance(load, CuteSortableLoad)
        loads.append(load)

    index_dtype = env.backend.dtype_str(env.index_dtype)
    out_pos = state.device_function.new_var("scan_out_pos")
    scan_i = state.device_function.new_var("scan_i")
    include = state.device_function.new_var("scan_include")
    initialized = state.device_function.new_var("scan_initialized")

    acc_vars = [state.device_function.new_var("scan_acc") for _ in loads]
    value_vars = [state.device_function.new_var("scan_value") for _ in loads]

    state.codegen.add_statement(
        statement_from_string(f"{out_pos} = {index_dtype}({axis.position})")
    )
    for acc_var, val in zip(acc_vars, vals, strict=True):
        dtype_str = env.backend.dtype_str(val.dtype)
        state.codegen.add_statement(
            statement_from_string(f"{acc_var} = {dtype_str}(0)")
        )
    state.codegen.add_statement(statement_from_string(f"{initialized} = False"))
    position, position_lines = _cute_serial_scan_position(
        state, scan_i, axis.extent, reverse
    )

    # Per-iteration value loads (re-load each stream at the scanned position).
    # Cast each loaded scalar to the *scanned* element dtype (``vals[i].dtype``):
    # the recovered load may have a different storage dtype than the value
    # entering the scan (e.g. the index stream is ``indices`` int64 but scanned
    # as float32 after ``idxs.float()``).  Each load is guarded so an
    # out-of-range position (partial final tile) reads 0 instead of faulting.
    value_lines: list[str] = []
    for val_var, load, val in zip(value_vars, loads, vals, strict=True):
        scan_dtype_str = env.backend.dtype_str(val.dtype)
        load_dtype_str = env.backend.dtype_str(load.dtype)
        index_exprs, mask_terms = cute_reread_index(state, load, axis, position)
        load_expr = f"{load.tensor_name}[{', '.join(index_exprs)}]"
        mask_expr = " and ".join(f"({term})" for term in mask_terms)
        load_expr = f"({load_expr} if {mask_expr} else {load_dtype_str}(0))"
        value_lines.append(f"    {val_var} = {scan_dtype_str}({load_expr})")

    include_expr = _cute_scan_include_expr(axis, position, out_pos, reverse)

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
                    f"for {scan_i} in range(cutlass.Int32(0), cutlass.Int32({axis.extent}), cutlass.Int32(1)):",
                    *position_lines,
                    f"    {include} = {include_expr}",
                    *value_lines,
                    *combine_lines,
                    *fold_lines,
                ]
            )
        )
    )

    return [expr_from_string(acc_var) for acc_var in acc_vars]


def _cute_scan_tile_base(state: CodegenState, block_id: int) -> str | None:
    """Uniform first index of the active tile along ``block_id``, if known."""
    from ..reduction_strategy import PersistentReductionStrategy
    from ..tile_strategy import PerThreadFlattenedTileStrategy
    from ..tile_strategy import PerThreadNDTileStrategy

    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return None
    strategy = loops[-1].strategy
    if isinstance(
        strategy,
        (
            PerThreadNDTileStrategy,
            PerThreadFlattenedTileStrategy,
            PersistentReductionStrategy,
        ),
    ):
        return strategy.cute_tile_base_expr(block_id)
    return None


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


def cute_rank_sort_lines(
    state: CodegenState,
    load: CuteSortableLoad,
    axis: CuteSerialAxis,
    descending: bool,
    out_pos: str,
    indent: str = "",
    select_if: str | None = None,
) -> tuple[list[str], str, str]:
    """A rank sort selecting the element of rank ``out_pos`` along ``axis``.

    Every candidate position is ranked against every probe position of the
    re-read load (``cute_reread_index``); returns the statements (two
    initializations, then the candidate loop), the selected value and the
    selected position (0 where ``select_if`` is false).
    """
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    dtype_str = env.backend.dtype_str(load.dtype)
    index_dtype = env.backend.dtype_str(env.index_dtype)
    new_var = state.device_function.new_var
    sorted_value = new_var("sorted_vals")
    sorted_index = new_var("sorted_indices")
    candidate = new_var("sort_k")
    probe = new_var("sort_j")
    candidate_rank = new_var("sort_rank")
    candidate_value = new_var("sort_candidate")
    probe_value = new_var("sort_probe")
    before = new_var("sort_before")
    selected = new_var("sort_selected")

    def reread(position: str) -> str:
        index_exprs, mask_terms = cute_reread_index(state, load, axis, position)
        expr = f"{load.tensor_name}[{', '.join(index_exprs)}]"
        mask_expr = " and ".join(f"({term})" for term in mask_terms)
        return f"({expr} if {mask_expr} else {dtype_str}(0))"

    cmp_op = ">" if descending else "<"
    # Positions past a masked block's end hold no element: they neither rank
    # before a candidate nor take a rank (``x[t, n]`` of a partial tile).
    probe_valid = "" if (valid := axis.valid(probe)) is None else f"({valid}) and "
    candidate_valid = (
        "" if (valid := axis.valid(candidate)) is None else f"({valid}) and "
    )
    loop = f"range(cutlass.Int32(0), cutlass.Int32({axis.extent}), cutlass.Int32(1))"
    loop_lines = [
        f"for {candidate} in {loop}:",
        f"    {candidate_value} = {reread(candidate)}",
        f"    {candidate_rank} = {index_dtype}(0)",
        f"    for {probe} in {loop}:",
        f"        {probe_value} = {reread(probe)}",
        f"        {before} = {probe_valid}(({probe_value} {cmp_op} {candidate_value}) or (({probe_value} == {candidate_value}) and ({probe} < {candidate})))",
        f"        {candidate_rank} = {candidate_rank} + ({index_dtype}(1) if {before} else {index_dtype}(0))",
        f"    {selected} = {candidate_valid}{candidate_rank} == {out_pos}"
        + ("" if select_if is None else f" and {select_if}"),
        f"    {sorted_value} = {candidate_value} if {selected} else {sorted_value}",
        f"    {sorted_index} = {index_dtype}({candidate}) if {selected} else {sorted_index}",
    ]
    statements = [
        f"{indent}{sorted_value} = {dtype_str}(0)",
        f"{indent}{sorted_index} = {index_dtype}(0)",
        "\n".join(f"{indent}{line}" for line in loop_lines),
    ]
    return statements, sorted_value, sorted_index


def _cute_sorted_value_lines(
    state: CodegenState,
    load: CuteSortableLoad,
    axis: CuteSerialAxis,
    descending: bool,
    out_pos: str,
    output_var: str,
) -> list[str]:
    lines, sorted_value, _ = cute_rank_sort_lines(
        state, load, axis, descending, out_pos, indent="    "
    )
    return [*lines, f"    {output_var} = {sorted_value}"]


# ---------------------------------------------------------------------------
# Register / warp-shuffle scan lowering
# ---------------------------------------------------------------------------

# One carry slot is needed per column a thread walks while the scan lane loop
# advances; beyond this many the lane-looped scan keeps the serial fallback.
_CUTE_SCAN_MAX_CARRY_SLOTS = 256
_CUTE_WARP_SIZE = 32


@dataclasses.dataclass(frozen=True)
class _CuteScanInnerLane:
    """A lane / vector loop over a non-scan axis nested inside the scan lane loop.

    Each such loop multiplies the number of distinct columns one thread walks
    while the scan lane loop advances, so the carried prefix needs one slot per
    combination of their coordinates.
    """

    index_expr: str
    extent: int


@dataclasses.dataclass(frozen=True)
class _CuteScanGeometry:
    """How the scan axis of one tile is distributed over threads and lanes."""

    block_id: int
    # Elements along the scan axis covered by the tile (``threads * lanes``).
    extent: int
    # Threads on CUDA thread axis 0 splitting the axis (1 = lane loop only).
    threads: int
    # Per-thread lane loop over the scan axis and its trip count.
    lane_var: str | None
    lane_steps: int
    # Constexpr vector loop over the scan axis (only with ``threads == 1``).
    vec_lane_var: str | None
    vec_width: int
    # Thread-split axis whose lane steps are ``threads``-wide chunks.
    strided: bool
    inner_lanes: tuple[_CuteScanInnerLane, ...]
    # Block mask naming the in-range rows of a partial tile (``None`` = all).
    mask_expr: str | None
    # Loop / grid state whose ``outer_prefix`` receives the carry declarations.
    owner: DeviceGridState | DeviceLoopState

    @property
    def carry_slots(self) -> int:
        slots = 1
        for lane in self.inner_lanes:
            slots *= lane.extent
        return slots


@dataclasses.dataclass(frozen=True)
class _CuteScanStream:
    """One scanned stream: the per-lane variable holding this element's value."""

    value: str
    dtype_str: str
    is_bool: bool


@dataclasses.dataclass(frozen=True)
class _CuteScanCarry:
    """Per-thread storage for the prefix carried across scan lane steps."""

    values: list[str]
    valid: str | None
    # Register-fragment slot for the current column (``None`` = scalar carry).
    slot: str | None


def _cute_scan_input_is_sorted(node: object) -> bool:
    from torch.fx.node import Node

    return (
        isinstance(node, Node)
        and node.target is operator.getitem
        and isinstance(node.args[0], Node)
        and node.args[0].target is torch.ops.aten.sort.default
    )


def _cute_scan_lane_slot(
    lane_var: str, vec_lane_var: str, extent: int, width: int
) -> _CuteScanInnerLane:
    if extent == width:
        return _CuteScanInnerLane(f"cutlass.Int32({vec_lane_var})", extent)
    return _CuteScanInnerLane(
        f"cutlass.Int32({lane_var}) * {width} + cutlass.Int32({vec_lane_var})",
        extent,
    )


def _cute_scan_inner_lanes(
    owner: DeviceGridState | DeviceLoopState,
    strategy: object,
    block_id: int,
    lane_var: str,
) -> tuple[_CuteScanInnerLane, ...] | None:
    """Lane / vector loops nested inside the scan block's lane loop.

    Grid lane loops are materialized by ``DeviceGridState.wrap_body`` in
    ``lane_loops`` order (first entry outermost); device-loop lane loops follow
    the strategy's ``loop_order``.  Returns ``None`` when a vector partition
    cannot be read back.
    """
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import PerThreadNDTileStrategy
    from ..tile_strategy import _static_lane_loop_extent

    lanes: list[_CuteScanInnerLane] = []
    if isinstance(owner, DeviceGridState):
        lane_vars = [var for var, _extent in owner.lane_loops]
        if lane_var not in lane_vars:
            return None
        for var, extent in owner.lane_loops[lane_vars.index(lane_var) + 1 :]:
            if extent <= 1:
                continue
            wrapper = owner.vec_lane_wrappers.get(var)
            if wrapper is None or wrapper.vec_lane_var == var:
                # Plain lane loop, or a persistent one-vector-per-thread lane
                # whose constexpr loop var *is* the lane var.
                lanes.append(_CuteScanInnerLane(f"cutlass.Int32({var})", extent))
                continue
            width = _static_lane_loop_extent(wrapper.vloop)
            if width is None or extent % width:
                return None
            lanes.append(_cute_scan_lane_slot(var, wrapper.vec_lane_var, extent, width))
        return tuple(lanes)
    if not isinstance(strategy, PerThreadNDTileStrategy):
        return None
    order = [strategy.block_ids[i] for i in strategy.loop_order]
    if block_id not in order:
        return None
    for other in order[order.index(block_id) + 1 :]:
        var = strategy._lane_var_by_block.get(other)
        if var is None:
            continue
        extent = strategy._elements_per_thread_for_block(other)
        if extent <= 1:
            continue
        vec_lane_var = strategy._cute_vec_lane_var_by_block.get(other)
        if vec_lane_var is None:
            lanes.append(_CuteScanInnerLane(f"cutlass.Int32({var})", extent))
            continue
        width = strategy._cute_lane_vec_width_by_block.get(other, 1)
        if extent % width:
            return None
        lanes.append(_cute_scan_lane_slot(var, vec_lane_var, extent, width))
    return tuple(lanes)


def _cute_scan_has_nested_device_loops(
    state: CodegenState, owner: DeviceGridState | DeviceLoopState
) -> bool:
    """Whether a device loop runs *inside* the owner's lane loops.

    A carried prefix is reset only at the first lane step, so a device loop
    nested inside the scan lane loop would interleave unrelated tiles into one
    carry.  Grid lane loops wrap the whole root body, so for a grid owner any
    active device loop is nested; for a device-loop owner another active device
    loop is fine only when it encloses the owner.
    """
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState

    seen: set[int] = set()
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            if (
                loop_state is owner
                or id(loop_state) in seen
                or not isinstance(loop_state, DeviceLoopState)
            ):
                continue
            seen.add(id(loop_state))
            if isinstance(owner, DeviceGridState):
                return True
            if not any(
                node is owner.for_node for node in ast.walk(loop_state.for_node)
            ):
                return True
    return False


def _cute_scan_geometry(state: CodegenState, block_id: int) -> _CuteScanGeometry | None:
    """Resolve the thread / lane distribution of the scan axis, or ``None``.

    ``None`` means the shape is outside what the parallel lowering proves
    correct and the caller keeps the serial fallback.
    """
    from ..reduction_strategy import PersistentReductionStrategy
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState
    from ..tile_strategy import PersistentReductionState
    from ..tile_strategy import PerThreadFlattenedTileStrategy
    from ..tile_strategy import PerThreadNDTileStrategy

    cg = state.codegen
    loops = cg.active_device_loops.get(block_id)
    if not loops:
        return None
    loop_state = loops[-1]
    strategy = loop_state.strategy
    grid = cg.current_grid_state
    thread_axis = loop_state.block_thread_axes.get(block_id)
    owner: DeviceGridState | DeviceLoopState
    if isinstance(loop_state, PersistentReductionState):
        if not isinstance(strategy, PersistentReductionStrategy) or grid is None:
            return None
        owner = grid
    elif isinstance(strategy, PerThreadNDTileStrategy) or (
        # A flattened tile only lives on the grid.
        isinstance(strategy, PerThreadFlattenedTileStrategy)
        and isinstance(loop_state, DeviceGridState)
    ):
        if not isinstance(loop_state, (DeviceGridState, DeviceLoopState)):
            return None
        owner = loop_state
    else:
        return None
    axis = strategy.cute_lane_axis(block_id)
    if axis is None or axis.extent <= 0:
        return None
    # Grid-owned declarations go to the kernel-scope statement list that later
    # receives the grid's ``outer_prefix`` and the (possibly lane-wrapped)
    # root body.
    if isinstance(owner, DeviceGridState) and owner.hoist_parent_statements is None:
        return None
    threads = axis.threads
    if threads > 1:
        # Consecutive threads of axis 0 are consecutive warp lanes, so a
        # power-of-two group of them is warp-aligned (or whole warps).
        if thread_axis != 0 or threads & (threads - 1):
            return None
        if threads > _CUTE_WARP_SIZE and threads % _CUTE_WARP_SIZE:
            return None
        # A vector chunk per thread is blocked *within* the chunk, and a
        # blocked multi-element split needs a second pass over the values.
        if axis.vec_lane_var is not None or (axis.lane_steps > 1 and not axis.strided):
            return None

    inner_lanes: tuple[_CuteScanInnerLane, ...] = ()
    if axis.lane_steps * axis.vec_width > 1:
        if axis.lane_var is None or _cute_scan_has_nested_device_loops(state, owner):
            return None
        resolved = _cute_scan_inner_lanes(owner, strategy, block_id, axis.lane_var)
        if resolved is None:
            return None
        inner_lanes = resolved
    geometry = _CuteScanGeometry(
        block_id=block_id,
        extent=axis.extent,
        threads=threads,
        lane_var=axis.lane_var,
        lane_steps=axis.lane_steps,
        vec_lane_var=axis.vec_lane_var,
        vec_width=axis.vec_width,
        strided=axis.strided,
        inner_lanes=inner_lanes,
        mask_expr=cg.mask_var(block_id),
        owner=owner,
    )
    if geometry.carry_slots > _CUTE_SCAN_MAX_CARRY_SLOTS:
        return None
    return geometry


def _cute_scan_lane_hosts_reduction(state: CodegenState) -> bool:
    """Whether a lane loop enclosing the scan will also host a lane reduction.

    A reduction over a lane-looped block leaves a ``_helion_lane_reduce``
    marker that ``split_lane_loop_reductions`` later turns into separate
    accumulate / consume passes over that lane loop.  The split admits only
    loop-carried scalars it can prove are updated once per tile; the scan's
    per-lane prefix carry is not one of them, and the consume pass would drop
    the carry update (and any ``sync_threads`` of a cross-warp exchange).  The
    reductions of the current graph are known before codegen reaches them
    through their :class:`ReductionLowering` (and the matmul facts of scalar
    contractions), so the scan can decline up front and keep the serial
    fallback, which carries nothing across lanes.
    """
    from ...language.matmul_ops import MATMUL_FACT_ID_META
    from ..compile_environment import CompileEnvironment
    from ..inductor_lowering import ReductionLowering
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState

    cg = state.codegen
    lane_blocks: set[int] = set()
    grid = cg.current_grid_state
    if isinstance(grid, DeviceGridState):
        lane_blocks |= grid.lane_loop_blocks
    for loops in cg.active_device_loops.values():
        for loop_state in loops:
            if isinstance(loop_state, DeviceLoopState):
                lane_blocks |= loop_state.lane_loop_blocks
    if not lane_blocks:
        return False
    env = CompileEnvironment.current()
    lane_canonical = {env.canonical_block_id(block_id) for block_id in lane_blocks}

    def reduces_lane_block(block_id: int | None) -> bool:
        return (
            block_id is not None and env.canonical_block_id(block_id) in lane_canonical
        )

    fx_node = state.fx_node
    assert fx_node is not None
    for node in fx_node.graph.nodes:
        lowering = node.meta.get("lowering")
        if isinstance(lowering, ReductionLowering) and reduces_lane_block(
            lowering.block_index
        ):
            return True
        fact_id = node.meta.get(MATMUL_FACT_ID_META)
        if isinstance(fact_id, int):
            fact = env.config_spec.matmul_facts[fact_id]
            if any(
                reduces_lane_block(block_id)
                for block_id in (fact.m_block_id, fact.n_block_id, fact.k_block_id)
            ):
                return True
    return False


def _cute_scan_prepare_lane_direction(
    state: CodegenState, geometry: _CuteScanGeometry, reverse: bool
) -> bool:
    """Claim the scan lane loop's traversal direction; reverse it if needed.

    Every parallel scan carried across a lane loop records the direction it
    needs; a later scan over the same lane loop must agree or it falls back.
    Grid lane loops are reversed when ``wrap_body`` materializes them; device
    loop lane loops already exist and are rewritten in place.  The rewrite is
    transactional: every loop is checked first and nothing is mutated or
    recorded unless all of them can be reversed, so a ``False`` result leaves
    the caller free to take the serial fallback.
    """
    from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import _lane_loop_reversible
    from ..tile_strategy import _reverse_lane_loop_iter

    lane_var = geometry.lane_var
    if lane_var is None or geometry.lane_steps * geometry.vec_width <= 1:
        return True
    directions = state.device_function.cute_state.scan_lane_directions
    previous = directions.get(lane_var)
    if previous is not None:
        return previous == reverse
    if reverse:
        owner = geometry.owner
        if isinstance(owner, DeviceGridState):
            # ``wrap_body`` creates the plain ``range(N)`` loop later; a
            # pre-built vector partition cannot be reversed (see the module
            # docstring), and the caller already declined that shape.
            if lane_var in owner.vec_lane_wrappers:
                return False
            owner.reversed_lane_vars.add(lane_var)
        else:
            loops = [
                node
                for node in ast.walk(owner.for_node)
                if isinstance(node, ast.For)
                and getattr(node, HELION_LANE_LOOP_VAR_ATTR, None) == lane_var
            ]
            if not loops or not all(_lane_loop_reversible(loop) for loop in loops):
                return False
            for loop in loops:
                reversed_ok = _reverse_lane_loop_iter(loop)
                assert reversed_ok
    directions[lane_var] = reverse
    return True


class _CuteScanEmitter:
    """Emits the register / shuffle scan for one ``hl.associative_scan``.

    Statements go to the current codegen position (inside the innermost lane
    loop, where the per-lane values live); carry and shared-memory
    declarations are hoisted ahead of every lane loop: to the kernel-scope
    parent statements for a grid owner, to ``outer_prefix`` for a device loop.
    """

    def __init__(
        self,
        state: CodegenState,
        geometry: _CuteScanGeometry,
        helper_graph_info: HelperFunctionGraphInfo,
        reverse: bool,
    ) -> None:
        from ..compile_environment import CompileEnvironment

        self.state = state
        self.geometry = geometry
        self.helper_graph_info = helper_graph_info
        self.reverse = reverse
        self.env = CompileEnvironment.current()
        # Forward scans need no validity bookkeeping: the block mask is a
        # suffix of the axis, so masked rows only ever feed other masked rows.
        self.track_valid = reverse and geometry.mask_expr is not None
        self._blocks: list[list[str]] = []

    def new_var(self, name: str) -> str:
        return self.state.device_function.new_var(name)

    def emit(self, text: str) -> None:
        from ..ast_extension import statement_from_string

        if self._blocks:
            self._blocks[-1].append(text)
        else:
            self.state.codegen.add_statement(statement_from_string(text))

    def hoist(self, text: str) -> None:
        from ..ast_extension import statement_from_string
        from ..tile_strategy import DeviceGridState

        owner = self.geometry.owner
        if isinstance(owner, DeviceGridState):
            assert owner.hoist_parent_statements is not None
            owner.hoist_parent_statements.append(statement_from_string(text))
        else:
            owner.outer_prefix.append(statement_from_string(text))

    @contextlib.contextmanager
    def block(self, header: str) -> Iterator[None]:
        body: list[str] = []
        self._blocks.append(body)
        try:
            yield
        finally:
            self._blocks.pop()
        lines = [f"    {line}" for text in body for line in text.split("\n")]
        self.emit("\n".join([header, *lines]))

    def stream(self, arg: ast.AST, fake_value: torch.Tensor) -> _CuteScanStream:
        dtype_str = self.env.backend.dtype_str(fake_value.dtype)
        var = self.new_var("scan_value")
        self.emit(f"{var} = {dtype_str}({ast.unparse(arg)})")
        return _CuteScanStream(var, dtype_str, fake_value.dtype is torch.bool)

    @staticmethod
    def shuffle(kind: str, expr: str, offset: str, is_bool: bool) -> str:
        function = {
            "up": "cute.arch.shuffle_sync_up",
            "down": "cute.arch.shuffle_sync_down",
            "idx": "cute.arch.shuffle_sync",
        }[kind]
        if is_bool:
            # ``shfl`` moves integer / float registers; booleans ride as Int32.
            return f"({function}(cutlass.Int32({expr}), {offset}) != cutlass.Int32(0))"
        return f"{function}({expr}, {offset})"

    def merge(
        self,
        streams: list[_CuteScanStream],
        left: list[str],
        right: list[str],
        take: str | None,
        *,
        left_valid: str | None = None,
        right_valid: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Fold ``left`` (earlier in scan order) into ``right`` (current).

        ``take`` says whether a left operand exists at all; ``left_valid`` /
        ``right_valid`` are optional validity flags of the operands (masked
        rows).  Returns the merged per-stream variables and the validity of
        the result (``None`` when no validity is tracked).
        """
        lines, exprs = _cute_inline_combine_graph(
            self.state, self.helper_graph_info, list(left), list(right), indent=""
        )
        for line in lines:
            self.emit(line)
        if left_valid is None:
            assert take is not None
            use = take
        elif take is None:
            use = left_valid
        else:
            use = self.new_var("scan_use")
            self.emit(f"{use} = ({take}) and ({left_valid})")
        outs: list[str] = []
        for stream, left_var, right_var, expr in zip(
            streams, left, right, exprs, strict=True
        ):
            out = self.new_var("scan_out")
            merged = f"{stream.dtype_str}({expr})"
            if right_valid is None:
                self.emit(f"{out} = ({merged}) if ({use}) else ({right_var})")
            else:
                self.emit(
                    f"{out} = ({merged}) if (({use}) and ({right_valid})) "
                    f"else (({left_var}) if ({use}) else ({right_var}))"
                )
            outs.append(out)
        if left_valid is None and right_valid is None:
            return outs, None
        if right_valid is None:
            # The right operand always exists, so the result is unconditionally
            # populated; keep the flag a DSL Boolean so loop carries type-check.
            return outs, "cutlass.Boolean(True)"
        valid = self.new_var("scan_valid")
        self.emit(f"{valid} = ({right_valid}) or ({use})")
        return outs, valid

    def declare_carry(self, streams: list[_CuteScanStream]) -> _CuteScanCarry:
        slots = self.geometry.carry_slots
        values = [self.new_var("scan_carry") for _ in streams]
        valid = self.new_var("scan_carry_valid") if self.track_valid else None
        if slots == 1:
            for var, stream in zip(values, streams, strict=True):
                self.hoist(f"{var} = {stream.dtype_str}(0)")
            if valid is not None:
                self.hoist(f"{valid} = cutlass.Boolean(False)")
            return _CuteScanCarry(values, valid, None)
        for var, stream in zip(values, streams, strict=True):
            self.hoist(f"{var} = cute.make_rmem_tensor(({slots},), {stream.dtype_str})")
        if valid is not None:
            self.hoist(f"{valid} = cute.make_rmem_tensor(({slots},), cutlass.Int32)")
        terms: list[str] = []
        stride = 1
        for lane in reversed(self.geometry.inner_lanes):
            terms.append(
                f"({lane.index_expr}) * {stride}"
                if stride > 1
                else f"({lane.index_expr})"
            )
            stride *= lane.extent
        slot = self.new_var("scan_slot")
        self.emit(f"{slot} = " + " + ".join(reversed(terms)))
        return _CuteScanCarry(values, valid, slot)

    def load_carry(self, carry: _CuteScanCarry) -> tuple[list[str], str | None]:
        if carry.slot is None:
            return list(carry.values), carry.valid
        previous: list[str] = []
        for var in carry.values:
            prev = self.new_var("scan_prev")
            self.emit(f"{prev} = {var}[{carry.slot}]")
            previous.append(prev)
        prev_valid: str | None = None
        if carry.valid is not None:
            prev_valid = self.new_var("scan_prev_valid")
            self.emit(f"{prev_valid} = {carry.valid}[{carry.slot}] != cutlass.Int32(0)")
        return previous, prev_valid

    def store_carry(
        self, carry: _CuteScanCarry, values: list[str], valid: str | None
    ) -> None:
        if carry.slot is None:
            for var, value in zip(carry.values, values, strict=True):
                self.emit(f"{var} = {value}")
            if carry.valid is not None:
                assert valid is not None
                self.emit(f"{carry.valid} = {valid}")
            return
        for var, value in zip(carry.values, values, strict=True):
            self.emit(f"{var}[{carry.slot}] = {value}")
        if carry.valid is not None:
            assert valid is not None
            self.emit(
                f"{carry.valid}[{carry.slot}] = cutlass.Int32(1) if ({valid}) "
                f"else cutlass.Int32(0)"
            )

    def emit_lane_loop_scan(self, streams: list[_CuteScanStream]) -> list[str]:
        """One thread owns the whole axis: a loop-carried in-thread prefix."""
        geometry = self.geometry
        assert geometry.lane_var is not None
        position = self.new_var("scan_pos")
        position_expr = f"cutlass.Int32({geometry.lane_var})"
        if geometry.vec_lane_var is not None:
            position_expr = (
                f"cutlass.Int32({geometry.lane_var}) * {geometry.vec_width}"
                f" + cutlass.Int32({geometry.vec_lane_var})"
            )
        self.emit(f"{position} = {position_expr}")
        first = geometry.extent - 1 if self.reverse else 0
        rest = self.new_var("scan_rest")
        self.emit(f"{rest} = {position} != cutlass.Int32({first})")
        carry = self.declare_carry(streams)
        previous, prev_valid = self.load_carry(carry)
        outs, out_valid = self.merge(
            streams,
            previous,
            [stream.value for stream in streams],
            rest,
            left_valid=prev_valid,
            right_valid=geometry.mask_expr if self.track_valid else None,
        )
        self.store_carry(carry, outs, out_valid)
        return outs

    def emit_thread_split_scan(self, streams: list[_CuteScanStream]) -> list[str]:
        """``threads`` consecutive lanes hold one chunk: Kogge-Stone warp scan."""
        geometry = self.geometry
        threads = geometry.threads
        group = min(threads, _CUTE_WARP_SIZE)
        lane = self.new_var("scan_lane")
        if group == _CUTE_WARP_SIZE:
            self.emit(f"{lane} = cutlass.Int32(cute.arch.lane_idx())")
        else:
            self.emit(f"{lane} = cutlass.Int32(cute.arch.thread_idx()[0]) % {group}")
        inclusive = [stream.value for stream in streams]
        inclusive_valid = geometry.mask_expr if self.track_valid else None
        kind = "down" if self.reverse else "up"
        distance = 1
        while distance < group:
            peers: list[str] = []
            for stream, value in zip(streams, inclusive, strict=True):
                peer = self.new_var("scan_peer")
                self.emit(
                    f"{peer} = {self.shuffle(kind, value, str(distance), stream.is_bool)}"
                )
                peers.append(peer)
            peer_valid: str | None = None
            if inclusive_valid is not None:
                peer_valid = self.new_var("scan_peer_valid")
                self.emit(
                    f"{peer_valid} = "
                    f"{self.shuffle(kind, inclusive_valid, str(distance), True)}"
                )
            take = (
                f"{lane} < {group - distance}"
                if self.reverse
                else f"{lane} >= {distance}"
            )
            inclusive, inclusive_valid = self.merge(
                streams,
                peers,
                inclusive,
                take,
                left_valid=peer_valid,
                right_valid=inclusive_valid,
            )
            distance *= 2
        chunk_total: tuple[list[str], str | None] | None = None
        if threads > _CUTE_WARP_SIZE:
            inclusive, inclusive_valid, chunk_total = self.emit_cross_warp(
                streams, lane, inclusive, inclusive_valid
            )
        if geometry.lane_steps <= 1:
            return inclusive
        # Strided chunks: fold the previous chunks' total into this chunk and
        # carry the new total (held by the chunk's last lane in scan order).
        assert geometry.lane_var is not None
        first = geometry.lane_steps - 1 if self.reverse else 0
        rest = self.new_var("scan_rest")
        self.emit(
            f"{rest} = cutlass.Int32({geometry.lane_var}) != cutlass.Int32({first})"
        )
        carry = self.declare_carry(streams)
        previous, prev_valid = self.load_carry(carry)
        outs, out_valid = self.merge(
            streams,
            previous,
            inclusive,
            rest,
            left_valid=prev_valid,
            right_valid=inclusive_valid,
        )
        if chunk_total is None:
            if group == _CUTE_WARP_SIZE:
                source = str(0 if self.reverse else _CUTE_WARP_SIZE - 1)
            else:
                source = self.new_var("scan_source_lane")
                base = f"(cutlass.Int32(cute.arch.lane_idx()) // {group}) * {group}"
                self.emit(
                    f"{source} = {base}"
                    if self.reverse
                    else f"{source} = {base} + {group - 1}"
                )
            new_values: list[str] = []
            for stream, out in zip(streams, outs, strict=True):
                new = self.new_var("scan_carry_next")
                self.emit(f"{new} = {self.shuffle('idx', out, source, stream.is_bool)}")
                new_values.append(new)
            new_valid: str | None = None
            if out_valid is not None:
                new_valid = self.new_var("scan_carry_next_valid")
                self.emit(
                    f"{new_valid} = {self.shuffle('idx', out_valid, source, True)}"
                )
        else:
            total, total_valid = chunk_total
            new_values, new_valid = self.merge(
                streams,
                previous,
                total,
                rest,
                left_valid=prev_valid,
                right_valid=total_valid,
            )
        self.store_carry(carry, new_values, new_valid)
        return outs

    def emit_cross_warp(
        self,
        streams: list[_CuteScanStream],
        lane: str,
        inclusive: list[str],
        inclusive_valid: str | None,
    ) -> tuple[list[str], str | None, tuple[list[str], str | None]]:
        """Exchange per-warp totals through shared memory.

        Each warp publishes its inclusive total (its last lane in scan order);
        every thread then folds the totals of the warps preceding its own in
        scan order into an exclusive prefix and, separately, all of the group's
        warps into the chunk total used for the lane-step carry.
        """
        geometry = self.geometry
        warps = geometry.threads // _CUTE_WARP_SIZE
        buffers: list[str] = []
        for stream in streams:
            pointer = self.new_var("scan_smem_ptr")
            buffer = self.new_var("scan_smem")
            self.hoist(
                f"{pointer} = cute.arch.alloc_smem({stream.dtype_str}, {_CUTE_WARP_SIZE})"
            )
            self.hoist(f"{buffer} = cute.make_tensor({pointer}, ({_CUTE_WARP_SIZE},))")
            buffers.append(buffer)
        valid_buffer: str | None = None
        if inclusive_valid is not None:
            pointer = self.new_var("scan_smem_valid_ptr")
            valid_buffer = self.new_var("scan_smem_valid")
            self.hoist(
                f"{pointer} = cute.arch.alloc_smem(cutlass.Int32, {_CUTE_WARP_SIZE})"
            )
            self.hoist(
                f"{valid_buffer} = cute.make_tensor({pointer}, ({_CUTE_WARP_SIZE},))"
            )
        warp_in_group = self.new_var("scan_warp")
        self.emit(
            f"{warp_in_group} = cutlass.Int32(cute.arch.thread_idx()[0]) // {_CUTE_WARP_SIZE}"
        )
        cta_warp = self.new_var("scan_cta_warp")
        self.emit(f"{cta_warp} = cutlass.Int32(cute.arch.warp_idx())")
        group_warp = self.new_var("scan_group_warp")
        self.emit(f"{group_warp} = {cta_warp} - {warp_in_group}")
        last_lane = 0 if self.reverse else _CUTE_WARP_SIZE - 1
        with self.block(f"if {lane} == {last_lane}:"):
            for buffer, value in zip(buffers, inclusive, strict=True):
                self.emit(f"{buffer}[{cta_warp}] = {value}")
            if valid_buffer is not None:
                self.emit(
                    f"{valid_buffer}[{cta_warp}] = cutlass.Int32(1) "
                    f"if ({inclusive_valid}) else cutlass.Int32(0)"
                )
        self.emit("cute.arch.sync_threads()")
        prefix = [self.new_var("scan_pre") for _ in streams]
        total = [self.new_var("scan_total") for _ in streams]
        for stream, pre_var, total_var in zip(streams, prefix, total, strict=True):
            self.emit(f"{pre_var} = {stream.dtype_str}(0)")
            self.emit(f"{total_var} = {stream.dtype_str}(0)")
        prefix_valid = self.new_var("scan_pre_valid")
        total_valid = self.new_var("scan_total_valid")
        self.emit(f"{prefix_valid} = cutlass.Boolean(False)")
        self.emit(f"{total_valid} = cutlass.Boolean(False)")
        step = self.new_var("scan_j")
        with self.block(
            f"for {step} in range(cutlass.Int32(0), cutlass.Int32({warps}), cutlass.Int32(1)):"
        ):
            order = self.new_var("scan_step")
            self.emit(
                f"{order} = cutlass.Int32({warps - 1}) - {step}"
                if self.reverse
                else f"{order} = {step}"
            )
            warp_totals: list[str] = []
            for buffer in buffers:
                warp_total = self.new_var("scan_warp_total")
                self.emit(f"{warp_total} = {buffer}[{group_warp} + {order}]")
                warp_totals.append(warp_total)
            warp_total_valid: str | None = None
            if valid_buffer is not None:
                warp_total_valid = self.new_var("scan_warp_total_valid")
                self.emit(
                    f"{warp_total_valid} = "
                    f"{valid_buffer}[{group_warp} + {order}] != cutlass.Int32(0)"
                )
            merged_total, merged_total_valid = self.merge(
                streams,
                total,
                warp_totals,
                None,
                left_valid=total_valid,
                right_valid=warp_total_valid,
            )
            for total_var, merged in zip(total, merged_total, strict=True):
                self.emit(f"{total_var} = {merged}")
            self.emit(f"{total_valid} = {merged_total_valid}")
            before = self.new_var("scan_before")
            self.emit(
                f"{before} = {order} > {warp_in_group}"
                if self.reverse
                else f"{before} = {order} < {warp_in_group}"
            )
            merged_prefix, merged_prefix_valid = self.merge(
                streams,
                prefix,
                warp_totals,
                None,
                left_valid=prefix_valid,
                right_valid=warp_total_valid,
            )
            for pre_var, merged in zip(prefix, merged_prefix, strict=True):
                self.emit(f"{pre_var} = ({merged}) if {before} else ({pre_var})")
            self.emit(
                f"{prefix_valid} = ({merged_prefix_valid}) if {before} else ({prefix_valid})"
            )
        # Every thread has read the totals before the next chunk overwrites them.
        self.emit("cute.arch.sync_threads()")
        outs, out_valid = self.merge(
            streams, prefix, inclusive, prefix_valid, right_valid=inclusive_valid
        )
        return outs, out_valid, (total, total_valid if self.track_valid else None)


def _cute_scan_axis_is_unit(size: object) -> bool:
    """Whether the scanned axis statically has exactly one element."""
    from ..compile_environment import CompileEnvironment

    if isinstance(size, int):
        return size == 1
    if isinstance(size, torch.SymInt):
        return CompileEnvironment.current().known_equal(size, 1)
    return False


def _cute_try_parallel_scan(
    state: CodegenState,
    helper_graph_info: HelperFunctionGraphInfo,
    input_nodes: list[object],
    dim: int,
    reverse: bool,
) -> list[ast.AST] | str:
    """Lower the scan on the already-loaded per-lane values when provable.

    Returns one output expression per stream, or the reason the serial
    fallback has to take the scan.  See the module docstring for the shapes
    handled.
    """
    from torch.fx.node import Node

    from ..ast_extension import expr_from_string
    from .cute_reshape import _resolve_dim_block_id

    if not input_nodes or not all(isinstance(node, Node) for node in input_nodes):
        return "an input is not a traced tile"
    if any(_cute_scan_input_is_sorted(node) for node in input_nodes):
        return "an input is torch.sort output"
    fake_values = [cast("Node", node).meta.get("val") for node in input_nodes]
    if not all(isinstance(value, torch.Tensor) for value in fake_values):
        return "an input is not a tensor"
    first = cast("torch.Tensor", fake_values[0])
    if dim < 0:
        dim += first.ndim
    if not 0 <= dim < first.ndim:
        return f"scan dim {dim} is out of range"
    ast_arg = state.ast_args[1]
    args = list(ast_arg) if isinstance(ast_arg, (tuple, list)) else [ast_arg]
    if len(args) != len(fake_values) or not all(
        isinstance(arg, ast.AST) for arg in args
    ):
        return "an input has no per-lane value"
    passthrough = [expr_from_string(ast.unparse(cast("ast.AST", arg))) for arg in args]
    if _cute_scan_axis_is_unit(first.shape[dim]):
        # An inclusive scan over one element is the identity for any combine;
        # a size-1 axis also has no tile block to resolve below.
        return passthrough
    block_id = _resolve_dim_block_id(state.codegen, first, dim)
    if block_id is None:
        return "the scan dim has no block"
    geometry = _cute_scan_geometry(state, block_id)
    if geometry is None:
        return (
            "the scan axis layout is not a lane loop or a strided thread "
            "split on thread axis 0"
        )
    if geometry.threads > _CUTE_WARP_SIZE and any(
        cast("torch.Tensor", value).dtype is torch.bool for value in fake_values
    ):
        return "a bool stream spans more than one warp"
    if geometry.extent == 1:
        return passthrough
    if reverse and geometry.vec_lane_var is not None:
        # The constexpr vector loop over the scan axis cannot run backwards:
        # the vector store protocol appends the ``V`` results in loop order.
        return "a reverse scan cannot run a vectorized lane loop backwards"
    if _cute_scan_lane_hosts_reduction(state):
        return (
            "a lane loop around it also hosts a reduction over a lane-looped "
            "block, whose two-pass split cannot carry the scan prefix (a "
            "config that puts the reduced block on threads avoids the lane loop)"
        )
    if not _cute_scan_prepare_lane_direction(state, geometry, reverse):
        return "the lane loop cannot be visited in scan order"
    emitter = _CuteScanEmitter(state, geometry, helper_graph_info, reverse)
    streams = [
        emitter.stream(cast("ast.AST", arg), cast("torch.Tensor", value))
        for arg, value in zip(args, fake_values, strict=True)
    ]
    if geometry.threads == 1:
        outs = emitter.emit_lane_loop_scan(streams)
    else:
        outs = emitter.emit_thread_split_scan(streams)
    return [expr_from_string(out) for out in outs]
