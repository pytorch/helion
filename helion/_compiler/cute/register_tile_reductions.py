"""Register-tile lowering of lane reductions nested in trace-time tile loops.

A persistent synthetic reduction lane registered OUTSIDE the one-vector tile
wrappers of its grid (``DeviceGridState.nest_reduction_lane_outside_vector_tiles``)
wraps a nest of ``cutlass.range_constexpr`` loops::

    for lane in cutlass.range_constexpr(L):          # reduction lane (owner)
        row = tid + lane * T
        for tile_lane in cutlass.range_constexpr(1): # one-vector tile wrapper
            base = ...
            packet = cute.arch.load(state + row * S + base, V)  # one LDG.128
            vals = []
            out_vals = []
            for vec_lane in cutlass.range_constexpr(V):
                elem = packet[vec_lane] ...
                new = elem * decay + k * v
                vals.append(new)
                r = _helion_lane_reduce(q * new, ..., 'lane')   # marker
                out_vals.append(r)
            _cute_store_u32_vec(state + row * S + base, vals)
            _cute_store_u32_vec(out + base, out_vals)

Every loop unrolls at trace time, so the per-thread body is a register tile.
This module rewrites the owner loop into:

1. an accumulate nest running every pure producer (all loads, no stores) and
   folding each marker input into a per-tile-element accumulator list, with
   values the later nests need stashed in trace-time Python lists;
2. one cross-thread combine per marker over that list: a warp shuffle per
   element for a one-warp group, or one shared-memory column reduce
   (``_cute_grouped_reduce_shared_columns``) for a multi-warp group;
3. a consume nest for lane-varying side effects (the state store) and a
   once-only tail nest for lane-invariant side effects (the reduced output),
   reading stashed values and finalized results instead of re-running loads.

Every lane's loads are therefore issued before the first store, each tile
element is reduced across threads exactly once, and no load is repeated.

The rewrite fails closed: anything outside this shape raises
``RegisterTileUnsupported`` naming the construct (rolled inner loops, control
flow, unknown effects, loop-carried scalars, reads across loop scopes,
dependent markers, stores inside element loops, a register or shared-memory
budget overrun, and any potentially aliasing write,
``_lane_split_reorders_aliasing_memory``).  ``generate_ast`` then regenerates
the kernel with the rolled lane nesting the persistent strategy used before
register tiles existed, so such a config compiles exactly as it did then.
Bodies the device IR already shows to be outside the shape never get here:
``register_tile_body_admitted`` (``register_tile_admission.py``) keeps the
rolled lane nesting for them without a second codegen pass.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import Callable
from typing import NoReturn

from .. import reduction_strategy as reductions
from .. import tile_strategy as lanes
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import ReadWrites
from ..ast_read_writes import ast_rename
from .fuse_two_pass_loads import _is_store_call
from .hoist_lane_invariant_reductions import _CUTLASS_SCALAR_TYPES
from .persistent_branch_vec import _memory_load_calls
from .register_tile_admission import RegisterTileUnsupported
from .thread_budget import CUTE_REGISTER_TILE_MAX_LIVE_VALUES

if TYPE_CHECKING:
    from collections.abc import Sequence

_IGNORED_READS = frozenset({*lanes._CUTE_UNIFORM_GLOBAL_NAMES, "ir"})


def _reject(reason: str) -> NoReturn:
    raise RegisterTileUnsupported(reason)


@dataclasses.dataclass
class _Loop:
    var: str
    extent: int
    node: ast.For


@dataclasses.dataclass
class _Leaf:
    index: int
    stmt: ast.AST
    path: tuple[int, ...]
    kind: str  # "marker" | "pure" | "append" | "store"
    defines: str | None
    reads: frozenset[str]
    marker: lanes._LaneReduceMarker | None = None
    appended_list: str | None = None


@dataclasses.dataclass
class _Nest:
    owner: str
    extent: int
    loops: list[_Loop]
    leaves: list[_Leaf]
    loop_ids: dict[int, int]
    leaf_ids: dict[int, _Leaf]


def _constexpr_extent(loop: ast.For) -> int | None:
    iterator = loop.iter
    if (
        not isinstance(iterator, ast.Call)
        or ast.unparse(iterator.func) != "cutlass.range_constexpr"
        or len(iterator.args) != 1
        or iterator.keywords
        or not isinstance(loop.target, ast.Name)
        or loop.orelse
    ):
        return None
    extent = iterator.args[0]
    if not isinstance(extent, ast.Constant) or not isinstance(extent.value, int):
        return None
    return extent.value if extent.value >= 1 else None


def _append_target(stmt: ast.AST) -> tuple[str, ast.expr] | None:
    """``NAME.append(value)`` staging a trace-time list."""
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "append"
        and isinstance(stmt.value.func.value, ast.Name)
        and len(stmt.value.args) == 1
        and not stmt.value.keywords
    ):
        return stmt.value.func.value.id, stmt.value.args[0]
    return None


def _is_store_statement(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and _is_store_call(stmt.value)
    )


def _collect_nest(loop: ast.For, lane_var: str) -> _Nest:
    extent = _constexpr_extent(loop)
    if extent is None or extent < 2:
        _reject("the owner lane is not a trace-time loop over several lanes")
    nest = _Nest(lane_var, extent, [], [], {}, {})
    loop_vars = {lane_var}

    def visit(statements: Sequence[ast.AST], path: tuple[int, ...]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.For):
                inner_extent = _constexpr_extent(stmt)
                if inner_extent is None:
                    _reject(
                        f"a rolled loop inside the tile: {ast.unparse(stmt.iter)[:60]}"
                    )
                target = stmt.target
                assert isinstance(target, ast.Name)
                if target.id in loop_vars:
                    _reject(f"element loop variable {target.id} is reused")
                loop_vars.add(target.id)
                index = len(nest.loops)
                nest.loops.append(_Loop(target.id, inner_extent, stmt))
                nest.loop_ids[id(stmt)] = index
                visit(stmt.body, (*path, index))
                continue
            reads = frozenset(ReadWrites.from_ast(stmt).reads) - _IGNORED_READS
            marker = lanes._is_lane_reduce_marker_assign(stmt)
            leaf: _Leaf
            if marker is not None:
                leaf = _Leaf(
                    len(nest.leaves), stmt, path, "marker", marker.result_var, reads
                )
                leaf.marker = marker
            elif (append := _append_target(stmt)) is not None:
                leaf = _Leaf(len(nest.leaves), stmt, path, "append", None, reads)
                leaf.appended_list = append[0]
            elif _is_store_statement(stmt):
                leaf = _Leaf(len(nest.leaves), stmt, path, "store", None, reads)
            elif (
                (name := lanes._plain_assignment_name(stmt)) is not None
                and lanes._is_proven_relocatable_assignment(stmt, allow_load=True)
                and not lanes._contains_unduplicatable_op(stmt)
            ):
                leaf = _Leaf(len(nest.leaves), stmt, path, "pure", name, reads)
            else:
                _reject(
                    "a statement that cannot move between passes: "
                    f"{ast.unparse(stmt)[:80]}"
                )
            nest.leaves.append(leaf)
            nest.leaf_ids[id(stmt)] = leaf

    visit(loop.body, ())
    if any(
        leaf.defines in loop_vars for leaf in nest.leaves if leaf.defines is not None
    ):
        _reject("a loop variable is reassigned inside the tile")
    return nest


def _flat_index(nest: _Nest, path: tuple[int, ...], *, with_owner: bool) -> str:
    """Trace-time position of an iteration within the (owner x) inner loops."""
    terms: list[str] = []
    inner = 1
    for index in path:
        inner *= nest.loops[index].extent
    if with_owner:
        terms.append(f"{nest.owner} * {inner}")
    for position, index in enumerate(path):
        stride = 1
        for later in path[position + 1 :]:
            stride *= nest.loops[later].extent
        loop = nest.loops[index]
        terms.append(loop.var if stride == 1 else f"{loop.var} * {stride}")
    return " + ".join(terms) if terms else "0"


def _tile_count(nest: _Nest, path: tuple[int, ...]) -> int:
    count = 1
    for index in path:
        count *= nest.loops[index].extent
    return count


def _has_carry(nest: _Nest, rename_groups: dict[str, str]) -> bool:
    """A name read before its definition and written in the body is carried."""
    loop_vars = {nest.owner, *(loop.var for loop in nest.loops)}
    written: set[str] = set()
    all_written: set[str] = set()
    renamed: list[ast.AST] = []
    for leaf in nest.leaves:
        clone = lanes._clone_stmt(leaf.stmt)
        ast_rename(clone, rename_groups)
        renamed.append(clone)
        all_written.update(ReadWrites.from_ast(clone).writes)
    for clone in renamed:
        effects = ReadWrites.from_ast(clone)
        reads = set(effects.reads) - loop_vars - _IGNORED_READS
        if (reads & all_written) - written:
            return True
        written.update(effects.writes)
    return False


def _clone_loop(loop: _Loop, body: list[ast.AST]) -> ast.For:
    cloned = create(
        ast.For,
        target=create(ast.Name, id=loop.var, ctx=ast.Store()),
        iter=expr_from_string(f"cutlass.range_constexpr({loop.extent})"),
        body=body,
        orelse=[],
        type_comment=None,
    )
    lane = getattr(loop.node, lanes.HELION_LANE_LOOP_VAR_ATTR, None)
    if isinstance(lane, str):
        setattr(cloned, lanes.HELION_LANE_LOOP_VAR_ATTR, lane)
    return cloned


def _emit_nest(
    nest: _Nest,
    statements: Sequence[ast.AST],
    select: Callable[[_Leaf], list[ast.AST]],
) -> list[ast.AST]:
    result: list[ast.AST] = []
    for stmt in statements:
        if isinstance(stmt, ast.For) and id(stmt) in nest.loop_ids:
            loop = nest.loops[nest.loop_ids[id(stmt)]]
            inner = _emit_nest(nest, stmt.body, select)
            if inner:
                result.append(_clone_loop(loop, inner))
            continue
        leaf = nest.leaf_ids.get(id(stmt))
        if leaf is not None:
            result.extend(select(leaf))
    return result


def _single_reduce_expr(marker: lanes._LaneReduceMarker, acc: str) -> str:
    """Cross-thread combine of one accumulator element within one warp; the
    caller routes multi-warp groups to the shared-memory forms first."""
    if marker.group_span > 32:
        _reject("a reduction group spans several warps without a warp multiple")
    if marker.group_span > 1 and marker.group_pre > 1 and marker.group_lane_expr:
        return lanes._grouped_warp_reduce_expr(
            marker.reduction_type,
            acc,
            marker.identity_expr,
            marker.group_lane_expr,
            pre=marker.group_pre,
            group_span=marker.group_span,
        )
    if marker.threads_in_group > 32:
        _reject("a reduction group spans several warps without a group lane")
    if marker.threads_in_group > 1:
        return lanes._warp_reduce_expr(
            marker.reduction_type, acc, marker.threads_in_group
        )
    return acc


def _static_smem_bytes(
    marker: lanes._LaneReduceMarker, count: int, slots_per_column: int
) -> int:
    """Static shared memory a combine of ``count`` columns allocates."""
    ctor = lanes._dtype_ctor_from_identity(marker.identity_expr)
    itemsize = _CUTLASS_SCALAR_TYPES.get(ctor or "", 8)
    return count * slots_per_column * itemsize


def _finalize_marker(
    marker: lanes._LaneReduceMarker, acc_list: str, results: str, count: int
) -> tuple[list[ast.AST], int]:
    """Combine every accumulator element across the live thread group.

    Emitted as straight-line statements (``count`` is a small register tile)
    so no later V-loop peephole mistakes the per-element combines for one
    reduction over the elements.  Returns the statements and the static
    shared memory (bytes) they allocate.
    """
    if marker.group_cluster_n > 1:
        _reject("cute_cluster_n > 1 cannot combine a register tile across CTAs")
    grouped = marker.group_span > 1 and bool(marker.group_lane_expr)
    cross_warp = grouped and marker.group_span > 32 and marker.group_span % 32 == 0
    # The shared-memory combines key their slots on the lane the emitter
    # chose for shared memory (the full runtime thread id when a redundant
    # thread axis may still be mapped later); ``group_count`` counts the
    # groups of that keying.  The warp shuffles below keep the static lane.
    shared_lane_expr = marker.shared_lane_expr or marker.group_lane_expr
    # One shared-memory column reduce for the whole tile: consecutive
    # reduction threads spanning several warps, or (any span) reduction
    # threads interleaved above ``pre`` sibling coordinates, where the
    # per-column masked-shuffle fold would cost ``pre`` shuffles per column.
    if (grouped and marker.group_pre > 1 or cross_warp) and marker.reduction_type in (
        "sum",
        "prod",
    ):
        lane_var = f"{acc_list}_lane"
        lane_in_group_var = f"{acc_list}_lane_in_group"
        reduced = f"{acc_list}_reduced"
        elements = [marker.finalize_expr(f"{reduced}[{t}]") for t in range(count)]
        # One slot per warp (``pre == 1``) or per thread (``pre > 1``) of every
        # group, per column (``_cute_grouped_reduce_shared_columns``).
        slots = marker.group_count * (
            marker.group_span // 32 if marker.group_pre == 1 else marker.group_span
        )
        return [
            statement_from_string(f"{lane_var} = {shared_lane_expr}"),
            statement_from_string(
                f"{lane_in_group_var} = ({lane_var}) % {marker.group_span}"
            ),
            statement_from_string(
                f"{reduced} = _cute_grouped_reduce_shared_columns("
                f"{acc_list}, {marker.reduction_type!r}, {marker.identity_expr}, "
                f"{lane_var}, {lane_in_group_var}, pre={marker.group_pre}, "
                f"group_span={marker.group_span}, group_count={marker.group_count})"
            ),
            statement_from_string(f"{results} = [{', '.join(elements)}]"),
        ], _static_smem_bytes(marker, count, slots)
    if cross_warp:
        statements: list[ast.AST] = [statement_from_string(f"{results} = []")]
        for t in range(count):
            reduced = f"{acc_list}_reduced_{t}"
            statements.extend(
                lanes._grouped_two_stage_reduce_stmts(
                    reduced,
                    marker.reduction_type,
                    f"{acc_list}[{t}]",
                    marker.identity_expr,
                    shared_lane_expr,
                    pre=marker.group_pre,
                    group_span=marker.group_span,
                    group_count=marker.group_count,
                )
            )
            statements.append(
                statement_from_string(
                    f"{results}.append({marker.finalize_expr(reduced)})"
                )
            )
        # Every two-stage call stages ``pre`` partials per warp plus one
        # result per sibling coordinate for each group.
        slots = marker.group_count * marker.group_pre * (marker.group_span // 32 + 1)
        return statements, _static_smem_bytes(marker, count, slots)
    elements = [
        marker.finalize_expr(_single_reduce_expr(marker, f"{acc_list}[{t}]"))
        for t in range(count)
    ]
    return [statement_from_string(f"{results} = [{', '.join(elements)}]")], 0


def lower_register_tile_lane_loop(
    loop: ast.For,
    lane_var: str,
    *,
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    proven_tensor_stride_values: dict[tuple[str, int], int],
    thread_axis_names: dict[str, frozenset[int]],
    scalar_definitions: dict[str, ast.AST],
    rename_groups: dict[str, str],
) -> list[ast.AST]:
    """Lower a constexpr owner lane whose markers sit in nested constexpr loops.

    Returns the replacement statements; a body outside the proved shape raises
    ``BackendUnsupported`` naming the offending construct.
    """
    nest = _collect_nest(loop, lane_var)
    leaves = nest.leaves
    markers = [leaf for leaf in leaves if leaf.kind == "marker"]
    if not markers:
        _reject("no reduction marker inside the tile")
    if any(leaf.path == () for leaf in markers):
        _reject("a reduction marker outside every element loop")
    if any(
        leaf.marker is None
        or leaf.marker.owner_lane != lane_var
        or leaf.marker.matmul_contribution
        for leaf in markers
    ):
        _reject("a reduction marker of another lane or of a matmul contraction")
    loop_vars = {lane_var, *(inner.var for inner in nest.loops)}

    # Unique definitions, no loop-carried scalars, and no read across a loop
    # scope (a name defined in an inner loop read after that loop ends).
    definers: dict[str, _Leaf] = {}
    for leaf in leaves:
        if leaf.defines is None:
            continue
        if leaf.defines in definers or leaf.defines in loop_vars:
            _reject(f"{leaf.defines} is defined twice inside the tile")
        definers[leaf.defines] = leaf
    if _has_carry(nest, rename_groups):
        _reject("a loop-carried scalar")
    for leaf in leaves:
        for name in leaf.reads:
            definer = definers.get(name)
            if definer is None:
                continue
            if definer.index >= leaf.index:
                _reject(f"{name} is read before its definition")
            if leaf.path[: len(definer.path)] != definer.path:
                _reject(f"{name} is read outside the loop that defines it")
        if leaf.kind == "append":
            assert leaf.appended_list is not None
            list_definer = definers.get(leaf.appended_list)
            if list_definer is None or list_definer.kind != "pure":
                _reject(f"staging list {leaf.appended_list} is not defined in the tile")
    # A store repeated inside an element loop would write the same or an
    # element-dependent address several times per lane; only hoisted flushes
    # (outside every multi-trip inner loop) are scheduled here.
    for leaf in leaves:
        if leaf.kind == "store" and _tile_count(nest, leaf.path) != 1:
            _reject("a store repeats inside an element loop")

    # Marker results (finalized after the accumulate nest) taint the pure
    # statements and lists that consume them; those run after finalize.
    result_names = {leaf.defines for leaf in markers if leaf.defines is not None}
    result_tainted = set(result_names)
    changed = True
    while changed:
        changed = False
        for leaf in leaves:
            if leaf.kind == "pure" and leaf.defines not in result_tainted:
                if leaf.reads & result_tainted:
                    result_tainted.add(leaf.defines)  # type: ignore[arg-type]
                    changed = True
            elif leaf.kind == "append" and leaf.appended_list not in result_tainted:
                if leaf.reads & result_tainted:
                    result_tainted.add(leaf.appended_list)  # type: ignore[arg-type]
                    changed = True
    post = {
        leaf.index
        for leaf in leaves
        if leaf.kind == "pure" and leaf.reads & result_tainted
    }
    if any(leaf.reads & result_tainted for leaf in markers):
        _reject("a reduction input depends on another reduction's result")

    # Statements that only rearrange coordinates (no loads, no data values)
    # are cheap to re-run in every nest; everything else computed before the
    # combine is stashed once and read back.
    index_only: set[str] = set()
    for leaf in leaves:
        if leaf.kind != "pure" or leaf.index in post:
            continue
        if _memory_load_calls(leaf.stmt):
            continue
        if all(
            name in loop_vars or name in index_only or name not in definers
            for name in leaf.reads
        ):
            assert leaf.defines is not None
            index_only.add(leaf.defines)

    # Lane variance with finalized markers: a marker result is uniform across
    # the owner lane, so only its inputs (not its consumers) vary with it.
    varying: set[str] = {lane_var}
    changed = True
    while changed:
        changed = False
        for leaf in leaves:
            if leaf.kind == "pure":
                if leaf.defines not in varying and leaf.reads & varying:
                    varying.add(leaf.defines)  # type: ignore[arg-type]
                    changed = True
            elif leaf.kind == "append":
                if leaf.appended_list not in varying and leaf.reads & varying:
                    varying.add(leaf.appended_list)  # type: ignore[arg-type]
                    changed = True

    def is_varying(leaf: _Leaf) -> bool:
        return bool(leaf.reads & varying)

    # Everything that runs after the combine: the side effects (staging appends
    # and stores) and the pure values derived from a reduction result.
    consumers = [
        leaf
        for leaf in leaves
        if leaf.kind in ("append", "store") or leaf.index in post
    ]
    stash_roots: set[str] = set()
    for leaf in consumers:
        for name in leaf.reads:
            definer = definers.get(name)
            if (
                definer is not None
                and definer.kind == "pure"
                and definer.index not in post
                and name not in index_only
            ):
                stash_roots.add(name)

    # Accumulate nest: backward slice of the marker inputs and stashed values
    # over the pure producers.
    marker_inputs = {leaf.marker.input_name for leaf in markers if leaf.marker}
    needed = set(marker_inputs) | stash_roots
    phase1: set[int] = set()
    for leaf in reversed(leaves):
        if leaf.kind != "pure" or leaf.index in post:
            continue
        if leaf.defines in needed:
            phase1.add(leaf.index)
            needed |= leaf.reads
    if needed & result_names:
        _reject("an accumulate-pass producer depends on a reduction result")

    # Register budget: every stashed value lives once per lane per element and
    # every accumulator once per element until the combine.
    live_values = sum(
        nest.extent * _tile_count(nest, definers[name].path) for name in stash_roots
    ) + sum(_tile_count(nest, leaf.path) for leaf in markers)
    if live_values > CUTE_REGISTER_TILE_MAX_LIVE_VALUES:
        _reject(
            f"{live_values} live values exceed the budget of "
            f"{CUTE_REGISTER_TILE_MAX_LIVE_VALUES}"
        )

    # Every load of the accumulate nest moves above every store of the
    # consume nest; prove no store can reach a load of another lane.
    flat: list[ast.AST] = []
    substitutions = {inner.var: "0" for inner in nest.loops if inner.extent == 1}
    for leaf in leaves:
        clone = lanes._clone_stmt(leaf.stmt)
        if substitutions:
            ast_rename(clone, substitutions)
        flat.append(clone)
    flat_markers = [
        (leaf.index, leaf.marker) for leaf in markers if leaf.marker is not None
    ]
    additional_inputs = [
        (definers[name].index + 1, name) for name in sorted(stash_roots)
    ]
    if lanes._lane_split_reorders_aliasing_memory(
        flat,
        flat_markers,
        proven_disjoint_tensor_pairs,
        lane_var,
        nest.extent,
        proven_tensor_stride_values,
        additional_inputs=additional_inputs,
    ):
        _reject("a load of one lane may alias a store of another lane")

    # Owner predicates for the once-only tail: physical-thread provenance is
    # tracked over the leaves in program order, and a list staged with a
    # marker result carries that result's dependence to its flush.
    thread_axes = dict(thread_axis_names)
    scalar_defs = dict(scalar_definitions)
    axes_before: dict[int, dict[str, frozenset[int]]] = {}
    defs_before: dict[int, dict[str, ast.AST]] = {}
    for leaf in leaves:
        axes_before[leaf.index] = dict(thread_axes)
        defs_before[leaf.index] = dict(scalar_defs)
        lanes._update_thread_axis_names(leaf.stmt, thread_axes)
        lanes._update_scalar_definitions(leaf.stmt, scalar_defs)
    marker_list = [(leaf.index, leaf.marker) for leaf in markers if leaf.marker]
    marker_dependencies: list[tuple[lanes._LaneReduceMarker, set[str]]] = []
    for leaf in markers:
        assert leaf.marker is not None
        dependent = {leaf.marker.result_var}
        changed = True
        while changed:
            changed = False
            for other in leaves:
                target = (
                    other.appended_list if other.kind == "append" else other.defines
                )
                if target is None or target in dependent:
                    continue
                if other.reads & dependent:
                    dependent.add(target)
                    changed = True
        marker_dependencies.append((leaf.marker, dependent))

    def guarded(leaf: _Leaf) -> ast.AST:
        stmt = lanes._clone_stmt(leaf.stmt)
        if leaf.kind != "store":
            return stmt
        owner_exprs = lanes._lane_reduction_owner_exprs_for_statement(
            stmt,
            marker_list,  # type: ignore[arg-type]
            marker_dependencies,
            axes_before[leaf.index],
            defs_before[leaf.index],
        )
        return lanes._guard_stmt_with_owner(stmt, owner_exprs) if owner_exprs else stmt

    stash_name = {name: f"{name}_lane_stash" for name in sorted(stash_roots)}
    acc_name: dict[int, str] = {}
    results_name: dict[int, str] = {}
    prologue: list[ast.AST] = []
    for leaf in markers:
        assert leaf.marker is not None
        acc = f"{leaf.marker.result_var}_lane_acc"
        acc_name[leaf.index] = acc
        results_name[leaf.index] = f"{leaf.marker.result_var}_lane_results"
        count = _tile_count(nest, leaf.path)
        prologue.append(
            statement_from_string(
                f"{acc} = [{', '.join([leaf.marker.identity_expr] * count)}]"
            )
        )
    for name in sorted(stash_roots):
        prologue.append(statement_from_string(f"{stash_name[name]} = []"))

    def select_phase1(leaf: _Leaf) -> list[ast.AST]:
        if leaf.kind == "marker":
            marker = leaf.marker
            assert marker is not None
            acc = acc_name[leaf.index]
            position = _flat_index(nest, leaf.path, with_owner=False)
            ctor = lanes._dtype_ctor_from_identity(marker.identity_expr)
            value = f"{ctor}({marker.input_name})" if ctor else marker.input_name
            combined = lanes._combine_expr(
                marker.reduction_type, f"{acc}[{position}]", value
            )
            return [statement_from_string(f"{acc}[{position}] = {combined}")]
        if leaf.index not in phase1:
            return []
        statements = [lanes._clone_stmt(leaf.stmt)]
        if leaf.defines is not None and leaf.defines in stash_roots:
            statements.append(
                statement_from_string(
                    f"{stash_name[leaf.defines]}.append({leaf.defines})"
                )
            )
        return statements

    def needed_names(selected: list[_Leaf]) -> set[str]:
        """Names a nest must define for ``selected``: stashed values and
        marker results are read back, coordinate-only statements and values
        derived from a reduction result are re-run (so their reads count)."""
        names: set[str] = set()
        pending = [name for leaf in selected for name in leaf.reads]
        while pending:
            name = pending.pop()
            if name in names or name not in definers:
                continue
            names.add(name)
            definer = definers[name]
            if name in index_only or definer.index in post:
                pending.extend(definer.reads)
        return names

    def select_consume(
        selected: set[int], required: set[str], *, with_owner: bool
    ) -> Callable[[_Leaf], list[ast.AST]]:
        def select(leaf: _Leaf) -> list[ast.AST]:
            if leaf.kind == "marker":
                if leaf.defines not in required:
                    return []
                position = _flat_index(nest, leaf.path, with_owner=False)
                return [
                    statement_from_string(
                        f"{leaf.defines} = {results_name[leaf.index]}[{position}]"
                    )
                ]
            if leaf.index in selected:
                return [guarded(leaf)]
            if leaf.defines is None or leaf.defines not in required:
                return []
            if leaf.defines in index_only or leaf.index in post:
                return [lanes._clone_stmt(leaf.stmt)]
            assert leaf.defines in stash_roots
            position = _flat_index(nest, leaf.path, with_owner=with_owner)
            return [
                statement_from_string(
                    f"{leaf.defines} = {stash_name[leaf.defines]}[{position}]"
                )
            ]

        return select

    # A lane-invariant side effect never reads a lane-varying value, so the
    # per-lane consume nest and the once-only tail nest are independent; the
    # pure values each needs (including those derived from a reduction result)
    # are re-emitted in place by ``select_consume``.
    side_effects = [leaf for leaf in consumers if leaf.kind in ("append", "store")]
    consume_leaves = [leaf for leaf in side_effects if is_varying(leaf)]
    tail_leaves = [leaf for leaf in side_effects if not is_varying(leaf)]

    result: list[ast.AST] = list(prologue)
    result.append(
        lanes._clone_lane_loop_with_body(
            loop, _emit_nest(nest, loop.body, select_phase1)
        )
    )
    smem_bytes = 0
    for leaf in markers:
        assert leaf.marker is not None
        statements, allocated = _finalize_marker(
            leaf.marker,
            acc_name[leaf.index],
            results_name[leaf.index],
            _tile_count(nest, leaf.path),
        )
        result.extend(statements)
        smem_bytes += allocated
    if smem_bytes and smem_bytes > reductions._cute_shared_memory_budget_bytes():
        _reject(f"{smem_bytes} bytes of static shared memory exceed the budget")
    if consume_leaves:
        required = needed_names(consume_leaves)
        body = _emit_nest(
            nest,
            loop.body,
            select_consume(
                {leaf.index for leaf in consume_leaves}, required, with_owner=True
            ),
        )
        result.append(lanes._clone_lane_loop_with_body(loop, body))
    if tail_leaves:
        required = needed_names(tail_leaves)
        result.extend(
            _emit_nest(
                nest,
                loop.body,
                select_consume(
                    {leaf.index for leaf in tail_leaves}, required, with_owner=False
                ),
            )
        )
    return result
