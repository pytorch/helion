"""Thread axes of free ``hl.arange`` dims (CuTe).

A free ``hl.arange(n)`` used directly as a load/store index is bound to no
tile, reduction or grid block, so no strategy gives it a thread axis.  Each
distinct arange gets a CUDA thread axis of its own (``allocate_arange_axis``)
or, when the joint thread count would overflow the 1024-thread budget, a
sequential lane loop on the grid body (``allocate_arange_coord``); the dynamic
``_if`` branch each was allocated under is recorded so aranges of mutually
exclusive branches share one axis (``branch_scope``,
``record_strategy_axis_branch_path``), and the launch block grows to cover the
lanes.  The state is ``CuteCodegenState``'s ``synthetic_arange_*``,
``branch_path`` and ``strategy_axis_branch_paths``.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from torch._inductor.runtime.runtime_utils import next_power_of_2

from .state import cute_codegen_state
from .thread_budget import MAX_THREADS_PER_BLOCK
from .thread_budget import check_thread_limit
from .thread_budget import referenced_thread_axes

if TYPE_CHECKING:
    import ast
    from collections.abc import Iterator

    from ..generate_ast import GenerateAST


@contextlib.contextmanager
def branch_scope(
    cg: GenerateAST, if_node_id: int, branch_side: int, *, divergent: bool
) -> Iterator[None]:
    """Mark codegen as inside one branch of a dynamic ``_if``.

    ``branch_side`` is 0 for the ``if`` body and 1 for the ``else`` body.
    Used so synthetic ``hl.arange`` axes allocated in mutually-exclusive
    branches can share a single thread axis.  ``divergent`` is False for a
    condition every thread of the CTA evaluates alike, where a block-wide
    barrier stays placeable.
    """
    branch_path = cute_codegen_state(cg).branch_path
    branch_path.append((if_node_id, branch_side))
    try:
        with cg.divergent_control_flow() if divergent else contextlib.nullcontext():
            yield
    finally:
        branch_path.pop()


def _branch_paths_mutually_exclusive(
    path_a: list[tuple[int, int]], path_b: list[tuple[int, int]]
) -> bool:
    """True when two branch paths can never both execute."""
    from ..device_ir import DeviceIR

    return DeviceIR.branch_paths_mutually_exclusive(path_a, path_b)


def allocate_arange_coord(
    cg: GenerateAST, key: tuple[object, ...], size: int
) -> str | None:
    """Resolve a free ``hl.arange`` dim to a per-thread coordinate expr.

    Returns ``cute.arch.thread_idx()[axis]`` when the arange fits a fresh (or
    reusable) CUDA thread axis within the 1024-thread budget. When it would
    instead overflow the budget, the arange is chunked onto a sequential lane
    loop (``thread_idx()[axis] * 1 + lane * 1`` collapses to ``lane``, or
    ``thread_idx()[axis] + lane * nt`` when some thread lanes still fit) so
    the full extent stays addressable; the lane loop wraps the grid body.
    Returns ``None`` when no synthetic axis can be assigned (axis index >= 3).
    """
    state = cute_codegen_state(cg)
    lane_expr = state.synthetic_arange_lane_exprs.get(key)
    if lane_expr is not None:
        return lane_expr
    if key not in state.synthetic_arange_axes and _arange_needs_lane_loop(
        cg, key, size
    ):
        return _allocate_arange_lane_loop(cg, key, size)
    axis = allocate_arange_axis(cg, key, size)
    if axis >= 3:
        return None
    return f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"


def _arange_proposed_total(cg: GenerateAST, axis: int, size: int) -> int:
    """Joint thread count if ``size`` were placed on a fresh ``axis``."""
    proposed_sizes = dict(cute_codegen_state(cg).synthetic_arange_axis_sizes)
    proposed_sizes[axis] = size
    strategy_threads = 1
    for strat_axis, strat_size in cg._strategy_thread_axis_sizes().items():
        if strat_axis not in proposed_sizes:
            strategy_threads *= strat_size
    total = strategy_threads
    for axis_size in proposed_sizes.values():
        total *= axis_size
    return total


def _arange_needs_lane_loop(
    cg: GenerateAST, key: tuple[object, ...], size: int
) -> bool:
    # The lane loop is hosted by the grid body wrapper, so only chunk when a
    # grid state exists to carry it; otherwise keep the (raising) thread-axis
    # path rather than emitting an un-iterated lane variable.
    if cg.current_grid_state is None:
        return False
    # A mutually-exclusive branch reuse never grows the budget, so prefer it.
    if _mutually_exclusive_axis(cg, size) is not None:
        return False
    used_axes = _strategy_thread_axes(cg)
    used_axes.update(cute_codegen_state(cg).synthetic_arange_axes.values())
    axis = 0
    while axis in used_axes:
        axis += 1
    if axis >= 3:
        return True
    return _arange_proposed_total(cg, axis, size) > MAX_THREADS_PER_BLOCK


def _allocate_arange_lane_loop(
    cg: GenerateAST, key: tuple[object, ...], size: int
) -> str:
    """Chunk a free ``hl.arange`` onto a sequential lane loop.

    Uses ``nt`` live thread lanes on a fresh axis (``nt`` is the largest
    power-of-2 that keeps the joint budget within 1024, possibly 1) and a
    ``ceil(size / nt)`` sequential lane loop covering the rest. The arange's
    per-thread coordinate is ``thread_idx()[axis] + lane * nt``.
    """
    state = cute_codegen_state(cg)
    used_axes = _strategy_thread_axes(cg)
    used_axes.update(state.synthetic_arange_axes.values())
    axis = 0
    while axis in used_axes and axis < 3:
        axis += 1

    # Threads already committed (strategy axes + other synthetic axes).
    committed = 1
    for strat_size in cg._strategy_thread_axis_sizes().values():
        committed *= strat_size
    for axis_size in state.synthetic_arange_axis_sizes.values():
        committed *= axis_size
    budget = max(1, MAX_THREADS_PER_BLOCK // max(1, committed))
    nt = 1
    if axis < 3:
        nt = min(next_power_of_2(size), 1 << (budget.bit_length() - 1))
        nt = max(1, min(nt, size))
    lane_extent = (size + nt - 1) // nt

    lane_var = cg.device_function.new_var(
        f"arange_lane_{len(state.synthetic_arange_lane_exprs)}", dce=False
    )
    grid_state = cg.current_grid_state
    if grid_state is not None:
        grid_state.add_lane_loop(-1, lane_var, lane_extent)
    if nt > 1 and axis < 3:
        state.synthetic_arange_axes[key] = axis
        state.synthetic_arange_axis_sizes[axis] = max(
            state.synthetic_arange_axis_sizes.get(axis, 1), nt
        )
        _record_arange_axis_branch_path(cg, axis)
        cg._record_active_thread_axis_sizes()
        coord = (
            f"(cutlass.Int32(cute.arch.thread_idx()[{axis}])"
            f" + cutlass.Int32({lane_var}) * {nt})"
        )
    else:
        coord = f"cutlass.Int32({lane_var})"
    state.synthetic_arange_lane_exprs[key] = coord
    return coord


def allocate_arange_axis(cg: GenerateAST, key: tuple[object, ...], size: int) -> int:
    """Allocate (or reuse) a per-thread axis for a free ``hl.arange`` dim.

    A free ``hl.arange(n)`` used directly as a load/store index is not
    bound to any tile/reduction/grid block id, so it has no strategy
    thread axis. We map each distinct arange onto its own CUDA thread
    axis: thread ``thread_idx()[axis]`` holds element ``axis``. Arange dims
    that share the same ``key`` (same length/start/step) describe the same
    logical lane and reuse the same axis so a value loaded on a lane is
    stored back on that lane.

    The chosen axis follows any thread axes already claimed by real tile
    strategies (so an ``hl.arange`` mixed with an ``hl.tile`` index does
    not collide with the tile's axis). The size is recorded so the
    launch-dim recovery enlarges the thread block accordingly.
    """
    state = cute_codegen_state(cg)
    existing = state.synthetic_arange_axes.get(key)
    if existing is not None:
        state.synthetic_arange_axis_sizes[existing] = max(
            state.synthetic_arange_axis_sizes.get(existing, 1), size
        )
        _record_arange_axis_branch_path(cg, existing)
        cg._record_active_thread_axis_sizes()
        return existing
    # Reuse a synthetic axis from a mutually-exclusive control-flow branch:
    # if every arange already mapped onto some axis lives in a branch that
    # can never co-execute with the current one, the axes never need lanes
    # at the same time, so they may share one thread axis (size = max). This
    # keeps the joint thread budget bounded for branch-by-grid kernels whose
    # branches each use a distinct free ``hl.arange``.
    shared = _mutually_exclusive_axis(cg, size)
    if shared is not None:
        state.synthetic_arange_axes[key] = shared
        state.synthetic_arange_axis_sizes[shared] = max(
            state.synthetic_arange_axis_sizes.get(shared, 1), size
        )
        _record_arange_axis_branch_path(cg, shared)
        cg._record_active_thread_axis_sizes()
        return shared
    used_axes = _strategy_thread_axes(cg)
    used_axes.update(state.synthetic_arange_axes.values())
    axis = 0
    while axis in used_axes:
        axis += 1
    # Validate the joint thread count once the new axis is added.
    check_thread_limit(
        _arange_proposed_total(cg, axis, size),
        context=f"free hl.arange axis size={size}",
    )
    state.synthetic_arange_axes[key] = axis
    state.synthetic_arange_axis_sizes[axis] = size
    _record_arange_axis_branch_path(cg, axis)
    cg._record_active_thread_axis_sizes()
    return axis


def _record_arange_axis_branch_path(cg: GenerateAST, axis: int) -> None:
    """Remember the current branch path for an arange mapped onto ``axis``."""
    state = cute_codegen_state(cg)
    paths = state.synthetic_arange_axis_branch_paths.setdefault(axis, [])
    current = list(state.branch_path)
    if current not in paths:
        paths.append(current)


def record_strategy_axis_branch_path(cg: GenerateAST, axis: int) -> None:
    """Remember the current branch path for a strategy (e.g. reduction) axis.

    Called from reduction codegen while the dynamic ``_if`` branch scope is
    live, so a free ``hl.arange`` in a mutually-exclusive sibling branch can
    reuse this axis rather than claiming a fresh one. Recording only happens
    inside a branch (an unbranched strategy axis is always co-live and must
    never be reused).
    """
    state = cute_codegen_state(cg)
    if not state.branch_path:
        return
    paths = state.strategy_axis_branch_paths.setdefault(axis, [])
    current = list(state.branch_path)
    if current not in paths:
        paths.append(current)


def _mutually_exclusive_axis(cg: GenerateAST, size: int) -> int | None:
    """Find a thread axis whose every recorded use is in a branch that can
    never co-execute with the current branch path, so it can be safely reused.

    Considers both synthetic ``hl.arange`` axes and strategy (reduction) axes:
    a free arange in one grid branch may reuse the axis a reduction claimed in
    a sibling branch when the two can never run together. An axis recorded in
    BOTH maps must be mutually exclusive across all of its recorded paths.
    A reduction's axis narrower than ``size`` is skipped: the reduction was
    emitted for its own lanes, and widening the launch would run surplus
    lanes through its cross-lane combine.
    """
    state = cute_codegen_state(cg)
    if not state.branch_path:
        return None
    current = list(state.branch_path)
    candidate_axes = (
        state.synthetic_arange_axis_branch_paths.keys()
        | state.strategy_axis_branch_paths.keys()
    )
    reserved = cg._all_strategy_reserved_axes()
    for axis in sorted(candidate_axes):
        if axis in state.strategy_axis_branch_paths and reserved.get(axis, 1) < size:
            continue
        paths = [
            *state.synthetic_arange_axis_branch_paths.get(axis, []),
            *state.strategy_axis_branch_paths.get(axis, []),
        ]
        if paths and all(
            _branch_paths_mutually_exclusive(current, path) for path in paths
        ):
            return axis
    return None


def _strategy_thread_axes(cg: GenerateAST) -> set[int]:
    """The thread axes the strategies occupy, sized or not."""
    axes = set(cg._strategy_thread_axis_sizes())
    axes.update(cg._all_strategy_reserved_axes())
    # A strategy can structurally occupy a thread axis even when its block
    # size is 1 (e.g. an ``hl.grid`` whose offset is
    # ``pid * BLOCK + thread_idx[0]`` with ``BLOCK == 1``). Such an axis is
    # not recorded in ``thread_axis_sizes`` (which only keeps sizes > 1) but
    # it is referenced in the already-emitted setup statements, so scan
    # those to avoid handing a synthetic arange an axis the grid already
    # uses (which would mis-filter most lanes via the grid's bounds mask).
    # The per-thread index definitions include a vec lane loop's base
    # (``(lane * NT + thread_idx[0]) * V`` with one thread), which sits in
    # the pre-built lane body rather than the setup statements.
    statement_groups: list[list[ast.AST]] = list(cg.statements_stack)
    grid_state = cg.current_grid_state
    if grid_state is not None:
        statement_groups.append(grid_state.lane_index_definitions()[0])
    for loops in cg.active_device_loops.values():
        for loop_state in loops:
            statement_groups.append(loop_state.lane_index_definitions()[0])
    for statements in statement_groups:
        axes.update(referenced_thread_axes(statements))
    return axes
