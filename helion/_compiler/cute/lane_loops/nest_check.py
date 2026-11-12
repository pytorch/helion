"""The lane loop nest of a device loop and the barrier pass of a root body (CuTe).

Role: device-body pass (run by the per-thread strategies, not from the
pipeline tables).  A per-thread strategy builds a device loop's lane loops
around the body before the body exists (``PerThreadNDTileStrategy.
codegen_device_loop``); once the body is complete, ``check_lane_loop_nest``
makes the nest the tile program (``CuteBackend.check_lane_loop_nest``): the
full-nest exactness check and the redistribution of ``placement.py``, then the
block-wide barriers of ``barriers.py``.  A root body is placed around its lane
loops by the shared ``DeviceGridState.wrap_body``; ``add_lane_nest_barriers``
is the CuTe pass it finishes with (``CuteBackend.add_lane_nest_barriers``).
``_thread_barrier_context`` gathers what the barrier pass needs to know about
the threads running a body from the device function being generated.
"""

from __future__ import annotations

import ast
import itertools
from typing import TYPE_CHECKING
from typing import cast

from ...ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ...ast_read_writes import ReadWrites
from ...compile_environment import CompileEnvironment
from ...device_function import DeviceFunction
from ...host_function import NoCurrentFunction
from ...reduction_strategy import LoopedReductionStrategy
from ...tile_strategy import DeviceGridState
from ...tile_strategy import DeviceLoopState
from ...tile_strategy import _apply_wrapper_barriers
from ...tile_strategy import _assign_lane_setup
from ...tile_strategy import _current_rename_groups
from ...tile_strategy import _cute_lane_coordinates
from ...tile_strategy import _cute_lane_scopes
from ...tile_strategy import _lane_loop_is_live
from ...tile_strategy import _live_lane_setup
from ...tile_strategy import _merge_carried_names
from ...tile_strategy import _vec_wrapper_statements
from ..access_regions import block_size_symbol
from ..access_regions import tile_begin_symbol
from .barriers import add_thread_barriers
from .markers import _encloses
from .markers import required_lane_reduce_owners
from .placement import distribute_lane_loops
from .placement import full_nest_inexactness
from .placement import inexact_nest_error
from .statements import LanePlacement
from .statements import _packets

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    import sympy

    from ...tile_strategy import DeviceLoopOrGridState
    from .statements import LaneScope


def tile_begin_shifts(
    state: DeviceLoopState, *, per_thread_loops: bool
) -> dict[sympy.Symbol, sympy.Expr | None]:
    """The begin symbol of each block this loop iterates and its step per iteration.

    The symbols the access regions of the body use for this loop's tiles
    (``tile_begin_symbol``, keyed on this loop instance).  A loop over one
    block advances its begin by the block size each iteration; a nest
    over several blocks steps them in turn, so the next iteration's
    begins are unrelated to the current ones (None).  With
    ``per_thread_loops`` the barrier pass also counts the per-thread loops
    around the body as iterations (``DeviceLoopState.per_thread_loops``):
    the next iteration may then be the next lane or vector element of the
    same tile, whose begins have not moved, so they are None too.
    """
    single = len(state.block_ids) == 1 and not per_thread_loops
    return {
        tile_begin_symbol(block_id, state): (
            block_size_symbol(state.strategy.fn.block_size_var(block_id))
            if single
            else None
        )
        for block_id in state.block_ids
    }


def check_lane_loop_nest(state: DeviceLoopState) -> None:
    """Make the lane loop nest around the body the tile program, or reject it.

    Every statement of the body runs inside every lane loop, so one whose
    values ignore a loop repeats once per iteration.  The full-nest check
    of the lane-loop distribution accepts the nest when every repetition
    is idempotent and pins an atomic that is uniform along a loop's tile
    axis to the loop's first lane (``cute/lane_loops/placement.py``);
    the body's carries count as the variables the phis after the loop
    make them (``carried_names``).  An inexact nest (a carry beside the
    per-K-lane products of the scalar matmul fallback updated once per
    lane) is redistributed like a grid body, each statement inside only
    the loops it depends on, or rejected.  Only the live loops are
    checked: a loop none of whose names the body reads (a tile indexed
    only by its ``begin``) is spliced away by the dead-code elimination
    and repeats nothing, as in ``DeviceGridState.wrap_body``, and a pin
    reading its lane variable would keep it alive.  Then, as for a grid
    body, block-wide barriers separate the accesses of one tensor that
    threads differing along a shared thread axis (this loop's or an
    enclosing loop's) could reorder (``add_thread_barriers``), and the
    body's accesses are checked across the loop's iterations too, a
    per-thread store of one iteration against a loop-invariant access of
    the next.
    """
    setup = {id(statement) for statement in state.lane_setup_statements}
    body = [
        statement for statement in state.inner_statements if id(statement) not in setup
    ]
    needed, kept_setup, spliced_wrappers = _live_lane_setup(
        body,
        state.lane_setup_statements,
        state.vec_lane_wrappers,
        semantic_reads=required_lane_reduce_owners(body),
    )
    live_loops = [
        (lane_var, extent)
        for lane_var, extent in state.lane_loops
        if _lane_loop_is_live(
            lane_var, state.vec_lane_wrappers.get(lane_var), needed, spliced_wrappers
        )
    ]
    coordinates = {
        lane_var: _cute_lane_coordinates(
            lane_var, state.vec_lane_wrappers.get(lane_var)
        )
        for lane_var, _extent in live_loops
    }
    setup_by_lane, _ = _assign_lane_setup(live_loops, coordinates, kept_setup)
    scopes = _cute_lane_scopes(
        live_loops,
        setup_by_lane,
        coordinates,
        {
            lane_var: _vec_wrapper_statements(state.vec_lane_wrappers.get(lane_var))
            for lane_var, _extent in live_loops
        },
        state.vec_lane_wrappers,
        state.tile_masks,
    )
    rename_groups = _current_rename_groups()
    # The repetition checks read a carry as the one variable the phi
    # after the loop makes of it.
    carried_groups = _merge_carried_names(rename_groups, state.carried_names)
    placement: list[ast.AST | LanePlacement] | None = None
    if live_loops:
        reason = full_nest_inexactness(body, scopes, rename_groups=carried_groups)
        if reason is not None:
            # A statement whose values ignore a loop repeats once per
            # iteration of the nest built around the body (a carry beside
            # the per-K-lane products of the scalar matmul fallback).
            # Place each statement inside only the loops it depends on, as
            # for a grid body (``DeviceGridState.wrap_body``), or reject.
            placement = distribute_lane_loops(
                body, scopes, rename_groups=carried_groups
            )
            if placement is None:
                raise inexact_nest_error(reason)
    # The nest as emitted: the body inside every live loop, or the placement.
    items: list[ast.AST | LanePlacement] = list(body)
    loops = [LanePlacement(lane_var, items) for lane_var, _extent in live_loops]
    for outer, inner in itertools.pairwise(loops):
        outer.items = [inner]
    nest: list[ast.AST | LanePlacement] = [loops[0]] if loops else items
    loop_body = items
    if placement is not None:
        nest = loop_body = placement
    axis_sizes, definitions, masks, constants, enclosing = _thread_barrier_context(
        state
    )
    # The per-thread loops between this loop and its body that the lane
    # nest does not hold (a looped reduction's lane loop and V-loop) run
    # inside each of its iterations: the pass takes them as further
    # iterations, in program order, and the statements they run before the
    # body (a lane's base index) as definitions reaching it.  Without them
    # the base index is a value of unknown origin, the same on every thread,
    # and a store through it looks like one element every thread writes.
    # Their iterations are lanes of the tile statements, not iterations of
    # the program: a pair that may meet in two of them is rejected
    # (``lane_variables``), where a barrier would order only other threads.
    # Their packets, loaded before the body and flushed after it, are
    # accesses of every element of a lane's vector (``packets_before``,
    # ``packets_after``).
    inner_loops, inner_variables, inner_definitions, inner_tails = (
        state.per_thread_loops()
    )
    packet_calls: dict[int, tuple[str, str]] = {}
    for loop in inner_loops:
        packet_calls.update(_packets(loop))

    def with_packets(statements: list[ast.AST]) -> list[ast.AST]:
        return [
            statement
            for statement in statements
            if any(id(call) in packet_calls for call in ast.walk(statement))
        ]

    add_thread_barriers(
        nest,
        scopes,
        rename_groups=rename_groups,
        axis_sizes=axis_sizes,
        definitions=[*definitions, *inner_definitions],
        masks=masks,
        constants=constants,
        # A branch condition may vary per thread; a block-wide barrier in
        # a branch some threads skip would deadlock.
        allow_barriers=not _in_divergent_control_flow(),
        loop_variables=state.loop_variables() | inner_variables,
        loop_body=loop_body,
        loop_shifts=tile_begin_shifts(state, per_thread_loops=bool(inner_loops)),
        # The loops around the body: those around this loop, then its own
        # (the per-thread loops among them iterate inside its iterations).
        loop_headers=[*enclosing, *state.loop_headers(), *inner_loops],
        lane_variables=inner_variables,
        packets_before=with_packets(inner_definitions),
        packets_after=with_packets(inner_tails),
        packet_calls=packet_calls,
        # A looped reduction's iterations are chunks of one tile statement,
        # which the program runs over every chunk before the next statement;
        # its tile begins move by a block per chunk.
        chunk_shifts=(
            tile_begin_shifts(state, per_thread_loops=False)
            if isinstance(state.strategy, LoopedReductionStrategy)
            else None
        ),
    )
    if placement is not None:
        _emit_lane_placement(state, placement, setup_by_lane)
        return
    for loop in loops:
        _apply_wrapper_barriers(
            state.vec_lane_wrappers.get(loop.lane_var), loop.barriers
        )
    # The innermost list holds the body statements and the barriers added
    # among them, never a loop.
    statements = [item for item in items if isinstance(item, ast.AST)]
    assert len(statements) == len(items)
    # The dead loops' coordinates and the setup computed from them leave
    # the body with the loops (``_splice_lane_loops``).
    live = dict(live_loops)
    dead = {lane_var for lane_var, _extent in state.lane_loops if lane_var not in live}
    dead_names: set[str] = set()
    for lane_var in dead:
        dead_names |= _cute_lane_coordinates(
            lane_var, state.vec_lane_wrappers.get(lane_var)
        )
    kept: list[ast.AST] = []
    for statement in state.lane_setup_statements:
        rw = ReadWrites.from_ast(statement)
        if set(rw.reads) & dead_names:
            dead_names |= set(rw.writes)
        else:
            kept.append(statement)
    kept_ids = {id(statement) for statement in kept}
    state.inner_statements[:] = [
        *(
            statement
            for statement in state.inner_statements
            if id(statement) in kept_ids
        ),
        *statements,
    ]
    _splice_lane_loops(state, dead)


def _emit_lane_placement(
    state: DeviceLoopState,
    placement: list[ast.AST | LanePlacement],
    setup_by_lane: dict[str, list[ast.AST]],
) -> None:
    """Replace the lane loop nest around the body with ``placement``.

    Each loop instance reuses its loop of the nest (for a vec wrapper, the
    outer loop with its attached statements, the instance in its V-loop)
    with the loop's setup at the top.  The dead loops leave with the
    nest, and the statements outside every loop join the body of the
    innermost device loop, which becomes ``inner_statements``.
    """

    def enclosing_child(loop: ast.For) -> ast.For:
        return next(
            child
            for child in loop.body
            if isinstance(child, ast.For) and _encloses(child, state.inner_statements)
        )

    parent = state.for_node
    nest = enclosing_child(parent)
    while getattr(nest, HELION_LANE_LOOP_VAR_ATTR, None) is None:
        parent, nest = nest, enclosing_child(nest)
    lane_loops: dict[str, ast.For] = {}
    loop = nest
    while True:
        if (lane_var := getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None)) is not None:
            lane_loops[lane_var] = loop
        if loop.body is state.inner_statements:
            break
        loop = enclosing_child(loop)

    def materialize(items: list[ast.AST | LanePlacement]) -> list[ast.AST]:
        result: list[ast.AST] = []
        for item in items:
            if not isinstance(item, LanePlacement):
                result.append(item)
                continue
            wrapper = state.vec_lane_wrappers.get(item.lane_var)
            _apply_wrapper_barriers(wrapper, item.barriers)
            inner = [*setup_by_lane[item.lane_var], *materialize(item.items)]
            if wrapper is None:
                lane_loops[item.lane_var].body = inner  # type: ignore[assignment]
            else:
                wrapper.vloop.body = inner  # type: ignore[assignment]
            result.append(lane_loops[item.lane_var])
        return result

    position = parent.body.index(nest)
    parent.body[position : position + 1] = materialize(placement)  # type: ignore[assignment]
    state.inner_statements = cast("list[ast.AST]", parent.body)


def _splice_lane_loops(state: DeviceLoopState, lane_vars: set[str]) -> None:
    """Take the lane loops of ``lane_vars`` out of the nest around the body, their bodies in their place.

    The nest is built around the body before the body exists
    (``codegen_device_loop``): a loop none of whose coordinates the body
    reads repeats the body for nothing, or wrongly (a read-modify-write
    nested in it runs once per lane), while ``check_lane_loop_nest``
    described the nest without it; the emitted nest is the checked one.
    A vectorized loop goes with its per-thread base and V-loop.  The
    setup computed from the loops' coordinates is out of the body
    already.
    """
    loop = state.for_node
    while loop.body is not state.inner_statements:
        child = next(
            child
            for child in loop.body
            if isinstance(child, ast.For) and _encloses(child, state.inner_statements)
        )
        lane_var = getattr(child, HELION_LANE_LOOP_VAR_ATTR, None)
        if lane_var is None or lane_var not in lane_vars:
            loop = child
            continue
        wrapper = state.vec_lane_wrappers.get(lane_var)
        inner = child.body if wrapper is None else wrapper.vloop.body
        position = loop.body.index(child)
        loop.body[position : position + 1] = inner
        if inner is state.inner_statements:
            state.inner_statements = cast("list[ast.AST]", loop.body)


def _thread_barrier_context(
    state: DeviceLoopOrGridState,
) -> tuple[
    dict[int, int], list[ast.AST], frozenset[str], dict[str, int], list[ast.For]
]:
    """What ``add_thread_barriers`` needs to know about the threads running ``state``'s body.

    The thread count along every launch axis (the body runs on every thread
    of the launch, those a loop nested in it addresses included), the
    per-thread index definitions of this loop and of every enclosing one
    (a device loop's body addresses the grid's axes too), the tile masks
    among them, the values of the block-size constants (the bounds of the
    loops inside the body name them) and the loops around this loop: the
    enclosing device loops, their per-thread loops and the enclosing loops'
    lane loops (a lane nest is materialized after its body, so a loop
    nested in the body sees the lane variables as names), whose variables
    are one value each throughout the body, nonnegative and below their
    bounds (``AddressMaps`` models the loops around a body).
    """
    axis_sizes = dict(state.thread_axis_sizes)
    own, masks = state.lane_index_definitions()
    constants: dict[str, int] = {}
    enclosing_loops: list[ast.For] = []
    try:
        fn = DeviceFunction.current()
    except NoCurrentFunction:
        # Unit tests build the loop states outside a device function.
        return axis_sizes, own, masks, constants, enclosing_loops
    env = CompileEnvironment.current()
    for key, name in fn.block_size_var_cache.items():
        if len(key) == 1:
            value = env.block_sizes[key[0]].from_config(fn.config)
            if isinstance(value, int):
                constants[name] = value
    codegen = fn.codegen
    for axis, size in codegen.launch_thread_axis_sizes().items():
        axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
    # In program order, for the address maps to resolve each name to the
    # definition reaching the body: whatever the enclosing scopes defined so
    # far (a flattened tile's per-thread base, a coordinate a strategy
    # emitted straight into the kernel, an earlier loop over the same block
    # with its own index) may carry a thread coordinate into the body too,
    # then the enclosing loops' index definitions, then this loop's own.
    # The list at the bottom of the stack is the host wrapper's: what it
    # defines (a dynamic size unpacked from a shape) reaches the kernel as
    # an argument, one uniform value, not as a definition the body reads.
    definitions: list[ast.AST] = []
    for statements in codegen.statements_stack[1:]:
        for statement in statements:
            # The loops enclosing the body (this one among them, already in
            # its parent's list) are the body's structure, not definitions
            # reaching it.
            if (
                isinstance(state, DeviceLoopState)
                and isinstance(statement, ast.For)
                and _encloses(statement, state.inner_statements)
            ):
                continue
            definitions.append(statement)
    seen = {id(state)}
    for loops in codegen.active_device_loops.values():
        for enclosing in loops:
            if id(enclosing) not in seen:
                seen.add(id(enclosing))
                enclosing_definitions, enclosing_masks = (
                    enclosing.lane_index_definitions()
                )
                definitions.extend(enclosing_definitions)
                masks |= enclosing_masks
                # An enclosing loop's ``for`` statements join their parent's
                # list once its body is complete, so the loops around this
                # one are read from their states: the device loop's own
                # headers, its per-thread loops (a looped reduction's lane
                # loop and V-loop, with what they run before its body: the
                # lane's base index) and the lane loops it nests around its
                # body.
                if isinstance(enclosing, DeviceLoopState):
                    enclosing_loops.extend(enclosing.loop_headers())
                    inner_loops, _variables, inner_definitions, _tails = (
                        enclosing.per_thread_loops()
                    )
                    enclosing_loops.extend(inner_loops)
                    definitions.extend(inner_definitions)
                enclosing_loops.extend(enclosing.lane_loop_headers())
    definitions.extend(own)
    return axis_sizes, definitions, masks, constants, enclosing_loops


def _in_divergent_control_flow() -> bool:
    """Whether the statements being generated sit in a branch or while loop
    (``GenerateAST.divergent_control_flow``), whose condition a CuTe SIMT
    thread may evaluate differently from its neighbours."""
    try:
        return DeviceFunction.current().codegen.divergent_control_flow_depth > 0
    except NoCurrentFunction:
        return False


def add_lane_nest_barriers(
    state: DeviceGridState,
    placement: list[ast.AST | LanePlacement],
    scopes: Sequence[LaneScope],
    rename_groups: Mapping[str, str],
) -> None:
    """Order across threads the accesses of a root body placed around its lane loops.

    The barriers separate the accesses of one tensor that threads differing
    along a shared thread axis could reorder (``add_thread_barriers``, in
    place on ``placement``).
    """
    axis_sizes, definitions, masks, constants, enclosing = _thread_barrier_context(
        state
    )
    add_thread_barriers(
        placement,
        scopes,
        rename_groups=rename_groups,
        axis_sizes=axis_sizes,
        definitions=definitions,
        masks=masks,
        constants=constants,
        loop_headers=enclosing,
    )
