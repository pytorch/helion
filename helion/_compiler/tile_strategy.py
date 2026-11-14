from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import itertools
import math
import operator
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeVar
from typing import cast
import weakref

import sympy
import torch

from .. import exc

# The ``.cute.lane_loops.markers`` imports (``_ascending_lane_loop_form``,
# ``_create_lane_loop``, ``_encloses``, ``_is_constexpr_lane_iter``,
# ``_lane_loop_extent``, ``_reverse_lane_loop_iter``) are the lane-loop forms
# of the per-thread strategies CuTe and Metal share; they belong to a
# backend-neutral module and leave with the layout move.
from .._compat import shape_env_size_hint
from .._utils import indexing_uses_tensor_descriptor
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from .ast_read_writes import HELION_VEC_LANE_OF_ATTR
from .compile_environment import CompileEnvironment
from .compile_environment import _has_unbacked
from .compile_environment import _to_sympy
from .cute.lane_loops.markers import _ascending_lane_loop_form
from .cute.lane_loops.markers import _create_lane_loop
from .cute.lane_loops.markers import _encloses
from .cute.lane_loops.markers import _is_constexpr_lane_iter
from .cute.lane_loops.markers import _lane_loop_extent
from .cute.lane_loops.markers import _reverse_lane_loop_iter
from .device_function import DeviceFunction
from .device_function import TensorArg
from .host_function import HostFunction
from .host_function import NoCurrentFunction
from .program_id import FlatProgramIDs
from .program_id import ForEachProgramID
from .program_id import L2GroupingProgramIDs
from .program_id import PersistentBlockedProgramIDs
from .program_id import PersistentInterleavedProgramIDs
from .program_id import PIDInfo
from .program_id import ProgramIDs
from .program_id import XYZProgramIDs

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .cute.integer_loop_reduction import Hoist
    from .cute.lane_loops.statements import LanePlacement
    from .cute.lane_loops.statements import LaneScope
    from .cute.memory_ops import CuteLaneRelocation
    from .inductor_lowering import CodegenState
    from .lane_loop_context import LaneLoopContext
    from .pallas.dma import DmaResources

    _T = TypeVar("_T")
    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


# Set once the missing ``tl.range(data_partition_factor=...)`` was warned about.
_warned_data_partition_factor: list[bool] = []

_loop_instances = itertools.count()


def new_loop_instance() -> int:
    """A serial for a loop state, unique for the process: the access regions
    name a loop's tile begins by it (``DeviceLoopOrGridState.region_instance``)."""
    return next(_loop_instances)


class ThreadAxisTracker:
    """Tracks thread axis assignments for block dimensions during codegen."""

    __slots__ = ("sizes", "block_axes")

    def __init__(self) -> None:
        self.sizes: dict[int, int] = {}
        self.block_axes: dict[int, int] = {}

    def record(self, block_idx: int, axis: int, size: int) -> None:
        """Record a thread axis mapping for a single block dimension."""
        self.sizes[axis] = max(self.sizes.get(axis, 1), size)
        self.block_axes[block_idx] = axis

    def record_all(self, block_ids: list[int], axis: int, size: int) -> None:
        """Record the same thread axis mapping for all block dimensions."""
        self.sizes[axis] = size
        for block_id in block_ids:
            self.block_axes[block_id] = axis

    def record_symbolic_axis(self, block_ids: Iterable[int], axis: int) -> None:
        """Record the thread axis of blocks whose extent is only known at launch.

        An argument-sized ``hl.tile`` (``block_size=bsz`` with ``bsz`` an int
        argument, ``static_shapes=False``) lives one element per thread on an
        axis the static shapes hold a one for, so no size is recorded; the
        axis itself still names the leader thread of a tile-uniform atomic
        (``cute.arch.thread_idx()[axis] == 0``), which ran once per thread of
        the axis while the block was missing here.  Only a backend planning
        the launch's thread layout from the recorded axes records it.
        """
        if not CompileEnvironment.current().backend.plans_launch_thread_axes():
            return
        for block_id in block_ids:
            self.block_axes[block_id] = axis


@dataclasses.dataclass(frozen=True)
class CuteLaneAxis:
    """How a CuTe per-thread strategy distributes one block axis of a tile.

    ``threads`` CUDA threads split the ``extent`` elements of the axis; each
    thread walks its remaining elements in ``lane_steps`` iterations of the
    ``lane_var`` lane loop, optionally as ``vec_width``-wide vectors whose
    elements the constexpr ``vec_lane_var`` loop (``vloop``) enumerates.
    ``strided`` lane steps are ``threads``-wide chunks of consecutive elements
    (``index = thread + step * threads``); blocked steps give every thread one
    contiguous slice.  Returned by the strategies' ``cute_lane_axis`` so
    consumers such as the scan lowering need not read strategy internals.
    """

    extent: int
    threads: int
    lane_var: str | None
    lane_steps: int
    vec_lane_var: str | None
    vec_width: int
    strided: bool


@dataclasses.dataclass
class LoopDimInfo:
    begin_var_name: str | None = None
    begin_expr: sympy.Expr | None = None
    end_var_name: str | None = None
    end_expr: sympy.Expr | None = None
    # Absolute end of a root-grid dim; None for device loops.
    grid_end_expr: sympy.Expr | None = None
    # True when the generated extent mask checks both the logical begin and end.
    mask_has_lower_bound: bool = False

    def is_end_matching(self, size: int | torch.SymInt) -> bool:
        expected = _to_sympy(size)
        if expected == self.end_expr:
            return True
        if (
            self.end_expr is None
            or _has_unbacked(self.end_expr)
            or _has_unbacked(expected)
        ):
            return False
        shape_env = CompileEnvironment.current().shape_env
        # TODO(jansel): current check is based on size hints, may need to guard here in the future
        return shape_env_size_hint(shape_env, expected) == shape_env_size_hint(
            shape_env, self.end_expr
        )

    def is_end_matching_keyed(self, size: int | torch.SymInt) -> bool:
        """``is_end_matching`` for every input the bound kernel serves: input
        sizes that match only at these values key the bound kernel on their
        equality (``CompileEnvironment.keyed_size_equalities``)."""
        expected = _to_sympy(size)
        if expected == self.end_expr:
            return True
        if (
            self.end_expr is None
            or _has_unbacked(self.end_expr)
            or _has_unbacked(expected)
        ):
            return False
        env = CompileEnvironment.current()
        # Specialized sizes are keyed by their values already.
        expected = env.specialize_expr(expected)
        end = env.specialize_expr(self.end_expr)
        if expected == end:
            return True
        static = env.shape_env._maybe_evaluate_static(sympy.Eq(expected, end))
        if static is not None:
            return bool(static)
        sources = env.shape_env.var_to_sources
        if (
            not isinstance(expected, sympy.Symbol)
            or not all(
                sources.get(symbol)
                for symbol in end.free_symbols
                if isinstance(symbol, sympy.Symbol)
            )
            or not sources.get(expected)
            or shape_env_size_hint(env.shape_env, expected)
            != shape_env_size_hint(env.shape_env, end)
        ):
            return False
        env.keyed_size_equalities.add((expected, end))
        return True


@dataclasses.dataclass
class DeviceLoopOrGridState:
    strategy: TileStrategy
    block_id_to_info: dict[int, LoopDimInfo]
    thread_axis_sizes: dict[int, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )
    block_thread_axes: dict[int, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )
    # The serial naming this state's tile begins in the access regions
    # (``cute/access_regions.py``).
    region_instance: int = dataclasses.field(
        default_factory=new_loop_instance, kw_only=True
    )

    @property
    def block_ids(self) -> list[int]:
        return self.strategy.block_ids

    def lane_index_definitions(self) -> tuple[list[ast.AST], frozenset[str]]:
        """The per-thread index and mask definitions this loop emits around its
        body, and the tile masks among them (``add_thread_barriers`` reads the
        thread axes through them)."""
        return [], frozenset()

    def lane_loop_headers(self) -> list[ast.For]:
        """The lane loops this loop nests around its body, as ``for`` statements
        for the barrier pass of a loop nested in the body
        (``cute/lane_loops/nest_check.py``)."""
        return []


@dataclasses.dataclass
class DeviceLoopState(DeviceLoopOrGridState):
    integer_reduction_hoist: Hoist | None = dataclasses.field(default=None, init=False)
    for_node: ast.For
    inner_statements: list[ast.AST]
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    # Block ids that this device loop distributes across a per-thread lane
    # loop (CuTe only). A reduction over one of these blocks needs the
    # two-pass lane structure (see ``split_lane_loop_reductions``).
    lane_loop_blocks: set[int] = dataclasses.field(default_factory=set)
    # The lane loops themselves (CuTe per-thread strategies), outermost first:
    # the blocks each distributes, the vec partitions among them and the
    # per-lane index / mask definitions at the top of the innermost body.
    # Unlike a grid's (``DeviceGridState.wrap_body``) the nest is built
    # around the body before the body exists; ``check_lane_loop_nest`` keeps
    # it when it is the tile program and redistributes or rejects it otherwise.
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_loop_block_ids: dict[str, frozenset[int]] = dataclasses.field(
        default_factory=dict
    )
    vec_lane_wrappers: dict[str, VecLaneWrapper] = dataclasses.field(
        default_factory=dict
    )
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    # The tile masks among those definitions (``LaneScope.masks``).
    tile_masks: frozenset[str] = frozenset()
    # Run once the loop body is complete (CuTe: restore the tile-vector stores
    # whose deferred flush a later access of their tensor would observe).
    body_finalizers: list[Callable[[], None]] = dataclasses.field(default_factory=list)
    # Each carried output of the body mapped to its initial value, the two
    # names the phi after the loop merges into one variable (set once the
    # body is complete, before ``check_lane_loop_nest``).
    carried_names: dict[str, str] = dataclasses.field(default_factory=dict)

    def lane_index_definitions(self) -> tuple[list[ast.AST], frozenset[str]]:
        return _lane_index_definitions(
            self.outer_prefix, self.lane_setup_statements, self.vec_lane_wrappers
        ), self.tile_masks

    def lane_loop_headers(self) -> list[ast.For]:
        return _lane_loop_headers(self.lane_loops, self.vec_lane_wrappers)

    def loop_headers(self) -> list[ast.For]:
        """The device loops around ``inner_statements``, outermost first.

        The ``for`` statements on the way from ``for_node`` down to the body
        (a multi-dimensional tile nests one loop per block); the synthetic
        lane loops and constexpr vector loops among them are per-thread
        structure, not iteration.
        """
        loops: list[ast.For] = []
        loop: ast.For | None = self.for_node
        while loop is not None:
            if getattr(
                loop, HELION_LANE_LOOP_VAR_ATTR, None
            ) is None and not _is_constexpr_lane_iter(loop):
                loops.append(loop)
            if loop.body is self.inner_statements:
                break
            loop = next(
                (
                    child
                    for child in loop.body
                    if isinstance(child, ast.For)
                    and _encloses(child, self.inner_statements)
                ),
                None,
            )
        return loops

    def loop_variables(self) -> frozenset[str]:
        """The names the device loops around ``inner_statements`` redefine per iteration."""
        return frozenset(
            node.id
            for loop in self.loop_headers()
            for node in ast.walk(loop.target)
            if isinstance(node, ast.Name)
        )

    def check_lane_loop_nest(self) -> None:
        """Make the lane loop nest around the body the tile program, or reject
        it, once the body is complete (``Backend.check_lane_loop_nest``: the
        nest is built around the body before the body exists)."""
        CompileEnvironment.current().backend.check_lane_loop_nest(self)


@dataclasses.dataclass
class EmitPipelineLoopState(DeviceLoopOrGridState):
    """State for emit_pipeline-based loops on TPU (Pallas backend)."""

    body_fn_name: str
    body_fn_def: ast.FunctionDef | None = None
    inner_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    pipeline_call: ast.AST | None = None
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    _tensor_to_dma_scratch: dict[str, str] = dataclasses.field(default_factory=dict)
    # Per downstream _mask_to node and block id, physical tensor bounds needed
    # to zero lanes left stale by a shortened input DMA.
    _deferred_physical_mask_bounds: dict[torch.fx.Node, dict[int, tuple[str, ...]]] = (
        dataclasses.field(default_factory=dict)
    )
    # Clean-region branches prove these physical bounds all-true for this tile.
    _proven_physical_mask_block_ids: set[int] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class ForiLoopState(DeviceLoopOrGridState):
    """State for fori_loop-based loops on TPU (Pallas backend).

    Uses jax.lax.fori_loop with pltpu.make_async_copy for tensors whose
    inner-block shape passes ``_check_dma_alignment``; tensors that fail
    are kept on their outer BlockSpec and accessed via ``pl.ds`` from the
    body. Per-tensor pipelining membership lives in
    ``_tensor_to_dma_scratch``; input tensors with an overlapped prefetch are
    recorded in ``_prefetched_load_tensors``.
    """

    body_fn_name: str
    loop_var_name: str  # The fori_loop index variable (e.g., "_j")
    iteration_count: str | None = None
    static_unroll: bool = False
    inner_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    _tensor_to_dma_scratch: dict[str, str] = dataclasses.field(default_factory=dict)
    _tensor_to_sem: dict[str, str] = dataclasses.field(default_factory=dict)
    _prefetched_load_tensors: set[str] = dataclasses.field(default_factory=set)
    _memory_op_to_dma_scratch: dict[torch.fx.Node, DmaResources] = dataclasses.field(
        default_factory=dict
    )


def _cute_epilogue_subtile_active(config: object) -> bool:
    """True when the config requests epilogue subtiling (>= 2)."""
    value = getattr(config, "config", {}).get("epilogue_subtile")
    return isinstance(value, int) and value > 1


@dataclasses.dataclass
class VecLaneWrapper:
    """Pre-built outer x constexpr-V lane structure for a GRID lane loop.

    Grid lane loops are materialized late (``DeviceGridState.wrap_body``),
    but the vec-load/store hoist protocol needs the outer-lane body list and
    the constexpr V-loop node to exist DURING body codegen (memory_ops
    splices hoisted ``cute.arch.load(..., V)`` statements into it).  So
    ``PerThreadNDTileStrategy.codegen_grid`` pre-builds the structure and
    ``wrap_body`` just plugs the user body into ``vloop.body``.
    """

    outer_for: ast.For
    vloop: ast.For
    vec_lane_var: str
    base_index_var: str
    # Persistent reductions whose complete per-thread slice is one vector do
    # not need a separate outer scalar-lane loop.  Keep ``outer_for`` as the
    # mutable container used by the load/store hoist protocol, but splice its
    # body directly into the enclosing scope when this flag is set.
    elide_outer_loop: bool = False

    def __post_init__(self) -> None:
        # The lane reduction split tells this V-loop from any other constexpr
        # loop in the lane body by the lane variable it is nested in.
        lane_target = self.outer_for.target
        assert isinstance(lane_target, ast.Name)
        setattr(self.vloop, HELION_VEC_LANE_OF_ATTR, lane_target.id)


def _cute_lane_coordinates(lane_var: str, wrapper: VecLaneWrapper | None) -> set[str]:
    """A lane loop's own coordinates: its lane variable and, when vectorized,
    its constexpr V-loop variable and per-thread lane base."""
    names = {lane_var}
    if wrapper is not None:
        names.update((wrapper.vec_lane_var, wrapper.base_index_var))
    return names


def _cute_first_lane_predicate(lane_var: str, wrapper: VecLaneWrapper | None) -> str:
    """The first iteration of ``lane_var``'s loop, and of its constexpr V-loop."""
    if wrapper is None or wrapper.vec_lane_var == lane_var:
        return f"{lane_var} == 0"
    return f"{lane_var} == 0 and {wrapper.vec_lane_var} == 0"


def _vec_wrapper_statements(wrapper: VecLaneWrapper | None) -> list[ast.AST]:
    """The statements a vec wrapper emits besides its V-loop: the per-thread
    lane base, the packet loads hoisted above the V-loop, the store flushes
    after it."""
    if wrapper is None:
        return []
    return [stmt for stmt in wrapper.outer_for.body if stmt is not wrapper.vloop]


def _live_lane_setup(
    body: list[ast.AST],
    setup: list[ast.AST],
    wrappers: dict[str, VecLaneWrapper],
    *,
    semantic_reads: Iterable[str] = (),
) -> tuple[set[str], list[ast.AST], set[str]]:
    """The names ``body`` needs, the lane setup statements defining them and
    the vec wrappers whose lane bodies received splices.

    Setup statements (per-lane index / mask definitions) whose results the
    body never reads are dropped, and a lane loop whose variable is then left
    unreferenced is dead (``_lane_loop_is_live``).  An unused rdim block
    (e.g. one allocated for a ``tile.index`` that codegen serves from the
    tile's own block) would otherwise wrap the whole body in a dead innermost
    loop, and the lane-reduce split pass would then try to split reductions
    on that loop's lane var instead of the loop that actually distributes
    them.
    """
    from .ast_read_writes import ReadWrites

    # A reduction's owner is a semantic dependency even when its value does
    # not read the lane coordinate (for example, sum over an expanded axis).
    # Proved reshape aliases remain prunable by the owner resolver.
    needed = set(semantic_reads)
    for stmt in body:
        needed |= set(ReadWrites.from_ast(stmt).reads)
    # A vec wrapper whose lane body received memory_ops splices (hoisted vec
    # loads before the V-loop, store flushes after it) must be kept
    # regardless of what the inner body reads: the flush IS the store.  Its
    # splices also read setup vars (masks, sibling index vars), so fold those
    # reads in before selecting the kept setup.
    spliced_wrappers: set[str] = set()
    for lane_var, wrapper in wrappers.items():
        if len(wrapper.outer_for.body) > 2:  # more than [base, vloop]
            spliced_wrappers.add(lane_var)
            needed |= set(ReadWrites.from_ast(wrapper.outer_for).reads)
    kept_setup: list[ast.AST] = []
    for stmt in reversed(setup):
        rw = ReadWrites.from_ast(stmt)
        if not rw.writes or set(rw.writes) & needed:
            kept_setup.append(stmt)
            needed |= set(rw.reads)
    kept_setup.reverse()
    return needed, kept_setup, spliced_wrappers


def _lane_loop_is_live(
    lane_var: str,
    wrapper: VecLaneWrapper | None,
    needed: set[str],
    spliced_wrappers: set[str],
) -> bool:
    """Whether the lane loop of ``lane_var`` survives the dead-code elimination."""
    if wrapper is None:
        return lane_var in needed
    # The per-element index var reads ``base_index_var`` + ``vec_lane_var``
    # (not ``lane_var`` directly), so test all three before dropping the
    # structure as dead.
    return lane_var in spliced_wrappers or bool(
        {lane_var, wrapper.vec_lane_var, wrapper.base_index_var} & needed
    )


def _current_rename_groups() -> dict[str, str]:
    """Every alias the device function will rename, mapped to its canonical name.

    A tensor argument viewing another's storage (``alias = x.view(...)`` on
    the host) is mapped to the first such argument too: the lane-loop and
    thread-barrier passes key memory dependences on these names, and a store
    through one view must order a load through the other.
    """
    try:
        fn = DeviceFunction.current()
    except NoCurrentFunction:
        # Unit tests build the loop states outside a device function.
        return {}
    groups = {name: aliases[0] for name, aliases in fn._variable_renames.items()}
    storage_owners: dict[int, str] = {}
    for arg in fn.arguments:
        if isinstance(arg, TensorArg):
            owner = storage_owners.setdefault(
                id(arg.fake_value.untyped_storage()), arg.name
            )
            if owner != arg.name:
                groups.setdefault(arg.name, owner)
    return groups


def _merge_carried_names(
    rename_groups: Mapping[str, str], carried_names: Mapping[str, str]
) -> dict[str, str]:
    """``rename_groups`` with each carried output and its initial value
    mapped to one canonical name.

    The phi after a device loop merges the two only once the loop statement
    is complete; the repetition checks of its body need them as one variable
    already (``DeviceLoopState.carried_names``).
    """
    groups = dict(rename_groups)
    for output, initial in carried_names.items():
        old = groups.get(output, output)
        new = groups.get(initial, initial)
        if old == new:
            continue
        for name, canonical in list(groups.items()):
            if canonical == old:
                groups[name] = new
        groups[old] = new
        groups[output] = new
    return groups


def _assign_lane_setup(
    lane_loops: list[tuple[str, int]],
    coordinates: dict[str, set[str]],
    setup: list[ast.AST],
) -> tuple[dict[str, list[ast.AST]], list[ast.AST]]:
    """Place each setup statement at the shallowest lane scope that defines
    every lane / base variable it reads.

    Historically all setup statements were placed in the innermost lane loop.
    That is semantically correct for scalar loops, but strands an outer-axis
    index definition *inside* an inner persistent vec loop while that vec
    loop's hoisted load needs the index outside its constexpr V loop.
    Explicit scope placement keeps outer tile coordinates available to nested
    reduction vec hoists and avoids redundantly recomputing them for every
    inner lane.  A statement reading no lane coordinate keeps the old
    innermost placement rather than speculatively widening its scope (a later
    setup statement reading its value must remain at least as deep), or is
    returned separately when there are no lane loops.
    """
    from .ast_read_writes import ReadWrites

    setup_by_lane: dict[str, list[ast.AST]] = {
        lane_var: [] for lane_var, _extent in lane_loops
    }
    fallback: list[ast.AST] = []
    setup_name_scopes: dict[str, int] = {}
    for stmt in setup:
        rw = ReadWrites.from_ast(stmt)
        reads = set(rw.reads)
        matching_depths = [
            depth
            for depth, (lane_var, _extent) in enumerate(lane_loops)
            if reads & coordinates[lane_var]
        ]
        matching_depths.extend(
            setup_name_scopes[name] for name in reads if name in setup_name_scopes
        )
        if matching_depths:
            # A statement depending on multiple lane coordinates belongs to
            # the deepest of those nested scopes.
            depth = max(matching_depths)
        elif lane_loops:
            depth = len(lane_loops) - 1
        else:
            fallback.append(stmt)
            continue
        setup_by_lane[lane_loops[depth][0]].append(stmt)
        for name in rw.writes:
            setup_name_scopes[name] = depth
    return setup_by_lane, fallback


def _cute_lane_scopes(
    live_loops: list[tuple[str, int]],
    setup_by_lane: dict[str, list[ast.AST]],
    coordinates: dict[str, set[str]],
    attached: dict[str, list[ast.AST]],
    wrappers: dict[str, VecLaneWrapper],
    masks: frozenset[str],
) -> list[LaneScope]:
    """The ``LaneScope`` of every live lane loop, outermost first, for the
    lane-loop distribution (``cute/lane_loops/``).

    A loop's names are its coordinates and the definitions of its setup and
    attached statements; it requires the loops whose names its setup reads;
    its masks are the tile masks its setup defines; its counts are its trip
    count and its constexpr V-loop's, by variable.
    """
    from .ast_read_writes import ReadWrites
    from .cute.lane_loops.statements import LaneScope

    def counts(lane_var: str, extent: int) -> dict[str, int]:
        wrapper = wrappers.get(lane_var)
        if wrapper is None:
            # A plain loop runs ``range(extent)``.
            return {lane_var: extent}
        # A vec wrapper's outer loop is pre-built (``extent`` counts the
        # thread's elements, not its iterations); its V-loop is constexpr.
        result: dict[str, int] = {}
        if wrapper.elide_outer_loop:
            result[lane_var] = 1
        elif (form := _ascending_lane_loop_form(wrapper.outer_for)) is not None:
            result[lane_var] = form[1]
        if (form := _ascending_lane_loop_form(wrapper.vloop)) is not None:
            result[wrapper.vec_lane_var] = form[1]
        return result

    setup_writes = {
        lane_var: frozenset().union(
            *(ReadWrites.from_ast(stmt).writes for stmt in setup_by_lane[lane_var])
        )
        for lane_var, _extent in live_loops
    }
    scope_names = {
        lane_var: frozenset(coordinates[lane_var]).union(
            setup_writes[lane_var],
            *(ReadWrites.from_ast(stmt).writes for stmt in attached[lane_var]),
        )
        for lane_var, _extent in live_loops
    }
    return [
        LaneScope(
            lane_var,
            scope_names[lane_var],
            frozenset(
                other
                for other, names in scope_names.items()
                if other != lane_var
                and any(
                    set(ReadWrites.from_ast(stmt).reads) & names
                    for stmt in setup_by_lane[lane_var]
                )
            ),
            tuple(attached[lane_var]),
            frozenset(coordinates[lane_var]),
            tuple(setup_by_lane[lane_var]),
            _cute_first_lane_predicate(lane_var, wrappers.get(lane_var)),
            masks & setup_writes[lane_var],
            _vloop_index(wrappers.get(lane_var)),
            counts(lane_var, extent),
        )
        for lane_var, extent in live_loops
    ]


def _vloop_index(wrapper: VecLaneWrapper | None) -> int:
    """How many of a vec wrapper's statements precede its V-loop (``LaneScope.vloop_index``).

    All of them when the V-loop is not among them (a wrapper built by hand).
    """
    if wrapper is None:
        return 0
    return next(
        (
            index
            for index, statement in enumerate(wrapper.outer_for.body)
            if statement is wrapper.vloop
        ),
        len(wrapper.outer_for.body),
    )


def _apply_wrapper_barriers(
    wrapper: VecLaneWrapper | None, barriers: list[int]
) -> None:
    """Insert the barriers ``add_thread_barriers`` placed among a wrapper's statements."""
    if not barriers:
        return
    assert wrapper is not None
    for position in sorted(barriers, reverse=True):
        wrapper.outer_for.body.insert(
            position, statement_from_string("cute.arch.sync_threads()")
        )


def _lane_index_definitions(
    prefix: list[ast.AST],
    setup: list[ast.AST],
    wrappers: dict[str, VecLaneWrapper],
) -> list[ast.AST]:
    """A loop's per-thread index definitions: those hoisted before its lane
    loops, the lane setup and the vec wrappers' own statements."""
    return [
        *prefix,
        *setup,
        *(
            statement
            for wrapper in wrappers.values()
            for statement in _vec_wrapper_statements(wrapper)
        ),
    ]


def _lane_loop_headers(
    lane_loops: list[tuple[str, int]], wrappers: dict[str, VecLaneWrapper]
) -> list[ast.For]:
    """A loop's lane loops as ``for`` statements: ``for lane in range(extent)``
    for each, and the V-loop of each vectorized one.

    A lane nest is materialized after its body (``wrap_body``), so a loop
    nested in the body sees the lane variables as names; the headers say
    what the names are (one value per pass, nonnegative, below the extent)
    to the address maps of the nested loop's own barrier pass.
    """
    headers: list[ast.For] = []
    for lane_var, extent in lane_loops:
        (header,) = ast.parse(f"for {lane_var} in range({extent}):\n    pass\n").body
        assert isinstance(header, ast.For)
        headers.append(header)
    headers.extend(wrapper.vloop for wrapper in wrappers.values())
    return headers


@dataclasses.dataclass
class DeviceGridState(DeviceLoopOrGridState):
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_loop_blocks: set[int] = dataclasses.field(default_factory=set)
    # CuTe: the index variable of the chunk lane loop standing in for a
    # rolled reduction block outside its loop, per block id
    # (``memory_ops._cute_make_rolled_replica``).  Per grid, like the lane
    # loop itself: another root loop sharing the block gets its own.
    rolled_reduction_replicas: dict[int, str] = dataclasses.field(default_factory=dict)
    # Preserve which logical blocks each synthetic lane variable distributes.
    # The aggregate set above is retained for existing reduction scheduling;
    # late structural lowerings need the exact association to prove coordinate
    # coverage without guessing from generated variable names.
    lane_loop_block_ids: dict[str, frozenset[int]] = dataclasses.field(
        default_factory=dict
    )
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    # The tile masks among the lane setup's definitions (``LaneScope.masks``).
    tile_masks: frozenset[str] = frozenset()
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    # Statement list that will receive ``wrap_body(...)``.  Definitions
    # already emitted there dominate every synthetic lane wrapper; lists
    # pushed after it are branch/loop bodies that the wrapper will enclose.
    hoist_parent_statements: list[ast.AST] | None = None
    # lane_var -> pre-built vec partition (see VecLaneWrapper).  Only grid
    # lane loops whose block has ``cute_vector_widths[block] > 1`` (and a
    # divisible elements-per-thread) get an entry.
    vec_lane_wrappers: dict[str, VecLaneWrapper] = dataclasses.field(
        default_factory=dict
    )
    deferred_vector_ops: list[tuple[ast.For, ast.AST, Callable[[], ast.AST | None]]] = (
        dataclasses.field(default_factory=list)
    )
    # Lane vars whose loops must visit lanes in descending order because a
    # reverse ``hl.associative_scan`` carries its suffix across them (see
    # ``cute/scan_ops.py``).  Applied when ``wrap_body`` materializes the
    # loops; never names a lane with a ``vec_lane_wrappers`` entry.
    reversed_lane_vars: set[str] = dataclasses.field(default_factory=set)
    # Lane vars whose loops a later pass rewrites structurally (resident
    # reductions, register-tile reductions); ``wrap_body`` keeps the full
    # nest around them.
    undistributable_lane_vars: set[str] = dataclasses.field(default_factory=set)
    # Body definitions that packet loads hoisted into a vec wrapper read
    # (``cute/memory_ops.py`` records them while lowering the body).
    # ``wrap_body`` emits them before that lane loop or, when it keeps the
    # full nest, moves them into the wrapper above the V-loop.
    pending_relocations: list[CuteLaneRelocation] = dataclasses.field(
        default_factory=list
    )
    # Synthetic reduction lanes materialized as ``cutlass.range_constexpr``
    # loops OUTSIDE the one-vector tile wrappers (register-tile reductions,
    # see ``nest_reduction_lane_outside_vector_tiles``).
    constexpr_lane_vars: set[str] = dataclasses.field(default_factory=set)
    # The thread extents of the root tile's own axes when the grid became
    # active, before the body's reductions added theirs to
    # ``thread_axis_sizes`` (``GenerateAST.set_active_loops``).
    root_axis_sizes: dict[int, int] = dataclasses.field(default_factory=dict)

    def has_lane_loops(self) -> bool:
        return bool(self.lane_loops)

    def lane_defined_names(self) -> set[str]:
        """The names only the lane loops define: the lane variables, the vec
        lanes' own, and the per-lane index / mask definitions."""
        from .ast_read_writes import ReadWrites

        names = {lane_var for lane_var, _extent in self.lane_loops}
        for wrapper in self.vec_lane_wrappers.values():
            names.update((wrapper.vec_lane_var, wrapper.base_index_var))
        for stmt in self.lane_setup_statements:
            names.update(ReadWrites.from_ast(stmt).writes)
        return names

    def add_lane_loop(
        self,
        block_id: int,
        lane_var: str,
        extent: int,
        *,
        position: int | None = None,
    ) -> None:
        if position is None:
            self.lane_loops.append((lane_var, extent))
        else:
            self.lane_loops.insert(position, (lane_var, extent))
        self.lane_loop_blocks.add(block_id)
        self.lane_loop_block_ids[lane_var] = self.lane_loop_block_ids.get(
            lane_var, frozenset()
        ) | {block_id}

    def nest_reduction_lane_outside_vector_tiles(
        self,
        block_id: int,
        lane_var: str,
        extent: int,
        *,
        max_unrolled_elements: int,
    ) -> bool:
        """Register a synthetic reduction lane as a trace-time loop OUTSIDE
        every one-vector tile wrapper, so the tile V-loops become the innermost
        element loops (a per-thread register tile).

        With the default nesting the constexpr tile V-loop wraps the rolled
        reduction lane loop, so a load whose row coordinate comes from the
        reduction lane is defined below the V-loop and can never be hoisted as
        one vector transaction; every lane iteration also waits for its own
        load before the next lane's load issues.  Placing the reduction lane
        outside the wrappers makes the row coordinate available above the
        V-loop (one LDG.128 per lane) and, because both loops unroll at trace
        time, lets ``split_lane_loop_reductions`` schedule every lane's loads
        before the first store.

        Applies only when every tile lane loop of this grid is a vector
        wrapper whose outer loop has exactly one trip (each thread owns one
        V-wide fragment per tile axis) and the unrolled per-thread element
        count stays within ``max_unrolled_elements``; otherwise the caller
        keeps the established rolled nesting.  Returns whether the lane was
        registered.
        """
        if not self.lane_loops or extent <= 1:
            return False
        unrolled = extent
        for tile_lane_var, _tile_extent in self.lane_loops:
            wrapper = self.vec_lane_wrappers.get(tile_lane_var)
            if wrapper is None or wrapper.elide_outer_loop:
                return False
            if _lane_loop_extent(wrapper.outer_for) != 1:
                return False
            unrolled *= _lane_loop_extent(wrapper.vloop)
        if unrolled > max_unrolled_elements:
            return False
        for tile_lane_var, _tile_extent in self.lane_loops:
            wrapper = self.vec_lane_wrappers[tile_lane_var]
            constexpr_iter = expr_from_string("cutlass.range_constexpr(1)")
            assert isinstance(constexpr_iter, ast.expr)
            wrapper.outer_for.iter = constexpr_iter
        self.constexpr_lane_vars.add(lane_var)
        # The register-tile lowering (``cute/register_tile_reductions.py``)
        # rewrites this nest as a whole, so ``wrap_body`` keeps it intact.
        self.undistributable_lane_vars.add(lane_var)
        self.add_lane_loop(block_id, lane_var, extent, position=0)
        return True

    def _live_lane_setup(
        self, body: list[ast.AST]
    ) -> tuple[set[str], list[ast.AST], set[str]]:
        return _live_lane_setup(
            body,
            self.lane_setup_statements,
            self.vec_lane_wrappers,
            semantic_reads=CompileEnvironment.current().backend.lane_loop_semantic_reads(
                body
            ),
        )

    def lane_index_definitions(self) -> tuple[list[ast.AST], frozenset[str]]:
        return _lane_index_definitions(
            self.outer_prefix, self.lane_setup_statements, self.vec_lane_wrappers
        ), self.tile_masks

    def lane_loop_headers(self) -> list[ast.For]:
        return _lane_loop_headers(self.lane_loops, self.vec_lane_wrappers)

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:

        needed, kept_setup, spliced_wrappers = self._live_lane_setup(body)
        if self.deferred_vector_ops:
            has_barrier = any(
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "cute.arch.sync_threads"
                for stmt in body
                for node in ast.walk(stmt)
            )
            # Root codegen can register synthetic reduction lanes that are
            # subsequently unused (for example aliases of tile.index). Decide
            # the innermost live lane only after the entire body is available.
            innermost = None
            for lane_var, _extent in self.lane_loops:
                wrapper = self.vec_lane_wrappers.get(lane_var)
                names = {lane_var}
                if wrapper is not None:
                    names.update((wrapper.vec_lane_var, wrapper.base_index_var))
                if names & needed or lane_var in spliced_wrappers:
                    innermost = wrapper
            replacements = {}
            for vloop, scalar, emit in self.deferred_vector_ops:
                # Hoisting a load or delaying a store across a RAW barrier
                # changes its value even when the lane scope is unchanged.
                if (
                    not has_barrier
                    and innermost is not None
                    and innermost.vloop is vloop
                ):
                    vector = emit()
                    if vector is not None:
                        replacements[id(scalar)] = vector
            self.deferred_vector_ops.clear()

            class ReplaceVectorOps(ast.NodeTransformer):
                def visit(self, node: ast.AST) -> ast.AST:
                    if replacement := replacements.get(id(node)):
                        return replacement
                    return super().visit(node)

            replace = ReplaceVectorOps()
            body = [replace.visit(stmt) for stmt in body]
            # Accepted operations added vector-load/store splices, whose
            # address and mask setup must participate in final liveness too.
            needed, kept_setup, spliced_wrappers = self._live_lane_setup(body)
        # Place each setup at the shallowest lane scope that defines every
        # lane/base variable it reads (``_assign_lane_setup``).
        lane_scope_names: dict[str, set[str]] = {
            lane_var: _cute_lane_coordinates(
                lane_var, self.vec_lane_wrappers.get(lane_var)
            )
            for lane_var, _extent in self.lane_loops
        }
        setup_by_lane, fallback_setup = _assign_lane_setup(
            self.lane_loops, lane_scope_names, kept_setup
        )

        live_loops = [
            (lane_var, extent)
            for lane_var, extent in self.lane_loops
            if self._lane_loop_is_live(lane_var, needed, spliced_wrappers)
        ]
        from .cute.lane_loops.placement import check_full_nest
        from .cute.lane_loops.placement import definitions_precede_loop
        from .cute.lane_loops.placement import distribute_lane_loops
        from .cute.lane_loops.statements import LanePlacement

        # A statement that reads none of a lane loop's coordinates runs once
        # per iteration of that loop for nothing; place each statement inside
        # only the loops it depends on (``cute/lane_loops/placement.py``).
        # The full nest stays when every statement depends on every loop or
        # the redistribution cannot be proven safe, and for loops other passes
        # rewrite structurally (resident reductions, reverse scans).  Either
        # nest then gets the barriers ordering the accesses threads sharing a
        # tile axis could race on (``add_thread_barriers``).
        # A vec wrapper's outer lane body carries statements of its own
        # besides the V-loop: the per-thread lane base, packet loads
        # hoisted above the V-loop and store flushes after it.  Their
        # definitions are the loop's, and the body statements that move
        # past the loop are ordered against their memory accesses.
        attached = {
            lane_var: _vec_wrapper_statements(self.vec_lane_wrappers.get(lane_var))
            for lane_var, _extent in live_loops
        }
        scopes = _cute_lane_scopes(
            live_loops,
            setup_by_lane,
            lane_scope_names,
            attached,
            self.vec_lane_wrappers,
            self.tile_masks,
        )
        rename_groups = _current_rename_groups()
        placement = None
        if (
            live_loops
            and not self.reversed_lane_vars
            and not self.undistributable_lane_vars & {lane for lane, _ in live_loops}
            and not any(
                setup_by_lane[lane_var]
                for lane_var, _extent in self.lane_loops
                if lane_var not in dict(live_loops)
            )
        ):
            placement = distribute_lane_loops(body, scopes, rename_groups=rename_groups)
            # The definitions a hoisted packet load reads stay body statements
            # here; they must come out before the loop that hoisted the load.
            # Otherwise the full nest stays, if it is exact.
            if placement is not None and not all(
                definitions_precede_loop(
                    placement,
                    self.vec_wrapper_lane_var(relocation.vloop),
                    relocation.statements,
                )
                for relocation in self.pending_relocations
            ):
                check_full_nest(body, scopes, rename_groups=rename_groups)
                placement = None
        extents = dict(live_loops)
        # Pass order.  The deferred vector operations were emitted on the
        # scalar-form body and the body is now placed; next the collected
        # stores a later statement of their loop observes return to their
        # scalar form (``demote_reordered_tile_vec_stores``).  A demotion
        # changes the body and the wrapper statements the scopes describe, so
        # the body is placed again from the start.  Only the final,
        # demotion-free pass reaches the barrier pass: it classifies the
        # statements as they are emitted (a restored scalar store is a plain
        # store of its tensor) and no barrier it records among a wrapper's
        # statements is moved or dropped afterwards.  Both passes are the
        # backend's (``Backend.demote_grid_tile_vec_stores`` /
        # ``add_lane_nest_barriers``).
        backend = CompileEnvironment.current().backend
        if placement is None:
            # The full nest keeps every statement inside every loop: a site's
            # later accesses are the body statements after it.
            if backend.demote_grid_tile_vec_stores(self, body, None):
                return self.wrap_body(body)
            for relocation in self.pending_relocations:
                relocation.apply(body)
            placement = self._full_nest(body, setup_by_lane, extents)
        elif backend.demote_grid_tile_vec_stores(self, body, placement):
            return self.wrap_body(body)
        # Later passes rewrite the nest around an undistributable lane as a
        # whole; a body with compiler markers is left alone by the pass.
        if not self.undistributable_lane_vars & set(extents):
            backend.add_lane_nest_barriers(self, placement, scopes, rename_groups)

        def materialize(items: list[ast.AST | LanePlacement]) -> list[ast.AST]:
            result: list[ast.AST] = []
            for item in items:
                if isinstance(item, LanePlacement):
                    result.extend(
                        self._materialize_lane_loop(
                            item.lane_var,
                            extents[item.lane_var],
                            [
                                *setup_by_lane[item.lane_var],
                                *materialize(item.items),
                            ],
                            barriers=item.barriers,
                        )
                    )
                else:
                    result.append(item)
            return result

        return [*fallback_setup, *materialize(placement)]

    def _full_nest(
        self,
        body: list[ast.AST],
        setup_by_lane: dict[str, list[ast.AST]],
        extents: dict[str, int],
    ) -> list[ast.AST | LanePlacement]:
        """The original nest, ``body`` inside every lane loop, as placement items.

        A live loop (in ``extents``) is a ``LanePlacement``; a dead loop is
        spliced away and only its setup statements stay, where the loop
        would have been.
        """
        from .cute.lane_loops.statements import LanePlacement

        items: list[ast.AST | LanePlacement] = list(body)
        for lane_var, _extent in reversed(self.lane_loops):
            if lane_var in extents:
                items = [LanePlacement(lane_var, items)]
            else:
                items = [*setup_by_lane[lane_var], *items]
        return items

    def vec_wrapper_lane_var(self, vloop: ast.For) -> str:
        """The lane var whose vec wrapper owns the constexpr V-loop ``vloop``."""
        for lane_var, wrapper in self.vec_lane_wrappers.items():
            if wrapper.vloop is vloop:
                return lane_var
        raise AssertionError("relocation recorded for an unknown V-loop")

    def _lane_loop_is_live(
        self, lane_var: str, needed: set[str], spliced_wrappers: set[str]
    ) -> bool:
        return _lane_loop_is_live(
            lane_var, self.vec_lane_wrappers.get(lane_var), needed, spliced_wrappers
        )

    def _materialize_lane_loop(
        self,
        lane_var: str,
        extent: int,
        wrapped: list[ast.AST],
        *,
        barriers: list[int] | None = None,
    ) -> list[ast.AST]:
        """Wrap ``wrapped`` in the (pre-built or plain) loop of ``lane_var``.

        ``barriers`` are the positions among a vec wrapper's statements where
        ``add_thread_barriers`` put a block-wide barrier.
        """
        wrapper = self.vec_lane_wrappers.get(lane_var)
        _apply_wrapper_barriers(wrapper, barriers or [])
        if wrapper is not None:
            wrapper.vloop.body = wrapped  # type: ignore[assignment]
            if lane_var in self.reversed_lane_vars:
                # The vector store protocol appends the V results of the
                # constexpr vector loop in iteration order, so this
                # partition cannot run backwards; the scan declines such
                # shapes before requesting a reversal.
                raise CompileEnvironment.current().backend.unsupported(
                    "reverse scan over a vectorised lane loop"
                )
            return (
                list(wrapper.outer_for.body)
                if wrapper.elide_outer_loop
                else [wrapper.outer_for]
            )
        lane_loop = _create_lane_loop(
            lane_var,
            extent,
            wrapped,
            constexpr=lane_var in self.constexpr_lane_vars,
        )
        if lane_var in self.reversed_lane_vars and not _reverse_lane_loop_iter(
            lane_loop
        ):
            raise CompileEnvironment.current().backend.unsupported(
                "reverse scan lane loop is not reversible"
            )
        return [lane_loop]


@dataclasses.dataclass
class PersistentReductionState(DeviceLoopOrGridState):
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)

    def has_lane_loops(self) -> bool:
        return bool(self.lane_loops)

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:
        wrapped: list[ast.AST] = [*self.lane_setup_statements, *body]
        for lane_var, extent in reversed(self.lane_loops):
            wrapped = [_create_lane_loop(lane_var, extent, wrapped)]
        return wrapped


class TileStrategy:
    _fn: weakref.ReferenceType[DeviceFunction]
    block_ids: list[int]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
    ) -> None:
        self._fn = weakref.ref(fn)
        self.block_ids = block_ids
        self.index_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"indices_{block_idx}", dce=True)
            for block_idx in block_ids
        }
        # CuTe DSL preprocessor counter collision: the preprocessor's
        # negative-step machinery (``_handle_negative_step`` in
        # ``cutlass.base_dsl.ast_preprocessor.DSLPreprocessor``) emits
        # ``offset_<counter>`` / ``start_<counter>`` / ``stop_<counter>`` /
        # ``step_<counter>`` / ``isNegative_<counter>`` helpers at the enclosing
        # scope of every for-loop whose step is not a positive Python literal.
        # Helion's tile-offset names share the same ``offset_<n>`` namespace —
        # Python's name-binding rule sees the late preprocessor assignment and
        # treats the variable as local for the whole function body, turning
        # earlier reads into ``UnboundLocalError``. The ``tile_`` prefix moves
        # Helion's names out of the reserved CuTe DSL namespace. Of the five
        # reserved suffixes, only ``offset_`` and ``step_`` are emitted by
        # Helion (``offset_<bid>`` here; ``step_<n>`` via ``codegen.lift(...,
        # prefix='step')`` in ``codegen_grid_loops`` / ``codegen_lane_loops``);
        # both are renamed by the CuTe backend (``Backend.loop_var_prefixes``).
        # ``start_/stop_/isNegative_`` collisions are not currently emitted by
        # Helion. The other backends keep the historical short names to
        # preserve existing goldens — this is a deliberate trade-off (see
        # ``cute_plan.md`` §7.6.5.2 for the trade-off rationale; search "CuTe
        # DSL preprocessor counter collision" for the diagnosis).
        env = CompileEnvironment.current()
        offset_prefix, _step_prefix = env.backend.loop_var_prefixes()
        self.offset_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"{offset_prefix}_{block_idx}", dce=True)
            for block_idx in block_ids
        }

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def offset_var(self, block_idx: int) -> str:
        return self.offset_vars[block_idx]

    def tile_begin_var(self, block_idx: int) -> str:
        """Uniform first index of the current tile along ``block_idx``.

        ``tile.begin`` / ``tile.end`` / ``tile.id`` render through this. It is
        the loop offset for every strategy whose ``offset_var`` names the tile
        start; strategies whose offset is already the per-element index (the
        CuTe per-thread flattened tile) override it with the tile base.
        """
        return self.offset_var(block_idx)

    def grid_origin_var(self, block_idx: int) -> str:
        """Return the CTA's logical tile origin, without a thread/lane offset."""
        return self.offset_var(block_idx)

    def index_var(self, block_idx: int) -> str:
        return self.index_vars[block_idx]

    def mask_var(self, block_idx: int) -> str | None:
        raise NotImplementedError

    def block_size_var(self, block_idx: int) -> str | None:
        return self.fn.block_size_var_cache.get((block_idx,))

    def supports_index_rank_expansion(self) -> bool:
        """Whether index expressions produced by this strategy are tensor-shaped."""
        return True

    def thread_axes_used(self) -> int:
        return 0

    def thread_block_sizes(self) -> list[int]:
        """Return the thread block size for each thread axis this strategy uses."""
        return []

    def thread_block_size_exprs(self) -> list[str]:
        """Return per-axis thread block sizes as launch-time expressions."""
        return [str(size) for size in self.thread_block_sizes()]

    @staticmethod
    def get_tl_range_kwargs(config: Config, block_idx: int) -> list[str]:
        """Get the range_extra string for loop unroll factor and num_stages based on config."""
        env = CompileEnvironment.current()
        kwargs = []

        range_unroll_factor = env.config_spec.range_unroll_factors.config_get(
            config.range_unroll_factors, block_idx, 0
        )
        range_warp_specialize = env.config_spec.range_warp_specialize.config_get(
            CompileEnvironment.current().backend.range_warp_specializes(config),
            block_idx,
            None,
        )
        range_num_stages = env.config_spec.range_num_stages.config_get(
            config.range_num_stages, block_idx, 0
        )
        num_stages = config.num_stages

        if indexing_uses_tensor_descriptor(
            config.indexing
        ) or indexing_uses_tensor_descriptor(config.atomic_indexing):
            # Device-created tensor descriptors combined with multi-stage
            # range pipelines tend to cause CUDA "misaligned address" or
            # "unspecified launch failure" errors. Host-created descriptors
            # do not carry that per-program construction through the range.
            if range_num_stages > 0 and not config.host_tensor_descriptors:
                range_num_stages = 0
            # Host construction has not established that descriptor indexing
            # is safe with both unrolling and a separate kernel-level pipeline
            # (Triton's: gluon has none, its num_stages is the ring's depth).
            if (
                range_unroll_factor > 0
                and num_stages > 1
                and env.backend.name != "gluon"
            ):
                range_unroll_factor = 0
        elif (
            range_num_stages > 1
            and range_unroll_factor > 1
            and env.block_sizes[block_idx].size
            and env.block_sizes[block_idx].numel.is_number
        ):
            loop_numel = int(env.block_sizes[block_idx].numel)
            block_size = int(env.block_sizes[block_idx].from_config_assert(config))
            if env.backend.effective_num_warps(config) == 1:
                # One-warp kernels have no cross-warp layout hazard. Keep the
                # pipeline within the number of complete unrolled iterations;
                # using only the final tile's remainder would collapse every
                # exactly divisible reduction to one stage.
                loop_iterations = int(math.ceil(loop_numel / block_size))
                unrolled_iterations = int(
                    math.ceil(loop_iterations / range_unroll_factor)
                )
                range_num_stages = min(max(1, unrolled_iterations), range_num_stages)
            else:
                # Multi-warp unrolling plus pipelining can cause CUDA IMA. Keep
                # the conservative legacy bound for those kernels.
                step = range_unroll_factor * block_size
                last_offset = ((loop_numel - 1) // block_size) * block_size
                remainder = loop_numel - last_offset
                range_num_stages = min(
                    max(1, int(math.ceil(remainder / step))), range_num_stages
                )

        if range_unroll_factor > 0:
            kwargs.append(f"loop_unroll_factor={range_unroll_factor}")
        if range_warp_specialize is not None:
            kwargs.append(f"warp_specialize={range_warp_specialize}")
        if range_num_stages > 0:
            kwargs.append(f"num_stages={range_num_stages}")

        range_multi_buffer = env.config_spec.range_multi_buffers.config_get(
            config.range_multi_buffers, block_idx, None
        )
        if range_multi_buffer is not None:
            kwargs.append(f"disallow_acc_multi_buffer={not range_multi_buffer}")

        range_flatten = env.config_spec.range_flattens.config_get(
            config.range_flattens, block_idx, None
        )
        if range_flatten is not None:
            kwargs.append(f"flatten={range_flatten}")

        dpf_range = config.get("_triton_range_id_data_partition_factor", None)
        dpf_value = config.get("_triton_range_value_data_partition_factor", None)

        if dpf_range is not None and dpf_value is not None and dpf_range == block_idx:
            from .._compat import range_supports_data_partition_factor
            from .compile_environment import warning

            if range_supports_data_partition_factor():
                kwargs.append(f"data_partition_factor={dpf_value}")
            elif not _warned_data_partition_factor:
                _warned_data_partition_factor.append(True)
                warning(exc.TritonOptionUnavailable("data_partition_factor"))

        return kwargs

    @staticmethod
    def get_range_call_str(
        config: Config,
        block_ids: list[int],
        *,
        begin: str | None = None,
        end: str,
        step: str | None = None,
    ) -> str:
        env = CompileEnvironment.current()

        # Allow backend to override the range expression entirely
        backend_range = env.backend.range_str(begin, end, step)
        if backend_range is not None:
            return backend_range

        use_static_range = all(
            env.config_spec.static_ranges.config_get(
                config.static_ranges, block_idx, None
            )
            is True
            for block_idx in block_ids
        )

        range_args = []
        if begin is not None:
            range_args.append(begin)
        range_args.append(end)
        if step is not None and step != "1":
            range_args.append(step)

        if use_static_range:
            return f"tl.static_range({', '.join(range_args)})"

        range_kwargs = TileStrategy.get_tl_range_kwargs(config, block_ids[0])
        return f"tl.range({', '.join(range_args + range_kwargs)})"

    def user_size(self, block_index: int) -> sympy.Expr:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        raise NotImplementedError

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        raise NotImplementedError

    def codegen_preamble(self, state: CodegenState) -> None:
        """Called after a *different* strategy has been used to generate the grid."""

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        raise NotImplementedError

    def _create_block_id_info_dict(
        self,
        state: CodegenState,
        use_proxy_ends: bool = False,
        ends_override: list[object] | None = None,
    ) -> dict[int, LoopDimInfo]:
        """Helper to create block_id_to_info dictionary with end bounds.

        Args:
            state: The codegen state
            use_proxy_ends: If True, use proxy_ends from state.proxy_args (for device loops)
            ends_override: The absolute ends of a root grid; like a device loop's
                ends they bound ``tile.end``, ``tile.count`` and the tile extent
        """
        env = CompileEnvironment.current()
        block_id_to_info = {}

        def begin_to_ast(value: object) -> ast.AST:
            if isinstance(value, ast.AST):
                return value
            if isinstance(value, int):
                return expr_from_string(repr(value))
            if isinstance(value, sympy.Expr):
                return expr_from_string(DeviceFunction.current().sympy_expr(value))
            if isinstance(value, torch.SymInt):
                return begin_to_ast(value._sympy_())
            if isinstance(value, torch.Tensor):
                tensor_arg = DeviceFunction.current().tensor_arg(value)
                return expr_from_string(env.backend.scalar_load_expr(tensor_arg.name))
            raise NotImplementedError(f"{type(value)} is not implemented.")

        def normalize_dim_values(value: object) -> list[object]:
            if isinstance(value, (list, tuple, torch.Size)):
                return list(value)
            return [value]

        begin_values: list[object] | None = None
        proxy_begins: list[object] | None = None
        if isinstance(state.ast_args, (list, tuple)):
            if len(state.ast_args) >= 2 and isinstance(state.ast_args[1], list):
                begin_values = state.ast_args[1]
        if isinstance(state.proxy_args, (list, tuple)):
            # Root grids pass ``(begin_or_end, end, step)``: a list in their
            # second slot holds the ends, not the begins.
            root_grid = state.ast_args is None and len(state.proxy_args) == 3
            if (
                not root_grid
                and len(state.proxy_args) >= 2
                and isinstance(state.proxy_args[1], (list, tuple, torch.Size))
            ):
                proxy_begins = normalize_dim_values(state.proxy_args[1])
                if begin_values is None:
                    begin_values = proxy_begins
            elif len(state.proxy_args) >= 2:
                begin_arg, end_arg = state.proxy_args[:2]
                if end_arg is None:
                    proxy_begins = [0] * len(normalize_dim_values(begin_arg))
                else:
                    proxy_begins = normalize_dim_values(begin_arg)
                if begin_values is None:
                    begin_values = proxy_begins

        if use_proxy_ends:
            _, _, proxy_ends, _, _ = state.proxy_args
            assert isinstance(proxy_ends, list)
            for idx, (block_idx, end) in enumerate(
                zip(self.block_ids, proxy_ends, strict=True)
            ):
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                else:
                    end_expr = None
                block_id_to_info[block_idx] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=None,
                    end_expr=end_expr,
                )
        else:
            assert ends_override is not None
            for idx, (block_id, end) in enumerate(
                zip(self.block_ids, ends_override, strict=True)
            ):
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                    end_var_name = state.sympy_expr(end_expr)
                else:
                    # Tensor (data-dependent): no sympy end, so name its load
                    end_expr = None
                    end_var_name = state.codegen.lift(
                        begin_to_ast(end), dce=True, prefix="end"
                    ).id
                block_id_to_info[block_id] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=end_var_name,
                    end_expr=end_expr,
                )

        # Root grid arguments carry the absolute logical bounds.
        if state.ast_args is None and len(state.proxy_args) == 3:
            begin_or_end, end, _ = state.proxy_args
            grid_ends = normalize_dim_values(begin_or_end if end is None else end)
            for block_id, grid_end in zip(self.block_ids, grid_ends, strict=True):
                if isinstance(grid_end, (int, torch.SymInt, sympy.Expr)):
                    block_id_to_info[block_id].grid_end_expr = _to_sympy(grid_end)

        return block_id_to_info

    def _setup_block_size_constexpr(
        self, state: CodegenState, block_size_var: str, block_size: SymIntLike
    ) -> None:
        """Helper to setup constexpr block size variable on host."""
        state.device_function.constexpr_arg_with_host_def(block_size_var, block_size)


def _clone_lane_loop_with_body(loop: ast.For, body: list[ast.AST]) -> ast.For:
    """Clone a synthetic lane loop while preserving its iterator form.

    Most lane loops use ordinary ``range``.  A persistent reduction whose
    complete per-thread slice is one vector instead uses
    ``cutlass.range_constexpr(V)`` so the load/store hoist machinery can form
    one native vector transaction.  Reduction splitting must retain that
    constexpr iterator rather than silently rebuilding it as ``range``.
    """
    assert isinstance(loop.target, ast.Name)
    target = create(ast.Name, id=loop.target.id, ctx=ast.Store())
    iterator = ast.parse(ast.unparse(loop.iter), mode="eval").body
    cloned = create(
        ast.For,
        target=target,
        iter=iterator,
        body=body,
        orelse=[],
        type_comment=None,
    )
    lane_var = getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None)
    if isinstance(lane_var, str):
        setattr(cloned, HELION_LANE_LOOP_VAR_ATTR, lane_var)
    return cloned


class BlockSizeTileStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=block_ids,
        )
        self.block_size = block_size
        self.loop_order = loop_order

    def _reorder(self, block_ids: list[_T]) -> list[_T]:
        if len(block_ids) <= 1:
            return block_ids
        order = self.loop_order
        assert len(order) == len(block_ids), (
            f"Invalid order length: {len(order)} != {len(block_ids)}"
        )
        assert {*order} == {*range(len(order))}, f"Invalid permutation: {order}"
        return [block_ids[i] for i in reversed(order)]

    def _get_data_dependent_numel(
        self, state: CodegenState, end: object, begin: object
    ) -> sympy.Expr | str:
        """Get numel for data-dependent bounds using the tensor end value.

        When the tile bound is a tensor (data-dependent), we need to pass
        the tensor to the kernel and use it to compute the number of elements.
        Returns either a sympy.Expr or a string expression.
        """
        from .device_function import DeviceFunction

        device_function = DeviceFunction.current()

        if isinstance(end, torch.Tensor):
            # For tensor bounds, we need to add it as a kernel argument
            # and load the scalar value
            tensor_arg = device_function.tensor_arg(end)
            end_expr = CompileEnvironment.current().backend.scalar_load_expr(
                tensor_arg.name
            )
        elif isinstance(end, (int, torch.SymInt)):
            end_expr = device_function.sympy_expr(_to_sympy(end))
        else:
            raise NotImplementedError(f"Unsupported end type: {type(end)}")

        if begin == 0:
            # Simple case: numel = end
            return end_expr  # type: ignore[return-value]
        if isinstance(begin, torch.Tensor):
            begin_arg = device_function.tensor_arg(begin)
            begin_expr = CompileEnvironment.current().backend.scalar_load_expr(
                begin_arg.name
            )
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        if isinstance(begin, (int, torch.SymInt)):
            begin_expr = device_function.sympy_expr(_to_sympy(begin))
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        raise NotImplementedError(f"Unsupported begin type: {type(begin)}")

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].symbol()

    def _fold_tile_end_op(
        self,
        state: CodegenState,
        end: object,
        block_size: int | torch.SymInt,
    ) -> sympy.Expr | None:
        """
        Compute more precise end bound for the pattern:

            for outer in hl.tile(...):
                for inner in hl.tile(outer.begin, outer.end):
                    ...
        """
        if isinstance(end, (int, torch.SymInt)):
            end = _to_sympy(end)
        elif not isinstance(end, sympy.Expr):
            return None

        var_info = state.device_function.expr_to_var_info.get(end)
        if var_info is None or not isinstance(block_size, int):
            return end

        from ..language.tile_ops import tile_end

        env = CompileEnvironment.current()
        fx_node = var_info.fx_node
        # check for the case where we have the same end bound a parent loop
        if (
            fx_node is not None
            and fx_node.target is tile_end
            and isinstance(arg := fx_node.args[0], torch.fx.Node)
            and (block_id := env.get_block_id(arg.meta["val"])) is not None
            and (device_loops := state.codegen.active_device_loops.get(block_id))
            and (loop_info := device_loops[-1].block_id_to_info.get(block_id))
            is not None
            # TODO(jansel): when parent block size is a SymInt, we fail to apply this optimization should fix this
            and isinstance(
                parent_block_size := state.device_function.resolved_block_size(
                    block_id
                ),
                int,
            )
            # If our block size is larger than the parent, then their will be gaps in the iteration space
            and block_size <= parent_block_size
        ):
            # Replace our end bound (a SymInt) will the parent loop's end bound
            return loop_info.end_expr
        return end

    def _compute_thread_axis_offset(
        self,
        active_device_loops: dict[int, list[DeviceLoopOrGridState]],
    ) -> int:
        """Compute the starting thread axis for the next strategy.

        Counts axes already claimed by active device loops, reserving at
        least one axis for reduction strategies when the backend places
        reductions first.

        When a ``CuTeGridExecutionPlan`` with ``block_axis_priority`` is
        in scope for this strategy's blocks, the offset is instead
        derived from ``thread_axis_for_strategy`` so the M/N axis order
        is dictated by the plan (e.g. the warp-per-row layout swaps the
        outer M-grid and inner N-tile axes so each warp owns one row).
        """
        from .reduction_strategy import ReductionStrategy

        env = CompileEnvironment.current()

        sibling_axis = self.fn.tile_strategy.collective_sibling_thread_axis(self)
        if sibling_axis is not None:
            return sibling_axis

        # Plan-driven path: honor ``block_axis_priority`` so the outer
        # grid loop can reserve an axis for a lower-priority inner tile
        # loop even when that inner loop has not yet entered
        # ``active_device_loops``.  Used by the warp-per-row layout where
        # the outer M-grid must take a HIGHER thread-axis index than the
        # inner N-tile so 32 contiguous threads on axis 0 form one warp
        # per row.
        plan = self.fn.tile_strategy.current_cute_grid_execution_plan(
            block_ids=self.block_ids
        )
        if plan is not None and any(
            plan.priority_for_block(block_id) is not None for block_id in self.block_ids
        ):
            offset = self.fn.tile_strategy.thread_axis_for_strategy(self)
            if offset is not None:
                return offset

        if (
            env.backend.plans_launch_thread_axes()
            and len(HostFunction.current().device_ir.phases) > 1
        ):
            # All barrier phases share the launch layout. Use the dispatcher's
            # kernel-wide reservation, including axes absent from this phase,
            # so the body's tile indices agree with the launch block dimensions.
            offset = self.fn.tile_strategy.thread_axis_for_strategy(self)
            if offset is not None:
                return offset

        seen: set[int] = set()
        active_reduction_axes = 0
        active_non_reduction_axes = 0
        for loops in active_device_loops.values():
            for loop_state in loops:
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                axes = loop_state.strategy.thread_axes_used()
                if env.backend.reduction_axis_first() and isinstance(
                    loop_state.strategy, ReductionStrategy
                ):
                    # A reduction with one live thread reserves no axis (see
                    # ``planned_reduction_axes`` below).
                    if loop_state.strategy._reduction_thread_count() > 1:
                        active_reduction_axes += axes
                else:
                    active_non_reduction_axes += axes

        if not env.backend.reduction_axis_first():
            return active_non_reduction_axes + active_reduction_axes

        # Reduction strategies claim axes 0..n-1 in creation order
        # (``_get_thread_axis``), so reserving only one axis when two
        # multi-thread reductions are live would place this strategy on the
        # same axis as the second reduction. That collision double-books the
        # axis (e.g. a tile axis planned for 2 threads sharing thread_idx[1]
        # with a 4-thread reduction), making the generated tile indices span
        # more elements than the tile holds. Reserve one axis per reduction
        # that actually spreads across threads; single-thread reductions
        # (thread_idx is constant 0 on their axis) may share an axis safely.
        # The reservation is kernel-wide (it also counts reductions of other
        # root loops and ``hl.barrier()`` phases);
        # ``TileStrategyDispatch.thread_axis_for_strategy`` applies the same
        # reservation so the launch block dims agree with the axes the body
        # indexes.
        planned_reduction_axes = self.fn.tile_strategy.reserved_reduction_axes()
        if plan is not None and any(
            plan.disables_reduction_axis_reservation(block_id)
            for block_id in self.block_ids
        ):
            return self._check_launch_thread_axis(
                active_non_reduction_axes + active_reduction_axes
            )
        reserved_reduction_axes = max(planned_reduction_axes, active_reduction_axes)
        offset = reserved_reduction_axes + active_non_reduction_axes
        tile_axes = set(range(offset, offset + self.thread_axes_used()))
        for strategy in self.fn.tile_strategy.strategies:
            if not isinstance(strategy, ReductionStrategy):
                continue
            if not self.fn.tile_strategy.strategies_can_coexecute(self, strategy):
                continue
            if not any(size > 1 for size in strategy.thread_block_sizes()):
                continue
            reduction_axis = self.fn.tile_strategy.thread_axis_for_strategy(strategy)
            if reduction_axis is None:
                continue
            reduction_axes = set(
                range(reduction_axis, reduction_axis + strategy.thread_axes_used())
            )
            if collision := tile_axes & reduction_axes:
                axes = ", ".join(map(str, sorted(collision)))
                raise exc.BackendUnsupported(
                    env.backend.name,
                    "thread-axis collision: tile blocks "
                    f"{self.block_ids} and reduction/slice block "
                    f"{strategy.block_index} both require axis {axes}",
                )
        return self._check_launch_thread_axis(offset)

    def _check_launch_thread_axis(self, offset: int) -> int:
        """Return ``offset``, the first thread axis the body gives this
        strategy, after checking that the launch layout agrees.

        The launch block dims, the surplus-thread masks and every shared
        memory exchange read the layout from
        ``TileStrategyDispatch.thread_axis_for_strategy``; a body indexing
        another axis would run with the wrong thread counts (rows of the tile
        never launched, or launched without their masks).
        """
        env = CompileEnvironment.current()
        if not env.backend.plans_launch_thread_axes() or self.thread_axes_used() == 0:
            return offset
        planned = self.fn.tile_strategy.thread_axis_for_strategy(self)
        if planned is not None and planned != offset:
            raise exc.BackendUnsupported(
                env.backend.name,
                f"thread-axis layout mismatch: the kernel body places tile "
                f"blocks {self.block_ids} on thread axis {offset} but the launch "
                f"layout plans axis {planned}",
            )
        return offset

    def select_pid_strategy(self) -> ProgramIDs:
        env = CompileEnvironment.current()
        if env.compact_worklist_plan is not None:
            # Compact worklist: the owner hl.grid becomes the work-item grid.
            from .program_id import WorklistProgramIDs

            return WorklistProgramIDs(upper_expr=str(env.compact_worklist_upper))
        pid_type = self.fn.config.pid_type
        if pid_type == "xyz":
            assert 1 < len(self.block_ids) <= 3
            return XYZProgramIDs()
        backend_pid = env.backend.create_program_ids(self.fn, pid_type, self.block_ids)
        if backend_pid is not None:
            return backend_pid
        if pid_type == "persistent_blocked":
            return PersistentBlockedProgramIDs()
        if pid_type == "persistent_interleaved":
            return PersistentInterleavedProgramIDs()
        assert pid_type == "flat"
        return FlatProgramIDs()


class FlattenedTileStrategy(BlockSizeTileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    # pyrefly: ignore [bad-override]
    block_size: SymIntLike

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, (int, torch.SymInt))
        super().__init__(fn, block_ids, block_size, loop_order)
        env = CompileEnvironment.current()
        self._mask_elidable = not env.backend.force_tile_mask() and env.known_multiple(
            functools.reduce(  # pyrefly: ignore[incompatible-overload-residual]
                operator.mul, [env.block_sizes[i].numel for i in block_ids]
            ),
            block_size,
        )
        # Elision also requires that no sibling section launches this
        # strategy's thread axis wider than the tile, which is unknowable
        # until every strategy is registered in the dispatch — ``mask_var``
        # decides lazily on first use during codegen.
        self._mask_elision_decided = False
        self._mask_var: str | None = self.new_var("mask", dce=True)
        self._offsets_var = self.new_var("offsets", dce=True)
        self._grid_origin_vars: dict[int, str] = {}
        # The trip count each block's index is decoded from the flat offset
        # with, recorded by the codegen.
        self.trip_counts: dict[int, sympy.Expr | str] = {}

        key = (*self.block_ids,)
        assert key not in fn.block_size_var_cache
        fn.block_size_var_cache[key] = bs_var = self.new_var("_BLOCK_SIZE")
        for block_index in block_ids:
            fn.block_size_var_cache[(block_index,)] = bs_var

    def new_var(self, prefix: str, dce: bool = False) -> str:
        return self.fn.new_var(
            f"{prefix}_{'_'.join(map(str, self.block_ids))}", dce=dce
        )

    def offset_var(self, block_idx: int) -> str:
        raise NotImplementedError("offset_var not used in FlattenedTileStrategy")

    def grid_origin_var(self, block_idx: int) -> str:
        if block_idx not in self._grid_origin_vars:
            raise exc.InvalidConfig(
                "computed fragments require rectangular flattened grid tiles"
            )
        return self._grid_origin_vars[block_idx]

    def _record_grid_origins(
        self,
        state: CodegenState,
        pid_var: str,
        block_size_var: str,
        begins: list[object],
        steps: list[object | None],
    ) -> None:
        if not CompileEnvironment.current().backend.tracks_flat_grid_origins():
            return
        if self.block_size == 1:
            # A singleton flat tile is a singleton Cartesian tile on every
            # axis, including nonzero begins and scalar iteration strides.
            self._grid_origin_vars.update(self.index_vars)
        elif len(self.block_ids) == 1 and steps[0] in (None, 1):
            block_id = self.block_ids[0]
            origin = self.offset_vars[block_id]
            expression = f"({pid_var}) * ({block_size_var})"
            if begins[0] != 0:
                expression = f"({self._expr_str(begins[0])}) + ({expression})"
            state.add_statement(f"{origin} = {expression}")
            self._grid_origin_vars[block_id] = origin
        # A multi-axis flattened interval is not a Cartesian tile. Do not
        # reinterpret its scalar per-thread coordinates as independent origins.

    def mask_var(self, block_idx: int) -> str | None:
        if not self._mask_elision_decided:
            self._mask_elision_decided = True
            if self._mask_elidable and not (
                CompileEnvironment.current().backend.launches_surplus_tile_threads()
                and self.fn.tile_strategy.has_surplus_threads_for_strategy(self)
            ):
                self._mask_var = None
        return self._mask_var

    def block_size_var(self, block_idx: int) -> str:
        return self.fn.block_size_var_cache[tuple(self.block_ids)]

    def thread_axes_used(self) -> int:
        return int(self._uses_thread_axis())

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis() or not isinstance(self.block_size, int):
            return []
        return [self.block_size]

    def thread_block_size_exprs(self) -> list[str]:
        if not self._uses_thread_axis():
            return []
        if isinstance(self.block_size, int):
            return [str(self.block_size)]
        bs_var = self.block_size_var(-1)
        if bs_var is None:
            return []
        return [bs_var]

    def _uses_thread_axis(self) -> bool:
        return not (isinstance(self.block_size, int) and self.block_size == 1)

    def _numel_str(self, state: CodegenState, value: sympy.Expr | str) -> str:
        if isinstance(value, str):
            return value
        return state.sympy_expr(value)

    def _range_trip_count(
        self,
        begin: object,
        end: object,
        step: object | None,
    ) -> sympy.Expr | str:
        return self._range_numel_expr(begin, end, step)

    def _range_numel_expr(
        self, begin: object, end: object, step: object | None
    ) -> sympy.Expr | str:
        begin_expr = (
            _to_sympy(begin)
            if isinstance(begin, (int, torch.SymInt, sympy.Expr))
            else None
        )
        end_expr = (
            _to_sympy(end) if isinstance(end, (int, torch.SymInt, sympy.Expr)) else None
        )
        diff_expr = (
            sympy.Add(end_expr, sympy.Mul(-1, begin_expr))
            if begin_expr is not None and end_expr is not None
            else None
        )
        if step is None or step == 1:
            if diff_expr is not None:
                return diff_expr
            return f"(({self._expr_str(end)}) - ({self._expr_str(begin)}))"
        assert isinstance(step, (int, torch.SymInt, sympy.Expr))
        step_expr = _to_sympy(step)
        if getattr(step_expr, "free_symbols", None):
            return (
                f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
                f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
            )
        if diff_expr is not None:
            return sympy.ceiling(sympy.Mul(diff_expr, sympy.Pow(step_expr, -1)))
        return (
            f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
            f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
        )

    def _expr_str(self, value: object) -> str:
        if isinstance(value, (int, torch.SymInt, sympy.Expr)):
            return self.fn.sympy_expr(_to_sympy(value))
        if isinstance(value, torch.Tensor):
            tensor_arg = DeviceFunction.current().tensor_arg(value)
            return CompileEnvironment.current().backend.scalar_load_expr(
                tensor_arg.name
            )
        if isinstance(value, str):
            return value
        raise NotImplementedError(f"{type(value)} is not implemented.")

    def _normalize_loop_steps(
        self, step_arg: object | None, ndim: int
    ) -> list[object | None]:
        if step_arg is None:
            return [None] * ndim
        if isinstance(step_arg, (list, tuple)):
            steps = list(step_arg)
            assert len(steps) == ndim
            return steps
        return [step_arg] * ndim

    def _extract_root_bounds(
        self, state: CodegenState
    ) -> tuple[list[object], list[object], list[object | None]]:
        assert len(state.proxy_args) == 3
        if state.proxy_args[1] is None:
            begins: list[object] = [0] * len(self.block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins_arg = state.proxy_args[0]
            begins = (
                list(begins_arg)
                if isinstance(begins_arg, (list, tuple))
                else [begins_arg]
            )
            ends_arg = state.proxy_args[1]
        ends = list(ends_arg) if isinstance(ends_arg, (list, tuple)) else [ends_arg]
        steps = self._normalize_loop_steps(state.proxy_args[2], len(self.block_ids))
        assert len(begins) == len(self.block_ids)
        assert len(ends) == len(self.block_ids)
        return begins, ends, steps

    def _extract_device_loop_bounds(
        self, state: CodegenState
    ) -> tuple[list[object], list[object], list[object | None]]:
        if len(state.ast_args) == 5:
            _, begins_arg, ends_arg, _, steps_arg = state.ast_args
        else:
            _, begins_arg, ends_arg, _ = state.ast_args
            steps_arg = None
        begins = (
            list(begins_arg) if isinstance(begins_arg, (list, tuple)) else [begins_arg]
        )
        ends = list(ends_arg) if isinstance(ends_arg, (list, tuple)) else [ends_arg]
        steps = self._normalize_loop_steps(steps_arg, len(self.block_ids))
        assert len(begins) == len(self.block_ids)
        assert len(ends) == len(self.block_ids)
        return begins, ends, steps

    def _codegen_common(
        self,
        state: CodegenState,
        *,
        begins: list[object] | None = None,
        ends: list[object] | None = None,
        steps: list[object | None] | None = None,
    ) -> tuple[str, str, sympy.Expr | str, list[ast.AST]]:
        offsets_var = self._offsets_var
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        if begins is None:
            begins = [0] * len(block_ids)
        if ends is None:
            ends = [env.block_sizes[block_id].numel for block_id in block_ids]
        if steps is None:
            steps = [None] * len(block_ids)
        total_numel: sympy.Expr | str = sympy.S.One
        statements = []

        # pyrefly: ignore [bad-assignment]
        for i, (block_idx, begin, end, step) in enumerate(
            self._reorder([*zip(block_ids, begins, ends, steps, strict=True)])
        ):
            # A strided single-block tile on a thread axis may walk the
            # elements of its range rather than its trips.
            elementwise_tile = (
                env.backend.flat_strided_tile_walks_elements()
                and len(block_ids) == 1
                and self._uses_thread_axis()
                and step not in (None, 1)
            )
            numel = (
                self._range_numel_expr(begin, end, None)
                if elementwise_tile
                else self._range_trip_count(begin, end, step)
            )
            self.trip_counts[block_idx] = numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({self._numel_str(state, total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({self._numel_str(state, numel)})"
            step_expr = self._expr_str(step) if step not in (None, 1) else None
            if step_expr is not None and not elementwise_tile:
                expr = f"({expr}) * ({step_expr})"
            if begin != 0:
                expr = f"({self._expr_str(begin)}) + ({expr})"
            statements.append(statement_from_string(f"{block_index_var} = {expr}"))
            if isinstance(total_numel, str) or isinstance(numel, str):
                total_numel = (
                    f"({self._numel_str(state, total_numel)})"
                    f" * ({self._numel_str(state, numel)})"
                )
            else:
                assert isinstance(total_numel, sympy.Expr)
                assert isinstance(numel, sympy.Expr)
                total_numel = sympy.Mul(total_numel, numel)

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            mask_terms = [f"{offsets_var} < ({self._numel_str(state, total_numel)})"]
            # Skip the ``thread_idx[axis] < block_size`` term for a block-size-1
            # axis that claims no thread axis (see ``codegen_grid``): the term
            # would otherwise pin the launch dim to 1 and block a synthetic
            # free-``hl.arange`` axis from reusing it.
            if env.backend.single_thread_tile_claims_axis() or self._uses_thread_axis():
                thread_mask = env.backend.thread_in_tile_mask_expr(
                    block_size_var, axis=self._flat_thread_axis()
                )
                if thread_mask is not None:
                    mask_terms.insert(0, f"({thread_mask})")
            mask_expr = " and ".join(mask_terms)
            statements.append(statement_from_string(f"{mask_var} = {mask_expr}"))
        # pyrefly: ignore [bad-return]
        return block_size_var, offsets_var, total_numel, statements

    def _flat_thread_axis(self) -> int:
        """Compute the thread axis for this flattened strategy.

        For CuTe, reduction strategies occupy earlier axes.
        """
        return self._compute_thread_axis_offset(self.fn.codegen.active_device_loops)

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        assert state.ast_args is None

        from .ast_extension import ExtendedAST
        from .type_info import GridIndexType
        from .type_info import IterType
        from .type_info import SequenceType

        type_info = ExtendedAST.current()[-1]._type_info
        scalar_grid_loop = False
        if isinstance(type_info, IterType):
            inner = (
                type_info.inner.unpack()
                if isinstance(type_info.inner, SequenceType)
                else [type_info.inner]
            )
            scalar_grid_loop = len(inner) == 1 and isinstance(inner[0], GridIndexType)

        if (
            scalar_grid_loop
            and len(self.block_ids) == 1
            and len(state.proxy_args) == 3
            and not isinstance(state.proxy_args[0], (list, tuple))
            and (
                state.proxy_args[1] is None
                or not isinstance(state.proxy_args[1], (list, tuple))
            )
            and not isinstance(state.proxy_args[2], (list, tuple))
        ):

            def _range_bound_to_sympy(value: object) -> sympy.Expr:
                assert isinstance(value, (int, torch.SymInt, sympy.Expr))
                return _to_sympy(value)

            step = state.proxy_args[2]
            if step not in (None, 1):
                block_id = self.block_ids[0]
                if state.proxy_args[1] is None:
                    begin = 0
                    end = state.proxy_args[0]
                else:
                    begin = state.proxy_args[0]
                    end = state.proxy_args[1]
                    if isinstance(begin, (list, tuple)):
                        assert len(begin) == 1
                        begin = begin[0]
                    if isinstance(end, (list, tuple)):
                        assert len(end) == 1
                        end = end[0]
                begin_expr = _range_bound_to_sympy(begin)
                end_expr = _range_bound_to_sympy(end)
                step_expr = _range_bound_to_sympy(step)
                trip_count = (
                    f"(({state.sympy_expr(end_expr)}) - ({state.sympy_expr(begin_expr)}) + "
                    f"({state.sympy_expr(step_expr)}) - 1) // ({state.sympy_expr(step_expr)})"
                )

                env = CompileEnvironment.current()
                dtype = env.index_type()
                pid_var = state.device_function.new_var("pid_flat", dce=True)
                offsets_var = self._offsets_var
                block_size_var = self.block_size_var(-1)
                self._setup_block_size_constexpr(state, block_size_var, self.block_size)
                pids = self.select_pid_strategy()
                if isinstance(state.device_function.pid, ForEachProgramID):
                    pids.shared_pid_var = state.device_function.pid.shared_pid_var
                pids.append(PIDInfo(pid_var, block_size_var, trip_count, block_id))
                self._record_grid_origins(
                    state, pid_var, block_size_var, [begin], [step]
                )
                state.add_statement(
                    env.backend.arange_expr(
                        offsets_var,
                        pid_var,
                        block_size_var,
                        dtype,
                        axis=self._flat_thread_axis(),
                    )
                )
                index_var = self.index_var(block_id)
                state.add_statement(
                    f"{index_var} = ({state.sympy_expr(begin_expr)}) + ({offsets_var}) * ({state.sympy_expr(step_expr)})"
                )
                mask_var = self.mask_var(-1)
                if mask_var is not None:
                    mask_terms = [f"{offsets_var} < ({trip_count})"]
                    thread_mask = env.backend.thread_in_tile_mask_expr(
                        block_size_var, axis=self._flat_thread_axis()
                    )
                    if thread_mask is not None:
                        mask_terms.insert(0, f"({thread_mask})")
                    state.add_statement(
                        statement_from_string(
                            f"{mask_var} = {' and '.join(mask_terms)}"
                        )
                    )
                pids.codegen(state)
                if isinstance(state.device_function.pid, ForEachProgramID):
                    shared_pid = state.device_function.pid
                    shared_pid.cases.append(pids)
                    shared_pid.codegen(state)
                else:
                    state.device_function.set_pid(pids)
                tracker = ThreadAxisTracker()
                if self._uses_thread_axis() and isinstance(self.block_size, int):
                    tracker.record_all(
                        self.block_ids, self._flat_thread_axis(), self.block_size
                    )
                elif self._uses_thread_axis():
                    tracker.record_symbolic_axis(
                        self.block_ids, self._flat_thread_axis()
                    )
                return DeviceGridState(
                    self,
                    block_id_to_info=self._create_block_id_info_dict(
                        state, ends_override=[end]
                    ),
                    thread_axis_sizes=tracker.sizes,
                    block_thread_axes=tracker.block_axes,
                )
        begins, ends, steps = self._extract_root_bounds(state)
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state,
            begins=begins,
            ends=ends,
            steps=steps,
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))
        self._record_grid_origins(state, pid_var, block_size_var, begins, steps)

        # A grid whose block size is 1 need not claim a thread axis: its
        # ``offsets = pid * 1 + thread_idx[axis]`` term is always 0 (launch dim
        # for the axis is 1). Emit ``offsets = pid * 1`` instead so the axis is
        # genuinely free for a synthetic free-``hl.arange`` thread axis to reuse
        # without the grid's ``thread_idx[axis] < 1`` mask filtering its lanes.
        if (
            not env.backend.single_thread_tile_claims_axis()
            and not self._uses_thread_axis()
        ):
            state.add_statement(
                statement_from_string(
                    f"{offsets_var} = ({pid_var}) * ({block_size_var})"
                )
            )
        else:
            state.add_statement(
                env.backend.arange_expr(
                    offsets_var,
                    pid_var,
                    block_size_var,
                    dtype,
                    axis=self._flat_thread_axis(),
                )
            )
        state.codegen.statements_stack[-1].extend(statements)

        pids.codegen(state)

        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis():
            thread_size: int | None = None
            if isinstance(self.block_size, int):
                thread_size = self.block_size
            elif isinstance(self.block_size, torch.SymInt):
                if (block_size_id := env.get_block_id(self.block_size)) is not None:
                    config_block_size = env.config_spec.block_sizes.config_get(
                        state.config.block_sizes,
                        block_size_id,
                    )
                    if isinstance(config_block_size, int):
                        thread_size = config_block_size
            if thread_size is not None:
                tracker.record_all(
                    self.block_ids, self._flat_thread_axis(), thread_size
                )
            else:
                tracker.record_symbolic_axis(self.block_ids, self._flat_thread_axis())
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        begins, ends, steps = self._extract_device_loop_bounds(state)
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state,
            begins=begins,
            ends=ends,
            steps=steps,
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()
        lid = self.new_var("lid")
        numel_str = self._numel_str(state, total_numel)
        end_var = env.backend.cdiv_expr(numel_str, block_size_var, is_device=True)
        # Mirror ``codegen_grid``: a block-size-1 loop axis need not claim a
        # thread axis, so drop the always-zero ``+ thread_idx[axis]`` term so the
        # axis stays free (and consistent with the mask emitted by
        # ``_codegen_common``, which also drops its thread term for this case).
        if (
            not env.backend.single_thread_tile_claims_axis()
            and not self._uses_thread_axis()
        ):
            arange_expr = f"{offsets_var} = ({lid}) * ({block_size_var})"
        else:
            arange_expr = env.backend.arange_expr(
                offsets_var, lid, block_size_var, dtype, axis=self._flat_thread_axis()
            )
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=(
                body := [
                    statement_from_string(arange_expr),
                    *statements,
                ]
            ),
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis():
            thread_size: int | None = None
            if isinstance(self.block_size, int):
                thread_size = self.block_size
            elif isinstance(self.block_size, torch.SymInt):
                if (block_size_id := env.get_block_id(self.block_size)) is not None:
                    config_block_size = env.config_spec.block_sizes.config_get(
                        state.config.block_sizes,
                        block_size_id,
                    )
                    if isinstance(config_block_size, int):
                        thread_size = config_block_size
            if thread_size is not None:
                tracker.record_all(
                    self.block_ids, self._flat_thread_axis(), thread_size
                )
            else:
                tracker.record_symbolic_axis(self.block_ids, self._flat_thread_axis())
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    @classmethod
    def update_allow_flattened(cls, shape: Sequence[sympy.Expr]) -> None:
        env = CompileEnvironment.current()
        used_indices = {}
        for i, x in enumerate(shape):
            block_idx = env.get_block_id(x)
            if block_idx is not None:
                used_indices[block_idx] = i
        flatten_loops = env.config_spec.flatten_loops
        for spec in [*flatten_loops]:
            block_ids = spec.block_ids
            disable = not (
                all(x in used_indices for x in block_ids)
                or all(x not in used_indices for x in block_ids)
            )
            if not disable:
                for i, j in itertools.pairwise(block_ids):
                    if i in used_indices and used_indices[i] + 1 != used_indices[j]:
                        # The block indices must be contiguous
                        disable = True
                        break
            if disable:
                flatten_loops.disable_block_id(block_ids[0])
                if env.backend.reflattens_pointwise_tiles():
                    # This gate exists for vector-model backends, where a
                    # partial-block access (``bias[tile_n]``) cannot
                    # broadcast against a flattened [BS] value vector.  A
                    # per-thread SCALAR model recomputes per-dim index
                    # vars from the flat offset per element, so a PURE
                    # pointwise kernel can flatten safely — record the spec
                    # and let device-IR analysis re-register it once the
                    # PointwiseElementwiseFact (no reductions / matmuls /
                    # accumulators) is known.
                    cands = env.config_spec.cute_reflatten_candidates
                    if all(c.block_ids != block_ids for c in cands):
                        cands.append(spec)

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # Keep axis structure intact for multi-phase kernels (e.g., barrier) to
        # avoid mismatched ranks in downstream reductions.
        if len(HostFunction.current().device_ir.root_ids) > 1:
            return shapes

        env = CompileEnvironment.current()
        # Filter out unit-sized blocks that don't need compacting
        compact_block_ids = [
            block_id
            for block_id in self.block_ids
            if not (
                isinstance(env.block_sizes[block_id].size, int)
                and env.block_sizes[block_id].size == 1
            )
        ]
        if not compact_block_ids:
            return shapes

        output = []
        shape_queue = collections.deque(shapes)
        while shape_queue:
            shape = shape_queue.popleft()
            # Check if this starts our flattened sequence
            if len(shape.block_ids) != 1 or shape.block_ids[0] != compact_block_ids[0]:
                output.append(shape)
                continue

            # Try to collect the full sequence
            group_shapes = [shape]
            found_complete_sequence = True
            for expected in compact_block_ids[1:]:
                if (
                    shape_queue
                    and len(shape_queue[0].block_ids) == 1
                    and shape_queue[0].block_ids[0] == expected
                ):
                    group_shapes.append(shape_queue.popleft())
                else:
                    # Partial match - don't combine
                    found_complete_sequence = False
                    output.extend(group_shapes)
                    break

            if found_complete_sequence:
                # Full match - combine into one
                for s in group_shapes[1:]:
                    shape = shape.combine(s)
                output.append(shape)
        return output


class _BaseNDTileStrategy(BlockSizeTileStrategy):
    # pyrefly: ignore [bad-override]
    block_size: list[SymIntLike]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, list)
        super().__init__(fn, block_ids, block_size, loop_order)
        for bs, block_idx in zip(block_size, block_ids, strict=True):
            if (block_idx,) not in fn.block_size_var_cache and bs != 1:
                fn.block_size_var_cache[(block_idx,)] = fn.new_var(
                    f"_BLOCK_SIZE_{block_idx}"
                )

    def _uses_thread_axis(self, block_size: SymIntLike) -> bool:
        return not (isinstance(block_size, int) and block_size == 1)

    def _uses_thread_axis_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> bool:
        """Hook: does ``block_id`` claim a CUDA thread axis under this strategy?

        Defaults to ``_uses_thread_axis(block_size)``. Subclasses that
        track per-block-id state (e.g. ``PerThreadNDTileStrategy``'s
        ``inactive_block_ids``) override this to return False for
        block_ids that don't claim an axis so the grid / device-loop
        codegen does not emit ``thread_idx[axis]`` for them.
        """
        return self._uses_thread_axis(block_size)

    def thread_axes_used(self) -> int:
        return sum(
            1 for block_size in self.block_size if self._uses_thread_axis(block_size)
        )

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if self._uses_thread_axis(bs) and isinstance(bs, int):
                sizes.append(bs)
        return sizes

    def thread_block_size_exprs(self) -> list[str]:
        exprs: list[str] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if not self._uses_thread_axis(bs):
                continue
            if isinstance(bs, int):
                exprs.append(str(bs))
            else:
                bs_var = self.block_size_var(block_id)
                if bs_var is None:
                    return []
                exprs.append(bs_var)
        return exprs

    def _thread_axis_offset(self, state: CodegenState) -> int:
        return self._compute_thread_axis_offset(state.codegen.active_device_loops)

    def _thread_axis_map(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis(block_size_by_id[block_id]):
                axis += 1
        return mapping

    def _normalize_loop_steps(
        self, step_arg: object | None, ndim: int
    ) -> list[object | None]:
        if step_arg is None:
            return [None] * ndim
        if isinstance(step_arg, (list, tuple)):
            steps = list(step_arg)
            assert len(steps) == ndim
            return steps
        return [step_arg] * ndim

    def _root_grid_steps(self, state: CodegenState) -> list[object | None]:
        from .ast_extension import ExtendedAST
        from .type_info import GridIndexType
        from .type_info import IterType
        from .type_info import SequenceType

        type_info = ExtendedAST.current()[-1]._type_info
        assert isinstance(type_info, IterType)
        inner = (
            type_info.inner.unpack()
            if isinstance(type_info.inner, SequenceType)
            else [type_info.inner]
        )
        if not all(isinstance(value, GridIndexType) for value in inner):
            return [None] * len(self.block_ids)
        return self._normalize_loop_steps(state.proxy_args[2], len(self.block_ids))

    def _range_numel_expr(
        self, begin: object, end: object, step: object | None
    ) -> sympy.Expr | str:
        begin_expr = (
            _to_sympy(begin)
            if isinstance(begin, (int, torch.SymInt, sympy.Expr))
            else None
        )
        end_expr = (
            _to_sympy(end) if isinstance(end, (int, torch.SymInt, sympy.Expr)) else None
        )
        diff_expr = (
            sympy.Add(end_expr, sympy.Mul(-1, begin_expr))
            if begin_expr is not None and end_expr is not None
            else None
        )
        if step is None or step == 1:
            if diff_expr is not None:
                return diff_expr
            return f"(({self._expr_str(end)}) - ({self._expr_str(begin)}))"
        assert isinstance(step, (int, torch.SymInt, sympy.Expr))
        step_expr = _to_sympy(step)
        if getattr(step_expr, "free_symbols", None):
            return (
                f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
                f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
            )
        if diff_expr is not None:
            return sympy.ceiling(sympy.Mul(diff_expr, sympy.Pow(step_expr, -1)))
        return (
            f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
            f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
        )

    def _expr_str(self, value: object) -> str:
        if isinstance(value, (int, torch.SymInt, sympy.Expr)):
            return self.fn.sympy_expr(_to_sympy(value))
        return ast.unparse(self._to_ast(value))

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        elif (
            isinstance(pids, FlatProgramIDs)
            and env.backend.name == "pallas"
            and len(block_ids) >= 2
        ):
            pids = XYZProgramIDs()

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)
        steps = self._root_grid_steps(state)

        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for i, (block_idx, block_size, begin, end, step) in enumerate(
            reversed(
                self._reorder(
                    [*zip(block_ids, block_sizes, begins, ends, steps, strict=True)]
                )
            )
        ):
            numel = self._range_numel_expr(begin, end, step)
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if step not in (None, 1):
                step_ast = self._to_ast(step, to_dtype=dtype)
                # CuTe DSL preprocessor reserves ``step_<counter>`` (see comment
                # in ``TileStrategy.__init__``) — the backend renames the lifted
                # step var to avoid the UnboundLocalError that drove the offset
                # rename.
                _offset_prefix, step_prefix = env.backend.loop_var_prefixes()
                step_var = state.codegen.lift(step_ast, dce=True, prefix=step_prefix).id
                block_size_var = "1"
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}({pid_var}) * {step_var}"
                )
            elif block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")
            axis = thread_axis_offset + thread_axis_map[block_idx]
            # Inactive block_ids never claim a CUDA thread axis (per
            # ``_thread_axis_map``); without the polymorphic
            # ``_uses_thread_axis_for_block`` hook the grid would emit
            # ``thread_idx[axis]`` for them and collide with the inner
            # device-loop on the same axis.
            uses_thread_axis = step in (
                None,
                1,
            ) and self._uses_thread_axis_for_block(block_idx, block_size)
            bs = block_size_var if uses_thread_axis else "1"
            idx_expr = env.backend.grid_index_expr(offset_var, bs, dtype, axis=axis)
            if uses_thread_axis and isinstance(block_size, int):
                tracker.record(block_idx, axis, block_size)
            elif uses_thread_axis:
                tracker.record_symbolic_axis([block_idx], axis)
            state.add_statement(f"{index_var} = {idx_expr}")
            if uses_thread_axis and isinstance(block_size, int):
                env.backend.record_launch_thread_extent(
                    self.fn, index_var, axis, block_size
                )
                if env.config_spec.pointwise_facts and not (
                    _cute_epilogue_subtile_active(self.fn.config)
                ):
                    # The non-lane ND path must retain the same producer-axis
                    # evidence as PerThreadNDTileStrategy's lane-loop path.
                    env.backend.record_grid_thread_extent(
                        self.fn, index_var, axis, block_size
                    )
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state,
                block_idx,
                block_size,
                index_var,
                end,
                thread_axis=axis if uses_thread_axis else None,
                block_size_var=bs if uses_thread_axis else None,
            )
            if mask_statement is not None:
                state.add_statement(mask_statement)
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def _to_ast(self, x: object, to_dtype: str | None = None) -> ast.AST:
        if isinstance(x, ast.AST):
            if to_dtype:
                cast_expr = CompileEnvironment.current().backend.ast_to_dtype_expr(
                    "{value}", to_dtype
                )
                return expr_from_string(cast_expr, value=x)
            return x
        if isinstance(x, int):
            return expr_from_string(repr(x))
        if isinstance(x, sympy.Expr):
            from .device_function import DeviceFunction

            return expr_from_string(DeviceFunction.current().sympy_expr(x))
        if isinstance(x, torch.SymInt):
            return self._to_ast(x._sympy_())
        if isinstance(x, torch.Tensor):
            # Handle tensor values (for data-dependent bounds)
            # For scalar tensors, we need to load the value using tl.load
            from .device_function import DeviceFunction

            tensor_arg = DeviceFunction.current().tensor_arg(x)
            return expr_from_string(
                CompileEnvironment.current().backend.scalar_load_expr(tensor_arg.name)
            )
        if isinstance(x, str):
            # Already a string expression (for data-dependent numel)
            return expr_from_string(x)
        raise NotImplementedError(f"{type(x)} is not implemented.")

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        # TODO(jansel): refactor this to share code with codegen_grid
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        dtype = env.index_type()
        block_sizes = self.block_size
        body = innermost_body = []
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        if len(state.ast_args) == 5:
            _, begins, ends, _, steps = state.ast_args
        else:
            _, begins, ends, _ = state.ast_args
            steps = None
        _, _, proxy_ends, *_ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        if steps is None:
            steps = [None] * len(block_ids)
        assert isinstance(steps, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for block_idx, block_size, begin, end, step, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, steps, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if step in (None, 1) and block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            begin_var_name = state.codegen.lift(
                self._to_ast(begin, to_dtype=dtype), dce=True, prefix="begin"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                begin_var_name=begin_var_name,
                begin_expr=_to_sympy(begin)
                if isinstance(begin, (int, torch.SymInt))
                else None,
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            # When the backend uses Python range() (e.g. Pallas), range
            # bounds must be plain Python ints — skip the dtype cast so
            # that concrete values stay as ints and are not wrapped in
            # backend-traced dtype conversions.
            range_dtype = None if env.backend.range_requires_python_int else dtype
            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=(
                            ast.unparse(self._to_ast(step, to_dtype=range_dtype))
                            if step not in (None, 1)
                            else block_size_var
                        ),
                    ),
                    begin=self._to_ast(begin, to_dtype=range_dtype),
                    end=self._to_ast(end, to_dtype=range_dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            assert for_node.body is body
            # Inactive block_ids never claim a CUDA thread axis (per
            # ``_thread_axis_map``); see ``codegen_grid`` above for the
            # collision this guards against.
            uses_thread_axis = step in (
                None,
                1,
            ) and self._uses_thread_axis_for_block(block_idx, block_size)
            axis = thread_axis_offset + thread_axis_map[block_idx]
            bs = block_size_var if uses_thread_axis else "1"
            idx_expr = env.backend.loop_index_expr(offset_var, bs, dtype, axis=axis)
            if uses_thread_axis and isinstance(block_size, int):
                tracker.record(block_idx, axis, block_size)
            elif uses_thread_axis:
                tracker.record_symbolic_axis([block_idx], axis)
            extra_body = [
                statement_from_string(f"{index_var} = {idx_expr}"),
            ]
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state,
                block_idx,
                block_size,
                index_var,
                end,
                thread_axis=axis if uses_thread_axis else None,
                block_size_var=bs if uses_thread_axis else None,
            )
            if mask_statement is not None:
                extra_body.append(mask_statement)
            # pyrefly: ignore [unsupported-operation]
            body[:] = [*extra_body, *body]
            body = [for_node]
        assert for_node is not None
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=innermost_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # TODO(jansel): we should combine size==1 dimensions here
        return shapes


class NDTileStrategy(_BaseNDTileStrategy):
    """Do up to 3D tiling using the kernel grid."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self.mask_vars: dict[int, str | None] = {}
        self.l2_grouping = l2_grouping

    def mask_var(self, block_idx: int) -> str | None:
        return self.mask_vars[block_idx]

    def _setup_mask(
        self,
        state: CodegenState,
        block_idx: int,
        block_size: SymIntLike,
        index_var: str,
        end: object,
        *,
        thread_axis: int | None = None,
        block_size_var: str | None = None,
    ) -> ast.stmt | None:
        """Build the bounds mask for one tile axis."""
        env = CompileEnvironment.current()
        if (
            not env.backend.force_tile_mask()
            and env.block_sizes[block_idx].known_multiple(block_size)
            and not env.is_jagged_tile(block_idx)
            and not (
                env.backend.launches_surplus_tile_threads()
                and self.fn.tile_strategy.has_surplus_threads_for_block_id(block_idx)
            )
        ):
            self.mask_vars[block_idx] = None
            return None
        self.mask_vars[block_idx] = mask_var = self.fn.new_var(
            f"mask_{block_idx}", dce=True
        )

        if env.is_jagged_tile(block_idx):
            jagged_tile_parents_ast = state.ast_args[3]
            jagged_tile_parents_proxy = state.proxy_args[3]
            assert isinstance(jagged_tile_parents_ast, list)
            assert isinstance(jagged_tile_parents_proxy, list)
            # We guarantee the first lifted loop input is the jagged_tile parent tensor.
            jagged_tile_parent = jagged_tile_parents_ast[0]
            jagged_tile_block_size = env.block_sizes[block_idx].var
            jagged_tile_parent_proxy = jagged_tile_parents_proxy[0]
            assert isinstance(jagged_tile_parent_proxy, torch.Tensor)
            parent_dims: list[torch.SymInt] = []
            for d in jagged_tile_parent_proxy.size():
                assert isinstance(d, torch.SymInt)
                parent_dims.append(d)
            assert len(parent_dims) >= 1
            env.jagged_tile_mask_shapes[block_idx] = [
                *parent_dims,
                jagged_tile_block_size,
            ]
            if not self.supports_index_rank_expansion():
                tile_mask = None
                if (
                    thread_axis is not None
                    and block_size_var is not None
                    and env.backend.launches_surplus_tile_threads()
                    and self.fn.tile_strategy.has_surplus_threads_for_block_id(
                        block_idx
                    )
                ):
                    tile_mask = env.backend.thread_in_tile_mask_expr(
                        block_size_var, axis=thread_axis
                    )
                prefix = f"({tile_mask}) and " if tile_mask is not None else ""
                return statement_from_string(
                    f"{mask_var} = {prefix}({index_var}) < {{parent}}",
                    parent=self._to_ast(jagged_tile_parent),
                )
            k = len(parent_dims)
            child_expand = "[" + ", ".join(["None"] * k + [":"]) + "]"
            parent_expand = "[" + ", ".join([":"] * k + ["None"]) + "]"
            return statement_from_string(
                f"{mask_var} = ({index_var}){child_expand} < {{parent}}{parent_expand}",
                parent=self._to_ast(jagged_tile_parent),
            )

        mask_terms = [f"({index_var}) < {{end}}"]
        if (
            thread_axis is not None
            and block_size_var is not None
            and env.backend.launches_surplus_tile_threads()
        ):
            thread_mask = env.backend.thread_in_tile_mask_expr(
                block_size_var, axis=thread_axis
            )
            if thread_mask is not None:
                mask_terms.insert(0, f"({thread_mask})")
        return statement_from_string(
            f"{mask_var} = {' and '.join(mask_terms)}", end=self._to_ast(end)
        )

    def select_pid_strategy(self) -> ProgramIDs:
        if self.l2_grouping > 1:
            return L2GroupingProgramIDs(
                group_size=self.l2_grouping,
                parent_strategy=super().select_pid_strategy(),
            )
        return super().select_pid_strategy()


class _LaneContextOwner:
    """A per-thread tile strategy's view of its blocks' ``LaneLoopContext``s.

    The contexts live on the device function by block id
    (``DeviceFunction.lane_contexts``); the strategy registers one per block
    when it is built and fills it in as it opens the lane loops.
    """

    fn: DeviceFunction
    block_ids: list[int]

    def lane_context(self, block_id: int) -> LaneLoopContext:
        return self.fn.lane_contexts[block_id]

    def lane_contexts_with_vec_stores(self) -> list[LaneLoopContext]:
        """The blocks holding collected vector stores, in the order their
        first store was registered (the order the demotion decides them in)."""
        contexts = [
            context
            for context in map(self.lane_context, self.block_ids)
            if context.vec_stores
        ]
        return sorted(contexts, key=lambda context: context.first_vec_store_serial or 0)


class PerThreadNDTileStrategy(_LaneContextOwner, NDTileStrategy):
    """N-D tiling for backends that give each thread one element of the tile.

    Adds, on top of :class:`NDTileStrategy`, the per-axis ``num_threads`` split
    and the lane loop it implies: an axis served by fewer threads than its
    block size has each thread walk ``block_size // num_threads`` elements.
    """

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
        num_threads: list[int] | None = None,
        mma_mode: bool = False,
        inactive_block_ids: set[int] | None = None,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order, l2_grouping)
        assert isinstance(block_size, list)
        if num_threads is None:
            num_threads = [0 for _ in block_ids]
        assert len(num_threads) == len(block_ids)
        self.num_threads = num_threads
        self._shared_thread_extents: dict[int, int] = {}
        self.mma_mode = mma_mode
        self.inactive_block_ids = inactive_block_ids or set()
        self._lane_var_by_block: dict[int, str] = {}
        # ``cute_tiles_per_block``: grid axis -> tiles packed per CTA (its
        # thread axis extent); see ``cute/tile_packing.py``.  Resolved here,
        # before layout planning, so the planner sees the packed axis as a
        # multi-row CTA and can order it above the inner tile axes (one warp
        # per tile when the inner extent is a warp).
        self._cute_pack_by_block: dict[int, int] = {}
        if not mma_mode:
            packed = CompileEnvironment.current().backend.packed_grid_block(fn.config)
            if packed is not None and packed[0] in block_ids:
                self._cute_pack_by_block[packed[0]] = packed[1]
                # The packed axis is a thread axis whether or not the body
                # observes the tile's elements: the pid itself varies per
                # sub-group.  A one-element axis used only through
                # ``tile.id`` / ``tile.begin`` arrives here inactive (no graph
                # distributes over it), which would otherwise leave the pid
                # reading a thread axis the launch never raises (or one the
                # next block owns).
                self.inactive_block_ids = self.inactive_block_ids - {packed[0]}
        # One ``LaneLoopContext`` per block, registered on the device
        # function by block id: the vector width of the lane
        # loop (1 = scalar) from the autotuner-selected ``cute_vector_widths``
        # when the block has a lane loop and its ``elements_per_thread`` is
        # divisible by the picked V (``codegen_device_loop`` then partitions
        # the lane loop into outer (epT/V) x inner constexpr V so memory_ops
        # can hoist a single ``cute.arch.load(..., V)`` per outer-lane iter),
        # the V-loop and lane body the memory emitters splice packets into,
        # and the thread-block-cluster split of the axis (``cute_cluster_n``:
        # the axis's elements are divided across ``cl`` cluster CTAs in
        # addition to the CTA's own threads, applied only to a single
        # lane-looped axis when the tile covers the full extent and all
        # sibling axes have thread extent 1; see ``_maybe_apply_cluster_split``).
        for block_id in block_ids:
            fn.register_lane(block_id)
        # Per-block lane layout ("blocked" | "strided") from the
        # ``cute_lane_layouts`` config knob.  Only differs from blocked
        # when the lane loop has more than one iteration.
        self._cute_lane_layout_by_block: dict[int, str] = {}
        # Lazy one-shot guard for ``_maybe_apply_cluster_split``.
        self._cute_cluster_checked: bool = False
        if not mma_mode:
            env_local = CompileEnvironment.current()
            cute_vec_widths_cfg = cast(
                "list[int]",
                fn.config.config.get("cute_vector_widths", []) or [],
            )
            cute_lane_layouts_cfg = cast(
                "list[str]",
                fn.config.config.get("cute_lane_layouts", []) or [],
            )
            for block_id, nt, bs in zip(
                block_ids, num_threads, block_size, strict=True
            ):
                if block_id in self.inactive_block_ids:
                    continue
                static_bs = self._configured_block_size_int(bs)
                if (
                    nt > 0
                    and static_bs is not None
                    and static_bs > nt
                    and static_bs % nt == 0
                ):
                    self._lane_var_by_block[block_id] = self.fn.new_var(
                        f"lane_{block_id}"
                    )
                    if (
                        block_id
                        in env_local.config_spec.cute_lane_layouts.valid_block_ids()
                    ):
                        layout = env_local.config_spec.cute_lane_layouts.config_get(
                            cute_lane_layouts_cfg, block_id, "blocked"
                        )
                        if isinstance(layout, str):
                            self._cute_lane_layout_by_block[block_id] = layout
                    elements_per_thread = static_bs // nt
                    # Vec slot is registered eagerly in device-IR analysis; read
                    # the tuned V.  Never append here — growing the spec during
                    # codegen breaks the autotuner's fixed-width unflatten.
                    if (
                        block_id
                        in env_local.config_spec.cute_vector_widths.valid_block_ids()
                    ):
                        vec_width = env_local.config_spec.cute_vector_widths.config_get(
                            cute_vec_widths_cfg,
                            block_id,
                            1,
                        )
                        if (
                            isinstance(vec_width, int)
                            and vec_width > 1
                            and elements_per_thread % vec_width == 0
                        ):
                            self.lane_context(block_id).vec_width = vec_width
                    else:
                        # Metal shares this strategy without vec-width tuning,
                        # and non-static sizes are eager-skipped — both run
                        # scalar.  A missing static slot on a backend tuning
                        # the widths is a bug.
                        assert (
                            not env_local.backend.tunes_lane_vector_widths()
                            or not isinstance(
                                env_local.block_sizes[block_id].size,
                                (int, torch.SymInt),
                            )
                        ), (
                            f"cute_vector_widths slot missing for static-size "
                            f"block_id={block_id}; it must be registered during "
                            f"device-IR analysis, not lazily during codegen"
                        )

    def _configured_block_size_int(self, block_size: SymIntLike) -> int | None:
        if isinstance(block_size, int):
            return block_size
        env = CompileEnvironment.current()
        resolved_block_id = env.resolve_block_id(block_size)
        if resolved_block_id is not None:
            configured_size = self.fn.resolved_block_size(resolved_block_id)
            if isinstance(configured_size, int):
                return configured_size
        block_size_expr = _to_sympy(block_size)
        block_size_expr = env.specialize_expr(block_size_expr)
        if getattr(block_size_expr, "free_symbols", None):
            return None
        return int(block_size_expr)

    def use_shared_thread_extents(self, extents: dict[int, int]) -> None:
        """Fit SIMT lane loops to the physical extents shared by sibling loops."""
        for block_id, threads in extents.items():
            idx = self.block_ids.index(block_id)
            size = self._configured_block_size_int(self.block_size[idx])
            if size is None or (size > threads and size % threads != 0):
                raise CompileEnvironment.current().backend.unsupported(
                    "a shared tile thread axis requires an evenly partitioned tile"
                )
            self._set_shared_thread_extent(block_id, size, threads)

    def _set_shared_thread_extent(self, block_id: int, size: int, threads: int) -> None:
        self._shared_thread_extents[block_id] = threads
        elements_per_thread = max(1, size // threads)
        # Collective caps can narrow a previously scalar tile into lane loops.
        if elements_per_thread > 1 and block_id not in self._lane_var_by_block:
            self._lane_var_by_block[block_id] = self.fn.new_var(f"lane_{block_id}")
        if elements_per_thread == 1:
            self._lane_var_by_block.pop(block_id, None)
        context = self.lane_context(block_id)
        if elements_per_thread % context.vec_width:
            context.vec_width = 1

    def thread_extent_for_masking(self, block_id: int, thread_extent: int) -> int:
        """Tile lanes that can own an element when a sibling widens the launch."""
        if block_id in self._shared_thread_extents:
            size = self._configured_block_size_int(
                self.block_size[self.block_ids.index(block_id)]
            )
            assert size is not None
            return min(thread_extent, size)
        return thread_extent

    def _demote_blocks_beyond_thread_axes(self, state: CodegenState) -> None:
        """Serve blocks that would land on CUDA thread axis >= 3 with lanes.

        A launch only has x/y/z thread axes. Once the enclosing loops and
        reductions have claimed them (``_thread_axis_offset``), every further
        block of this strategy is demoted to what ``num_threads=1`` means:
        thread extent 1 and a single thread's lane loop walking the whole
        tile. This only changes the thread layout, never the elements the tile
        covers, so it is safe to decide per call site at codegen time.
        """
        backend = CompileEnvironment.current().backend
        axis_limit = backend.thread_axis_limit()
        if self.mma_mode or axis_limit is None:
            return
        axis = self._thread_axis_offset(state)
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            block_size = block_size_by_id[block_id]
            if not self._uses_thread_axis_for_block(block_id, block_size):
                continue
            if axis < axis_limit:
                axis += 1
                continue
            size = self._configured_block_size_int(block_size)
            if size is None:
                raise backend.unsupported(
                    f"thread axis {axis}: block {block_id} needs a static block "
                    "size to run as a lane loop"
                )
            self._set_shared_thread_extent(block_id, size, 1)

    def _maybe_apply_cluster_split(
        self, env: CompileEnvironment, state: CodegenState
    ) -> None:
        """Ask the backend once whether a cluster split applies to this loop
        (``Backend.apply_lane_cluster_split``; invoked lazily on the first
        ``codegen_device_loop`` call so the active grid state is visible)."""
        if self._cute_cluster_checked:
            return
        self._cute_cluster_checked = True
        if self.mma_mode:
            return
        env.backend.apply_lane_cluster_split(self, state)

    def _elements_per_thread_for_block(self, block_id: int) -> int:
        """Elements per thread for *block_id* (derived from num_threads and
        the cluster split)."""
        if block_id in self.inactive_block_ids:
            return 1
        if block_id in self._cute_pack_by_block:
            return 1  # every thread sub-group owns one tile of one element
        idx = self.block_ids.index(block_id)
        nt = self._shared_thread_extents.get(block_id, self.num_threads[idx])
        if nt == 0:
            return 1
        bs = self._configured_block_size_int(self.block_size[idx])
        assert isinstance(bs, int)  # validated by _thread_extent_for_axis
        return max(1, bs // nt) // self.lane_context(block_id).cluster_n

    def _thread_extent_for_axis(
        self, block_id: int, block_size: SymIntLike
    ) -> SymIntLike:
        if block_id in self.inactive_block_ids:
            return 1
        if self.mma_mode:
            return 1  # MMA handles element distribution, no CUDA threads needed
        if block_id in self._cute_pack_by_block:
            return self._cute_pack_by_block[block_id]
        idx = self.block_ids.index(block_id)
        nt = self._shared_thread_extents.get(block_id, self.num_threads[idx])
        if nt == 0:
            return block_size
        backend_name = CompileEnvironment.current().backend.name
        resolved_block_size = block_size
        if not isinstance(resolved_block_size, int):
            static_block_size = self._configured_block_size_int(resolved_block_size)
            if static_block_size is None:
                raise exc.BackendUnsupported(
                    backend_name,
                    f"num_threads requires static ND block sizes for {backend_name}",
                )
            resolved_block_size = static_block_size
        if resolved_block_size % nt != 0 and not (
            (block_id in self._shared_thread_extents and resolved_block_size < nt)
            or CompileEnvironment.current().backend.collective_owns_tile(
                self.fn, block_id
            )
        ):
            raise exc.BackendUnsupported(
                backend_name,
                (
                    f"block size must be divisible by num_threads for "
                    f"{backend_name} axis "
                    f"{block_id}: {resolved_block_size} is not divisible by {nt}"
                ),
            )
        return nt

    def _uses_thread_axis_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> bool:
        if block_id in self.inactive_block_ids:
            return False
        thread_extent = self._thread_extent_for_axis(block_id, block_size)
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def _thread_axis_map(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis_for_block(block_id, block_size_by_id[block_id]):
                axis += 1
        return mapping

    def thread_axes_used(self) -> int:
        return sum(
            1
            for block_idx, block_size in zip(
                self.block_ids, self.block_size, strict=True
            )
            if self._uses_thread_axis_for_block(block_idx, block_size)
        )

    def _static_thread_extent_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> int | None:
        thread_extent = self._thread_extent_for_axis(block_id, block_size)
        if isinstance(thread_extent, int):
            return thread_extent
        return self._configured_block_size_int(thread_extent)

    def cute_tile_base_expr(self, block_id: int) -> str | None:
        """Uniform expression of the tile's first index along ``block_id``.

        The per-element index is ``base + <thread / lane partition>``, so
        ``index - base`` is the block-local position of the current element.
        """
        if self.mma_mode or block_id not in self.block_ids:
            return None
        return self.offset_var(block_id)

    def cute_lane_axis(self, block_id: int) -> CuteLaneAxis | None:
        """Static thread / lane distribution of ``block_id``, or ``None``.

        ``None`` when the axis has no plain per-thread partition: MMA mode, an
        inactive or cluster-split block, a dynamic extent, or a thread count
        that does not tile the extent exactly.
        """
        if (
            self.mma_mode
            or block_id not in self.block_ids
            or block_id in self.inactive_block_ids
            or self.lane_context(block_id).cluster_n > 1
        ):
            return None
        block_size = self.block_size[self.block_ids.index(block_id)]
        extent = self._configured_block_size_int(block_size)
        threads = self._static_thread_extent_for_block(block_id, block_size)
        if extent is None or threads is None or threads <= 0:
            return None
        elements_per_thread = self._elements_per_thread_for_block(block_id)
        if threads * elements_per_thread != extent:
            return None
        lane_var = self._lane_var_by_block.get(block_id)
        context = self.lane_context(block_id)
        vec_lane_var = context.vec_lane_var
        vec_width = context.vec_width if vec_lane_var is not None else 1
        if lane_var is None:
            if elements_per_thread != 1:
                return None
            lane_steps = 1
        elif elements_per_thread % vec_width:
            return None
        else:
            lane_steps = elements_per_thread // vec_width
        return CuteLaneAxis(
            extent=extent,
            threads=threads,
            lane_var=lane_var,
            lane_steps=lane_steps,
            vec_lane_var=vec_lane_var,
            vec_width=vec_width,
            strided=(
                lane_var is not None
                and threads > 1
                and self._cute_lane_layout_by_block.get(block_id, "blocked")
                == "strided"
            ),
        )

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            thread_extent = self._thread_extent_for_axis(
                block_id, block_size_by_id[block_id]
            )
            if self._uses_thread_axis_for_block(block_id, block_size_by_id[block_id]):
                static_extent = thread_extent
                if not isinstance(static_extent, int):
                    static_extent = self._configured_block_size_int(static_extent)
                if isinstance(static_extent, int):
                    sizes.append(static_extent)
        return sizes

    def thread_block_size_exprs(self) -> list[str]:
        exprs: list[str] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if not self._uses_thread_axis_for_block(block_id, bs):
                continue
            thread_extent = self._thread_extent_for_axis(block_id, bs)
            if isinstance(thread_extent, int):
                exprs.append(str(thread_extent))
                continue
            if not isinstance(bs, torch.SymInt):
                return []
            bs_var = self.block_size_var(block_id)
            if bs_var is None:
                return []
            elements_per_thread = self._elements_per_thread_for_block(block_id)
            if elements_per_thread == 1:
                exprs.append(bs_var)
            else:
                exprs.append(f"({bs_var}) // {elements_per_thread}")
        return exprs

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        self._demote_blocks_beyond_thread_axes(state)
        env = CompileEnvironment.current()
        # ``cute_tiles_per_block``: a packed grid axis is a thread axis of
        # this strategy even when no lane loop is needed.
        if (
            not self._lane_var_by_block
            and not self._shared_thread_extents
            and not self._cute_pack_by_block
        ):
            return super().codegen_grid(state)

        block_ids = self.block_ids
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)
        steps = self._root_grid_steps(state)

        lane_setup_statements: list[ast.AST] = []
        outer_setup_statements: list[ast.AST] = []
        vec_wrappers: dict[str, VecLaneWrapper] = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        for i, (block_idx, block_size, begin, end, step) in enumerate(
            reversed(
                self._reorder(
                    [*zip(block_ids, block_sizes, begins, ends, steps, strict=True)]
                )
            )
        ):
            numel = self._range_numel_expr(begin, end, step)
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)
            pack = self._cute_pack_by_block.get(block_idx, 1)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if step not in (None, 1):
                step_ast = self._to_ast(step, to_dtype=dtype)
                # CuTe DSL preprocessor reserves ``step_<counter>`` (see comment
                # in ``TileStrategy.__init__``) — the backend renames the lifted
                # step var to avoid the UnboundLocalError that drove the offset
                # rename.
                _offset_prefix, step_prefix = env.backend.loop_var_prefixes()
                step_var = state.codegen.lift(step_ast, dce=True, prefix=step_prefix).id
                block_size_var = "1"
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}({pid_var}) * {step_var}"
                )
            elif block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")

            elements_per_thread = self._elements_per_thread_for_block(block_idx)
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis_for_block(
                block_idx, block_size
            )
            axis = thread_axis_offset + thread_axis_map[block_idx]
            static_extent: int | None = None
            if uses_thread_axis:
                idx_expr = env.backend.lane_index_expr(
                    offset_var, elements_per_thread, axis=axis
                )
                if pack > 1:
                    # The pid already carries this sub-group's tile.
                    idx_expr = offset_var
                thread_extent = self._thread_extent_for_axis(block_idx, block_size)
                static_extent = (
                    thread_extent
                    if isinstance(thread_extent, int)
                    else self._static_thread_extent_for_block(block_idx, block_size)
                )
                if isinstance(static_extent, int):
                    tracker.record(block_idx, axis, static_extent)
                else:
                    tracker.record_symbolic_axis([block_idx], axis)
            else:
                idx_expr = offset_var
            if lane_var := self._lane_var_by_block.get(block_idx):
                vec_width = self.lane_context(block_idx).vec_width
                lane_strided = (
                    self._cute_lane_layout_by_block.get(block_idx, "blocked")
                    == "strided"
                    and uses_thread_axis
                    and isinstance(static_extent, int)
                )
                if (
                    vec_width > 1
                    and uses_thread_axis
                    and elements_per_thread % vec_width == 0
                    # A matmul fallback may suppress root lane loops AFTER
                    # loads already hoisted into the wrapper (cute_mma's
                    # request_root_lane_loop_suppression), which would strand
                    # the hoists; keep grid vec to matmul-free kernels.
                    and (
                        not env.config_spec.matmul_facts
                        or set(block_ids).issubset(
                            env.config_spec.cute_pointwise_region_block_ids
                        )
                    )
                    # Epilogue subtiling stages stores through smem with a
                    # sync_threads INSIDE the per-element pipeline, which the
                    # vec store-collection protocol silently corrupts (50%
                    # wrong output measured); keep such configs on the
                    # (correct) scalar form.
                    and not _cute_epilogue_subtile_active(self.fn.config)
                ):
                    # Same outer x constexpr-V lane partition as
                    # ``codegen_device_loop`` (see the comments there), so
                    # the memory_ops vec-load/store hoist protocol applies
                    # to GRID lane loops too.  The loops themselves are
                    # materialized later by ``DeviceGridState.wrap_body``
                    # from the pre-built ``VecLaneWrapper``.
                    vec_lane_var = self.fn.new_var(f"vec_lane_{block_idx}", dce=False)
                    self.lane_context(block_idx).vec_lane_var = vec_lane_var
                    inner_for = cast(
                        "ast.For",
                        ast.parse(
                            f"for {vec_lane_var} in cutlass.range_constexpr({vec_width}):\n"
                            f"    pass"
                        ).body[0],
                    )
                    lane_body: list[ast.AST] = [inner_for]
                    self.lane_context(block_idx).lane_body = lane_body
                    self.lane_context(block_idx).vloop = inner_for
                    base_index_var = self.fn.new_var(
                        f"lane_base_{block_idx}", dce=False
                    )
                    self.lane_context(block_idx).base_index_var = base_index_var
                    if isinstance(static_extent, int):
                        env.backend.record_grid_thread_extent(
                            self.fn, base_index_var, axis, static_extent
                        )
                        env.backend.record_launch_thread_extent(
                            self.fn, base_index_var, axis, static_extent
                        )
                    if lane_strided:
                        # ``base = offset + (outer*NT + tid) * V``
                        base_expr = (
                            f"{offset_var} + "
                            f"({env.backend.lane_offset_expr(lane_var)} "
                            f"* {static_extent} + "
                            f"{env.backend.thread_index_expr(axis=axis)}) "
                            f"* {vec_width}"
                        )
                    else:
                        # ``base = offset + tid*EPT + outer*V``
                        base_expr = (
                            f"{idx_expr} + "
                            f"{env.backend.lane_offset_expr(lane_var)} "
                            f"* {vec_width}"
                        )
                    lane_body.insert(
                        0,
                        statement_from_string(f"{base_index_var} = {base_expr}"),
                    )
                    outer_for = _create_lane_loop(
                        lane_var, elements_per_thread // vec_width, lane_body
                    )
                    vec_wrappers[lane_var] = VecLaneWrapper(
                        outer_for=outer_for,
                        vloop=inner_for,
                        vec_lane_var=vec_lane_var,
                        base_index_var=base_index_var,
                    )
                    idx_expr = f"{base_index_var} + cutlass.Int32({vec_lane_var})"
                else:
                    if (
                        lane_strided
                        and not env.config_spec.matmul_facts
                        and not _cute_epilogue_subtile_active(self.fn.config)
                    ):
                        # Collective matmul/subtile lowerings may replace the
                        # lane body later; retain their established mapping.
                        idx_expr = (
                            f"{offset_var} + {env.backend.thread_index_expr(axis=axis)}"
                            f" + {env.backend.lane_offset_expr(lane_var)} * {static_extent}"
                        )
                    else:
                        idx_expr = (
                            f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)}"
                        )
                target = lane_setup_statements
            else:
                # Setup that does not depend on a lane variable can be hoisted
                # out of the lane loops. This avoids reassignments inside the
                # lane-loop body that confuse the CuTe DSL preprocessor when
                # its internal negative-step machinery emits identifiers like
                # ``offset_<n>`` that collide with helion's tile offsets.
                target = outer_setup_statements
            target.append(statement_from_string(f"{index_var} = {idx_expr}"))

            if isinstance(static_extent, int):
                env.backend.record_launch_thread_extent(
                    self.fn, index_var, axis, static_extent
                )
                if pack > 1:
                    env.backend.record_launch_thread_extent(
                        self.fn, offset_var, axis, static_extent
                    )
                if not env.config_spec.matmul_facts and not (
                    _cute_epilogue_subtile_active(self.fn.config)
                ):
                    env.backend.record_grid_thread_extent(
                        self.fn, index_var, axis, static_extent
                    )
                    if pack > 1:
                        # The body may only name the tile begin
                        # (``tile.begin``); the launch must still raise the
                        # sub-group axis.
                        env.backend.record_grid_thread_extent(
                            self.fn, offset_var, axis, static_extent
                        )

            # Bound by the *thread* extent rather than the block size: this
            # axis advances ``elements_per_thread`` per thread, so the threads
            # belonging to the tile number block_size / that stride.
            mask_statement = self._setup_mask(
                state,
                block_idx,
                block_size,
                index_var,
                end,
                thread_axis=axis if isinstance(static_extent, int) else None,
                block_size_var=(
                    str(self.thread_extent_for_masking(block_idx, static_extent))
                    if isinstance(static_extent, int)
                    else None
                ),
            )
            if mask_statement is not None:
                target.append(mask_statement)
            wrapper = vec_wrappers.get(self._lane_var_by_block.get(block_idx, ""))
            if wrapper is not None:
                env.backend.record_vloop_wrapper(
                    self.fn,
                    wrapper,
                    block_id=block_idx,
                    lane_var=self._lane_var_by_block[block_idx],
                    index_var=index_var,
                    mask_var=self.mask_vars.get(block_idx),
                    begin=begin,
                    extent=numel,
                    step=None,
                )
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            if pack > 1:
                pid = pid._replace(pack=pack, pack_axis=axis)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        lane_loops = [
            (
                self._lane_var_by_block[block_id],
                self._elements_per_thread_for_block(block_id),
            )
            for block_id in (self.block_ids[i] for i in self.loop_order)
            if block_id in self._lane_var_by_block
        ]
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_loop_blocks=set(self._lane_var_by_block),
            lane_loop_block_ids={
                lane_var: frozenset(
                    block_id
                    for block_id, candidate in self._lane_var_by_block.items()
                    if candidate == lane_var
                )
                for lane_var, _extent in lane_loops
            },
            lane_setup_statements=lane_setup_statements,
            tile_masks=frozenset(
                mask
                for block_id in block_ids
                if (mask := self.mask_vars.get(block_id)) is not None
            ),
            outer_prefix=outer_setup_statements,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            vec_lane_wrappers=vec_wrappers,
        )

    def _lane_nesting_order(self, state: CodegenState) -> list[int]:
        """Block ids of this device loop's serial lanes, outermost first.

        A thread visits the same elements in any lane nesting, so follow
        ``loop_order`` except that a lane whose block is reduced in the loop
        body nests inside the others: the lane reduction lowering finalizes a
        reduction in its owner lane, which must be the innermost rolled lane
        (``validate_lane_reduce_owners``).
        """
        from .inductor_lowering import ReductionLowering

        env = CompileEnvironment.current()
        graph_info = getattr(state.codegen, "_cute_active_graph_info", None)
        reduced = (
            set()
            if graph_info is None
            else {
                env.canonical_block_id(lowering.block_index)
                for node in graph_info.graph.nodes
                if isinstance(lowering := node.meta.get("lowering"), ReductionLowering)
            }
        )
        return sorted(
            (
                block_id
                for block_id in (self.block_ids[i] for i in self.loop_order)
                if block_id in self._lane_var_by_block
            ),
            key=lambda block_id: env.canonical_block_id(block_id) in reduced,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        self._demote_blocks_beyond_thread_axes(state)
        if (
            not self._lane_var_by_block
            and not self._shared_thread_extents
            and not self.mma_mode
        ):
            return super().codegen_device_loop(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
        self._maybe_apply_cluster_split(env, state)
        dtype = env.index_type()
        block_sizes = self.block_size
        user_body: list[ast.AST] = []
        body: list[ast.AST] = user_body
        # Capture per-block (lane_var, full_extent, vec_width).  When
        # vec_width > 1, the outer lane runs (full_extent // vec_width)
        # iters; the inner constexpr-V loop handles V elements per outer
        # iter.  The memory_ops vec-load dispatcher splices a single
        # ``cute.arch.load(..., V)`` between the outer lane setup and the
        # inner constexpr loop, so per-thread bytes-per-load grow from
        # ``sizeof(dtype)`` to ``V * sizeof(dtype)`` (LDG.64 / LDG.128).
        lane_loops_meta: list[tuple[int, str, int, int]] = []
        for block_id in self._lane_nesting_order(state):
            lane_var = self._lane_var_by_block[block_id]
            extent = self._elements_per_thread_for_block(block_id)
            vec_width = self.lane_context(block_id).vec_width
            lane_loops_meta.append((block_id, lane_var, extent, vec_width))
        outer_for_by_block: dict[int, ast.For] = {}
        vloop_blocks: list[int] = []
        for block_id, lane_var, extent, vec_width in reversed(lane_loops_meta):
            if vec_width > 1 and extent > 0 and extent % vec_width == 0:
                vloop_blocks.append(block_id)
                # Partition the lane loop into outer x inner constexpr V.
                # The inner constexpr-V loop's body re-runs the user body
                # for each of the V lanes (the user body's per-lane
                # ``index_var = ...`` setup keys off the COMPOSITE lane =
                # outer*V + inner so the per-element index is correct).
                vec_lane_var = self.fn.new_var(f"vec_lane_{block_id}", dce=False)
                self.lane_context(block_id).vec_lane_var = vec_lane_var
                inner_for = cast(
                    "ast.For",
                    ast.parse(
                        f"for {vec_lane_var} in cutlass.range_constexpr({vec_width}):\n"
                        f"    pass"
                    ).body[0],
                )
                inner_for.body = body  # type: ignore[assignment]
                # ``lane_body`` is what's INSIDE the outer lane loop:
                # statements above the constexpr V-loop, plus the loop
                # itself (last entry).  memory_ops.py splices a hoisted
                # ``cute.arch.load(..., V)`` into ``lane_body[-1:]`` via
                # the same protocol ``LoopedReductionStrategy`` uses.
                lane_body: list[ast.AST] = [inner_for]
                self.lane_context(block_id).lane_body = lane_body
                self.lane_context(block_id).vloop = inner_for
                outer_extent = extent // vec_width
                # Always emit the outer lane loop even when ``outer_extent
                # == 1`` (i.e. EPT == V): the lane-base index expression
                # references ``lane_var``, which must be defined in scope.
                # The CuTe DSL constant-folds the 1-iter loop away.
                outer_for = _create_lane_loop(lane_var, outer_extent, lane_body)
                outer_for_by_block[block_id] = outer_for
                body = [outer_for]
            else:
                lane_for = _create_lane_loop(lane_var, extent, body)
                body = [lane_for]
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        if len(state.ast_args) == 5:
            _, begins, ends, _, steps = state.ast_args
        else:
            _, begins, ends, _ = state.ast_args
            steps = None
        _, _, proxy_ends, *_ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        if steps is None:
            steps = [None] * len(block_ids)
        assert isinstance(steps, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        thread_axis_map = self._thread_axis_map()
        index_setup: list[ast.AST] = []
        vec_wrappers: dict[str, VecLaneWrapper] = {}
        for block_idx, block_size, begin, end, step, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, steps, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if step in (None, 1) and block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            begin_var_name = state.codegen.lift(
                self._to_ast(begin, to_dtype=dtype), dce=True, prefix="begin"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                begin_var_name=begin_var_name,
                begin_expr=_to_sympy(begin)
                if isinstance(begin, (int, torch.SymInt))
                else None,
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            # When the backend uses Python range() (e.g. Pallas), range
            # bounds must be plain Python ints — skip the dtype cast so
            # that concrete values stay as ints and are not wrapped in
            # backend-traced dtype conversions.
            range_dtype = None if env.backend.range_requires_python_int else dtype
            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=(
                            ast.unparse(self._to_ast(step, to_dtype=range_dtype))
                            if step not in (None, 1)
                            else block_size_var
                        ),
                    ),
                    begin=self._to_ast(begin, to_dtype=range_dtype),
                    end=self._to_ast(end, to_dtype=range_dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            elements_per_thread = self._elements_per_thread_for_block(block_idx)
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis_for_block(
                block_idx, block_size
            )
            axis = thread_axis_offset + thread_axis_map[block_idx]
            static_extent: int | None = None
            if uses_thread_axis:
                idx_expr = env.backend.lane_index_expr(
                    offset_var, elements_per_thread, axis=axis
                )
                thread_extent = self._thread_extent_for_axis(block_idx, block_size)
                static_extent = (
                    thread_extent
                    if isinstance(thread_extent, int)
                    else self._static_thread_extent_for_block(block_idx, block_size)
                )
                if isinstance(static_extent, int):
                    tracker.record(block_idx, axis, static_extent)
                else:
                    tracker.record_symbolic_axis([block_idx], axis)
            else:
                idx_expr = offset_var
            block_vec_width = self.lane_context(block_idx).vec_width
            vec_lane_var = self.lane_context(block_idx).vec_lane_var
            # Strided lane layout: thread ``t`` owns V-wide chunks strided by
            # the thread count instead of one contiguous EPT-element span, so
            # each warp instruction touches consecutive chunks (coalesced).
            lane_strided = (
                self._cute_lane_layout_by_block.get(block_idx, "blocked") == "strided"
                and uses_thread_axis
                and isinstance(static_extent, int)
            )
            # Cluster split: each of the ``cluster_n`` CTAs in the cluster
            # owns a contiguous ``block/cluster_n`` slice of the tile (the
            # launch adds a grid dim of ``cluster_n`` CTAs per outer index).
            # The slice offset folds into the tile offset; the per-CTA lane
            # layout applies within the slice unchanged
            # (``_elements_per_thread_for_block`` already divides by the
            # cluster width).
            lane_cluster_n = self.lane_context(block_idx).cluster_n
            if lane_cluster_n > 1:
                assert uses_thread_axis and isinstance(static_extent, int)
                cluster_block = self._configured_block_size_int(block_size)
                assert isinstance(cluster_block, int)
                slice_len = cluster_block // lane_cluster_n
                offset_var = (
                    f"({offset_var} + "
                    f"{env.backend.program_id_expr(1, index_dtype=dtype)}"
                    f" * {slice_len})"
                )
                idx_expr = env.backend.lane_index_expr(
                    offset_var, elements_per_thread, axis=axis
                )
            if lane_var := self._lane_var_by_block.get(block_idx):
                if block_vec_width > 1 and vec_lane_var is not None:
                    # Composite per-element lane index = outer*V + inner.
                    # outer (``lane_var``) ranges [0, EPT/V); inner
                    # (``vec_lane_var``) ranges [0, V).  Per-thread base
                    # (the start of the V-wide chunk this thread owns
                    # for this outer iter) is stashed in
                    # the block's ``LaneLoopContext`` so the vec
                    # load can use it directly (mirrors the
                    # ``LoopedReductionStrategy`` unroll path).
                    base_index_var = self.fn.new_var(
                        f"lane_base_{block_idx}", dce=False
                    )
                    self.lane_context(block_idx).base_index_var = base_index_var
                    # ``base = offset + tid*EPT + outer*V``  (per-thread
                    # V-aligned base) — emitted INSIDE the outer lane
                    # loop's body (above the constexpr V-loop) so a
                    # single ``cute.arch.load(..., V)`` can be hoisted
                    # at the same level by memory_ops.
                    context = self.lane_context(block_idx)
                    lane_body_list = context.lane_body
                    if lane_body_list is not None:
                        assert context.vloop is not None
                        if lane_strided:
                            # ``base = offset + (outer*NT + tid) * V``
                            base_expr = (
                                f"{offset_var} + "
                                f"({env.backend.lane_offset_expr(lane_var)} "
                                f"* {static_extent} + "
                                f"{env.backend.thread_index_expr(axis=axis)}) "
                                f"* {block_vec_width}"
                            )
                        else:
                            base_expr = (
                                f"{idx_expr} + "
                                f"{env.backend.lane_offset_expr(lane_var)} "
                                f"* {block_vec_width}"
                            )
                        lane_body_list.insert(
                            0,
                            statement_from_string(f"{base_index_var} = {base_expr}"),
                        )
                        vec_wrappers[lane_var] = VecLaneWrapper(
                            outer_for=outer_for_by_block[block_idx],
                            vloop=context.vloop,
                            vec_lane_var=vec_lane_var,
                            base_index_var=base_index_var,
                        )
                    # The user-body's per-element index uses the base +
                    # the inner constexpr-V var so the existing scalar
                    # pipeline (mask + cast + reduce-or-store) keeps
                    # working unchanged.
                    idx_expr = f"{base_index_var} + cutlass.Int32({vec_lane_var})"
                elif lane_strided:
                    # ``idx = offset + tid + lane * NT``
                    idx_expr = (
                        f"{offset_var} + {env.backend.thread_index_expr(axis=axis)}"
                        f" + {env.backend.lane_offset_expr(lane_var)}"
                        f" * {static_extent}"
                    )
                else:
                    idx_expr = f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)}"
            index_setup.append(statement_from_string(f"{index_var} = {idx_expr}"))
            # Same thread-extent bound as the grid lane path: a device loop is
            # equally able to be launched wider than the tile it indexes.
            mask_statement = self._setup_mask(
                state,
                block_idx,
                block_size,
                index_var,
                end,
                thread_axis=axis if isinstance(static_extent, int) else None,
                block_size_var=(
                    str(self.thread_extent_for_masking(block_idx, static_extent))
                    if isinstance(static_extent, int)
                    else None
                ),
            )
            if mask_statement is not None:
                index_setup.append(mask_statement)
            wrapper = vec_wrappers.get(self._lane_var_by_block.get(block_idx, ""))
            if wrapper is not None:
                # The device-loop twin of the grid wrapper: the tile's range
                # is this loop's.
                env.backend.record_vloop_wrapper(
                    self.fn,
                    wrapper,
                    block_id=block_idx,
                    lane_var=self._lane_var_by_block[block_idx],
                    index_var=index_var,
                    mask_var=self.mask_vars.get(block_idx),
                    begin=begin,
                    extent=proxy_end,
                    step=step,
                )
            body = [for_node]
        assert for_node is not None
        # Run index/mask setup once per loop-offset and per-lane before user body.
        user_body[:0] = index_setup
        device_loop = DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            lane_loop_blocks=set(self._lane_var_by_block),
            lane_loops=[
                (lane_var, extent)
                for _block_id, lane_var, extent, _vec_width in lane_loops_meta
            ],
            lane_loop_block_ids={
                lane_var: frozenset({block_id})
                for block_id, lane_var, _extent, _vec_width in lane_loops_meta
            },
            vec_lane_wrappers=vec_wrappers,
            lane_setup_statements=index_setup,
            tile_masks=frozenset(
                mask
                for block_id in block_ids
                if (mask := self.mask_vars.get(block_id)) is not None
            ),
        )
        if vloop_blocks:
            # A store site is collected while its V-loop body is still being
            # built; the statements that follow it are known once the loop
            # body is complete.
            device_loop.body_finalizers.append(
                lambda: env.backend.demote_loop_tile_vec_stores(self, vloop_blocks)
            )
        return device_loop

    def supports_index_rank_expansion(self) -> bool:
        return False


class PerThreadFlattenedTileStrategy(_LaneContextOwner, FlattenedTileStrategy):
    """Flattened tiling with one scalar index per thread.

    The flattened counterpart of :class:`PerThreadNDTileStrategy`: a single
    ``num_threads`` splits the flattened tile, and a lane loop covers the rest
    when it is narrower than the block size.
    """

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        num_threads: int = 0,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self._num_threads = num_threads
        self._lane_var: str | None = None
        if num_threads > 0 and isinstance(block_size, int) and num_threads < block_size:
            self._lane_var = self.new_var("lane", dce=False)
        # One ``LaneLoopContext`` per block (registered on the device
        # function's cute state by block id) for the memory_ops ``tile_unroll``
        # hoist protocol (same shape as ``PerThreadNDTileStrategy``).  The vec
        # slot lives on the LAST (stride-1) block; for the multi-block
        # (flattened N-D) form ``flat_multi`` is set and the hoist emits FLAT
        # base pointers (``t.iterator + lane_base``) -- valid only for
        # contiguous tensors covering the whole iteration space, which
        # ``_cute_vector_load_ctx`` gates per tensor.
        for block_id in block_ids:
            fn.register_lane(block_id, flat_multi=len(block_ids) > 1)
        self._cute_lane_layout: str = "blocked"
        # ``begin + pid * BLOCK`` of the lane-looped grid: the flattened index
        # var aliases its offset var, so the tile base is recorded separately.
        self._cute_tile_base_expr: str | None = None
        if self._lane_var is not None:
            env = CompileEnvironment.current()
            block_id = block_ids[-1]
            cfg = fn.config.config
            if block_id in env.config_spec.cute_lane_layouts.valid_block_ids():
                layout = env.config_spec.cute_lane_layouts.config_get(
                    cast("list[str]", cfg.get("cute_lane_layouts", []) or []),
                    block_id,
                    "blocked",
                )
                if isinstance(layout, str):
                    self._cute_lane_layout = layout
            if block_id in env.config_spec.cute_vector_widths.valid_block_ids():
                vec_width = env.config_spec.cute_vector_widths.config_get(
                    cast("list[int]", cfg.get("cute_vector_widths", []) or []),
                    block_id,
                    1,
                )
                assert isinstance(block_size, int)
                elements_per_thread = block_size // num_threads
                if (
                    isinstance(vec_width, int)
                    and vec_width > 1
                    and elements_per_thread % vec_width == 0
                ):
                    self.lane_context(block_id).vec_width = vec_width

    def _configured_block_size_int(self, block_size: SymIntLike) -> int | None:
        # Shared with the tile_unroll hoist protocol (see the identically
        # named method on PerThreadNDTileStrategy); the flattened block size
        # is already a plain int whenever the lane path is active.
        return block_size if isinstance(block_size, int) else None

    @property
    def _elements_per_thread(self) -> int:
        """Elements per thread (derived from num_threads and block_size)."""
        if self._num_threads == 0:
            return 1
        assert isinstance(self.block_size, int)
        return self.block_size // self._num_threads

    def _thread_extent(self) -> SymIntLike:
        if self._num_threads == 0:
            return self.block_size
        backend_name = CompileEnvironment.current().backend.name
        if not isinstance(self.block_size, int):
            raise exc.BackendUnsupported(
                backend_name,
                f"num_threads requires static flattened block sizes for {backend_name}",
            )
        if self.block_size % self._num_threads != 0:
            raise exc.BackendUnsupported(
                backend_name,
                (
                    f"block size must be divisible by num_threads for "
                    f"{backend_name}: "
                    f"{self.block_size} is not divisible by {self._num_threads}"
                ),
            )
        return self._num_threads

    def cute_tile_base_expr(self, block_id: int) -> str | None:
        """Uniform ``begin + pid * BLOCK`` of a single lane-looped flattened block.

        ``offset_var`` is the per-element index here (``indices = offsets``),
        so the tile base is recorded by ``codegen_grid`` instead.  ``None``
        for the multi-block form and for grids without a lane loop.
        """
        if self.block_ids != [block_id]:
            return None
        return self._cute_tile_base_expr

    def tile_begin_var(self, block_idx: int) -> str:
        # ``offset_var`` is the per-element index on this strategy, so
        # ``tile.begin`` must not fall back to it (it would broadcast
        # ``w[tile.begin]`` per element). A single lane-looped block records
        # its uniform ``begin + pid * BLOCK`` base; otherwise (multi-block flattened
        # tiles, grids without a lane loop) derive the tile start from the
        # per-element index: tiles are aligned to their static block size, so
        # ``index - index % BLOCK`` is the same value for every element of the
        # tile even though it is rendered per element.
        base = self.cute_tile_base_expr(block_idx)
        if base is not None:
            return base
        if self.block_ids != [block_idx]:
            # A flattened multi-block tile is a contiguous range of the
            # flattened iteration space, not a rectangle, so its
            # per-dimension begin is the per-element coordinate (as on the
            # Triton flattened strategy).
            return self.offset_var(block_idx)
        extent = self._configured_block_size_int(self.block_size)
        if extent is None:
            raise CompileEnvironment.current().backend.unsupported(
                "tile.begin / tile.end / tile.id of a flattened per-thread tile "
                "needs a static block size"
            )
        index = self.index_var(block_idx)
        if extent == 1:
            return index
        return f"(({index}) - (({index}) % {extent}))"

    def cute_lane_axis(self, block_id: int) -> CuteLaneAxis | None:
        """Static thread / lane distribution of a single flattened block.

        A flattened multi-block tile walks one merged index whose lanes do not
        follow any single block axis, so only the one-block form qualifies.
        """
        if self.block_ids != [block_id] or not isinstance(self.block_size, int):
            return None
        extent = self.block_size
        threads = self._num_threads if self._num_threads > 0 else extent
        elements_per_thread = self._elements_per_thread
        if threads <= 0 or threads * elements_per_thread != extent:
            return None
        lane_var = self._lane_var
        context = self.lane_context(block_id)
        vec_lane_var = context.vec_lane_var
        vec_width = context.vec_width if vec_lane_var is not None else 1
        if lane_var is None:
            if elements_per_thread != 1:
                return None
            lane_steps = 1
        elif elements_per_thread % vec_width:
            return None
        else:
            lane_steps = elements_per_thread // vec_width
        return CuteLaneAxis(
            extent=extent,
            threads=threads,
            lane_var=lane_var,
            lane_steps=lane_steps,
            vec_lane_var=vec_lane_var,
            vec_width=vec_width,
            strided=(
                lane_var is not None
                and threads > 1
                and self._cute_lane_layout == "strided"
            ),
        )

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis():
            return []
        thread_extent = self._thread_extent()
        if not isinstance(thread_extent, int):
            return []
        return [thread_extent]

    def thread_block_size_exprs(self) -> list[str]:
        if not self._uses_thread_axis():
            return []
        thread_extent = self._thread_extent()
        if isinstance(thread_extent, int):
            return [str(thread_extent)]
        if not isinstance(self.block_size, torch.SymInt):
            return []
        bs_var = self.block_size_var(-1)
        if bs_var is None:
            return []
        if self._num_threads == 0:
            return [bs_var]
        return [f"({bs_var}) // {self._elements_per_thread}"]

    def _uses_thread_axis(self) -> bool:
        thread_extent = self._thread_extent()
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        if self._lane_var is None:
            return super().codegen_grid(state)

        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []
        vec_wrappers: dict[str, VecLaneWrapper] = {}
        axis = self._flat_thread_axis()
        thread_extent = self._thread_extent()
        vec_width = self.lane_context(block_ids[-1]).vec_width
        lane_strided = self._cute_lane_layout == "strided" and isinstance(
            thread_extent, int
        )
        begins, ends, steps = self._extract_root_bounds(state)
        # ``offsets`` is the element's index (``indices = offsets``), so the
        # tile begin folds into the uniform tile base.  A tile with begins
        # never flattens, so only a single block can start past zero; its
        # bounds are static here (data-dependent extents take the N-D
        # strategy).
        assert len(block_ids) == 1 or all(begin == 0 for begin in begins)
        begin = _to_sympy(cast("int | torch.SymInt", begins[0]))

        if vec_width > 1 and (
            (
                env.config_spec.matmul_facts
                and not set(block_ids).issubset(
                    env.config_spec.cute_pointwise_region_block_ids
                )
            )
            or _cute_epilogue_subtile_active(self.fn.config)
        ):
            # See the matmul-fallback lane-loop-suppression and the
            # epilogue-subtile smem-staging notes in
            # ``PerThreadNDTileStrategy.codegen_grid``.
            vec_width = 1
        if vec_width > 1 and not env.known_multiple(begin, vec_width):
            # A packet is V-aligned only when every lane base is: the tile
            # begin must be a multiple of V.
            vec_width = 1
        if vec_width > 1:
            # Same outer x constexpr-V lane partition as
            # ``PerThreadNDTileStrategy.codegen_grid`` so the memory_ops
            # tile_unroll vec-hoist protocol applies to flattened grid
            # tiles too (the multi-block form via flat base pointers).
            block_id = block_ids[-1]
            vec_lane_var = self.new_var("vec_lane", dce=False)
            self.lane_context(block_id).vec_lane_var = vec_lane_var
            inner_for = cast(
                "ast.For",
                ast.parse(
                    f"for {vec_lane_var} in cutlass.range_constexpr({vec_width}):\n"
                    f"    pass"
                ).body[0],
            )
            lane_body: list[ast.AST] = [inner_for]
            self.lane_context(block_id).lane_body = lane_body
            self.lane_context(block_id).vloop = inner_for
            base_index_var = self.new_var("lane_base", dce=False)
            self.lane_context(block_id).base_index_var = base_index_var
            if lane_strided:
                # ``base = pid*BS + (outer*NT + tid) * V``
                base_expr = (
                    f"{offsets_base_var} + "
                    f"({env.backend.lane_offset_expr(self._lane_var)} "
                    f"* {thread_extent} + "
                    f"{env.backend.thread_index_expr(axis=axis)}) "
                    f"* {vec_width}"
                )
            else:
                # ``base = pid*BS + tid*EPT + outer*V``
                base_expr = (
                    f"{offsets_base_var} + "
                    f"{env.backend.lane_offset_expr(self._lane_var)} "
                    f"* {vec_width}"
                )
            lane_body.insert(
                0, statement_from_string(f"{base_index_var} = {base_expr}")
            )
            outer_for = _create_lane_loop(
                self._lane_var, self._elements_per_thread // vec_width, lane_body
            )
            vec_wrappers[self._lane_var] = VecLaneWrapper(
                outer_for=outer_for,
                vloop=inner_for,
                vec_lane_var=vec_lane_var,
                base_index_var=base_index_var,
            )
            lane_setup_statements.append(
                statement_from_string(
                    f"{offsets_var} = {base_index_var} + cutlass.Int32({vec_lane_var})"
                )
            )
        elif lane_strided:
            # ``offsets = pid*BS + tid + lane * NT`` — consecutive threads
            # touch consecutive elements each lane iter (coalesced even
            # without vectorization).
            lane_setup_statements.append(
                statement_from_string(
                    f"{offsets_var} = {offsets_base_var} + "
                    f"{env.backend.lane_offset_expr(self._lane_var)}"
                    f" * {thread_extent}"
                )
            )
        else:
            lane_setup_statements.append(
                statement_from_string(
                    f"{offsets_var} = {offsets_base_var} + {env.backend.lane_offset_expr(self._lane_var)}"
                )
            )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            self.trip_counts[block_idx] = numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < "
                    f"({state.sympy_expr(begin + total_numel)})"
                )
            )

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        tile_base = f"({pid_var}) * ({block_size_var})"
        if begin != 0:
            # Parenthesized: consumers splice the base into larger
            # expressions (``tile.id`` divides it by the block size).
            tile_base = f"(({state.sympy_expr(begin)}) + {tile_base})"
        self._cute_tile_base_expr = tile_base
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))
        self._record_grid_origins(state, pid_var, block_size_var, begins, steps)
        if vec_width > 1 and lane_strided:
            # The strided vec base folds the thread index in itself.
            state.add_statement(f"{offsets_base_var} = {tile_base}")
        elif lane_strided:
            # ``offsets_base = begin + pid*BS + tid`` (per-lane term adds
            # lane*NT).
            state.add_statement(
                f"{offsets_base_var} = {env.backend.lane_index_expr(tile_base, 1, axis=axis)}"
            )
        else:
            state.add_statement(
                f"{offsets_base_var} = {env.backend.lane_index_expr(tile_base, self._elements_per_thread, axis=axis)}"
            )
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)
        # Absolute ends, as in ``FlattenedTileStrategy.codegen_grid``:
        # ``tile.end`` and ``tile.count`` read them next to ``begin``.
        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        lane_loops = []
        if self._lane_var is not None:
            lane_loops = [(self._lane_var, self._elements_per_thread)]
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis() and isinstance(thread_extent, int):
            tracker.record_all(self.block_ids, axis, thread_extent)
        elif self._uses_thread_axis():
            tracker.record_symbolic_axis(self.block_ids, axis)
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_loop_blocks=set(self.block_ids) if lane_loops else set(),
            lane_loop_block_ids=(
                {self._lane_var: frozenset(self.block_ids)}
                if self._lane_var is not None
                else {}
            ),
            lane_setup_statements=lane_setup_statements,
            tile_masks=frozenset() if mask_var is None else frozenset({mask_var}),
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            vec_lane_wrappers=vec_wrappers,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        if self._lane_var is None:
            return super().codegen_device_loop(state)

        env = CompileEnvironment.current()
        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []

        lane_setup_statements.append(
            statement_from_string(
                f"{offsets_var} = {offsets_base_var} + {env.backend.lane_offset_expr(self._lane_var)}"
            )
        )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            self.trip_counts[block_idx] = numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )

        lid = self.new_var("lid")
        end_var = env.backend.cdiv_expr(
            state.sympy_expr(total_numel), block_size_var, is_device=True
        )
        axis = self._flat_thread_axis()
        user_body: list[ast.AST] = []
        body: list[ast.AST] = user_body
        user_body[:0] = lane_setup_statements
        if self._lane_var is not None:
            lane_for = _create_lane_loop(
                self._lane_var,
                self._elements_per_thread,
                body,
            )
            body = [lane_for]
        body[:0] = [
            statement_from_string(
                f"{offsets_base_var} = {env.backend.lane_index_expr(f'{lid} * ({block_size_var})', self._elements_per_thread, axis=axis)}"
            )
        ]
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, use_proxy_ends=True)
        tracker = ThreadAxisTracker()
        thread_extent = self._thread_extent()
        if self._uses_thread_axis() and isinstance(thread_extent, int):
            tracker.record_all(self.block_ids, axis, thread_extent)
        elif self._uses_thread_axis():
            tracker.record_symbolic_axis(self.block_ids, axis)
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            lane_loops=[(self._lane_var, self._elements_per_thread)],
            lane_loop_block_ids={self._lane_var: frozenset(self.block_ids)},
            lane_setup_statements=lane_setup_statements,
            tile_masks=frozenset() if mask_var is None else frozenset({mask_var}),
        )

    def offset_var(self, block_idx: int) -> str:
        return self._offsets_var

    def supports_index_rank_expansion(self) -> bool:
        return False


class CompactedShape(NamedTuple):
    size_str: str
    user_indices: list[int]
    block_ids: list[int]

    def combine(self, other: CompactedShape) -> CompactedShape:
        size_str = self.size_str
        if size_str == "1":
            size_str = other.size_str
        else:
            assert other.size_str in ("1", size_str)
        return CompactedShape(
            size_str=size_str,
            user_indices=[*self.user_indices, *other.user_indices],
            block_ids=[*self.block_ids, *other.block_ids],
        )
