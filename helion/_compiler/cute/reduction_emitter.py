"""CuTe reduction emitter: the backend's half of the reduction strategies.

Role: general-purpose emitter (with the planning that sizes it).  The generic
``ReductionStrategy`` classes in ``helion/_compiler/reduction_strategy.py``
reach this module through ``Backend.reduction_emitter(strategy)``
(``CuteBackend`` returns one of the three emitters below); a backend without
an emitter takes the generic path at every hook.

What lives here, per strategy kind:

* ``CuteReductionEmitter`` -- the shared lane-loop reduction machinery: the
  two-pass lane-reduce marker owner / group proofs, the reshape-merged group
  parameters and the fold-site records of ``cute/fold_lane_reductions.py``.
* ``CutePersistentReductionEmitter`` -- thread-count planning (the
  ``num_threads`` knob, indexed-reduction warp cap, sibling-axis partner,
  resident rows, synthetic lanes with one-vector-per-thread rows and register
  tiles), the synthetic lane preamble and the cross-warp / sibling-axis
  combines of a full-extent reduction.
* ``CuteLoopedReductionEmitter`` -- the rolled reduction's lane loop and
  vector lattice, the ``cute_cluster_n`` cluster split, constexpr trip
  counts and the cross-warp / cluster finalize.
* ``CuteBlockReductionEmitter`` -- tile-dim reductions: the strided thread
  reduction (grouped warp / two-stage shared / shared tree), the device
  lane-loop marker and the sequence-reduction hook.

Every emitter holds a reference to its strategy; the ``_cute_*`` planning
facts the rest of the backend reads (``_synthetic_cute_lane_var``,
``_cute_reduction_vec_width``, ...) stay on the strategy object, and the lane
body the memory lowering splices packets into is the strategy's
``lane_context`` (``lane_loop_context.py``).
``thread_budget.shared_memory_budget_bytes`` and ``register_tile_shape`` are
read through that module at call time (tests patch them there).
"""

from __future__ import annotations

import ast
from itertools import starmap
import logging
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.runtime.runtime_utils import next_power_of_2

from ... import exc
from ..._compat import shape_env_size_hint
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_function import find_block_size_symbols
from ..host_function import HostFunction
from ..matmul_utils import kernel_has_half_precision_matmul
from ..matmul_utils import matmul_contraction_block_ids
from ..reduction_strategy import BlockReductionEmitter
from ..reduction_strategy import BlockReductionStrategy
from ..reduction_strategy import LoopedReductionEmitter
from ..reduction_strategy import LoopedReductionStrategy
from ..reduction_strategy import PersistentReductionEmitter
from ..reduction_strategy import PersistentReductionStrategy
from ..reduction_strategy import ReductionEmitter
from ..reduction_strategy import ReductionStrategy
from ..reduction_strategy import _dtype_str
from ..reduction_strategy import _int_argument_symbol
from ..reduction_strategy import reduction_acc_dtype
from ..tile_strategy import DeviceGridState
from ..tile_strategy import DeviceLoopState
from ..tile_strategy import LoopDimInfo
from ..tile_strategy import PerThreadFlattenedTileStrategy
from ..tile_strategy import PerThreadNDTileStrategy
from ..tile_strategy import TileStrategy
from ..tile_strategy import VecLaneWrapper
from ..tile_strategy import _to_sympy
from . import thread_budget
from .analysis.broadcast_axis import explicitly_replicated_axis
from .argreduce import codegen_packed_reduction
from .device_state import CuteVloopWrapperFact
from .fold_lane_reductions import fold_site_key
from .integer_loop_reduction import defer_collective
from .lane_loops.markers import HELION_VECTOR_REDUCTION_OWNER_ATTR
from .lane_loops.markers import _create_lane_loop
from .lane_loops.markers import _grouped_two_stage_reduce_stmts
from .lane_loops.markers import _grouped_warp_reduce_expr
from .lane_loops.markers import _lane_reduce_marker_expr
from .math_templates import argreduce_candidate_expr
from .matmul_fallback import _widen_lane_layout_for_barrier_phases
from .reduce_call import ReduceCall
from .reduce_call import ReduceScope
from .reduce_call import runtime_lane_expr
from .register_tile_admission import RegisterTileUnsupported
from .resident_reductions import ResidentReductionLayout
from .resident_reductions import feature_index_expression
from .resident_reductions import supports_resident_threads
from .resident_sequence import sequence_reduction_expr
from .state import cute_env_state
from .synthetic_axes import record_strategy_axis_branch_path
from .thread_budget import CUTE_REGISTER_TILE_MAX_ELEMENTS
from .thread_budget import MAX_THREADS_PER_BLOCK
from .tile_packing import packed_grid_block

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..inductor_lowering import CodegenState

log = logging.getLogger(__name__)

S = TypeVar("S", bound=ReductionStrategy)


class CutePersistentReductionStrategy(PersistentReductionStrategy):
    """The persistent strategy on CuTe (``CuteBackend.create_reduction_strategy``):
    a synthetic-lane reduction inside the grid sets its index and mask per lane
    (``CutePersistentReductionEmitter.emit_synthetic_lane_preamble``); every
    other reduction uses the generic per-thread form."""

    def _codegen_index(
        self, state: CodegenState, block_size_var: str, numel: sympy.Expr
    ) -> None:
        current_grid = state.codegen.current_grid_state
        if self._synthetic_cute_lane_var is None or current_grid is None:
            super()._codegen_index(state, block_size_var, numel)
            return
        emitter = self.emitter
        assert isinstance(emitter, CutePersistentReductionEmitter), type(emitter)
        emitter.emit_synthetic_lane_preamble(state, current_grid, numel, block_size_var)


def reduction_emitter_for(strategy: ReductionStrategy) -> ReductionEmitter[Any]:
    """The CuTe emitter matching ``strategy``'s kind (``CuteBackend.reduction_emitter``)."""
    if isinstance(strategy, PersistentReductionStrategy):
        return CutePersistentReductionEmitter(strategy)
    if isinstance(strategy, LoopedReductionStrategy):
        return CuteLoopedReductionEmitter(strategy)
    if isinstance(strategy, BlockReductionStrategy):
        return CuteBlockReductionEmitter(strategy)
    return CuteReductionEmitter(strategy)


def _cute_reduction_smem_bytes(num_elements: int, dtype: torch.dtype) -> int:
    return num_elements * torch.empty((), dtype=dtype).element_size()


_CUTE_LOOPED_REDUCTION_MAX_ELEMENTS_PER_THREAD = 256
_CUTE_WARP_REDUCTION_THREADS = 32


def cute_looped_reduction_block_size(size_hint: int, max_threads: int) -> int:
    """Pick the default CuTe loop chunk for reductions wider than one warp."""
    return min(size_hint, max_threads * _CUTE_LOOPED_REDUCTION_MAX_ELEMENTS_PER_THREAD)


def cute_live_reduction_threads(max_threads: int) -> int:
    # Persistent reductions on CuTe can recruit threads beyond a single warp
    # (cross-warp combining uses _cute_grouped_reduce_shared_two_stage). The
    # autotuner / config_spec keeps the size_hint <= max_threads case here so
    # no synthetic lane wrap is required.
    return max_threads


def _cute_blocks_share_a_value(
    env: CompileEnvironment, first: int, second: int
) -> bool:
    """Whether some tile value of the kernel carries both blocks as dims.

    Two full-slice axes held by one value (rope's ``q[tile_b, :, tile_t,
    :half]`` holds the head axis and the half-dim axis) need the cross
    product of their coordinates, which only distinct thread axes (or a lane
    loop) provide; axes that never meet in a value (the q heads and the k
    heads) may index one thread axis.  A literal dim is matched to a block
    by its extent, as the literal-dim holders do; an extent several blocks
    share counts as both (fails closed).
    """

    def blocks_of(size: object) -> set[int]:
        if isinstance(size, torch.SymInt):
            block_id = env.get_block_id(size)
            if block_id is not None:
                return {block_id}
        if not isinstance(size, (int, torch.SymInt)):
            return set()
        return {
            info.block_id
            for info in env.block_sizes
            if info.reduction
            and isinstance(info.size, (int, torch.SymInt))
            and env.known_equal(info.size, size)
        }

    for graph_info in HostFunction.current().device_ir.graphs:
        for node in graph_info.graph.nodes:
            value = node.meta.get("val")
            if not isinstance(value, torch.Tensor):
                continue
            found: set[int] = set()
            for size in value.shape:
                found |= blocks_of(size)
            if first in found and second in found:
                return True
    return False


def _cute_cluster_reduce_smem_vars(
    fn: DeviceFunction, group_span: int, cluster_n: int
) -> tuple[str, str]:
    """Allocate + initialize the SMEM receive buffer and single-phase
    mbarrier for one ``_cute_grouped_reduce_cluster`` site in the kernel
    preamble, and count the site so ``codegen_function_def`` emits ONE
    ``mbarrier_init_fence`` + cluster arrive/wait covering every site (a
    per-site cluster barrier costs ~10% of kernel time at cluster_n=16).
    """
    slots = (group_span // 32) * cluster_n
    buf_var = fn.new_var("_cluster_red_buf", dce=False)
    mbar_var = fn.new_var("_cluster_red_mbar", dce=False)
    fn.preamble.append(
        statement_from_string(
            f"{buf_var} = cute.arch.alloc_smem(cutlass.Float32, {slots})"
        )
    )
    fn.preamble.append(
        statement_from_string(f"{mbar_var} = cute.arch.alloc_smem(cutlass.Int64, 1)")
    )
    fn.preamble.append(
        statement_from_string(
            "if cutlass.Int32(cute.arch.thread_idx()[0]) == 0:"
            f"\n    cute.arch.mbarrier_init({mbar_var}, 1)"
        )
    )
    fn.cute_state.lanes.simt_cluster_reduce_sites += 1
    return buf_var, mbar_var


def _block_has_indexed_reduction(fn: DeviceFunction, block_index: int) -> bool:
    """Return True when ``block_index`` is the reduction axis of any
    argmin/argmax in the device IR.

    Populated by :meth:`DeviceIR.register_rollable_reductions` so this is
    just a set lookup on ConfigSpec.

    Used to cap CuTe reduction strategies' thread_count at the warp width
    when an indexed reduction is present — CuTe argreduce uses
    cute.arch.warp_reduction which is only correct for threads_in_group<=32.
    """
    env = CompileEnvironment.current()
    return block_index in env.config_spec.cute_indexed_reduction_block_ids


def _cute_is_folded_matmul_contraction(block_index: int) -> bool:
    """Whether ``block_index`` is the contraction of a matmul the scalar
    fallback lowers (no 16-bit operand a warp-MMA plan may take)."""
    env = CompileEnvironment.current()
    return (
        env.canonical_block_id(block_index) in matmul_contraction_block_ids()
        and not kernel_has_half_precision_matmul()
    )


class CuteReductionEmitter(ReductionEmitter[S]):
    """The lane-loop reduction machinery every CuTe reduction strategy shares.

    Constructed for a bare ``ReductionStrategy``; the three strategy-kind
    emitters below extend it.
    """

    def claims_thread_axis(self) -> bool:
        # A reduction left with a single live thread claims no axis (see
        # ``ReductionStrategy._claims_thread_axis``).
        return self.strategy._reduction_thread_count() > 1

    def record_axis_branch_path(self, state: CodegenState) -> None:
        # Record the branch path under which this reduction claims its thread
        # axis, so a free ``hl.arange`` in a mutually-exclusive sibling grid
        # branch reuses this axis instead of claiming a fresh one that would
        # widen the launch block and race this single-axis reduction. No-op
        # outside a dynamic ``_if`` branch.
        if self.claims_thread_axis():
            record_strategy_axis_branch_path(
                state.codegen, self.strategy._get_thread_axis()
            )

    def mark_cross_block_vec_lanes(self, state: CodegenState) -> None:
        """Record the constexpr V-loops that enclose a reduction over ``block_id``
        but iterate another block (``CuteDeviceFunctionState.unfoldable_vec_lanes``).

        The V-fold of ``cute/hoist_warp_reduce.py`` combines a V-loop's lanes
        before the cross-thread reduce, which is the reduction itself only when
        the lanes are elements of the reduced block.  A vector width on the tile
        of ``s[tile] = x[tile, :].sum(-1)`` gives each lane its own row, so those
        lanes must each reduce on their own and feed one element of the vector
        store.
        """
        strategy = self.strategy
        env = CompileEnvironment.current()
        codegen = state.codegen
        active_block_ids = {
            block for block, loops in codegen.active_device_loops.items() if loops
        }
        if codegen.current_grid_state is not None:
            active_block_ids.update(codegen.current_grid_state.block_ids)
        reduced = env.canonical_block_id(strategy.block_index)
        unfoldable = state.device_function.cute_state.lanes.unfoldable_vec_lanes
        for tile_strategy in state.device_function.tile_strategy.strategies:
            if not isinstance(
                tile_strategy, (PerThreadNDTileStrategy, PerThreadFlattenedTileStrategy)
            ):
                continue
            for other in tile_strategy.block_ids:
                vec_lane_var = tile_strategy.lane_context(other).vec_lane_var
                if (
                    vec_lane_var is not None
                    and other in active_block_ids
                    and env.canonical_block_id(other) != reduced
                ):
                    unfoldable.add(vec_lane_var)

    def _vector_lane_loop_var(self) -> str | None:
        """The lane variable of the VECTORIZED device lane loop that
        distributes this reduction block, or ``None``.

        Such a loop walks several packets per thread per trip, and the
        two-pass lane marker declines it
        (``CuteBlockReductionEmitter.emit_device_lane_loop_marker``), so the
        strided reduce below is emitted once per lane and reduces one lane's
        packet.  ``cute/fold_lane_reductions.py`` folds those per-lane
        partials across the unrolled lanes; it finds the sites through
        ``cute_state.lane_fold_reduce_sites`` (``record_lane_fold_site``).
        """
        strategy = self.strategy
        if not isinstance(strategy, BlockReductionStrategy):
            return None
        for loops in strategy._codegen.active_device_loops.values():
            for loop_state in loops:
                if (
                    isinstance(loop_state, DeviceLoopState)
                    and strategy.block_index in loop_state.lane_loop_blocks
                    and isinstance(loop_state.strategy, PerThreadNDTileStrategy)
                    and loop_state.strategy.lane_context(strategy.block_index).vec_width
                    != 1
                ):
                    return loop_state.strategy._lane_var_by_block.get(
                        strategy.block_index
                    )
        return None

    def record_lane_fold_site(self, call_text: str) -> None:
        """Register a cross-thread reduce emitted inside a vectorized lane
        loop over its own block (see ``_vector_lane_loop_var``): the grouped
        strided helpers and the direct single-warp ``warp_reduction_*``."""
        strategy = self.strategy
        lane_var = self._vector_lane_loop_var()
        if lane_var is None:
            return
        sites = strategy.fn.cute_state.lanes.lane_fold_reduce_sites
        sites.setdefault(lane_var, set()).add(fold_site_key(call_text))

    def _lane_reduce_cluster_n(self) -> int:
        """Thread-block-cluster width of the lane-looped lane_strategy that
        distributes this reduction block, or 1 when not cluster-split."""
        strategy = self.strategy
        if not isinstance(strategy, BlockReductionStrategy):
            return 1
        for loops in strategy._codegen.active_device_loops.values():
            for loop_state in loops:
                if (
                    isinstance(loop_state, DeviceLoopState)
                    and strategy.block_index in loop_state.lane_loop_blocks
                ):
                    lane_strategy = loop_state.strategy
                    if isinstance(lane_strategy, PerThreadNDTileStrategy):
                        return lane_strategy.lane_context(
                            strategy.block_index
                        ).cluster_n
        return 1

    def _lane_reduce_threads_in_group(self) -> int | None:
        """Return ``threads_in_group`` for a two-pass lane reduction over this
        block, or ``None`` when this reduction is not over a lane-distributed
        block.

        When the block is split across a per-thread lane loop, the per-lane
        partials must first be accumulated across the lane loop and then
        combined across the live thread axis (``threads_in_group``). A value of
        1 means the block has no live thread axis (a pure lane loop), so the
        accumulator alone is the result.
        """
        strategy = self.strategy
        # A synthetic reduction lane (PersistentReductionStrategy) always
        # distributes the reduction axis across a lane loop; the live thread
        # axis is ``_reduction_thread_count`` wide.
        if (
            isinstance(strategy, PersistentReductionStrategy)
            and strategy._synthetic_cute_lane_var is not None
        ):
            return max(1, strategy._reduction_thread_count())
        if not (
            strategy._reduction_block_has_lane_loops()
            or strategy._reduction_block_in_device_lane_loop()
        ):
            return None
        threads = strategy._reduction_thread_count()
        return max(1, threads)

    def _reshape_merged_reduction_group_params(
        self,
    ) -> tuple[int, int, str] | None:
        """Return ``(pre, group_span, lane_expr)`` for a reshape-merged
        reduction, including a contiguous or entirely serial source layout.

        When ``x[tile0, tile1, tile2].reshape(tile0, -1).sum(-1)`` merges
        ``tile1`` (a live thread axis) and ``tile2`` (a lane loop) into a
        single synthetic reduction block, the reduction's live thread axis
        (``tile1``) shares the launch warp with the *unrelated* ``tile0``
        row axis. A plain ``cute.arch.warp_reduction_*(threads_in_group=N)``
        folds together CONSECUTIVE warp lanes, so it would sum across both
        ``tile1`` AND ``tile0`` (cross-contaminating rows). Instead the
        reduction must be grouped/strided so each lane only combines the
        lanes that share its ``tile0`` coordinate.

        This computes the ``pre`` (product of live thread extents on axes
        *below* the reduce axis) and ``group_span`` (``pre`` times the
        reduce axis extent) used by ``_cute_grouped_reduce_warp``. A contiguous
        group still needs an explicit source-lane owner when reshaping removes
        the synthetic coordinate. Unsupported source layouts return ``None``.
        """
        strategy = self.strategy
        env = CompileEnvironment.current()
        backend = env.backend
        numel = env.block_sizes[strategy.block_index].numel
        if not isinstance(numel, sympy.Expr):
            return None
        if numel != sympy.prod(numel.free_symbols):
            return None
        # Source block ids merged into this reduction dim by the reshape.
        source_block_ids: set[int] = set()
        for symbol in numel.free_symbols:
            if not isinstance(symbol, sympy.Symbol):
                return None
            block_id = env.get_block_id(symbol)
            if block_id is None:
                return None
            source_block_ids.add(env.canonical_block_id(block_id))
        if len(source_block_ids) < 2:
            # A single source block needs no de-interleaving.
            return None
        tile_strategy = strategy.fn.tile_strategy
        # The reduce axis is the (single) live thread axis spanned by the
        # source blocks. A lane-looped source block has ``extent is None``.
        reduce_axis: int | None = None
        reduce_extent = 1
        for block_id in source_block_ids:
            axis = tile_strategy.thread_axis_for_block_id(block_id)
            extent = tile_strategy.thread_extent_for_block_id(block_id)
            if axis is None or extent is None or extent <= 1:
                continue
            if reduce_axis is not None and reduce_axis != axis:
                # More than one live thread axis among the source blocks is
                # not expressible as a single grouped warp reduce.
                return None
            reduce_axis = axis
            reduce_extent = max(reduce_extent, extent)
        if reduce_axis is None:
            # No source axis has hardware lanes. A single serial source lane
            # plus active singleton tiles needs no physical thread reduction.
            owners: set[str] = set()
            for block_id in source_block_ids:
                active = {
                    id(loop): loop
                    for loop in strategy.fn.codegen.active_device_loops.get(
                        block_id, []
                    )
                    if isinstance(loop, DeviceLoopState)
                    and isinstance(loop.strategy, PerThreadNDTileStrategy)
                }
                if len(active) != 1:
                    return None
                loop = next(iter(active.values()))
                if block_id in loop.lane_loop_blocks:
                    owner = cast(
                        "PerThreadNDTileStrategy", loop.strategy
                    )._lane_var_by_block.get(block_id)
                    if owner is None:
                        return None
                    owners.add(owner)
                elif strategy.fn.resolved_block_size(block_id) != 1:
                    return None
            if len(owners) != 1:
                return None
            return 1, 1, "0"
        # Live thread extents of ALL blocks (siblings included) so the linear
        # lane index strides are computed correctly. The reduction block's own
        # synthetic thread axis is excluded -- it is fictional (no real warp
        # lanes back it); the source live axis carries the actual data.
        logical_axis_sizes: dict[int, int] = {reduce_axis: reduce_extent}
        for info in env.block_sizes:
            block_id = info.block_id
            if block_id == strategy.block_index or block_id in source_block_ids:
                continue
            axis = tile_strategy.thread_axis_for_block_id(block_id)
            extent = tile_strategy.thread_extent_for_block_id(block_id)
            if axis is None or extent is None or extent <= 1:
                continue
            logical_axis_sizes[axis] = max(logical_axis_sizes.get(axis, 1), extent)
        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        group_span = pre * reduce_extent
        if group_span > 32:
            # Cross-warp grouped reduction is not handled by the marker path.
            return None
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if lane_expr is None:
            return None
        return pre, group_span, lane_expr

    def _reshape_physical_reduction_group_params(
        self, state: CodegenState
    ) -> tuple[int, int, str] | None:
        """Prove a merged reduction is distributed entirely over hardware lanes.

        Unlike the serial-lane fallback, every source tile must fit its own
        active physical axis. Two adjacent axes may form one warp group; an
        unrelated axis between them, a partial tile, or a serial source cannot.
        This records layout only. The post-wrap pass must also prove uniform
        execution and the absence of the synthetic coordinate and loop.
        """
        strategy = self.strategy
        env = CompileEnvironment.current()
        numel = env.block_sizes[strategy.block_index].numel
        if (
            strategy.fn.cute_state.lanes.simt_cluster_n != 1
            or not isinstance(numel, sympy.Expr)
            or numel != sympy.prod(numel.free_symbols)
        ):
            return None
        sources: set[int] = set()
        for symbol in numel.free_symbols:
            block_id = env.get_block_id(symbol)
            if block_id is None:
                return None
            sources.add(env.canonical_block_id(block_id))
        if len(sources) < 2:
            return None
        dispatch = strategy.fn.tile_strategy
        source_axes: set[int] = set()
        for block_id in sources:
            active = {
                id(loop): loop
                for loop in state.codegen.active_device_loops.get(block_id, [])
                if isinstance(loop, DeviceLoopState)
                and isinstance(loop.strategy, PerThreadNDTileStrategy)
            }
            if len(active) != 1:
                return None
            loop = next(iter(active.values()))
            if block_id in loop.lane_loop_blocks:
                return None
            size = strategy.fn.resolved_block_size(block_id)
            if size is None:
                return None
            axis = loop.block_thread_axes.get(block_id)
            if axis is None:
                if size != 1:
                    return None
                continue
            if (
                axis not in range(3)
                or dispatch.thread_axis_for_block_id(block_id) != axis
                or dispatch.thread_extent_for_block_id(block_id) != size
                or axis in source_axes
            ):
                return None
            if size > 1:
                source_axes.add(axis)

        # Use only actual active source/sibling axes, not the fictional merged
        # strategy. Final launch dimensions are likewise based on live indices.
        sizes: dict[int, int] = {}
        for block_id, loops in state.codegen.active_device_loops.items():
            for loop in loops:
                if not isinstance(loop, (DeviceGridState, DeviceLoopState)):
                    continue
                axis = loop.block_thread_axes.get(block_id)
                if axis is None or block_id == strategy.block_index:
                    continue
                extent = dispatch.thread_extent_for_block_id(block_id)
                if extent is None or axis not in range(3):
                    return None
                if extent > 1:
                    if axis in sizes and sizes[axis] != extent:
                        return None
                    sizes[axis] = extent
                    if axis in source_axes and block_id not in sources:
                        return None
        if not source_axes:
            return 1, 1, "0"
        first, last = min(source_axes), max(source_axes)
        if any(axis not in source_axes for axis in sizes if first <= axis <= last):
            return None
        pre = 1
        span = 1
        for axis in range(last + 1):
            extent = sizes.get(axis, 1)
            if extent <= 0 or extent & (extent - 1):
                return None
            if axis < first:
                pre *= extent
            span *= extent
        if span > 32:
            return None
        lane_expr = env.backend.thread_linear_index_expr(sizes)
        return (pre, span, lane_expr) if lane_expr is not None else None

    def _lane_reduce_marker_unsupported(self, state: CodegenState) -> bool:
        """Return True when the two-pass lane-reduction marker cannot be
        handled by the ``split_lane_loop_reductions`` post-pass, so the caller
        must fall back to the existing single-pass path.

        Two situations are unsupported:

        * An active *serial* device loop (over a different block) wraps this
          reduction inside the lane scope. The post-pass splits the lane loop
          at its top level, but here the reduction needs the lanes reduced
          *per* serial iteration (the lane loop is outside the serial loop).
        * An active ``LoopedReductionStrategy`` already rolls this block: the
          rolled loop carries its own accumulator (and vec-fold) for the lane
          reduction, so emitting a marker would double-handle it.
        """
        strategy = self.strategy
        # Bound at call time: tests patch ``generate_ast.GenerateAST`` with a
        # fake codegen class and expect this ``isinstance`` to see it.
        from ..generate_ast import GenerateAST

        # A serial device loop may enclose the lane that owns this reduction.
        # Its complete lane reduction then remains inside each serial iteration.
        # Use the live statement-list frontier to establish this order; block
        # numbers and insertion order do not describe lexical nesting.
        scope_positions: dict[int, int] = {}
        owner_scope: int | None = None
        if isinstance(state.codegen, GenerateAST):
            scope_positions = {
                id(statements): index
                for index, statements in enumerate(state.codegen.statements_stack)
            }
            owner_scopes = {
                scope_positions[id(loop.inner_statements)]
                for loops in state.codegen.active_device_loops.values()
                for loop in loops
                if isinstance(loop, DeviceLoopState)
                and strategy.block_index in loop.lane_loop_blocks
                and id(loop.inner_statements) in scope_positions
            }
            if len(owner_scopes) == 1:
                owner_scope = next(iter(owner_scopes))

        for block_id, loops in state.codegen.active_device_loops.items():
            for loop_state in loops:
                if not isinstance(loop_state, DeviceLoopState):
                    continue
                if isinstance(loop_state.strategy, LoopedReductionStrategy):
                    # The rolled reduction owns the lane reduction over its
                    # block; defer to its accumulate / vec-fold machinery.
                    if block_id == strategy.block_index:
                        return True
                    continue
                if (
                    block_id != strategy.block_index
                    and block_id not in loop_state.block_thread_axes
                    and block_id not in loop_state.lane_loop_blocks
                ):
                    serial_scope = scope_positions.get(id(loop_state.inner_statements))
                    if (
                        owner_scope is not None
                        and serial_scope is not None
                        and serial_scope < owner_scope
                    ):
                        continue
                    # The reduction's lane loop is OUTSIDE this serial device
                    # loop. A synthetic-lane PersistentReductionStrategy can be
                    # repaired by the ``interchange_lane_outside_serial_reductions``
                    # post-pass (it splits the lane loop into a lane-inside-mb
                    # two-pass nest for the broadcast consumer plus a
                    # lane-outside-mb nest for any per-feature accumulators), so
                    # keep emitting the marker in that case. Other (non-synthetic)
                    # situations remain unsupported.
                    if (
                        isinstance(strategy, PersistentReductionStrategy)
                        and strategy._synthetic_cute_lane_var is not None
                    ):
                        continue
                    return True
        return False

    def _lane_reduce_owner(
        self,
        state: CodegenState,
        *,
        reshape_group: tuple[int, int, str] | None = None,
    ) -> str:
        """Return the actual serial lane that distributes this reduction axis."""
        strategy = self.strategy
        if isinstance(strategy, PersistentReductionStrategy):
            if strategy._synthetic_cute_lane_var is not None:
                # Reshape may merge existing tile axes without redistributing
                # their values. In that case the synthetic index is unused
                # and DeviceGridState drops its loop. Recover the one concrete
                # serial lane from the merged dimension's source-block symbols.
                env = CompileEnvironment.current()
                numel = env.block_sizes[strategy.block_index].numel
                source_blocks: set[int] = set()
                if isinstance(numel, sympy.Expr) and numel == sympy.prod(
                    numel.free_symbols
                ):
                    for symbol in numel.free_symbols:
                        if not isinstance(symbol, sympy.Symbol):
                            source_blocks.clear()
                            break
                        block_id = env.get_block_id(symbol)
                        if block_id is None:
                            source_blocks.clear()
                            break
                        source_blocks.add(env.canonical_block_id(block_id))
                if len(source_blocks) >= 2:
                    owners: set[str] = set()
                    covered: set[int] = set()
                    for block_id in source_blocks:
                        active = {
                            id(loop): loop
                            for loop in state.codegen.active_device_loops.get(
                                block_id, []
                            )
                            if isinstance(loop, DeviceLoopState)
                            and isinstance(loop.strategy, PerThreadNDTileStrategy)
                        }
                        if len(active) != 1:
                            continue
                        loop = next(iter(active.values()))
                        if block_id in loop.lane_loop_blocks:
                            owner = cast(
                                "PerThreadNDTileStrategy", loop.strategy
                            )._lane_var_by_block.get(block_id)
                            if owner is not None:
                                owners.add(owner)
                                covered.add(block_id)
                        elif block_id in loop.block_thread_axes:
                            covered.add(block_id)
                        elif strategy.fn.resolved_block_size(block_id) == 1:
                            # An active singleton tile contributes no lane or
                            # hardware coordinate to the merged reduction.
                            covered.add(block_id)
                    if (
                        covered == source_blocks
                        and len(owners) == 1
                        and strategy.fn.cute_state.lanes.simt_cluster_n == 1
                        and reshape_group is not None
                    ):
                        # Keep the synthetic owner for now: a value can still
                        # use that coordinate despite having a merged shape.
                        # Only the post-wrap body can prove its loop was dead.
                        record = (next(iter(owners)), *reshape_group)
                        previous = strategy.fn.cute_state.lanes.reshape_lane_fallbacks.setdefault(
                            strategy._synthetic_cute_lane_var, record
                        )
                        if previous != record:
                            raise exc.BackendUnsupported(
                                "cute", "reshape reduction has conflicting lane owners"
                            )
                return strategy._synthetic_cute_lane_var
        states = [
            loop
            for loops in state.codegen.active_device_loops.values()
            for loop in loops
        ]
        if state.codegen.current_grid_state is not None:
            states.append(state.codegen.current_grid_state)
        for loop in states:
            if (
                not isinstance(loop, (DeviceGridState, DeviceLoopState))
                or strategy.block_index not in loop.lane_loop_blocks
            ):
                continue
            owner_strategy = loop.strategy
            if isinstance(owner_strategy, PerThreadNDTileStrategy):
                owner = owner_strategy._lane_var_by_block.get(strategy.block_index)
                if owner is not None:
                    vector_loop = owner_strategy.lane_context(
                        strategy.block_index
                    ).vloop
                    if vector_loop is not None:
                        setattr(vector_loop, HELION_VECTOR_REDUCTION_OWNER_ATTR, owner)
                    return owner
            elif isinstance(owner_strategy, PerThreadFlattenedTileStrategy):
                if owner_strategy._lane_var is not None:
                    vector_loop = owner_strategy.lane_context(
                        owner_strategy.block_ids[-1]
                    ).vloop
                    if vector_loop is not None:
                        setattr(
                            vector_loop,
                            HELION_VECTOR_REDUCTION_OWNER_ATTR,
                            owner_strategy._lane_var,
                        )
                    return owner_strategy._lane_var
        raise exc.BackendUnsupported("cute", "reduction lane owner is not proven")


class CutePersistentReductionEmitter(
    CuteReductionEmitter[PersistentReductionStrategy], PersistentReductionEmitter
):
    """Thread layout, synthetic lanes and cross-warp combines of a
    full-extent (persistent) reduction."""

    def claims_thread_axis(self) -> bool:
        if self.strategy._cute_axis_partner is not None:
            # Indexes a sibling's axis (``plan_axis_partner``).
            return False
        return super().claims_thread_axis()

    def reject_int_argument_extent(self, numel: object) -> None:
        env = CompileEnvironment.current()
        if (int_argument := _int_argument_symbol(numel)) is not None:
            # The thread layout below is sized from the bound value's hint
            # and fixed at codegen; a register dim ``hl.zeros([tile, bsz])``
            # with ``bsz`` an int argument then holds the hint's elements at
            # every call, dropping or repeating the dim's at another value
            # (an int argument is not part of the specialization key).
            raise exc.BackendUnsupported(
                env.backend.name,
                f"a reduction dim sized by the int argument {int_argument}: the "
                "persistent reduction's thread layout is fixed at codegen from "
                "the bound value and would not follow the argument",
            )

    def plan_hint_bucket(self, extent: object, size_hint: int) -> int | None:
        if isinstance(extent, sympy.Expr) and extent.free_symbols:
            # The same layout follows a dynamic tensor size only within the
            # hint's power-of-2 bucket; the bound kernel keys that bucket.
            cute_env_state(
                CompileEnvironment.current()
            ).hint_sized_reduction_extents.add(extent)
            return next_power_of_2(size_hint)
        return None

    def plan_max_threads(self, max_threads: int) -> int:
        strategy = self.strategy
        max_threads = cute_live_reduction_threads(max_threads)
        # Indexed reductions (argmin/argmax) on CuTe only have a
        # warp-level reduction primitive that takes
        # ``threads_in_group <= 32``. Cap the persistent thread
        # count to the warp size so the emitted
        # ``cute.arch.warp_reduction`` is correct.
        if _block_has_indexed_reduction(strategy.fn, strategy.block_index):
            max_threads = min(max_threads, _CUTE_WARP_REDUCTION_THREADS)
        return max_threads

    def plan_thread_count(self, size_hint: int, max_threads: int) -> None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        # Persistent reductions use the same per-block thread-count
        # knob as rolled reductions.  A smaller live subgroup keeps
        # unrelated tile axes from inflating the CTA and lets each
        # thread retain a short contiguous reduction fragment.
        requested = env.config_spec.num_threads.config_get(
            cast("list[int]", strategy.fn.config.config.get("num_threads", []) or []),
            strategy.block_index,
            0,
        )
        if isinstance(requested, int):
            strategy._cute_requested_threads = requested
        if isinstance(requested, int) and 0 < requested < strategy._thread_count:
            strategy._thread_count = requested
        elif (
            requested == 0
            and size_hint > max_threads
            and _cute_is_folded_matmul_contraction(strategy.block_index)
        ):
            # A matmul contraction longer than the CTA's threads is
            # split into synthetic lanes, which the scalar fallback
            # folds serially per output element
            # (``emit_cute_synthetic_lane_fold_mm``): more threads on
            # it would only repeat that fold.  The free tile axes take
            # them instead.
            strategy._thread_count = 1

    def plan_resident_reduction(self) -> bool:
        return supports_resident_threads(self.strategy.fn, self.strategy.block_index)

    def prove_merged_tiled_extent(self, numel: object) -> bool:
        env = CompileEnvironment.current()
        return (
            isinstance(numel, sympy.Expr)
            and len(numel.free_symbols) >= 2
            and numel == sympy.prod(numel.free_symbols)
            and all(
                env.get_block_id(symbol) is not None for symbol in numel.free_symbols
            )
        )

    def plan_synthetic_lanes(
        self,
        lane_extent: int | None,
        size_hint: int,
        mask_var: str | None,
        merged_tiled_extent: bool,
    ) -> None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        block_index = strategy.block_index
        fn = strategy.fn
        if (
            lane_extent is not None
            and strategy._thread_count * lane_extent
            != next_power_of_2(max(1, size_hint))
        ):
            # Uneven synthetic slices would let an unmasked vector or
            # register row access beyond its padded reduction extent.
            # Keep a power-of-two live count within the thread budget
            # so every slice fits, then plan the layout from that count.
            strategy._thread_count = 1 << (strategy._thread_count.bit_length() - 1)
            lane_extent = env.backend.create_synthetic_reduction_lanes(
                strategy._thread_count, size_hint
            )
        # A merged tiled axis can still contain a serial source lane
        # when its configured total extent fits in one warp. Keep an
        # explicit logical owner for the reduction; after wrapping,
        # resolve_pruned_lane_owners maps unused synthetic coordinates
        # back to the proven source lane and its physical subgroup.
        if lane_extent is None and merged_tiled_extent:
            lane_extent = 1
        # One complete vector per live thread: the requested V exactly
        # covers each thread's synthetic slice, so the lane loop is a
        # constexpr V-fold with no loop-carried lanes.  The memory-op
        # gate independently caps each load/store at 16 bytes for its
        # dtype.
        requested_vec = 1
        if mask_var is None and not strategy._cute_resident_reduction:
            configured_vec = env.config_spec.cute_vector_widths.config_get(
                cast(
                    "list[int]",
                    fn.config.config.get("cute_vector_widths", []) or [],
                ),
                block_index,
                1,
            )
            if isinstance(configured_vec, int) and configured_vec > 1:
                requested_vec = configured_vec
        # The synthetic lane loop folds each lane into a per-thread
        # accumulator that is then combined across the live threads.
        # A single warp reduction is only correct within one warp, so
        # a lane-looped multi-warp group caps its thread count to the
        # warp size and lets the lane loop grow to cover the rest;
        # otherwise the group silently drops lanes (#2643).  A
        # one-vector-per-thread row keeps its warp-aligned multi-warp
        # thread count instead (``row_len / V`` threads, e.g. 128 for
        # a 1024-wide bf16 row): its finalize combines the per-thread
        # V-folds with the cross-warp two-stage shared reduce (see
        # ``_sibling_axis_group_params``), matching the one-LDG.128
        # -per-thread shape of the Triton kernel instead of a 32-thread
        # CTA looping over scalar lanes.
        vector_row = (
            lane_extent is not None
            and requested_vec > 1
            and lane_extent == requested_vec
        )
        multiwarp_vector_row = (
            vector_row and strategy._thread_count % _CUTE_WARP_REDUCTION_THREADS == 0
        )
        # A scalar lane over one-vector tile wrappers nests OUTSIDE
        # them as a trace-time register tile (see
        # ``DeviceGridState.nest_reduction_lane_outside_vector_tiles``)
        # whose finalize combines each tile element across the group
        # with the cross-warp column reduce, so it keeps a multi-warp
        # thread count too.  The prediction mirrors the grid-time
        # check; ``codegen_preamble`` rejects the config if the grid
        # turns out not to match.
        strategy._cute_register_tile_predicted = (
            not vector_row
            and mask_var is None
            and lane_extent is not None
            and lane_extent > 1
            and not strategy._cute_resident_reduction
            and thread_budget.register_tile_shape(fn, block_index, lane_extent)
        )
        multiwarp_register_tile = (
            strategy._cute_register_tile_predicted
            and strategy._thread_count % _CUTE_WARP_REDUCTION_THREADS == 0
        )
        if (
            lane_extent is not None
            and strategy._thread_count > _CUTE_WARP_REDUCTION_THREADS
            and not strategy._cute_resident_reduction
            and not multiwarp_vector_row
            and not multiwarp_register_tile
        ):
            strategy._thread_count = _CUTE_WARP_REDUCTION_THREADS
            lane_extent = env.backend.create_synthetic_reduction_lanes(
                strategy._thread_count, size_hint
            )
        if lane_extent is not None:
            strategy._synthetic_cute_lane_var = fn.new_var(
                f"synthetic_lane_{block_index}",
                dce=False,
            )
            strategy._synthetic_cute_lane_extent = lane_extent
            if strategy._cute_resident_reduction:
                width = env.config_spec.cute_vector_widths.config_get(
                    cast(
                        "list[int]",
                        fn.config.config.get("cute_vector_widths", []) or [],
                    ),
                    block_index,
                    1,
                )
                fn.cute_state.lanes.resident_reduction_layouts[
                    strategy._synthetic_cute_lane_var
                ] = ResidentReductionLayout(
                    block_id=block_index,
                    feature_extent=size_hint,
                    threads=strategy._thread_count,
                    lane_extent=lane_extent,
                    vector_width=min(width, lane_extent),
                    index_name=strategy.index_var(block_index),
                )
            # Tested against the lane extent AFTER the cap: a row the
            # cap brings down to exactly V lanes per thread (a 256-wide
            # bf16 row asked for 64 or 128 threads with V=8) keeps its
            # one-warp vector loads exactly as a 32-thread request does.
            if requested_vec > 1 and lane_extent == requested_vec:
                strategy._cute_reduction_vec_width = requested_vec

    def check_shared_thread_extent(self, extent: int) -> None:
        strategy = self.strategy
        if strategy._synthetic_cute_lane_var is not None or (
            extent > _CUTE_WARP_REDUCTION_THREADS
            and _block_has_indexed_reduction(strategy.fn, strategy.block_index)
        ):
            # The lane loop strides by the thread count, so the surplus
            # threads would alias the elements of later lanes; argmin/argmax
            # combine within one warp only (see ``__init__``).
            raise strategy._shared_thread_extent_error(extent)

    def plan_axis_partner(
        self,
        concurrent: list[TileStrategy],
        size_hint: int,
    ) -> PersistentReductionStrategy | None:
        """A concurrent sibling full-slice block whose thread axis this block
        shares, or ``None``.

        Two full-slice axes of one pure pointwise tile body (rope's q heads
        and k heads) are independent per-thread lane loops.  When one owns a
        thread axis, every statement of the other runs once per thread of
        that axis, repeating its loads and stores ``threads`` times; when the
        config spells the same explicit thread count for both, this block
        indexes the sibling's axis instead (``thread_idx()[axis]`` is a head
        of each), each thread walking its own share of both axes.  Only a
        kernel without reductions qualifies (``pointwise_facts``: there is no
        cross-thread combine to keep per axis), and both static extents must
        be multiples of the shared count so no thread falls outside either.
        ``concurrent`` already excludes the strategies of mutually exclusive
        branches (``_strategies_concurrent_with_block``).
        """
        strategy = self.strategy
        env = CompileEnvironment.current()
        requested = strategy._cute_requested_threads
        if (
            not env.config_spec.pointwise_facts
            or requested <= 1
            or strategy._thread_count != requested
            or size_hint % requested
        ):
            return None
        numel = env.block_sizes[strategy.block_index].numel
        if not isinstance(numel, (int, sympy.Integer)) or int(numel) != size_hint:
            return None
        for sibling in concurrent:
            if (
                isinstance(sibling, PersistentReductionStrategy)
                and sibling is not strategy
                and sibling._cute_axis_partner is None
                and sibling._cute_requested_threads == requested
                and sibling._thread_count == requested
            ):
                partner_numel = env.block_sizes[sibling.block_index].numel
                if (
                    isinstance(partner_numel, (int, sympy.Integer))
                    and int(partner_numel) % requested == 0
                    and not _cute_blocks_share_a_value(
                        env, strategy.block_index, sibling.block_index
                    )
                ):
                    return sibling
        return None

    def _cute_runtime_lane_group_params(self, group_span: int) -> tuple[str, int]:
        """Lane expression and group count that key a two-stage shared reduce's
        shared memory on the FULL runtime thread id.

        The thread-axis sizes known at this point only reflect the axes
        discovered so far. A sibling control-flow branch can still introduce a
        *redundant* thread axis later in codegen -- e.g. a free ``hl.arange``
        that another (mutually-exclusive) branch maps onto thread axis 1/2 --
        which enlarges the launch block beyond the threads counted here. Those
        extra threads re-run the reduction; if every redundant row keyed its
        shared memory on the same slots the cross-warp combine would race
        (producing intermittently wrong partial reductions). Keying the
        per-group shared memory on the flattened thread id (from the runtime
        block dims) gives each redundant row its own region; when the reduction
        owns the whole launch block (``blockDim.x == group_span``) the extra
        groups simply go unused. Only valid for a reduce group at the bottom of
        the linear lane index (thread axis 0), where ``group_span`` consecutive
        linear lanes form one group.
        """
        env = CompileEnvironment.current()
        backend = env.backend
        lane_expr = runtime_lane_expr(backend.index_type_str(env.index_dtype))
        group_count = (MAX_THREADS_PER_BLOCK + group_span - 1) // group_span
        return lane_expr, group_count

    def emit_hint_bucket_guard(
        self, state: CodegenState, block_size_var: str, expr_str: str
    ) -> None:
        if (bucket := self.strategy._cute_hint_bucket) is not None:
            # The bound kernel keys this bucket, but a compiled
            # config called directly at another size bypasses
            # the key.  The layout spans the whole bucket and the
            # dynamic extent is always masked (needs_mask), so a
            # smaller extent runs exactly; a larger one would
            # leave the rest of the dim uncovered.
            message = (
                f"CuTe kernel compiled for {expr_str} in the "
                f"power-of-2 bucket {bucket}: a config compiled at "
                "one size cannot run a larger bucket; call the "
                "Kernel itself (not the compiled config) so it "
                "rebinds"
            )
            state.codegen.host_statements.append(
                statement_from_string(
                    f"if {block_size_var} > {bucket}:\n"
                    f"    raise RuntimeError({message!r})"
                )
            )

    def emit_synthetic_lane_preamble(
        self,
        state: CodegenState,
        current_grid: DeviceGridState,
        numel: sympy.Expr,
        block_size_var: str,
    ) -> None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        block_idx = strategy.block_index
        index_var = strategy.index_var(block_idx)
        mask_var = strategy._mask_var
        synthetic_lane_var = strategy._synthetic_cute_lane_var
        assert synthetic_lane_var is not None
        axis = strategy._get_thread_axis()
        vec_width = strategy._cute_reduction_vec_width
        if vec_width > 1:
            # With exactly one V-wide chunk per thread, blocked and strided
            # layouts coincide: thread ``t`` owns ``[t*V, (t+1)*V)``.
            # Build the same mutable wrapper used by rolled/tile vec loops
            # so memory lowering can splice one LDG/STG vector around the
            # constexpr per-element loop.  The dummy one-trip outer loop is
            # retained only as a container and elided by ``wrap_body``.
            base_index_var = strategy.fn.new_var(
                f"reduction_lane_base_{block_idx}", dce=False
            )
            context = strategy.lane_context
            context.base_index_var = base_index_var
            base_expr = (
                f"({strategy._index_init_expr(block_size_var, env.index_type(), block_idx)})"
                f" * {vec_width}"
            )
            vec_for = _create_lane_loop(synthetic_lane_var, vec_width, [])
            vec_iter = expr_from_string(f"cutlass.range_constexpr({vec_width})")
            assert isinstance(vec_iter, ast.expr)
            vec_for.iter = vec_iter
            lane_body: list[ast.AST] = [
                statement_from_string(f"{base_index_var} = {base_expr}"),
                vec_for,
            ]
            outer_for = _create_lane_loop(synthetic_lane_var, 1, lane_body)
            context.lane_body = lane_body
            context.vloop = vec_for
            context.vec_lane_var = synthetic_lane_var
            current_grid.add_lane_loop(block_idx, synthetic_lane_var, vec_width)
            current_grid.vec_lane_wrappers[synthetic_lane_var] = VecLaneWrapper(
                outer_for=outer_for,
                vloop=vec_for,
                vec_lane_var=synthetic_lane_var,
                base_index_var=base_index_var,
                elide_outer_loop=True,
            )
            # The vector-loop sinking pass (``cute/sink_vector_loops.py``,
            # ``cute_vloop_sink``) interchanges this constexpr V-loop into
            # the lane loops of the other full-slice axes it encloses, so a
            # load addressed by one of their lanes can become one packet
            # (the branch-local persistent vectorizer widens the markers
            # inside each sunk V-loop; this pass widens none itself).
            strategy.fn.cute_state.lanes.vloop_sink_wrappers[synthetic_lane_var] = (
                CuteVloopWrapperFact(
                    block_id=block_idx,
                    vec_width=vec_width,
                    lane_var=synthetic_lane_var,
                    vec_lane_var=synthetic_lane_var,
                    base_index_var=base_index_var,
                    index_var=strategy.index_var(block_idx),
                    mask_var=strategy._mask_var,
                    uniform_vector_mask=strategy._mask_var is None,
                )
            )
            index_expr = f"{base_index_var} + cutlass.Int32({synthetic_lane_var})"
        else:
            # A scalar synthetic lane over a grid of one-vector tile
            # wrappers becomes a trace-time loop OUTSIDE those wrappers
            # (a per-thread register tile: one vector transaction per
            # lane, every lane's loads issued before the first store).
            # Otherwise the lane keeps its rolled innermost position.
            register_tile = (
                strategy._cute_register_tile_predicted
                and current_grid.nest_reduction_lane_outside_vector_tiles(
                    block_idx,
                    synthetic_lane_var,
                    strategy._synthetic_cute_lane_extent,
                    max_unrolled_elements=CUTE_REGISTER_TILE_MAX_ELEMENTS,
                )
            )
            if not register_tile:
                if (
                    strategy._cute_register_tile_predicted
                    and strategy._thread_count > _CUTE_WARP_REDUCTION_THREADS
                ):
                    # The multi-warp thread count was kept for a register
                    # tile that the grid does not provide; a rolled
                    # multi-warp lane loop would drop lanes (#2643), so
                    # ``generate_ast`` regenerates the kernel with the
                    # rolled nesting and its warp-capped thread count.
                    raise RegisterTileUnsupported(
                        "the grid does not provide one-vector tile "
                        "wrappers for the reduction lane"
                    )
                current_grid.add_lane_loop(
                    block_idx,
                    synthetic_lane_var,
                    strategy._synthetic_cute_lane_extent,
                )
            index_expr = (
                f"({strategy._index_init_expr(block_size_var, env.index_type(), block_idx)})"
                f" + cutlass.Int32({synthetic_lane_var}) * {strategy._thread_count}"
            )
            if strategy._cute_resident_reduction:
                index_expr = feature_index_expression(
                    strategy.fn.cute_state.lanes.resident_reduction_layouts[
                        synthetic_lane_var
                    ],
                    synthetic_lane_var,
                )
        if self.claims_thread_axis():
            current_grid.thread_axis_sizes[axis] = max(
                current_grid.thread_axis_sizes.get(axis, 1),
                strategy._thread_count,
            )
            current_grid.block_thread_axes[block_idx] = axis
        elif strategy._cute_axis_partner is not None:
            # The partner sized the axis; this block's coordinate is the
            # same thread index.
            current_grid.block_thread_axes[block_idx] = axis
        if strategy._cute_resident_reduction:
            # ``materialize_resident_reductions`` rewrites this loop and
            # its prelude as a unit.
            current_grid.undistributable_lane_vars.add(synthetic_lane_var)
        current_grid.lane_setup_statements.append(
            statement_from_string(f"{index_var} = {index_expr}")
        )
        if mask_var is not None:
            current_grid.lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {index_var} < {strategy.fn.sympy_expr(numel)}"
                )
            )

    def _cute_cross_warp_reduction_expr(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        default_value: float | bool,
        dtype: torch.dtype,
    ) -> str | None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        backend = env.backend
        if (
            strategy._thread_count <= 32
            or strategy._synthetic_cute_lane_var is not None
            or backend.is_indexed_reduction(reduction_type)
        ):
            return None

        current_grid = state.codegen.current_grid_state
        axis_sizes: dict[int, int] = {}
        if isinstance(current_grid, DeviceGridState):
            for axis, size in current_grid.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        reduction_axis = strategy._get_thread_axis()
        axis_sizes[reduction_axis] = max(
            axis_sizes.get(reduction_axis, 1), strategy._thread_count
        )

        num_threads = 1
        for size in axis_sizes.values():
            num_threads *= size
        group_span = strategy._thread_count
        if num_threads % group_span != 0:
            return None

        identity_expr = backend.cast_expr(
            constant_repr(default_value), _dtype_str(dtype)
        )
        # The two-stage shared reduce takes ``dtype`` (the accumulation dtype,
        # ``reduction_acc_dtype``) from ``type(identity)``.
        # Upcast the (possibly fp16/bf16) masked input to that same dtype so the
        # helper's ``input if mask else identity`` selection unifies cleanly and
        # the reduction still accumulates in the wider accumulation dtype.
        input_expr = backend.cast_expr(input_name, _dtype_str(dtype))

        if reduction_axis == 0:
            # A redundant thread axis can still appear later in codegen, so key
            # the shared memory on the full runtime thread id (see
            # ``_cute_runtime_lane_group_params``).
            lane_expr, group_count = self._cute_runtime_lane_group_params(group_span)
        else:
            # The two-stage shared-memory reduction assumes its ``lane_var`` is
            # the linear thread index across ALL of the launch block's threads.
            # If ``axis_sizes`` only covers a subset of the planned block dims
            # (e.g. an inner reduction strategy contributes another thread axis
            # that hasn't been entered yet), the emitted reduction would race
            # across the missing axis. Bail out and fall back to the warp-level
            # path in that case.
            planned_dims = strategy._planned_thread_dims()
            planned_block_threads = planned_dims[0] * planned_dims[1] * planned_dims[2]
            if num_threads != planned_block_threads:
                if len(HostFunction.current().device_ir.phases) > 1:
                    # The warp-level fallback shuffles at most 32 lanes, so for
                    # a >32-thread reduce it would silently drop lanes.
                    raise exc.BackendUnsupported(
                        "cute",
                        f"persistent reduction over {group_span} lanes cannot "
                        "prove its thread group under the hl.barrier() launch "
                        f"{planned_dims}",
                    )
                return None
            lane_expr = backend.thread_linear_index_expr(axis_sizes)
            if lane_expr is None:
                return None
            group_count = num_threads // group_span

        lane_var = strategy.fn.new_var("persistent_reduce_lane", dce=True)
        lane_in_group_var = strategy.fn.new_var(
            "persistent_reduce_lane_in_group", dce=True
        )
        lane_mod_pre_var = strategy.fn.new_var(
            "persistent_reduce_lane_mod_pre", dce=True
        )
        result_var = strategy.fn.new_var("persistent_reduce_result", dce=True)
        state.add_statement(f"{lane_var} = {lane_expr}")
        state.add_statement(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
        state.add_statement(f"{lane_mod_pre_var} = ({lane_in_group_var}) % 1")
        reduce = ReduceCall(
            scope=ReduceScope.SHARED_TWO_STAGE,
            op=reduction_type,
            input=input_expr,
            identity=identity_expr,
            lane_expr=lane_var,
            lane_in_group=lane_in_group_var,
            lane_mod_pre=lane_mod_pre_var,
            pre=1,
            group_span=group_span,
            group_count=group_count,
        )
        state.add_statement(f"{result_var} = {reduce.render()}")
        return result_var

    def _live_thread_axis_sizes(self, state: CodegenState) -> dict[int, int]:
        """Thread extent per launch axis.

        The active loops and the grid, and the axes every strategy of the
        kernel reserves: a sibling strategy whose loop is not active here
        (``x[t, 1::2].argmax(-1)`` rolled over thread axis 0, then a
        persistent ``x[t, 8:32].argmin(-1)`` on axis 1) still occupies its
        launch axis, so it still interleaves this reduction's lanes.
        """
        return state.codegen._strategy_thread_axis_sizes()

    def _sibling_axis_group_params(
        self, state: CodegenState
    ) -> tuple[int, int, str, int, str] | None:
        """Prove a lane-reduce marker's physical thread group.

        A plain ``warp_reduction_*(threads_in_group=N)`` folds N consecutive
        linear lanes, which is the reduction group only when the reduce axis
        occupies the lowest strides. With a sibling axis below (e.g. a
        128-thread matmul contraction on axis 0 under this 4-thread reduce axis
        on axis 1), the reduce lanes are strided by the sibling extent, so the
        finalize must use the grouped (pre-strided) reduction instead. A
        bottom-axis reduce group wider than one warp (a resident row or a
        one-vector-per-thread multi-warp row) also needs these params: its
        finalize is the cross-warp two-stage shared reduce, since a warp
        shuffle cannot span warps. Returns ``(pre, group_span, lane_expr,
        group_count, shared_lane_expr)`` -- ``lane_expr`` is the static linear
        lane the marker analysis parses, ``shared_lane_expr`` the (possibly
        different) lane keying the two-stage helper's shared memory, or ``""``
        for the same -- or ``None`` when the layout cannot be proved. Even a
        consecutive fold retains its physical coordinates: a broadcast input
        may not read the reduction index, but its owner guard must still keep
        that thread axis in the launch.
        """
        strategy = self.strategy
        env = CompileEnvironment.current()
        backend = env.backend
        reduce_extent = strategy._thread_count
        if reduce_extent <= 1:
            return None
        reduce_axis = strategy._get_thread_axis()
        axis_sizes = self._live_thread_axis_sizes(state)
        if axis_sizes.get(reduce_axis, reduce_extent) != reduce_extent:
            # Another block shares the reduce axis with a different extent;
            # the linear-lane stride math below would not isolate this axis.
            return None
        axis_sizes[reduce_axis] = reduce_extent
        pre = 1
        for axis in range(reduce_axis):
            pre *= axis_sizes.get(axis, 1)
        group_span = pre * reduce_extent
        if group_span > 32 and group_span % 32 != 0:
            return None
        num_threads = 1
        for size in axis_sizes.values():
            num_threads *= size
        if num_threads % group_span != 0:
            return None
        if strategy._cute_resident_reduction and (
            strategy._planned_thread_dims() != (reduce_extent, 1, 1) or reduce_axis != 0
        ):
            raise exc.BackendUnsupported(
                "cute", "resident reduction must own the whole physical CTA"
            )
        lane_expr = backend.thread_linear_index_expr(axis_sizes)
        if lane_expr is None:
            return None
        if (
            pre <= 1
            and group_span > _CUTE_WARP_REDUCTION_THREADS
            and not strategy._cute_resident_reduction
        ):
            # A one-vector-per-thread multi-warp row or a register tile
            # (``_cute_register_tile_predicted``): the only other ways a
            # synthetic-lane reduction keeps more than one warp.  Their
            # finalize is the two-stage shared reduce (per tile element or
            # column for the register tile), which keys its per-group shared
            # memory on the linear thread index across ALL launch-block
            # threads.  Like the non-synthetic cross-warp path, key that shared
            # memory on the full runtime thread id: a redundant thread axis can
            # still appear later in codegen, and the thread axes counted so far
            # would then let the redundant rows race on the same slots (see
            # ``_cute_runtime_lane_group_params``).  The marker's own lane
            # stays the static ``lane_expr``: the post-pass parses it to find
            # the reduce axis and the consume stores that need an owner.
            if reduce_axis == 0:
                shared_lane_expr, group_count = self._cute_runtime_lane_group_params(
                    group_span
                )
                return pre, group_span, lane_expr, group_count, shared_lane_expr
            # Above thread axis 0 the runtime id cannot isolate the group, so
            # the axes counted so far must already cover the planned launch
            # block; otherwise the emitted reduce would race across the
            # missing axis.
            planned_dims = strategy._planned_thread_dims()
            if num_threads != planned_dims[0] * planned_dims[1] * planned_dims[2]:
                return None
        return pre, group_span, lane_expr, num_threads // group_span, ""

    def _cute_sibling_axis_reduction_expr(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
        default_value: float | bool,
    ) -> str | None:
        """Thread combine for a reduce axis above live sibling thread axes.

        A plain ``warp_reduction_*`` (or the ``pre=1`` cross-warp reduce)
        folds consecutive linear lanes, which here are the sibling rows below
        the reduce axis -- e.g. ``x[tile, :, :].sum(-1)`` with the middle full
        slice on thread axis 0 -- rather than this axis.  Use the grouped
        reductions that keep the ``pre`` interleaved sibling lanes distinct.
        Returns ``None`` when the reduce axis is at the bottom of the lanes.
        """
        strategy = self.strategy
        env = CompileEnvironment.current()
        backend = env.backend
        if strategy._thread_count <= 1 or strategy._synthetic_cute_lane_var is not None:
            return None
        reduce_axis = strategy._get_thread_axis()
        axis_sizes = self._live_thread_axis_sizes(state)
        pre = 1
        for axis in range(reduce_axis):
            pre *= axis_sizes.get(axis, 1)
        if pre <= 1:
            return None
        params = self._sibling_axis_group_params(state)
        if params is None:
            raise exc.BackendUnsupported(
                "cute",
                f"{reduction_type} over thread axis {reduce_axis} interleaved "
                f"with {pre} sibling lanes below it (thread axes {axis_sizes})",
            )
        pre, group_span, lane_expr, group_count, _ = params
        # The warp combine needs the launch to match the axes up to this one
        # (they order the warp lanes); the two-stage combine keys shared
        # memory on the linear thread index, so it needs every launch axis.
        planned_dims = strategy._planned_thread_dims()
        checked_axes = (
            range(reduce_axis + 1)
            if group_span <= _CUTE_WARP_REDUCTION_THREADS
            else range(len(planned_dims))
        )
        if any(axis_sizes.get(axis, 1) != planned_dims[axis] for axis in checked_axes):
            launch = (
                "hl.barrier() launch"
                if len(HostFunction.current().device_ir.phases) > 1
                else "launch"
            )
            raise exc.BackendUnsupported(
                "cute",
                f"persistent reduction over {strategy._thread_count} lanes cannot "
                f"prove its thread group under the {launch} {planned_dims}",
            )

        def grouped_reduce(value: str, combine: str, identity: str) -> str:
            if group_span <= _CUTE_WARP_REDUCTION_THREADS:
                return _grouped_warp_reduce_expr(
                    combine, value, identity, lane_expr, pre=pre, group_span=group_span
                )
            result_var = strategy.fn.new_var("persistent_reduce_result", dce=True)
            for stmt in _grouped_two_stage_reduce_stmts(
                result_var,
                combine,
                value,
                identity,
                lane_expr,
                pre=pre,
                group_span=group_span,
                group_count=group_count,
            ):
                state.add_statement(stmt)
            return result_var

        indexed = backend.is_indexed_reduction(reduction_type)
        dtype_str = _dtype_str(
            reduction_acc_dtype(reduction_type, fake_input, fake_output)
        )
        identity_expr = backend.cast_expr(constant_repr(default_value), dtype_str)
        input_expr = backend.cast_expr(input_name, dtype_str)
        if not indexed:
            return grouped_reduce(input_expr, reduction_type, identity_expr)
        # argmin/argmax: the winning value, then the lowest index holding it.
        value_var = strategy.fn.new_var("persistent_reduce_value", dce=True)
        state.add_statement(
            f"{value_var} = "
            + grouped_reduce(
                input_expr,
                "min" if reduction_type == "argmin" else "max",
                identity_expr,
            )
        )
        index_dtype_str = backend.index_type_str(env.index_dtype)
        max_index = backend.cast_expr(
            repr(torch.iinfo(env.index_dtype).max), index_dtype_str
        )
        index = strategy.broadcast_str(
            strategy.argreduce_position_var(), fake_input, dim
        )
        candidate = argreduce_candidate_expr(index, input_expr, value_var, max_index)
        return backend.cast_expr(
            grouped_reduce(candidate, "min", max_index),
            backend.dtype_str(fake_output.dtype),
        )

    def emit_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST | None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        backend = env.backend
        if backend.is_indexed_reduction(reduction_type):
            packed = codegen_packed_reduction(
                state,
                strategy,
                None,
                input_name,
                reduction_type,
                dim,
                fake_input,
                fake_output,
            )
            if packed is not None:
                return packed
        acc_dtype = reduction_acc_dtype(reduction_type, fake_input, fake_output)
        default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
        if (
            strategy._synthetic_cute_lane_var is not None
            and isinstance(default, (float, int, bool))
            and not self._lane_reduce_marker_unsupported(state)
            and (threads := self._lane_reduce_threads_in_group()) is not None
        ):
            # The reduction axis is split across a synthetic per-thread lane
            # loop: the single warp reduction only covers one lane's worth of
            # elements. Emit a marker so the ``split_lane_loop_reductions``
            # post-pass produces the two-pass (accumulate across lanes ->
            # warp-combine across ``threads`` -> consume) structure.
            identity_expr = backend.cast_expr(
                constant_repr(default), _dtype_str(acc_dtype)
            )
            group_params = self._reshape_merged_reduction_group_params()
            physical_group = self._reshape_physical_reduction_group_params(state)
            if physical_group is not None:
                group_params = physical_group
                previous = (
                    strategy.fn.cute_state.lanes.reshape_physical_fallbacks.setdefault(
                        strategy._synthetic_cute_lane_var, physical_group
                    )
                )
                if previous != physical_group:
                    raise exc.BackendUnsupported(
                        "cute", "reshape reduction has conflicting physical groups"
                    )
            owner_lane = self._lane_reduce_owner(state, reshape_group=group_params)
            broadcast_owner = (
                state.fx_node is not None
                and isinstance(state.fx_node.args[0], torch.fx.Node)
                and explicitly_replicated_axis(state.fx_node.args[0], dim)
            )
            sibling_params = (
                self._sibling_axis_group_params(state) if group_params is None else None
            )

            def marker(value: str, combine: str, identity: str) -> str:
                if group_params is not None:
                    group_pre, group_span, group_lane_expr = group_params
                    return _lane_reduce_marker_expr(
                        value,
                        combine,
                        identity,
                        threads,
                        group_pre=group_pre,
                        group_span=group_span,
                        group_lane_expr=group_lane_expr,
                        owner_lane=owner_lane,
                        broadcast_owner=broadcast_owner,
                    )
                if sibling_params is not None:
                    (
                        group_pre,
                        group_span,
                        group_lane_expr,
                        group_count,
                        shared_lane_expr,
                    ) = sibling_params
                    return _lane_reduce_marker_expr(
                        value,
                        combine,
                        identity,
                        threads,
                        group_pre=group_pre,
                        group_span=group_span,
                        group_lane_expr=group_lane_expr,
                        group_count=group_count,
                        owner_lane=owner_lane,
                        broadcast_owner=broadcast_owner,
                        shared_lane_expr=shared_lane_expr,
                    )
                return _lane_reduce_marker_expr(
                    value,
                    combine,
                    identity,
                    threads,
                    owner_lane=owner_lane,
                    broadcast_owner=broadcast_owner,
                )

            if not backend.is_indexed_reduction(reduction_type):
                expr = marker(input_name, reduction_type, identity_expr)
            else:
                # argmin/argmax: the winning value across the lanes, then the
                # lowest index holding it, as two dependent lane reductions.
                # The post-pass slices each marker's input by name, so it
                # stays a plain variable; the accumulator applies the cast.
                value_var = strategy.fn.new_var("lane_reduce_value", dce=True)
                state.add_statement(
                    f"{value_var} = "
                    + marker(
                        input_name,
                        "min" if reduction_type == "argmin" else "max",
                        identity_expr,
                    )
                )
                max_index = backend.cast_expr(
                    repr(torch.iinfo(env.index_dtype).max),
                    backend.index_type_str(env.index_dtype),
                )
                index = strategy.broadcast_str(
                    strategy.index_var(strategy.block_index), fake_input, dim
                )
                input_expr = backend.cast_expr(input_name, _dtype_str(acc_dtype))
                candidate_var = strategy.fn.new_var("lane_reduce_index", dce=True)
                state.add_statement(
                    f"{candidate_var} = "
                    + argreduce_candidate_expr(index, input_expr, value_var, max_index)
                )
                expr = backend.cast_expr(
                    marker(candidate_var, "min", max_index),
                    backend.dtype_str(fake_output.dtype),
                )
            return expr_from_string(
                strategy.maybe_reshape(expr, dim, fake_input, fake_output)
            )
        if (
            strategy._synthetic_cute_lane_var is not None
            and backend.is_indexed_reduction(reduction_type)
        ):
            # Without the lane marker each synthetic lane would overwrite the
            # previous lane's result.
            raise exc.BackendUnsupported(
                "cute",
                f"{reduction_type} over a reduction axis split across synthetic "
                "lanes whose lane reduction cannot be split into two passes",
            )
        if isinstance(default, (float, int, bool)):
            cross_warp = self._cute_sibling_axis_reduction_expr(
                state,
                input_name,
                reduction_type,
                dim,
                fake_input,
                fake_output,
                default,
            ) or self._cute_cross_warp_reduction_expr(
                state, input_name, reduction_type, default, acc_dtype
            )
        else:
            cross_warp = None
        if cross_warp is None:
            return None
        return expr_from_string(
            strategy.maybe_reshape(cross_warp, dim, fake_input, fake_output)
        )


class CuteLoopedReductionEmitter(
    CuteReductionEmitter[LoopedReductionStrategy], LoopedReductionEmitter
):
    """Lane loop, vector lattice, cluster split and finalize of a rolled
    reduction."""

    def plan_max_threads(
        self, fn: DeviceFunction, block_index: int, max_threads: int
    ) -> int:
        # CuTe argreduce uses cute.arch.warp_reduction which is only
        # correct for threads_in_group<=32. Cap to warp size whenever
        # the rolled reduction will fold an indexed reduction over this
        # block.
        if _block_has_indexed_reduction(fn, block_index):
            max_threads = min(max_threads, _CUTE_WARP_REDUCTION_THREADS)
        return max_threads

    def plan_requested_thread_count(
        self, fn: DeviceFunction, block_index: int, thread_count: int
    ) -> int:
        env = CompileEnvironment.current()
        # Autotunable per-rdim thread count: fewer threads per row
        # (each covering more elements via the lane loop) is often
        # faster for memory-bound row reductions. 0 = auto (keep the
        # chunk-derived count above).
        requested = env.config_spec.num_threads.config_get(
            cast("list[int]", fn.config.config.get("num_threads", []) or []),
            block_index,
            0,
        )
        if isinstance(requested, int) and 0 < requested < thread_count:
            thread_count = requested
        return thread_count

    def plan_lane_loop(
        self, fn: DeviceFunction, block_index: int, block_size: int, thread_count: int
    ) -> None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        if thread_count > 0 and block_size > thread_count:
            strategy._cute_reduction_lane_extent = (
                block_size + thread_count - 1
            ) // thread_count
            strategy._loop_block_size = (
                thread_count * strategy._cute_reduction_lane_extent
            )
            strategy._cute_reduction_lane_var = fn.new_var(
                f"reduction_lane_{block_index}",
                dce=False,
            )
            # Read autotuner-selected vector width and partition lane extent
            # into outer × inner = lane_extent/V × V.  When V==1 (default)
            # this preserves the original scalar codegen.
            cute_vector_widths = cast(
                "list[int]",
                fn.config.config.get("cute_vector_widths", []) or [],
            )
            vec_width = env.config_spec.cute_vector_widths.config_get(
                cute_vector_widths,
                block_index,
                1,
            )
            if (
                isinstance(vec_width, int)
                and vec_width > 1
                and strategy._cute_reduction_lane_extent % vec_width == 0
            ):
                strategy._cute_reduction_vec_width = vec_width
                strategy._cute_reduction_lane_extent = (
                    strategy._cute_reduction_lane_extent // vec_width
                )

    def plan_chunk_mask(self, block_index: int, mask_var: str | None) -> None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        if mask_var is not None and strategy._cute_reduction_vec_width > 1:
            # ``memory_ops`` is a registered codegen module ``backend_registry``
            # reloads; bind the helper at call time, not at import.
            from .memory_ops import cute_known_multiple

            strategy._cute_reduction_chunk_uniform_mask = cute_known_multiple(
                env,
                env.block_sizes[block_index].numel,
                strategy._cute_reduction_vec_width,
            )

    def plan_rolled_cluster(self, state: CodegenState, numel: sympy.Expr) -> sympy.Expr:
        self._maybe_apply_cute_rolled_cluster(state)
        if self.strategy._cute_rolled_cluster_n > 1:
            # Only the successfully claimed cluster needs a concrete extent.
            # The coverage proof used this exact metadata-guarded bound value.
            env = CompileEnvironment.current()
            numel = sympy.Integer(shape_env_size_hint(env.shape_env, numel))
        return numel

    def _maybe_apply_cute_rolled_cluster(self, state: CodegenState) -> None:
        """Decide whether ``cute_cluster_n`` applies to this rolled
        reduction (mirrors ``lane_cluster_split.apply_cluster_split`` for the
        tile-loop form).

        When applied, each of the ``cluster_n`` CTAs of a thread-block
        cluster rolls over its own contiguous ``numel/cluster_n`` slice of
        the reduction range, the finalize combines the partials across the
        cluster with one DSM exchange (every CTA receives the full result,
        so downstream row-scalar stores stay redundant-but-identical), and
        the consume sweep re-iterates only the CTA's slice — so loads and
        stores are sliced automatically.  The owning ``DeviceFunction``'s
        ``cute_state.simt_cluster_n`` is set so the host-side call emits the
        extra grid dim + cluster launch shape.
        """
        strategy = self.strategy
        if strategy._cute_rolled_cluster_checked:
            return
        strategy._cute_rolled_cluster_checked = True
        env = CompileEnvironment.current()
        cl = strategy.fn.config.config.get("cute_cluster_n", 1)
        if not isinstance(cl, int) or cl <= 1:
            return
        if strategy.fn.cute_state.lanes.simt_cluster_n > 1:
            # Another device loop's strategy already claimed the cluster
            # rank; splitting a second axis on the same rank would leave
            # only the diagonal rank x rank slices covered.
            return
        # The finalize must go through the cross-warp two-stage path (the
        # warp-shuffle path never crosses CTAs), and argreduce has no
        # cluster combine.
        if (
            strategy._thread_count <= 32
            or strategy._thread_count % 32 != 0
            or _block_has_indexed_reduction(strategy.fn, strategy.block_index)
        ):
            return
        # Statements outside the roll loop execute once per cluster CTA —
        # benign for plain (idempotent) stores, but a read-modify-write
        # would repeat ``cluster_n`` times.
        if HostFunction.current().device_ir.has_atomic_ops():
            return
        # Sibling thread axes (e.g. a multi-row grid tile with block_m > 1)
        # would make the whole-CTA cluster reduce fold unrelated rows.
        grid_state = state.codegen.current_grid_state
        if grid_state is not None and any(
            size > 1 for size in grid_state.thread_axis_sizes.values()
        ):
            return
        numel = env.specialize_expr(env.block_sizes[strategy.block_index].numel)
        # Static tails can be padded to whole chunks per CTA. Each rank owns
        # a disjoint contiguous padded slice, and the original extent still
        # masks the loads and reduction inputs. Dynamic extents retain the
        # exact input-metadata guard used by the unmasked cluster schedule.
        static_numel = isinstance(numel, sympy.Integer)
        if not static_numel and not env.specialized_multiple(
            numel, cl * strategy._loop_block_size
        ):
            return
        numel_int = shape_env_size_hint(env.shape_env, numel)
        if numel_int <= 0:
            return
        slice_len = (
            -(-numel_int // (cl * strategy._loop_block_size))
            * strategy._loop_block_size
            if static_numel
            else numel_int // cl
        )
        # Cluster rank-relative begin/end expressions use cutlass.Int32.
        # A logical extent can fit while the padded final slice overflows.
        if static_numel and slice_len * cl > 2**31 - 1:
            return
        strategy._cute_rolled_cluster_n = cl
        strategy._cute_rolled_cluster_slice_len = slice_len
        strategy.fn.cute_state.lanes.simt_cluster_n = cl
        if slice_len * cl == numel_int:
            strategy._mask_var = None
        elif strategy._mask_var is None:
            strategy._mask_var = strategy.fn.new_var(
                f"mask_{strategy.block_index}", dce=True
            )
            strategy._cute_reduction_chunk_uniform_mask = env.known_multiple(
                numel, strategy._cute_reduction_vec_width
            )

    def _active_thread_axis_sizes(
        self, state: CodegenState, device_loop: DeviceLoopState
    ) -> dict[int, int]:
        strategy = self.strategy
        axis_sizes = strategy.fn.tile_strategy.thread_axis_sizes()
        # Earlier phases may have emitted axes outside this root's strategy
        # plan. Those threads still participate in shared-memory reductions.
        for axis, size in enumerate(state.codegen.max_thread_block_dims):
            axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        seen: set[int] = set()
        for loops in state.codegen.active_device_loops.values():
            for loop_state in loops:
                if not isinstance(loop_state, (DeviceLoopState, DeviceGridState)):
                    continue
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                for axis, size in loop_state.thread_axis_sizes.items():
                    axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        current_grid = state.codegen.current_grid_state
        if isinstance(current_grid, DeviceGridState):
            for axis, size in current_grid.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        for axis, size in device_loop.thread_axis_sizes.items():
            axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        return axis_sizes

    def _cute_lanes_below_reduce_axis(
        self, state: CodegenState, device_loop: DeviceLoopState
    ) -> tuple[int, dict[int, int]]:
        """``(pre, axis_sizes)``: the live threads on the launch axes below this
        rolled reduction's thread axis, which interleave with its lanes."""
        strategy = self.strategy
        axis_sizes = self._active_thread_axis_sizes(state, device_loop)
        pre = 1
        for axis in range(strategy._get_thread_axis()):
            pre *= axis_sizes.get(axis, 1)
        return pre, axis_sizes

    def _cute_sibling_axis_reduction_expr(
        self,
        state: CodegenState,
        device_loop: DeviceLoopState,
        acc: str,
        reduction_type: str,
        default_value: float | bool,
        dtype: torch.dtype,
    ) -> str | None:
        """Thread combine of a rolled reduction above live sibling thread axes.

        A plain ``warp_reduction_*`` (and the ``pre=1`` cross-warp reduce of
        ``_cute_cross_warp_reduction_expr``) folds consecutive linear lanes,
        which here are the sibling lanes below the reduce axis:
        ``out[tile] = buf[:, :].sum(-1)`` with the row dim on thread axis 0 and
        a rolled column chunk on axis 1 summed rows instead of column lanes.
        Use the grouped reductions that keep the ``pre`` interleaved sibling
        lanes distinct, as ``PersistentReductionStrategy`` does.  Returns
        ``None`` when the reduce axis is at the bottom of the lanes.
        """
        strategy = self.strategy
        env = CompileEnvironment.current()
        backend = env.backend
        if strategy._thread_count <= 1:
            return None
        pre, axis_sizes = self._cute_lanes_below_reduce_axis(state, device_loop)
        if pre <= 1:
            return None
        reduce_axis = strategy._get_thread_axis()

        def refuse(reason: str) -> NoReturn:
            raise exc.BackendUnsupported(
                "cute",
                f"rolled {reduction_type} over thread axis {reduce_axis} "
                f"interleaved with {pre} sibling lanes below it: {reason}",
            )

        if strategy._cute_rolled_cluster_n > 1:
            refuse("the cluster-split reduce combines whole CTAs")
        if (
            axis_sizes.get(reduce_axis, strategy._thread_count)
            != strategy._thread_count
        ):
            refuse("another block shares the reduce axis")
        axis_sizes[reduce_axis] = strategy._thread_count
        group_span = pre * strategy._thread_count
        num_threads = 1
        for size in axis_sizes.values():
            num_threads *= size
        if (
            group_span > _CUTE_WARP_REDUCTION_THREADS
            and group_span % _CUTE_WARP_REDUCTION_THREADS != 0
        ) or num_threads % group_span != 0:
            refuse("the thread axes do not tile the reduce group")
        # The warp combine needs the launch to match the axes up to this one
        # (they order the warp lanes); the two-stage combine keys shared memory
        # on the linear thread index, so it needs every launch axis.
        planned_dims = tuple(
            starmap(
                max,
                zip(
                    strategy._planned_thread_dims(),
                    state.codegen.max_thread_block_dims,
                    strict=True,
                ),
            )
        )
        checked_axes = (
            range(reduce_axis + 1)
            if group_span <= _CUTE_WARP_REDUCTION_THREADS
            else range(len(planned_dims))
        )
        if any(axis_sizes.get(axis, 1) != planned_dims[axis] for axis in checked_axes):
            refuse(f"its thread group is not provable under the launch {planned_dims}")
        lane_expr = backend.thread_linear_index_expr(axis_sizes)
        if lane_expr is None:
            refuse("no linear thread index for the block layout")
        identity_expr = backend.cast_expr(
            constant_repr(default_value), _dtype_str(dtype)
        )
        if group_span <= _CUTE_WARP_REDUCTION_THREADS:
            return _grouped_warp_reduce_expr(
                reduction_type,
                acc,
                identity_expr,
                lane_expr,
                pre=pre,
                group_span=group_span,
            )
        result_var = strategy.fn.new_var("looped_reduce_result", dce=True)
        device_loop.outer_suffix.extend(
            _grouped_two_stage_reduce_stmts(
                result_var,
                reduction_type,
                acc,
                identity_expr,
                lane_expr,
                pre=pre,
                group_span=group_span,
                group_count=num_threads // group_span,
            )
        )
        return result_var

    def _cute_cross_warp_reduction_expr(
        self,
        state: CodegenState,
        device_loop: DeviceLoopState,
        input_name: str,
        reduction_type: str,
        default_value: float | bool,
        dtype: torch.dtype,
    ) -> str | None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        backend = env.backend
        # Once ``_maybe_apply_cute_rolled_cluster`` sliced the roll range,
        # every CTA holds only a partial sum — any fallback to a
        # non-cluster reduction would silently drop the peer CTAs'
        # contributions, so every bail-out below must hard-fail instead.
        cluster_n = strategy._cute_rolled_cluster_n

        def unsupported(reason: str) -> None:
            if cluster_n > 1:
                raise exc.BackendUnsupported(
                    "cute",
                    f"cute_cluster_n > 1 rolled reduction: {reason}",
                )

        if strategy._thread_count <= 32 or backend.is_indexed_reduction(reduction_type):
            unsupported("requires a cross-warp non-indexed reduce")
            return None

        axis_sizes = self._active_thread_axis_sizes(state, device_loop)
        reduction_axis = strategy._get_thread_axis()
        axis_sizes[reduction_axis] = max(
            axis_sizes.get(reduction_axis, 1), strategy._thread_count
        )
        num_threads = 1
        for size in axis_sizes.values():
            num_threads *= size
        group_span = strategy._thread_count
        if num_threads % group_span != 0:
            unsupported("thread axes do not tile the reduce group")
            return None
        if reduction_axis == 0 and cluster_n == 1:
            # An independent output axis can be entered after this reduction,
            # but still enlarge the launch. Each duplicate reduction needs its
            # own shared-memory slots, keyed by the complete physical thread ID.
            lane_expr = runtime_lane_expr(backend.index_type_str(env.index_dtype))
            group_count = (MAX_THREADS_PER_BLOCK + group_span - 1) // group_span
        else:
            # Other layouts require every planned or previously emitted axis
            # to participate in the shared reduction's linear thread index.
            planned_dims = tuple(
                starmap(
                    max,
                    zip(
                        strategy._planned_thread_dims(),
                        state.codegen.max_thread_block_dims,
                        strict=True,
                    ),
                )
            )
            planned_block_threads = planned_dims[0] * planned_dims[1] * planned_dims[2]
            if num_threads != planned_block_threads:
                unsupported("another strategy contributes unentered thread axes")
                return None
            lane_expr = backend.thread_linear_index_expr(axis_sizes)
            if lane_expr is None:
                unsupported("no linear thread index for the block layout")
                return None
            group_count = num_threads // group_span

        identity_expr = backend.cast_expr(
            constant_repr(default_value), _dtype_str(dtype)
        )
        if cluster_n > 1:
            if group_count != 1:
                # The cluster reduce combines across the WHOLE CTA; a
                # multi-group reduce (several rows per CTA) would fold
                # unrelated rows together.
                raise exc.BackendUnsupported(
                    "cute",
                    "cute_cluster_n > 1 requires a whole-CTA reduce group "
                    "for the rolled reduction",
                )
            if reduction_type not in ("sum", "max", "min"):
                raise exc.BackendUnsupported(
                    "cute",
                    f"cute_cluster_n > 1 does not support {reduction_type!r} "
                    "rolled reductions",
                )
            if dtype != torch.float32:
                # The cluster exchange buffers every partial as Float32
                # (see _cute_grouped_reduce_cluster) — a wider accumulator
                # would silently lose precision through the round trip.
                raise exc.BackendUnsupported(
                    "cute",
                    "cute_cluster_n > 1 requires an fp32 reduction "
                    f"accumulator, got {dtype}",
                )
            return self._emit_rolled_cluster_reduce(
                device_loop=device_loop,
                input_name=input_name,
                reduction_type=reduction_type,
                identity_expr=identity_expr,
                lane_expr=lane_expr,
                group_span=group_span,
                cluster_n=cluster_n,
            )
        lane_var = strategy.fn.new_var("looped_reduce_lane", dce=True)
        lane_in_group_var = strategy.fn.new_var("looped_reduce_lane_in_group", dce=True)
        lane_mod_pre_var = strategy.fn.new_var("looped_reduce_lane_mod_pre", dce=True)
        result_var = strategy.fn.new_var("looped_reduce_result", dce=True)
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_var} = {lane_expr}")
        )
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
        )
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_mod_pre_var} = ({lane_in_group_var}) % 1")
        )
        reduce = ReduceCall(
            scope=ReduceScope.SHARED_TWO_STAGE,
            op=reduction_type,
            input=input_name,
            identity=identity_expr,
            lane_expr=lane_var,
            lane_in_group=lane_in_group_var,
            lane_mod_pre=lane_mod_pre_var,
            pre=1,
            group_span=group_span,
            group_count=group_count,
        )
        device_loop.outer_suffix.append(
            statement_from_string(f"{result_var} = {reduce.render()}")
        )
        return result_var

    def _emit_rolled_cluster_reduce(
        self,
        *,
        device_loop: DeviceLoopState,
        input_name: str,
        reduction_type: str,
        identity_expr: str,
        lane_expr: str,
        group_span: int,
        cluster_n: int,
    ) -> str:
        """Finalize a cluster-split rolled reduction: combine the CTA's
        partial across its warps AND the ``cluster_n`` peer CTAs with one
        DSM exchange (every CTA receives the full result).  The SMEM
        receive buffer and mbarrier are allocated + initialized in the
        kernel preamble; ``codegen_function_def`` emits ONE
        ``mbarrier_init_fence`` + cluster arrive/wait covering every site.
        """
        strategy = self.strategy
        buf_var, mbar_var = _cute_cluster_reduce_smem_vars(
            strategy.fn, group_span, cluster_n
        )
        lane_var = strategy.fn.new_var("looped_reduce_lane", dce=True)
        result_var = strategy.fn.new_var("looped_reduce_result", dce=True)
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_var} = {lane_expr}")
        )
        reduce = ReduceCall(
            scope=ReduceScope.CLUSTER,
            op=reduction_type,
            input=input_name,
            identity=identity_expr,
            lane_expr=lane_var,
            buffers=(buf_var, mbar_var),
            group_span=group_span,
            cluster_n=cluster_n,
        )
        device_loop.outer_suffix.append(
            statement_from_string(f"{result_var} = {reduce.render()}")
        )
        return result_var

    def register_dynamic_trip_count(
        self,
        state: CodegenState,
        numel: sympy.Expr,
        offset_var: str,
        block_size_var: str,
    ) -> None:
        """Expose a symbolic extent's trip count as a constexpr kernel param.

        A symbolic extent has no static trip count, so the host computes
        ``ceil(extent / block)`` and passes it as ``cutlass.Constexpr``.  The
        roll then iterates ``range(0, TRIPS * block, block)``: the same trips
        as ``range(0, extent, block)`` but with a trace-time constant bound
        the DSL can unroll, and the two-pass load fuser sizes its per-thread
        cache as ``TRIPS * lanes * V`` (exact for every runtime extent) while
        making its profitability decision from the bound size hint.  Each
        distinct trip count is a distinct constexpr value and so traces the
        kernel again: launch schemas that bake tensor shapes already trace
        once per shape, while shape-agnostic schemas (matmul wrapper plans)
        gain one trace per trip count.  The entry is keyed by the roll's
        offset variable, the loop target the later AST passes match on.
        """
        strategy = self.strategy
        trips = state.device_function.cute_state.lanes.dynamic_reduction_trips
        if offset_var in trips or isinstance(numel, (int, sympy.Integer)):
            return
        if strategy._cute_rolled_cluster_n > 1:
            return
        env = CompileEnvironment.current()
        block_mapping, _ = find_block_size_symbols(numel)
        if block_mapping:
            return
        trips_var = f"_REDUCTION_TRIPS_{strategy.block_index}"
        host_numel = HostFunction.current().sympy_expr(
            numel, state.device_function.config
        )
        state.device_function.constexpr_arg_with_host_def(
            trips_var,
            f"({host_numel} + {block_size_var} - 1) // {block_size_var}",
        )
        size_hint = shape_env_size_hint(env.shape_env, numel)
        hint_trips = max(1, -(-size_hint // strategy._loop_block_size))
        trips[offset_var] = (hint_trips, trips_var)

    def emit_lane_loop_body(
        self,
        state: CodegenState,
        inner_body: list[ast.AST],
        numel: sympy.Expr,
        offset_var: str,
        index_var: str,
        block_size_var: str,
    ) -> list[ast.AST]:
        strategy = self.strategy
        env = CompileEnvironment.current()
        block_index = strategy.block_index
        reduction_lane_var = strategy._cute_reduction_lane_var
        vec = strategy._cute_reduction_vec_width
        # A vec width above one unrolls the lane body via a constexpr
        # V-loop: the hoisted integer-vector load is bitcast per element so
        # both the reduce sweep and the consume sweep stay on the scalar
        # pipeline (the CuTe DSL cannot safely subscript bf16/fp16 vectors).
        consume_unroll = vec > 1
        # The lane body of this sweep (``LaneLoopContext``): the per-(tensor,
        # base pointer, guard) cache of hoisted vec loads so the dispatcher
        # reuses one hoist per pair instead of emitting a fresh vec load on
        # every call, the per-lane-iter base index the ``unroll``-mode hoists
        # compute the vec pointer from, the constexpr V-loop node that
        # positions the hoists and the store flushes, the lane body list the
        # dispatcher splices into (None until the consume-unroll form below
        # builds it) and the store flush sites already spliced in, in source
        # order.  Rebuilt per sweep; the chunk predicate persists.
        context = strategy.lane_context
        context.vec_loads = {}
        context.base_index_var = None
        context.vloop = None
        context.lane_body = None
        context.vec_stores = []
        context.vec_lane_var = None
        vec_lane_var: str | None = None
        base_expr: str = ""
        if reduction_lane_var is not None:
            if vec > 1:
                # base = offset + thread_idx*V + lane*(THREADS*V)
                base_expr = (
                    f"{offset_var} + "
                    f"{strategy._index_init_expr(f'({block_size_var})', env.index_type(), block_index)} "
                    f"* {vec} + "
                    f"cutlass.Int32({reduction_lane_var}) * {strategy._thread_count * vec}"
                )
                if consume_unroll:
                    vec_lane_var = strategy.fn.new_var(
                        f"reduction_vec_lane_{block_index}",
                        dce=False,
                    )
                    context.base_index_var = strategy.fn.new_var(
                        f"reduction_lane_base_{block_index}",
                        dce=False,
                    )
                    context.vec_lane_var = vec_lane_var
                    # index_var = base + vi  (used inside the constexpr loop)
                    inner_body[0] = statement_from_string(
                        f"{index_var} = {context.base_index_var} + cutlass.Int32({vec_lane_var})"
                    )
                else:
                    inner_body[0] = statement_from_string(f"{index_var} = {base_expr}")
            else:
                inner_body[0] = statement_from_string(
                    f"{index_var} = {offset_var} + {strategy._index_init_expr(f'({block_size_var})', env.index_type(), block_index)} + cutlass.Int32({reduction_lane_var}) * {strategy._thread_count}"
                )
        chunk_mask_stmts: list[ast.AST] = []
        if (mask_var := strategy._mask_var) is not None:
            numel_expr = state.sympy_expr(numel)
            if consume_unroll and context.base_index_var is not None:
                chunk_in_bounds = strategy.cute_chunk_in_bounds_expr(state)
                if strategy._cute_reduction_chunk_uniform_mask:
                    # ``numel % V == 0``: the chunk base decides every lane of
                    # the chunk, so the mask is defined once above the V-loop
                    # and vector loads/stores can be predicated on it.
                    chunk_mask_stmts.append(
                        statement_from_string(f"{mask_var} = {chunk_in_bounds}")
                    )
                else:
                    # The tail chunk may straddle the extent: keep the
                    # per-element mask and expose the whole-chunk predicate
                    # for the vector fast path; its negation selects the
                    # per-element tail.
                    context.chunk_full_var = strategy.fn.new_var(
                        f"reduction_chunk_full_{block_index}", dce=True
                    )
                    chunk_mask_stmts.append(
                        statement_from_string(
                            f"{context.chunk_full_var} = {chunk_in_bounds}"
                        )
                    )
            if not strategy._cute_reduction_chunk_uniform_mask or not chunk_mask_stmts:
                inner_body.append(
                    statement_from_string(f"{mask_var} = {index_var} < {numel_expr}")
                )
        body: list[ast.AST] = inner_body
        if reduction_lane_var is not None:
            if consume_unroll and vec_lane_var is not None:
                # for vi in cutlass.range_constexpr(V): ...
                vec_for = cast(
                    "ast.For",
                    ast.parse(
                        f"for {vec_lane_var} in cutlass.range_constexpr({vec}):\n"
                        f"    pass"
                    ).body[0],
                )
                vec_for.body = inner_body  # type: ignore[assignment]
                # The lane-loop body holds the per-lane base index, then any
                # dispatcher-requested vec hoists, then the constexpr loop.
                base_stmt = statement_from_string(
                    f"{context.base_index_var} = {base_expr}"
                )
                lane_body: list[ast.AST] = [
                    base_stmt,
                    *chunk_mask_stmts,
                    vec_for,
                ]
                body = [
                    _create_lane_loop(
                        reduction_lane_var,
                        strategy._cute_reduction_lane_extent,
                        lane_body,
                    )
                ]
                # Record the V-loop node so hoists (inserted before it) and
                # vec-store flushes (appended after it) can locate it even
                # once it is no longer the last lane_body entry.
                context.vloop = vec_for
                # Stash the lane body list so the dispatcher can splice
                # hoists in (BETWEEN base_stmt and vec_for) as it runs.
                context.lane_body = lane_body
            else:
                body = [
                    _create_lane_loop(
                        reduction_lane_var,
                        strategy._cute_reduction_lane_extent,
                        inner_body,
                    )
                ]
        return body

    def rolled_range(
        self,
        state: CodegenState,
        offset_var: str,
        block_size_var: str,
        range_begin: str,
        range_end: str,
    ) -> tuple[str, str]:
        strategy = self.strategy
        env = CompileEnvironment.current()
        dynamic_trips = (
            state.device_function.cute_state.lanes.dynamic_reduction_trips.get(
                offset_var
            )
        )
        if dynamic_trips is not None:
            # Same ``ceil(extent / block)`` trips as ``range(0, extent, block)``
            # (the bounds mask still covers the tail), but the bound is a
            # trace-time constant, so the DSL can unroll the short roll and
            # keep the trip-indexed register cache in registers instead of
            # dynamically indexed local memory.
            range_end = f"{dynamic_trips[1]} * {block_size_var}"
        if strategy._cute_rolled_cluster_n > 1:
            # Cluster split: each of the ``cluster_n`` CTAs rolls over its
            # own contiguous slice (the launch adds a grid dim of
            # ``cluster_n`` CTAs per outer index); the finalize combines
            # the partials across the cluster.
            slice_len = strategy._cute_rolled_cluster_slice_len
            rank_expr = env.backend.program_id_expr(1, index_dtype="cutlass.Int32")
            range_begin = f"{rank_expr} * {slice_len}"
            range_end = f"{rank_expr} * {slice_len} + {slice_len}"
        return range_begin, range_end

    def emit_packed_argreduce(
        self,
        state: CodegenState,
        device_loop: DeviceLoopState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST | None:
        strategy = self.strategy
        backend = CompileEnvironment.current().backend
        if not backend.is_indexed_reduction(reduction_type):
            return None
        return codegen_packed_reduction(
            state,
            strategy,
            device_loop,
            input_name,
            reduction_type,
            dim,
            fake_input,
            fake_output,
        )

    def emit_cross_thread_reduction(
        self,
        state: CodegenState,
        device_loop: DeviceLoopState,
        acc: str,
        reduction_type: str,
        default: float | bool,
        acc_dtype: torch.dtype,
    ) -> str | None:
        return self._cute_sibling_axis_reduction_expr(
            state, device_loop, acc, reduction_type, default, acc_dtype
        ) or self._cute_cross_warp_reduction_expr(
            state, device_loop, acc, reduction_type, default, acc_dtype
        )

    def check_indexed_reduction_lanes(
        self, state: CodegenState, device_loop: DeviceLoopState, reduction_type: str
    ) -> None:
        strategy = self.strategy
        if (
            strategy._thread_count > 1
            and self._cute_lanes_below_reduce_axis(state, device_loop)[0] > 1
        ):
            # The argreduce finalize folds consecutive lanes too.
            raise exc.BackendUnsupported(
                "cute",
                f"rolled {reduction_type} over thread axis "
                f"{strategy._get_thread_axis()} interleaved with sibling "
                "lanes below it",
            )


class CuteBlockReductionEmitter(
    CuteReductionEmitter[BlockReductionStrategy], BlockReductionEmitter
):
    """Cross-thread combines of a reduction over a tile dim."""

    def check_static_thread_extent(self, reduction_type: str) -> None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        if (
            symbolic_extent := strategy.fn.tile_strategy.symbolic_thread_extent_expr(
                strategy.block_index
            )
        ) is not None:
            # The dim lives one element per thread on a launch axis whose
            # extent is a kernel argument (``hl.tile(n, block_size=bsz)``).
            # Every combine across threads is sized at codegen, where the
            # extent is unknown: the strided and direct forms find no thread
            # axis and the fallback below passes each thread's own element
            # through as the total.
            raise exc.BackendUnsupported(
                env.backend.name,
                f"{reduction_type} over tile dim {strategy.block_index}, whose "
                f"{symbolic_extent} threads are sized by a kernel argument: the "
                "combine across them needs a static thread extent",
            )

    def emit_sequence_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        fake_output: torch.Tensor,
        default: float | bool,
    ) -> str | None:
        if (
            self.strategy.fn.config.config.get("cute_reduction_sequence", "scalar")
            != "scalar"
        ):
            return sequence_reduction_expr(
                self, state, input_name, reduction_type, fake_output, default
            )
        return None

    def emit_carried_lane_loop_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        acc_dtype: torch.dtype,
        default: float | bool,
    ) -> str | None:
        if (
            not self._lane_reduce_marker_unsupported(state)
            and (threads := self._lane_reduce_threads_in_group()) is not None
        ):
            # The block is split across a per-thread lane loop. The
            # single-pass lane loop can only produce a per-lane partial,
            # but every consumer needs the full reduction. Emit a marker
            # that the ``split_lane_loop_reductions`` post-pass rewrites
            # into a two-pass (accumulate across lanes -> combine across
            # ``threads`` -> consume) lane structure.
            return self._lane_loop_reduction_expr(
                state,
                input_name,
                reduction_type,
                dim,
                fake_input,
                acc_dtype,
                default,
                threads,
            )
        return None

    def _lane_loop_group_params(
        self,
    ) -> tuple[int, int, int, str] | None:
        """Return ``(pre, group_span, group_count, lane_expr)`` for a tile block
        that is reduced over its inner (tiled) dim AND carries a runtime lane
        loop, or ``None`` when no de-interleaving is required.

        When the reduced block is mapped to a thread axis ABOVE a sibling tile
        axis (e.g. ``hl.tile([o, d])`` where ``d`` is reduced, ``d`` on
        ``thread_idx[1]`` and ``o`` on ``thread_idx[0]``), the threads that
        share a row are strided by the sibling axis extent (stride 32 here), so
        the reduce group is spread across warps. A plain
        ``cute.arch.warp_reduction_*`` would fold together CONSECUTIVE lanes
        (different rows). Compute the grouped/strided parameters that the
        cross-warp ``_cute_grouped_reduce_shared_two_stage`` helper needs:

        * ``pre`` — product of live thread extents on axes *below* the reduce
          axis (the sibling rows that must stay distinct);
        * ``group_span`` — ``pre`` times the reduce-axis extent (the lanes that
          form one reduction);
        * ``group_count`` — the number of independent groups in the CTA;
        * ``lane_expr`` — the linear thread index across all live thread axes.

        Returns ``None`` (so the caller keeps the plain warp-reduce / no-op
        finalize) for contiguous groups (``pre == 1``) fitting one warp.
        Strided groups use a single-warp grouped reduce when
        ``group_span <= 32``. Wider groups, including contiguous ones, use
        the two-stage shared reduce when ``group_span`` is a multiple of 32;
        groups straddling a warp boundary without that alignment are rejected.
        """
        strategy = self.strategy
        env = CompileEnvironment.current()
        backend = env.backend
        block_axes, axis_sizes = self._active_thread_layout()
        reduce_axis = block_axes.get(strategy.block_index)
        if reduce_axis is None:
            reduce_axis = self._aliased_active_thread_axis(block_axes)
        if reduce_axis is None:
            return None
        # Under an ``hl.barrier()`` launch another phase may run more lanes on
        # the sibling axes than this phase's loops; the finalize must stride
        # by the shared launch, and the launcher checks the assumed layout.
        _widen_lane_layout_for_barrier_phases(
            strategy._codegen,
            axis_sizes,
            subject=f"lane-loop reduction over tile block {strategy.block_index}",
        )
        # Live thread extents per axis (sibling axes included) so the linear
        # lane index strides are computed correctly.
        logical_axis_sizes = {
            axis: size for axis, size in axis_sizes.items() if size > 1
        }
        if reduce_axis not in logical_axis_sizes:
            return None
        # Earlier full-slice reductions can own physical axes absent from
        # the active tile mapping. Their lanes still stride this reduction.
        for axis, size in enumerate(strategy._planned_thread_dims()):
            if size > 1:
                logical_axis_sizes.setdefault(axis, size)
        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        reduce_extent = logical_axis_sizes[reduce_axis]
        if pre <= 1 and reduce_extent <= 32:
            # The reduce axis is already at the bottom of the linear lane
            # index and fits one warp: consecutive warp lanes belong to the
            # reduction, so the plain warp reduce is correct.  A wider
            # bottom-axis group (> 32 threads) still needs the cross-warp
            # shared reduce below: a warp shuffle cannot span warps.
            return None
        group_span = pre * reduce_extent
        num_threads = 1
        for size in logical_axis_sizes.values():
            num_threads *= size
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if (
            (group_span > 32 and group_span % 32 != 0)
            or num_threads % group_span != 0
            or lane_expr is None
        ):
            # A reduce group that straddles a warp boundary (or an uneven
            # group tiling of the CTA) cannot be de-interleaved by either the
            # single-warp grouped reduce or the two-stage shared reduce.
            raise exc.BackendUnsupported(
                "cute",
                "lane-loop reduction group is interleaved with a sibling thread "
                f"axis but is not warp-aligned (pre={pre}, span={group_span}, "
                f"threads={num_threads})",
            )
        return pre, group_span, num_threads // group_span, lane_expr

    def _active_thread_layout(self) -> tuple[dict[int, int], dict[int, int]]:
        strategy = self.strategy
        axis_sizes: dict[int, int] = {}
        block_axes: dict[int, int] = {}
        seen: set[int] = set()
        for loops in strategy._codegen.active_device_loops.values():
            for loop_state in loops:
                if not isinstance(loop_state, (DeviceLoopState, DeviceGridState)):
                    continue
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                for axis, size in loop_state.thread_axis_sizes.items():
                    axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
                block_axes.update(loop_state.block_thread_axes)
        current_grid = strategy._codegen.current_grid_state
        if current_grid is not None:
            for axis, size in current_grid.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
            block_axes.update(current_grid.block_thread_axes)
        # Full-slice axes need not be in the active loop nest. Their threads
        # still determine the stride between rows of a tiled reduction.
        for other in strategy.fn.tile_strategy.strategies:
            if isinstance(
                other, (PersistentReductionStrategy, LoopedReductionStrategy)
            ):
                count = other._reduction_thread_count()
                axis = strategy.fn.tile_strategy.thread_axis_for_strategy(other)
                if count > 1 and axis is not None:
                    block_axes[other.block_index] = axis
                    axis_sizes[axis] = max(axis_sizes.get(axis, 1), count)
        return block_axes, axis_sizes

    def _aliased_active_thread_axis(self, block_axes: dict[int, int]) -> int | None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        target_block = strategy.block_index
        for candidate_block_id, axis in block_axes.items():
            if candidate_block_id == target_block:
                return axis
            source = env.block_sizes[candidate_block_id].block_size_source
            value = getattr(source, "value", None)
            if isinstance(value, torch.SymInt):
                if env.get_block_id(value) == target_block:
                    return axis
            elif isinstance(value, int):
                target_size = env.block_sizes[target_block].size
                if isinstance(target_size, (int, torch.SymInt)) and env.known_equal(
                    target_size, value
                ):
                    return axis
        return None

    def _aliased_strategy_block_id(self) -> int | None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        target_block = strategy.block_index
        for other in strategy.fn.tile_strategy.strategies:
            for candidate_block_id in other.block_ids:
                if candidate_block_id == target_block:
                    return candidate_block_id
                source = env.block_sizes[candidate_block_id].block_size_source
                value = getattr(source, "value", None)
                if isinstance(value, torch.SymInt):
                    if env.get_block_id(value) == target_block:
                        return candidate_block_id
                elif isinstance(value, int):
                    target_size = env.block_sizes[target_block].size
                    if isinstance(target_size, (int, torch.SymInt)) and env.known_equal(
                        target_size, value
                    ):
                        return candidate_block_id
        return None

    def _launch_thread_axis_extent(self, axis: int, extent: int) -> int:
        """``extent`` widened to the launch's thread count along ``axis``.

        Sibling loop paths reuse a CUDA axis and the launch takes the widest
        of them, so a narrower sibling (a 16-row ``hl.tile`` beside a
        32-thread K loop) runs with surplus threads: their tile mask fails,
        so they hold the reduction's identity, but they execute every
        collective of the body.  A cross-thread combine over the axis has to
        span them.  A group sized to the strategy's own extent leaves the
        surplus threads reducing among themselves, so a value every thread of
        the row is meant to hold -- a per-column sum multiplied back into the
        row and stored by all of them -- differs between the real and the
        surplus writers of one address, and the shared-memory combines index
        slots past their allocation.  ``_normalize_shared_tile_thread_extents``
        widens SIMT tile strategies to the launch ahead of codegen, but not in
        a kernel with a matmul, whose layouts keep their own thread contracts.
        """
        strategy = self.strategy
        return max(extent, strategy._codegen.launch_thread_axis_sizes().get(axis, 1))

    def emit_strided_thread_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        default_value: float | bool,
        acc_dtype: torch.dtype,
    ) -> str | None:
        strategy = self.strategy
        env = CompileEnvironment.current()
        backend = env.backend
        current_grid = strategy._codegen.current_grid_state
        allow_lane_axis_fallback = (
            current_grid is not None and current_grid.has_lane_loops()
        )
        normalized_dim = dim if dim >= 0 else fake_input.ndim + dim

        def debug(*parts: object) -> None:
            return None

        def block_thread_extent_hint(block_id: int) -> int | None:
            extent = strategy.fn.tile_strategy.thread_extent_for_block_id(block_id)
            if extent is not None:
                return extent
            configured_threads = env.config_spec.num_threads.config_get(
                strategy.fn.config.num_threads, block_id, 0
            )
            if configured_threads > 0:
                return configured_threads
            configured_block_size = strategy.fn.resolved_block_size(block_id)
            return (
                configured_block_size
                if isinstance(configured_block_size, int)
                else None
            )

        def active_loop_states() -> list[DeviceLoopState | DeviceGridState]:
            loop_states: list[DeviceLoopState | DeviceGridState] = []
            seen: set[int] = set()
            for loops in strategy._codegen.active_device_loops.values():
                for loop_state in loops:
                    if not isinstance(loop_state, (DeviceLoopState, DeviceGridState)):
                        continue
                    key = id(loop_state)
                    if key in seen:
                        continue
                    seen.add(key)
                    loop_states.append(loop_state)
            return loop_states

        loop_states = active_loop_states()
        info_by_block: dict[int, LoopDimInfo] = {}
        if isinstance(current_grid, DeviceGridState):
            info_by_block.update(current_grid.block_id_to_info)
        for loop_state in loop_states:
            for block_id, info in loop_state.block_id_to_info.items():
                info_by_block.setdefault(block_id, info)
        planned_dims = strategy._planned_thread_dims()
        active_thread_blocks: list[tuple[int, int, int, LoopDimInfo]] = []
        seen_thread_blocks: set[int] = set()
        active_block_axes, active_axis_sizes = self._active_thread_layout()
        active_block_ids = set(info_by_block) | set(active_block_axes)
        for block_id in active_block_ids:
            if block_id in seen_thread_blocks:
                continue
            axis = active_block_axes.get(block_id)
            if axis is None:
                continue
            live_extent = active_axis_sizes.get(axis, 1)
            if live_extent <= 1:
                continue
            extent = block_thread_extent_hint(block_id)
            if extent is None:
                extent = live_extent
            else:
                extent = min(extent, live_extent)
            if extent <= 1:
                continue
            if extent > live_extent:
                continue
            info = info_by_block.get(block_id)
            if info is None:
                size = env.block_sizes[block_id].size
                if not isinstance(size, (int, torch.SymInt)):
                    size = extent
                end_expr = _to_sympy(size)
                info = LoopDimInfo(
                    end_var_name=state.sympy_expr(end_expr),
                    end_expr=end_expr,
                )
            active_thread_blocks.append((block_id, axis, extent, info))
            seen_thread_blocks.add(block_id)
        active_thread_blocks.sort(key=operator.itemgetter(1, 0))
        active_block_axes = {
            block_id: axis for block_id, axis, _, _ in active_thread_blocks
        }
        active_axis_sizes: dict[int, int] = {}
        for _, axis, extent, _ in active_thread_blocks:
            active_axis_sizes[axis] = max(active_axis_sizes.get(axis, 1), extent)

        def resolve_tensor_dim_mapping() -> dict[int, tuple[int, int, int]]:
            mapping: dict[int, tuple[int, int, int]] = {}
            used_block_ids: set[int] = set()
            used_axes: set[int] = set()
            for dim_idx in range(fake_input.ndim):
                dim_size = fake_input.size(dim_idx)
                candidates: dict[tuple[int, int, int], int] = {}
                block_id = env.resolve_block_id(dim_size)
                if block_id is not None and block_id in active_block_axes:
                    axis = active_block_axes[block_id]
                    extent = block_thread_extent_hint(block_id)
                    if extent is not None:
                        candidates[(block_id, axis, extent)] = 0
                for candidate_block_id, axis, extent, info in active_thread_blocks:
                    matches_end = isinstance(
                        dim_size, (int, torch.SymInt)
                    ) and info.is_end_matching(dim_size)
                    matches_thread_extent = isinstance(
                        dim_size, (int, torch.SymInt)
                    ) and env.known_equal(dim_size, extent)
                    candidate_source = getattr(
                        env.block_sizes[candidate_block_id].block_size_source,
                        "value",
                        None,
                    )
                    matches_source_value = (
                        isinstance(dim_size, torch.SymInt)
                        and isinstance(candidate_source, torch.SymInt)
                        and candidate_source._sympy_() == dim_size._sympy_()
                    )
                    if (
                        not matches_end
                        and not matches_thread_extent
                        and not matches_source_value
                    ):
                        continue
                    priority = 3
                    if matches_source_value:
                        priority = 1
                    elif matches_end:
                        priority = 2
                    candidate = (
                        candidate_block_id,
                        axis,
                        extent,
                    )
                    previous = candidates.get(candidate)
                    if previous is None or priority < previous:
                        candidates[candidate] = priority
                chosen: tuple[int, int, int] | None = None
                ordered_candidates = sorted(
                    candidates.items(),
                    key=lambda item: (item[1], item[0][1], item[0][0]),
                )
                for candidate, _priority in ordered_candidates:
                    block_id, axis, _ = candidate
                    if block_id in used_block_ids or axis in used_axes:
                        continue
                    chosen = candidate
                    break
                if (
                    chosen is None
                    and allow_lane_axis_fallback
                    and dim_idx != normalized_dim
                ):
                    for candidate_block_id, axis, extent, _info in sorted(
                        active_thread_blocks, key=operator.itemgetter(1, 0)
                    ):
                        if candidate_block_id in used_block_ids or axis in used_axes:
                            continue
                        chosen = (candidate_block_id, axis, extent)
                        break
                if chosen is None and ordered_candidates:
                    chosen = ordered_candidates[0][0]
                if chosen is None:
                    continue
                mapping[dim_idx] = chosen
                used_block_ids.add(chosen[0])
                used_axes.add(chosen[1])
            return mapping

        if strategy._reduction_block_is_serial():
            debug("skip serial", strategy.block_index)
            return None
        if state.fx_node is not None:
            for arg in state.fx_node.args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                target_name = getattr(arg.target, "__name__", "")
                if any(
                    name in target_name
                    for name in ("sum", "prod", "mean", "amax", "amin")
                ):
                    debug("skip nested reduction arg", target_name)
                    return None
        if strategy._reduction_block_has_lane_loops():
            # Lane loops serialize part of the logical tile in Python rather
            # than mapping it to actual threads. Thread-reduction fast paths
            # assume every participating axis is backed by a live thread, so
            # they are invalid under active lane loops.
            debug("skip lane loops")
            return None

        tensor_dim_mapping = resolve_tensor_dim_mapping()
        mapped_block_ids = {block_id for block_id, _, _ in tensor_dim_mapping.values()}
        logical_axes = {
            axis for _, axis, _ in tensor_dim_mapping.values() if axis is not None
        }
        reduce_axis: int | None = None
        reduce_thread_extent: int | None = None
        if 0 <= normalized_dim < fake_input.ndim:
            mapping = tensor_dim_mapping.get(normalized_dim)
            if mapping is not None:
                _, reduce_axis, reduce_thread_extent = mapping

        block_axes = dict(active_block_axes)
        axis_sizes = dict(active_axis_sizes)
        if reduce_axis is not None and reduce_thread_extent is not None:
            axis_sizes[reduce_axis] = max(
                axis_sizes.get(reduce_axis, 1), reduce_thread_extent
            )
            if 0 <= reduce_axis < len(strategy._codegen.max_thread_block_dims):
                strategy._codegen.max_thread_block_dims[reduce_axis] = max(
                    strategy._codegen.max_thread_block_dims[reduce_axis],
                    reduce_thread_extent,
                )
        if reduce_axis is None:
            reduce_axis = self._aliased_active_thread_axis(block_axes)
        if reduce_axis is None:
            aliased_block_id = self._aliased_strategy_block_id()
            # Only treat the reduce dim as a strided thread reduction when the
            # aliased block is actually backed by a *live* thread axis. A block
            # with ``block_size == 1`` (a grid/serial dim such as a size-1
            # contributor axis) reports no live thread extent; its
            # ``_thread_axis_map`` entry still records a phantom local axis that
            # collides with an unrelated sibling block's real thread axis (e.g.
            # the M tile on CuTe). Using that phantom axis would fold the
            # reduction across the sibling's tile instead of squeezing the
            # size-1 dim, so bail to the loop-carried / passthrough path.
            if (
                aliased_block_id is not None
                and strategy.fn.tile_strategy.thread_extent_for_block_id(
                    aliased_block_id
                )
                is None
            ):
                aliased_block_id = None
            if aliased_block_id is not None:
                reduce_axis = strategy.fn.tile_strategy.thread_axis_for_block_id(
                    aliased_block_id
                )
                reduce_thread_extent = block_thread_extent_hint(aliased_block_id)
                if reduce_axis is not None and reduce_thread_extent is not None:
                    if (
                        reduce_axis >= len(planned_dims)
                        or planned_dims[reduce_axis] <= 1
                        or reduce_thread_extent > planned_dims[reduce_axis]
                    ):
                        reduce_axis = None
                        reduce_thread_extent = None
                    else:
                        axis_sizes[reduce_axis] = max(
                            axis_sizes.get(reduce_axis, 1), reduce_thread_extent
                        )
                        if (
                            0
                            <= reduce_axis
                            < len(strategy._codegen.max_thread_block_dims)
                        ):
                            strategy._codegen.max_thread_block_dims[reduce_axis] = max(
                                strategy._codegen.max_thread_block_dims[reduce_axis],
                                reduce_thread_extent,
                            )
        if reduce_axis is None:
            block_strategy = strategy.fn.tile_strategy.block_id_to_strategy.get(
                (strategy.block_index,)
            )
            # A block_strategy's starting axis belongs to its first block that
            # holds threads.  A block of one holds none (its
            # ``_thread_axis_map`` entry is the next block's axis), so a
            # reduction over it has nothing to combine across: taking the
            # axis would fold the sibling block's tile instead of passing
            # the single element through.  A single-thread reduction block
            # reports an extent of 1 but claims no axis either
            # (``_claims_thread_axis``): the axis the dispatcher parks it on
            # may be a wider tile's, so it must not be reduced across.
            if (
                block_strategy is not None
                and block_strategy.thread_axes_used() > 0
                and strategy.fn.tile_strategy.thread_extent_for_block_id(
                    strategy.block_index
                )
                is not None
            ):
                reduce_axis = strategy.fn.tile_strategy.thread_axis_for_strategy(
                    block_strategy
                )
            if reduce_axis is not None:
                hint = backend.reduction_threads_hint(
                    strategy.block_size_var(strategy.block_index)
                )
                if (
                    hint is not None
                    and reduce_axis < len(planned_dims)
                    and planned_dims[reduce_axis] > 1
                    and hint <= planned_dims[reduce_axis]
                ):
                    axis_sizes[reduce_axis] = max(axis_sizes.get(reduce_axis, 1), hint)
                else:
                    reduce_axis = None
        if reduce_axis is None:
            debug("skip no reduce axis", tuple(fake_input.size()), dim)
            return None
        logical_axes.add(reduce_axis)
        logical_axis_sizes: dict[int, int] = {}
        # A grid axis packed several tiles per CTA (``cute_tiles_per_block``)
        # lives above the reduce axis as independent sub-groups: it must count
        # toward the group count and the linear lane, or the sub-groups would
        # share the helper's shared-memory slots.
        packed = packed_grid_block(env, strategy.fn.config)
        packed_block = None if packed is None else packed[0]
        for block_id, axis, extent, _info in active_thread_blocks:
            if (
                block_id in mapped_block_ids
                or block_id >= strategy.block_index
                or axis < reduce_axis
                or block_id == packed_block
            ):
                # Even a broadcast input (for example a rank-one validity
                # count) is executed by the physical row threads below this
                # axis. Omitting their stride makes a consecutive-lane warp
                # reduction combine different rows instead of this axis.
                logical_axis_sizes[axis] = max(
                    logical_axis_sizes.get(axis, 1),
                    extent,
                )
        if reduce_axis not in logical_axis_sizes and 0 <= reduce_axis < len(
            strategy._codegen.max_thread_block_dims
        ):
            reduce_size = axis_sizes.get(reduce_axis, 1)
            if reduce_thread_extent is None:
                reduce_size = max(
                    reduce_size,
                    strategy._codegen.max_thread_block_dims[reduce_axis],
                )
            logical_axis_sizes[reduce_axis] = reduce_size
        if not logical_axis_sizes:
            debug("skip no logical axis sizes", tuple(fake_input.size()), dim)
            return None
        for axis, size in logical_axis_sizes.items():
            if 0 <= axis < len(strategy._codegen.max_thread_block_dims):
                strategy._codegen.max_thread_block_dims[axis] = max(
                    strategy._codegen.max_thread_block_dims[axis], size
                )
        if reduce_thread_extent is None and 0 <= reduce_axis < len(
            strategy._codegen.max_thread_block_dims
        ):
            logical_axis_sizes[reduce_axis] = max(
                logical_axis_sizes.get(reduce_axis, 1),
                strategy._codegen.max_thread_block_dims[reduce_axis],
            )

        # Earlier full-slice reductions can own physical axes absent from
        # the active tile mapping. Their lanes still stride this reduction.
        for axis, size in enumerate(strategy._planned_thread_dims()):
            if size > 1:
                logical_axis_sizes.setdefault(axis, size)

        # Every axis the tile is distributed over spans the launch's threads
        # along it: the surplus threads of a sibling narrower than the launch
        # execute this combine holding the identity, and the lane expression
        # has to follow the physical layout (``_launch_thread_axis_extent``).
        # An axis of extent 1 is not distributed: every thread holds the same
        # element there, so the launch's threads along it are not combined.
        strategy_axis_sizes = dict(logical_axis_sizes)
        for axis, size in strategy_axis_sizes.items():
            if size > 1:
                logical_axis_sizes[axis] = self._launch_thread_axis_extent(axis, size)
        surplus_threads = logical_axis_sizes != strategy_axis_sizes

        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        reduce_extent = logical_axis_sizes.get(reduce_axis, 1)
        group_span = pre * reduce_extent
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if lane_expr is None:
            debug("skip no lane expr", tuple(fake_input.size()), dim)
            return None

        # Inductor's scalar input has already applied reduction promotion (for
        # example Int32 sum -> Int64). Shared selections and storage must use
        # that same type, including when only the broadcast count is reduced.
        dtype = _dtype_str(acc_dtype)
        identity_expr = backend.cast_expr(constant_repr(default_value), dtype)
        input_expr = (
            input_name
            if acc_dtype == fake_input.dtype
            else backend.cast_expr(input_name, dtype)
        )
        num_threads = 1
        for size in logical_axis_sizes.values():
            num_threads *= size
        tensor_thread_axes: set[int] = set()
        tensor_thread_footprint = 1
        for _block_id, axis, extent in tensor_dim_mapping.values():
            if axis is None or extent is None or axis in tensor_thread_axes:
                continue
            tensor_thread_axes.add(axis)
            tensor_thread_footprint *= extent
        if (
            reduce_axis is not None
            and reduce_thread_extent is not None
            and reduce_axis not in tensor_thread_axes
        ):
            tensor_thread_axes.add(reduce_axis)
            tensor_thread_footprint *= reduce_thread_extent
        actual_threads = 1
        planned_dims = strategy.fn.tile_strategy.thread_block_dims()
        for axis, (recorded, planned) in enumerate(
            zip(strategy._codegen.max_thread_block_dims, planned_dims, strict=True)
        ):
            if axis not in logical_axis_sizes:
                continue
            size = max(recorded, planned)
            actual_threads *= max(size, 1)
        if num_threads > actual_threads:
            # Some logical axes are being serialized (for example via lane loops)
            # rather than mapped to actual threads. The strided thread-reduction
            # path assumes every participating lane is backed by a live thread, so
            # using it here would read unwritten SMEM partials.
            debug(
                "skip actual threads",
                tuple(fake_input.size()),
                dim,
                num_threads,
                actual_threads,
                logical_axis_sizes,
            )
            return None
        # Skip to the direct ``cute.arch.warp_reduction_*`` path when the
        # entire CTA is a single warp (num_threads == group_span <= 32):
        # the standard ``call_reduction_function`` can emit a one-shot
        # warp_reduction with ``threads_in_group=group_span``.
        #
        # When ``num_threads > group_span`` (e.g. warp-per-row layouts
        # with multiple warps per CTA, each owning one row), keep the
        # ``_cute_grouped_reduce_warp`` path at the bottom — it picks
        # the right per-warp reduce even when other thread axes coexist
        # within the CTA.  The "skip" shortcut would route through
        # ``_needs_loop_carried_accumulator``, which returns True when
        # the reduction block is no longer in ``active_device_loops``
        # (e.g. ``cute_dynamic_row_sum``'s ``acc.sum(-1)`` after the
        # inner ``hl.tile`` exits) and would silently drop the reduce.
        # Under surplus threads the direct path would size its group by the
        # block's own extent, so the strided form is kept.  The same drop
        # happens with one warp per CTA when the block's loop has exited
        # (one row per CTA on 32 column threads: ``acc.mean(dim=1)`` after
        # the loop became each thread's own column), so the shortcut is
        # only taken by a block with a live thread axis.
        if (
            pre <= 1
            and group_span <= 32
            and num_threads == group_span
            and not surplus_threads
            and strategy._reduction_block_has_live_thread_axis()
        ):
            debug(
                "skip small direct",
                tuple(fake_input.size()),
                dim,
                "block",
                strategy.block_index,
                "reduce_axis",
                reduce_axis,
                "pre",
                pre,
                "group_span",
                group_span,
                "mapping",
                tensor_dim_mapping,
                "active_thread_blocks",
                active_thread_blocks,
                "logical_axis_sizes",
                logical_axis_sizes,
            )
            return None
        debug(
            "use strided",
            tuple(fake_input.size()),
            dim,
            "block",
            strategy.block_index,
            "reduce_axis",
            reduce_axis,
            "pre",
            pre,
            "group_span",
            group_span,
            "mapping",
            tensor_dim_mapping,
            "active_thread_blocks",
            active_thread_blocks,
            "logical_axis_sizes",
            logical_axis_sizes,
        )
        if group_span > 32:
            assert num_threads % group_span == 0, (
                f"num_threads ({num_threads}) must be divisible by "
                f"group_span ({group_span})"
            )
            group_count = num_threads // group_span
            lane_var = strategy.fn.new_var("strided_lane", dce=True)
            lane_in_group_var = strategy.fn.new_var("strided_lane_in_group", dce=True)
            lane_mod_pre_var = strategy.fn.new_var("strided_lane_mod_pre", dce=True)
            state.add_statement(f"{lane_var} = {lane_expr}")
            state.add_statement(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
            state.add_statement(f"{lane_mod_pre_var} = ({lane_in_group_var}) % {pre}")
        elif self._lane_reduce_cluster_n() > 1:
            raise exc.BackendUnsupported(
                "cute",
                "cute_cluster_n > 1 requires a cross-warp reduce group; "
                "this config reduces within single warps",
            )

        def grouped_reduce(
            value: str, combine: str, identity: str, value_dtype: torch.dtype
        ) -> str:
            if group_span <= 32:
                call = ReduceCall(
                    scope=ReduceScope.WARP,
                    op=combine,
                    input=value,
                    identity=identity,
                    lane_expr=lane_expr,
                    pre=pre,
                    group_span=group_span,
                ).render()
                self.record_lane_fold_site(call)
                return call
            smem_budget_bytes = thread_budget.shared_memory_budget_bytes()
            if group_span % 32 == 0:
                warps_per_group = group_span // 32
                partials_size = group_count * pre * warps_per_group
                results_size = group_count * pre
                if (
                    _cute_reduction_smem_bytes(
                        partials_size + results_size, value_dtype
                    )
                    > smem_budget_bytes
                ):
                    raise exc.BackendUnsupported(
                        "cute", "strided reduction exceeds the shared-memory budget"
                    )
                if self._lane_reduce_cluster_n() > 1 and (pre != 1 or group_count != 1):
                    # The cluster split covers the whole CTA; an
                    # interleaved / multi-group reduce cannot combine
                    # across the cluster and would silently drop the
                    # peer CTAs' contributions.
                    raise exc.BackendUnsupported(
                        "cute",
                        "cute_cluster_n > 1 requires a whole-CTA reduce "
                        "group (pre == 1 and group_count == 1)",
                    )
                return self._strided_thread_reduction_expr_shared_two_stage(
                    state=state,
                    input_name=value,
                    reduction_type=combine,
                    acc_dtype=value_dtype,
                    identity_expr=identity,
                    lane_var=lane_var,
                    lane_in_group_var=lane_in_group_var,
                    lane_mod_pre_var=lane_mod_pre_var,
                    pre=pre,
                    group_span=group_span,
                    group_count=group_count,
                )
            if (
                _cute_reduction_smem_bytes(num_threads + group_count * pre, value_dtype)
                > smem_budget_bytes
            ):
                raise exc.BackendUnsupported(
                    "cute", "strided reduction exceeds the shared-memory budget"
                )
            if self._lane_reduce_cluster_n() > 1:
                raise exc.BackendUnsupported(
                    "cute",
                    "cute_cluster_n > 1 is not supported with the "
                    "shared-tree reduction layout",
                )
            return self._strided_thread_reduction_expr_shared_tree(
                state=state,
                input_name=value,
                reduction_type=combine,
                fake_input=fake_input,
                identity_expr=identity,
                lane_var=lane_var,
                lane_in_group_var=lane_in_group_var,
                lane_mod_pre_var=lane_mod_pre_var,
                pre=pre,
                group_span=group_span,
                num_threads=num_threads,
                group_count=group_count,
            )

        if not backend.is_indexed_reduction(reduction_type):
            return grouped_reduce(input_expr, reduction_type, identity_expr, acc_dtype)
        # argmin/argmax: the winning value, then the lowest index holding it,
        # as the persistent block_strategy's grouped combine does.
        value_var = strategy.fn.new_var("strided_reduce_value", dce=True)
        state.add_statement(
            f"{value_var} = "
            + grouped_reduce(
                input_expr,
                "min" if reduction_type == "argmin" else "max",
                identity_expr,
                acc_dtype,
            )
        )
        index_dtype = env.index_dtype
        index_dtype_str = backend.index_type_str(index_dtype)
        max_index = backend.cast_expr(
            repr(torch.iinfo(index_dtype).max), index_dtype_str
        )
        index = strategy.broadcast_str(
            strategy.argreduce_position_var(), fake_input, dim
        )
        candidate = argreduce_candidate_expr(index, input_expr, value_var, max_index)
        return backend.cast_expr(
            grouped_reduce(candidate, "min", max_index, index_dtype),
            backend.dtype_str(torch.int64),
        )

    def _lane_loop_marker_expr(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        acc_dtype: torch.dtype,
        default: float | bool,
        threads: int,
        *,
        strided_restore: bool = False,
    ) -> str:
        """Emit the two-pass lane-reduction marker for a lane-looped block.

        The ``split_lane_loop_reductions`` post-pass rewrites the marker into
        accumulate-across-lanes -> combine-across-``threads`` -> consume.  When
        the reduced tile dim sits ABOVE a sibling tile axis on the linear
        thread index, the grouped/strided params are attached so the finalize
        de-interleaves the sibling rows (single-warp grouped reduce or the
        cross-warp two-stage shared reduction) instead of folding consecutive
        lanes that belong to different rows.
        """
        strategy = self.strategy
        env = CompileEnvironment.current()
        identity_expr = env.backend.cast_expr(
            constant_repr(default), _dtype_str(acc_dtype)
        )
        reduce_axis = strategy.fn.tile_strategy.thread_axis_for_block_id(
            strategy.block_index
        )
        if (
            threads > 1
            and reduce_axis is not None
            and self._launch_thread_axis_extent(reduce_axis, threads) != threads
        ):
            # The surplus threads of the launch walk this lane loop too; with
            # a strided lane layout their elements alias the real threads'
            # and a group spanning them would count those twice, while a
            # group of the block's own threads leaves them a partial every
            # owner-guarded consume store would race with.
            raise exc.BackendUnsupported(
                "cute",
                "lane-looped reduction over a tile axis narrower than the "
                f"launch (threads={threads}, launch="
                f"{self._launch_thread_axis_extent(reduce_axis, threads)})",
            )
        group_params = self._lane_loop_group_params()
        owner_lane = self._lane_reduce_owner(state)
        cluster_n = self._lane_reduce_cluster_n()
        if cluster_n > 1 and group_params is None:
            raise exc.BackendUnsupported(
                "cute",
                "cute_cluster_n > 1 requires the cross-warp grouped reduce path",
            )
        if group_params is None:
            return _lane_reduce_marker_expr(
                input_name,
                reduction_type,
                identity_expr,
                threads,
                owner_lane=owner_lane,
                strided_restore=strided_restore,
            )
        group_pre, group_span, group_count, group_lane_expr = group_params
        return _lane_reduce_marker_expr(
            input_name,
            reduction_type,
            identity_expr,
            threads,
            group_pre=group_pre,
            group_span=group_span,
            group_lane_expr=group_lane_expr,
            group_count=group_count,
            group_cluster_n=cluster_n,
            owner_lane=owner_lane,
            strided_restore=strided_restore,
        )

    def emit_device_lane_loop_marker(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        acc_dtype: torch.dtype,
        default: float | bool,
    ) -> str | None:
        """Two-pass marker for a block distributed by a ``DeviceLoopState``
        lane loop, or ``None`` when this reduction is not lane-looped.

        Without this, a lane-looped tile reduction whose block also owns a
        live thread axis falls through to the per-element strided thread
        reduction: every synthetic lane then pays a full cross-thread (warp
        shuffle or shared-memory two-stage) combine, e.g. 256 CTA-wide
        shared reductions per row tile instead of one after the lane loop.
        """
        strategy = self.strategy
        if (
            not isinstance(default, (float, int, bool))
            or not strategy._reduction_block_in_device_lane_loop()
            or self._lane_reduce_marker_unsupported(state)
        ):
            return None
        for loops in strategy._codegen.active_device_loops.values():
            for loop_state in loops:
                if (
                    isinstance(loop_state, DeviceLoopState)
                    and strategy.block_index in loop_state.lane_loop_blocks
                    and isinstance(loop_state.strategy, PerThreadNDTileStrategy)
                    and loop_state.strategy.lane_context(strategy.block_index).vec_width
                    != 1
                ):
                    # The marker would sit inside the block's constexpr vector
                    # loop, which the two-pass lane splitter does not fold;
                    # the vector-fold (``hoist_warp_reduce``) and resident
                    # sequence lowerings own that shape.
                    return None
        threads = self._lane_reduce_threads_in_group()
        if threads is None:
            return None
        # The loop body keeps its per-element strided form; when the two-pass
        # split is unsafe the marker is finalized per lane, which is complete
        # for lane-carry consumers, and the shares are totalled over the lanes
        # for any other lane-invariant consumer (``_restore_lane_markers``).
        return self._lane_loop_reduction_expr(
            state,
            input_name,
            reduction_type,
            dim,
            fake_input,
            acc_dtype,
            default,
            threads,
            strided_restore=True,
        )

    def _lane_loop_reduction_expr(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        acc_dtype: torch.dtype,
        default: float | bool,
        threads: int,
        *,
        strided_restore: bool = False,
    ) -> str:
        """The lane-loop marker of a value reduction, or for argmin/argmax
        the winning value across the lanes and then the lowest index holding
        it, as two dependent lane reductions (as the persistent strategy's
        synthetic lanes do)."""
        strategy = self.strategy
        backend = CompileEnvironment.current().backend
        if not backend.is_indexed_reduction(reduction_type):
            return self._lane_loop_marker_expr(
                state,
                input_name,
                reduction_type,
                acc_dtype,
                default,
                threads,
                strided_restore=strided_restore,
            )
        index_dtype = CompileEnvironment.current().index_dtype
        value_var = strategy.fn.new_var("lane_reduce_value", dce=True)
        state.add_statement(
            f"{value_var} = "
            + self._lane_loop_marker_expr(
                state,
                input_name,
                "min" if reduction_type == "argmin" else "max",
                acc_dtype,
                default,
                threads,
                strided_restore=strided_restore,
            )
        )
        max_index = torch.iinfo(index_dtype).max
        index = strategy.broadcast_str(
            strategy.argreduce_position_var(), fake_input, dim
        )
        input_expr = backend.cast_expr(input_name, _dtype_str(acc_dtype))
        # The post-pass slices each marker's input by name, so the candidate
        # stays a plain variable.
        candidate_var = strategy.fn.new_var("lane_reduce_index", dce=True)
        max_index_expr = backend.cast_expr(
            repr(max_index), backend.index_type_str(index_dtype)
        )
        state.add_statement(
            f"{candidate_var} = "
            + argreduce_candidate_expr(index, input_expr, value_var, max_index_expr)
        )
        return backend.cast_expr(
            self._lane_loop_marker_expr(
                state,
                candidate_var,
                "min",
                index_dtype,
                max_index,
                threads,
                strided_restore=strided_restore,
            ),
            backend.dtype_str(torch.int64),
        )

    def _strided_thread_reduction_expr_shared_two_stage(
        self,
        *,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        acc_dtype: torch.dtype,
        identity_expr: str,
        lane_var: str,
        lane_in_group_var: str,
        lane_mod_pre_var: str,
        pre: int,
        group_span: int,
        group_count: int,
    ) -> str:
        strategy = self.strategy
        cluster_n = self._lane_reduce_cluster_n()
        if cluster_n == 1 and defer_collective(
            state,
            block=strategy.block_index,
            operation=reduction_type,
            dtype=acc_dtype,
            identity=identity_expr,
            pre=pre,
            span=group_span,
            groups=group_count,
        ):
            return input_name
        result_var = strategy.fn.new_var("strided_reduce_result", dce=True)
        if cluster_n > 1 and pre == 1 and group_count == 1:
            # The reduced axis is additionally split across the CTAs of a
            # thread-block cluster (``cute_cluster_n``): combine within-CTA
            # warps and across the cluster with one DSM exchange.
            if acc_dtype != torch.float32:
                # The cluster exchange buffers every partial as Float32
                # (see _cute_grouped_reduce_cluster) — a wider accumulator
                # would silently lose precision through the round trip.
                raise exc.BackendUnsupported(
                    "cute",
                    "cute_cluster_n > 1 requires an fp32 reduction "
                    f"accumulator, got {acc_dtype}",
                )
            buf_var, mbar_var = _cute_cluster_reduce_smem_vars(
                strategy.fn, group_span, cluster_n
            )
            call = ReduceCall(
                scope=ReduceScope.CLUSTER,
                op=reduction_type,
                input=input_name,
                identity=identity_expr,
                lane_expr=lane_var,
                buffers=(buf_var, mbar_var),
                group_span=group_span,
                cluster_n=cluster_n,
            ).render()
            self.record_lane_fold_site(call)
            state.add_statement(f"{result_var} = {call}")
            return result_var
        call = ReduceCall(
            scope=ReduceScope.SHARED_TWO_STAGE,
            op=reduction_type,
            input=input_name,
            identity=identity_expr,
            lane_expr=lane_var,
            lane_in_group=lane_in_group_var,
            lane_mod_pre=lane_mod_pre_var,
            pre=pre,
            group_span=group_span,
            group_count=group_count,
        ).render()
        self.record_lane_fold_site(call)
        state.add_statement(f"{result_var} = {call}")
        return result_var

    def _strided_thread_reduction_expr_shared_tree(
        self,
        *,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        fake_input: torch.Tensor,
        identity_expr: str,
        lane_var: str,
        lane_in_group_var: str,
        lane_mod_pre_var: str,
        pre: int,
        group_span: int,
        num_threads: int,
        group_count: int,
    ) -> str:
        strategy = self.strategy
        result_var = strategy.fn.new_var("strided_reduce_result", dce=True)
        call = ReduceCall(
            scope=ReduceScope.SHARED_TREE,
            op=reduction_type,
            input=input_name,
            identity=identity_expr,
            lane_expr=lane_var,
            lane_in_group=lane_in_group_var,
            lane_mod_pre=lane_mod_pre_var,
            pre=pre,
            group_span=group_span,
            num_threads=num_threads,
            group_count=group_count,
        ).render()
        self.record_lane_fold_site(call)
        state.add_statement(f"{result_var} = {call}")
        return result_var
