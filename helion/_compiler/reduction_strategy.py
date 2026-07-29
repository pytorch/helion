from __future__ import annotations

import ast
import logging
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._prims_common import get_computation_dtype

from .. import exc
from .._compat import shape_env_size_hint
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cute.layout import LayoutTag as _CuteLayoutTag
from .cute.layout_propagation import META_KEY as _CUTE_LAYOUT_META_KEY
from .device_function import find_block_size_symbols
from .host_function import HostFunction
from .inductor_lowering import ReductionLowering
from .inductor_lowering import install_inductor_kernel_handlers
from .tile_strategy import CompactedShape
from .tile_strategy import DeviceGridState
from .tile_strategy import DeviceLoopState
from .tile_strategy import LoopDimInfo
from .tile_strategy import PersistentReductionState
from .tile_strategy import ThreadAxisTracker
from .tile_strategy import TileStrategy
from .tile_strategy import _to_sympy

if TYPE_CHECKING:
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState

log = logging.getLogger(__name__)


def _dtype_str(dtype: torch.dtype) -> str:
    return CompileEnvironment.current().backend.dtype_str(dtype)


def _cute_shared_memory_budget_bytes() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    default_shared = int(props.shared_memory_per_block)
    optin_shared = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
    return max(default_shared, optin_shared)


def _log_cute_reduction_layout(state: CodegenState) -> None:
    """Log the CuTe layout annotation for the current reduction node, if any."""
    if state.fx_node is None:
        return
    constraint = state.fx_node.meta.get(_CUTE_LAYOUT_META_KEY)
    if constraint is None or constraint.input_layout is None:
        return
    layout = constraint.input_layout
    log.debug(
        "cute reduction %s: layout tag=%s thread=%s value=%s",
        state.fx_node.name,
        layout.tag.value,
        layout.thread_shape,
        layout.value_shape,
    )


def _reduction_threads_from_annotation(state: CodegenState) -> int | None:
    """Read reduction thread count from the layout annotation, if available.

    Returns the thread count from the layout annotation when the node has
    a REDUCTION-tagged layout with a concrete integer thread count.
    Falls back to ``None`` so the caller can use ``reduction_threads_hint()``.
    """
    if state.fx_node is None:
        return None
    constraint = state.fx_node.meta.get(_CUTE_LAYOUT_META_KEY)
    if constraint is None or constraint.input_layout is None:
        return None
    layout = constraint.input_layout
    if layout.tag != _CuteLayoutTag.REDUCTION:
        return None
    nt = layout.num_threads()
    if isinstance(nt, int) and nt > 0:
        return nt
    return None


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


def _strategies_concurrent_with_block(
    tile_dispatch: object,
    block_index: int,
) -> list[TileStrategy]:
    """Return strategies that can co-execute with reduction ``block_index``.

    Drops reduction strategies that live in a control-flow branch mutually
    exclusive with ``block_index``'s branch so the per-block thread budget is
    not over-counted (CuTe branch-by-pid kernels). Outside that pattern (no
    branch paths) this returns every strategy unchanged.
    """
    from .device_ir import DeviceIR
    from .host_function import HostFunction

    strategies = list(getattr(tile_dispatch, "strategies", []))
    device_ir = HostFunction.current().device_ir
    red_paths = device_ir.reduction_block_id_branch_paths()
    own_paths = red_paths.get(block_index)
    if not own_paths:
        return strategies
    own_path = own_paths[0]
    result: list[TileStrategy] = []
    for strategy in strategies:
        other_path = None
        for other_block in strategy.block_ids:
            paths = red_paths.get(other_block)
            if paths:
                other_path = paths[0]
                break
        if DeviceIR.branch_paths_mutually_exclusive(own_path, other_path):
            continue
        result.append(strategy)
    return result


def _cute_vec_kernel_mode() -> str:
    """Return ``"vec"`` when all reduction-feeding loads are vec-eligible
    without a degrading dtype cast (i.e. fp32 -> fp32 pipeline), ``"unroll"``
    when at least one load uses the bf16/fp16 -> fp32 cast pattern that the
    CuTe DSL's ``Float32(vec)`` constructor would silently scalarise, or
    ``"none"`` when there's no looped reduction at all.

    ``"vec"`` lets the strategy emit a single ``cute.arch.load(..., V)`` +
    V-fold per lane iter.  ``"unroll"`` falls back to a constexpr V-loop
    around per-element scalar loads — the CUTLASS DSL cannot iterate the
    elements of a bf16/fp16 vector without crashing during compile.
    """
    from .host_function import HostFunction
    from .host_function import NoCurrentFunction

    try:
        hf = HostFunction.current()
    except NoCurrentFunction:
        return "none"
    if hf._device_ir is None:
        return "none"
    cast_targets = {
        "convert_element_type.default",
        "convert_element_type",
        "_to_copy.default",
        "_to_copy",
    }
    from ..language import memory_ops as _memory_ops

    load_target = _memory_ops.load
    has_cast = False
    for graph_info in hf.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.target is not load_target:
                continue
            for user in node.users:
                target_name = getattr(user.target, "__name__", "") or ""
                if target_name in cast_targets:
                    has_cast = True
                    break
            if has_cast:
                break
        if has_cast:
            break
    return "unroll" if has_cast else "vec"


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


class ReductionStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        mask_var: str | None,
        block_size_var: str | None,
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=[block_index],
        )
        self._mask_var = mask_var
        if block_size_var is not None:
            fn.block_size_var_cache[(block_index,)] = block_size_var

    def mask_var(self, block_idx: int) -> str | None:
        assert block_idx == self.block_index
        return self._mask_var

    @property
    def block_index(self) -> int:
        return self.block_ids[0]

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].numel

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        return shapes

    def _reduction_thread_count(self) -> int:
        """Return threads used for this reduction on thread-aware backends."""
        return 0

    def thread_axes_used(self) -> int:
        return 1 if self._reduction_thread_count() > 0 else 0

    def thread_block_sizes(self) -> list[int]:
        count = self._reduction_thread_count()
        return [count] if count > 0 else []

    def _reduction_block_has_lane_loops(self) -> bool:
        """Return True when this reduction block is being traversed via a
        lane loop on the cute backend (synthetic per-thread iteration
        inside a ``DeviceGridState`` that does not have a live thread for
        every logical lane).

        Lane loops serialize part of the logical tile in Python rather
        than mapping it to actual threads, so reductions over the looped
        block cannot be fast-pathed via a warp-level reduction (every
        participating axis must be backed by a live thread).
        """
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        current_grid = codegen.current_grid_state
        if (
            isinstance(current_grid, DeviceGridState)
            and current_grid.has_lane_loops()
            and self.block_index in current_grid.lane_loop_blocks
        ):
            return True
        for loops in codegen.active_device_loops.values():
            for loop_state in loops:
                if (
                    isinstance(loop_state, DeviceGridState)
                    and loop_state.has_lane_loops()
                    and self.block_index in loop_state.lane_loop_blocks
                ):
                    return True
        return False

    def _reduction_block_in_device_lane_loop(self) -> bool:
        """Return True when a ``DeviceLoopState`` distributes this reduction
        block across a per-thread lane loop (CuteNDTileStrategy lanes).

        Unlike :meth:`_reduction_block_has_lane_loops`, this does NOT feed
        :meth:`_needs_loop_carried_accumulator` — it is a dedicated signal for
        the two-pass marker so the existing warp / vec-fold paths keep their
        tuned behavior.
        """
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        for loops in codegen.active_device_loops.values():
            for loop_state in loops:
                if (
                    isinstance(loop_state, DeviceLoopState)
                    and self.block_index in loop_state.lane_loop_blocks
                ):
                    return True
        return False

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
        # A synthetic reduction lane (PersistentReductionStrategy) always
        # distributes the reduction axis across a lane loop; the live thread
        # axis is ``_reduction_thread_count`` wide.
        if getattr(self, "_synthetic_cute_lane_var", None) is not None:
            return max(1, self._reduction_thread_count())
        if not (
            self._reduction_block_has_lane_loops()
            or self._reduction_block_in_device_lane_loop()
        ):
            return None
        threads = self._reduction_thread_count()
        return max(1, threads)

    def _reshape_merged_reduction_group_params(
        self,
    ) -> tuple[int, int, str] | None:
        """Return ``(pre, group_span, lane_expr)`` for a reshape-merged
        reduction whose live thread axis is interleaved with a *sibling*
        thread axis, or ``None`` when no such interleaving exists.

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
        reduce axis extent) used by ``_cute_grouped_reduce_warp``. Returns
        ``None`` when ``pre == 1`` (no sibling axis below the reduce axis),
        in which case the plain consecutive-lane warp reduction is already
        correct.
        """
        env = CompileEnvironment.current()
        backend = env.backend
        if backend.name != "cute":
            return None
        numel = env.block_sizes[self.block_index].numel
        if not isinstance(numel, sympy.Expr):
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
        tile_strategy = self.fn.tile_strategy
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
            return None
        # Live thread extents of ALL blocks (siblings included) so the linear
        # lane index strides are computed correctly. The reduction block's own
        # synthetic thread axis is excluded -- it is fictional (no real warp
        # lanes back it); the source live axis carries the actual data.
        logical_axis_sizes: dict[int, int] = {reduce_axis: reduce_extent}
        for info in env.block_sizes:
            block_id = info.block_id
            if block_id == self.block_index or block_id in source_block_ids:
                continue
            axis = tile_strategy.thread_axis_for_block_id(block_id)
            extent = tile_strategy.thread_extent_for_block_id(block_id)
            if axis is None or extent is None or extent <= 1:
                continue
            logical_axis_sizes[axis] = max(logical_axis_sizes.get(axis, 1), extent)
        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        if pre <= 1:
            # No sibling thread axis below the reduce axis: the reduce axis is
            # already at the bottom of the linear lane index, so consecutive
            # warp lanes belong to the reduction and the plain warp reduce is
            # correct.
            return None
        group_span = pre * reduce_extent
        if group_span > 32:
            # Cross-warp grouped reduction is not handled by the marker path.
            return None
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if lane_expr is None:
            return None
        return pre, group_span, lane_expr

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
        for block_id, loops in state.codegen.active_device_loops.items():
            for loop_state in loops:
                if not isinstance(loop_state, DeviceLoopState):
                    continue
                if isinstance(loop_state.strategy, LoopedReductionStrategy):
                    # The rolled reduction owns the lane reduction over its
                    # block; defer to its accumulate / vec-fold machinery.
                    if block_id == self.block_index:
                        return True
                    continue
                if (
                    block_id != self.block_index
                    and block_id not in loop_state.block_thread_axes
                    and block_id not in loop_state.lane_loop_blocks
                ):
                    # The reduction's lane loop is OUTSIDE this serial device
                    # loop. A synthetic-lane PersistentReductionStrategy can be
                    # repaired by the ``interchange_lane_outside_serial_reductions``
                    # post-pass (it splits the lane loop into a lane-inside-mb
                    # two-pass nest for the broadcast consumer plus a
                    # lane-outside-mb nest for any per-feature accumulators), so
                    # keep emitting the marker in that case. Other (non-synthetic)
                    # situations remain unsupported. ``getattr`` guards strategies
                    # without a synthetic lane var (e.g. BlockReductionStrategy).
                    if getattr(self, "_synthetic_cute_lane_var", None) is not None:
                        continue
                    return True
        return False

    def _reduction_block_is_serial(self) -> bool:
        """Return True when this reduction block is being traversed by a
        serial ``DeviceLoopState`` (a Python ``for`` loop) rather than a
        live thread axis.

        Reductions over a serially-iterated block cannot be fast-pathed
        via a warp-level reduction; the surrounding loop has to carry the
        accumulator.
        """
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        for loop_state in codegen.active_device_loops.get(self.block_index, []):
            if (
                isinstance(loop_state, DeviceLoopState)
                and self.block_index not in loop_state.block_thread_axes
            ):
                return True
        return False

    def _reduction_block_has_live_thread_axis(self) -> bool:
        """Return True when this reduction block is mapped to a live thread
        axis in the active loop nest (in either the current grid or any
        active device loop).

        A ``False`` return on the cute backend means a warp-level reduction
        across this block would fold together unrelated tensor elements,
        because no real threads back the block. The caller falls back to
        loop-carried accumulation.
        """
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        current_grid = codegen.current_grid_state
        if (
            isinstance(current_grid, DeviceGridState)
            and self.block_index in current_grid.block_thread_axes
        ):
            return True
        for loop_state in codegen.active_device_loops.get(self.block_index, []):
            if self.block_index in loop_state.block_thread_axes:
                return True
        for loops in codegen.active_device_loops.values():
            for loop_state in loops:
                if self.block_index in loop_state.block_thread_axes:
                    return True
        return False

    def _needs_loop_carried_accumulator(self) -> bool:
        """Return True when the surrounding loop nest must perform the
        reduction via loop-carried accumulation instead of a warp-level
        reduction across threads.

        This consolidates the three "no live thread axis" conditions:

        * :meth:`_reduction_block_is_serial` — the block is iterated by
          a serial ``DeviceLoopState`` rather than a thread axis;
        * :meth:`_reduction_block_has_lane_loops` — the block is
          iterated by a lane loop (synthetic per-thread iteration);
        * ``not _reduction_block_has_live_thread_axis()`` — the block
          is not mapped to any thread axis at all.

        In every case the conclusion is the same: there is no live
        thread axis to reduce across, so the surrounding loop must
        accumulate the partial values across iterations.

        Always returns False for tile-level backends (Triton / Pallas /
        TileIR) which use their native reduction primitives.
        """
        if CompileEnvironment.current().backend.max_reduction_threads() is None:
            return False
        return (
            self._reduction_block_is_serial()
            or self._reduction_block_has_lane_loops()
            or not self._reduction_block_has_live_thread_axis()
        )

    def _planned_thread_dims(self) -> tuple[int, int, int]:
        return self.fn.tile_strategy.thread_block_dims()

    def _get_thread_axis(self) -> int:
        """Compute the thread axis index for this reduction strategy.

        Some backends place reduction strategies first so reduction threads share
        a warp. Others keep the natural strategy order.
        """
        env = CompileEnvironment.current()
        if (axis := self.fn.tile_strategy.thread_axis_for_strategy(self)) is not None:
            return axis
        if env.backend.reduction_axis_first():
            axis = 0
            for strategy in self.fn.tile_strategy.strategies:
                if strategy is self:
                    break
                if isinstance(strategy, ReductionStrategy):
                    axis += strategy.thread_axes_used()
            return axis
        axis = 0
        for strategy in self.fn.tile_strategy.strategies:
            if strategy is self:
                break
            axis += strategy.thread_axes_used()
        return axis

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        raise NotImplementedError

    def call_reduction_function(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> str:
        backend = CompileEnvironment.current().backend
        if backend.is_indexed_reduction(reduction_type):
            index_var = self.index_var(self.block_index)
            return self.call_indexed_reduction(
                input_name,
                self.broadcast_str(index_var, fake_input, dim),
                reduction_type,
                dim,
                fake_output,
            )
        return backend.reduction_expr(
            input_name,
            reduction_type,
            dim,
            block_size_var=self.block_size_var(self.block_index),
        )

    def _index_init_expr(self, block_size_var: str, dtype: str, block_idx: int) -> str:
        env = CompileEnvironment.current()
        backend = env.backend
        size = env.block_sizes[block_idx].size
        if isinstance(size, int) and size == 0:
            return backend.reduction_index_zero_expr(dtype)
        if isinstance(size, torch.SymInt) and env.known_equal(size, 0):
            return backend.reduction_index_zero_expr(dtype)
        return backend.reduction_index_expr(
            block_size_var, dtype, block_idx, axis=self._get_thread_axis()
        )

    def call_indexed_reduction(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        fake_output: torch.Tensor,
    ) -> str:
        env = CompileEnvironment.current()
        return env.backend.argreduce_result_expr(
            input_name,
            index_value,
            reduction_type,
            dim,
            fake_output.dtype,
            block_size_var=self.block_size_var(self.block_index),
            index_dtype=env.index_dtype,
        )

    def maybe_reshape(
        self,
        expr: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> str:
        size = [*fake_input.size()]
        size.pop(dim)
        if [*fake_output.size()] == size:
            return expr
        backend = CompileEnvironment.current().backend
        shape = self.fn.tile_strategy.shape_str([*fake_output.size()])
        return backend.maybe_reshape_reduction(
            expr,
            source_shape=size,
            target_shape=[*fake_output.size()],
            target_shape_expr=shape,
        )

    def broadcast_str(self, base: str, fake_input: torch.Tensor, dim: int) -> str:
        input_size = [*fake_input.size()]
        expand = self.fn.tile_strategy.expand_str(input_size, dim)
        shape = self.fn.tile_strategy.shape_str(input_size)
        return CompileEnvironment.current().backend.broadcast_to_expr(
            f"{base}{expand}", shape
        )


class PersistentReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
    ) -> None:
        from .device_ir import ReductionLoopGraphInfo

        env = CompileEnvironment.current()
        numel = env.block_sizes[block_index].numel
        # Skip the mask when RDIM_SIZE == numel (no padding needed).
        # This is true when numel is a power of 2 (Triton doesn't round),
        # or when the backend uses exact RDIM sizes (e.g., Pallas).
        needs_mask = True
        # Guard numel > 0: on PyTorch 2.9, next_power_of_2(0) returns 0
        # (the n <= 0 guard was added later), so static_rdim_size(0) == 0
        # would incorrectly skip the mask for zero-size reductions.
        if isinstance(numel, (int, sympy.Integer)) and int(numel) > 0:
            needs_mask = env.backend.static_rdim_size(int(numel)) != int(numel)
        mask_var: str | None = (
            fn.new_var(f"mask_{block_index}", dce=True) if needs_mask else None
        )
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_RDIM_SIZE_{block_index}"),
        )
        self.offset_vars[block_index] = "0"
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            if env.backend.name == "cute":
                max_threads = cute_live_reduction_threads(max_threads)
                # Indexed reductions (argmin/argmax) on CuTe only have a
                # warp-level reduction primitive that takes
                # ``threads_in_group <= 32``. Cap the persistent thread
                # count to the warp size so the emitted
                # ``cute.arch.warp_reduction`` is correct.
                if _block_has_indexed_reduction(fn, block_index):
                    max_threads = min(max_threads, _CUTE_WARP_REDUCTION_THREADS)
            if isinstance(numel, (int, sympy.Integer)):
                size_hint = int(numel)
            elif isinstance(numel, sympy.Expr):
                size_hint = shape_env_size_hint(env.shape_env, numel)
            else:
                size_hint = env.size_hint(numel)
            self._thread_count = next_power_of_2(min(size_hint, max_threads))
        else:
            self._thread_count = 0
        # On cute, the launch block dim is capped at MAX_THREADS_PER_BLOCK.
        # If the existing tile strategies already claim that budget, the
        # reduction's Y/Z axis silently collapses to 1, producing kernels
        # whose ``thread_idx[axis] + synthetic_lane * thread_count`` indexing
        # only covers ``padded_size // thread_count`` of the reduction extent.
        # Shrink ``_thread_count`` here so the full extent stays addressable
        # via the synthetic lane loop.
        # Tile strategies are added before reduction strategies, so they are
        # already on the dispatcher by the time we get here.
        tile_dispatch = getattr(fn, "tile_strategy", None)
        if tile_dispatch is not None:
            # Reductions in mutually-exclusive control-flow branches share a
            # thread axis (see ``TileStrategyDispatch._branch_by_control_flow``),
            # so they never co-execute and must not be multiplied into this
            # reduction's thread budget. Drop them before adjusting.
            concurrent = _strategies_concurrent_with_block(tile_dispatch, block_index)
            self._thread_count = env.backend.adjust_reduction_thread_count(
                self._thread_count, concurrent
            )
        self._synthetic_cute_lane_var: str | None = None
        self._synthetic_cute_lane_extent = 1
        is_graph_reduction_dim = any(
            isinstance(graph, ReductionLoopGraphInfo) and block_index in graph.block_ids
            for graph in fn.codegen.codegen_graphs
        )
        if self._thread_count > 0:
            if isinstance(numel, (int, sympy.Integer)):
                size_hint = int(numel)
            elif isinstance(numel, sympy.Expr):
                size_hint = shape_env_size_hint(env.shape_env, numel)
            else:
                size_hint = env.size_hint(numel)
            # For a non-graph-reduction dim we always try to recover the
            # full extent through a synthetic lane loop. For a graph
            # reduction dim we only need the synthetic lane loop when
            # ``adjust_reduction_thread_count`` shrank ``thread_count``
            # below the (padded) reduction extent — otherwise every logical
            # lane is already backed by a live thread and the warp/cross-warp
            # reduction covers the whole axis. Without this, a shrunk
            # graph-reduction dim only addresses the first ``thread_count``
            # elements (e.g. layer_norm_bwd's feature axis), leaving the
            # remaining columns/partial sums uncomputed.
            needs_synthetic = not is_graph_reduction_dim or (
                self._thread_count < next_power_of_2(min(size_hint, max_threads))
                if max_threads is not None
                else False
            )
            if needs_synthetic:
                lane_extent = env.backend.create_synthetic_reduction_lanes(
                    self._thread_count, size_hint
                )
                # The synthetic lane loop folds each lane into a per-thread
                # accumulator that is then combined with a single warp reduction
                # over ``_reduction_thread_count`` threads. That warp reduction
                # is only correct within one warp, so cap the thread count to
                # the warp size and let the lane loop grow to cover the rest;
                # otherwise a multi-warp group silently drops lanes (#2643).
                if (
                    lane_extent is not None
                    and self._thread_count > _CUTE_WARP_REDUCTION_THREADS
                ):
                    self._thread_count = _CUTE_WARP_REDUCTION_THREADS
                    lane_extent = env.backend.create_synthetic_reduction_lanes(
                        self._thread_count, size_hint
                    )
                if lane_extent is not None:
                    self._synthetic_cute_lane_var = fn.new_var(
                        f"synthetic_lane_{block_index}",
                        dce=False,
                    )
                    self._synthetic_cute_lane_extent = lane_extent

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def offset_var(self, block_idx: int) -> str:
        assert block_idx == self.block_index
        return "0"

    def codegen_preamble(self, state: CodegenState) -> None:
        env = CompileEnvironment.current()
        backend = env.backend
        block_idx = self.block_index
        numel = env.block_sizes[block_idx].numel
        index_var = self.index_var(block_idx)
        mask_var = self._mask_var
        block_size_var = self.block_size_var(self.block_index)
        assert block_size_var is not None
        if state.device_function.constexpr_arg(block_size_var):
            if isinstance(numel, sympy.Integer):
                # Static size - issue statement immediately
                stmt = statement_from_string(
                    f"{block_size_var} = {backend.static_rdim_size(int(numel))}"
                )
                state.codegen.host_statements.append(stmt)
            else:
                # Check for block size dependencies
                block_mapping, _ = find_block_size_symbols(numel)
                if block_mapping:
                    # Defer issuing statement until block sizes are known
                    state.device_function.deferred_rdim_defs.append(
                        (block_size_var, numel)
                    )
                else:
                    # No dependencies - issue statement immediately
                    expr_str = HostFunction.current().sympy_expr(numel)
                    stmt = statement_from_string(
                        f"{block_size_var} = {backend.dynamic_rdim_size_expr(expr_str)}"
                    )
                    state.codegen.host_statements.append(stmt)
        current_grid = state.codegen.current_grid_state
        synthetic_lane_var = self._synthetic_cute_lane_var
        if synthetic_lane_var is not None and current_grid is not None:
            axis = self._get_thread_axis()
            current_grid.add_lane_loop(
                block_idx,
                synthetic_lane_var,
                self._synthetic_cute_lane_extent,
            )
            current_grid.thread_axis_sizes[axis] = max(
                current_grid.thread_axis_sizes.get(axis, 1),
                self._thread_count,
            )
            current_grid.block_thread_axes[block_idx] = axis
            index_expr = (
                f"({self._index_init_expr(block_size_var, env.index_type(), block_idx)})"
                f" + cutlass.Int32({synthetic_lane_var}) * {self._thread_count}"
            )
            current_grid.lane_setup_statements.append(
                statement_from_string(f"{index_var} = {index_expr}")
            )
            if mask_var is not None:
                current_grid.lane_setup_statements.append(
                    statement_from_string(
                        f"{mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
                    )
                )
        else:
            state.add_statement(
                f"{index_var} = {self._index_init_expr(block_size_var, env.index_type(), block_idx)}"
            )
            if mask_var is not None:
                state.add_statement(
                    f"{mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
                )
        # Extract end_var_name from the numel expression
        from .tile_strategy import LoopDimInfo

        end_var_name = self.fn.sympy_expr(numel)
        block_id_to_info = {
            self.block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        tracker = ThreadAxisTracker()
        if self._thread_count > 0:
            tracker.record(
                self.block_index, self._get_thread_axis(), self._thread_count
            )
        state.codegen.push_active_loops(
            PersistentReductionState(
                self,
                block_id_to_info=block_id_to_info,
                thread_axis_sizes=tracker.sizes,
                block_thread_axes=tracker.block_axes,
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
        env = CompileEnvironment.current()
        backend = env.backend
        if (
            backend.name != "cute"
            or self._thread_count <= 32
            or self._synthetic_cute_lane_var is not None
            or backend.is_indexed_reduction(reduction_type)
        ):
            return None

        current_grid = state.codegen.current_grid_state
        axis_sizes: dict[int, int] = {}
        if isinstance(current_grid, DeviceGridState):
            for axis, size in current_grid.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        reduction_axis = self._get_thread_axis()
        axis_sizes[reduction_axis] = max(
            axis_sizes.get(reduction_axis, 1), self._thread_count
        )

        num_threads = 1
        for size in axis_sizes.values():
            num_threads *= size
        group_span = self._thread_count
        if num_threads % group_span != 0:
            return None

        identity_expr = backend.cast_expr(
            constant_repr(default_value), _dtype_str(dtype)
        )
        # The two-stage shared reduce takes ``dtype`` (the accumulation dtype,
        # ``get_computation_dtype(fake_input.dtype)``) from ``type(identity)``.
        # Upcast the (possibly fp16/bf16) masked input to that same dtype so the
        # helper's ``input if mask else identity`` selection unifies cleanly and
        # the reduction still accumulates in the wider accumulation dtype.
        input_expr = backend.cast_expr(input_name, _dtype_str(dtype))

        if reduction_axis == 0:
            # ``axis_sizes`` only reflects the thread axes discovered so far. A
            # sibling control-flow branch can still introduce a *redundant*
            # thread axis later in codegen -- e.g. a free ``hl.arange`` that
            # another (mutually-exclusive) branch maps onto thread axis 1/2 --
            # which enlarges the launch block beyond ``num_threads``. Those extra
            # threads re-run this reduction; if every redundant row keyed its
            # shared memory on the same slots the cross-warp combine would race
            # (producing intermittently wrong partial reductions). Key the
            # per-group shared memory on the FULL flattened thread id (from the
            # runtime block dims) so each redundant row reduces into its own
            # region. When the reduction owns thread axis 0
            # (``blockDim.x == group_span``) this is identical to the
            # single-axis path whenever there is no redundancy; the extra
            # groups simply go unused.
            from .cute.thread_budget import MAX_THREADS_PER_BLOCK

            index_type = backend.index_type_str(env.index_dtype)
            tid0 = backend.cast_expr("cute.arch.thread_idx()[0]", index_type)
            tid1 = backend.cast_expr("cute.arch.thread_idx()[1]", index_type)
            tid2 = backend.cast_expr("cute.arch.thread_idx()[2]", index_type)
            bdim0 = backend.cast_expr("cute.arch.block_dim()[0]", index_type)
            bdim1 = backend.cast_expr("cute.arch.block_dim()[1]", index_type)
            lane_expr = (
                f"{tid0} + ({tid1}) * ({bdim0}) + ({tid2}) * ({bdim0}) * ({bdim1})"
            )
            group_count = (MAX_THREADS_PER_BLOCK + group_span - 1) // group_span
        else:
            # The two-stage shared-memory reduction assumes its ``lane_var`` is
            # the linear thread index across ALL of the launch block's threads.
            # If ``axis_sizes`` only covers a subset of the planned block dims
            # (e.g. an inner reduction strategy contributes another thread axis
            # that hasn't been entered yet), the emitted reduction would race
            # across the missing axis. Bail out and fall back to the warp-level
            # path in that case.
            planned_dims = self._planned_thread_dims()
            planned_block_threads = planned_dims[0] * planned_dims[1] * planned_dims[2]
            if num_threads != planned_block_threads:
                return None
            lane_expr = backend.thread_linear_index_expr(axis_sizes)
            if lane_expr is None:
                return None
            group_count = num_threads // group_span

        lane_var = self.fn.new_var("persistent_reduce_lane", dce=True)
        lane_in_group_var = self.fn.new_var("persistent_reduce_lane_in_group", dce=True)
        lane_mod_pre_var = self.fn.new_var("persistent_reduce_lane_mod_pre", dce=True)
        result_var = self.fn.new_var("persistent_reduce_result", dce=True)
        state.add_statement(f"{lane_var} = {lane_expr}")
        state.add_statement(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
        state.add_statement(f"{lane_mod_pre_var} = ({lane_in_group_var}) % 1")
        state.add_statement(
            f"{result_var} = _cute_grouped_reduce_shared_two_stage("
            f"{input_expr}, {reduction_type!r}, {identity_expr}, "
            f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
            f"pre=1, group_span={group_span}, group_count={group_count})"
        )
        return result_var

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        env = CompileEnvironment.current()
        backend = env.backend
        # Record (for the CuTe backend) the branch path under which this reduction
        # claims its thread axis, so a free ``hl.arange`` in a mutually-exclusive
        # sibling grid branch reuses this axis instead of claiming a fresh one that
        # would widen the launch block and race this single-axis reduction. No-op
        # outside a dynamic ``_if`` branch.
        if backend.name == "cute":
            state.codegen.record_cute_strategy_axis_branch_path(self._get_thread_axis())
        numel = env.block_sizes[self.block_index].numel
        if isinstance(numel, sympy.Integer) and numel == 0:
            default = ir.Reduction.default_accumulator(reduction_type, fake_input.dtype)
            assert isinstance(default, (float, int, bool))
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_output.size()])
            return expr_from_string(
                backend.full_expr(shape_dims, constant_repr(default), fake_output.dtype)
            )
        acc_dtype = get_computation_dtype(fake_input.dtype)
        default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
        if (
            self._synthetic_cute_lane_var is not None
            and not backend.is_indexed_reduction(reduction_type)
            and isinstance(default, (float, int, bool))
            and not self._lane_reduce_marker_unsupported(state)
            and (threads := self._lane_reduce_threads_in_group()) is not None
        ):
            # The reduction axis is split across a synthetic per-thread lane
            # loop: the single warp reduction only covers one lane's worth of
            # elements. Emit a marker so the ``split_lane_loop_reductions``
            # post-pass produces the two-pass (accumulate across lanes ->
            # warp-combine across ``threads`` -> consume) structure.
            from .tile_strategy import _lane_reduce_marker_expr

            identity_expr = backend.cast_expr(
                constant_repr(default), _dtype_str(acc_dtype)
            )
            group_params = self._reshape_merged_reduction_group_params()
            if group_params is not None:
                group_pre, group_span, group_lane_expr = group_params
                expr = _lane_reduce_marker_expr(
                    input_name,
                    reduction_type,
                    identity_expr,
                    threads,
                    group_pre=group_pre,
                    group_span=group_span,
                    group_lane_expr=group_lane_expr,
                )
            else:
                expr = _lane_reduce_marker_expr(
                    input_name, reduction_type, identity_expr, threads
                )
            return expr_from_string(
                self.maybe_reshape(expr, dim, fake_input, fake_output)
            )
        if isinstance(default, (float, int, bool)):
            cross_warp = self._cute_cross_warp_reduction_expr(
                state, input_name, reduction_type, default, acc_dtype
            )
        else:
            cross_warp = None
        if cross_warp is not None:
            expr = cross_warp
        else:
            expr = self.call_reduction_function(
                input_name,
                reduction_type,
                dim,
                fake_input,
                fake_output,
            )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))


class LoopedReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        block_size: int,
    ) -> None:
        env = CompileEnvironment.current()
        if block_size <= 1:
            raise exc.InvalidConfig(
                f"LoopedReductionStrategy requires block_size > 1, got {block_size}"
            )
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            # CuTe argreduce uses cute.arch.warp_reduction which is only
            # correct for threads_in_group<=32. Cap to warp size whenever
            # the rolled reduction will fold an indexed reduction over this
            # block.
            if env.backend.name == "cute" and _block_has_indexed_reduction(
                fn, block_index
            ):
                max_threads = min(max_threads, _CUTE_WARP_REDUCTION_THREADS)
            thread_count = next_power_of_2(min(block_size, max_threads))
        else:
            thread_count = 0
        tile_dispatch = getattr(fn, "tile_strategy", None)
        if tile_dispatch is not None:
            thread_count = env.backend.adjust_reduction_thread_count(
                thread_count, tile_dispatch.strategies
            )
        self._thread_count = thread_count
        self.block_size = block_size
        self._loop_block_size = block_size
        self._cute_reduction_lane_var: str | None = None
        self._cute_reduction_lane_extent = 1
        self._cute_reduction_vec_width = 1
        # ``"vec"`` (fp32 fast path) or ``"unroll"`` (bf16/fp16 fallback)
        # — controls how the lane body emits each per-iter load.
        self._cute_reduction_vec_mode = "vec"
        # Masks queued by vec loads inside the lane loop; consumed by
        # codegen_reduction to wrap the V-fold scalar.
        self._cute_pending_vec_masks: list[str] = []
        # Set when a vec load was actually emitted for the current
        # reduction's lane body — codegen_reduction inspects this to
        # decide whether to emit the V-fold step.
        self._cute_emitted_vec_load = False
        if (
            env.backend.name == "cute"
            and thread_count > 0
            and block_size > thread_count
        ):
            self._cute_reduction_lane_extent = (
                block_size + thread_count - 1
            ) // thread_count
            self._loop_block_size = thread_count * self._cute_reduction_lane_extent
            self._cute_reduction_lane_var = fn.new_var(
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
                and self._cute_reduction_lane_extent % vec_width == 0
            ):
                mode = _cute_vec_kernel_mode()
                if mode in ("vec", "unroll"):
                    self._cute_reduction_vec_width = vec_width
                    self._cute_reduction_vec_mode = mode
                    self._cute_reduction_lane_extent = (
                        self._cute_reduction_lane_extent // vec_width
                    )
        if env.known_multiple(
            env.block_sizes[block_index].numel, self._loop_block_size
        ):
            mask_var: str | None = None
        else:
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_REDUCTION_BLOCK_{block_index}"),
        )
        self.offset_vars[block_index] = fn.new_var(f"roffset_{block_index}", dce=True)
        self.index_vars[block_index] = fn.new_var(f"rindex_{block_index}", dce=True)

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def _active_thread_axis_sizes(
        self, state: CodegenState, device_loop: DeviceLoopState
    ) -> dict[int, int]:
        axis_sizes: dict[int, int] = {}
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

    def _cute_cross_warp_reduction_expr(
        self,
        state: CodegenState,
        device_loop: DeviceLoopState,
        input_name: str,
        reduction_type: str,
        default_value: float | bool,
        dtype: torch.dtype,
    ) -> str | None:
        env = CompileEnvironment.current()
        backend = env.backend
        if (
            backend.name != "cute"
            or self._thread_count <= 32
            or backend.is_indexed_reduction(reduction_type)
        ):
            return None

        axis_sizes = self._active_thread_axis_sizes(state, device_loop)
        reduction_axis = self._get_thread_axis()
        axis_sizes[reduction_axis] = max(
            axis_sizes.get(reduction_axis, 1), self._thread_count
        )
        num_threads = 1
        for size in axis_sizes.values():
            num_threads *= size
        group_span = self._thread_count
        if num_threads % group_span != 0:
            return None
        # The two-stage shared-memory reduction assumes its ``lane_var`` is
        # the linear thread index across ALL of the launch block's threads.
        # If ``axis_sizes`` only covers a subset of the planned block dims
        # (e.g. an inner reduction strategy contributes another thread axis
        # that hasn't been entered yet), the emitted reduction would race
        # across the missing axis. Bail out and fall back to the warp-level
        # path in that case.
        planned_dims = self._planned_thread_dims()
        planned_block_threads = planned_dims[0] * planned_dims[1] * planned_dims[2]
        if num_threads != planned_block_threads:
            return None
        lane_expr = backend.thread_linear_index_expr(axis_sizes)
        if lane_expr is None:
            return None

        identity_expr = backend.cast_expr(
            constant_repr(default_value), _dtype_str(dtype)
        )
        group_count = num_threads // group_span
        lane_var = self.fn.new_var("looped_reduce_lane", dce=True)
        lane_in_group_var = self.fn.new_var("looped_reduce_lane_in_group", dce=True)
        lane_mod_pre_var = self.fn.new_var("looped_reduce_lane_mod_pre", dce=True)
        result_var = self.fn.new_var("looped_reduce_result", dce=True)
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_var} = {lane_expr}")
        )
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
        )
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_mod_pre_var} = ({lane_in_group_var}) % 1")
        )
        device_loop.outer_suffix.append(
            statement_from_string(
                f"{result_var} = _cute_grouped_reduce_shared_two_stage("
                f"{input_name}, {reduction_type!r}, {identity_expr}, "
                f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
                f"pre=1, group_span={group_span}, group_count={group_count})"
            )
        )
        return result_var

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        env = CompileEnvironment.current()
        block_index = self.block_index
        numel = env.block_sizes[block_index].numel
        offset_var = self.offset_var(block_index)
        index_var = self.index_var(block_index)
        block_size_var = self.block_size_var(block_index)
        assert block_size_var is not None
        if state.device_function.constexpr_arg(block_size_var):
            state.codegen.host_statements.append(
                statement_from_string(f"{block_size_var} = {self._loop_block_size!r}")
            )
        inner_body: list[ast.AST] = [
            statement_from_string(
                f"{index_var} = {offset_var} + {self._index_init_expr(f'({block_size_var})', env.index_type(), block_index)}"
            ),
        ]
        reduction_lane_var = self._cute_reduction_lane_var
        vec = self._cute_reduction_vec_width
        # Detect whether the upcoming graph contains a reduction op so we
        # can choose between the reduce-sweep shape (single vec load +
        # V-fold) and the consume-sweep shape (V scalar elementwise ops in
        # an inner constexpr loop).
        active_graph_info = getattr(state.codegen, "_cute_active_graph_info", None)
        graph_has_reduction = True
        if vec > 1 and active_graph_info is not None:
            graph = getattr(active_graph_info, "graph", None)
            if graph is not None:
                graph_has_reduction = any(
                    isinstance(n.meta.get("lowering"), ReductionLowering)
                    for n in graph.nodes
                )
        # Unroll the lane body via a constexpr V-loop when the consume sweep
        # mixes scalars (no reduction in graph), OR when the reduce sweep is
        # in ``"unroll"`` mode (bf16/fp16 inputs that the CuTe DSL can't
        # safely subscript as a vector).
        consume_unroll = vec > 1 and (
            not graph_has_reduction or self._cute_reduction_vec_mode == "unroll"
        )
        # Map from (tensor_name, base_expr) -> (hoist_var, dtype) so the
        # dispatcher can reuse one hoist per (tensor, base) pair instead of
        # emitting a fresh vec load on every dispatcher call.
        self._cute_lane_vec_loads: dict[tuple[str, str], tuple[str, torch.dtype]] = {}
        # Variable name holding the per-lane-iter base index for vec hoists
        # in ``unroll`` mode — the dispatcher uses this to compute the vec
        # pointer offset once.
        self._cute_lane_base_index_var: str | None = None
        vec_lane_var: str | None = None
        base_expr: str = ""
        if reduction_lane_var is not None:
            if vec > 1:
                # base = offset + thread_idx*V + lane*(THREADS*V)
                base_expr = (
                    f"{offset_var} + "
                    f"{self._index_init_expr(f'({block_size_var})', env.index_type(), block_index)} "
                    f"* {vec} + "
                    f"cutlass.Int32({reduction_lane_var}) * {self._thread_count * vec}"
                )
                if consume_unroll:
                    vec_lane_var = self.fn.new_var(
                        f"reduction_vec_lane_{block_index}",
                        dce=False,
                    )
                    self._cute_lane_base_index_var = self.fn.new_var(
                        f"reduction_lane_base_{block_index}",
                        dce=False,
                    )
                    # index_var = base + vi  (used inside the constexpr loop)
                    inner_body[0] = statement_from_string(
                        f"{index_var} = {self._cute_lane_base_index_var} + cutlass.Int32({vec_lane_var})"
                    )
                else:
                    inner_body[0] = statement_from_string(f"{index_var} = {base_expr}")
            else:
                inner_body[0] = statement_from_string(
                    f"{index_var} = {offset_var} + {self._index_init_expr(f'({block_size_var})', env.index_type(), block_index)} + cutlass.Int32({reduction_lane_var}) * {self._thread_count}"
                )
        if (mask_var := self._mask_var) is not None:
            inner_body.append(
                statement_from_string(
                    f"{mask_var} = {index_var} < {state.sympy_expr(numel)}"
                )
            )
        body = inner_body
        if reduction_lane_var is not None:
            from .tile_strategy import _create_lane_loop

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
                    f"{self._cute_lane_base_index_var} = {base_expr}"
                )
                lane_body: list[ast.AST] = [
                    base_stmt,
                    vec_for,
                ]
                body = [
                    _create_lane_loop(
                        reduction_lane_var,
                        self._cute_reduction_lane_extent,
                        lane_body,
                    )
                ]
                # Stash the lane body list so the dispatcher can splice
                # hoists in (BETWEEN base_stmt and vec_for) as it runs.
                self._cute_lane_body = lane_body
            else:
                body = [
                    _create_lane_loop(
                        reduction_lane_var,
                        self._cute_reduction_lane_extent,
                        inner_body,
                    )
                ]

        for_node = create(
            ast.For,
            target=create(ast.Name, id=offset_var, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(
                    state.config,
                    [self.block_index],
                    begin="0",
                    end=state.sympy_expr(numel),
                    step=block_size_var,
                ),
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        # Extract end_var_name from the actual numel expression used in the range()
        from .tile_strategy import LoopDimInfo

        end_var_name = state.sympy_expr(numel)
        block_id_to_info = {
            block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        tracker = ThreadAxisTracker()
        if self._thread_count > 0:
            tracker.record(block_index, self._get_thread_axis(), self._thread_count)
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=inner_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        _log_cute_reduction_layout(state)
        # See ``PersistentReductionStrategy.codegen_reduction``: record the branch
        # path of this reduction's thread axis so a mutually-exclusive sibling
        # branch's free ``hl.arange`` can reuse the axis (CuTe backend only).
        if CompileEnvironment.current().backend.name == "cute":
            state.codegen.record_cute_strategy_axis_branch_path(self._get_thread_axis())
        with install_inductor_kernel_handlers(state.codegen, {}):
            env = CompileEnvironment.current()
            backend = env.backend
            device_loop = state.codegen.active_device_loops[self.block_index][-1]
            assert isinstance(device_loop, DeviceLoopState)
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_input.size()])
            acc_dtype = get_computation_dtype(fake_input.dtype)  # promote fp16 to fp32
            default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
            assert isinstance(default, (float, int, bool))
            assert state.fx_node is not None
            acc = self.fn.new_var(f"{state.fx_node.name}_acc", dce=True)
            acc_full = backend.full_expr(shape_dims, constant_repr(default), acc_dtype)
            device_loop.outer_prefix.append(
                statement_from_string(f"{acc} = {acc_full}")
            )
            result = self.fn.new_var(state.fx_node.name, dce=True)
            if not backend.is_indexed_reduction(reduction_type):
                vec_input = input_name
                if (
                    backend.name == "cute"
                    and self._cute_reduction_vec_width > 1
                    and self._cute_emitted_vec_load
                ):
                    # The vec load + downstream elementwise ops produced a
                    # length-V vector; fold it into a scalar so the warp-level
                    # reduction stays unchanged.
                    folded = self.fn.new_var(f"{state.fx_node.name}_vfold", dce=True)
                    state.add_statement(
                        f"{folded} = _cute_pre_vec_fold({vec_input}, "
                        f"{reduction_type!r}, V={self._cute_reduction_vec_width})"
                    )
                    # If any vec load was masked, gate the folded scalar by
                    # the same mask so masked-out rows don't pollute acc.
                    if self._cute_pending_vec_masks:
                        identity_repr = constant_repr(default)
                        identity_expr = backend.cast_expr(
                            identity_repr, _dtype_str(acc_dtype)
                        )
                        mask_combined = " and ".join(
                            f"({m})" for m in self._cute_pending_vec_masks
                        )
                        gated = self.fn.new_var(
                            f"{state.fx_node.name}_vfold_gated", dce=True
                        )
                        state.add_statement(
                            f"{gated} = {folded} if ({mask_combined}) "
                            f"else {identity_expr}"
                        )
                        vec_input = gated
                        self._cute_pending_vec_masks.clear()
                    else:
                        vec_input = folded
                # Reset for the next reduction's lane body (the consume
                # sweep may also be codegen'd later but with no vec load).
                self._cute_emitted_vec_load = False
                combine_expr = backend.reduction_combine_expr(
                    reduction_type, acc, vec_input, acc_dtype
                )
                state.add_statement(f"{acc} = {combine_expr}")
                expr = self._cute_cross_warp_reduction_expr(
                    state,
                    device_loop,
                    acc,
                    reduction_type,
                    default,
                    acc_dtype,
                ) or self.call_reduction_function(
                    acc,
                    reduction_type,
                    dim,
                    fake_input,
                    fake_output,
                )
            else:
                acc_index = self.fn.new_var(f"{state.fx_node.name}_acc_index", dce=True)
                index_dtype = env.index_dtype
                device_loop.outer_prefix.append(
                    statement_from_string(
                        f"{acc_index} = {backend.reduction_index_init_expr(shape_dims, index_dtype)}"
                    )
                )
                index = self.broadcast_str(
                    self.index_var(self.block_index), fake_input, dim
                )
                for stmt in backend.argreduce_loop_update_statements(
                    reduction_type=reduction_type,
                    acc=acc,
                    acc_index=acc_index,
                    value=input_name,
                    index=index,
                ):
                    state.add_statement(stmt)
                expr = self.call_indexed_reduction(
                    acc,
                    acc_index,
                    reduction_type,
                    dim,
                    fake_output,
                )
            # Ensure the final reduction result matches torch.* dtype semantics
            expr = self.maybe_reshape(expr, dim, fake_input, fake_output)
            expr = backend.cast_expr(expr, _dtype_str(fake_output.dtype))
            device_loop.outer_suffix.append(statement_from_string(f"{result} = {expr}"))

            # Optional: emit a dtype static assert right after the assignment when enabled
            if env.settings.debug_dtype_asserts:
                device_loop.outer_suffix.append(
                    statement_from_string(
                        f"tl.static_assert({result}.dtype == {_dtype_str(fake_output.dtype)})"
                    )
                )
            return expr_from_string(result)


class BlockReductionStrategy(ReductionStrategy):
    """This is used when we are reducing over a tile rather than an entire tensor."""

    def __init__(
        self,
        state: CodegenState,
        block_index: int,
    ) -> None:
        super().__init__(
            fn=state.device_function,
            block_index=block_index,
            mask_var=state.codegen.mask_var(block_index),
            block_size_var=None,
        )
        self.offset_vars[block_index] = "0"
        # Store reference to codegen to access existing index variables
        self._codegen = state.codegen

    def index_var(self, block_idx: int) -> str:
        # Use the existing index variable from the active device loop
        # instead of the newly created one from TileStrategy.__init__
        return self._codegen.index_var(block_idx)

    def _reduction_thread_count(self) -> int:
        """Return the live thread extent of the reduced tile block.

        Unlike a real reduction axis, a tile block reduced over its inner
        (tiled) dim is mapped to a normal tile thread axis (plus, when the
        block is wider than its thread extent, a runtime lane loop). When that
        block has a live thread axis the partials must be combined ACROSS those
        threads, so report the thread extent (0 when the block has no live
        thread axis, e.g. a pure lane loop / serial dim — the base behavior).
        """
        extent = self.fn.tile_strategy.thread_extent_for_block_id(self.block_index)
        return extent if extent is not None and extent > 0 else 0

    def _lane_loop_cross_warp_group_params(
        self,
    ) -> tuple[int, int, int, str] | None:
        """Return ``(pre, group_span, group_count, lane_expr)`` for a tile block
        that is reduced over its inner (tiled) dim AND carries a runtime lane
        loop, or ``None`` when no cross-warp de-interleaving is required.

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
        finalize) unless the reduce group is genuinely cross-warp
        (``pre > 1`` and ``group_span`` a multiple of 32 greater than 32).
        """
        env = CompileEnvironment.current()
        backend = env.backend
        if backend.name != "cute":
            return None
        block_axes, axis_sizes = self._active_thread_layout()
        reduce_axis = block_axes.get(self.block_index)
        if reduce_axis is None:
            reduce_axis = self._aliased_active_thread_axis(block_axes)
        if reduce_axis is None:
            return None
        # Live thread extents per axis (sibling axes included) so the linear
        # lane index strides are computed correctly.
        logical_axis_sizes = {
            axis: size for axis, size in axis_sizes.items() if size > 1
        }
        if reduce_axis not in logical_axis_sizes:
            return None
        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        if pre <= 1:
            # The reduce axis is already at the bottom of the linear lane
            # index: consecutive warp lanes belong to the reduction, so the
            # plain warp reduce is correct (no de-interleaving needed).
            return None
        reduce_extent = logical_axis_sizes[reduce_axis]
        group_span = pre * reduce_extent
        if group_span <= 32 or group_span % 32 != 0:
            # Single-warp (or non-warp-aligned) groups are not handled by the
            # cross-warp two-stage path.
            return None
        num_threads = 1
        for size in logical_axis_sizes.values():
            num_threads *= size
        if num_threads % group_span != 0:
            return None
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if lane_expr is None:
            return None
        return pre, group_span, num_threads // group_span, lane_expr

    def _active_thread_layout(self) -> tuple[dict[int, int], dict[int, int]]:
        axis_sizes: dict[int, int] = {}
        block_axes: dict[int, int] = {}
        seen: set[int] = set()
        for loops in self._codegen.active_device_loops.values():
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
        current_grid = getattr(self._codegen, "current_grid_state", None)
        if isinstance(current_grid, DeviceGridState):
            for axis, size in current_grid.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
            block_axes.update(current_grid.block_thread_axes)
        return block_axes, axis_sizes

    def _aliased_active_thread_axis(self, block_axes: dict[int, int]) -> int | None:
        env = CompileEnvironment.current()
        target_block = self.block_index
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
        env = CompileEnvironment.current()
        target_block = self.block_index
        for strategy in self.fn.tile_strategy.strategies:
            for candidate_block_id in strategy.block_ids:
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

    def _strided_thread_reduction_expr(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        default_value: float | bool,
    ) -> str | None:
        env = CompileEnvironment.current()
        backend = env.backend
        current_grid = getattr(self._codegen, "current_grid_state", None)
        allow_lane_axis_fallback = (
            isinstance(current_grid, DeviceGridState) and current_grid.has_lane_loops()
        )
        normalized_dim = dim if dim >= 0 else fake_input.ndim + dim

        def debug(*parts: object) -> None:
            return None

        def block_thread_extent_hint(block_id: int) -> int | None:
            extent = self.fn.tile_strategy.thread_extent_for_block_id(block_id)
            if extent is not None:
                return extent
            configured_threads = env.config_spec.num_threads.config_get(
                self.fn.config.num_threads, block_id, 0
            )
            if configured_threads > 0:
                return configured_threads
            configured_block_size = self.fn.resolved_block_size(block_id)
            return (
                configured_block_size
                if isinstance(configured_block_size, int)
                else None
            )

        def active_loop_states() -> list[DeviceLoopState | DeviceGridState]:
            loop_states: list[DeviceLoopState | DeviceGridState] = []
            seen: set[int] = set()
            for loops in self._codegen.active_device_loops.values():
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
        planned_dims = self._planned_thread_dims()
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

        if backend.name != "cute":
            debug("skip backend", backend.name)
            return None
        if backend.is_indexed_reduction(reduction_type):
            debug("skip indexed", reduction_type)
            return None
        if self._reduction_block_is_serial():
            debug("skip serial", self.block_index)
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
        if self._reduction_block_has_lane_loops():
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
            if 0 <= reduce_axis < len(self._codegen.max_thread_block_dims):
                self._codegen.max_thread_block_dims[reduce_axis] = max(
                    self._codegen.max_thread_block_dims[reduce_axis],
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
                and self.fn.tile_strategy.thread_extent_for_block_id(aliased_block_id)
                is None
            ):
                aliased_block_id = None
            if aliased_block_id is not None:
                reduce_axis = self.fn.tile_strategy.thread_axis_for_block_id(
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
                        if 0 <= reduce_axis < len(self._codegen.max_thread_block_dims):
                            self._codegen.max_thread_block_dims[reduce_axis] = max(
                                self._codegen.max_thread_block_dims[reduce_axis],
                                reduce_thread_extent,
                            )
        if reduce_axis is None:
            strategy = self.fn.tile_strategy.block_id_to_strategy.get(
                (self.block_index,)
            )
            if strategy is not None:
                reduce_axis = self.fn.tile_strategy.thread_axis_for_strategy(strategy)
            if reduce_axis is not None:
                hint = _reduction_threads_from_annotation(state)
                if hint is None:
                    hint = backend.reduction_threads_hint(
                        self.block_size_var(self.block_index)
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
        for block_id, axis, extent, _info in active_thread_blocks:
            if block_id in mapped_block_ids or block_id >= self.block_index:
                logical_axis_sizes[axis] = max(
                    logical_axis_sizes.get(axis, 1),
                    extent,
                )
        if reduce_axis not in logical_axis_sizes and 0 <= reduce_axis < len(
            self._codegen.max_thread_block_dims
        ):
            reduce_size = axis_sizes.get(reduce_axis, 1)
            if reduce_thread_extent is None:
                reduce_size = max(
                    reduce_size,
                    self._codegen.max_thread_block_dims[reduce_axis],
                )
            logical_axis_sizes[reduce_axis] = reduce_size
        if not logical_axis_sizes:
            debug("skip no logical axis sizes", tuple(fake_input.size()), dim)
            return None
        for axis, size in logical_axis_sizes.items():
            if 0 <= axis < len(self._codegen.max_thread_block_dims):
                self._codegen.max_thread_block_dims[axis] = max(
                    self._codegen.max_thread_block_dims[axis], size
                )
        if reduce_thread_extent is None and 0 <= reduce_axis < len(
            self._codegen.max_thread_block_dims
        ):
            logical_axis_sizes[reduce_axis] = max(
                logical_axis_sizes.get(reduce_axis, 1),
                self._codegen.max_thread_block_dims[reduce_axis],
            )

        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        reduce_extent = logical_axis_sizes.get(reduce_axis, 1)
        group_span = pre * reduce_extent
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if lane_expr is None:
            debug("skip no lane expr", tuple(fake_input.size()), dim)
            return None

        dtype = _dtype_str(fake_input.dtype)
        identity_expr = backend.cast_expr(constant_repr(default_value), dtype)
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
        planned_dims = self.fn.tile_strategy.thread_block_dims()
        for axis, (recorded, planned) in enumerate(
            zip(self._codegen.max_thread_block_dims, planned_dims, strict=True)
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
        if pre <= 1 and group_span <= 32 and num_threads == group_span:
            debug(
                "skip small direct",
                tuple(fake_input.size()),
                dim,
                "block",
                self.block_index,
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
            self.block_index,
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
            smem_budget_bytes = _cute_shared_memory_budget_bytes()
            group_count = num_threads // group_span
            lane_var = self.fn.new_var("strided_lane", dce=True)
            lane_in_group_var = self.fn.new_var("strided_lane_in_group", dce=True)
            lane_mod_pre_var = self.fn.new_var("strided_lane_mod_pre", dce=True)
            state.add_statement(f"{lane_var} = {lane_expr}")
            state.add_statement(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
            state.add_statement(f"{lane_mod_pre_var} = ({lane_in_group_var}) % {pre}")
            if group_span % 32 == 0:
                warps_per_group = group_span // 32
                partials_size = group_count * pre * warps_per_group
                results_size = group_count * pre
                if (
                    _cute_reduction_smem_bytes(
                        partials_size + results_size, fake_input.dtype
                    )
                    > smem_budget_bytes
                ):
                    return None
                return self._strided_thread_reduction_expr_shared_two_stage(
                    state=state,
                    input_name=input_name,
                    reduction_type=reduction_type,
                    fake_input=fake_input,
                    identity_expr=identity_expr,
                    lane_var=lane_var,
                    lane_in_group_var=lane_in_group_var,
                    lane_mod_pre_var=lane_mod_pre_var,
                    pre=pre,
                    group_span=group_span,
                    group_count=group_count,
                )
            if (
                _cute_reduction_smem_bytes(
                    num_threads + group_count * pre, fake_input.dtype
                )
                > smem_budget_bytes
            ):
                return None
            return self._strided_thread_reduction_expr_shared_tree(
                state=state,
                input_name=input_name,
                reduction_type=reduction_type,
                fake_input=fake_input,
                identity_expr=identity_expr,
                lane_var=lane_var,
                lane_in_group_var=lane_in_group_var,
                lane_mod_pre_var=lane_mod_pre_var,
                pre=pre,
                group_span=group_span,
                num_threads=num_threads,
                group_count=group_count,
            )

        return (
            "_cute_grouped_reduce_warp("
            f"{input_name}, {reduction_type!r}, {identity_expr}, {lane_expr}, "
            f"pre={pre}, group_span={group_span})"
        )

    def _strided_thread_reduction_expr_shared_two_stage(
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
        group_count: int,
    ) -> str:
        result_var = self.fn.new_var("strided_reduce_result", dce=True)
        state.add_statement(
            f"{result_var} = _cute_grouped_reduce_shared_two_stage("
            f"{input_name}, {reduction_type!r}, {identity_expr}, "
            f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
            f"pre={pre}, group_span={group_span}, group_count={group_count})"
        )
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
        result_var = self.fn.new_var("strided_reduce_result", dce=True)
        state.add_statement(
            f"{result_var} = _cute_grouped_reduce_shared_tree("
            f"{input_name}, {reduction_type!r}, {identity_expr}, "
            f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
            f"pre={pre}, group_span={group_span}, "
            f"num_threads={num_threads}, group_count={group_count})"
        )
        return result_var

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        _log_cute_reduction_layout(state)
        default = ir.Reduction.default_accumulator(reduction_type, fake_input.dtype)
        assert isinstance(default, (float, int, bool))
        env = CompileEnvironment.current()
        dim_size = fake_input.size(dim)
        is_zero_dim = False
        if (
            isinstance(dim_size, int)
            and dim_size == 0
            or isinstance(dim_size, torch.SymInt)
            and env.known_equal(dim_size, 0)
        ):
            is_zero_dim = True
        if is_zero_dim:
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_output.size()])
            return expr_from_string(
                env.backend.full_expr(
                    shape_dims, constant_repr(default), fake_output.dtype
                )
            )
        if (
            strided_expr := self._strided_thread_reduction_expr(
                state, input_name, reduction_type, dim, fake_input, default
            )
        ) is not None:
            expr = strided_expr
        elif self._needs_loop_carried_accumulator():
            # The reduction block is not backed by a live thread axis in the
            # active loop nest (it is iterated either by a serial device loop,
            # by a lane loop, or has no thread axis at all).
            if (
                (
                    self._reduction_block_has_lane_loops()
                    or self._reduction_block_in_device_lane_loop()
                )
                and not self._lane_reduce_marker_unsupported(state)
                and (threads := self._lane_reduce_threads_in_group()) is not None
            ):
                # The block is split across a per-thread lane loop. The
                # single-pass lane loop can only produce a per-lane partial,
                # but every consumer needs the full reduction. Emit a marker
                # that the ``split_lane_loop_reductions`` post-pass rewrites
                # into a two-pass (accumulate across lanes -> combine across
                # ``threads`` -> consume) lane structure.
                from .tile_strategy import _lane_reduce_marker_expr

                acc_dtype = get_computation_dtype(fake_input.dtype)
                identity_expr = env.backend.cast_expr(
                    constant_repr(default), _dtype_str(acc_dtype)
                )
                group_params = self._lane_loop_cross_warp_group_params()
                if group_params is not None:
                    # The reduce group is spread across warps (the reduced tile
                    # dim sits ABOVE a sibling tile axis on the linear thread
                    # index). Carry the strided/grouped params so the post-pass
                    # finalize uses the cross-warp two-stage shared reduction
                    # instead of a (row-cross-contaminating) consecutive-lane
                    # warp reduce.
                    group_pre, group_span, group_count, group_lane_expr = group_params
                    expr = _lane_reduce_marker_expr(
                        input_name,
                        reduction_type,
                        identity_expr,
                        threads,
                        group_pre=group_pre,
                        group_span=group_span,
                        group_lane_expr=group_lane_expr,
                        group_count=group_count,
                    )
                else:
                    expr = _lane_reduce_marker_expr(
                        input_name, reduction_type, identity_expr, threads
                    )
            else:
                # A serial device loop (or no thread axis at all). A warp-level
                # reduction would fold together unrelated tensor elements, so
                # each iteration contributes only its current scalar value and
                # the surrounding loop-carried accumulator performs the real
                # reduction.
                expr = input_name
        else:
            expr = self.call_reduction_function(
                input_name,
                reduction_type,
                dim,
                fake_input,
                fake_output,
            )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))
