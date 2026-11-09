from __future__ import annotations

import abc
import ast
import functools
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import TypeVar

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
from .device_function import find_block_size_symbols
from .host_function import HostFunction
from .inductor_lowering import install_inductor_kernel_handlers
from .roll_reduction import FREE_IOTA_REDUCTION_META
from .tile_strategy import CompactedShape
from .tile_strategy import CuteLaneAxis
from .tile_strategy import DeviceGridState
from .tile_strategy import DeviceLoopState
from .tile_strategy import LoopDimInfo
from .tile_strategy import PersistentReductionState
from .tile_strategy import ThreadAxisTracker
from .tile_strategy import TileStrategy

if TYPE_CHECKING:
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState
    from .lane_loop_context import LaneLoopContext
    from .tile_dispatch import TileStrategyDispatch

log = logging.getLogger(__name__)


def _dtype_str(dtype: torch.dtype) -> str:
    return CompileEnvironment.current().backend.dtype_str(dtype)


def reduction_acc_dtype(
    reduction_type: str, fake_input: torch.Tensor, fake_output: torch.Tensor
) -> torch.dtype:
    """The dtype a reduction accumulates and combines in.

    The reduction input already carries torch's promotion (a bool or integer
    sum arrives as Int64), so a value reduction accumulates in the output's
    computation dtype; argmin/argmax track the input's values.
    """
    if CompileEnvironment.current().backend.is_indexed_reduction(reduction_type):
        return get_computation_dtype(fake_input.dtype)
    return get_computation_dtype(fake_output.dtype)


def _strategies_concurrent_with_block(
    tile_dispatch: TileStrategyDispatch,
    block_index: int,
) -> list[TileStrategy]:
    """Return strategies that can co-execute with reduction ``block_index``.

    Drops reduction strategies that live in a control-flow branch mutually
    exclusive with ``block_index``'s branch so the per-block thread budget is
    not over-counted (CuTe branch-by-pid kernels). Outside that pattern (no
    branch paths) this returns every strategy unchanged.
    """
    from .device_ir import DeviceIR

    strategies = list(tile_dispatch.strategies)
    red_paths = tile_dispatch.reduction_branch_paths()
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


S = TypeVar("S", bound="ReductionStrategy")
E = TypeVar("E", bound="ReductionEmitter[Any]")


class ReductionEmitter(abc.ABC, Generic[S]):
    """A backend's half of a reduction strategy (``Backend.reduction_emitter``).

    The generic strategies call these hooks where a thread-level backend
    shapes the reduction: thread counts and lane loops at construction, the
    index preamble and the cross-thread combine.  A backend without an
    emitter takes the generic path at every hook.  Hooks that run before
    ``TileStrategy.__init__`` receive the strategy's ``fn`` and
    ``block_index`` explicitly.
    """

    def __init__(self, strategy: S) -> None:
        self.strategy: S = strategy

    @abc.abstractmethod
    def claims_thread_axis(self) -> bool:
        """Whether the reduction owns a launch thread axis."""

    @abc.abstractmethod
    def mark_cross_block_vec_lanes(self, state: CodegenState) -> None:
        """Record the enclosing vector lanes of other blocks before the reduce."""

    @abc.abstractmethod
    def record_axis_branch_path(self, state: CodegenState) -> None:
        """Record the control-flow branch under which the axis is claimed."""


class PersistentReductionEmitter(ReductionEmitter["PersistentReductionStrategy"]):
    """Hooks of a full-extent reduction (``PersistentReductionStrategy``)."""

    @abc.abstractmethod
    def reject_int_argument_extent(self, numel: object) -> None:
        """Refuse an extent sized by an int argument (before ``__init__``)."""

    @abc.abstractmethod
    def plan_hint_bucket(self, extent: object, size_hint: int) -> int | None:
        """The power-of-2 bucket a hint-sized extent is compiled for."""

    @abc.abstractmethod
    def plan_max_threads(self, max_threads: int) -> int:
        """Cap the backend's thread budget for this reduction."""

    @abc.abstractmethod
    def plan_thread_count(self, size_hint: int, max_threads: int) -> None:
        """Apply the per-block thread-count knob to ``strategy._thread_count``."""

    @abc.abstractmethod
    def plan_axis_partner(
        self, concurrent: list[TileStrategy], size_hint: int
    ) -> PersistentReductionStrategy | None:
        """A concurrent sibling whose thread axis this block indexes instead."""

    @abc.abstractmethod
    def plan_resident_reduction(self) -> bool:
        """Whether the reduction keeps its row resident in registers."""

    @abc.abstractmethod
    def prove_merged_tiled_extent(self, numel: object) -> bool:
        """Whether the extent is a reshape-merged product of tile blocks."""

    @abc.abstractmethod
    def plan_synthetic_lanes(
        self,
        lane_extent: int | None,
        size_hint: int,
        mask_var: str | None,
        merged_tiled_extent: bool,
    ) -> None:
        """Size the synthetic lane loop, vector row or register tile."""

    @abc.abstractmethod
    def check_shared_thread_extent(self, extent: int) -> None:
        """Refuse widening to a wider sibling's launch extent when unsafe."""

    @abc.abstractmethod
    def emit_hint_bucket_guard(
        self, state: CodegenState, block_size_var: str, expr_str: str
    ) -> None:
        """Host-side guard that the dynamic extent stays in its bucket."""

    @abc.abstractmethod
    def emit_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST | None:
        """The backend's combine, or ``None`` for ``call_reduction_function``."""


class LoopedReductionEmitter(ReductionEmitter["LoopedReductionStrategy"]):
    """Hooks of a rolled reduction (``LoopedReductionStrategy``)."""

    @abc.abstractmethod
    def plan_max_threads(
        self, fn: DeviceFunction, block_index: int, max_threads: int
    ) -> int:
        """Cap the backend's thread budget for this reduction."""

    @abc.abstractmethod
    def plan_requested_thread_count(
        self, fn: DeviceFunction, block_index: int, thread_count: int
    ) -> int:
        """Apply the per-block thread-count knob."""

    @abc.abstractmethod
    def plan_lane_loop(
        self, fn: DeviceFunction, block_index: int, block_size: int, thread_count: int
    ) -> None:
        """Split the chunk into per-thread lanes and a vector lattice."""

    @abc.abstractmethod
    def plan_chunk_mask(self, block_index: int, mask_var: str | None) -> None:
        """Whether a masked vector lattice keeps whole-chunk predicates."""

    @abc.abstractmethod
    def plan_rolled_cluster(self, state: CodegenState, numel: sympy.Expr) -> sympy.Expr:
        """Claim a thread-block-cluster split; the extent the roll covers."""

    @abc.abstractmethod
    def register_dynamic_trip_count(
        self,
        state: CodegenState,
        numel: sympy.Expr,
        offset_var: str,
        block_size_var: str,
    ) -> None:
        """Expose a symbolic extent's trip count as a constexpr parameter."""

    @abc.abstractmethod
    def emit_lane_loop_body(
        self,
        state: CodegenState,
        inner_body: list[ast.AST],
        numel: sympy.Expr,
        offset_var: str,
        index_var: str,
        block_size_var: str,
    ) -> list[ast.AST]:
        """The roll body: ``inner_body`` wrapped in its lane / vector loops."""

    @abc.abstractmethod
    def rolled_range(
        self,
        state: CodegenState,
        offset_var: str,
        block_size_var: str,
        range_begin: str,
        range_end: str,
    ) -> tuple[str, str]:
        """The roll's range bounds under a constexpr trip count or a cluster split."""

    @abc.abstractmethod
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
        """One ordered-key collective for a bounded low-precision argreduction."""

    @abc.abstractmethod
    def emit_cross_thread_reduction(
        self,
        state: CodegenState,
        device_loop: DeviceLoopState,
        acc: str,
        reduction_type: str,
        default: float | bool,
        acc_dtype: torch.dtype,
    ) -> str | None:
        """The cross-warp / sibling-axis finalize, or ``None`` for the warp reduce."""

    @abc.abstractmethod
    def check_indexed_reduction_lanes(
        self, state: CodegenState, device_loop: DeviceLoopState, reduction_type: str
    ) -> None:
        """Refuse an argreduce finalize interleaved with sibling lanes."""


class BlockReductionEmitter(ReductionEmitter["BlockReductionStrategy"]):
    """Hooks of a reduction over a tile dim (``BlockReductionStrategy``)."""

    @abc.abstractmethod
    def check_static_thread_extent(self, reduction_type: str) -> None:
        """Refuse a combine over an argument-sized thread extent."""

    @abc.abstractmethod
    def emit_sequence_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        fake_output: torch.Tensor,
        default: float | bool,
    ) -> str | None:
        """The resident sequence-reduction marker, when the config asks for it."""

    @abc.abstractmethod
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
        """Two-pass marker for a block distributed by a device lane loop."""

    @abc.abstractmethod
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
        """Grouped / shared-memory combine across the tile's thread axis."""

    @abc.abstractmethod
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
        """Two-pass marker for a lane-looped block without a live thread axis."""

    @abc.abstractmethod
    def record_lane_fold_site(self, call_text: str) -> None:
        """Register a per-lane cross-thread reduce for the lane fold pass."""


def _backend_emitter(strategy: ReductionStrategy, kind: type[E]) -> E | None:
    emitter = CompileEnvironment.current().backend.reduction_emitter(strategy)
    if emitter is None:
        return None
    assert isinstance(emitter, kind), (type(emitter), kind)
    return emitter


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

    @functools.cached_property
    def emitter(self) -> ReductionEmitter[Any] | None:
        """The backend's emitter for this reduction (``None``: generic path)."""
        return _backend_emitter(self, ReductionEmitter)

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

    def use_shared_thread_extent(self, extent: int) -> None:
        """Spread the reduction over ``extent`` threads of its launch axis.

        See ``TileStrategyDispatch._normalize_shared_reduction_thread_extents``.
        """
        raise self._shared_thread_extent_error(extent)

    def _shared_thread_extent_error(self, extent: int) -> exc.BackendUnsupported:
        return exc.BackendUnsupported(
            CompileEnvironment.current().backend_name,
            f"reduction block {self.block_index} spans "
            f"{self._reduction_thread_count()} lanes but its launch axis has "
            f"{extent} threads; the surplus lanes would corrupt the cross-lane "
            "reduction",
        )

    def _claims_thread_axis(self) -> bool:
        """Whether this reduction owns a launch thread axis.

        A reduction spread across several live threads owns one axis.  On the
        CuTe backend a reduction left with a single live thread (shrunk by
        ``CuteBackend.adjust_reduction_thread_count`` once the tile axes fill
        the CTA, or a dim whose padded extent is 1) walks its whole extent in
        its synthetic lane loop and indexes with the constant 0
        (``_index_init_expr``).  Its thread coordinate is never read, so it
        claims no axis and records no ``block_thread_axes`` entry: the
        dispatcher parks it on the lowest axis no threaded reduction holds
        (``TileStrategyDispatch.thread_axis_for_strategy``), which a tile may
        use.  Giving every such reduction a private axis ran past the three
        hardware axes as soon as a kernel held four reduction dims (an
        attention kernel whose value head dim differs from the q/k head dim,
        plus the two tile-sized dims that ``tile.index[:, None]`` slices
        allocate).
        """
        if (emitter := self.emitter) is not None:
            return emitter.claims_thread_axis()
        return self._reduction_thread_count() > 0

    def thread_axes_used(self) -> int:
        return 1 if self._claims_thread_axis() else 0

    def thread_block_sizes(self) -> list[int]:
        if not self._claims_thread_axis():
            return []
        return [self._reduction_thread_count()]

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
        block across a per-thread lane loop (PerThreadNDTileStrategy lanes).

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
        if (
            isinstance(self, PersistentReductionStrategy)
            and self._cute_axis_partner is not None
        ):
            return self._cute_axis_partner._get_thread_axis()
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
        acc_dtype = reduction_acc_dtype(reduction_type, fake_input, fake_output)
        if backend.is_indexed_reduction(reduction_type):
            return self.call_indexed_reduction(
                input_name,
                self.broadcast_str(self.argreduce_position_var(), fake_input, dim),
                reduction_type,
                dim,
                fake_output,
                dtype=acc_dtype,
            )
        return backend.reduction_expr(
            input_name,
            reduction_type,
            dim,
            block_size_var=self.block_size_var(self.block_index),
            dtype=acc_dtype,
        )

    def _index_init_expr(self, block_size_var: str, dtype: str, block_idx: int) -> str:
        env = CompileEnvironment.current()
        backend = env.backend
        size = env.block_sizes[block_idx].size
        if isinstance(size, int) and size == 0:
            return backend.reduction_index_zero_expr(dtype)
        if isinstance(size, torch.SymInt) and env.known_equal(size, 0):
            return backend.reduction_index_zero_expr(dtype)
        if self._reduction_thread_count() == 1:
            # On CuTe a single-thread reduction claims no thread axis of its
            # own (``_claims_thread_axis``); the axis the dispatcher parks it
            # on may belong to a sibling tile and must not shift this index.
            # On the other thread-aware backends a reduction shrunk to one
            # live thread (``adjust_reduction_thread_count``) still reserves a
            # thread axis (``thread_axes_used() == 1``).  Alone, that axis is
            # one thread wide and ``thread_idx()[axis]`` is always 0.  Beside
            # a sibling reduction with more threads the reductions share one
            # reservation and ``TileStrategy._compute_thread_axis_offset``
            # lets a tile strategy take this axis, so ``thread_idx()[axis]``
            # would alias the tile's thread id.  The constant 0 is the lane
            # index in both cases.
            return backend.reduction_index_zero_expr(dtype)
        return backend.reduction_index_expr(
            block_size_var, dtype, block_idx, axis=self._get_thread_axis()
        )

    def argreduce_position_var(self) -> str:
        """Positions along the reduced dim, as argmin/argmax return them.

        A reduction dim walks its whole extent from 0, so its index is the
        position.
        """
        return self.index_var(self.block_index)

    def call_indexed_reduction(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        fake_output: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
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
            dtype=dtype,
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

    @property
    def lane_context(self) -> LaneLoopContext:
        """The reduction lane's V-loop body the memory lowering splices packets
        into (``lane_loop_context.py``); registered by the looped and
        persistent strategies, which own a lane loop."""
        return self.fn.lane_contexts[self.block_index]


def _int_argument_symbol(numel: object) -> str | None:
    """The name of the int argument a symbolic extent depends on, by provenance.

    ``CompileEnvironment.to_fake`` records a bare input ``LocalSource`` for the
    unbacked symbol of an int argument and nothing else does; a tensor size's
    symbol carries a ``TensorPropertySource``, whatever host local names it
    (``m, n = x.size()`` gives the size symbol a ``NameOrigin`` too, which is
    why the origin's class cannot tell them apart).  ``None`` for a static
    extent and for an extent of tensor sizes and block sizes.
    """
    from torch._dynamo.source import LocalSource

    expr = numel._sympy_() if isinstance(numel, torch.SymInt) else numel
    if not isinstance(expr, sympy.Expr):
        return None
    var_to_sources = CompileEnvironment.current().shape_env.var_to_sources
    for symbol in sorted(expr.free_symbols, key=str):
        if not isinstance(symbol, sympy.Symbol):
            continue
        for source in var_to_sources.get(symbol, ()):
            if type(source) is LocalSource and source.is_input:
                return source.local_name
    return None


class PersistentReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
    ) -> None:
        from .device_ir import ReductionLoopGraphInfo

        env = CompileEnvironment.current()
        numel = env.specialize_expr(env.block_sizes[block_index].numel)
        if isinstance(numel, (int, sympy.Integer)):
            size_hint = int(numel)
        elif isinstance(numel, sympy.Expr):
            # Reduction extents can refer to tile sizes. Resolve those symbols
            # from this config before planning physical threads or lane loops;
            # the tracing hint can be larger or smaller than the chosen tile.
            block_symbols, _ = find_block_size_symbols(numel)
            configured_sizes = {
                symbol: sympy.Integer(size)
                for symbol, block_id in block_symbols.items()
                if isinstance(size := fn.resolved_block_size(block_id), int)
            }
            resolved_numel = numel.xreplace(configured_sizes)
            size_hint = shape_env_size_hint(env.shape_env, resolved_numel)
        else:
            size_hint = env.size_hint(numel)
        emitter = self.emitter
        if emitter is not None:
            emitter.reject_int_argument_extent(numel)
        extent = numel._sympy_() if isinstance(numel, torch.SymInt) else numel
        hint_bucket: int | None = None
        if emitter is not None:
            hint_bucket = emitter.plan_hint_bucket(extent, size_hint)
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
        self._cute_hint_bucket = hint_bucket
        self.offset_vars[block_index] = "0"
        # The ``num_threads`` entry the config spells for this block (0 when
        # it leaves the count to the backend).
        self._cute_requested_threads = 0
        # A concurrent sibling persistent block whose thread axis this block
        # indexes instead of claiming one (``plan_axis_partner``).
        self._cute_axis_partner: PersistentReductionStrategy | None = None
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            if emitter is not None:
                max_threads = emitter.plan_max_threads(max_threads)
            self._thread_count = next_power_of_2(min(size_hint, max_threads))
            if emitter is not None:
                emitter.plan_thread_count(size_hint, max_threads)
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
            if emitter is not None:
                partner = emitter.plan_axis_partner(concurrent, size_hint)
                if partner is not None:
                    # The shared axis is already in the budget as the partner.
                    self._cute_axis_partner = partner
                    concurrent = [s for s in concurrent if s is not partner]
            self._thread_count = env.backend.adjust_reduction_thread_count(
                self._thread_count, concurrent
            )
        self._synthetic_cute_lane_var: str | None = None
        self._synthetic_cute_lane_extent = 1
        # Persistent reductions expose the same unroll-vec protocol as rolled
        # reductions when one V-wide vector exactly covers each thread's
        # synthetic slice.  Restricting the first implementation to one vector
        # per thread avoids introducing a second loop-carried reduction level;
        # wider slices keep the established scalar synthetic-lane path.
        self._cute_reduction_lane_extent = 1
        self._cute_reduction_vec_width = 1
        # The synthetic lane's V-loop and lane body, filled in by the
        # emitter's ``emit_synthetic_lane_preamble`` for the memory lowering
        # (``lane_loop_context.py``).
        fn.register_lane(block_index)
        self._cute_resident_reduction = False
        self._cute_register_tile_predicted = False
        if emitter is not None:
            self._cute_resident_reduction = emitter.plan_resident_reduction()
        is_graph_reduction_dim = any(
            isinstance(graph, ReductionLoopGraphInfo) and block_index in graph.block_ids
            for graph in fn.codegen.codegen_graphs
        )
        merged_tiled_extent = emitter is not None and emitter.prove_merged_tiled_extent(
            numel
        )
        if self._thread_count > 0:
            # For a non-graph-reduction dim we always try to recover the
            # full extent through a synthetic lane loop. For a graph
            # reduction dim we need the synthetic lane loop whenever the
            # live thread count is below the full padded reduction extent.
            # This includes the backend's hardware thread cap, not just a
            # later adjustment for competing tile axes. Only a live thread for
            # every padded element makes the warp/cross-warp reduction cover
            # the whole axis without a lane loop. Without this, a shrunk
            # graph-reduction dim only addresses the first ``thread_count``
            # elements (e.g. layer_norm_bwd's feature axis), leaving the
            # remaining columns/partial sums uncomputed.
            needs_synthetic = (
                merged_tiled_extent
                or not is_graph_reduction_dim
                or (
                    self._thread_count < next_power_of_2(size_hint)
                    if max_threads is not None
                    else False
                )
            )
            if needs_synthetic:
                lane_extent = env.backend.create_synthetic_reduction_lanes(
                    self._thread_count, size_hint
                )
                if emitter is not None:
                    emitter.plan_synthetic_lanes(
                        lane_extent, size_hint, mask_var, merged_tiled_extent
                    )

    @functools.cached_property
    def emitter(self) -> PersistentReductionEmitter | None:
        return _backend_emitter(self, PersistentReductionEmitter)

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def use_shared_thread_extent(self, extent: int) -> None:
        if (emitter := self.emitter) is not None:
            emitter.check_shared_thread_extent(extent)
        # One thread per element: the extent mask also drops the surplus
        # threads, which then hold the identity through the widened combine.
        self._thread_count = extent
        if self._mask_var is None:
            self._mask_var = self.fn.new_var(f"mask_{self.block_index}", dce=True)

    def cute_tile_base_expr(self, block_id: int) -> str | None:
        """The persistent axis covers its whole extent: the tile base is 0."""
        if block_id != self.block_index:
            return None
        return self.offset_var(block_id)

    def cute_lane_axis(self, block_id: int) -> CuteLaneAxis | None:
        """Thread / synthetic-lane distribution of the persistent axis.

        ``None`` for another block, for a vectorised or resident row (their
        per-thread values are not one scalar per lane step) and when no live
        thread count is known.  Each synthetic lane step is one
        ``threads``-wide chunk of consecutive elements
        (``index = thread + step * threads``).
        """
        if (
            block_id != self.block_index
            or self._thread_count <= 0
            or self._cute_reduction_vec_width > 1
            or self._cute_resident_reduction
        ):
            return None
        lane_var = self._synthetic_cute_lane_var
        lane_steps = self._synthetic_cute_lane_extent if lane_var is not None else 1
        return CuteLaneAxis(
            extent=self._thread_count * lane_steps,
            threads=self._thread_count,
            lane_var=lane_var,
            lane_steps=lane_steps,
            vec_lane_var=None,
            vec_width=1,
            strided=lane_var is not None,
        )

    def offset_var(self, block_idx: int) -> str:
        assert block_idx == self.block_index
        return "0"

    def _codegen_index(
        self, state: CodegenState, block_size_var: str, numel: sympy.Expr
    ) -> None:
        """Set this thread's index (and mask) along the reduction dimension.

        Backends whose strategy walks the dimension in a lane loop override
        this to emit them per lane.
        """
        env = CompileEnvironment.current()
        index_var = self.index_var(self.block_index)
        state.add_statement(
            f"{index_var} = {self._index_init_expr(block_size_var, env.index_type(), self.block_index)}"
        )
        if self._mask_var is not None:
            state.add_statement(
                f"{self._mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
            )

    def codegen_preamble(self, state: CodegenState) -> None:
        env = CompileEnvironment.current()
        backend = env.backend
        emitter = self.emitter
        block_idx = self.block_index
        numel = env.specialize_expr(env.block_sizes[block_idx].numel)
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
                    if emitter is not None:
                        emitter.emit_hint_bucket_guard(state, block_size_var, expr_str)
        self._codegen_index(state, block_size_var, numel)
        # Extract end_var_name from the numel expression
        end_var_name = self.fn.sympy_expr(numel)
        block_id_to_info = {
            self.block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        tracker = ThreadAxisTracker()
        if self._claims_thread_axis():
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
        emitter = self.emitter
        if emitter is not None:
            emitter.record_axis_branch_path(state)
        numel = env.block_sizes[self.block_index].numel
        if isinstance(numel, sympy.Integer) and numel == 0:
            default = ir.Reduction.default_accumulator(
                reduction_type,
                reduction_acc_dtype(reduction_type, fake_input, fake_output),
            )
            assert isinstance(default, (float, int, bool))
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_output.size()])
            return expr_from_string(
                backend.full_expr(shape_dims, constant_repr(default), fake_output.dtype)
            )
        if (
            emitter is not None
            and (
                emitted := emitter.emit_reduction(
                    state, input_name, reduction_type, dim, fake_input, fake_output
                )
            )
            is not None
        ):
            return emitted
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
        emitter = self.emitter
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            if emitter is not None:
                max_threads = emitter.plan_max_threads(fn, block_index, max_threads)
            thread_count = next_power_of_2(min(block_size, max_threads))
            if emitter is not None:
                thread_count = emitter.plan_requested_thread_count(
                    fn, block_index, thread_count
                )
        else:
            thread_count = 0
        tile_dispatch = getattr(fn, "tile_strategy", None)
        if tile_dispatch is not None:
            thread_count = env.backend.adjust_reduction_thread_count(
                thread_count, tile_dispatch.strategies
            )
        self._thread_count = thread_count
        if thread_count > 0:
            # Thread-level backends (e.g. FlyDSL) may override the per-block thread
            # count of a looped whole-row reduction; tile-level backends return
            # None here and keep the default.
            _override = env.backend.looped_reduction_thread_count(
                requested=thread_count,
                block_size=block_size,
                block_index=block_index,
                config=fn.config,
                config_spec=env.config_spec,
            )
            if _override is not None:
                self._thread_count = _override
        self.block_size = block_size
        self._loop_block_size = block_size
        self._cute_reduction_lane_var: str | None = None
        self._cute_reduction_lane_extent = 1
        self._cute_reduction_vec_width = 1
        if emitter is not None:
            emitter.plan_lane_loop(fn, block_index, block_size, thread_count)
        numel = env.specialize_expr(env.block_sizes[block_index].numel)
        if env.known_multiple(numel, self._loop_block_size):
            mask_var: str | None = None
        else:
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)
        # A masked vec lattice keeps its vector transactions when every
        # V-wide chunk is provably all-in or all-out of the extent: a chunk
        # base is a multiple of V, so ``extent % V == 0`` makes the bounds
        # mask uniform across the chunk and it is evaluated once per chunk
        # (``_cute_reduction_chunk_uniform_mask``).  Without that proof
        # (a symbolic extent whose cache-key residue is not a multiple of V)
        # the lane body carries an explicit whole-chunk predicate so full
        # chunks still use one vector load/store while the straddling tail
        # chunk falls back to per-element accesses.
        self._cute_reduction_chunk_uniform_mask = False
        if emitter is not None:
            emitter.plan_chunk_mask(block_index, mask_var)
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_REDUCTION_BLOCK_{block_index}"),
        )
        self.offset_vars[block_index] = fn.new_var(f"roffset_{block_index}", dce=True)
        self.index_vars[block_index] = fn.new_var(f"rindex_{block_index}", dce=True)
        # ``cute_cluster_n`` split of the rolled range across a thread-block
        # cluster; decided lazily at the first ``codegen_device_loop`` (see
        # ``_maybe_apply_cute_rolled_cluster``).
        self._cute_rolled_cluster_n = 1
        self._cute_rolled_cluster_slice_len = 0
        self._cute_rolled_cluster_checked = False
        # Per-sweep lane body state the memory lowering splices vector hoists
        # and store flushes into (``lane_loop_context.py``); rebuilt by
        # every ``codegen_device_loop``.
        fn.register_lane(block_index)

    @functools.cached_property
    def emitter(self) -> LoopedReductionEmitter | None:
        return _backend_emitter(self, LoopedReductionEmitter)

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def _register_block_size_constexpr(
        self, state: CodegenState, block_size_var: str
    ) -> None:
        # Register the loop block size as a constexpr kernel param, defined host-side.
        if CompileEnvironment.current().backend.reduction_block_size_is_inlined_constexpr():
            # FlyDSL's scf.for step must be a value inside the loop, not an
            # external constexpr param, so inline the block size as a literal.
            state.device_function.constexpr_arg(block_size_var, self._loop_block_size)
            return
        if state.device_function.constexpr_arg(block_size_var):
            state.codegen.host_statements.append(
                statement_from_string(f"{block_size_var} = {self._loop_block_size!r}")
            )

    def cute_chunk_in_bounds_expr(self, state: CodegenState) -> str:
        """Predicate on the chunk base admitting one whole V-wide packet.

        When ``extent % V == 0`` the chunk base decides every lane, so the
        packet is in bounds exactly when its base is; otherwise its last lane
        must be in bounds.  The chunk-level mask statements of
        ``codegen_device_loop`` and the packet guard of the vector hoist are
        both spelled from this expression, so the load pipeliner rebases them
        alike.
        """
        base_var = self.lane_context.base_index_var
        assert base_var is not None
        numel_expr = state.sympy_expr(
            CompileEnvironment.current().block_sizes[self.block_index].numel
        )
        if self._cute_reduction_chunk_uniform_mask:
            return f"{base_var} < {numel_expr}"
        return f"{base_var} + {self._cute_reduction_vec_width - 1} < {numel_expr}"

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        env = CompileEnvironment.current()
        emitter = self.emitter
        block_index = self.block_index
        numel = env.specialize_expr(env.block_sizes[block_index].numel)
        if emitter is not None:
            numel = emitter.plan_rolled_cluster(state, numel)
        offset_var = self.offset_var(block_index)
        index_var = self.index_var(block_index)
        block_size_var = self.block_size_var(block_index)
        assert block_size_var is not None
        self._register_block_size_constexpr(state, block_size_var)
        if emitter is not None:
            emitter.register_dynamic_trip_count(
                state, numel, offset_var, block_size_var
            )
        inner_body: list[ast.AST] = [
            statement_from_string(
                f"{index_var} = {offset_var} + {self._index_init_expr(f'({block_size_var})', env.index_type(), block_index)}"
            ),
        ]
        body: list[ast.AST] = inner_body
        if emitter is not None:
            body = emitter.emit_lane_loop_body(
                state, inner_body, numel, offset_var, index_var, block_size_var
            )
        elif (mask_var := self._mask_var) is not None:
            inner_body.append(
                statement_from_string(
                    f"{mask_var} = {index_var} < {state.sympy_expr(numel)}"
                )
            )

        range_begin = "0"
        range_end = state.sympy_expr(numel)
        if emitter is not None:
            range_begin, range_end = emitter.rolled_range(
                state, offset_var, block_size_var, range_begin, range_end
            )
        for_node = create(
            ast.For,
            target=create(ast.Name, id=offset_var, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(
                    state.config,
                    [self.block_index],
                    begin=range_begin,
                    end=range_end,
                    step=block_size_var,
                ),
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        # Extract end_var_name from the actual numel expression used in the range()
        end_var_name = state.sympy_expr(numel)
        block_id_to_info = {
            block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        tracker = ThreadAxisTracker()
        if self._claims_thread_axis():
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
        emitter = self.emitter
        if emitter is not None:
            emitter.record_axis_branch_path(state)
        assert state.fx_node is not None
        if state.fx_node.meta.get(FREE_IOTA_REDUCTION_META):
            # Its input is computed outside the loop from a free hl.arange, so
            # every chunk would fold the same full-extent value; only the
            # computed-fragment lowering re-reads it per chunk.
            raise exc.BackendUnsupported(
                CompileEnvironment.current().backend.name,
                "a rolled reduction of a value computed outside its loop",
            )
        with install_inductor_kernel_handlers(state.codegen, {}):
            env = CompileEnvironment.current()
            backend = env.backend
            device_loop = state.codegen.active_device_loops[self.block_index][-1]
            assert isinstance(device_loop, DeviceLoopState)
            if emitter is not None:
                packed = emitter.emit_packed_argreduce(
                    state,
                    device_loop,
                    input_name,
                    reduction_type,
                    dim,
                    fake_input,
                    fake_output,
                )
                if packed is not None:
                    return packed
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_input.size()])
            acc_dtype = reduction_acc_dtype(reduction_type, fake_input, fake_output)
            default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
            assert isinstance(default, (float, int, bool))
            acc = self.fn.new_var(f"{state.fx_node.name}_acc", dce=True)
            acc_full = backend.reduction_acc_init_expr(
                shape_dims, constant_repr(default), acc_dtype
            )
            acc_full = backend.wrap_reduction_accumulator(
                acc_full,
                thread_count=self._thread_count,
                loop_block_size=self._loop_block_size,
                acc_dtype=acc_dtype,
            )
            device_loop.outer_prefix.append(
                statement_from_string(f"{acc} = {acc_full}")
            )
            result = self.fn.new_var(state.fx_node.name, dce=True)
            if not backend.is_indexed_reduction(reduction_type):
                combine_expr = backend.reduction_combine_expr(
                    reduction_type, acc, input_name, acc_dtype
                )
                state.add_statement(f"{acc} = {combine_expr}")
                expr = (
                    emitter.emit_cross_thread_reduction(
                        state, device_loop, acc, reduction_type, default, acc_dtype
                    )
                    if emitter is not None
                    else None
                ) or self.call_reduction_function(
                    acc,
                    reduction_type,
                    dim,
                    fake_input,
                    fake_output,
                )
            else:
                if emitter is not None:
                    emitter.check_indexed_reduction_lanes(
                        state, device_loop, reduction_type
                    )
                acc_index = self.fn.new_var(f"{state.fx_node.name}_acc_index", dce=True)
                index_dtype = env.index_dtype
                device_loop.outer_prefix.append(
                    statement_from_string(
                        f"{acc_index} = {backend.full_expr(shape_dims, repr(torch.iinfo(index_dtype).max), index_dtype)}"
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
                    dtype=acc_dtype,
                ):
                    state.add_statement(stmt)
                expr = self.call_indexed_reduction(
                    acc,
                    acc_index,
                    reduction_type,
                    dim,
                    fake_output,
                    dtype=acc_dtype,
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

    @functools.cached_property
    def emitter(self) -> BlockReductionEmitter | None:
        return _backend_emitter(self, BlockReductionEmitter)

    def index_var(self, block_idx: int) -> str:
        # Use the existing index variable from the active device loop
        # instead of the newly created one from TileStrategy.__init__
        return self._codegen.index_var(block_idx)

    def argreduce_position_var(self) -> str:
        # argmax over ``x[tm, tn]`` counts positions from the tile's first
        # element, as torch does on the tile; the index holds tile.begin +
        # position.
        return (
            f"(({self.index_var(self.block_index)}) - "
            f"({self._codegen.tile_begin_var(self.block_index)}))"
        )

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

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        emitter = self.emitter
        acc_dtype = reduction_acc_dtype(reduction_type, fake_input, fake_output)
        default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
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
        if emitter is not None:
            emitter.check_static_thread_extent(reduction_type)
        has_lane_loop = (
            self._reduction_block_has_lane_loops()
            or self._reduction_block_in_device_lane_loop()
        )
        if has_lane_loop and not env.backend.supports_lane_loop_reductions():
            raise exc.BackendUnsupported(
                env.backend.name,
                f"{reduction_type} reduction over an axis strided by a tile lane "
                "loop; this backend does not support lane-loop reductions",
            )
        if (
            emitter is not None
            and (
                sequence_expr := emitter.emit_sequence_reduction(
                    state, input_name, reduction_type, fake_output, default
                )
            )
            is not None
        ):
            expr = sequence_expr
        elif (
            emitter is not None
            and (
                lane_marker_expr := emitter.emit_device_lane_loop_marker(
                    state,
                    input_name,
                    reduction_type,
                    dim,
                    fake_input,
                    acc_dtype,
                    default,
                )
            )
            is not None
        ):
            expr = lane_marker_expr
        elif (
            env.backend.is_indexed_reduction(reduction_type)
            and self._reduction_block_in_device_lane_loop()
        ):
            # The per-element combine below would leave each lane's own
            # winner; only the two-pass lane marker sees every lane.
            raise exc.BackendUnsupported(
                env.backend.name,
                f"{reduction_type} over tile dim {self.block_index} split across "
                "a lane loop the two-pass lane reduction cannot place",
            )
        elif (
            emitter is not None
            and (
                strided_expr := emitter.emit_strided_thread_reduction(
                    state,
                    input_name,
                    reduction_type,
                    dim,
                    fake_input,
                    default,
                    acc_dtype,
                )
            )
            is not None
        ):
            expr = strided_expr
        elif self._needs_loop_carried_accumulator():
            # The reduction block is not backed by a live thread axis in the
            # active loop nest (it is iterated either by a serial device loop,
            # by a lane loop, or has no thread axis at all).
            if (
                has_lane_loop
                and emitter is not None
                and (
                    lane_loop_expr := emitter.emit_carried_lane_loop_reduction(
                        state,
                        input_name,
                        reduction_type,
                        dim,
                        fake_input,
                        acc_dtype,
                        default,
                    )
                )
                is not None
            ):
                # The block is split across a per-thread lane loop: the
                # single-pass lane loop can only produce a per-lane partial,
                # so the emitter's two-pass marker carries the full reduction.
                expr = lane_loop_expr
            else:
                # A serial device loop (or no thread axis at all). A warp-level
                # reduction would fold together unrelated tensor elements, so
                # each iteration contributes only its current scalar value and
                # the surrounding loop-carried accumulator performs the real
                # reduction.
                expr = input_name
                if env.backend.is_indexed_reduction(reduction_type):
                    # That one element wins at its own position, not its value.
                    expr = env.backend.cast_expr(
                        self.argreduce_position_var(),
                        env.backend.dtype_str(torch.int64),
                    )
        else:
            expr = self.call_reduction_function(
                input_name,
                reduction_type,
                dim,
                fake_input,
                fake_output,
            )
            # The direct single-warp ``cute.arch.warp_reduction_*`` (the
            # strided path's "skip small direct"): inside a vectorized lane
            # loop it reduces one lane's packet like the strided forms, so
            # it is a fold site too.
            if emitter is not None:
                emitter.record_lane_fold_site(expr)
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))
