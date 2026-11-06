"""Metal's reduction strategy subclasses.

A persistent reduction dimension can be wider than the threads the tile axes
leave it.  Each thread then walks its share of the dimension in a lane loop
(see ``MetalBackend.create_reduction_strategy``).  A dimension that is only
sliced (``x[tile, :]``) needs nothing more.  A reduction over the dimension
first folds each thread's lanes (``_metal_lane_fold``, unrolled by
:mod:`.lane_fold`), then combines the folds across threads as usual.  A
slice block that one value or access repeats walks each extra position in a
lane loop of its own (see ``repeated_blocks``).

A reduction over a tile block after the block's loop has exited must still
combine across the threads that loop used (``MetalBlockReductionStrategy``).
One over a block that its own loop walks in a lane loop, on one thread or
several, splits that lane loop into passes (``lane_reduction.py``).  A count
it sums there (``_is_count``) folds and combines in int32, not torch's int64.

A reduction block that is only the power-of-two padding of a ``:`` slice
block (see ``padded_twin``) owns no threads, lanes or mask: its reductions
run on the slice block's strategy (``MetalPaddedTwinStrategy``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _tracing_ops
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..reduction_strategy import BlockReductionStrategy
from ..reduction_strategy import PersistentReductionStrategy
from ..reduction_strategy import reduction_acc_dtype
from ..tile_strategy import LoopDimInfo
from ..tile_strategy import PersistentReductionState
from .backend import MetalBackend
from .backend import budget_note
from .lane_fold import LANE_FOLD
from .lane_fold import MAX_LANE_COPIES
from .lane_reduction import lane_split_for
from .msl_reduction import SUPPORTED_REDUCTIONS
from .padded_twin import cone_violation
from .padded_twin import padded_twins
from .reduction import exited_block_thread_axis

if TYPE_CHECKING:
    import ast

    import sympy

    from ..device_function import DeviceFunction
    from ..inductor_lowering import CodegenState


class MetalPersistentReductionStrategy(PersistentReductionStrategy):
    """A persistent reduction strategy that can serialize across lanes."""

    _lane_var: str | None = None
    _lane_extent = 1
    _lane_span = 0
    _serialized_extent = 0
    _folds = False
    _position_lanes: dict[int, str] | None = None
    _position_extent = 1
    _position_hosts: dict[int, tuple[int, int]] | None = None

    @property
    def serializes_lanes(self) -> bool:
        """Whether each thread walks several positions in a lane loop."""
        return self._lane_var is not None

    def serialize_lanes(self, extent: int) -> None:
        """Cover *extent* positions with the threads available, in a lane loop.

        Thread ``t`` covers positions ``t, t + span, t + 2 * span, ...``, so
        neighbouring threads still touch neighbouring elements.  With a single
        thread the strategy claims no thread axis at all and that thread walks
        the whole dimension, leaving the axis to a dimension that needs it.
        """
        span = self._thread_count
        self._lane_var = self.fn.new_var(f"slice_lane_{self.block_index}", dce=False)
        self._lane_extent = -(-extent // span)
        self._lane_span = span
        self._serialized_extent = extent
        if extent % span != 0 and self._mask_var is None:
            # A non-dividing extent masks the tail, per lane.
            self._mask_var = self.fn.new_var(f"mask_{self.block_index}", dce=True)
        if span == 1:
            self._thread_count = 0

    def thread_lanes(self) -> tuple[int | None, int, str | None]:
        """``(axis, threads, lane)``: position ``e`` of the dimension is on
        thread ``e % threads`` of ``axis`` and on lane ``e // threads`` of the
        lane loop ``lane`` (None without one).  One thread walking every
        position has ``threads`` 0 and no axis (``serialize_lanes``)."""
        threads = self._reduction_thread_count()
        axis = self._get_thread_axis() if threads else None
        return axis, threads, self._lane_var

    def fold_lanes(self) -> None:
        """Make reductions over the laned dimension fold each thread's lanes
        before combining across threads (``codegen_reduction``)."""
        assert self._lane_var is not None
        block_id = self.block_index
        twins = padded_twins(block_id)
        others = [
            info.block_id
            for info in CompileEnvironment.current().block_sizes
            if info.reduction
            and info.block_id != block_id
            and info.block_id not in twins
        ]
        if others:
            # Folds are shown correct only as the kernel's single reduction
            # dim.  Its padded twins are the same dim and fold with it.
            raise exc.BackendUnsupported(
                "metal",
                f"reduction over persistent dim {block_id} in a lane loop beside "
                f"other reduction dims {others}{budget_note()}",
            )
        if self._lane_extent > MAX_LANE_COPIES:
            span = self._lane_span
            threads = "1 thread" if span == 1 else f"{span} threads"
            raise exc.BackendUnsupported(
                "metal",
                f"persistent reduction over {self._serialized_extent} elements "
                f"would need {self._lane_extent} lanes per thread on {threads} "
                f"(max {MAX_LANE_COPIES}){budget_note()}",
            )
        self._folds = True

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        if not self._folds:
            return super().codegen_reduction(
                state, input_name, reduction_type, dim, fake_input, fake_output
            )
        backend = CompileEnvironment.current().backend
        if backend.is_indexed_reduction(reduction_type):
            raise exc.BackendUnsupported(
                "metal",
                f"{reduction_type} over a persistent dim walked in a lane loop"
                f"{budget_note()}",
            )
        # Checked here rather than left to the cross-thread combine, which a
        # single-thread dimension never emits.
        if reduction_type not in SUPPORTED_REDUCTIONS:
            raise exc.BackendUnsupported("metal", f"reduction {reduction_type!r}")
        # Resolved now: the fold is unrolled at launch, with no compile
        # environment to map the dtype.
        acc_type = backend.acc_type(
            reduction_acc_dtype(reduction_type, fake_input, fake_output)
        )
        fold = self.fn.new_var("lane_fold", dce=True)
        state.add_statement(
            f"{fold} = {LANE_FOLD}({reduction_type!r}, {input_name}, "
            f"{self._lane_var}, {acc_type!r})"
        )
        if self._thread_count == 0:
            return expr_from_string(
                self.maybe_reshape(fold, dim, fake_input, fake_output)
            )
        return super().codegen_reduction(
            state, fold, reduction_type, dim, fake_input, fake_output
        )

    def add_position_lanes(
        self,
        positions: list[int],
        extent: int,
        hosts: dict[int, tuple[int, int]],
    ) -> None:
        """Walk the block's whole *extent* at each of *positions*, counted
        from the right of a value's shape.  *hosts* maps a position to the
        thread axis it takes and that axis's thread count; a position it
        lacks walks a lane loop.  The lowest position keeps the block's own
        index."""
        self._position_lanes = {
            position: self.fn.new_var(
                f"position_lane_{self.block_index}_{position}", dce=False
            )
            for position in positions
        }
        self._position_extent = extent
        self._position_hosts = hosts

    def position_lane_mask(self, position: int) -> str | None:
        """The mask of a position on a thread axis with more threads than
        the block's extent."""
        assert self._position_hosts is not None
        host = self._position_hosts.get(position)
        if host is None or host[1] == self._position_extent:
            return None
        assert self._position_lanes is not None
        return f"({self._position_lanes[position]} < {self._position_extent})"

    def position_lane(self, position: int) -> str | None:
        if self._position_lanes is None:
            return None
        return self._position_lanes.get(position)

    def _codegen_index(
        self, state: CodegenState, block_size_var: str, numel: sympy.Expr
    ) -> None:
        self._codegen_thread_index(state, block_size_var, numel)
        if not self._position_lanes:
            return
        current_grid = state.codegen.current_grid_state
        if current_grid is None:
            raise exc.BackendUnsupported(
                "metal", "a repeated `:` slice block's lane loops need a grid loop"
            )
        backend = CompileEnvironment.current().backend
        assert isinstance(backend, MetalBackend) and self._position_hosts is not None
        for position, lane in self._position_lanes.items():
            if (host := self._position_hosts.get(position)) is not None:
                state.add_statement(f"{lane} = {backend.thread_index(host[0])}")
            else:
                # Full extent, so a lane needs no mask.
                current_grid.add_lane_loop(
                    self.block_index, lane, self._position_extent
                )

    def _codegen_thread_index(
        self, state: CodegenState, block_size_var: str, numel: sympy.Expr
    ) -> None:
        current_grid = state.codegen.current_grid_state
        lane_var = self._lane_var
        if lane_var is None or current_grid is None:
            super()._codegen_index(state, block_size_var, numel)
            return
        env = CompileEnvironment.current()
        block_idx = self.block_index
        index_var = self.index_var(block_idx)
        # The index and mask are evaluated per lane, inside the loop
        # DeviceGridState.wrap_body emits, like a tile's lane loop.
        current_grid.add_lane_loop(block_idx, lane_var, self._lane_extent)
        if self._thread_count == 0:
            index_expr = lane_var
        else:
            base = self._index_init_expr(block_size_var, env.index_type(), block_idx)
            index_expr = f"({base}) + {lane_var} * {self._thread_count}"
            axis = self._get_thread_axis()
            current_grid.thread_axis_sizes[axis] = max(
                current_grid.thread_axis_sizes.get(axis, 1), self._thread_count
            )
            current_grid.block_thread_axes[block_idx] = axis
        current_grid.lane_setup_statements.append(
            statement_from_string(f"{index_var} = {index_expr}")
        )
        if self._mask_var is not None:
            current_grid.lane_setup_statements.append(
                statement_from_string(
                    f"{self._mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
                )
            )


class MetalPaddedTwinStrategy(MetalPersistentReductionStrategy):
    """A reduction block that is the power-of-two padding of block *owner*.

    It claims no thread axis, emits no index, and reports the owner's mask,
    so ``_mask_to`` drops the owner's padding from every reduction over it.
    Its reductions are the owner's: they fold the owner's lanes and combine
    over the owner's threads.  Its own index is never defined, so a use of it
    fails MSL compilation rather than reading the wrong elements.
    """

    def __init__(self, fn: DeviceFunction, block_index: int, owner: int) -> None:
        super().__init__(fn, block_index)
        self._owner_id = owner
        self._thread_count = 0
        self._mask_var = None

    def owner(self) -> MetalPersistentReductionStrategy:
        # The owner is never rolled: once typing merges its dim with the
        # padded one, every reduction over the dim sees the padded size and
        # is recorded on the twin, and only recorded blocks roll.
        strategy = self.fn.tile_strategy.block_id_to_strategy[(self._owner_id,)]
        assert isinstance(strategy, MetalPersistentReductionStrategy)
        assert not isinstance(strategy, MetalPaddedTwinStrategy)
        return strategy

    def mask_var(self, block_idx: int) -> str | None:
        assert block_idx == self.block_index
        return self.owner().mask_var(self._owner_id)

    def codegen_preamble(self, state: CodegenState) -> None:
        numel = CompileEnvironment.current().block_sizes[self._owner_id].numel
        state.codegen.push_active_loops(
            PersistentReductionState(
                self,
                block_id_to_info={
                    self.block_index: LoopDimInfo(
                        end_var_name=self.fn.sympy_expr(numel), end_expr=numel
                    )
                },
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
        padded = env.block_sizes[self.block_index].size
        real = env.block_sizes[self._owner_id].size
        assert isinstance(padded, int)
        assert isinstance(real, int)
        assert state.fx_node is not None
        reason = cone_violation(
            state.fx_node, padded, real, self.fn.codegen.codegen_graphs
        )
        if reason is not None:
            raise exc.BackendUnsupported(
                "metal",
                f"{reduction_type} over dim {self.block_index}, the power-of-two "
                f"padding of dim {self._owner_id}: {reason}, so its positions "
                f"past {real} may be real elements; reduce only values sliced "
                f"from a {real}-long dim",
            )
        return self.owner().codegen_reduction(
            state, input_name, reduction_type, dim, fake_input, fake_output
        )


def _is_count(node: torch.fx.Node) -> bool:
    """Whether every element of an FX value is 0 or 1: a bool, widened to any
    integer dtype, its masked-off elements filled with 0."""
    if node.target is _tracing_ops._mask_to:
        operand, fill = node.args
        # Helion masks a sum's operand with the sum's identity.
        assert fill == 0, fill
    elif node.target is torch.ops.prims.convert_element_type.default:
        operand, dtype = node.args
        assert isinstance(dtype, torch.dtype)
        if dtype.is_floating_point or dtype.is_complex or dtype == torch.bool:
            return False
    else:
        return node.meta["val"].dtype == torch.bool
    assert isinstance(operand, torch.fx.Node)
    return _is_count(operand)


class MetalBlockReductionStrategy(BlockReductionStrategy):
    """A tile-block reduction that also combines after the block's loop exits.

    The shared strategy passes each thread's value through whenever the block
    has no live thread axis in the active loop nest.  That is right inside the
    block's loop, where the loop-carried accumulator combines, but after the
    loop the carried value still holds one partial per thread of the loop's
    axis, and nothing would combine them.
    """

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        split = lane_split_for(state, self.fn, self.block_index, reduction_type)
        if split is None:
            return super().codegen_reduction(
                state, input_name, reduction_type, dim, fake_input, fake_output
            )
        threads = self._reduction_thread_count()
        assert (threads > 1) == (self.block_index in split.loop.block_thread_axes)
        if self._sums_count_in_int32(state, reduction_type, fake_output):
            fake_input = fake_input.new_empty(fake_input.shape, dtype=torch.int32)
            fake_output = fake_output.new_empty(fake_output.shape, dtype=torch.int32)
        across_threads = None
        if threads > 1:

            def across_threads(partial: str) -> str:
                return self.call_reduction_function(
                    partial, reduction_type, dim, fake_input, fake_output
                )

        result = split.emit_marker(
            state, input_name, reduction_type, fake_input, fake_output, across_threads
        )
        return expr_from_string(
            self.maybe_reshape(result, dim, fake_input, fake_output)
        )

    def _sums_count_in_int32(
        self, state: CodegenState, reduction_type: str, fake_output: torch.Tensor
    ) -> bool:
        """Whether a sum into torch's int64 may fold and combine in int32.

        An int64 threadgroup collective is markedly slower than an int32 one
        on Apple GPUs.  A count over one block of fewer than 2**31 elements
        never leaves [0, 2**31 - 1] in any partial or combine, so int32 holds
        it exactly; the reduction's lowering widens the result back.
        """
        if reduction_type != "sum" or fake_output.dtype != torch.int64:
            return False
        assert state.fx_node is not None
        operand = state.fx_node.args[0]
        assert isinstance(operand, torch.fx.Node)
        block = self.fn.resolved_block_size(self.block_index)
        assert isinstance(block, int)
        return block < 2**31 and _is_count(operand)

    def _needs_loop_carried_accumulator(self) -> bool:
        """Combine an exited block across its thread axis when the layout is
        proven, raise when it is not.

        The combine is correct only when both hold:

        - the block is one element per thread (its size equals its thread
          extent), so each thread's carried value is its own element.  With
          lanes, a thread's value already folded its lanes with the carry's op;
          with surplus threads, some threads hold no element;
        - a grid execution plan fixes the block's axis, so the axis the body
          used is the one :func:`exited_block_thread_axis` returns.

        Every other exited block spread over more than one thread raises.  A
        block on at most one thread keeps the passthrough.
        """
        if not super()._needs_loop_carried_accumulator():
            return False
        block_id = self.block_index
        if self._codegen.active_device_loops.get(block_id):
            return True
        extent = self.fn.tile_strategy.thread_extent_for_block_id(block_id)
        if extent is None or extent <= 1:
            return True
        size = self.fn.resolved_block_size(block_id)
        if size != extent:
            raise exc.BackendUnsupported(
                "metal",
                f"reduction over tile dim {block_id} after its loop: its block "
                f"of {size} is spread over {extent} threads, not one element "
                "per thread; set num_threads to equal the block size",
            )
        if exited_block_thread_axis(self.fn, block_id) is None:
            raise exc.BackendUnsupported(
                "metal",
                f"reduction over tile dim {block_id} after its loop: no layout "
                "plan fixes the thread axis its loop used (threaded sibling "
                "loops, a reduction dimension or a matmul prevent one); reduce "
                "inside the loop, or run the other sibling loops on one thread",
            )
        return False
