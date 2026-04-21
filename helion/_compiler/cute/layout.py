"""Core data structures for CuTe thread-value layout planning.

A ThreadLayout describes how a tile's elements are distributed across
(thread_id, value_id) pairs.  Shapes and strides use SymIntLike so they
can reference autotuned block sizes and dynamic tensor dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import functools
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    SymIntLike = torch.SymInt | int


class LayoutTag(enum.Enum):
    """Why a particular layout was chosen -- useful for debugging / cost model."""

    COALESCED = "coalesced"
    REDUCTION = "reduction"
    MMA_OPERAND_A = "mma_a"
    MMA_OPERAND_B = "mma_b"
    MMA_ACCUMULATOR = "mma_c"
    INHERITED = "inherited"
    IDENTITY = "identity"


class MatmulAxisRole(enum.Enum):
    """Semantic role of a logical matmul axis."""

    M = "m"
    N = "n"
    K = "k"


class MatmulExecutionKind(enum.Enum):
    """Planner-selected CuTe execution scheme for a matmul node."""

    DIRECT_GROUPED_N = "direct_grouped_n"


@dataclass(frozen=True)
class MatmulOperandAxes:
    """Logical matmul-axis assignment for one operand/output tensor."""

    m_dim: int | None = None
    n_dim: int | None = None
    k_dim: int | None = None


@dataclass(frozen=True)
class MatmulAxisModel:
    """Planner-owned mapping from tensor dims to matmul M/N/K roles."""

    lhs: MatmulOperandAxes
    rhs: MatmulOperandAxes
    out: MatmulOperandAxes


@dataclass(frozen=True)
class MatmulExecutionPlan:
    """Planner-owned execution layout for a CuTe matmul node."""

    kind: MatmulExecutionKind
    m_block_id: int
    scalar_block_id: int
    bm: int
    bn: int
    bk: int
    groups_per_lane: int
    lane_extent: int


@dataclass(frozen=True)
class CuTeGridExecutionPlan:
    """Branch-scoped CuTe thread-axis policy chosen by layout planning."""

    scoped_block_ids: frozenset[int]
    block_axis_priority: dict[int, int]
    disable_reduction_axis_reservation_for: frozenset[int]

    def applies_to_block(self, block_id: int) -> bool:
        return block_id in self.scoped_block_ids

    def applies_to_any_block(self, block_ids: tuple[int, ...] | list[int]) -> bool:
        return any(block_id in self.scoped_block_ids for block_id in block_ids)

    def priority_for_block(self, block_id: int) -> int | None:
        if block_id not in self.scoped_block_ids:
            return None
        return self.block_axis_priority.get(block_id)

    def disables_reduction_axis_reservation(self, block_id: int) -> bool:
        return (
            block_id in self.scoped_block_ids
            and block_id in self.disable_reduction_axis_reservation_for
        )


def _checked_div(a: SymIntLike, b: SymIntLike) -> SymIntLike:
    """Floor-divide *a* by *b*, asserting exact divisibility for concrete values."""
    if isinstance(a, int) and isinstance(b, int):
        if a % b != 0:
            raise ValueError(f"{a} is not divisible by {b}")
        return a // b
    return a // b  # type: ignore[operator, return-value]


def _sym_product(values: tuple[SymIntLike, ...]) -> SymIntLike:
    """Multiply all elements, returning int when possible."""
    return functools.reduce(operator.mul, values, 1)


@dataclass(frozen=True)
class ThreadLayout:
    """Thread-value layout for a CuTe tile.

    Maps (thread_id, value_id) -> element coordinate within a tile.

    * thread_shape / thread_stride  -- shape and stride of the thread dimension
    * value_shape / value_stride    -- shape and stride of the per-thread value dim
    * tag                           -- why this layout was picked (for debugging)

    All shapes/strides are tuples of ``SymIntLike`` so they can refer to
    autotuned block sizes (``torch.SymInt``) or concrete integers.
    """

    thread_shape: tuple[SymIntLike, ...]
    thread_stride: tuple[SymIntLike, ...]
    value_shape: tuple[SymIntLike, ...]
    value_stride: tuple[SymIntLike, ...]
    tag: LayoutTag = LayoutTag.IDENTITY

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def num_threads(self) -> SymIntLike:
        """Total number of threads used by this layout (may be symbolic)."""
        return _sym_product(self.thread_shape)

    def num_values(self) -> SymIntLike:
        """Number of values each thread handles (may be symbolic)."""
        return _sym_product(self.value_shape)

    def tile_numel(self) -> SymIntLike:
        """Total number of elements in the tile (threads * values)."""
        return self.num_threads() * self.num_values()  # type: ignore[return-value]

    def is_compatible(self, other: ThreadLayout) -> bool:
        """True if *other* produces the same thread-to-element mapping."""
        return (
            self.thread_shape == other.thread_shape
            and self.thread_stride == other.thread_stride
            and self.value_shape == other.value_shape
            and self.value_stride == other.value_stride
        )

    def with_tag(self, tag: LayoutTag) -> ThreadLayout:
        """Return a copy with a different tag."""
        return ThreadLayout(
            thread_shape=self.thread_shape,
            thread_stride=self.thread_stride,
            value_shape=self.value_shape,
            value_stride=self.value_stride,
            tag=tag,
        )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @staticmethod
    def make_row_major(
        rows: SymIntLike,
        cols: SymIntLike,
        *,
        num_threads: SymIntLike,
        tag: LayoutTag = LayoutTag.COALESCED,
    ) -> ThreadLayout:
        """Threads vary along cols (stride-1 dim) for coalesced access.

        For a row-major (M, K) tile where stride-1 is along K:
          thread = num_threads threads along K
          value = (rows, cols // num_threads) values per thread
        """
        vals_per_thread = _checked_div(cols, num_threads)
        return ThreadLayout(
            thread_shape=(num_threads,),
            thread_stride=(1,),
            value_shape=(rows, vals_per_thread),
            value_stride=(cols, 1),
            tag=tag,
        )

    @staticmethod
    def make_col_major(
        rows: SymIntLike,
        cols: SymIntLike,
        *,
        num_threads: SymIntLike,
        tag: LayoutTag = LayoutTag.COALESCED,
    ) -> ThreadLayout:
        """Threads vary along rows (stride-1 dim) for coalesced access.

        For a col-major (M, K) tile where stride-1 is along M:
          thread = num_threads threads along M
          value = (rows // num_threads, cols) values per thread
        """
        vals_per_thread = _checked_div(rows, num_threads)
        return ThreadLayout(
            thread_shape=(num_threads,),
            thread_stride=(1,),
            value_shape=(vals_per_thread, cols),
            value_stride=(1, rows),
            tag=tag,
        )

    @staticmethod
    def make_1d(
        numel: SymIntLike,
        *,
        num_threads: SymIntLike,
        tag: LayoutTag = LayoutTag.IDENTITY,
    ) -> ThreadLayout:
        """Simple 1D layout: each thread gets numel // num_threads elements."""
        vals_per_thread = _checked_div(numel, num_threads)
        return ThreadLayout(
            thread_shape=(num_threads,),
            thread_stride=(1,),
            value_shape=(vals_per_thread,),
            value_stride=(1,),
            tag=tag,
        )


@dataclass
class LayoutConstraint:
    """Attached to a node via ``node.meta["cute_layout_constraint"]``.

    Layouts are tracked per edge direction:

    * *preferred_input* / *input_layout* describe how this node consumes tensor
      inputs.
    * *preferred_output* / *output_layout* describe how this node produces its
      tensor output.
    * *required* marks the constraint as non-negotiable for propagation.

    Pointwise / passthrough ops typically resolve both sides to the same
    layout. Ops like ``load`` only have an output layout, while ``store`` and
    ``reduce`` only have an input layout.
    """

    preferred_input: ThreadLayout | None = None
    preferred_output: ThreadLayout | None = None
    input_layout: ThreadLayout | None = None
    output_layout: ThreadLayout | None = None
    matmul_axes: MatmulAxisModel | None = None
    matmul_plan: MatmulExecutionPlan | None = None
    required: bool = False

    def primary_layout(self) -> ThreadLayout | None:
        """Return the node's main resolved layout for logging/debugging."""
        return self.input_layout or self.output_layout

    def primary_preferred_layout(self) -> ThreadLayout | None:
        """Return the node's main preferred layout for logging/debugging."""
        return self.preferred_input or self.preferred_output
