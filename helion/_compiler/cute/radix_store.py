"""Mixed-radix destination views for rank-two tcgen05 tile stores.

Inductor fuses view and permute ops into the store of a native matmul: a
head split writes ``acc[y, x]`` to ``out[y // 512, x // 64, y % 512, x % 64]``
of a ``[16, 16, 512, 64]`` destination.  Every element's address is then
``rows(y) + cols(x)``, each a CuTe layout over one tile coordinate
(``(512, 16):(64, 524288)`` and ``(64, 16):(1, 32768)`` here), so a TMA store
through one hierarchical ``[M, N]`` tensor addresses the destination and the
epilogue is otherwise the identity store.

The proof interprets the store's index arithmetic on concrete coordinates; it
never matches an expression pattern.  It evaluates every ``(y, x)`` of the
problem once (in row chunks), so it is exact for the static shapes it admits.
Block-size symbols are admitted only as the extents of a coordinate reshape
(their value differs per config; the proof does not depend on the config).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.node import Node
from torch.utils._python_dispatch import _disable_current_modes

from ...language import tile_index
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

# Elements of the (M, N) problem the proof evaluates, and per chunk.
_MAX_PROOF_ELEMENTS = 1 << 28
_CHUNK_ELEMENTS = 1 << 22
# A TMA descriptor addresses at most five dimensions.
_TMA_MAX_RANK = 5

_INDEX_OPS: dict[object, Callable[..., object]] = {
    torch.ops.aten.add.Tensor: torch.add,
    torch.ops.aten.add.Scalar: torch.add,
    torch.ops.aten.sub.Tensor: torch.sub,
    torch.ops.aten.sub.Scalar: torch.sub,
    torch.ops.aten.mul.Tensor: torch.mul,
    torch.ops.aten.mul.Scalar: torch.mul,
    torch.ops.aten.remainder.Scalar: torch.remainder,
    torch.ops.aten.remainder.Tensor: torch.remainder,
    torch.ops.aten.floor_divide.default: torch.floor_divide,
}
_INDEX_DTYPES = frozenset({torch.int8, torch.int16, torch.int32, torch.int64})
_SHAPE_OPS = (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
)


@dataclasses.dataclass(frozen=True)
class RadixStoreLayout:
    """``(extent, stride)`` modes, innermost first, of each tile coordinate."""

    rows: tuple[tuple[int, int], ...]
    cols: tuple[tuple[int, int], ...]

    @staticmethod
    def _mode_expr(modes: tuple[tuple[int, int], ...], part: int) -> str:
        values = [str(mode[part]) for mode in modes]
        return values[0] if len(values) == 1 else f"({', '.join(values)})"

    def cute_layout_args(self) -> tuple[str, str]:
        """``(shape, stride)`` expressions of the hierarchical ``[M, N]`` layout."""
        shape = f"({self._mode_expr(self.rows, 0)}, {self._mode_expr(self.cols, 0)})"
        stride = f"({self._mode_expr(self.rows, 1)}, {self._mode_expr(self.cols, 1)})"
        return shape, stride

    @property
    def extents(self) -> tuple[int, int]:
        """The ``[M, N]`` extents the layout covers."""
        rows = cols = 1
        for extent, _stride in self.rows:
            rows *= extent
        for extent, _stride in self.cols:
            cols *= extent
        return rows, cols

    @staticmethod
    def _tiles(modes: tuple[tuple[int, int], ...], tile: int) -> bool:
        """Whether ``tile``-long blocks of the coordinate are CuTe tiles of
        ``modes``: walking innermost first, the tile divides the current
        extent or is a whole multiple of it (the quotient carries on)."""
        for extent, _stride in modes:
            if tile <= extent:
                return extent % tile == 0
            if tile % extent:
                return False
            tile //= extent
        return tile == 1

    def tiles(self, rows_per_tile: int, cols_per_tile: int) -> bool:
        """Whether ``cute.local_tile`` partitions the layout into
        ``rows_per_tile x cols_per_tile`` tiles (a tile that neither divides
        nor spans whole runs is partitioned silently wrong)."""
        return self._tiles(self.rows, rows_per_tile) and self._tiles(
            self.cols, cols_per_tile
        )

    def tma_store_supported(
        self, *, rows_per_store: int, cols_per_store: int, element_bytes: int
    ) -> bool:
        """Whether a ``rows_per_store x cols_per_store`` TMA box stays inside
        one innermost run of each coordinate, with the TensorMap's contiguous
        innermost dimension and 16-byte outer strides."""
        modes = (*self.rows, *self.cols)
        return (
            len(modes) <= _TMA_MAX_RANK
            and self.cols[0][1] == 1
            and self.rows[0][0] % rows_per_store == 0
            and self.cols[0][0] % cols_per_store == 0
            and all(
                (stride * element_bytes) % 16 == 0
                for extent, stride in modes
                if stride != 1 and extent > 1
            )
        )


def _mixed_radix(values: torch.Tensor) -> tuple[tuple[int, int], ...] | None:
    """The coalesced mixed-radix layout reproducing ``values`` (``values[0] ==
    0``) over ``range(len(values))``, or ``None``."""
    length = values.numel()
    flat = values.tolist()
    modes: list[tuple[int, int]] = []
    period = 1
    while period < length:
        stride = flat[period]
        if stride <= 0:
            return None
        extent = 1
        while extent * period < length and flat[extent * period] == extent * stride:
            extent += 1
        if length % (period * extent):
            return None
        # Coalesced: the run stops where the next stride is not its
        # continuation.
        modes.append((extent, stride))
        period *= extent
    if not modes:
        modes.append((1, 0))
    index = torch.arange(length, dtype=torch.int64, device="cpu")
    rebuilt = torch.zeros_like(index)
    period = 1
    for extent, stride in modes:
        rebuilt += (index // period) % extent * stride
        period *= extent
    if not torch.equal(rebuilt, values):
        return None
    return tuple(modes)


def _injective(modes: Sequence[tuple[int, int]]) -> bool:
    """Non-overlapping strides: every mode starts past the span of the ones
    with smaller strides."""
    span = 0
    for stride, extent in sorted((stride, extent) for extent, stride in modes):
        if extent == 1:
            continue
        if stride < span:
            return False
        span = stride * extent
    return True


class _Unsupported(Exception):
    pass


class _IndexEvaluator:
    """Evaluate tile-index arithmetic on concrete row / column coordinates.

    ``fragment_epilogue`` walks the same grammar for one tile's logical
    indices; this walker evaluates the whole problem and must match the
    kernel's integer arithmetic exactly (block sizes only as reshape extents,
    every intermediate inside its traced dtype), so it stays its own."""

    def __init__(
        self, block_ids: tuple[int, int], coords: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        self.env = CompileEnvironment.current()
        self.block_ids = block_ids
        self.coords = coords
        self.cache: dict[Node, object] = {}

    def _block_axis(self, size: object) -> int:
        value = size.meta.get("val") if isinstance(size, Node) else size
        if not isinstance(value, (int, torch.SymInt)):
            raise _Unsupported
        block_id = self.env.get_block_id(value)
        if block_id not in self.block_ids:
            raise _Unsupported
        return self.block_ids.index(block_id)

    def _shape(self, shape: object) -> tuple[int, ...]:
        """A coordinate reshape's extents: block-size symbols of the two tile
        axes (the evaluated coordinate counts), ``1`` and ``-1`` only."""
        if not isinstance(shape, (list, tuple)):
            raise _Unsupported
        extents = []
        for entry in shape:
            if isinstance(entry, Node):
                extents.append(self.coords[self._block_axis(entry)].numel())
            elif type(entry) is int and entry in (1, -1):
                extents.append(entry)
            else:
                raise _Unsupported
        return tuple(extents)

    def _value(self, value: object) -> object:
        if isinstance(value, Node):
            if value not in self.cache:
                self.cache[value] = self._in_range(value, self._compute(value))
            return self.cache[value]
        if type(value) is int:
            return value
        raise _Unsupported

    @staticmethod
    def _in_range(node: Node, value: object) -> object:
        """``value`` when it fits the node's traced integer dtype: the
        evaluation runs in ``int64`` and matches the kernel's (usually
        ``int32``) index arithmetic only while nothing overflows."""
        traced = node.meta.get("val")
        if isinstance(value, torch.Tensor) and value.numel():
            if (
                not isinstance(traced, torch.Tensor)
                or traced.dtype not in _INDEX_DTYPES
            ):
                raise _Unsupported
            limits = torch.iinfo(traced.dtype)
            if int(value.min()) < limits.min or int(value.max()) > limits.max:
                raise _Unsupported
        return value

    def _compute(self, node: Node) -> object:
        if node.op != "call_function":
            raise _Unsupported
        if node.target is tile_index:
            if len(node.args) != 1 or node.kwargs:
                raise _Unsupported
            return self.coords[self._block_axis(node.args[0])]
        if node.target in _SHAPE_OPS:
            if len(node.args) != 2 or node.kwargs:
                raise _Unsupported
            source = self._value(node.args[0])
            if not isinstance(source, torch.Tensor):
                raise _Unsupported
            try:
                return source.reshape(self._shape(node.args[1]))
            except RuntimeError as error:  # the coordinates do not fill it
                raise _Unsupported from error
        args = tuple(self._value(arg) for arg in node.args)
        if not any(isinstance(arg, torch.Tensor) for arg in args):
            # Scalar arithmetic on symbols (a block size in an index
            # expression) depends on the config the proof does not see.
            raise _Unsupported
        if node.target is torch.ops.aten.div.Tensor_mode:
            if node.kwargs != {"rounding_mode": "floor"}:
                raise _Unsupported
            if len(args) != 2:
                raise _Unsupported
            return self._apply(
                lambda lhs, rhs: torch.div(lhs, rhs, rounding_mode="floor"), args
            )
        op = _INDEX_OPS.get(node.target)
        if (
            op is None
            or len(args) != 2
            or set(node.kwargs) - {"alpha"}
            or node.kwargs.get("alpha", 1) != 1
        ):
            raise _Unsupported
        return self._apply(op, args)

    @staticmethod
    def _apply(op: Callable[..., object], args: tuple[object, ...]) -> object:
        """``op(*args)``; operands that do not broadcast and integer division
        by zero (both a RuntimeError in torch) refuse the store."""
        try:
            return op(*args)
        except RuntimeError as error:
            raise _Unsupported from error

    def evaluate(self, node: Node) -> torch.Tensor:
        value = self._value(node)
        if not isinstance(value, torch.Tensor) or value.dtype.is_floating_point:
            raise _Unsupported
        return value


def prove_radix_store_layout(
    store: Node, *, block_ids: tuple[int, int], extents: tuple[int, int]
) -> RadixStoreLayout | None:
    """The mixed-radix layout of ``store``'s destination, or ``None``.

    ``store`` writes a rank-two ``[M, N]`` tile (``block_ids`` and their
    static ``extents``); its subscript may be any tile-index arithmetic whose
    address, ``sum(index_d * stride_d)``, separates into injective row and
    column layouts that stay inside the destination.
    """
    if len(store.args) < 3 or (len(store.args) > 3 and store.args[3] is not None):
        return None
    tensor_node, subscript = store.args[0], store.args[1]
    tensor = tensor_node.meta.get("val") if isinstance(tensor_node, Node) else None
    m, n = extents
    if (
        not isinstance(tensor, torch.Tensor)
        or not isinstance(subscript, (list, tuple))
        or len(subscript) != tensor.ndim
        or not all(isinstance(index, Node) for index in subscript)
        or any(type(size) is not int for size in tensor.shape)
        or any(type(stride) is not int for stride in tensor.stride())
        or type(tensor.storage_offset()) is not int
        or tensor.storage_offset() != 0
        # an empty problem stores nothing to lay out
        or m <= 0
        or n <= 0
        or m * n > _MAX_PROOF_ELEMENTS
    ):
        return None
    indices = cast("list[Node]", list(subscript))
    sizes_and_strides = (tuple(tensor.shape), tuple(tensor.stride()))

    def addresses(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        evaluator = _IndexEvaluator(block_ids, (rows, cols))
        shape = (rows.numel(), cols.numel())
        values = [evaluator.evaluate(index).to(torch.int64) for index in indices]
        try:
            stacked = torch.stack(
                [torch.broadcast_to(value, shape) for value in values], dim=-1
            )
        except RuntimeError as error:  # an index that is not [rows, cols]
            raise _Unsupported from error
        sizes, strides = (
            torch.tensor(values, dtype=torch.int64, device="cpu")
            for values in sizes_and_strides
        )
        if bool((stacked < 0).any()) or bool((stacked >= sizes).any()):
            raise _Unsupported
        return (stacked * strides).sum(dim=-1)

    # Real CPU tensors: no fake mode, and no Helion factory padding mode.
    with unset_fake_temporarily(), _disable_current_modes():
        try:
            return _prove(addresses, m, n)
        except _Unsupported:
            return None


def _prove(
    addresses: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], m: int, n: int
) -> RadixStoreLayout | None:
    """Separate the evaluated addresses into row and column layouts."""
    all_rows = torch.arange(m, dtype=torch.int64, device="cpu")
    all_cols = torch.arange(n, dtype=torch.int64, device="cpu")
    if int(addresses(all_rows[:1], all_cols[:1]).item()) != 0:
        return None
    row_map = addresses(all_rows, all_cols[:1])[:, 0]
    col_map = addresses(all_rows[:1], all_cols)[0, :]
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // n)
    for start in range(0, m, rows_per_chunk):
        chunk = all_rows[start : start + rows_per_chunk]
        expected = row_map[chunk][:, None] + col_map[None, :]
        if not torch.equal(addresses(chunk, all_cols), expected):
            return None
    rows = _mixed_radix(row_map)
    cols = _mixed_radix(col_map)
    if rows is None or cols is None or not _injective((*rows, *cols)):
        return None
    return RadixStoreLayout(rows, cols)


_PROOF_KEY = "helion_radix_store_proof"
_ACCEPTED_KEY = "helion_radix_store_layout"


def radix_store_layout(
    store: Node, *, block_ids: tuple[int, int], extents: tuple[int, int]
) -> RadixStoreLayout | None:
    """``prove_radix_store_layout`` memoized on the store node (the proof
    alone; ``accept_radix_store_layout`` records the store analysis' verdict)."""
    key = (block_ids, extents)
    cached = store.meta.get(_PROOF_KEY)
    if cached is not None and cached[0] == key:
        return cached[1]
    layout = prove_radix_store_layout(store, block_ids=block_ids, extents=extents)
    store.meta[_PROOF_KEY] = (key, layout)
    return layout


def accept_radix_store_layout(store: Node, layout: RadixStoreLayout) -> None:
    """Record that the MMA store analysis admitted ``store`` through ``layout``."""
    store.meta[_ACCEPTED_KEY] = layout


def proven_radix_store_layout(store: Node) -> RadixStoreLayout | None:
    """The layout the MMA store analysis admitted ``store`` through, if any."""
    return store.meta.get(_ACCEPTED_KEY)
