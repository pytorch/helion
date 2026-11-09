"""Mixed-radix destination views for tcgen05 tile stores.

Inductor fuses view and permute ops into the store of a native matmul: a
head split writes ``acc[y, x]`` to ``out[y // 512, x // 64, y % 512, x % 64]``
of a ``[16, 16, 512, 64]`` destination.  Every element's address is then
``rows(y) + cols(x)``, each a CuTe layout over one tile coordinate
(``(512, 16):(64, 524288)`` and ``(64, 16):(1, 32768)`` here), so a TMA store
through one hierarchical ``[M, N]`` tensor addresses the destination.

A batched matmul's store adds the batch coordinate: an attention head merge
writes ``acc[z, y, x]`` (``z = b * 16 + h`` of a ``[B * 16, S, 64]`` bmm) to
``out[z // 16, y, (z % 16) * 64 + x]`` of a ``[B, S, 1024]`` destination, and
the address is ``batch(z) + rows(y) + cols(x)`` (``(16, B):(64, S * 1024)``
for ``z``), the TMA store's third, L, mode.

The proof interprets the store's index arithmetic on concrete coordinates; it
never matches an expression pattern.  It evaluates every coordinate of the
problem once (in chunks), so it is exact for the static shapes it admits.
Block-size symbols are admitted only as the extents of a coordinate reshape
(their value differs per config; the proof does not depend on the config).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx.node import Node
from torch.utils._python_dispatch import _disable_current_modes

from ...language import tile_index
from ..compile_environment import CompileEnvironment
from .tcgen05_constants import TCGEN05_RADIX_STORE_BOX_N

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

# Elements of the ([B,] M, N) problem the proof evaluates, and per chunk.
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
    """``(extent, stride)`` modes, innermost first, of each tile coordinate:
    the rows and columns of the ``[M, N]`` tile, and a batched matmul's batch
    coordinate (empty for a rank-two one)."""

    rows: tuple[tuple[int, int], ...]
    cols: tuple[tuple[int, int], ...]
    batch: tuple[tuple[int, int], ...] = ()

    @staticmethod
    def _mode_expr(modes: tuple[tuple[int, int], ...], part: int) -> str:
        values = [str(mode[part]) for mode in modes]
        return values[0] if len(values) == 1 else f"({', '.join(values)})"

    def cute_layout_args(self) -> tuple[str, str]:
        """``(shape, stride)`` expressions of the hierarchical ``[M, N]``
        layout, ``[M, N, L]`` with the batch coordinate's L mode."""
        coordinates = (self.rows, self.cols, *((self.batch,) if self.batch else ()))
        shape = ", ".join(self._mode_expr(modes, 0) for modes in coordinates)
        stride = ", ".join(self._mode_expr(modes, 1) for modes in coordinates)
        return f"({shape})", f"({stride})"

    @property
    def column_major(self) -> bool:
        """Whether the rows, not the columns, hold the contiguous run (a
        batched head merge that writes the tile transposed)."""
        return self.rows[0][1] == 1 and self.cols[0][1] != 1

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

    def one_cta_tile(self, element_bytes: int) -> tuple[int, int] | None:
        """The widest one-CTA ``bm x bn`` tile (``bm`` 128 or 64 rows, ``bn``
        256, 128, 64 or 32 columns) whose CTA tiles partition the layout and
        whose epilogue subtile stores ``(bm, 32)`` boxes inside its runs: the
        tile a batched store's default takes (``_narrow_batched_default``),
        or ``None``."""
        for rows in (128, 64):
            for cols in (256, 128, 64, 32):
                if self.tiles(rows, cols) and self.tma_store_supported(
                    rows_per_store=rows,
                    cols_per_store=TCGEN05_RADIX_STORE_BOX_N,
                    element_bytes=element_bytes,
                ):
                    return rows, cols
        return None

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
        innermost dimension (the columns', or the rows' for the column-major
        D tile) and 16-byte outer strides."""
        modes = (*self.rows, *self.cols, *self.batch)
        return (
            len(modes) <= _TMA_MAX_RANK
            and (self.rows if self.column_major else self.cols)[0][1] == 1
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
        self, block_ids: tuple[int, ...], coords: tuple[torch.Tensor, ...]
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
        """A coordinate reshape's extents: block-size symbols of the tile
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
        traced = node.meta.get("val")
        if isinstance(traced, torch.Tensor) and traced.dtype in _INDEX_DTYPES:
            limits = torch.iinfo(traced.dtype)
            if any(
                type(arg) is int and not limits.min <= arg <= limits.max for arg in args
            ):
                # Torch narrows a Python scalar to the tensor's dtype
                # (``y % (2**32 + 512)`` on an int32 ``y`` computes
                # ``y % 512``), which the int64 evaluation would not.
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
    store: Node, *, block_ids: tuple[int, ...], extents: tuple[int, ...]
) -> RadixStoreLayout | None:
    """The mixed-radix layout of ``store``'s destination, or ``None``.

    ``store`` writes an ``[M, N]`` tile, or a batched matmul's ``[1, M, N]``
    tile of a ``[B, M, N]`` problem (``block_ids`` and their static
    ``extents``, the batch first); its subscript may be any tile-index
    arithmetic whose address, ``sum(index_d * stride_d)``, separates into
    injective (batch,) row and column layouts that stay inside the
    destination.
    """
    if len(store.args) < 3 or (len(store.args) > 3 and store.args[3] is not None):
        return None
    tensor_node, subscript = store.args[0], store.args[1]
    tensor = tensor_node.meta.get("val") if isinstance(tensor_node, Node) else None
    if (
        len(block_ids) != len(extents)
        or len(extents) not in (2, 3)
        or not isinstance(tensor, torch.Tensor)
        or not isinstance(subscript, (list, tuple))
        or len(subscript) != tensor.ndim
        or not all(isinstance(index, Node) for index in subscript)
        or any(type(size) is not int for size in tensor.shape)
        or any(type(stride) is not int for stride in tensor.stride())
        or type(tensor.storage_offset()) is not int
        or tensor.storage_offset() != 0
        # an empty problem stores nothing to lay out
        or any(extent <= 0 for extent in extents)
        or math.prod(extents) > _MAX_PROOF_ELEMENTS
    ):
        return None
    indices = cast("list[Node]", list(subscript))
    sizes_and_strides = (tuple(tensor.shape), tuple(tensor.stride()))

    def addresses(*coords: torch.Tensor) -> torch.Tensor:
        evaluator = _IndexEvaluator(block_ids, coords)
        shape = tuple(coord.numel() for coord in coords)
        values = [evaluator.evaluate(index).to(torch.int64) for index in indices]
        try:
            stacked = torch.stack(
                [torch.broadcast_to(value, shape) for value in values], dim=-1
            )
        except RuntimeError as error:  # an index that is not [(batch,) rows, cols]
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
            return _prove(addresses, extents)
        except _Unsupported:
            return None


def _prove(
    addresses: Callable[..., torch.Tensor], extents: tuple[int, ...]
) -> RadixStoreLayout | None:
    """Separate the evaluated addresses into one layout per coordinate."""
    ranges = [
        torch.arange(extent, dtype=torch.int64, device="cpu") for extent in extents
    ]
    origin = [coords[:1] for coords in ranges]
    if int(addresses(*origin).reshape(-1)[0].item()) != 0:
        return None
    maps = []
    for axis, coords in enumerate(ranges):
        line = list(origin)
        line[axis] = coords
        maps.append(addresses(*line).reshape(-1))
    # Every coordinate, in chunks: the trailing axes whole while they fit a
    # chunk, the next one sliced, any outer ones a coordinate at a time.
    axis = len(extents) - 1
    inner = 1
    while axis > 0 and inner * extents[axis] <= _CHUNK_ELEMENTS:
        inner *= extents[axis]
        axis -= 1
    step = max(1, _CHUNK_ELEMENTS // inner)
    for prefix in itertools.product(*(range(extent) for extent in extents[:axis])):
        for start in range(0, extents[axis], step):
            chunk = [
                *(
                    ranges[outer][index : index + 1]
                    for outer, index in enumerate(prefix)
                ),
                ranges[axis][start : start + step],
                *ranges[axis + 1 :],
            ]
            expected = sum(
                (
                    maps[dim][coords].reshape(
                        [-1 if other == dim else 1 for other in range(len(chunk))]
                    )
                    for dim, coords in enumerate(chunk)
                ),
                torch.zeros((), dtype=torch.int64, device="cpu"),
            )
            if not torch.equal(
                addresses(*chunk), expected.expand(*(c.numel() for c in chunk))
            ):
                return None
    layouts = [_mixed_radix(values) for values in maps]
    if any(layout is None for layout in layouts):
        return None
    modes = cast("list[tuple[tuple[int, int], ...]]", layouts)
    if not _injective([mode for layout in modes for mode in layout]):
        return None
    *batch, rows, cols = modes
    return RadixStoreLayout(rows, cols, batch[0] if batch else ())


_PROOF_KEY = "helion_radix_store_proof"
_ACCEPTED_KEY = "helion_radix_store_layout"


def radix_store_layout(
    store: Node, *, block_ids: tuple[int, ...], extents: tuple[int, ...]
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
