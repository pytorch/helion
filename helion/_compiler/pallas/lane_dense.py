"""Lane-dense layouts of the inputs of a sequential-roots Pallas program.

XLA lays out a TPU array in the dim order whose tiled footprint is smallest
(``tpu_default_layout``), so an array whose minor dim is much narrower than
the 128 lanes, e.g. a ``[5120, 6]`` projection weight or a ``[48, 5120, 6]``
stack of them, is laid out with its dims permuted.  A Pallas call takes its
operands row major, so XLA copies such an operand into row major ahead of
every call (and an operand the call writes in place back after it), and the
kernel's copies of it are padded to 128 lanes.  Passing the operand in its
physical dim order instead (a ``jnp.transpose`` in the launcher, which XLA
lowers to a bitcast) skips those copies: the kernel indexes the ref in
physical order and transposes values to the logical order at each load and
back at each store.

An integer index can then land on a minor dim of the physical order, e.g.
layer ``i`` of that stack is row ``i`` of each of the 6 ``[48, 5120]``
planes.  Mosaic indexes and copies a row of a 32-bit ref at any index, but a
row of a packed (sub-32-bit) one only at a multiple of its sublane tile, so a
row of a bfloat16 ref is read through the ref's 32-bit words: those of rows
``i // 2 * 2`` and the next, of which the read keeps the half that holds row
``i`` (``row_packing``, ``packed_row_load``, ``unpack_row``).
"""

from __future__ import annotations

import ast
import itertools
import math
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import torch

from ...language.matmul_ops import dot
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..inductor_lowering import CodegenState

_T = TypeVar("_T")

# The minor dim of an XLA TPU tile.
_LANES = 128


def _row_tile(rows: int, itemsize: int) -> int:
    """The second-minor tile size XLA picks for a TPU array with ``rows``
    second-minor elements of ``itemsize`` bytes: a small dim gets the
    smallest power of two that holds it, at least a 32-bit word of packed
    rows; a larger one the large tile that pads it least (the larger on a
    tie)."""
    tile = max(4 // itemsize, 1 << (rows - 1).bit_length())
    if tile <= 8:
        return tile
    large = (8, 32) if itemsize == 1 else (8,)
    return min(large, key=lambda size: (-(-rows // size) * size, -size))


def tpu_default_layout(shape: Sequence[int], dtype: torch.dtype) -> tuple[int, ...]:
    """The dim order, major to minor, of XLA's default layout of a TPU array
    of ``shape`` and ``dtype``: the permutation of the dims whose tiled
    footprint (the leading dims times the second-minor dim padded to its
    ``_row_tile`` times the minor dim padded to 128 lanes) is smallest, on a
    tie the one with the fewest inversions (the closest to row major), then
    the first in lexicographic order."""
    identity = tuple(range(len(shape)))
    if len(shape) < 2 or dtype.itemsize not in (1, 2, 4) or 0 in shape:
        return identity

    def cost(perm: tuple[int, ...]) -> tuple[int, int]:
        *major, rows, cols = (shape[dim] for dim in perm)
        row_tile = _row_tile(rows, dtype.itemsize)
        footprint = (
            math.prod(major)
            * (-(-rows // row_tile) * row_tile)
            * (-(-cols // _LANES) * _LANES)
        )
        inversions = sum(a > b for a, b in itertools.combinations(perm, 2))
        return footprint, inversions

    # ``permutations`` yields lexicographic order, and ``min`` keeps the
    # first of equal costs.
    return min(itertools.permutations(identity), key=cost)


def minor_swap(ndim: int) -> tuple[int, ...]:
    """The permutation that swaps the last two of ``ndim`` dims."""
    return (*range(ndim - 2), ndim - 1, ndim - 2)


def lane_dense_perm(shape: Sequence[int], dtype: torch.dtype) -> tuple[int, ...] | None:
    """The dim order to pass an input of ``shape`` and ``dtype`` lane dense
    in, or None: XLA's default layout of it, so the permuted array is a
    bitcast of it and never a copy, when that is not row major and its minor
    dim is narrower than the 128 lanes, so the dots that take it transposed
    (slower than plain ones) stay narrow."""
    if len(shape) < 2 or shape[-1] >= _LANES:
        return None
    perm = tpu_default_layout(shape, dtype)
    return None if perm == tuple(range(len(shape))) else perm


def row_packing(dtype: torch.dtype, rows: int) -> int | None:
    """How many rows of ``dtype`` a 32-bit word holds, when a load can read
    one of ``rows`` second-minor rows at an integer index (``packed_row``):
    1 for a 32-bit dtype, whose rows Mosaic indexes and copies at any index;
    2 for bfloat16 with an even ``rows``, whose rows it reads through 32-bit
    words (``packed_row_load``); None for any other."""
    if dtype.itemsize == 4:
        return 1
    if dtype == torch.bfloat16 and rows % 2 == 0:
        return 2
    return None


def packed_row(dtype: torch.dtype, perm: Sequence[int], kept: Sequence[bool]) -> bool:
    """Whether a load over the dims ``kept`` (per tensor dim) of a ``dtype``
    tensor laid out in dim order ``perm`` reads one row of a packed dtype:
    an integer index on its second-minor dim and none on its minor one."""
    *_, rows, cols = perm
    return dtype.itemsize < 4 and not kept[rows] and kept[cols]


def row_word(row: str, packing: int) -> str:
    """The index of the 32-bit word row that holds row ``row`` of a dtype
    ``packing`` rows to a word."""
    if packing == 1:
        return row
    if row.isdigit():
        return str(int(row) // packing)
    return f"({row}) // {packing}"


def packed_row_load(
    ref: str, parts: Sequence[str], dtype: torch.dtype, rows: int, lanes: int
) -> ast.AST:
    """``ref[parts]``, a load of the row at integer index ``parts[-2]`` of a
    ``dtype`` ref (``packed_row``) of ``rows`` rows of ``lanes`` lanes, read
    through the ref's 32-bit words.

    Mosaic reads a word row at a runtime index only from a ref of whole
    128-lane rows; a narrower ref is read whole and the row selected.
    """
    if lanes < _LANES:
        block = f"{ref}[{', '.join([*parts[:-2], ':', parts[-1]])}]"
        return expr_from_string(
            "jnp.sum(jnp.where("
            f"lax.broadcasted_iota(jnp.int32, ({rows}, 1), 0) == {parts[-2]}, "
            f"{block}.astype(jnp.float32), 0.0), axis=-2).astype(jnp.bfloat16)"
        )
    words = [*parts[:-2], row_word(parts[-2], 2), parts[-1]]
    value = expr_from_string(f"{ref}.bitcast(jnp.uint32)[{', '.join(words)}]")
    return unpack_row(value, parts[-2], dtype)


def unpack_row(words: ast.AST, row: str, dtype: torch.dtype) -> ast.AST:
    """Row ``row`` of a bfloat16 tensor out of ``words``, the 32-bit words
    of row ``row // 2`` of its ref bitcast to ``uint32``: a word holds an
    even row in its low half and the next in its high half, and a bfloat16
    is the high half of a float32."""
    assert dtype == torch.bfloat16
    shift = f"lax.convert_element_type(({row}) % 2 * 16, jnp.uint32)"
    return expr_from_string(
        f"lax.bitcast_convert_type(({{words}} >> {shift}) << 16, jnp.float32)"
        ".astype(jnp.bfloat16)",
        words=words,
    )


def physical_order(items: Sequence[_T], perm: Sequence[int]) -> list[_T]:
    """Per-dim ``items`` of a tensor laid out in dim order ``perm``, in
    physical order."""
    return [items[dim] for dim in perm]


def transpose_axes(
    perm: Sequence[int], kept: Sequence[bool], *, to_physical: bool
) -> tuple[int, ...] | None:
    """Axes of the transpose that takes a value over the dims ``kept`` (per
    tensor dim) of a tensor laid out in dim order ``perm`` from logical to
    physical order (``to_physical``) or back; None when that is no
    transpose."""
    logical = [dim for dim in range(len(perm)) if kept[dim]]
    physical = [dim for dim in perm if kept[dim]]
    source, target = (logical, physical) if to_physical else (physical, logical)
    axes = tuple(source.index(dim) for dim in target)
    return None if axes == tuple(range(len(axes))) else axes


def lowered_transpose(perm: Sequence[int], kept: Sequence[bool]) -> bool:
    """Whether a value over the dims ``kept`` of a tensor laid out in dim
    order ``perm`` takes a transpose Mosaic lowers between the two orders:
    none, or a swap of the last two dims."""
    axes = transpose_axes(perm, kept, to_physical=False)
    return axes is None or axes == minor_swap(len(axes))


def transpose_expr(value: ast.AST, axes: tuple[int, ...]) -> ast.AST:
    """``value`` transposed by ``axes``."""
    if axes == minor_swap(len(axes)):
        return expr_from_string("jnp.swapaxes({value}, -1, -2)", value=value)
    return expr_from_string(f"jnp.transpose({{value}}, {axes!r})", value=value)


def physical_store_value(
    state: CodegenState,
    value: ast.AST,
    perm: Sequence[int],
    kept: Sequence[bool],
) -> ast.AST:
    """``value``, the value store ``state.fx_node`` stores over the dims
    ``kept`` of a tensor laid out in dim order ``perm``, in that order.

    A stored transpose of a value that is in that order already (a
    lane-dense slab stored as a transposed view) stores that value instead
    of transposing it back.
    """
    axes = transpose_axes(perm, kept, to_physical=True)
    if axes is None:
        return value
    assert state.fx_node is not None
    node = state.fx_node.args[2]
    if (
        isinstance(node, torch.fx.Node)
        and node.target is torch.ops.aten.permute.default
        and state.env.get(node) is value
    ):
        dims = [int(dim) % len(axes) for dim in cast("list[int]", node.args[1])]
        source = state.env.get(cast("torch.fx.Node", node.args[0]))
        if [dims[axis] for axis in axes] == list(range(len(axes))) and isinstance(
            source, ast.AST
        ):
            return source
    return transpose_expr(value, axes)


def logical_load(
    state: CodegenState,
    value: ast.AST,
    perm: Sequence[int],
    kept: Sequence[bool],
) -> ast.AST:
    """``value``, read over the dims ``kept`` of a tensor laid out in dim
    order ``perm``, in logical order.

    A 2-D value that only dots take, as their right operand against a 2-D
    left one, stays in physical order: each of them contracts its minor dim
    instead (``pallas_lane_dense_rhs_loads``).  So does a value that only
    permutes back to that order take, directly or through dtype
    conversions (``pallas_physical_values``).
    """
    axes = transpose_axes(perm, kept, to_physical=False)
    if axes is None:
        return value
    node = state.fx_node
    assert node is not None
    if axes == (1, 0) and all(_dot_rhs(user, node) for user in node.users):
        state.device_function.pallas_lane_dense_rhs_loads.add(node)
        return value
    converts: list[torch.fx.Node] = []
    if _undone_by_users(node, axes, converts):
        state.device_function.pallas_physical_values.update([node, *converts])
        return value
    return transpose_expr(value, axes)


def _undone_by_users(
    node: torch.fx.Node, axes: tuple[int, ...], converts: list[torch.fx.Node]
) -> bool:
    """Whether every use of ``node``, a value transposed by ``axes``, is a
    permute that transposes it back, directly or through dtype conversions,
    which go to ``converts``."""
    if not node.users:
        return False
    for user in node.users:
        if user.target is torch.ops.aten.permute.default:
            dims = [int(dim) % len(axes) for dim in cast("list[int]", user.args[1])]
            if [axes[dim] for dim in dims] != list(range(len(axes))):
                return False
        elif user.target is torch.ops.prims.convert_element_type.default:
            if not _undone_by_users(user, axes, converts):
                return False
            converts.append(user)
        else:
            return False
    return True


def _dot_rhs(user: torch.fx.Node, node: torch.fx.Node) -> bool:
    """Whether ``user`` is a dot of a 2-D left operand by ``node`` that uses
    ``node`` only as its right operand."""
    if user.target is not dot or user.args[1] is not node:
        return False
    lhs, _, *rest = user.args
    return (
        isinstance(lhs, torch.fx.Node)
        and lhs is not node
        and node not in rest
        and lhs.meta["val"].ndim == 2
    )
