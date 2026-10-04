"""Raw 2-D views of a ring arena's slots (``pallas_stream_arena``).

An arena is a ``(n, sublanes, 128)`` VMEM ref, one native VMEM tile per
leading index.  ``ring_tile(arena, offset, (rows, cols))`` views the
``rows // sublanes * cols // 128`` tiles from ``offset`` as a ``(rows, cols)``
ref laid out row-major by tile, which is how a DMA or a vector store writes a
``(rows, cols)`` array, so tiles of different shapes can share the arena's
slots.  Mosaic has the op (``tpu.reinterpret_cast``) but no Pallas transform
reaches it, so this module adds one: a ``ReshapeTransform`` that Mosaic lowers
to the cast, and that interpret mode reads and writes as a plain reshape of the
tiles (any layout is fine there, as long as the copy into a slot and the reads
of it agree).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Any

from jax import tree_util
from jax._src.lib import tpu as tpu_ops
from jax._src.lib.mlir import ir
from jax._src.pallas.mosaic import lowering as mosaic_lowering
from jax._src.state import discharge as state_discharge
from jax._src.state import types as state_types
from jax.experimental import pallas as pl

if TYPE_CHECKING:
    import jax

# NOTE: this module is embedded verbatim into helion-free
# `to_code(allow_helion_deps=False)` output (via
# PallasBackend.embedded_helper_source), so its non-docstring code must not
# import anything from the `helion` package.


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TileView(state_types.ReshapeTransform):
    """View a run of native VMEM tiles as a ``shape`` ref."""


_plain_reshape_memref = mosaic_lowering._reshape_memref
_plain_transform_swap_array = state_discharge.transform_swap_array


def _reshape_memref(
    ref: ir.Value,
    reshaper: state_types.ReshapeTransform,
    ref_aval: Any,  # noqa: ANN401
    ref_block_shape: tuple[Any, ...],
) -> tuple[ir.Value, tuple[int, ...]]:
    if not isinstance(reshaper, TileView):
        return _plain_reshape_memref(ref, reshaper, ref_aval, ref_block_shape)
    rows, cols = reshaper.shape
    source = ir.MemRefType(ref.type)
    packing = 4 // ref_aval.dtype.itemsize
    tiling = f"({8 * packing},128)" + (f"({packing},1)" if packing > 1 else "")
    layout = ir.Attribute.parse(f"#tpu.tiled<{tiling},[{cols // 128},1]>")
    tiled = ir.MemRefType.get(
        (rows, cols),
        source.element_type,
        layout=layout,
        memory_space=source.memory_space,
    )
    plain = ir.MemRefType.get(
        (rows, cols), source.element_type, memory_space=source.memory_space
    )
    view = tpu_ops.reinterpret_cast(tiled, ref)
    return tpu_ops.erase_memref_layout(plain, view), (rows, cols)


def _transform_swap_array(
    x: jax.Array, transforms: tuple[object, ...], val: jax.Array
) -> tuple[jax.Array, jax.Array]:
    for position, transform in enumerate(transforms):
        if isinstance(transform, TileView):
            outer, inner = transforms[:position], transforms[position + 1 :]
            tiles = state_discharge.transform_array(x, outer)
            old, view = _transform_swap_array(
                tiles.reshape(transform.shape), inner, val
            )
            _, new_x = _plain_transform_swap_array(x, outer, view.reshape(tiles.shape))
            return old, new_x
    return _plain_transform_swap_array(x, transforms, val)


mosaic_lowering._reshape_memref = _reshape_memref
state_discharge.transform_swap_array = _transform_swap_array


def ring_tile(
    arena: state_types.AbstractRef | state_types.TransformedRef,
    offset: int | jax.Array,
    shape: tuple[int, int],
) -> state_types.TransformedRef:
    """The ``shape`` view of the arena's native tiles from ``offset``."""
    sublanes = arena.shape[-2]
    rows, cols = shape
    chunk = arena.at[pl.ds(offset, rows // sublanes * (cols // 128))]  # pyrefly: ignore[bad-index]
    return state_types.TransformedRef(
        chunk.ref, (*chunk.transforms, TileView((rows, cols)))
    )
