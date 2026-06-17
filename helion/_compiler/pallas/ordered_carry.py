"""Ordered carry for jagged row tiles on the Pallas emit_pipeline path.

A ``hl.tile(s, e)`` with runtime bounds rounds its rows out to a sublane-aligned
window and masks the extra.  Neighbouring groups then share one boundary,
which a small VMEM ``carry`` stitches back together.  Only valid when each
output row comes from its own input row.

Vocabulary (S = backend.sublane_tiling(dtype): the row granularity TPU can
dynamically slice):
  row       one logical row of the stored tensor.
  window    the S-aligned range a tile rounds its rows out to.
  boundary  the S rows two adjacent groups share, moved by the carry.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helion._compiler.inductor_lowering import CodegenState


@dataclasses.dataclass(frozen=True)
class CarryBoundaryTile:
    """What the block-spec, mask, and store code share about one carried tile."""

    block_id: int
    begin_var: str  # host var for the tile's runtime begin (e.g. ``s``)
    end_var: str  # host var for the tile's runtime end (e.g. ``e``)
    sublane: int  # S = backend.sublane_tiling(dtype)
    carry_scratch_name: str | None = None


def is_dynamic_bound_tile(state: CodegenState, block_id: int) -> bool:
    """True for a jagged tile: its begin and end are runtime tensor values.

    A static ``hl.tile(0, K)`` keeps a constant ``end_expr``; a ``hl.tile(s, e)``
    with runtime bounds has both ``begin_expr`` and ``end_expr`` set to None.
    """
    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return False
    info = loops[-1].block_id_to_info.get(block_id)
    if info is None:
        return False
    return info.begin_expr is None and info.end_expr is None


def is_row_map_axis(state: CodegenState, block_id: int) -> bool:
    """Whether each output row comes only from its own input row (a map axis).

    True only when the row is written straight back (row i to output row i),
    never summed, scattered, or shifted; the over-read and carry are only
    correct in that case.
    """
    # TODO(implement): the map-axis check above; for now reject every jagged tile.
    return False


def needs_ordered_carry(state: CodegenState, block_id: int) -> bool:
    """Whether this row tile needs the sublane carry.

    Carry stitches the boundary tile two groups share, so it only applies to a
    straight map axis (is_row_map_axis) feeding a 2-D store with a tiled column,
    which is the block the carry folds and saves.  An untiled ``:`` column cannot
    carry and needs none, a higher-rank store shares no boundary, and a reduction or
    scatter is not a map.
    """
    from helion._compiler.pallas.plan_tiling import TilePattern
    from helion.language.memory_ops import store

    if not is_row_map_axis(state, block_id):
        return False
    for ginfo in state.codegen.codegen_graphs:
        for node in ginfo.graph.nodes:
            if node.target is not store:
                continue
            patterns = node.meta.get("indexing_patterns")
            if not patterns or len(patterns) != 2:
                continue
            row_pat, col_pat = patterns
            if getattr(row_pat, "block_id", None) != block_id:
                continue
            if isinstance(row_pat, TilePattern) and (
                getattr(col_pat, "block_id", None) is not None
            ):
                return True
    return False


def begin_end_vars(state: CodegenState, block_id: int) -> tuple[str, str] | None:
    """Return the (begin_var, end_var) host names for a dynamic row tile."""
    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return None
    info = loops[-1].block_id_to_info.get(block_id)
    if info is None or info.begin_var_name is None:
        return None
    end_var = info.end_var_name
    if end_var is None:
        return None
    return info.begin_var_name, end_var


def emit_carry_store(
    state: CodegenState,
    tensor: object,
    subscript: list[object] | tuple[object, ...],
    name: str,
    idx_str: str,
    value: object,
) -> bool:
    """Store to a jagged row tile with the boundary carry.

    Not implemented yet.  Returns False for a normal store; for a jagged-row
    store it raises, because writing it directly would overwrite the previous
    group's data in the boundary they share.
    """
    fn = state.device_function
    patterns = state.fx_node.meta.get("indexing_patterns") if state.fx_node else None
    if not patterns:
        return False
    for pat in patterns:
        bid = getattr(pat, "block_id", None)
        if bid is not None and bid in fn.carry_tiles:
            raise NotImplementedError(
                "Pallas ordered carry store for a jagged row tile is not "
                "implemented yet."
            )
    return False
