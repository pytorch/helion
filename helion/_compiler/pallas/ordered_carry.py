"""Ordered carry for jagged row tiles on the Pallas emit_pipeline path.

A ``hl.tile(s, e)`` with runtime bounds rounds its rows out to a sublane-aligned
window and masks the extra.  Neighbouring groups then share one boundary row,
which a small VMEM ``carry`` stitches back together.  Only valid when each
output row comes from its own input row.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from helion._compiler.inductor_lowering import CodegenState


@dataclasses.dataclass(frozen=True)
class CarryRowTile:
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
    # TODO(tcombes): conservative.  Scans how the row is indexed and only accepts
    # a straight store; exercised only on the bmm and elementwise forms, so
    # revisit for completeness (aliasing, multiple stores, broadcast/expand).
    from helion._compiler.device_ir import ReductionLoopGraphInfo
    from helion._compiler.pallas.plan_tiling import TilePattern
    from helion.language.memory_ops import store

    has_straight_store = False
    for ginfo in state.codegen.codegen_graphs:
        if isinstance(ginfo, ReductionLoopGraphInfo) and block_id in ginfo.block_ids:
            return False  # the row is summed away
        for node in ginfo.graph.nodes:
            patterns = node.meta.get("indexing_patterns")
            if not patterns:
                continue
            for dim, pat in enumerate(patterns):
                if getattr(pat, "block_id", None) != block_id:
                    continue
                if not isinstance(pat, TilePattern):
                    return False  # the row is offset or scattered, not straight
                if node.target is store and dim == 0:
                    has_straight_store = True
    return has_straight_store


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
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    name: str,
    idx_str: str,
    value: object,
) -> bool:
    """Store a row tile to a jagged output, stitching the shared boundary row.

    Rounding each group's rows out to a sublane-aligned window makes neighbouring
    groups overlap on one row, and each group zeros the rows it does not own.  A
    one-row VMEM ``carry`` keeps that shared row across groups:

      * fold (first tile of a group, ``begin`` not S-aligned):
            ``out[:S] += carry``    add the previous group's piece back in
      * save (last tile of a group, ``end`` not S-aligned):
            ``carry = out[-S:]``    hand this group's piece forward

    Saving the folded value keeps it cumulative, so more than two tiny groups in
    one row still work.
    """
    from helion._compiler.ast_extension import statement_from_string
    from helion._compiler.host_function import HostFunction

    fn = state.device_function
    patterns = state.fx_node.meta.get("indexing_patterns") if state.fx_node else None
    if not patterns:
        return False

    # Find the carried (jagged) row dim, if this store has one.
    row_dim = None
    row_block_id = None
    for d, pat in enumerate(patterns):
        bid = getattr(pat, "block_id", None)
        if bid is not None and bid in fn.carry_tiles:
            row_dim, row_block_id = d, bid
            break
    if row_block_id is None:
        return False

    # The fold guard below counts groups with pl.program_id(0), so the group must
    # be the only grid dim.
    # TODO(tcombes): thread the group's real program id through CarryRowTile so
    # multi-grid kernels work, instead of refusing them here.
    grid_block_ids = HostFunction.current().device_ir.grid_block_ids
    n_grid = sum(len(bids) for bids in grid_block_ids)
    if n_grid != 1:
        raise NotImplementedError(
            "Pallas ordered carry assumes the group loop is the only grid "
            f"dimension (program_id(0)); got {n_grid} grid dims."
        )

    # The carry only handles a 2D store with the jagged row leading.
    if row_dim != 0 or tensor.ndim != 2:
        raise NotImplementedError(
            "Pallas ordered carry supports only a leading-row 2D store "
            f"(got ndim={tensor.ndim}, jagged row at dim {row_dim})."
        )
    col_block_id = getattr(patterns[1], "block_id", None)
    if col_block_id is None:
        raise NotImplementedError(
            "Pallas ordered carry: the column dim of a jagged store must be tiled."
        )
    rec = fn.carry_tiles[row_block_id]
    S = rec.sublane
    block_row = fn.resolved_block_size(row_block_id)
    block_col = fn.resolved_block_size(col_block_id)
    n_cols = tensor.size(1)
    if not (
        isinstance(block_row, int)
        and isinstance(block_col, int)
        and isinstance(n_cols, int)
    ):
        raise NotImplementedError(
            "Pallas ordered carry needs static block/column sizes "
            f"(block_row={block_row!r}, block_col={block_col!r}, cols={n_cols!r})."
        )
    if block_row % S != 0:
        raise NotImplementedError(
            f"Pallas ordered carry: block_row ({block_row}) must be a multiple of the "
            f"sublane tiling S ({S})."
        )
    # TODO(tcombes): block_row larger than the whole tensor (L < block_row) is not
    # handled and crashes downstream with a shape mismatch; only L >= block_row
    # is exercised.

    # Only the sublane (first) dim of a VMEM ref can be sliced at runtime on TPU,
    # not the column (last) dim.  So give each column tile its own boundary row,
    # stacked on dim 0: carry[num_col_tiles*S, block_col].
    num_col_tiles = (n_cols + block_col - 1) // block_col
    carry = fn.carry_scratch.get(row_block_id)
    if carry is None:
        carry = fn.register_scratch(
            (num_col_tiles * S, block_col),
            tensor.dtype,
            name_hint="carry",
            scratch_type="vmem",
        )
        fn.carry_scratch[row_block_id] = carry

    row_off = state.codegen.offset_var(row_block_id)
    col_off = state.codegen.offset_var(col_block_id)
    begin, end = rec.begin_var, rec.end_var
    a_start = f"({begin} - {begin} % {S})"
    a_end = f"(({end} + {S} - 1) // {S} * {S})"
    # The first group (program_id 0) has nothing before it, so it never folds:
    # the carry scratch is not written yet.  Later groups fold only after a
    # predecessor saved, so the scratch is always written first.
    # TODO(tcombes): assumes contiguous group offsets (no gaps); a gap would fold
    # a stale carry from a non-adjacent predecessor.
    fold_guard = (
        f"(({row_off} == {a_start}) & ({begin} % {S} != 0) & (pl.program_id(0) != 0))"
    )
    save_guard = f"(({row_off} + {block_row} >= {a_end}) & ({end} % {S} != 0))"
    col_sub = f"pl.multiple_of(({col_off}) // {block_col} * {S}, {S})"
    # carry is a normal (non-buffered) scratch, so a partial-row slice is fine.
    carry_slice = f"{carry}[pl.ds({col_sub}, {S}), :]"
    # The group's last row block sits at offset (a_end - S - row_off) inside the
    # last tile: S-aligned and within [0, block_row - S].
    last_off = f"pl.multiple_of({a_end} - {S} - ({row_off}), {S})"
    out_last = f"{name}[pl.ds({last_off}, {S}), :]"

    # Add the carry into the store value so the output gets one full-tile write;
    # Pallas rejects a partial write to the double-buffered output.  So pad the
    # [S, block_col] carry up to the full [block_row, block_col] tile (carry on
    # top, zeros below) and add the whole thing.
    n = block_row // S
    if n > 1:
        promoted = (
            f"jnp.concatenate([{carry_slice}] + "
            f"[jnp.zeros_like({carry_slice})] * {n - 1}, 0)"
        )
    else:
        promoted = carry_slice
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{idx_str}] = ({{value}}) + jnp.where({fold_guard}, {promoted}, 0)",
            value=value,  # pyrefly: ignore[bad-argument-type]
        )
    )
    # Save this group's last row block for the next group's fold.  When that
    # block is also the group's first (folded) one, saving it keeps the carry
    # cumulative across several tiny groups in one row.
    state.codegen.add_statement(
        statement_from_string(
            f"{carry_slice} = jnp.where({save_guard}, {out_last}, {carry_slice})"
        )
    )
    return True
