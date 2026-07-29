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
from typing import NamedTuple

if TYPE_CHECKING:
    import torch

    from helion._compiler.inductor_lowering import CodegenState


@dataclasses.dataclass(frozen=True)
class CarryBoundaryTile:
    """What the block-spec, mask, and store code share about one carried tile."""

    block_id: int
    begin_var: str  # host var for the tile's runtime begin (e.g. ``s``)
    end_var: str  # host var for the tile's runtime end (e.g. ``e``)
    sublane: int  # S = backend.sublane_tiling(dtype)
    carry_scratch_name: str | None = None


class CarryScratchKey(NamedTuple):
    """Identifies one carry scratch: a (row tile, output buffer) pair."""

    row_block_id: int
    output_name: str


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


class CarryStorePlan(NamedTuple):
    """Static layout a carried jagged store needs to emit fold + save."""

    row_block_id: int
    col_block_id: int
    sublane: int
    block_row: int
    block_col: int
    n_cols: int | torch.SymInt


def _carry_store_plan(
    state: CodegenState, tensor: torch.Tensor
) -> CarryStorePlan | None:
    """Carried layout of a jagged store, or None if it carries no row.  Raises
    on a carried store the carry cannot lower yet.
    """
    from helion._compiler.host_function import HostFunction

    fn = state.device_function
    patterns = state.fx_node.meta.get("indexing_patterns") if state.fx_node else None
    if not patterns:
        return None
    row_dim = row_block_id = None
    for d, pat in enumerate(patterns):
        bid = getattr(pat, "block_id", None)
        if bid is not None and bid in fn.carry_tiles:
            row_dim, row_block_id = d, bid
            break
    if row_block_id is None:
        return None

    # Fold guard counts groups with program_id(0), so the group is the only grid dim.
    # TODO(tcombes): thread the group's program id through CarryBoundaryTile for multi-grid.
    n_grid = sum(len(b) for b in HostFunction.current().device_ir.grid_block_ids)
    if n_grid != 1:
        raise NotImplementedError(
            "Pallas ordered carry assumes the group loop is the only grid "
            f"dimension (program_id(0)); got {n_grid} grid dims."
        )
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
    S = fn.carry_tiles[row_block_id].sublane
    block_row = fn.resolved_block_size(row_block_id)
    block_col = fn.resolved_block_size(col_block_id)
    n_cols = tensor.size(1)
    if not (isinstance(block_row, int) and isinstance(block_col, int)):
        raise NotImplementedError(
            "Pallas ordered carry needs static block sizes "
            f"(block_row={block_row!r}, block_col={block_col!r})."
        )
    if block_row % S != 0:
        raise NotImplementedError(
            f"Pallas ordered carry: block_row ({block_row}) must be a multiple of the "
            f"sublane tiling S ({S})."
        )
    # TODO(tcombes): block_row > L (tensor shorter than a block) crashes downstream.
    return CarryStorePlan(
        row_block_id=row_block_id,
        col_block_id=col_block_id,
        sublane=S,
        block_row=block_row,
        block_col=block_col,
        n_cols=n_cols,
    )


def emit_carry_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    name: str,
    idx_str: str,
    value: object,
) -> bool:
    """Store a jagged row tile, stitching the boundary two groups share.

    Each group rounds its rows out to an S-aligned window and zeros the rows it
    does not own.  A VMEM ``carry`` moves the shared boundary between groups:
    fold (group's first tile, ``begin`` unaligned) adds the predecessor's piece
    back; save (group's last tile, ``end`` unaligned) hands this group's forward.
    Saving the folded value keeps it cumulative, so several tiny groups in one
    boundary still work.
    """
    from helion._compiler.ast_extension import statement_from_string
    from helion._compiler.compile_environment import CompileEnvironment

    plan = _carry_store_plan(state, tensor)
    if plan is None:
        return False
    row_block_id = plan.row_block_id
    col_block_id = plan.col_block_id
    S = plan.sublane
    block_row = plan.block_row
    block_col = plan.block_col
    n_cols = plan.n_cols
    fn = state.device_function
    rec = fn.carry_tiles[row_block_id]

    # One boundary per column tile, stacked on dim 0: only the sublane (first)
    # dim of a VMEM ref is runtime-sliceable on TPU, not the column.
    num_col_tiles = (n_cols + block_col - 1) // block_col
    # One scratch per (tile, output): a tile feeding several stores (out1, out2)
    # carries each independently, otherwise the second store clobbers the first.
    carry = fn.carry_scratch.get(CarryScratchKey(row_block_id, name))
    if carry is None:
        carry = fn.register_scratch(
            (num_col_tiles * S, block_col),
            tensor.dtype,
            name_hint="carry",
            scratch_type="vmem",
        )
        fn.carry_scratch[CarryScratchKey(row_block_id, name)] = carry

    row_off = state.codegen.offset_var(row_block_id)
    col_off = state.codegen.offset_var(col_block_id)
    begin, end = rec.begin_var, rec.end_var
    a_start = f"({begin} - {begin} % {S})"
    a_end = f"(({end} + {S} - 1) // {S} * {S})"
    # Fold past the first group (program_id 0 has no predecessor) when begin is
    # unaligned; save the last row block when end is unaligned.
    # TODO(tcombes): assumes contiguous group offsets; a gap folds a stale carry.
    fold_guard = (
        f"(({row_off} == {a_start}) & ({begin} % {S} != 0) & (pl.program_id(0) != 0))"
    )
    save_guard = f"(({row_off} + {block_row} >= {a_end}) & ({end} % {S} != 0))"
    col_sub = f"pl.multiple_of(({col_off}) // {block_col} * {S}, {S})"
    carry_slice = f"{carry}[pl.ds({col_sub}, {S}), :]"
    # Group's last row block: S-aligned, within [0, block_row - S].
    last_off = f"pl.multiple_of({a_end} - {S} - ({row_off}), {S})"
    out_last = f"{name}[pl.ds({last_off}, {S}), :]"

    # Pad the [S, block_col] carry to a full [block_row, block_col] tile (carry on
    # top, zeros below): Pallas rejects a partial write to the double-buffered out.
    promoted = f"jnp.pad({carry_slice}, ((0, {block_row - S}), (0, 0)))"
    # Zero the over-read rows so an f(0) != 0 op (e.g. bias add) can't corrupt the
    # carried boundary; astype before reshape (Mosaic won't reshape a bool vector).
    dtype_str = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    # A carry tile is always a jagged tile, which _setup_mask always masks.
    row_mask = state.codegen.mask_var(row_block_id)
    assert row_mask is not None, "carry tile must be a masked jagged tile"
    owned = f"{row_mask}.astype({dtype_str})[:, None]"
    state.codegen.add_statement(
        statement_from_string(
            f"{name}[{idx_str}] = (({{value}}) * {owned}) "
            f"+ jnp.where({fold_guard}, {promoted}, 0)",
            value=value,  # pyrefly: ignore[bad-argument-type]
        )
    )
    state.codegen.add_statement(
        statement_from_string(
            f"{carry_slice} = jnp.where({save_guard}, {out_last}, {carry_slice})"
        )
    )
    return True
