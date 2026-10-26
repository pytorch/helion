"""Ordered carry for jagged row tiles under emit_pipeline and fori_loop.

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
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    import torch

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import GraphInfo
    from helion._compiler.inductor_lowering import CodegenState
    from helion._compiler.pallas.backend import PallasBackend
    from helion.runtime.config import Config


def aligned_window_sublane(env: CompileEnvironment) -> int | None:
    """Strictest float sublane across all host tensors, including unused ones."""
    from helion._compiler.host_function import HostFunction

    backend = cast("PallasBackend", env.backend)
    return max(
        (
            backend.sublane_tiling(t.dtype)
            for t in HostFunction.current().tensor_to_origin
            if t.is_floating_point()
        ),
        default=None,
    )


class AccessLayout(NamedTuple):
    """One memory access: the host tensor and the block id tiling each dim."""

    tensor: torch.Tensor
    block_ids: tuple[int | None, ...]


def store_aliases_a_load(
    tensor: torch.Tensor,
    graphs: Sequence[GraphInfo],
) -> bool:
    """Whether ``tensor`` may share its host allocation with any loaded tensor.

    The carry stores each aligned row window in full, so storing in place
    overwrites rows of the next group before it loads them.
    """
    import torch

    from helion._compiler.pallas.memory_access import tensors_share_origin_or_storage
    from helion.language import memory_ops

    return any(
        node.op == "call_function"
        and node.target is memory_ops.load
        and isinstance(node.args[0], torch.fx.Node)
        and isinstance((val := node.args[0].meta.get("val")), torch.Tensor)
        and tensors_share_origin_or_storage(val, tensor)
        for info in graphs
        for node in info.graph.nodes
    )


def check_carry_lowering(
    env: CompileEnvironment,
    config: Config,
    row_block_id: int,
    store: AccessLayout,
    graphs: Sequence[GraphInfo],
    store_node: torch.fx.Node,
) -> None:
    """Raise unless the carry can lower ``store`` on jagged ``row_block_id``.

    Kernel-shape reasons no config can fix (grid, store rank, aliasing,
    conditionals) raise ``BackendUnsupported``.  Reasons another config may fix
    (block sizes) raise ``InvalidConfig`` so the autotuner keeps searching
    instead of reporting a backend failure.  ``store_node`` is the store in
    ``graphs``.
    """
    from helion import exc

    reason = _unsupported_reason(env, row_block_id, store, graphs, store_node)
    if reason is not None:
        raise exc.BackendUnsupported("pallas", f"ordered carry: {reason}")
    reason = _invalid_config_reason(env, config, row_block_id, store)
    if reason is not None:
        raise exc.InvalidConfig(f"Pallas ordered carry {reason}.")


def _unsupported_reason(
    env: CompileEnvironment,
    row_block_id: int,
    store: AccessLayout,
    graphs: Sequence[GraphInfo],
    store_node: torch.fx.Node,
) -> str | None:
    """Why no config can lower ``store`` with the carry, or None."""
    import torch

    from helion._compiler.host_function import HostFunction

    # Fold guard counts groups with program_id(0), so the group is the only grid dim.
    # TODO(tcombes): thread the group's program id through CarryBoundaryTile for multi-grid.
    n_grid = sum(len(b) for b in HostFunction.current().device_ir.grid_block_ids)
    if n_grid != 1:
        return (
            "assumes the group loop is the only grid dimension "
            f"(program_id(0)); got {n_grid} grid dims"
        )
    if store.tensor.ndim != 2 or store.block_ids[0] != row_block_id:
        return f"supports only leading-row 2D store (got {store.block_ids})"
    if store.block_ids[1] is None:
        return "the column dim of a jagged store must be tiled"
    if store_node.args[3] is not None:
        return "extra_mask on a carried store is not supported"
    if store_aliases_a_load(store.tensor, graphs):
        return "store target aliases a loaded tensor"
    if (
        not isinstance(
            col_size := env.block_sizes[store.block_ids[1]].size,
            (int, torch.SymInt),
        )
        or not env.known_equal(col_size, store.tensor.shape[1])
        or may_skip_nonempty_group(row_block_id, graphs, store_node)
    ):
        return "an `if` may skip the row loop or its store for a non-empty group"
    return None


def capture_map(
    graphs: Sequence[GraphInfo],
    parents: Mapping[int, tuple[torch.fx.Node, int]],
) -> dict[torch.fx.Node, torch.fx.Node]:
    """Map each subgraph placeholder to the node captured at its call site."""
    import torch

    return {
        ph: captured
        for info in graphs
        if (entry := parents.get(info.graph_id)) is not None
        and isinstance(captures := entry[0].args[entry[1]], (list, tuple))
        for ph, captured in zip(
            info.graph.find_nodes(op="placeholder"), captures, strict=True
        )
        if isinstance(captured, torch.fx.Node)
    }


def resolve_node(
    value: object,
    outer: Mapping[torch.fx.Node, torch.fx.Node],
) -> object:
    """Follow ``_new_var`` copies and subgraph placeholders to the root node."""
    import torch

    from helion.language import _tracing_ops

    while isinstance(value, torch.fx.Node):
        if value.target is _tracing_ops._new_var:
            value = value.args[0]
        elif value in outer:
            value = outer[value]
        else:
            break
    return value


def may_skip_nonempty_group(
    row_block_id: int,
    graphs: Sequence[GraphInfo],
    store: torch.fx.Node | None = None,
) -> bool:
    """Crude check that no ``if`` skips a non-empty group's row loop or ``store``.

    At an unaligned group boundary, each group adds the shared boundary rows
    saved in the carry buffer by whichever group stored last. If a conditional
    like ``if e - s > 8:`` wraps ``for st in hl.tile(s, e):`` or
    ``out[st, dt] = ...`` and skips a 5-row group, the next group adds a stale
    boundary from an earlier group into its first rows. Only empty-group checks
    (``!= 0`` or ``> 0``) around the loop and inner ``if/else`` branches that
    both store are allowed.
    """
    import torch

    from helion._compiler.device_ir import ForLoopGraphInfo
    from helion._compiler.device_ir import control_flow_parent_entries
    from helion.language import _tracing_ops
    from helion.language import memory_ops

    aten = torch.ops.aten
    parents = control_flow_parent_entries(graphs)
    gid_of = {info.graph: info.graph_id for info in graphs}
    outer = capture_map(graphs, parents)

    def has_store(gid: object) -> bool:
        return isinstance(gid, int) and any(
            n.op == "call_function" and n.target is memory_ops.store
            for n in graphs[gid].graph.nodes
        )

    row_gids = {
        info.graph_id
        for info in graphs
        if isinstance(info, ForLoopGraphInfo) and row_block_id in info.block_ids
    }
    # Walk from `store` up to the row loop: only loops and `if/else` blocks
    # that store in both branches may sit between them.
    if store is not None:
        gid = gid_of[store.graph]
        while gid not in row_gids and (entry := parents.get(gid)) is not None:
            call, _ = entry
            # `_if` args: (test, true_graph_id, false_graph_id, true_args, false_args).
            if call.target is not _tracing_ops._for_loop and not (
                call.target is _tracing_ops._if
                and has_store(call.args[1])
                and has_store(call.args[2])
            ):
                return True
            gid = gid_of[call.graph]

    # Walk from each row loop up to the root graph: every enclosing `if` must
    # only skip empty groups (`!= 0` or `> 0`).
    for gid in row_gids:
        call, _ = parents[gid]
        gid = gid_of[call.graph]
        while (entry := parents.get(gid)) is not None:
            call, slot = entry
            test = resolve_node(call.args[0], outer) if call.args else None
            if not (
                call.target is _tracing_ops._if
                and slot == 3
                and isinstance(test, torch.fx.Node)
                and test.target in (aten.ne.Scalar, aten.gt.Scalar)
                and test.args[1] == 0
            ):
                return True
            gid = gid_of[call.graph]
    return False


def _invalid_config_reason(
    env: CompileEnvironment, config: Config, row_block_id: int, store: AccessLayout
) -> str | None:
    """Why ``config`` cannot lower ``store`` with the carry, or None."""
    import torch

    from helion._compiler.pallas.dma import is_tpu_dma_aligned_shape

    blocks: dict[int, int] = {}
    for bid in store.block_ids:
        if bid is None or bid in blocks:
            continue
        block = env.block_sizes[bid].from_config(config)
        if type(block) is not int or block <= 0:
            return f"needs static block sizes (block {bid} = {block!r})"
        blocks[bid] = block
    sublane = aligned_window_sublane(env)
    assert sublane is not None
    if blocks[row_block_id] % sublane != 0:
        return f"block_row ({blocks[row_block_id]}) must be a multiple of S ({sublane})"
    shape = tuple(blocks[bid] for bid in store.block_ids if bid is not None)
    full_shape = tuple(
        env.size_hint(dim)
        if isinstance(size := env.block_sizes[bid].size, (int, torch.SymInt))
        and env.known_equal(size, dim)
        else None
        for bid, dim in zip(store.block_ids, store.tensor.shape, strict=True)
        if bid is not None
    )
    if not is_tpu_dma_aligned_shape(shape, store.tensor.dtype, full_shape):
        return f"store block shape {shape} is not TPU DMA aligned"
    return None


@dataclasses.dataclass(frozen=True)
class CarryBoundaryTile:
    """What the block-spec, mask, and store code share about one carried tile."""

    block_id: int
    begin_var: str
    end_var: str
    sublane: int


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


def needs_ordered_carry(state: CodegenState, block_id: int) -> bool:
    """Whether this row tile needs the sublane carry.

    Carry stitches the boundary tile two groups share, so it only applies to a
    straight map axis (never summed, scattered, or shifted) feeding a 2-D store
    with a tiled column, which is the block the carry folds and saves.
    """
    from helion._compiler.device_ir import ReductionLoopGraphInfo
    from helion._compiler.pallas.plan_tiling import TilePattern
    from helion.language.memory_ops import store

    has_tiled_2d_store = False
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
                if (
                    node.target is store
                    and dim == 0
                    and len(patterns) == 2
                    and getattr(patterns[1], "block_id", None) is not None
                ):
                    has_tiled_2d_store = True
    return has_tiled_2d_store


def emit_carry_store(
    state: CodegenState,
    tensor: torch.Tensor,
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

    fn = state.device_function
    assert state.fx_node is not None
    patterns = state.fx_node.meta.get("indexing_patterns")
    if not patterns:
        return False
    block_ids = tuple(
        bid if isinstance(bid := getattr(pat, "block_id", None), int) else None
        for pat in patterns
    )
    row_block_id = next((bid for bid in block_ids if bid in fn.carry_tiles), None)
    if row_block_id is None:
        return False
    env = CompileEnvironment.current()
    check_carry_lowering(
        env,
        fn.config,
        row_block_id,
        AccessLayout(tensor, block_ids),
        state.codegen.codegen_graphs,
        state.fx_node,
    )
    col_block_id = block_ids[1]
    assert col_block_id is not None
    block_row = fn.resolved_block_size(row_block_id)
    block_col = fn.resolved_block_size(col_block_id)
    assert isinstance(block_row, int) and isinstance(block_col, int)
    rec = fn.carry_tiles[row_block_id]
    S = rec.sublane
    n_cols = tensor.size(1)

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
    dtype_str = env.backend.dtype_str(tensor.dtype)
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
