"""Place one local warp alternative beside its ordinary multiroot carrier."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from .....language import _tracing_ops
from .....language import memory_ops
from .....runtime.config import Config
from ...loop_nesting import boundary_only_grid_blocks
from ...row_carry_stripes import stripe_tiles
from ...thread_budget import tile_loop_thread_count
from ...warp_row_dot_spec import BF16_PACKET_LANES
from ...warp_row_dot_spec import ROW_DOT_K
from ...warp_row_dot_spec import ROW_DOT_ROWS
from ...warp_row_dot_spec import row_dot_geometry
from ...warp_row_partition import KEY
from ...warp_row_partition import WARP
from ..config_keys import COMPACT_ROWS_KEY
from ..config_keys import ROW_DOT_KEY
from .cute_row_publication import compatible_warp_rows

if TYPE_CHECKING:
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR


def _singleton_grid_schedule(env: CompileEnvironment, carrier: Config) -> Config | None:
    """Keep the order among active grid axes; place known singleton axes last."""
    spec = env.config_spec
    if not spec.supports_config_key("loop_orders"):
        return None
    # Only a literal extent of one is unconditionally a single grid tile.
    # A hint of one does not establish this for a dynamic extent.
    singletons = {
        axis.block_id
        for axis in env.block_sizes
        if type(axis.size) is int and axis.size == 1
    }
    if not singletons:
        return None
    orders = cast("list[list[int]] | None", carrier.get("loop_orders"))
    if orders is None:
        orders = [item._fill_missing() for item in spec.loop_orders]
    # LoopOrderSpec indexes a tile nest by position. Only change root grids;
    # do not rewrite the order of a nested device loop.
    grid_ids = set(spec.grid_block_ids)
    updated = [
        sorted(order, key=lambda pos: item.block_ids[pos] in singletons)
        if grid_ids.issuperset(item.block_ids)
        else order
        for order, item in zip(orders, spec.loop_orders, strict=True)
    ]
    if updated == orders:
        return None
    return Config.from_dict(carrier.config | {"loop_orders": updated})


def _row_dot_launch_variant(env: CompileEnvironment, carrier: Config) -> Config | None:
    """Try a register allocation hint even if the grid is already ordered."""
    alternate = _singleton_grid_schedule(env, carrier)
    if env.config_spec.supports_config_key(
        "cute_min_blocks_per_mp"
    ) and not carrier.get("cute_min_blocks_per_mp"):
        # Preserve the ordinary carrier, and add at most one launch variant.
        # One resident block gives ptxas a separate register allocation target.
        options = (alternate if alternate is not None else carrier).config
        return Config.from_dict(options | {"cute_min_blocks_per_mp": 1})
    return alternate


def insert_warp_row_seed(
    env: CompileEnvironment, ir: DeviceIR, configs: list[Config]
) -> list[Config]:
    spec = env.config_spec
    facts = spec.cute_warp_row_reduction_roots
    if not facts:
        return configs
    # A wide pointwise sibling supplies x. Preserve complete contractions,
    # other roots' geometry and every option in a previously legal carrier.
    axes = spec.block_sizes.valid_block_ids()
    row_ids = {fact[0] for fact in facts.values()}
    peers = [
        axis
        for root, grid in zip(ir.root_ids, ir.grid_block_ids, strict=True)
        if root not in facts and len([n for n in grid if n in axes]) == 1
        for axis in grid
        if axis in axes
        and axis not in row_ids
        and spec.block_sizes.block_id_lookup(axis).max_size >= WARP
    ]
    if not peers:
        return configs
    # Prefer a peer with no sequential tile loops, so widening its grid does
    # not also enlarge a matrix contraction. Later equal-width peers retain
    # their original execution order.
    pure = {
        axis
        for root, grid in zip(ir.root_ids, ir.grid_block_ids, strict=True)
        if all(
            n.target is not _tracing_ops._for_loop for n in ir.graphs[root].graph.nodes
        )
        for axis in grid
    }
    peer = max(
        peers,
        key=lambda a: (a in pure, spec.block_sizes.block_id_lookup(a).max_size, a),
    )
    # A fragment row-carry owner is a regular carrier when siblings contain
    # rolled contractions; otherwise the first ordinary seed stays the carrier.
    base = next((c for c in configs if c.config.get("cute_fragment_row_carry")), None)
    if base is None:
        base = configs[0] if configs else spec.autotune_reference_config()
    default = spec.autotune_reference_config()
    blocks = list(
        cast("list[int]", base.config.get("block_sizes", default.config["block_sizes"]))
    )
    peer_spec = spec.block_sizes.block_id_lookup(peer)
    width = 1 << (min(256, peer_spec.max_size).bit_length() - 1)
    if not max(peer_spec.min_size, peer_spec.autotuner_min) <= width:
        return configs
    blocks[axes.index(peer)] = width
    changed = False
    for axis in row_ids:
        if axis not in axes:
            continue
        item = spec.block_sizes.block_id_lookup(axis)
        rows = 1 << (min(width // WARP, item.max_size).bit_length() - 1)
        if max(item.min_size, item.autotuner_min) <= rows:
            blocks[axes.index(axis)] = rows
            changed = True
    if not changed:
        return configs
    # Unspecified synthetic/fragment axes can consume the peer's threads
    # when the ordinary search flattens and repairs its candidate. Select a
    # single physical x axis explicitly, with one warp per output in this
    # root. Fragment emitters partition that same linear CTA themselves.
    thread_axes = spec.num_threads.valid_block_ids()
    if peer not in thread_axes or not row_ids <= set(thread_axes):
        return configs
    requested = dict.fromkeys(thread_axes, 1)
    requested[peer] = width
    for row in row_ids:
        if row not in axes:
            return configs
        requested[row] = blocks[axes.index(row)]
    opts = dict(base.config)
    opts["num_threads"] = [requested[axis] for axis in thread_axes]
    # The single-resident-CTA default is omitted by the flat search surface.
    if opts.get("num_sm_multiplier") == 1:
        opts.pop("num_sm_multiplier")
    # This transform only owns rolled reductions: a persistent default
    # is not a viable witness. Keep each other reduction's reference mode.
    reductions = spec.reduction_loops.valid_block_ids()
    chunks = list(cast("list[int | None]", default.config.get("reduction_loops", [])))
    if len(chunks) != len(reductions):
        return configs
    for _, kid, length, _count in facts.values():
        if kid not in reductions:
            return configs
        slot = reductions.index(kid)
        upper = min(length & -length, spec.max_reduction_loop or length, 1024)
        if upper < WARP:
            return configs
        chunks[slot] = 1 << (upper.bit_length() - 1)
    opts.update(reduction_loops=chunks)
    opts[KEY] = True
    # The wide fragment projection is an independent row tile; pack a full
    # warp of output rows there while retaining the companion projection's
    # original tile. Where no fragment projection exists, keep its carrier.
    adjacent = [
        i
        for i in spec.block_sizes
        if i.block_id in spec.cute_fragment_row_carry_row_axes
        and i.block_id not in row_ids | {peer}
        and max(i.min_size, i.autotuner_min) <= WARP <= i.max_size
    ]
    if adjacent:
        largest = max(adjacent, key=lambda item: (item.size_hint, item.block_id))
        blocks[axes.index(largest.block_id)] = WARP
    opts["block_sizes"] = blocks
    companions = [Config.from_dict(opts)]
    # A row-dot seed built in isolation can lose the warp partition used by
    # an earlier reduction in the same launch. Give the tuner a candidate
    # that preserves that partition and assigns one MMA stripe to each warp.
    # Scalar carry siblings keep their own loop; amortize their setup across
    # one contiguous register packet's worth of rows per warp.
    mma_rows = {
        row
        for pairs in spec.cute_warp_row_dot_roots.values()
        for row, k in pairs
        if k in axes and blocks[axes.index(k)] == ROW_DOT_K
    }
    if mma_rows and row_dot_geometry(spec, opts):
        warps = width // WARP
        updated = list(blocks)
        for item in adjacent:
            rows_per_warp = (
                ROW_DOT_ROWS if item.block_id in mma_rows else BF16_PACKET_LANES
            )
            capacity = min(warps * rows_per_warp, item.max_size)
            tile = 1 << (capacity.bit_length() - 1)
            if tile >= max(item.min_size, item.autotuner_min):
                updated[axes.index(item.block_id)] = tile
        dot = opts | {"block_sizes": updated, ROW_DOT_KEY: True}
        if row_dot_geometry(spec, dot):
            regular = Config.from_dict(dot)
            companions.append(regular)
            alternate = _row_dot_launch_variant(env, regular)
            if alternate is not None:
                companions.append(alternate)
            if stripe_tiles(spec, dot):
                compact = alternate if alternate is not None else regular
                companions.append(
                    Config.from_dict(compact.config | {COMPACT_ROWS_KEY: True})
                )
    return [*configs[:1], *companions, *configs[1:]]


def add_singleton_row_seed(
    env: CompileEnvironment, ir: DeviceIR, configs: list[Config]
) -> list[Config]:
    """Offer more independent row programs when another root supplies the warps."""
    spec = env.config_spec
    facts = spec.cute_warp_row_reduction_roots
    if len(facts) != 1 or ir.host_function is None:
        return configs
    ((root, (row, _kid, _extent, _count)),) = facts.items()
    grid = ir.grid_block_ids[ir.root_ids.index(root)]
    # Changing one root's tile must not repack another root or an axis
    # sharing its config slot. Other grid axes must be scalar.
    if (
        row not in spec.block_sizes.valid_block_ids()
        or row not in spec.num_threads.valid_block_ids()
        or spec.block_sizes.block_id_lookup(row).min_size > 1
        or spec.num_threads.block_id_lookup(row).block_ids != [row]
        or any(row in axes for axes in ir.grid_block_ids if axes is not grid)
        or any(
            type(env.block_sizes[axis].size) is not int
            or env.block_sizes[axis].size != 1
            for axis in grid
            if axis != row
        )
    ):
        return configs
    with ir.host_function:
        # An immutable row-by-K input justifies distributing the independent
        # work. The emitter verifies the scalar address and disjoint winner.
        matrix = any(
            n.target is memory_ops.load
            and isinstance(indices := n.args[1], (list, tuple))
            and len(indices) == 2
            and indices[1] == slice(None)
            and isinstance(ix := indices[0], torch.fx.Node)
            and ix.target is _tracing_ops._get_symnode
            and isinstance(sym := ix.meta.get("val"), torch.SymInt)
            and env.get_block_id(sym) == row
            and isinstance(src := n.args[0], torch.fx.Node)
            and src.target is _tracing_ops._host_tensor
            and isinstance(v := src.meta.get("val"), torch.Tensor)
            and v.ndim == 2
            for n in ir.graphs[root].graph.nodes
        )
        if not matrix:
            return configs
        inactive = boundary_only_grid_blocks(env, ir, ir.graphs)
        for seed in reversed(configs):
            if (
                seed.get(KEY) is not True
                or seed.get("pid_type") != "persistent_interleaved"
            ):
                continue
            bs = seed.get("block_sizes")
            nt = seed.get("num_threads")
            if not isinstance(bs, list) or not isinstance(nt, list):
                continue
            bx = spec.block_sizes.block_id_to_index(row)
            tx = spec.num_threads.block_id_to_index(row)
            if not 1 < nt[tx] < bs[bx] or not compatible_warp_rows(
                env, ir, seed, nt[tx]
            ):
                continue
            blocks, threads = list(bs), list(nt)
            blocks[bx] = threads[tx] = 1
            candidate = Config.from_dict(
                seed.config | {"block_sizes": blocks, "num_threads": threads}
            )
            total = tile_loop_thread_count(
                env, ir, ir.graphs, candidate, inactive_block_ids=inactive
            )
            if WARP <= total <= 1024 and total % WARP == 0 and candidate not in configs:
                return [*configs, candidate]
    return configs
