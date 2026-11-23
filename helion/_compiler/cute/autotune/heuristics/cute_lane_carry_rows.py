from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from .....language.memory_ops import load as language_load
from .....language.memory_ops import store as language_store
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...reduction_lanes import WARP_THREADS
from ...reduction_lanes import fits_row_packets
from ..search_surface import cute_lane_carry_row
from .cute import _cute_static_block_extent
from .cute import _seq_config_list
from .cute_nest_storage import host_tensor_accesses
from .cute_nest_storage import inputs
from .cute_nest_storage import proven_apart
from .cute_nest_storage import reduces

if TYPE_CHECKING:
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR

# The ``helion.language`` modules whose operations a seeded nest may hold:
# tracing plumbing, loads and stores, views, tile coordinates and the
# carries' initializers.  Anything else (an atomic, ``hl.reduce``, a scan, a
# barrier, inline assembly) is one of the things the lowering declines.
_PLAIN_LANGUAGE_MODULES = frozenset(
    {
        "helion.language._tracing_ops",
        "helion.language.memory_ops",
        "helion.language.view_ops",
        "helion.language.tile_ops",
        "helion.language.creation_ops",
    }
)


@dataclasses.dataclass(frozen=True)
class _CarryRowPlan:
    # The grid tile's block and the device loop's.
    grid_block_id: int
    loop_block_id: int
    reduction_block_id: int
    # The row's length and its ``(threads, vec)`` layouts, widest vector first.
    size: int
    layouts: tuple[tuple[int, int], ...]
    # The rows a thread may walk per trip of the device loop
    # (``lane_loops/tile_trips.py``), most first; empty where the tile's
    # reductions feed more than the carried values.
    trips: tuple[int, ...]


def _nest_facts(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    loop_block_id: int,
    row_block_id: int,
    row_size: int,
) -> bool | None:
    """What the device IR shows of the conditions under which the lanes of
    a vector row carry their values through the device loop in one nest
    (``lane_loops/split.py``, ``_lane_carry_nest``): ``None`` where the
    lowering declines, else whether the tile's own reductions feed nothing
    but the carried values (what running several rows per trip takes,
    ``lane_loops/tile_trips.py``).

    * The kernel is a grid tile around one device loop over
      ``loop_block_id`` and nothing else: no branch, no further loop.
    * Its operations are plain ones (``_PLAIN_LANGUAGE_MODULES``).
    * The loop carries values, none computed from a reduction along the
      row, and stores something that is.
    * Every tensor it loads is proven apart from every one it stores, and
      so are the tensors of two stores of the loop and of one of the loop
      and one after it, by the storage rule the lowering applies
      (``cute_nest_storage.proven_apart``).

    The lowering still proves each config (the affine store addresses, the
    scalars private to a lane); a nest these facts admit and it declines
    costs the search one compile per seed.
    """
    from ....device_ir import ForLoopGraphInfo
    from ....device_ir import RootGraphInfo

    roots = [info for info in device_ir.graphs if isinstance(info, RootGraphInfo)]
    loops = [info for info in device_ir.graphs if isinstance(info, ForLoopGraphInfo)]
    if (
        len(roots) != 1
        or len(loops) != 1
        or len(device_ir.graphs) != 2
        or list(loops[0].block_ids) != [loop_block_id]
    ):
        return None
    row = env.block_sizes[row_block_id].symbol()
    tile = env.block_sizes[loop_block_id].symbol()
    for info in device_ir.graphs:
        for node in info.graph.nodes:
            module = getattr(node.target, "__module__", "") or ""
            if (
                node.op == "call_function"
                and module.startswith("helion.language")
                and (
                    module not in _PLAIN_LANGUAGE_MODULES
                    or (
                        module == "helion.language.memory_ops"
                        and node.target not in (language_load, language_store)
                    )
                )
            ):
                return None
    accesses = host_tensor_accesses(device_ir)
    if accesses is None:
        return None
    loop_stores = accesses.stored(loops[0].graph_id)
    later_stores = accesses.stored(roots[0].graph_id)
    apart = proven_apart(env, device_ir, accesses.tensors)
    if apart is None or not all(
        frozenset(pair) in apart
        for pair in (
            *itertools.product(accesses.loaded, {*loop_stores, *later_stores}),
            *itertools.combinations(loop_stores, 2),
            *itertools.product(loop_stores, later_stores),
        )
    ):
        return None
    nodes = list(loops[0].graph.nodes)
    reductions = {node for node in nodes if reduces(node, row, row_size)}
    folds = {node for node in nodes if reduces(node, tile, None)}
    carried = [
        value
        for node in nodes
        if node.op == "output"
        for value in cast("list[object]", node.args[0])
        if isinstance(value, torch.fx.Node)
    ]
    stores = [node for node in nodes if node.target is language_store]
    if (
        not carried
        or any(inputs(value) & reductions for value in carried)
        or not any(inputs(node) & reductions for node in stores)
    ):
        return None
    return not any(inputs(node) & folds for node in stores)


class CuteLaneCarryRowHeuristic(AutotunerHeuristic):
    """Seed the multi-warp vector row of a persistent row reduction inside a
    device loop under a grid tile: a mix-order reduction, whose rows are
    reduced over their columns while per-column partials are carried over a
    split's rows (a LayerNorm backward).

    One thread owns V consecutive columns (one packet per operand), the row
    takes ``row_len / V`` threads in whole warps, and both tile axes run on
    one thread.  The lanes then carry their partials through the row loop in
    one nest (``lane_loops/split.py``, ``_lane_carry_nest``): each row's
    packets are loaded once, where the scalar lanes the search reaches
    otherwise walk the rows twice.  The layouts are V = 8, 4 and 2 where one
    packet holds V of the row's widest elements and the row takes at most
    ``MAX_THREADS``: 96 x 8, 192 x 4 and 384 x 2 for 768 columns, 256 x 8
    and 512 x 4 for 2048.  Which one wins depends on what the rows feed (a
    row whose sums are only stored ran fastest on the most threads, one that
    rewrites every element on the widest packets), so each is planted.  A
    row whose length is not a power of two runs unpadded
    (``cute_unpadded_rows``), as the lone row of
    ``CutePersistentUnpaddedRowsHeuristic`` does; a row no whole warps of
    vectors tile gets no seed.

    Behind the one-row seeds come the same layouts at several rows per trip
    of the device loop: its tile of ``U`` rows on the one thread, a lane
    loop of rows (``lane_loops/tile_trips.py``), with ``cute_lane_unroll``
    issuing the loads of a trip's rows ahead of the first row's reductions.
    A small grid leaves few warps per SM, and a row's loads then wait behind
    the previous row's barriers: four rows per trip ran a 512-CTA LayerNorm
    backward 15% faster than one.  ``U`` is 4 and 2 where it divides the
    loop's static extent.

    The seeds are planted where the device IR shows what the lowering
    proves (``_nest_facts``): where it declines (the partials read a row
    reduction, the kernel updates a tensor in place, an output may alias
    an input) there is none.
    """

    name = "cute_lane_carry_row"
    backend = "cute"
    priority = 103
    VECTORS = (8, 4, 2)
    TRIPS = (4, 2)
    # The most threads a planted row takes, 16 warps.  The layout at the
    # cap was already behind the wider packets where both were timed (2048
    # columns: 512 x 4 was 12% behind 256 x 8), so none beyond it is planted.
    MAX_THREADS = 16 * WARP_THREADS

    @classmethod
    def _plan(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> _CarryRowPlan | None:
        spec = env.config_spec
        blocks = cute_lane_carry_row(spec)
        if (
            blocks is None
            or len(blocks[0]) != 2
            or any(item.min_size > 1 for item in spec.block_sizes)
        ):
            return None
        (grid_block_id, loop_block_id), reduction_block_id = blocks
        size = spec.cute_static_reduction_extents[reduction_block_id]
        unpadded = size & (size - 1) != 0
        if unpadded and not spec.supports_config_key("cute_unpadded_rows"):
            return None
        itemsize = spec.cute_row_itemsizes.get(reduction_block_id)
        maximum = min(cls.MAX_THREADS, cast("int", spec.max_reduction_threads))
        layouts = tuple(
            (size // vec, vec)
            for vec in cls.VECTORS
            if size % vec == 0
            and (size // vec) % WARP_THREADS == 0
            and size // vec <= maximum
            and fits_row_packets(itemsize, vec, spec.target_device_capability)
        )
        if not layouts:
            return None
        folds_feed_carries = _nest_facts(
            env, device_ir, loop_block_id, reduction_block_id, size
        )
        if folds_feed_carries is None:
            return None
        rows = _cute_static_block_extent(env, loop_block_id)
        loop = spec.block_sizes.block_id_lookup(loop_block_id)
        trips = tuple(
            trip
            for trip in cls.TRIPS
            if folds_feed_carries
            and rows is not None
            and rows % trip == 0
            and loop.max_size >= trip
            and spec.supports_config_key("cute_lane_unroll")
        )
        return _CarryRowPlan(
            grid_block_id, loop_block_id, reduction_block_id, size, layouts, trips
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return cls._plan(env, device_ir) is not None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        plan = cls._plan(env, device_ir)
        if plan is None:
            return None
        spec = env.config_spec
        tiles = (plan.grid_block_id, plan.loop_block_id)
        seeds: list[Config] = []
        for trip in (1, *plan.trips):
            for threads, vec in plan.layouts:
                seed: dict[str, Any] = {
                    "block_sizes": _seq_config_list(
                        spec.block_sizes,
                        {plan.grid_block_id: 1, plan.loop_block_id: trip},
                    ),
                    "num_threads": _seq_config_list(
                        spec.num_threads,
                        {**dict.fromkeys(tiles, 1), plan.reduction_block_id: threads},
                    ),
                    "cute_vector_widths": _seq_config_list(
                        spec.cute_vector_widths, {plan.reduction_block_id: vec}
                    ),
                }
                if plan.size & (plan.size - 1):
                    seed["cute_unpadded_rows"] = True
                if spec.cute_sequence_reduction_blocks:
                    seed["cute_reduction_sequence"] = "scalar"
                if spec.cute_resident_reduction_blocks:
                    seed["cute_reduction_schedule"] = "scalar"
                if trip > 1:
                    seed["cute_lane_unroll"] = trip
                seeds.append(Config(**seed))
        return seeds
