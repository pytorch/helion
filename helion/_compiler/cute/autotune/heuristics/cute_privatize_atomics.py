from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch

from .....language.atomic_ops import atomic_add
from .....language.memory_ops import load as language_load
from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ....compile_environment import _symint_sympy_expr
from ...atomic_privatization import privatizable_atomic_targets
from ...emitters.lane_packets import _CUTE_VECTOR_MAX_BYTES
from ...loop_nesting import tile_loop_paths
from ...reduction_lanes import WARP_THREADS
from .cute import _key_chunk
from .cute import _reversed_root_loop_orders
from .cute import _seq_config_list
from .cute import _stride_one_loads

if TYPE_CHECKING:
    from ....autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR
    from ...loop_nesting import TileLoopPath


class CutePrivatizeAtomicsHeuristic(AutotunerHeuristic):
    """Offer per-CTA shared-memory accumulation for small atomic-add targets.

    ``cute_privatize_atomics`` (``cute/atomic_privatization.py``) turns the
    adds of a CTA into shared-memory atomics and flushes each changed cell to
    global memory once per CTA, so a CTA's tile size sets how many updates
    one global atomic carries.  The seeds are the knob alone and the knob
    with a tile of ``SEED_TILE_ELEMENTS`` updates over ``SEED_THREADS``
    threads on every nest of tile loops (``tile_loop_paths``; sibling loops
    and grid roots each get their own): its innermost axis takes up to
    ``SEED_INNER_ELEMENTS`` elements and one thread per element up to the
    thread budget, every axis outward takes what is left of both, and each
    thread runs its reductions serially, so the tile's threads are all of
    the CTA's.

    Between those two, ``_jam_seed`` turns on ``cute_grid_lane_jam`` for one
    grid tile whose innermost axis is the stride-1 dimension of a tensor the
    atomic's value is reduced from: each thread's lanes are one 16-byte
    packet of that axis, which the jam loads as vectors and whose adds into
    one cell it sums before the shared-memory add.  Measured on B200 (cold L2, the
    better-benchmark inputs, whose buckets clamp 69% of the keys into one):
    T5's scatter (``[8, 8, 1024, 1024]`` bf16 gradients) takes 166-178 us at
    that seed against 274-307 us at the config a full search picked without
    it.
    """

    name = "cute_privatize_atomics"
    backend = "cute"
    priority = 93

    # Sized on T5's bias-gradient scatter (8M adds into 256 cells), where
    # 128-thread CTAs beat the same tile with its batch sum left on 8
    # threads per tile thread (1024-thread CTAs).
    SEED_TILE_ELEMENTS = 2048
    SEED_INNER_ELEMENTS = 256
    SEED_THREADS = 128
    # The jam seed's rolled reduction chunk: the smallest a reduction loop
    # takes, which keeps the batch sum a device loop for the jam.
    JAM_REDUCTION_CHUNK = 8

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        host = device_ir.host_function
        if host is not None:
            with host:
                if privatizable_atomic_targets(env, device_ir, allow_unbound=True):
                    env.config_spec.cute_privatize_atomics_enabled = True
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_privatize_atomics_enabled

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        if not cls.is_eligible(env, device_ir):
            return []
        spec = env.config_spec
        seeds = [Config(cute_privatize_atomics=True)]
        tiled = set(spec.block_sizes.valid_block_ids())
        blocks: dict[int, int] = {}
        threads: dict[int, int] = {}
        paths = tile_loop_paths(device_ir, device_ir.graphs)
        for path in paths:
            elements = cls.SEED_TILE_ELEMENTS
            budget = cls.SEED_THREADS
            axes = list(
                dict.fromkeys(
                    spec.block_sizes.block_id_lookup(block_id).block_id
                    for loop in path
                    for block_id in loop
                    if block_id in tiled
                )
            )
            # An axis an earlier nest sized (the grid tile of sibling inner
            # loops) keeps its size and threads, and this nest's own axes
            # share what it leaves of both, innermost axis first.
            for block_id in axes:
                if block_id in blocks:
                    elements = max(1, elements // blocks[block_id])
                    budget = max(1, budget // threads[block_id])
            new = [block_id for block_id in axes if block_id not in blocks]
            for position, block_id in enumerate(reversed(new)):
                fragment = spec.block_sizes.block_id_lookup(block_id)._fragment(spec)
                want = cls.SEED_INNER_ELEMENTS if position == 0 else elements
                size = max(fragment.low, min(fragment.high, want, elements))
                blocks[block_id] = 1 << (size.bit_length() - 1)
                threads[block_id] = min(budget, blocks[block_id])
                elements = max(1, elements // blocks[block_id])
                budget = max(1, budget // threads[block_id])
        if not blocks:
            return seeds
        jam = cls._jam_seed(env, device_ir, paths)
        if jam is not None:
            seeds.append(jam)
        values: dict[str, object] = {
            "block_sizes": _seq_config_list(spec.block_sizes, blocks),
            "cute_privatize_atomics": True,
        }
        threaded = set(spec.num_threads.valid_block_ids())
        counts = dict.fromkeys(threaded - tiled, 1)
        counts.update(
            (block_id, count)
            for block_id, count in threads.items()
            if block_id in threaded
        )
        if counts:
            values["num_threads"] = _seq_config_list(spec.num_threads, counts)
        seeds.append(Config.from_dict(values))
        return seeds

    @classmethod
    def _jam_seed(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
        paths: tuple[TileLoopPath, ...],
    ) -> Config | None:
        """The knob with ``cute_grid_lane_jam``: the one grid tile's innermost
        axis as each thread's lanes of one 16-byte packet of the narrowest
        tensor the atomic's value is reduced from along it
        (``_atomic_value_loads``; ``_key_chunk`` sizes the tile, unmasked for
        the jam), each axis outward, innermost first, a thread per element
        until a CTA has a warp and one element after that, and the one
        reduction rolled in ``JAM_REDUCTION_CHUNK`` chunks, the device loop
        the jam interleaves the lanes into.  None for another loop structure,
        a value reduced from no tensor along that axis, or a dynamic extent,
        whose masked tile the jam leaves rolled.  Every static shape gets the
        seed, so the knob it makes searchable is searchable at all of them
        (``autotune_multi`` needs one search space): a short reduction just
        leaves the lanes rolled."""
        spec = env.config_spec
        if (
            spec.matmul_facts
            or len(device_ir.grid_block_ids) != 1
            or len(paths) != 1
            or len(paths[0]) != 1
            or len(spec.reduction_loops) != 1
        ):
            return None
        grid = device_ir.grid_block_ids[0]
        inner = grid[-1]
        if not all(
            inner in sequence.valid_block_ids()
            for sequence in (spec.num_threads, spec.cute_vector_widths)
        ):
            return None
        host = device_ir.host_function
        if host is None:
            return None
        with host:
            targets = privatizable_atomic_targets(env, device_ir, allow_unbound=True)
        symbol = _symint_sympy_expr(env.block_sizes[inner].var)
        values = _atomic_value_loads(device_ir, targets)
        widths = [
            _CUTE_VECTOR_MAX_BYTES // tensor.dtype.itemsize
            for graph in device_ir.graphs
            for tensor, sizes, unit in _stride_one_loads(graph.graph, values)
            if sizes[unit] == symbol
        ]
        extent = env.block_sizes[inner].numel
        if not widths or not isinstance(extent, (int, sympy.Integer)):
            return None
        vec = max(widths)
        chunk = _key_chunk(int(extent), vec)
        blocks = dict.fromkeys(grid, 1)
        blocks[inner] = chunk
        threads = dict.fromkeys(spec.num_threads.valid_block_ids(), 1)
        threads[inner] = max(1, chunk // vec)
        # A short axis would leave a CTA under a warp (eight heads of bf16:
        # one thread): the axes outward give it threads of their own.
        wanted = WARP_THREADS // threads[inner]
        for block_id in reversed(grid[:-1]):
            outer = env.block_sizes[block_id].numel
            if (
                wanted <= 1
                or block_id not in threads
                or not isinstance(outer, (int, sympy.Integer))
            ):
                continue
            fragment = spec.block_sizes.block_id_lookup(block_id)._fragment(spec)
            size = max(fragment.low, min(fragment.high, wanted, int(outer)))
            blocks[block_id] = threads[block_id] = 1 << (size.bit_length() - 1)
            wanted = max(1, wanted // blocks[block_id])
        config: dict[str, object] = {
            "block_sizes": _seq_config_list(spec.block_sizes, blocks),
            "num_threads": _seq_config_list(spec.num_threads, threads),
            # One element per vector: the lanes stay a lane loop, which the
            # jam unrolls and loads in packets itself.
            "cute_vector_widths": _seq_config_list(spec.cute_vector_widths, {inner: 1}),
            "reduction_loops": [cls.JAM_REDUCTION_CHUNK],
            "cute_privatize_atomics": True,
            "cute_grid_lane_jam": True,
        }
        # Consecutive CTAs on adjacent rows (T5's scatter: 183-188 us with
        # the heads fastest, 166-176 us with the queries).
        orders = _reversed_root_loop_orders(env)
        if orders is not None:
            config["loop_orders"] = orders
        return Config.from_dict(config)


def _atomic_value_loads(
    device_ir: DeviceIR, targets: frozenset[int]
) -> set[torch.fx.Node]:
    """The loads the values of the atomic adds into ``targets`` (storages,
    ``privatizable_atomic_targets``) are computed from: those of the adds
    privatization keeps in shared memory, which the jam sums per cell, and
    not those of their indices (``buckets`` in ``hl.atomic_add(out,
    [buckets[tile]], total)``)."""
    loads: set[torch.fx.Node] = set()
    for info in device_ir.graphs:
        for node in info.graph.nodes:
            if not (
                node.op == "call_function"
                and node.target is atomic_add
                and isinstance(target := node.args[0], torch.fx.Node)
                and isinstance(tensor := target.meta.get("val"), torch.Tensor)
                and id(tensor.untyped_storage()) in targets
            ):
                continue
            todo = [node.args[2]]
            seen: set[torch.fx.Node] = set()
            while todo:
                arg = todo.pop()
                if not isinstance(arg, torch.fx.Node) or arg in seen:
                    continue
                seen.add(arg)
                if arg.target is language_load:
                    loads.add(arg)
                else:
                    todo.extend(arg.all_input_nodes)
    return loads
