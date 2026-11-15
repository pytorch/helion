from __future__ import annotations

from typing import TYPE_CHECKING

from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...atomic_privatization import privatizable_atomic_targets
from ...loop_nesting import tile_loop_paths
from .cute import _seq_config_list

if TYPE_CHECKING:
    from ....autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR


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
        for path in tile_loop_paths(device_ir, device_ir.graphs):
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
