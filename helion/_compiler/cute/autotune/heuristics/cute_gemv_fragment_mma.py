"""Seeds for ``cute_dot_fragment_mma``: a GEMV's dot on ``mma.sync`` fragments.

``hl.dot(lhs[m, k], rhs[k, 1])`` takes the scalar fallback (two converts, a
select and an FMA per element).  A 128 MB-per-call GEMV is memory-bound at
boost clocks but issue-bound once the sustained power cap pulls the SM
clock down (B200, gather_gemv (8, 8192, 2): memory-only body 46 us, the
fallback 53, Triton's tensor-core dot 49).  ``cute/fragment_mma_dot.py``
runs the contraction on tensor cores when the thread layout is eight row
threads by four K threads with 32-byte K packets; this heuristic seeds that
layout (the knob, the layout and the loads-first lane unroll that keeps
several packets in flight per thread), since random configs rarely land on
all of them at once.  The seeds only give the search a head start.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...fragment_mma_dot import FRAGMENT_MMA_K_THREADS
from ...fragment_mma_dot import FRAGMENT_MMA_KEY
from ...fragment_mma_dot import FRAGMENT_MMA_ROW_THREADS
from ..flash_surface import pdl_searched
from ..search_surface import _cute_fragment_mma_searchable
from ..search_surface import cute_fragment_mma_fact
from .cute import _seq_config_list

if TYPE_CHECKING:
    from .....autotuner.config_spec import MatmulFact
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR

# Rows per CTA (one or two per thread) x K chunk x lane unroll.
_SEED_SHAPES: tuple[tuple[int, int | None, int], ...] = (
    (8, 2048, 4),
    (8, None, 4),
    (16, 2048, 4),
    (8, 2048, 8),
)


class CuteGemvFragmentMmaHeuristic(AutotunerHeuristic):
    """Eight row threads x four K threads, 32-byte packets, the dot on MMA fragments."""

    name = "cute_gemv_fragment_mma"
    priority = 43
    backend = "cute"

    @classmethod
    def _fact(cls, env: CompileEnvironment) -> MatmulFact | None:
        spec = env.config_spec
        if not _cute_fragment_mma_searchable(spec):
            return None
        fact = cute_fragment_mma_fact(spec)
        if fact is None or fact.m_block_id is None or fact.k_block_id is None:
            return None
        for block_id in (fact.m_block_id, fact.k_block_id):
            if (
                block_id not in spec.num_threads.valid_block_ids()
                or block_id not in spec.block_sizes.valid_block_ids()
            ):
                return None
        if fact.k_block_id not in spec.cute_vector_widths.valid_block_ids():
            return None
        return fact

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return cls._fact(env) is not None

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
        fact = cls._fact(env)
        if fact is None:
            return []
        spec = env.config_spec
        m_block_id = cast("int", fact.m_block_id)
        k_block_id = cast("int", fact.k_block_id)
        m_spec = spec.block_sizes.block_id_lookup(m_block_id)
        k_spec = spec.block_sizes.block_id_lookup(k_block_id)
        # 32-byte packets of the 16-bit operands.
        packet = 32 // fact.lhs_dtype.itemsize
        # The read-only evict-first packets for the row operand (the memory
        # proofs keep ordinary loads where they do not apply).
        eviction: list[object] | None = None
        if spec.load_eviction_policies.length > 0:
            eviction = [""] * spec.load_eviction_policies.length
            for memory_fact in spec.memory_op_facts:
                if (
                    memory_fact.kind == "load"
                    and memory_fact.matmul_operand == "lhs"
                    and memory_fact.eviction_index is not None
                    and 0 <= memory_fact.eviction_index < len(eviction)
                ):
                    eviction[memory_fact.eviction_index] = "nc_l2_first"
        seeds: list[Config] = []
        seen: set[tuple[int, int, int]] = set()
        for rows, chunk, unroll in _SEED_SHAPES:
            m_block = max(rows, m_spec.min_size, m_spec.autotuner_min)
            if m_block > m_spec.max_size or m_block % FRAGMENT_MMA_ROW_THREADS:
                continue
            k_extent = k_spec.size_hint
            k_block = k_extent if chunk is None else min(chunk, k_extent)
            k_block = max(k_block, k_spec.min_size, k_spec.autotuner_min)
            if (
                k_block > k_spec.max_size
                or k_block & (k_block - 1)
                or k_block % (FRAGMENT_MMA_K_THREADS * packet)
            ):
                continue
            if (m_block, k_block, unroll) in seen:
                continue
            seen.add((m_block, k_block, unroll))
            values: dict[str, Any] = {
                "block_sizes": _seq_config_list(
                    spec.block_sizes, {m_block_id: m_block, k_block_id: k_block}
                ),
                "num_threads": _seq_config_list(
                    spec.num_threads,
                    {
                        m_block_id: FRAGMENT_MMA_ROW_THREADS,
                        k_block_id: FRAGMENT_MMA_K_THREADS,
                    },
                ),
                "cute_vector_widths": _seq_config_list(
                    spec.cute_vector_widths, {k_block_id: packet}
                ),
                FRAGMENT_MMA_KEY: True,
                "cute_lane_unroll": unroll,
            }
            if k_block_id in spec.cute_lane_layouts.valid_block_ids():
                values["cute_lane_layouts"] = _seq_config_list(
                    spec.cute_lane_layouts,
                    {
                        block_id: "strided"
                        for block_id in (m_block_id, k_block_id)
                        if block_id in spec.cute_lane_layouts.valid_block_ids()
                    },
                )
            if eviction is not None:
                values["load_eviction_policies"] = list(eviction)
            if spec.supports_config_key("cute_min_blocks_per_mp"):
                # ptxas keeps the hoisted packets in flight only with the
                # launch bound (without it the loads sink next to the MMAs).
                values["cute_min_blocks_per_mp"] = 1
            if pdl_searched(spec):
                values["cute_pdl"] = True
            seeds.append(Config.from_dict(deepcopy(values)))
        return seeds


__all__ = ["CuteGemvFragmentMmaHeuristic"]
