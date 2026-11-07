"""Seeds for ``cute_dot_ring_mma``: a GEMV's dot through a cp.async ring on
``mma.sync`` fragments.

``hl.dot(lhs[m, k], rhs[k, 1])`` with 16-bit operands takes the scalar
fallback, whose bytes in flight from DRAM all sit in registers.  A
128 MB-per-call GEMV on a power-capped B200 then reads 52-54 us where
helion-triton's cp.async ring + ``tl.dot`` reads 49 and the memory floor is
45-47 (gather_gemv (8, 8192, 2)).  ``cute/ring_mma_dot.py`` streams the row
tile through a shared-memory ring and multiplies on tensor cores when the
rows are a rolled lane loop (one thread on the dot's M block) and the K
threads are whole warps; this heuristic seeds the geometries that read
45-48 us in that kernel's hand-edit sweep (64 rows x 256 K with 3 stages on
4 warps, 32 x 128 with 4 stages, 64 x 128 with 8 stages, 128 x 256 with the
warps on the row atoms), since random configs rarely land on a legal ring
layout at once.  The seeds only give the search a head start.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from .....runtime.config import Config
from ....autotuner_heuristics.registry import AutotunerHeuristic
from ...ring_mma_dot import RING_MMA_KEY
from ...ring_mma_dot import RING_SPLIT_KEY
from ...ring_mma_dot import RING_STAGES_KEY
from ...ring_mma_dot import ring_mma_geometry_reason
from ...ring_mma_dot import ring_mma_smem_budget
from ..search_surface import _cute_dot_ring_searchable
from ..search_surface import cute_fragment_mma_fact
from .cute import _seq_config_list

if TYPE_CHECKING:
    from .....autotuner.config_spec import MatmulFact
    from ....compile_environment import CompileEnvironment
    from ....device_ir import DeviceIR

# (rows, K chunk, ring stages, threads, warp split); the K vector width is
# the chunk's elements per thread (at least 2: the lowering must form packets).
_SEED_SHAPES: tuple[tuple[int, int, int, int, str], ...] = (
    (64, 256, 3, 128, "k"),
    (32, 128, 4, 64, "k"),
    (64, 128, 8, 64, "k"),
    (128, 256, 3, 128, "m"),
    (64, 256, 3, 64, "m"),
    (32, 256, 4, 128, "k"),
)


class CuteGemvRingMmaHeuristic(AutotunerHeuristic):
    """Rolled rows, whole-warp K threads, the dot through the cp.async ring on MMA fragments."""

    name = "cute_gemv_ring_mma"
    priority = 44
    backend = "cute"

    @classmethod
    def _fact(cls, env: CompileEnvironment) -> MatmulFact | None:
        spec = env.config_spec
        if not _cute_dot_ring_searchable(spec):
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
        seeds: list[Config] = []
        seen: set[tuple[int, int, int, int, str]] = set()
        for rows, chunk, stages, threads, split in _SEED_SHAPES:
            m_block = max(rows, m_spec.min_size, m_spec.autotuner_min)
            k_extent = k_spec.size_hint
            k_block = max(min(chunk, k_extent), k_spec.min_size, k_spec.autotuner_min)
            width = k_block // threads
            if (
                m_block > m_spec.max_size
                or k_block > k_spec.max_size
                or k_block & (k_block - 1)
                or width < 2
                or ring_mma_geometry_reason(
                    rows=m_block,
                    chunk=k_block,
                    threads=threads,
                    stages=stages,
                    split=split,
                    elem_bytes=fact.lhs_dtype.itemsize,
                    k_width=width,
                    smem_budget=ring_mma_smem_budget(spec.device),
                )
                is not None
            ):
                continue
            key = (m_block, k_block, stages, threads, split)
            if key in seen:
                continue
            seen.add(key)
            values: dict[str, Any] = {
                "block_sizes": _seq_config_list(
                    spec.block_sizes, {m_block_id: m_block, k_block_id: k_block}
                ),
                "num_threads": _seq_config_list(
                    spec.num_threads, {m_block_id: 1, k_block_id: threads}
                ),
                "cute_vector_widths": _seq_config_list(
                    spec.cute_vector_widths, {k_block_id: min(width, 8)}
                ),
                RING_MMA_KEY: True,
                RING_STAGES_KEY: stages,
                RING_SPLIT_KEY: split,
            }
            capability = spec.target_device_capability
            if spec.supports_config_key("cute_pdl") and (
                capability is not None and capability >= (9, 0)
            ):
                values["cute_pdl"] = True
            seeds.append(Config.from_dict(deepcopy(values)))
        return seeds


__all__ = ["CuteGemvRingMmaHeuristic"]
