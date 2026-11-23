"""The Metal autotuner's seeds for a large K-loop matmul on MPP.

MPP's default config is a small tile (``[16, 16, 16]`` or ``[32, 32, 16]``
at 4 SIMD groups), and the search's neighbours step one block or
``num_warps`` at a time from it.  On a fast kernel the search gets to the
large tiles anyway; on a slow one the initial population's budget
(``PopulationBasedSearch._initial_population_budget``) leaves a few
generations, which end near the default.  On the M4 Pro, five of the eight
GEMMs of the Metal benchmark (1024 to 12288 per side, bf16) tuned to a
32 x 64 tile with a 256-wide K block at 2 SIMD groups; a grouped GEMM of four
8192-square groups ran 1018 ms on the fragment tile its budget-cut search
kept and 720 ms on that MPP tile (torch eager 738 ms), and 719 ms on
32 x 128 x 256 at 4 SIMD groups.

Each seed is one of :data:`TILES` with the K block capped at the next power
of two of K, for a kernel whose matmuls (one, or several over the same
blocks) all lower on MPP's tiles without staging an operand (staged matmuls
have ``staged_matmul_heuristic``) and whose rows and columns are both at
least :data:`MIN_EXTENT` (few-row matmuls have ``fragment_matmul_heuristic``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch._inductor.runtime.runtime_utils import next_power_of_2

from ..autotuner_heuristics.registry import AutotunerHeuristic
from .block_seeds import block_seeds
from .search_surface import METAL_MATMUL
from .search_surface import kernel_facts

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR

#: ``(rows, columns, K, num_warps)`` of each seed, the benchmark's most
#: common tuned tile first.
TILES = ((32, 64, 256, 2), (32, 128, 256, 4))
#: Fewest rows and columns a seeded matmul has: 32 x 64 tiles then make at
#: least 32 threadgroups.
MIN_EXTENT = 256


class MetalMppMatmulHeuristic(AutotunerHeuristic):
    name = "metal_mpp_matmul"
    backend = "metal"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(cls.get_seed_configs(env, device_ir))

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
        spec = env.config_spec
        if not spec.matmul_facts or not kernel_facts(spec).mpp_unstaged:
            return []
        blocks = {
            (fact.m_block_id, fact.n_block_id, fact.k_block_id)
            for fact in spec.matmul_facts
        }
        if len(blocks) != 1:
            return []
        ((m, n, k),) = blocks
        tunable = set(spec.block_sizes.valid_block_ids())
        if m is None or n is None or k is None or not {m, n, k} <= tunable:
            return []
        specs = {b: spec.block_sizes.block_id_lookup(b) for b in (m, n, k)}
        fact = spec.matmul_facts[0]
        extents = {
            m: fact.static_m or specs[m].size_hint,
            n: fact.static_n or specs[n].size_hint,
            k: fact.static_k or specs[k].size_hint,
        }
        if min(extents[m], extents[n]) < MIN_EXTENT:
            return []
        return block_seeds(
            spec,
            (
                (
                    {m: rows, n: columns, k: min(k_block, next_power_of_2(extents[k]))},
                    {"num_warps": num_warps, METAL_MATMUL: "mpp"},
                )
                for rows, columns, k_block, num_warps in TILES
            ),
        )
