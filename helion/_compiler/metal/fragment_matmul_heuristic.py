"""The Metal autotuner's seeds for a few-row K-loop matmul on fragments.

A fragment run (``fragment_mma``) unrolls a K tile's 8-deep MMA steps, and
each lane computes its elements of a computed operand (a dequantized weight)
from loads over one run of K positions per tile, ``K block / 8`` long.  How
cheap that decode is turns on the run: a run of 16 positions holds a whole
number of words for every code width up to 8 bits (16 codes of 5 bits are 80
bits, five 16-bit words), so a lane's word offsets and shifts fold to
constants per step (``affine_fold``) and its loads are a few contiguous
words; shorter runs of 5- or 6-bit codes start at a different bit in each
lane, so every element shifts by a computed amount and reads two words.
The cost is not monotone in the K block: on an M4 Max, the 5-bit
202048 x 6656 matmul of a 30B model's output head at 8 rows ran 3.2 ms at a
32-wide K block, 3.6 ms at 64 and 2.1 ms at 128 (85% of DRAM bandwidth).
The search's neighbours step one block at a time from the default
``[16, 16, 16]``, and a full search settled on the 32-wide block; seeded
with these tiles it picks the 128-wide one.

Each seed takes the largest fragment K tile (``MAX_STEPS`` steps) the K
extent allows, a row block of 16 (a Metal matmul block's least), a column
block of 16 or 32, and, where SIMD groups can split the K loop
(``fragment_mma.split_refusal``), 8 or 16 of them, capped at the largest
power of two the K loop has tiles for: the few threadgroups a few-row
matmul's output columns make need the split to fill the GPU. Six of that
model's eight few-row matmul shapes tuned to a 128-wide K block at 16 or 32
columns and 8 or 16 SIMD groups at 8 rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch._inductor.runtime.runtime_utils import next_power_of_2

from ...runtime.config import Config
from ..autotuner_heuristics.registry import AutotunerHeuristic
from ._constants import SIMD_WIDTH
from .search_surface import METAL_MATMUL
from .search_surface import kernel_facts

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR

#: Most rows of x a seed is for: past them the matmul is not few-row, and
#: MPP's and the scalar lowerings' own seeds and defaults apply.
MAX_ROWS = 32
#: The row block: a Metal matmul block's least (``min_dot_size``).
ROW_BLOCK = 16
#: The column blocks the seeds try.
COLUMN_BLOCKS = (16, 32)
#: The SIMD groups a seed splits a K loop over, where it can split.
SPLITS = (8, 16)


class MetalFragmentMatmulHeuristic(AutotunerHeuristic):
    name = "metal_fragment_matmul"
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
        # Imported here: fragment_mma imports the compiler, which imports the
        # backends' heuristics as helion loads.
        from .fragment_mma import MAX_STEPS

        spec = env.config_spec
        facts = kernel_facts(spec)
        blocks = set(facts.fragment_blocks)
        # One K-loop matmul (or several over the same blocks, as a fused gate
        # and up projection), all of the kernel's matmuls.
        if len(blocks) != 1:
            return []
        if len(facts.fragment_blocks) != len(spec.matmul_facts):
            return []
        ((m, n, k),) = blocks
        tunable = set(spec.block_sizes.valid_block_ids())
        if not {m, n, k} <= tunable:
            return []
        specs = {b: spec.block_sizes.block_id_lookup(b) for b in (m, n, k)}
        fragments = {b: s._fragment(spec) for b, s in specs.items()}
        rows_extent = specs[m].size_hint
        if rows_extent > MAX_ROWS:
            return []
        k_tile = min(8 * MAX_STEPS, next_power_of_2(specs[k].size_hint))
        # A SIMD group past the K loop's tile count would idle, and still
        # pay the combine after the loop.
        k_tiles = -(-specs[k].size_hint // k_tile)
        cap = 1 << (k_tiles.bit_length() - 1)
        splits = (1,)
        if facts.fragment_splits:
            splits = tuple(sorted({min(s, cap) for s in SPLITS}))
        # K's block on one SIMD group's threads, M's and N's on one: a lane's
        # elements then lie at K positions affine in its thread, whose packed
        # words and shifts fold (``affine_fold``); left to the default thread
        # counts, the 5-bit head above ran 2.5 ms at the seed's tile, 2.1 ms
        # this way.
        threaded = set(spec.num_threads.valid_block_ids()) == {m, n, k}
        default = spec.default_config()
        seeds: list[Config] = []
        for columns in COLUMN_BLOCKS:
            for num_warps in splits:
                chosen = {m: ROW_BLOCK, n: columns, k: k_tile}
                if not all(
                    fragments[b].low <= size <= fragments[b].high
                    for b, size in chosen.items()
                ):
                    continue
                block_sizes = [
                    chosen.get(item.block_id, size)
                    for item, size in zip(
                        spec.block_sizes, default.block_sizes, strict=True
                    )
                ]
                config: dict[str, object] = {
                    **default.config,
                    "block_sizes": block_sizes,
                    "num_warps": num_warps,
                    METAL_MATMUL: "fragment",
                }
                if threaded:
                    config["num_threads"] = [
                        SIMD_WIDTH if item.block_id == k else 1
                        for item in spec.num_threads
                    ]
                seed = Config.from_dict(config)
                if seed not in seeds:
                    seeds.append(seed)
        return seeds
