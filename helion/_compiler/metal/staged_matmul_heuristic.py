"""The Metal autotuner's seeds for a K-loop matmul that stages an operand.

An MPP matmul whose operand is computed per K tile (a dequantized weight,
``MetalKernelFacts.staged_matmuls``) writes that tile into threadgroup memory
every K step, and every SIMD group's run reads it.  The staging is paid once
per threadgroup and K tile, so its cost per output element falls with the
rows that share it: the other operand's side (the view's, "tall") wants as
many rows as the threadgroup's SIMD groups can hold, the staged side
("narrow") as few columns as an MPP run takes, so the grid still covers the
problem.  The search's neighbours step one block or ``num_warps`` at a time
from the default ``[16, 16, 16]`` at 4 SIMD groups, through tiles where the
staging dominates, so it rarely gets there.

Each seed is a tall block of ``ROWS_PER_SIMD_GROUP`` rows per SIMD group, at
most ``MAX_SIMD_GROUPS`` of them and never past the next power of two of the
tall extent, beside a narrow block of 16 or 32 and a K block of 128 or 256.
A tall block under 64 rows (a tall extent of 32 or less) gets no seed.  On the M4 Pro, over
138 tiles of 4-bit dequantize-and-matmul at 511 x 4096 x 6656 (``int4_qmm``),
every tile within 2% of the best was a 256-row block at 8 SIMD groups with a
16 or 32 narrow block and a 128 or 256 K block; 16 SIMD groups, 64-row blocks
and 64-wide narrow blocks ran 10-30% slower.  Those tiles ran 1.04-1.06x
MLX's ``quantized_matmul`` on the four large shapes of a 30B model's prefill
(N x K from 4096 x 6656 to 39936 x 6656) and 0.97x on 256 x 6656.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch._inductor.runtime.runtime_utils import next_power_of_2

from ...runtime.config import Config
from ..autotuner_heuristics.registry import AutotunerHeuristic
from ._constants import SIMD_WIDTH
from .search_surface import kernel_facts

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR

#: Rows of the tall block each SIMD group of the run takes.
ROWS_PER_SIMD_GROUP = SIMD_WIDTH
#: Most SIMD groups a seed's threadgroup runs.
MAX_SIMD_GROUPS = 8
#: The staged side's block: an MPP run's narrowest tiles.
NARROW_BLOCKS = (16, 32)
#: The K block: whole 4-bit packs and quantization groups per lane, 4-8 MPP
#: K steps per staged tile.
K_BLOCKS = (256, 128)


class MetalStagedMatmulHeuristic(AutotunerHeuristic):
    name = "metal_staged_matmul"
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
        staged = kernel_facts(spec).staged_matmuls
        # Every matmul stages an operand over the same blocks (one, or
        # several sharing a K loop, as a fused gate and up projection), so
        # one tall tile serves them all.
        if len(set(staged)) != 1 or len(staged) != len(spec.matmul_facts):
            return []
        tall, narrow, k = staged[0]
        tunable = set(spec.block_sizes.valid_block_ids())
        if not {tall, narrow, k} <= tunable:
            return []
        specs = {b: spec.block_sizes.block_id_lookup(b) for b in (tall, narrow, k)}
        fragments = {b: s._fragment(spec) for b, s in specs.items()}
        rows = min(
            next_power_of_2(specs[tall].size_hint),
            ROWS_PER_SIMD_GROUP * MAX_SIMD_GROUPS,
            fragments[tall].high,
        )
        # A short tall side gains nothing from sharing; few-row matmuls have
        # their own lowerings (fragments, row kernels).
        if rows < 2 * ROWS_PER_SIMD_GROUP or rows < fragments[tall].low:
            return []
        default = spec.default_config()
        seeds: list[Config] = []
        for k_block in K_BLOCKS:
            for narrow_block in NARROW_BLOCKS:
                chosen = {tall: rows, narrow: narrow_block, k: k_block}
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
                seed = Config.from_dict(
                    {
                        **default.config,
                        "block_sizes": block_sizes,
                        "num_warps": rows // ROWS_PER_SIMD_GROUP,
                    }
                )
                if seed not in seeds:
                    seeds.append(seed)
        return seeds
