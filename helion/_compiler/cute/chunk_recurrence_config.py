"""Validated resource budgets for resident tensor-memory recurrences."""

from __future__ import annotations

from dataclasses import dataclass

CUTE_CHUNK_RECURRENCE_PIPELINE_KEY = "cute_chunk_recurrence_pipeline"
CUTE_CHUNK_RECURRENCE_PIPELINES = ("wide", "compact")


@dataclass(frozen=True)
class ChunkRecurrencePipeline:
    input_stages: int
    tma_stages: int
    output_acc_stages: int
    tmem_cols: int
    compute_registers: int
    service_registers: int
    min_blocks_per_mp: int

    @property
    def smem_bytes(self) -> int:
        # Each input stage holds three K128 factors, V64, a K128 FP32
        # decay, and a 16x16 BF16 factor. Seven V64 output stages and
        # 1024 bytes of barrier/alignment storage complete the allocation.
        return self.input_stages * 15_360 + 7 * 2_048 + 1_024


_PIPELINES = {
    "wide": ChunkRecurrencePipeline(8, 6, 2, 512, 136, 56, 1),
    # One output accumulator leaves the live TMEM footprint at 256 columns.
    # The register budget is (3*72 + 40)*128 = 32768 registers per CTA,
    # including dynamic redistribution between the four warpgroups.
    "compact": ChunkRecurrencePipeline(6, 4, 1, 256, 72, 40, 2),
}


def chunk_recurrence_pipeline(name: str) -> ChunkRecurrencePipeline:
    return _PIPELINES[name]
