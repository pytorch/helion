"""Kernels and configs shared by the ``cute_tail_peel`` tests: the online
softmax statistics of a bf16 row whose length is not a multiple of the tile
(better-benchmark's cross-entropy stats, at test sizes), and a persistent
row sum followed by a dependent update across a barrier."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# One warp per row, eight bf16 per lane: 256-column tiles.
STATS_CONFIG = {
    "block_sizes": [1, 256],
    "num_threads": [1, 32],
    "cute_vector_widths": [8, 8],
}


@helion.kernel(backend="cute", static_shapes=True)
def online_softmax_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's max and sum of ``exp(x - max)``, folded tile by tile the
    way Inductor's online softmax does."""
    rows, vocab = logits.shape
    row_max = torch.empty([rows], dtype=torch.float32, device=logits.device)
    row_sum = torch.empty([rows], dtype=torch.float32, device=logits.device)
    for tile_r in hl.tile(rows):
        run_max = hl.full([tile_r], float("-inf"), dtype=torch.float32)
        run_sum = hl.zeros([tile_r], dtype=torch.float32)
        for tile_v in hl.tile(vocab):
            x = logits[tile_r, tile_v].to(torch.float32)
            block_max = torch.amax(x, dim=-1)
            delta = torch.where(
                block_max[:, None] == float("-inf"), 0.0, x - block_max[:, None]
            )
            block_sum = torch.sum(torch.exp(delta), dim=-1)
            new_max = torch.maximum(run_max, block_max)
            scale = torch.exp(-torch.abs(run_max - block_max))
            run_scale = torch.where(run_max == new_max, 1.0, scale)
            block_scale = torch.where(block_max == new_max, 1.0, scale)
            run_sum = run_sum * run_scale + block_sum * block_scale
            run_max = new_max
        row_max[tile_r] = run_max
        row_sum[tile_r] = run_sum
    return row_max, row_sum


def padded_rows(
    rows: int, vocab: int, device: torch.device | str, pad: int = 8
) -> torch.Tensor:
    """A ``[rows, vocab]`` bf16 view of rows padded to a multiple of ``pad``
    elements, so each row starts 16-byte aligned (better-benchmark's sliced
    logits)."""
    stride = -(-vocab // pad) * pad
    return torch.randn(rows, stride, device=device, dtype=torch.bfloat16)[:, :vocab]


# One warp reduces the partial row; the second phase updates the scalar.
PHASE_CONFIG = {
    "block_sizes": [1, 256, 1],
    "num_threads": [1, 32, 1],
    "cute_vector_widths": [8, 8, 1],
}


@helion.kernel(backend="cute", static_shapes=True)
def phased_tail_rows(x: torch.Tensor) -> torch.Tensor:
    """Sum each row up to its logical end, then add one in a later phase."""
    rows, cols = x.shape
    out = torch.empty((rows,), device=x.device, dtype=torch.float32)
    for i in hl.tile(rows):
        acc = hl.zeros([i], dtype=torch.float32)
        for j in hl.tile(cols):
            acc = acc + x[i, j].to(torch.float32).sum(-1)
        out[i] = acc
    hl.barrier()
    for i in hl.tile(rows):
        out[i] = out[i] + 1
    return out
