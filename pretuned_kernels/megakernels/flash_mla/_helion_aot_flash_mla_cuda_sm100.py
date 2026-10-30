"""Checked-in B200 config for the one-request DeepSeek-V3 MLA decode megakernel.

Context length is runtime metadata (one compiled kernel serves every length up to
the block-table capacity).
"""

from __future__ import annotations

from copy import deepcopy

import torch

# block_sizes = [QK contraction chunk, combine splits per step, output columns per partial task,
#                combine query rows, combine value columns].  loop_orders put the column slice
# innermost so a short context's live split tasks take the first dispatch tickets.
CONFIG = {
    "block_sizes": [64, 16, 128, 1, 512],
    "loop_orders": [[1, 0], [1, 0]],
    "num_warps": 4,
    "num_stages": 4,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
}

_HEADS = 16
_QK_DIM = 576
_PAGE = 64


def key_flash_mla(*args) -> int:
    """Validate the one-request TP8 geometry; context length and KV pool size stay runtime."""
    if len(args) != 5:
        raise ValueError("flash_mla takes (query, kv_cache, block_tables, seq_lens, scale)")
    query, kv_cache, block_tables, seq_lens, scale = args
    if (
        query.dim() != 4
        or query.size(0) != 1
        or tuple(query.shape[2:]) != (_HEADS, _QK_DIM)
        or query.dtype != torch.bfloat16
        or kv_cache.dim() != 4
        or tuple(kv_cache.shape[1:]) != (1, _PAGE, _QK_DIM)
        or kv_cache.dtype != torch.bfloat16
        or block_tables.dim() != 2
        or block_tables.size(0) != 1
        or block_tables.dtype != torch.int32
        or tuple(seq_lens.shape) != (1,)
        or seq_lens.dtype != torch.int32
        or scale != _QK_DIM**-0.5
    ):
        raise ValueError(
            "flash_mla is pretuned for one request, 16 heads, 576-wide BF16 keys in 64-token pages"
        )
    return 0


def autotune_flash_mla(*args) -> dict[str, object]:
    """Return the validated B200 dynamic-pipeline configuration."""
    key_flash_mla(*args)
    return deepcopy(CONFIG)
