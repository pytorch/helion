"""B200 configs for the CuTe flash attention kernel."""

from __future__ import annotations

import torch


# The third config is the long-sequence head-dim-64 fallback used by the
# comparison harness's dense_causal8 suite, beyond the representative SHAPES
# timed by pretuned_kernels/attention/attention.py.
_CONFIGS = [
    {
        "block_sizes": [1, 128, 128],
        "cute_flash_causal_kv_order": "ascending",
        "cute_flash_causal_loop_split": False,
        "cute_flash_causal_lpt_swizzle": 0,
        "cute_flash_corr_regs": 64,
        "cute_flash_disc_pipe": 4,
        "cute_flash_e2e_offset": 2,
        "cute_flash_e2e_offset0": 0,
        "cute_flash_e2e_schedule": "16/4",
        "cute_flash_epi_tma": False,
        "cute_flash_kv_stage": 3,
        "cute_flash_masked_e2e_schedule": "inherit",
        "cute_flash_packed_reduce": False,
        "cute_flash_persistent": True,
        "cute_flash_rescale_chunk_cols": 32,
        "cute_flash_rescale_threshold": 8.0,
        "cute_flash_role_map": "helion",
        "cute_flash_s_stage": 2,
        "cute_flash_small_biased": True,
        "cute_flash_softmax_regs": 200,
        "cute_flash_topology": "fa4",
    },
    {
        "block_sizes": [1, 128, 128],
        "cute_flash_causal_kv_order": "ascending",
        "cute_flash_causal_loop_split": False,
        "cute_flash_causal_lpt_swizzle": 0,
        "cute_flash_corr_regs": 64,
        "cute_flash_disc_pipe": 2,
        "cute_flash_e2e_offset": 0,
        "cute_flash_e2e_offset0": 0,
        "cute_flash_e2e_schedule": "8/2",
        "cute_flash_epi_tma": True,
        "cute_flash_kv_stage": 3,
        "cute_flash_masked_e2e_schedule": "inherit",
        "cute_flash_packed_reduce": False,
        "cute_flash_persistent": True,
        "cute_flash_rescale_chunk_cols": 16,
        "cute_flash_rescale_threshold": 8.0,
        "cute_flash_role_map": "helion",
        "cute_flash_s_stage": 2,
        "cute_flash_small_biased": True,
        "cute_flash_softmax_regs": 200,
        "cute_flash_topology": "fa4",
    },
    {
        "block_sizes": [1, 128, 128],
        "cute_flash_causal_kv_order": "ascending",
        "cute_flash_causal_loop_split": False,
        "cute_flash_causal_lpt_swizzle": 0,
        "cute_flash_corr_regs": 64,
        "cute_flash_disc_pipe": 3,
        "cute_flash_e2e_offset": 2,
        "cute_flash_e2e_offset0": 0,
        "cute_flash_e2e_schedule": "16/4",
        "cute_flash_epi_tma": False,
        "cute_flash_kv_stage": 2,
        "cute_flash_masked_e2e_schedule": "inherit",
        "cute_flash_packed_reduce": True,
        "cute_flash_persistent": True,
        "cute_flash_rescale_chunk_cols": 32,
        "cute_flash_rescale_threshold": 8.0,
        "cute_flash_role_map": "helion",
        "cute_flash_s_stage": 2,
        "cute_flash_small_biased": True,
        "cute_flash_softmax_regs": 200,
        "cute_flash_topology": "fa4",
    },
]


def key_attention(*args) -> int:
    """Select a config from the query sequence length and head dimension."""
    q = args[0]
    seq_len = int(q.shape[-2]) if isinstance(q, torch.Tensor) and q.ndim >= 2 else 0
    head_dim = int(q.shape[-1]) if isinstance(q, torch.Tensor) and q.ndim >= 1 else 0
    if head_dim >= 128:
        return 1
    if seq_len <= 4096:
        return 0
    return 2


def autotune_attention(*args) -> dict:
    """Return the checked-in B200 config for the given attention shape."""
    return _CONFIGS[key_attention(*args)]
