"""B200 configs for the CuTe flash attention kernel.

Long dense and causal schedules are tuned per sequence length at both
head_dim 64 and head_dim 128 for the sweep in
benchmarks/cute/compare_attention_backends.py. Slots 0-2 keep the
short-sequence and generic fallback schedules.

Every long-sequence entry is the fastest schedule found for its shape across
twelve full-effort autotune runs plus a cross-shape pool of every config those
runs produced, re-timed with the sweep's own cudagraph timer. Against cuDNN
SDPA the sweep lands at geomean 1.00x, winning on 6 of 16 shapes; the dense
head_dim 128 64K/128K rows are the furthest behind at ~0.96x and did not
improve across ~900-config searches, so that gap looks kernel-side rather than
a matter of config selection.
"""

from __future__ import annotations

import torch


_LONG_DENSE_HD64_BASE = {
    "block_sizes": [1, 128, 128],
    "cute_flash_causal_kv_order": "ascending",
    "cute_flash_causal_loop_split": False,
    "cute_flash_causal_lpt_swizzle": 0,
    "cute_flash_clc_heads_per_batch": 0,
    "cute_flash_clc_pdl": False,
    "cute_flash_clc_stages": 1,
    "cute_flash_corr_regs": 72,
    "cute_flash_corr_tile_size": 8,
    "cute_flash_disc_pipe": 1,
    "cute_flash_e2e_schedule": "16/6",
    "cute_flash_epi_stg": False,
    "cute_flash_epi_stg_gmem": "stage",
    "cute_flash_epi_stg_store": "slice",
    "cute_flash_epi_tma": True,
    "cute_flash_exp2_packet": "deg2_16x6",
    "cute_flash_kv_order": "descending",
    "cute_flash_kv_stage": 2,
    "cute_flash_masked_e2e_schedule": "inherit",
    "cute_flash_mma_interleave": True,
    "cute_flash_other_regs": 40,
    "cute_flash_p_store_rep": 16,
    "cute_flash_packed_reduce": True,
    "cute_flash_persistent": False,
    "cute_flash_persistent_ctas_per_sm": 1,
    "cute_flash_pipeline_family": "fa4_2cta",
    "cute_flash_precompute_qk_desc": True,
    "cute_flash_q_tile_count": 2,
    "cute_flash_recompute_tile_coords": False,
    "cute_flash_rescale_chunk_cols": 8,
    "cute_flash_s_load_rep": 32,
    "cute_flash_s_stage": 2,
    "cute_flash_skip_rescale_stats": False,
    "cute_flash_small_biased": True,
    "cute_flash_softmax_disc": False,
    "cute_flash_softmax_regs": 192,
    "cute_flash_split_p_arrive": True,
    "cute_flash_stat_transport": "single_final",
    "cute_flash_wait_hint": 0,
}


_LONG_DENSE_HD128_BASE = {
    "block_sizes": [1, 128, 128],
    "cute_flash_causal_kv_order": "ascending",
    "cute_flash_causal_loop_split": False,
    "cute_flash_causal_lpt_swizzle": 0,
    "cute_flash_clc_heads_per_batch": 0,
    "cute_flash_clc_pdl": False,
    "cute_flash_clc_stages": 1,
    "cute_flash_e2e_schedule": "8/2",
    "cute_flash_epi_stg": False,
    "cute_flash_epi_stg_gmem": "stage",
    "cute_flash_epi_stg_store": "slice",
    "cute_flash_epi_tma": False,
    "cute_flash_epi_tma_setup": "shared",
    "cute_flash_exp2_packet": "1x1",
    "cute_flash_masked_e2e_schedule": "inherit",
    "cute_flash_mma_interleave": True,
    "cute_flash_p_store_rep": 16,
    "cute_flash_packed_reduce": True,
    "cute_flash_persistent": False,
    "cute_flash_persistent_ctas_per_sm": 1,
    "cute_flash_persistent_loop": "while",
    "cute_flash_pipeline_family": "fa4_2cta",
    "cute_flash_precompute_qk_desc": False,
    "cute_flash_q_tile_count": 2,
    "cute_flash_recompute_tile_coords": False,
    "cute_flash_role_chain": False,
    "cute_flash_s_load_rep": 32,
    "cute_flash_s_stage": 2,
    "cute_flash_skip_rescale_stats": False,
    "cute_flash_small_biased": True,
    "cute_flash_softmax_disc": True,
    "cute_flash_softmax_regs": 192,
    "cute_flash_softmax_setup": "shared",
    "cute_flash_sp_row_sum": "fragment",
    "cute_flash_split_p_arrive": True,
    "cute_flash_stat_transport": "ring2",
}


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
    },  # slot 0
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
    },  # slot 1
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
    },  # slot 2
    # Long dense head_dim=64 schedules.
    {
        **_LONG_DENSE_HD64_BASE,
        "cute_flash_e2e_offset": 0,
        "cute_flash_e2e_offset0": 1,
        "cute_flash_first_load_order": 4,
        "cute_flash_rescale_threshold": 12.0,
        "cute_flash_role_map": "fa4",
    },  # seq_len=32768, 65536
    {
        **_LONG_DENSE_HD64_BASE,
        "cute_flash_e2e_offset": 0,
        "cute_flash_e2e_offset0": 0,
        "cute_flash_first_load_order": 0,
        "cute_flash_rescale_threshold": 8.0,
        "cute_flash_role_map": "helion",
    },  # seq_len=131072
    {
        **_LONG_DENSE_HD64_BASE,
        "cute_flash_e2e_offset": 12,
        "cute_flash_e2e_offset0": 2,
        "cute_flash_first_load_order": 4,
        "cute_flash_rescale_threshold": 8.0,
        "cute_flash_role_map": "fa4",
    },  # seq_len=262144
    # Long dense head_dim=128 schedules.
    {
        **_LONG_DENSE_HD128_BASE,
        "cute_flash_corr_regs": 64,
        "cute_flash_corr_tile_size": 16,
        "cute_flash_disc_pipe": 4,
        "cute_flash_e2e_offset": 0,
        "cute_flash_e2e_offset0": 4,
        "cute_flash_first_load_order": 0,
        "cute_flash_kv_order": "descending",
        "cute_flash_kv_stage": 4,
        "cute_flash_other_regs": 64,
        "cute_flash_rescale_chunk_cols": 16,
        "cute_flash_rescale_threshold": 12.0,
        "cute_flash_role_map": "helion",
        "cute_flash_wait_hint": 0,
    },  # seq_len=32768
    {
        **_LONG_DENSE_HD128_BASE,
        "cute_flash_corr_regs": 64,
        "cute_flash_corr_tile_size": 32,
        "cute_flash_disc_pipe": 3,
        "cute_flash_e2e_offset": 2,
        "cute_flash_e2e_offset0": 2,
        "cute_flash_first_load_order": 2,
        "cute_flash_kv_order": "ascending",
        "cute_flash_kv_stage": 4,
        "cute_flash_other_regs": 48,
        "cute_flash_rescale_chunk_cols": 8,
        "cute_flash_rescale_threshold": 4.0,
        "cute_flash_role_map": "fa4",
        "cute_flash_wait_hint": 10000000,
    },  # seq_len=65536
    {
        **_LONG_DENSE_HD128_BASE,
        "cute_flash_corr_regs": 80,
        "cute_flash_corr_tile_size": 16,
        "cute_flash_disc_pipe": 4,
        "cute_flash_e2e_offset": 3,
        "cute_flash_e2e_offset0": 2,
        "cute_flash_first_load_order": 4,
        "cute_flash_kv_order": "descending",
        "cute_flash_kv_stage": 4,
        "cute_flash_other_regs": 48,
        "cute_flash_rescale_chunk_cols": 16,
        "cute_flash_rescale_threshold": 12.0,
        "cute_flash_role_map": "helion",
        "cute_flash_wait_hint": 0,
    },  # seq_len=131072
    {
        **_LONG_DENSE_HD128_BASE,
        "cute_flash_corr_regs": 64,
        "cute_flash_corr_tile_size": 32,
        "cute_flash_disc_pipe": 3,
        "cute_flash_e2e_offset": 0,
        "cute_flash_e2e_offset0": 2,
        "cute_flash_first_load_order": 4,
        "cute_flash_kv_order": "ascending",
        "cute_flash_kv_stage": 3,
        "cute_flash_other_regs": 64,
        "cute_flash_rescale_chunk_cols": 16,
        "cute_flash_rescale_threshold": 8.0,
        "cute_flash_role_map": "helion",
        "cute_flash_wait_hint": 10000000,
    },  # seq_len=262144
]


_LONG_DENSE_HD64_CONFIGS = {
    32768: 3,
    65536: 3,
    131072: 4,
    262144: 5,
}


_LONG_DENSE_HD128_CONFIGS = {
    32768: 6,
    65536: 7,
    131072: 8,
    262144: 9,
}


def key_attention(*args: torch.Tensor) -> int:
    """Select a config from the query sequence length and head dimension."""
    q = args[0]
    seq_len = int(q.shape[-2]) if isinstance(q, torch.Tensor) and q.ndim >= 2 else 0
    head_dim = int(q.shape[-1]) if isinstance(q, torch.Tensor) and q.ndim >= 1 else 0
    if q.dtype == torch.float16:
        if head_dim == 64 and seq_len in _LONG_DENSE_HD64_CONFIGS:
            return _LONG_DENSE_HD64_CONFIGS[seq_len]
        if head_dim == 128 and seq_len in _LONG_DENSE_HD128_CONFIGS:
            return _LONG_DENSE_HD128_CONFIGS[seq_len]
    if head_dim >= 128:
        return 1
    if seq_len <= 4096:
        return 0
    return 2


def autotune_attention(*args: torch.Tensor) -> dict[str, object]:
    """Return the checked-in B200 config for the given attention shape."""
    return _CONFIGS[key_attention(*args)]


_LONG_CAUSAL_HD64_BASE = {
    "block_sizes": [1, 128, 128],
    "cute_flash_causal_kv_order": "descending",
    "cute_flash_causal_loop_split": True,
    "cute_flash_clc_heads_per_batch": 0,
    "cute_flash_clc_pdl": False,
    "cute_flash_clc_stages": 1,
    "cute_flash_corr_regs": 64,
    "cute_flash_e2e_offset": 0,
    "cute_flash_e2e_schedule": "16/6",
    "cute_flash_epi_stg": False,
    "cute_flash_epi_stg_gmem": "stage",
    "cute_flash_epi_stg_store": "slice",
    "cute_flash_exp2_packet": "deg2_16x6",
    "cute_flash_first_load_order": 0,
    "cute_flash_kv_order": "ascending",
    "cute_flash_masked_e2e_schedule": "16/6",
    "cute_flash_mma_interleave": True,
    "cute_flash_other_regs": 48,
    "cute_flash_p_store_rep": 16,
    "cute_flash_packed_reduce": True,
    "cute_flash_persistent": False,
    "cute_flash_persistent_ctas_per_sm": 1,
    "cute_flash_pipeline_family": "fa4",
    "cute_flash_q_tile_count": 2,
    "cute_flash_recompute_tile_coords": False,
    "cute_flash_rescale_chunk_cols": 16,
    "cute_flash_rescale_threshold": 8.0,
    "cute_flash_role_map": "fa4",
    "cute_flash_s_load_rep": 32,
    "cute_flash_s_stage": 2,
    "cute_flash_skip_rescale_stats": False,
    "cute_flash_small_biased": True,
    "cute_flash_softmax_disc": True,
    "cute_flash_split_p_arrive": True,
    "cute_flash_stat_transport": "ring2",
    "cute_flash_wait_hint": 0,
}


_LONG_CAUSAL_HD128_BASE = {
    "block_sizes": [1, 128, 128],
    "cute_flash_causal_kv_order": "descending",
    "cute_flash_causal_loop_split": True,
    "cute_flash_causal_lpt_swizzle": 1,
    "cute_flash_clc_heads_per_batch": 0,
    "cute_flash_clc_pdl": False,
    "cute_flash_clc_stages": 1,
    "cute_flash_corr_regs": 80,
    "cute_flash_disc_pipe": 1,
    "cute_flash_e2e_schedule": "8/2",
    "cute_flash_epi_stg_gmem": "stage",
    "cute_flash_epi_stg_store": "slice",
    "cute_flash_epi_tma_setup": "shared",
    "cute_flash_exp2_packet": "1x1",
    "cute_flash_kv_order": "ascending",
    "cute_flash_mma_interleave": True,
    "cute_flash_p_store_rep": 16,
    "cute_flash_packed_reduce": True,
    "cute_flash_persistent": False,
    "cute_flash_persistent_ctas_per_sm": 1,
    "cute_flash_persistent_loop": "while",
    "cute_flash_pipeline_family": "fa4",
    "cute_flash_precompute_qk_desc": False,
    "cute_flash_q_tile_count": 2,
    "cute_flash_recompute_tile_coords": False,
    "cute_flash_role_chain": False,
    "cute_flash_role_map": "helion",
    "cute_flash_s_load_rep": 16,
    "cute_flash_s_stage": 2,
    "cute_flash_skip_rescale_stats": False,
    "cute_flash_small_biased": True,
    "cute_flash_softmax_disc": True,
    "cute_flash_softmax_setup": "shared",
    "cute_flash_sp_row_sum": "fragment",
    "cute_flash_split_p_arrive": True,
    "cute_flash_stat_transport": "ring2",
    "cute_flash_wait_hint": 10000000,
}


_CAUSAL_CONFIGS = [
    # Long causal head_dim=64 schedules.
    {
        **_LONG_CAUSAL_HD64_BASE,
        "cute_flash_causal_lpt_swizzle": 0,
        "cute_flash_corr_tile_size": 16,
        "cute_flash_disc_pipe": 3,
        "cute_flash_e2e_offset0": 14,
        "cute_flash_epi_tma": True,
        "cute_flash_kv_stage": 3,
        "cute_flash_precompute_qk_desc": False,
        "cute_flash_softmax_regs": 200,
    },  # seq_len=65536
    {
        **_LONG_CAUSAL_HD64_BASE,
        "cute_flash_causal_lpt_swizzle": 1,
        "cute_flash_corr_tile_size": 8,
        "cute_flash_disc_pipe": 2,
        "cute_flash_e2e_offset0": 13,
        "cute_flash_epi_tma": False,
        "cute_flash_epi_tma_setup": "shared",
        "cute_flash_kv_stage": 2,
        "cute_flash_persistent_loop": "while",
        "cute_flash_precompute_qk_desc": True,
        "cute_flash_role_chain": False,
        "cute_flash_softmax_regs": 192,
        "cute_flash_softmax_setup": "shared",
        "cute_flash_sp_row_sum": "fragment",
    },  # seq_len=131072, 262144, 524288
    # Long causal head_dim=128 schedules.
    {
        **_LONG_CAUSAL_HD128_BASE,
        "cute_flash_corr_tile_size": 16,
        "cute_flash_e2e_offset": 0,
        "cute_flash_e2e_offset0": 2,
        "cute_flash_epi_stg": True,
        "cute_flash_epi_tma": False,
        "cute_flash_first_load_order": 4,
        "cute_flash_kv_stage": 3,
        "cute_flash_masked_e2e_schedule": "inherit",
        "cute_flash_other_regs": 32,
        "cute_flash_rescale_chunk_cols": 32,
        "cute_flash_rescale_threshold": 12.0,
        "cute_flash_softmax_regs": 176,
    },  # seq_len=65536
    {
        **_LONG_CAUSAL_HD128_BASE,
        "cute_flash_corr_tile_size": 32,
        "cute_flash_e2e_offset": 5,
        "cute_flash_e2e_offset0": 3,
        "cute_flash_epi_stg": False,
        "cute_flash_epi_tma": True,
        "cute_flash_first_load_order": 0,
        "cute_flash_kv_stage": 2,
        "cute_flash_masked_e2e_schedule": "xu",
        "cute_flash_other_regs": 40,
        "cute_flash_rescale_chunk_cols": 16,
        "cute_flash_rescale_threshold": 8.0,
        "cute_flash_softmax_regs": 184,
    },  # seq_len=131072, 524288
    {
        **_LONG_CAUSAL_HD128_BASE,
        "cute_flash_corr_tile_size": 16,
        "cute_flash_e2e_offset": 2,
        "cute_flash_e2e_offset0": 2,
        "cute_flash_epi_stg": False,
        "cute_flash_epi_tma": True,
        "cute_flash_first_load_order": 4,
        "cute_flash_kv_stage": 2,
        "cute_flash_masked_e2e_schedule": "inherit",
        "cute_flash_other_regs": 56,
        "cute_flash_rescale_chunk_cols": 32,
        "cute_flash_rescale_threshold": 12.0,
        "cute_flash_softmax_regs": 176,
    },  # seq_len=262144
]

_CAUSAL_HD64_CONFIGS = {
    65536: 0,
    131072: 1,
    262144: 1,
    524288: 1,
}


_CAUSAL_HD128_CONFIGS = {
    65536: 2,
    131072: 3,
    262144: 4,
    524288: 3,
}


def key_causal_attention(*args: torch.Tensor) -> int:
    """Select the B200 long-sequence causal config."""
    q = args[0]
    seq_len = int(q.shape[-2]) if isinstance(q, torch.Tensor) and q.ndim >= 2 else 0
    head_dim = int(q.shape[-1]) if isinstance(q, torch.Tensor) and q.ndim >= 1 else 0
    if q.dtype == torch.float16 and head_dim == 128:
        return _CAUSAL_HD128_CONFIGS.get(seq_len, 2)
    return _CAUSAL_HD64_CONFIGS.get(seq_len, 0)


def autotune_causal_attention(*args: torch.Tensor) -> dict[str, object]:
    """Return the checked-in B200 causal config for the given shape."""
    return _CAUSAL_CONFIGS[key_causal_attention(*args)]
