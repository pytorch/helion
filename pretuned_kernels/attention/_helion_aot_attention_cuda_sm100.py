"""
Heuristic for kernel: attention (B200 / sm100, CuTe tcgen05 flash backend)
Backend: decision_tree

Provides:
- key_attention(*args): Returns config index (also serves as cache key)
- autotune_attention(*args): Returns config dict for the given arguments

Configs are the B200 compiler-selected Flash seed configs captured from
benchmark runs (paste P2418822908) at Helion revision
9ca92a66d199a917f4fbe65dfc313879505649b4 on an NVIDIA B200 (1965 MHz, 1000 W).
The kernel is non-causal, so config selection depends on (seq_len, head_dim):

- head_dim >= 128                -> config_02 (index 1)
- head_dim == 64, seq_len <= 4096 -> config_01 (index 0)
- head_dim == 64, seq_len  > 4096 -> config_06 (index 2)
"""

import torch


def key_attention(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    q = args[0]
    seq_len = int(q.shape[-2]) if isinstance(q, torch.Tensor) and q.ndim >= 2 else 0
    head_dim = int(q.shape[-1]) if isinstance(q, torch.Tensor) and q.ndim >= 1 else 0
    if head_dim >= 128:
        return 1
    if seq_len <= 4096:
        return 0
    return 2


def autotune_attention(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 128, 128], 'cute_flash_causal_kv_order': 'ascending', 'cute_flash_causal_loop_split': False, 'cute_flash_causal_lpt_swizzle': 0, 'cute_flash_corr_regs': 64, 'cute_flash_disc_pipe': 4, 'cute_flash_e2e_offset': 2, 'cute_flash_e2e_offset0': 0, 'cute_flash_e2e_schedule': '16/4', 'cute_flash_epi_tma': False, 'cute_flash_kv_stage': 3, 'cute_flash_masked_e2e_schedule': 'inherit', 'cute_flash_packed_reduce': False, 'cute_flash_persistent': True, 'cute_flash_rescale_chunk_cols': 32, 'cute_flash_rescale_threshold': 8.0, 'cute_flash_role_map': 'helion', 'cute_flash_s_stage': 2, 'cute_flash_small_biased': True, 'cute_flash_softmax_regs': 200, 'cute_flash_topology': 'fa4'},
        {'block_sizes': [1, 128, 128], 'cute_flash_causal_kv_order': 'ascending', 'cute_flash_causal_loop_split': False, 'cute_flash_causal_lpt_swizzle': 0, 'cute_flash_corr_regs': 64, 'cute_flash_disc_pipe': 2, 'cute_flash_e2e_offset': 0, 'cute_flash_e2e_offset0': 0, 'cute_flash_e2e_schedule': '8/2', 'cute_flash_epi_tma': True, 'cute_flash_kv_stage': 3, 'cute_flash_masked_e2e_schedule': 'inherit', 'cute_flash_packed_reduce': False, 'cute_flash_persistent': True, 'cute_flash_rescale_chunk_cols': 16, 'cute_flash_rescale_threshold': 8.0, 'cute_flash_role_map': 'helion', 'cute_flash_s_stage': 2, 'cute_flash_small_biased': True, 'cute_flash_softmax_regs': 200, 'cute_flash_topology': 'fa4'},
        {'block_sizes': [1, 128, 128], 'cute_flash_causal_kv_order': 'ascending', 'cute_flash_causal_loop_split': False, 'cute_flash_causal_lpt_swizzle': 0, 'cute_flash_corr_regs': 64, 'cute_flash_disc_pipe': 3, 'cute_flash_e2e_offset': 2, 'cute_flash_e2e_offset0': 0, 'cute_flash_e2e_schedule': '16/4', 'cute_flash_epi_tma': False, 'cute_flash_kv_stage': 2, 'cute_flash_masked_e2e_schedule': 'inherit', 'cute_flash_packed_reduce': True, 'cute_flash_persistent': True, 'cute_flash_rescale_chunk_cols': 32, 'cute_flash_rescale_threshold': 8.0, 'cute_flash_role_map': 'helion', 'cute_flash_s_stage': 2, 'cute_flash_small_biased': True, 'cute_flash_softmax_regs': 200, 'cute_flash_topology': 'fa4'},
    ]
    return _C[key_attention(*args)]
