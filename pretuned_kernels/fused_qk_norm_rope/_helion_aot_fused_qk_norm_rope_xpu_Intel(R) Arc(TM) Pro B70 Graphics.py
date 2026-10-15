"""
Auto-generated heuristic for kernel: fused_qk_norm_rope
Backend: decision_tree

Provides:
- key_fused_qk_norm_rope(*args): Returns config index (cache key)
- autotune_fused_qk_norm_rope(*args): Returns config dict for the given arguments
"""

import torch


def key_fused_qk_norm_rope(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_numel = int(args[0].numel()) if len(args) > 0 and isinstance(args[0], torch.Tensor) else 0
    if _arg0_dim0 <= 8.0:
        return 1
    else:
        if _arg0_numel <= 50331648.0:
            return 0
        else:
            return 1


def autotune_fused_qk_norm_rope(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [4], 'loop_orders': [[0, 2, 1]], 'l2_groupings': [1], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [True], 'range_flattens': [False], 'load_eviction_policies': ['', 'first', '', 'first', '', 'last', 'last', 'first'], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 128},
        {'block_sizes': [16], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', '', 'last', 'last', '', '', 'first'], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_fused_qk_norm_rope(*args)]
