"""
Auto-generated heuristic for kernel: rms_norm
Backend: decision_tree

Provides:
- key_rms_norm(*args): Returns config index (cache key)
- autotune_rms_norm(*args): Returns config dict for the given arguments
"""

import torch


def key_rms_norm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune_rms_norm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', ''], 'num_warps': 16, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'flat'},
    ]
    return _C[key_rms_norm(*args)]
