"""
Auto-generated heuristic for kernel: scaled_mm
Backend: decision_tree

Provides:
- key_scaled_mm(*args): Returns config index (cache key)
- autotune_scaled_mm(*args): Returns config dict for the given arguments

Pretuned on NVIDIA GB10 (sm121) with quick autotune effort on the two
decode shapes from the correctness test, (M, K, N) = (16, 4096, 4096) and
(64, 2048, 2048). The single selected config fits the GB10's 101376-byte
shared-memory-per-block limit, unlike the sm90 heuristics it otherwise
falls back to (which require up to 147456 bytes and fail to launch).
"""

import torch


def key_scaled_mm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune_scaled_mm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [64, 16, 256], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_scaled_mm(*args)]
