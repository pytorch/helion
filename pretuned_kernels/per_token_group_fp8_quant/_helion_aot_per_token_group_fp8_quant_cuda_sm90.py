"""
Auto-generated heuristic for kernel: per_token_group_fp8_quant
Ported from vLLM Helion config nvidia_h100.json (sm90).

Provides:
- key_per_token_group_fp8_quant(*args): Returns config index (cache key)
- autotune_per_token_group_fp8_quant(*args): Returns config dict for the given arguments

Config selection mirrors the vLLM config picker: closest match on the
hidden_size, group_size dimension(s), then the smallest tuned num_tokens >= the input
(falling back to the largest).
"""

import functools

# Tuned key tuples (hidden_size, group_size, num_tokens), parallel to the config list in autotune_per_token_group_fp8_quant.
_KEYS = [
    (5120, 128, 16),
    (2048, 128, 1),
    (2048, 128, 2),
    (2048, 128, 4),
    (2048, 128, 8),
    (2048, 128, 16),
    (2048, 128, 32),
    (2048, 128, 64),
    (2048, 128, 128),
    (2048, 128, 256),
    (2048, 128, 512),
    (2048, 128, 1024),
    (2048, 128, 2048),
    (2048, 128, 4096),
    (2048, 128, 8192),
    (4096, 128, 1),
    (4096, 128, 2),
    (4096, 128, 4),
    (4096, 128, 8),
    (4096, 128, 16),
    (4096, 128, 32),
    (4096, 128, 64),
    (4096, 128, 128),
    (4096, 128, 256),
    (4096, 128, 512),
    (4096, 128, 1024),
    (4096, 128, 2048),
    (4096, 128, 4096),
    (4096, 128, 8192),
    (5120, 128, 1),
    (5120, 128, 2),
    (5120, 128, 4),
    (5120, 128, 8),
    (5120, 128, 32),
    (5120, 128, 64),
    (5120, 128, 128),
    (5120, 128, 256),
    (5120, 128, 512),
    (5120, 128, 1024),
    (5120, 128, 2048),
    (5120, 128, 4096),
    (5120, 128, 8192),
]

_INDEX_BY_KEY = {key: i for i, key in enumerate(_KEYS)}


@functools.cache
def _pick_index(hidden_size: int, group_size: int, num_tokens: int) -> int:
    target = (hidden_size, group_size, num_tokens,)
    exact = _INDEX_BY_KEY.get(target)
    if exact is not None:
        return exact
    # Narrow to the closest value on every dimension except the last.
    cands = _KEYS
    for i in range(len(target) - 1):
        best = min({k[i] for k in cands}, key=lambda v: abs(v - target[i]))
        cands = [k for k in cands if k[i] == best]
    # Last dimension: smallest tuned value >= the input, else the largest.
    last = target[-1]
    avail = sorted({k[-1] for k in cands})
    chosen = next((v for v in avail if v >= last), avail[-1])
    cands = [k for k in cands if k[-1] == chosen]
    return _INDEX_BY_KEY[cands[0]]


def key_per_token_group_fp8_quant(*args) -> int:
    """Select config index for the given arguments (also serves as cache key).

    On the per-call specialization-key path, so the pick is memoized on the
    extracted int key via functools.cache to stay O(1) per call.
    """
    a = args[0]
    hidden_size = a.shape[1]
    group_size = args[3]
    num_tokens = a.shape[0]
    return _pick_index(hidden_size, group_size, num_tokens)


def autotune_per_token_group_fp8_quant(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 2, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 2, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 1, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 1, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 2, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 2, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [2], 'range_warp_specializes': [], 'range_multi_buffers': [True], 'range_flattens': [True], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 128, 'maxnreg': 256},
        {'block_sizes': [8], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 7, 'indexing': ['tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_per_token_group_fp8_quant(*args)]
