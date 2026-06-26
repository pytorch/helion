"""
Auto-generated heuristic for kernel: fused_qk_norm_rope
Ported from vLLM Helion config fused_qk_norm_rope_h100.json (sm90).

Provides:
- key_fused_qk_norm_rope(*args): Returns config index (cache key)
- autotune_fused_qk_norm_rope(*args): Returns config dict for the given arguments

Config selection mirrors the vLLM config picker: closest match on the
q_heads, kv_heads dimension(s), then the smallest tuned num_tokens >= the input
(falling back to the largest).
"""

import functools

# Tuned key tuples (q_heads, kv_heads, num_tokens), parallel to the config list in autotune_fused_qk_norm_rope.
_KEYS = [
    (16, 8, 1),
    (32, 8, 1),
    (64, 8, 1),
    (16, 8, 2),
    (32, 8, 2),
    (64, 8, 2),
    (16, 8, 4),
    (32, 8, 4),
    (64, 8, 4),
    (16, 8, 8),
    (32, 8, 8),
    (64, 8, 8),
    (16, 8, 16),
    (32, 8, 16),
    (64, 8, 16),
    (16, 8, 32),
    (32, 8, 32),
    (64, 8, 32),
    (16, 8, 64),
    (32, 8, 64),
    (64, 8, 64),
    (16, 8, 128),
    (32, 8, 128),
    (64, 8, 128),
    (16, 8, 256),
    (32, 8, 256),
    (64, 8, 256),
    (16, 8, 512),
    (32, 8, 512),
    (64, 8, 512),
    (16, 8, 1024),
    (32, 8, 1024),
    (64, 8, 1024),
    (16, 8, 2048),
    (32, 8, 2048),
    (64, 8, 2048),
    (16, 8, 4096),
    (32, 8, 4096),
    (64, 8, 4096),
    (16, 8, 8192),
    (32, 8, 8192),
    (64, 8, 8192),
    (16, 8, 16384),
    (32, 8, 16384),
    (64, 8, 16384),
]

_INDEX_BY_KEY = {key: i for i, key in enumerate(_KEYS)}


@functools.cache
def _pick_index(q_heads: int, kv_heads: int, num_tokens: int) -> int:
    target = (q_heads, kv_heads, num_tokens,)
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


def key_fused_qk_norm_rope(*args) -> int:
    """Select config index for the given arguments (also serves as cache key).

    On the per-call specialization-key path, so the pick is memoized on the
    extracted int key via functools.cache to stay O(1) per call.
    """
    q_heads = args[1]
    kv_heads = args[2]
    num_tokens = args[0].shape[0]
    return _pick_index(q_heads, kv_heads, num_tokens)


def autotune_fused_qk_norm_rope(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [64], 'range_unroll_factors': [4], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [True], 'load_eviction_policies': ['', '', 'first', '', '', 'last', '', 'first'], 'num_warps': 1, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 2},
        {'block_sizes': [1], 'loop_orders': [[0, 2, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first', 'first', 'last', '', 'last', ''], 'num_warps': 1, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[2, 0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last', 'first', 'first', 'last', 'last', 'first'], 'num_warps': 2, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', 'first', 'first', 'last', 'first', 'last'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'last', 'first', 'first', 'first', 'last', ''], 'num_warps': 1, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', '', '', 'last', '', '', 'first'], 'num_warps': 1, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', '', '', 'last', 'last', '', 'last'], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', 'last', '', 'first', '', 'first', 'last'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last', 'first', 'last', 'last', 'last', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'first', 'last', 'first', 'first', 'last', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'flat', 'atomic_indexing': [], 'range_warp_specializes': [], 'range_num_stages': []},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last', 'first', 'last', 'last', 'last', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last', 'first', 'last', 'last', 'last', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', '', '', 'last', 'last', '', 'last'], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'last', 'first', 'last', 'last', 'last', ''], 'num_warps': 1, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', '', 'first', 'last', 'last', 'first', 'first'], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', 'first', 'last', '', 'first', 'first'], 'num_warps': 1, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', '', 'first', 'last', '', 'last', 'last'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'flat', 'atomic_indexing': [], 'range_warp_specializes': [], 'range_num_stages': []},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'last', 'last', 'first', 'last', 'first', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', 'last', 'last', 'first', 'first'], 'num_warps': 1, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', 'last', '', 'last', 'last', 'last', 'first'], 'num_warps': 2, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', 'last', 'last', 'last', 'first', 'first', ''], 'num_warps': 2, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', 'last', 'last', 'first', 'first'], 'num_warps': 1, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', 'last', 'last', 'last', ''], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', '', '', 'first', '', 'first', 'first'], 'num_warps': 2, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', 'last', 'last', 'last', ''], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [2], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', 'last', 'first', 'first', 'last', 'first'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 32, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [2], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', 'last', 'first', 'first', 'last', 'first'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 32, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [2], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', 'last', 'first', 'first', 'last', 'first'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 32, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [2], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', 'last', 'first', 'first', 'last', 'first'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 32, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [4], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [True], 'load_eviction_policies': ['first', '', 'first', 'first', 'last', 'last', '', ''], 'num_warps': 1, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 64, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [2], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', 'last', 'first', 'first', 'last', 'first'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 32, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [2], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', 'last', 'first', 'first', 'last', 'first'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 32, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'first', 'last', '', 'last', '', 'first'], 'num_warps': 1, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 128, 'maxnreg': 64, 'atomic_indexing': [], 'range_warp_specializes': []},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [16], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [False], 'load_eviction_policies': ['', '', '', 'last', '', 'last', 'first', ''], 'num_warps': 1, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 64, 'maxnreg': 256},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [True], 'load_eviction_policies': ['', 'first', 'first', 'last', '', '', 'last', 'first'], 'num_warps': 1, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 64, 'maxnreg': 128},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [True], 'load_eviction_policies': ['', 'first', 'first', 'last', '', '', 'last', 'first'], 'num_warps': 1, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 64, 'maxnreg': 128},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [16], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [False], 'load_eviction_policies': ['', '', '', 'last', '', 'last', 'first', ''], 'num_warps': 1, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 64, 'maxnreg': 256},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', '', '', '', 'first'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', '', '', '', 'first'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', '', '', '', 'first'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', '', '', '', 'first'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', '', '', '', 'first'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_multi_buffers': [None], 'range_flattens': [True], 'load_eviction_policies': ['', 'last', 'last', '', 'last', 'last', '', ''], 'num_warps': 1, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128, 'maxnreg': 64, 'atomic_indexing': [], 'range_warp_specializes': []},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', '', '', '', 'first'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128, 'maxnreg': 64},
        {'block_sizes': [2], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [False], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', '', '', '', 'first'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128, 'maxnreg': 64},
    ]
    return _C[key_fused_qk_norm_rope(*args)]
