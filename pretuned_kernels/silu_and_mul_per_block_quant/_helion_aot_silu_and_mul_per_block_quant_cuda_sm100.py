"""
Auto-generated heuristic for kernel: silu_and_mul_per_block_quant
Ported from vLLM Helion config silu_and_mul_per_block_quant_b200.json (sm100).

Provides:
- key_silu_and_mul_per_block_quant(*args): Returns config index (cache key)
- autotune_silu_and_mul_per_block_quant(*args): Returns config dict for the given arguments

Config selection mirrors the vLLM config picker: closest match on the
intermediate_size, group_size dimension(s), then the smallest tuned num_tokens >= the input
(falling back to the largest).
"""

import functools

# Tuned key tuples (intermediate_size, group_size, num_tokens), parallel to the config list in autotune_silu_and_mul_per_block_quant.
_KEYS = [
    (6144, 128, 1),
    (12288, 128, 1),
    (25600, 128, 1),
    (6144, 128, 2),
    (12288, 128, 2),
    (25600, 128, 2),
    (6144, 128, 4),
    (12288, 128, 4),
    (25600, 128, 4),
    (6144, 128, 8),
    (12288, 128, 8),
    (25600, 128, 8),
    (6144, 128, 16),
    (12288, 128, 16),
    (25600, 128, 16),
    (6144, 128, 32),
    (12288, 128, 32),
    (25600, 128, 32),
    (6144, 128, 64),
    (12288, 128, 64),
    (25600, 128, 64),
    (6144, 128, 128),
    (12288, 128, 128),
    (25600, 128, 128),
    (6144, 128, 256),
    (12288, 128, 256),
    (25600, 128, 256),
    (6144, 128, 512),
    (12288, 128, 512),
    (25600, 128, 512),
    (6144, 128, 1024),
    (12288, 128, 1024),
    (25600, 128, 1024),
    (6144, 128, 2048),
    (12288, 128, 2048),
    (25600, 128, 2048),
    (6144, 128, 4096),
    (12288, 128, 4096),
    (25600, 128, 4096),
    (6144, 128, 8192),
    (12288, 128, 8192),
    (25600, 128, 8192),
]

_INDEX_BY_KEY = {key: i for i, key in enumerate(_KEYS)}


@functools.cache
def _pick_index(intermediate_size: int, group_size: int, num_tokens: int) -> int:
    target = (intermediate_size, group_size, num_tokens,)
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


def key_silu_and_mul_per_block_quant(*args) -> int:
    """Select config index for the given arguments (also serves as cache key).

    On the per-call specialization-key path, so the pick is memoized on the
    extracted int key via functools.cache to stay O(1) per call.
    """
    out = args[0]
    group_size = args[3]
    num_tokens = out.shape[0]
    intermediate_size = out.shape[1]
    return _pick_index(intermediate_size, group_size, num_tokens)


def autotune_silu_and_mul_per_block_quant(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', ''], 'num_warps': 2, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [32], 'range_unroll_factors': [4], 'range_warp_specializes': [None], 'range_multi_buffers': [False], 'range_flattens': [None], 'load_eviction_policies': ['last', '', ''], 'num_warps': 2, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 4, 'maxnreg': 128},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', ''], 'num_warps': 2, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', ''], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', ''], 'num_warps': 2, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', ''], 'num_warps': 2, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', 'last'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'first'], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'last'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'last'], 'num_warps': 2, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'last'], 'num_warps': 1, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', 'last'], 'num_warps': 1, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last'], 'num_warps': 2, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'last'], 'num_warps': 2, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last'], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', 'last'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last'], 'num_warps': 2, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'last'], 'num_warps': 2, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last'], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last'], 'num_warps': 2, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last'], 'num_warps': 2, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_silu_and_mul_per_block_quant(*args)]
