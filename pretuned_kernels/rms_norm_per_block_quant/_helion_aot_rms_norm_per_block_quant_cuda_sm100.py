"""
Auto-generated heuristic for kernel: rms_norm_per_block_quant
Ported from vLLM Helion config nvidia_b200.json (sm100).

Provides:
- key_rms_norm_per_block_quant(*args): Returns config index (cache key)
- autotune_rms_norm_per_block_quant(*args): Returns config dict for the given arguments

Config selection mirrors the vLLM config picker: closest match on the
hidden_size, group_size dimension(s), then the smallest tuned num_tokens >= the input
(falling back to the largest).
"""

import functools

# Tuned key tuples (hidden_size, group_size, num_tokens), parallel to the config list in autotune_rms_norm_per_block_quant.
_KEYS = [
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
    (5120, 128, 16),
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


def key_rms_norm_per_block_quant(*args) -> int:
    """Select config index for the given arguments (also serves as cache key).

    On the per-call specialization-key path, so the pick is memoized on the
    extracted int key via functools.cache to stay O(1) per call.
    """
    input = args[1]
    group_size = args[7]
    num_tokens = input.shape[0]
    hidden_size = input.shape[1]
    return _pick_index(hidden_size, group_size, num_tokens)


def autotune_rms_norm_per_block_quant(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [2048, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 0, 0], 'range_warp_specializes': [None, False, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, True, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', '', '', 'last', 'last', 'last'], 'num_warps': 16, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 0, 0], 'range_warp_specializes': [None, False, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, True, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', '', '', 'last', 'last', 'last'], 'num_warps': 16, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 0, 0], 'range_warp_specializes': [None, False, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, True, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', '', '', 'last', 'last', 'last'], 'num_warps': 16, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 0, 1, 0], 'range_warp_specializes': [None, True, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, True, None, None], 'range_flattens': [None, True, False, None], 'static_ranges': [True], 'load_eviction_policies': ['first', '', 'first', 'first', 'first', 'first'], 'num_warps': 8, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 4, 2], 'range_warp_specializes': [None, None, None, False], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, True, None, False], 'static_ranges': [False], 'load_eviction_policies': ['last', 'first', 'last', 'last', '', 'last'], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 2, 0], 'range_warp_specializes': [None, None, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, False, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', 'first', 'last', 'last', '', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 0, 0], 'range_warp_specializes': [None, False, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, True, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', '', '', 'last', 'last', 'last'], 'num_warps': 16, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 0, 0], 'range_warp_specializes': [None, False, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, True, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', '', '', 'last', 'last', 'last'], 'num_warps': 16, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 1, 0, 2], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, True, False, True], 'range_flattens': [None, True, True, False], 'static_ranges': [False], 'load_eviction_policies': ['', 'last', 'first', '', 'first', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 8], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 0, 0], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, None, None], 'range_flattens': [None, True, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', 'first', 'last', 'last', '', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 0, 0], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, True, None], 'range_flattens': [None, True, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', 'first', 'last', 'last', '', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 2, 4, 0], 'range_warp_specializes': [None, False, False, False], 'range_num_stages': [], 'range_multi_buffers': [None, False, None, True], 'range_flattens': [None, True, True, False], 'static_ranges': [False], 'load_eviction_policies': ['', '', '', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 16], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 4, 4, 0], 'range_warp_specializes': [None, False, False, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, None, True, None], 'static_ranges': [True], 'load_eviction_policies': ['', '', '', 'first', 'first', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 8], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 4, 4, 2], 'range_warp_specializes': [None, False, False, False], 'range_num_stages': [], 'range_multi_buffers': [None, None, True, True], 'range_flattens': [None, False, False, None], 'static_ranges': [False], 'load_eviction_policies': ['', '', '', 'first', '', ''], 'num_warps': 2, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 3, 0], 'range_warp_specializes': [None, False, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, None, False, None], 'static_ranges': [True], 'load_eviction_policies': ['', 'first', 'last', '', '', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 3, 0], 'range_warp_specializes': [None, False, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, None, False, None], 'static_ranges': [True], 'load_eviction_policies': ['', 'first', 'last', '', '', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 0, 1, 4], 'range_warp_specializes': [None, None, False, False], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, None, None, False], 'static_ranges': [False], 'load_eviction_policies': ['first', '', 'first', 'first', '', ''], 'num_warps': 32, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 1, 0], 'range_warp_specializes': [None, False, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, False, None], 'range_flattens': [None, None, True, None], 'static_ranges': [True], 'load_eviction_policies': ['first', 'last', 'first', 'first', 'first', 'last'], 'num_warps': 32, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 2, 3, 0], 'range_warp_specializes': [None, None, False, True], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, True, False, False], 'static_ranges': [False], 'load_eviction_policies': ['last', 'first', '', '', '', 'first'], 'num_warps': 16, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 0, 1, 0], 'range_warp_specializes': [None, True, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, False, None], 'range_flattens': [None, True, False, False], 'static_ranges': [False], 'load_eviction_policies': ['first', '', 'first', 'first', 'first', ''], 'num_warps': 32, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 4, 4, 0], 'range_warp_specializes': [None, None, False, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, True, None], 'range_flattens': [None, None, False, None], 'static_ranges': [True], 'load_eviction_policies': ['', 'last', 'first', 'first', '', 'first'], 'num_warps': 32, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 0, 1, 0], 'range_warp_specializes': [None, False, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, True, None], 'range_flattens': [None, None, None, None], 'static_ranges': [True], 'load_eviction_policies': ['', 'first', 'last', '', 'last', 'last'], 'num_warps': 32, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 0, 3, 2], 'range_warp_specializes': [None, True, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, True, True, True], 'range_flattens': [None, None, None, False], 'static_ranges': [False], 'load_eviction_policies': ['first', '', 'first', '', 'last', 'last'], 'num_warps': 32, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 32], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 0, 4, 2], 'range_warp_specializes': [None, True, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, False, True], 'range_flattens': [None, True, True, None], 'static_ranges': [False], 'load_eviction_policies': ['last', 'first', '', 'first', 'last', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 0, 0, 0], 'range_warp_specializes': [None, False, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, False, None, None], 'static_ranges': [True], 'load_eviction_policies': ['first', '', '', 'first', 'first', 'last'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 2, 0], 'range_warp_specializes': [None, False, None, True], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, False, False, False], 'static_ranges': [False], 'load_eviction_policies': ['', '', 'first', 'first', 'last', ''], 'num_warps': 2, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 16], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 4, 0, 1], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, True, True, False], 'range_flattens': [None, False, False, False], 'static_ranges': [False], 'load_eviction_policies': ['last', 'last', 'first', '', 'last', ''], 'num_warps': 2, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4096, 16], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 2, 2, 0], 'range_warp_specializes': [None, False, None, True], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, False, False, False], 'static_ranges': [False], 'load_eviction_policies': ['', '', 'first', 'first', 'last', ''], 'num_warps': 2, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 64], 'loop_orders': [[0, 1]], 'range_unroll_factors': [3, 1, 1, 0], 'range_warp_specializes': [False, False, None, None], 'range_multi_buffers': [True, True, None, None], 'range_flattens': [None, None, True, None], 'static_ranges': [True], 'load_eviction_policies': ['first', 'last', '', 'last', '', ''], 'num_warps': 16, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 8, 'maxnreg': 128},
        {'block_sizes': [8192, 64], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 0, 0, 4], 'range_warp_specializes': [None, None, None, False], 'range_num_stages': [], 'range_multi_buffers': [None, False, True, None], 'range_flattens': [None, False, False, False], 'static_ranges': [False], 'load_eviction_policies': ['', 'first', 'first', '', 'first', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 64], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 4, 0], 'range_warp_specializes': [None, None, False, None], 'range_num_stages': [], 'range_multi_buffers': [None, True, None, None], 'range_flattens': [None, None, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', 'first', 'last', 'last', '', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 64], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 2, 0, 0], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, None, False, None], 'static_ranges': [True], 'load_eviction_policies': ['last', 'first', 'last', 'first', '', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 64], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 0, 0, 4], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, True, None], 'range_flattens': [None, False, False, False], 'static_ranges': [False], 'load_eviction_policies': ['', 'first', '', '', 'first', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 64], 'loop_orders': [[0, 1]], 'range_unroll_factors': [3, 1, 1, 0], 'range_warp_specializes': [False, False, None, None], 'range_multi_buffers': [True, True, None, None], 'range_flattens': [None, None, True, None], 'static_ranges': [True], 'load_eviction_policies': ['first', 'last', '', 'last', '', ''], 'num_warps': 16, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 8, 'maxnreg': 128},
        {'block_sizes': [8192, 64], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 0, 0], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, None, True, None], 'static_ranges': [True], 'load_eviction_policies': ['last', 'first', 'last', 'last', '', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 64], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 1, 3, 0], 'range_warp_specializes': [None, False, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, True, None, None], 'range_flattens': [None, False, True, None], 'static_ranges': [True], 'load_eviction_policies': ['first', 'last', '', 'last', '', ''], 'num_warps': 16, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 8], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 1, 4, 0], 'range_warp_specializes': [None, True, None, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, True, False, None], 'static_ranges': [True], 'load_eviction_policies': ['first', 'first', 'last', '', 'first', 'first'], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 8], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 0, 0], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, True, None, None], 'range_flattens': [None, None, True, None], 'static_ranges': [True], 'load_eviction_policies': ['last', 'first', 'first', 'first', '', 'last'], 'num_warps': 1, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 8], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 4, 0, 0], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, False, None], 'range_flattens': [None, True, True, None], 'static_ranges': [True], 'load_eviction_policies': ['', '', 'first', 'last', 'last', ''], 'num_warps': 1, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2048, 8], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 4, 1, 3], 'range_warp_specializes': [None, None, False, False], 'range_num_stages': [], 'range_multi_buffers': [None, True, True, False], 'range_flattens': [None, True, True, None], 'static_ranges': [False], 'load_eviction_policies': ['', '', 'first', 'first', '', 'last'], 'num_warps': 2, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 8], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 0, 2], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, None, False, False], 'static_ranges': [False], 'load_eviction_policies': ['', '', 'first', 'last', '', 'last'], 'num_warps': 1, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [8192, 8], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 0, 2], 'range_warp_specializes': [None, None, True, None], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, None, False, False], 'static_ranges': [False], 'load_eviction_policies': ['', '', 'first', 'last', '', 'last'], 'num_warps': 1, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_rms_norm_per_block_quant(*args)]
