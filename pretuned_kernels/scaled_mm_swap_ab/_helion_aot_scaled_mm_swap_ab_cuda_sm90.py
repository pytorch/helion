"""
Auto-generated heuristic for kernel: scaled_mm_swap_ab
Backend: decision_tree

Provides:
- key_scaled_mm_swap_ab(*args): Returns config index (cache key)
- autotune_scaled_mm_swap_ab(*args): Returns config dict for the given arguments

Re-tuned on NVIDIA H100 (sm90) for the swapAB FP8 scaled_mm body,
seeded from the shipped non-swap configs. Same (K, N, M) sweep and
nearest-neighbor selection as the non-swap heuristic.
"""

import functools

_KEYS = [
    (2048, 4096, 16),
    (2048, 2048, 16),
    (2048, 12288, 16),
    (6144, 2048, 16),
    (4096, 6144, 16),
    (4096, 4096, 16),
    (4096, 24576, 16),
    (12288, 4096, 16),
    (5120, 10240, 16),
    (5120, 5120, 16),
    (5120, 51200, 16),
    (25600, 5120, 16),
    (2048, 4096, 32),
    (2048, 2048, 32),
    (2048, 12288, 32),
    (6144, 2048, 32),
    (4096, 6144, 32),
    (4096, 4096, 32),
    (4096, 24576, 32),
    (12288, 4096, 32),
    (5120, 10240, 32),
    (5120, 5120, 32),
    (5120, 51200, 32),
    (25600, 5120, 32),
]

_CONFIGS_BY_KN: dict[int, dict[int, list[int]]] = {}
for _kk, _nn, _mm in _KEYS:
    _CONFIGS_BY_KN.setdefault(_kk, {}).setdefault(_nn, []).append(_mm)
for _by_n in _CONFIGS_BY_KN.values():
    for _ms in _by_n.values():
        _ms.sort()
_INDEX_BY_KNM: dict[tuple[int, int, int], int] = {
    key: i for i, key in enumerate(_KEYS)
}


@functools.cache
def _pick_index(M: int, K: int, N: int) -> int:
    exact = _INDEX_BY_KNM.get((K, N, M))
    if exact is not None:
        return exact
    best_K = min(_CONFIGS_BY_KN, key=lambda s: abs(s - K))
    best_N = min(_CONFIGS_BY_KN[best_K], key=lambda s: abs(s - N))
    available_M = _CONFIGS_BY_KN[best_K][best_N]
    best_M = next((m for m in available_M if m >= M), available_M[-1])
    return _INDEX_BY_KNM[(best_K, best_N, best_M)]


def key_scaled_mm_swap_ab(*args) -> int:
    a = args[1]
    b = args[2]
    return _pick_index(int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))


def autotune_scaled_mm_swap_ab(*args) -> dict:
    _C = [
        {'block_sizes': [16, 64, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', 'first', '', '', 'last', 'last'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 256], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'first', 'first', 'last', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 256], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'first', 'last', 'first', 'last', 'first', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 256], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'first', 'last', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'first', 'first', 'first', '', 'last', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'first', 'first', '', '', 'first', 'last'], 'num_warps': 8, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 512], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', 'first', '', '', ''], 'num_warps': 8, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 32, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', 'first', '', '', 'first', 'last'], 'num_warps': 2, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 16, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', '', 'first', 'first', 'first', 'last', 'first'], 'num_warps': 1, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'last', 'first', 'first', 'last', 'last'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 32, 256], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', '', '', 'last', ''], 'num_warps': 2, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 16, 512], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', 'first', 'last', '', 'first', 'last'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 256], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'first', 'last', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [64, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['last', '', '', 'last', 'last', 'first', 'last'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [32, 64, 256], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'first', 'last', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [32, 64, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'first', '', 'first', '', 'last', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 64, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', 'first', '', '', 'first', 'last'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [32, 64, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [4], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'last', 'first', 'last', '', 'first', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 16, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', 'first', '', '', 'first', 'last'], 'num_warps': 1, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [32, 128, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'last', '', 'first', '', 'last', 'last'], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [32, 64, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'last', 'first', 'first', 'last', 'last'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [32, 128, 256], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', '', 'first', '', '', 'last', 'last'], 'num_warps': 2, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [32, 32, 256], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [1, 1], 'range_warp_specializes': [], 'range_multi_buffers': [False, None], 'range_flattens': [False, True], 'load_eviction_policies': ['', '', 'last', 'last', '', 'last', 'last'], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 4, 'maxnreg': 256},
    ]
    return _C[key_scaled_mm_swap_ab(*args)]
