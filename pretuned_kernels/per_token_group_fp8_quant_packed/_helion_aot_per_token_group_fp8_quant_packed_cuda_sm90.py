"""
Auto-generated heuristic for kernel: per_token_group_fp8_quant_packed
Ported from vLLM Helion config nvidia_h100.json (sm90), autotuned on H100.

Config selection mirrors the vLLM config picker: closest match on the
hidden_size, group_size dimension(s), then the smallest tuned num_tokens >= the
input (falling back to the largest).
"""

import functools

# Tuned key tuples (hidden_size, group_size, num_tokens), parallel to the config list in autotune_per_token_group_fp8_quant_packed.
_KEYS = [
    (2048, 128, 4),
    (2048, 128, 8),
    (2048, 128, 16),
    (2048, 128, 32),
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


def key_per_token_group_fp8_quant_packed(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    a = args[0]
    hidden_size = a.shape[1]
    group_size = args[3]
    num_tokens = a.shape[0]
    return _pick_index(hidden_size, group_size, num_tokens)


def autotune_per_token_group_fp8_quant_packed(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first', ''], 'num_warps': 1, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
        {'block_sizes': [1], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [1], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_multi_buffers': [None], 'range_flattens': [False], 'load_eviction_policies': ['', '', 'last', 'last'], 'num_warps': 1, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
        {'block_sizes': [2], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first', 'last'], 'num_warps': 2, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', 'first', 'last'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_per_token_group_fp8_quant_packed(*args)]
