"""
Pretuned H100 (sm90) configs for the Helion linear-attention kernels.

Tuned with HELION_AUTOTUNE_EFFORT=full over the benchmarks/run_linattn.py SHAPES
and VARLEN_SHAPES sweeps; each call keeps its own best config.

A call is keyed on its tensor arguments' shapes together with its None and bool
arguments, which choose a code path.

An unseen call falls back to the nearest tuned call with the same None/bool
arguments, by total element count.
"""

import math

import torch


def _call_key(args):
    """(tensor shapes, None/bool arguments) for the given call."""
    shapes = tuple(
        tuple(int(d) for d in a.shape) for a in args if isinstance(a, torch.Tensor)
    )
    flags = tuple(repr(a) for a in args if a is None or isinstance(a, bool))
    return shapes, flags


def _select(keys, args):
    """Exact match if tuned, else the nearest same-flags call by element count."""
    key = _call_key(args)
    for i, k in enumerate(keys):
        if k == key:
            return i
    shapes, flags = key
    target = sum(math.prod(s) for s in shapes)
    same = [i for i, k in enumerate(keys) if k[1] == flags] or range(len(keys))
    return min(same, key=lambda i: abs(sum(math.prod(s) for s in keys[i][0]) - target))


_KEYS_chunk_bwd_dh_diag_fused = [
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 128, 128)), ('None', 'None', 'False', 'False', 'False')),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 64, 64)), ('None', 'None', 'False', 'False', 'False')),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 128, 128)), ('None', 'None', 'False', 'False', 'False')),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128)), ('None', 'None', 'False', 'False', 'False')),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 256, 256)), ('None', 'None', 'False', 'False', 'False')),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 128, 128)), ('None', 'None', 'False', 'False', 'False')),
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 128), (32, 128, 128), (32, 256, 64, 128)), ('True', 'False', 'True')),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64), (64, 64, 64), (64, 16, 64, 64)), ('True', 'False', 'True')),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 128), (64, 128, 128), (64, 32, 64, 128)), ('True', 'False', 'True')),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128), (96, 128, 128), (96, 128, 64, 128)), ('True', 'False', 'True')),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 256), (256, 256, 256), (256, 32, 64, 256)), ('True', 'False', 'True')),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 128), (256, 128, 128), (256, 64, 64, 128)), ('True', 'False', 'True')),
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256), (32, 128, 128), (32, 256, 64)), ('True', 'True', 'False')),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16), (64, 64, 64), (64, 16, 64)), ('True', 'True', 'False')),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32), (64, 128, 128), (64, 32, 64)), ('True', 'True', 'False')),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128), (96, 128, 128), (96, 128, 64)), ('True', 'True', 'False')),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32), (256, 256, 256), (256, 32, 64)), ('True', 'True', 'False')),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64), (256, 128, 128), (256, 64, 64)), ('True', 'True', 'False')),
]

_CONFIGS_chunk_bwd_dh_diag_fused = [
    {'block_sizes': [64, 64], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    {'block_sizes': [32, 64], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [8], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'last', 'first'], 'num_warps': 8, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    {'block_sizes': [64, 128], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', '', ''], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [64, 64], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'first', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'first', 'first'], 'num_warps': 16, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [1], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'first', 'first'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 128], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [4], 'range_unroll_factors': [1, 2], 'range_warp_specializes': [], 'range_num_stages': [4, 3], 'range_multi_buffers': [False, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', 'first', 'last', 'first'], 'num_warps': 16, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 32, 'maxnreg': 128},
    {'block_sizes': [32, 64], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'first', 'last', 'last'], 'num_warps': 16, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 128], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [64], 'range_unroll_factors': [2, 2], 'range_warp_specializes': [], 'range_num_stages': [3, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['last', '', 'last', '', ''], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1, 'maxnreg': 128},
    {'block_sizes': [128, 128], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [4], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'last', '', '', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 256], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last', '', 'last', 'last'], 'num_warps': 16, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [64], 'range_unroll_factors': [2, 4], 'range_warp_specializes': [], 'range_num_stages': [3, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', '', '', '', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [64, 32], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'first', '', '', 'last'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 64], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'last', 'last', 'first', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 64], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'last', '', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    {'block_sizes': [64, 64], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'first', 'last', 'first', ''], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'first', '', 'first', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[0, 2, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last', '', 'last', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_dh_diag_fused(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_dh_diag_fused, args)


def autotune_chunk_bwd_dh_diag_fused(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_dh_diag_fused[key_chunk_bwd_dh_diag_fused(*args)]


_KEYS_chunk_bwd_dk_delta_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64), (1024, 64, 64)), ()),
    (((2048, 64, 128), (2048, 64, 64), (2048, 64, 128), (2048, 64), (2048, 64, 128)), ()),
    (((8192, 64, 128), (8192, 64, 64), (8192, 64, 128), (8192, 64), (8192, 64, 128)), ()),
    (((8192, 64, 256), (8192, 64, 64), (8192, 64, 256), (8192, 64), (8192, 64, 256)), ()),
    (((12288, 64, 128), (12288, 64, 64), (12288, 64, 128), (12288, 64), (12288, 64, 128)), ()),
    (((16384, 64, 128), (16384, 64, 64), (16384, 64, 128), (16384, 64), (16384, 64, 128)), ()),
]

_CONFIGS_chunk_bwd_dk_delta_helion = [
    {'block_sizes': [64], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['first', '', 'first', '', '', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', '', '', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['last', '', 'last', 'last', 'first', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', 'first', 'first', 'first', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_dk_delta_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_dk_delta_helion, args)


def autotune_chunk_bwd_dk_delta_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_dk_delta_helion[key_chunk_bwd_dk_delta_helion(*args)]


_KEYS_chunk_bwd_dqk_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ('None', 'None', 'False')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 128, 128), (2048, 64, 128), (2048, 128, 128)), ('None', 'None', 'False')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 128, 128), (8192, 64, 128), (8192, 128, 128)), ('None', 'None', 'False')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 256, 256), (8192, 64, 256), (8192, 256, 256)), ('None', 'None', 'False')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 128, 128), (12288, 64, 128), (12288, 128, 128)), ('None', 'None', 'False')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 128, 128), (16384, 64, 128), (16384, 128, 128)), ('None', 'None', 'False')),
]

_CONFIGS_chunk_bwd_dqk_helion = [
    {'block_sizes': [1, 64, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'first', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'last', 'first', 'first', 'last', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['last', '', 'last', 'first', 'last', 'first'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'first', '', 'first', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 32], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'first', 'first', 'first', '', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last', 'last', 'first', '', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_dqk_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_dqk_helion, args)


def autotune_chunk_bwd_dqk_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_dqk_helion[key_chunk_bwd_dqk_helion(*args)]


_KEYS_chunk_bwd_dqkg_scalar_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024,)), ('False', 'False')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64), (2048, 128, 128), (2048, 64, 128), (2048, 128, 128), (2048,)), ('False', 'False')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64), (8192, 128, 128), (8192, 64, 128), (8192, 128, 128), (8192,)), ('False', 'False')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64), (8192, 256, 256), (8192, 64, 256), (8192, 256, 256), (8192,)), ('False', 'False')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64), (12288, 128, 128), (12288, 64, 128), (12288, 128, 128), (12288,)), ('False', 'False')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64), (16384, 128, 128), (16384, 64, 128), (16384, 128, 128), (16384,)), ('False', 'False')),
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ('None', 'True', 'True')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 128, 128), (2048, 64, 128), (2048, 128, 128)), ('None', 'True', 'True')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 128, 128), (8192, 64, 128), (8192, 128, 128)), ('None', 'True', 'True')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 256, 256), (8192, 64, 256), (8192, 256, 256)), ('None', 'True', 'True')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 128, 128), (12288, 64, 128), (12288, 128, 128)), ('None', 'True', 'True')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 128, 128), (16384, 64, 128), (16384, 128, 128)), ('None', 'True', 'True')),
]

_CONFIGS_chunk_bwd_dqkg_scalar_helion = [
    {'block_sizes': [1, 64, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'last', 'first', 'first', 'first', 'first', 'last', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['first', '', 'first', 'last', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 32], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'last', 'last', '', '', 'last', '', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 32], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['first', 'last', 'last', 'first', '', '', 'last', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last', 'first', 'last', 'first', '', '', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['first', 'last', 'last', 'last', '', '', '', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', '', 'first', '', '', '', 'last'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 32, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last', '', 'last', 'first', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 32, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'last', 'first', 'last', 'first', '', '', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 32, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['', '', '', 'last', '', '', 'last', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 32, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'last', 'last', 'last', 'first', 'last', 'last', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 32, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['first', '', '', 'last', '', '', '', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_dqkg_scalar_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_dqkg_scalar_helion, args)


def autotune_chunk_bwd_dqkg_scalar_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_dqkg_scalar_helion[key_chunk_bwd_dqkg_scalar_helion(*args)]


_KEYS_chunk_bwd_dqkw_delta_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ('None', 'None', 'False')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 128, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 128, 128)), ('None', 'None', 'False')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 128, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 128, 128)), ('None', 'None', 'False')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 256, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 256, 256)), ('None', 'None', 'False')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 128, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 128, 128)), ('None', 'None', 'False')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 128, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 128, 128)), ('None', 'None', 'False')),
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64), (1024,)), ('True',)),
    (((2048, 64, 128), (2048, 64, 128), (2048, 128, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 128, 128), (2048, 64), (2048,)), ('True',)),
    (((8192, 64, 128), (8192, 64, 128), (8192, 128, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 128, 128), (8192, 64), (8192,)), ('True',)),
    (((8192, 64, 256), (8192, 64, 256), (8192, 256, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 256, 256), (8192, 64), (8192,)), ('True',)),
    (((12288, 64, 128), (12288, 64, 128), (12288, 128, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 128, 128), (12288, 64), (12288,)), ('True',)),
    (((16384, 64, 128), (16384, 64, 128), (16384, 128, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 128, 128), (16384, 64), (16384,)), ('True',)),
]

_CONFIGS_chunk_bwd_dqkw_delta_helion = [
    {'block_sizes': [64, 64, 64], 'range_unroll_factors': [0, 3, 2, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 4, 3], 'range_multi_buffers': [None, False, False, None], 'range_flattens': [None, None, False, True], 'load_eviction_policies': ['', 'last', 'first', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 64, 64], 'range_unroll_factors': [0, 4, 4, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 2, 4], 'range_multi_buffers': [None, True, False, False], 'range_flattens': [None, None, False, None], 'load_eviction_policies': ['last', 'last', 'first', 'first', '', 'first', 'first', 'last', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 64, 128], 'range_unroll_factors': [0, 4, 2, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 3, 0], 'range_multi_buffers': [None, True, True, None], 'range_flattens': [None, None, False, True], 'load_eviction_policies': ['last', 'last', 'first', 'first', 'first', 'last', '', '', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128, 64], 'range_unroll_factors': [0, 0, 3, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 2, 4], 'range_multi_buffers': [None, True, False, None], 'range_flattens': [None, True, None, True], 'load_eviction_policies': ['last', '', 'first', 'first', 'last', '', '', '', 'first'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 64, 32], 'range_unroll_factors': [0, 3, 2, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 1, 2], 'range_multi_buffers': [None, None, True, False], 'range_flattens': [None, None, None, False], 'load_eviction_policies': ['first', 'last', 'first', 'first', '', '', 'last', 'first', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 64, 64], 'range_unroll_factors': [4, 2, 2, 0], 'range_warp_specializes': [], 'range_num_stages': [1, 1, 2, 1], 'range_multi_buffers': [None, True, True, None], 'range_flattens': [False, None, False, False], 'load_eviction_policies': ['first', 'last', 'first', 'first', '', 'last', 'last', 'last', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 128},
    {'block_sizes': [64, 64, 64, 32, 64], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 2, 4, 0, 3, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 1, 1, 3, 2], 'range_multi_buffers': [None, True, True, None, True, True], 'range_flattens': [None, False, False, False, False, True], 'load_eviction_policies': ['last', 'first', 'last', 'last', 'last', 'last', 'last', 'first', 'last', 'first', 'first', 'first', 'first', 'last', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 64, 32, 128, 128], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 0, 3, 0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 3, 3, 4, 3], 'range_multi_buffers': [None, False, None, True, True, True], 'range_flattens': [None, False, None, True, False, False], 'load_eviction_policies': ['last', 'first', '', 'last', 'last', 'last', 'last', '', 'first', 'last', 'first', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 32, 128, 128, 128], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 0, 0, 0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 0, 0, 0, 0], 'range_multi_buffers': [None, None, None, None, None, None], 'range_flattens': [None, None, None, None, None, False], 'load_eviction_policies': ['first', '', '', '', '', '', '', '', '', '', '', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128, 32, 256, 128], 'loop_orders': [[1, 0]], 'range_unroll_factors': [0, 3, 2, 2, 1, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 4, 2, 3, 3], 'range_multi_buffers': [None, None, None, True, None, None], 'range_flattens': [None, False, False, True, True, True], 'load_eviction_policies': ['last', '', 'last', '', '', 'last', 'last', 'first', 'first', '', 'last', 'first', 'first', 'first', 'last'], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 64, 32, 128, 128], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 0, 0, 3, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 0, 0, 4, 0], 'range_multi_buffers': [None, True, None, None, None, None], 'range_flattens': [None, True, None, None, None, True], 'load_eviction_policies': ['', '', 'first', 'last', 'last', 'last', 'first', 'first', 'first', 'first', '', 'last', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 64, 32, 128, 128], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 3, 2, 1, 3, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 0, 0, 3, 0], 'range_multi_buffers': [None, False, True, False, False, None], 'range_flattens': [None, None, None, False, None, None], 'load_eviction_policies': ['first', '', 'first', '', 'last', 'last', '', '', 'first', 'first', '', 'first', '', '', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_dqkw_delta_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_dqkw_delta_helion, args)


def autotune_chunk_bwd_dqkw_delta_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_dqkw_delta_helion[key_chunk_bwd_dqkw_delta_helion(*args)]


_KEYS_chunk_bwd_dstate_delta_helion = [
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 128, 128)), ('None', 'None', 'False')),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 64, 64)), ('None', 'None', 'False')),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 128, 128)), ('None', 'None', 'False')),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128)), ('None', 'None', 'False')),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 256, 256)), ('None', 'None', 'False')),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 128, 128)), ('None', 'None', 'False')),
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 128, 128), (32, 256, 64), (32, 256)), ('True',)),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 64, 64), (64, 16, 64), (64, 16)), ('True',)),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 128, 128), (64, 32, 64), (64, 32)), ('True',)),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128), (96, 128, 64), (96, 128)), ('True',)),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 256, 256), (256, 32, 64), (256, 32)), ('True',)),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 128, 128), (256, 64, 64), (256, 64)), ('True',)),
]

_CONFIGS_chunk_bwd_dstate_delta_helion = [
    {'block_sizes': [32], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['first', '', 'first', 'last', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['last', '', 'last', 'last', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', '', '', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [64], 'loop_orders': [[0, 1]], 'l2_groupings': [4], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'last', 'first', 'first', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [32], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'last', 'first', '', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [1, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', 'last', 'first'], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [2, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [False, False], 'range_flattens': [False, None], 'load_eviction_policies': ['last', 'first', 'first', 'last', '', 'last', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1, 'maxnreg': 256},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'last', '', 'first', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    {'block_sizes': [64], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', '', 'first', 'first', '', 'last', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [2, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [False, True], 'range_flattens': [False, True], 'load_eviction_policies': ['first', 'last', 'first', 'first', '', '', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1, 'maxnreg': 256},
    {'block_sizes': [32], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', '', 'last', 'last', 'last', 'first', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1, 'maxnreg': 256},
    {'block_sizes': [64], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [2, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [False, True], 'range_flattens': [False, False], 'load_eviction_policies': ['first', 'first', 'first', 'first', '', 'last', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 2, 'maxnreg': 256},
]


def key_chunk_bwd_dstate_delta_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_dstate_delta_helion, args)


def autotune_chunk_bwd_dstate_delta_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_dstate_delta_helion[key_chunk_bwd_dstate_delta_helion(*args)]


_KEYS_chunk_bwd_dv_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 128, 128)), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 128, 128)), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 256, 256)), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 128, 128)), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 128, 128)), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ('None', 'True', 'False', 'True')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 128, 128), (2048, 64, 64)), ('None', 'True', 'False', 'True')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 128, 128), (8192, 64, 64)), ('None', 'True', 'False', 'True')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 256, 256), (8192, 64, 64)), ('None', 'True', 'False', 'True')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 128, 128), (12288, 64, 64)), ('None', 'True', 'False', 'True')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 128, 128), (16384, 64, 64)), ('None', 'True', 'False', 'True')),
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64), (1024, 64, 64), (1024, 64, 64), (1024,)), ('None', 'True', 'True', 'False')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64), (2048, 64, 128), (2048, 128, 128), (2048,)), ('None', 'True', 'True', 'False')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64), (8192, 64, 128), (8192, 128, 128), (8192,)), ('None', 'True', 'True', 'False')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64), (8192, 64, 256), (8192, 256, 256), (8192,)), ('None', 'True', 'True', 'False')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64), (12288, 64, 128), (12288, 128, 128), (12288,)), ('None', 'True', 'True', 'False')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64), (16384, 64, 128), (16384, 128, 128), (16384,)), ('None', 'True', 'True', 'False')),
]

_CONFIGS_chunk_bwd_dv_helion = [
    {'block_sizes': [1, 64, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'first', '', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'first', 'first', 'last', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 256, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last', 'last', 'last', 'first'], 'num_warps': 8, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['last', '', '', 'last', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'first', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['first', 'first', 'first', 'last', 'first', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'first', 'first', 'first', '', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last', '', 'last', 'first', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 256, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', '', '', 'last', 'first', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'first', '', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'first', '', '', '', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['first', 'first', 'first', '', '', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['first', 'first', '', 'last', 'first', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'first', 'first', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 256, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'last', '', 'last', '', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', 'first', 'first', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last', '', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_dv_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_dv_helion, args)


def autotune_chunk_bwd_dv_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_dv_helion[key_chunk_bwd_dv_helion(*args)]


_KEYS_chunk_bwd_gram2_kda_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ()),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64, 64), (2048, 64, 64)), ()),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64, 64), (8192, 64, 64)), ()),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64, 64), (8192, 64, 64)), ()),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64, 64), (12288, 64, 64)), ()),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64, 64), (16384, 64, 64)), ()),
]

_CONFIGS_chunk_bwd_gram2_kda_helion = [
    {'block_sizes': [32], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 3, 0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 0, 4], 'range_multi_buffers': [None, False, True, True], 'range_flattens': [None, None, None, True], 'load_eviction_policies': ['first', 'first', 'last', '', '', 'first', 'first', 'first', 'last', 'last', 'last', 'last', 'last', 'first', '', '', 'last', '', 'last', 'first', 'first', ''], 'num_warps': 1, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4, 1, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 0, 1], 'range_multi_buffers': [None, True, True, True], 'range_flattens': [None, False, None, True], 'load_eviction_policies': ['', 'first', '', '', '', 'last', 'first', 'first', '', 'last', 'first', 'last', 'last', 'first', 'last', 'last', 'first', 'last', 'first', '', 'first', ''], 'num_warps': 1, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4, 4, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 1, 3, 2], 'range_multi_buffers': [None, False, True, False], 'range_flattens': [None, False, None, False], 'load_eviction_policies': ['first', 'first', 'first', 'first', 'last', 'last', 'first', 'first', '', 'last', 'last', 'last', 'last', 'first', 'first', 'last', '', 'first', 'last', '', 'first', ''], 'num_warps': 1, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 3, 0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 3, 1, 2], 'range_multi_buffers': [None, None, False, True], 'range_flattens': [None, True, False, False], 'load_eviction_policies': ['last', 'first', 'first', 'first', 'last', '', 'first', '', 'first', 'first', 'last', 'last', 'last', 'first', 'last', 'first', 'first', '', 'first', 'first', 'first', 'first'], 'num_warps': 1, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 4, 4, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 3, 0, 0], 'range_multi_buffers': [None, None, True, None], 'range_flattens': [None, True, True, True], 'load_eviction_policies': ['last', 'first', 'first', 'first', '', '', 'first', 'first', '', 'last', 'last', 'last', 'last', '', 'first', 'last', 'first', 'first', 'last', '', '', ''], 'num_warps': 1, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 1, 4, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 0, 2], 'range_multi_buffers': [None, True, False, None], 'range_flattens': [None, None, True, True], 'load_eviction_policies': ['first', 'first', 'first', '', 'first', 'first', 'first', 'first', 'first', 'last', 'first', 'last', 'last', 'last', '', 'first', 'first', 'last', '', '', 'first', ''], 'num_warps': 1, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_gram2_kda_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_gram2_kda_helion, args)


def autotune_chunk_bwd_gram2_kda_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_gram2_kda_helion[key_chunk_bwd_gram2_kda_helion(*args)]


_KEYS_chunk_bwd_o_kda_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ()),
    (((2048, 64, 128), (2048, 64, 128), (2048, 128, 128), (2048, 64, 64), (2048, 64, 128), (2048, 64, 128)), ()),
    (((8192, 64, 128), (8192, 64, 128), (8192, 128, 128), (8192, 64, 64), (8192, 64, 128), (8192, 64, 128)), ()),
    (((8192, 64, 256), (8192, 64, 256), (8192, 256, 256), (8192, 64, 64), (8192, 64, 256), (8192, 64, 256)), ()),
    (((12288, 64, 128), (12288, 64, 128), (12288, 128, 128), (12288, 64, 64), (12288, 64, 128), (12288, 64, 128)), ()),
    (((16384, 64, 128), (16384, 64, 128), (16384, 128, 128), (16384, 64, 64), (16384, 64, 128), (16384, 64, 128)), ()),
]

_CONFIGS_chunk_bwd_o_kda_helion = [
    {'block_sizes': [32, 64, 64], 'range_unroll_factors': [0, 2, 2, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 2, 1], 'range_multi_buffers': [None, True, None, None], 'range_flattens': [None, None, True, None], 'load_eviction_policies': ['last', 'last', '', 'first', 'first', '', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 64, 128], 'range_unroll_factors': [1, 2, 0, 2], 'range_warp_specializes': [], 'range_num_stages': [2, 4, 0, 4], 'range_multi_buffers': [False, None, True, True], 'range_flattens': [None, True, None, None], 'load_eviction_policies': ['', 'last', 'last', 'last', '', 'first', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 32, 'maxnreg': 256},
    {'block_sizes': [128, 64, 128], 'range_unroll_factors': [0, 1, 3, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 0, 4], 'range_multi_buffers': [None, None, None, True], 'range_flattens': [None, True, None, None], 'load_eviction_policies': ['last', 'last', 'first', '', '', 'last', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128, 64], 'range_unroll_factors': [0, 0, 2, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 3, 1], 'range_multi_buffers': [None, None, None, False], 'range_flattens': [None, False, None, None], 'load_eviction_policies': ['last', 'last', 'first', 'last', 'last', 'first', 'last'], 'num_warps': 16, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 64, 128], 'range_unroll_factors': [0, 4, 4, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 2, 4], 'range_multi_buffers': [None, False, None, True], 'range_flattens': [None, False, False, None], 'load_eviction_policies': ['', 'last', 'last', '', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 64, 128], 'range_unroll_factors': [0, 0, 0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 4, 1, 3], 'range_multi_buffers': [None, False, True, True], 'range_flattens': [None, False, None, False], 'load_eviction_policies': ['', 'last', 'last', '', '', '', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_o_kda_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_o_kda_helion, args)


def autotune_chunk_bwd_o_kda_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_o_kda_helion[key_chunk_bwd_o_kda_helion(*args)]


_KEYS_chunk_bwd_state_du_kda_helion = [
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 128, 128), (32, 128, 128)), ()),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 64, 64)), ()),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 128, 128), (64, 128, 128)), ()),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128, 128), (96, 128, 128)), ()),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 256, 256), (256, 256, 256)), ()),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 128, 128), (256, 128, 128)), ()),
]

_CONFIGS_chunk_bwd_state_du_kda_helion = [
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', '', '', 'first', 'first', 'first'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', '', 'first', 'last', 'first', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[0, 1]], 'l2_groupings': [4], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'last', 'last', 'last', '', 'last', 'last'], 'num_warps': 8, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'last', 'last', 'last', 'last', '', 'last'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['first', '', '', '', '', '', 'first'], 'num_warps': 8, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'last', '', 'last', 'last', 'last', 'last'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_state_du_kda_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_state_du_kda_helion, args)


def autotune_chunk_bwd_state_du_kda_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_state_du_kda_helion[key_chunk_bwd_state_du_kda_helion(*args)]


_KEYS_chunk_bwd_state_dwk_kda_helion = [
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 128, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 128, 128)), ()),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64)), ()),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 128, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 128, 128)), ()),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128, 128)), ()),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 256, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 256, 256)), ()),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 128, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 128, 128)), ()),
]

_CONFIGS_chunk_bwd_state_dwk_kda_helion = [
    {'block_sizes': [1, 64, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last', '', 'first', '', '', 'last'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [4], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['first', '', 'first', '', '', '', ''], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last', '', 'first', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', '', 'last', '', 'first', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['last', '', '', 'last', '', 'last', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', '', 'last', 'first', 'last', 'last'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_state_dwk_kda_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_state_dwk_kda_helion, args)


def autotune_chunk_bwd_state_dwk_kda_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_state_dwk_kda_helion[key_chunk_bwd_state_dwk_kda_helion(*args)]


_KEYS_chunk_bwd_wu_kda_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ()),
    (((2048, 64, 64), (2048, 64, 128), (2048, 64, 128), (2048, 64), (2048, 64, 128), (2048, 64, 64), (2048, 64, 128), (2048, 64, 128)), ()),
    (((8192, 64, 64), (8192, 64, 128), (8192, 64, 128), (8192, 64), (8192, 64, 128), (8192, 64, 64), (8192, 64, 128), (8192, 64, 128)), ()),
    (((8192, 64, 64), (8192, 64, 256), (8192, 64, 256), (8192, 64), (8192, 64, 256), (8192, 64, 64), (8192, 64, 256), (8192, 64, 256)), ()),
    (((12288, 64, 64), (12288, 64, 128), (12288, 64, 128), (12288, 64), (12288, 64, 128), (12288, 64, 64), (12288, 64, 128), (12288, 64, 128)), ()),
    (((16384, 64, 64), (16384, 64, 128), (16384, 64, 128), (16384, 64), (16384, 64, 128), (16384, 64, 64), (16384, 64, 128), (16384, 64, 128)), ()),
]

_CONFIGS_chunk_bwd_wu_kda_helion = [
    {'block_sizes': [32, 64], 'range_unroll_factors': [0, 1, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 0], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, True], 'load_eviction_policies': ['last', 'first', 'first', '', '', 'last', 'last', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16], 'range_unroll_factors': [0, 4, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 2], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['last', 'last', '', '', '', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16], 'range_unroll_factors': [0, 4, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 2], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['last', 'last', '', '', '', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 32], 'range_unroll_factors': [0, 2, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 0], 'range_multi_buffers': [None, None, False], 'range_flattens': [None, None, None], 'load_eviction_policies': ['first', '', 'first', '', 'first', '', '', 'last'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16], 'range_unroll_factors': [0, 4, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 2], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['last', 'last', '', '', '', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16], 'range_unroll_factors': [0, 4, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 2], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['last', 'last', '', '', '', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_wu_kda_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_wu_kda_helion, args)


def autotune_chunk_bwd_wu_kda_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_wu_kda_helion[key_chunk_bwd_wu_kda_helion(*args)]


_KEYS_chunk_bwd_wy_dL_delta_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ('None', 'False')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64), (2048, 64, 64), (2048, 64, 128), (2048, 64, 128)), ('None', 'False')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64), (8192, 64, 64), (8192, 64, 128), (8192, 64, 128)), ('None', 'False')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64), (8192, 64, 64), (8192, 64, 256), (8192, 64, 256)), ('None', 'False')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64), (12288, 64, 64), (12288, 64, 128), (12288, 64, 128)), ('None', 'False')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64), (16384, 64, 64), (16384, 64, 128), (16384, 64, 128)), ('None', 'False')),
    (((1024, 64, 64), (1024, 64, 64), (1024, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64)), ('True',)),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64), (2048, 64, 64), (2048, 64, 128), (2048, 64, 128), (2048, 64)), ('True',)),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64), (8192, 64, 64), (8192, 64, 128), (8192, 64, 128), (8192, 64)), ('True',)),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64), (8192, 64, 64), (8192, 64, 256), (8192, 64, 256), (8192, 64)), ('True',)),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64), (12288, 64, 64), (12288, 64, 128), (12288, 64, 128), (12288, 64)), ('True',)),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64), (16384, 64, 64), (16384, 64, 128), (16384, 64, 128), (16384, 64)), ('True',)),
]

_CONFIGS_chunk_bwd_wy_dL_delta_helion = [
    {'block_sizes': [64, 32, 16], 'range_unroll_factors': [4, 1, 3, 0], 'range_warp_specializes': [], 'range_num_stages': [4, 2, 1, 1], 'range_multi_buffers': [True, True, True, True], 'range_flattens': [None, False, False, True], 'load_eviction_policies': ['', '', '', 'first', '', 'last', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 8, 'maxnreg': 256},
    {'block_sizes': [32, 32, 32], 'range_unroll_factors': [0, 2, 2, 3], 'range_warp_specializes': [], 'range_num_stages': [1, 2, 4, 2], 'range_multi_buffers': [True, True, False, True], 'range_flattens': [False, True, None, False], 'load_eviction_policies': ['', 'last', '', '', 'last', 'last', 'last'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 16},
    {'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 1, 1, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 1, 4, 0], 'range_multi_buffers': [None, None, None, False], 'range_flattens': [None, None, None, None], 'load_eviction_policies': ['', 'first', '', '', '', 'first', 'last'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 32, 256], 'range_unroll_factors': [0, 2, 1, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2, 0, 4], 'range_multi_buffers': [None, None, None, True], 'range_flattens': [None, False, None, False], 'load_eviction_policies': ['first', '', '', 'first', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 1, 1, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 1, 4, 0], 'range_multi_buffers': [None, None, None, False], 'range_flattens': [None, None, None, None], 'load_eviction_policies': ['', 'first', '', '', '', 'first', 'last'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 1, 1, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 1, 4, 0], 'range_multi_buffers': [None, None, None, False], 'range_flattens': [None, None, None, None], 'load_eviction_policies': ['', 'first', '', '', '', 'first', 'last'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 2, 4, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 3, 0, 2], 'range_multi_buffers': [None, True, False, True], 'range_flattens': [None, True, None, False], 'load_eviction_policies': ['', '', '', 'last', 'first', 'first', 'first', 'last'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 32, 32], 'range_unroll_factors': [0, 1, 3, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 4, 2], 'range_multi_buffers': [None, None, True, None], 'range_flattens': [None, True, True, None], 'load_eviction_policies': ['last', '', 'last', '', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 4, 3, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 3, 3, 3], 'range_multi_buffers': [None, None, False, False], 'range_flattens': [None, None, False, True], 'load_eviction_policies': ['first', 'first', 'last', '', 'last', 'first', '', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 4, 3, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 2, 1], 'range_multi_buffers': [None, True, None, None], 'range_flattens': [None, None, True, True], 'load_eviction_policies': ['first', 'first', 'last', 'first', 'first', 'first', 'first', 'last'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 4, 3, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 3, 3, 3], 'range_multi_buffers': [None, None, False, False], 'range_flattens': [None, None, False, True], 'load_eviction_policies': ['first', 'first', 'last', '', 'last', 'first', '', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 4, 3, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 3, 3, 3], 'range_multi_buffers': [None, None, False, False], 'range_flattens': [None, None, False, True], 'load_eviction_policies': ['first', 'first', 'last', '', 'last', 'first', '', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_bwd_wy_dL_delta_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_bwd_wy_dL_delta_helion, args)


def autotune_chunk_bwd_wy_dL_delta_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_bwd_wy_dL_delta_helion[key_chunk_bwd_wy_dL_delta_helion(*args)]


_KEYS_chunk_cumsum_gc_helion = [
    (((1024, 64, 64),), ()),
    (((2048, 64, 128),), ()),
    (((8192, 64, 128),), ()),
    (((8192, 64, 256),), ()),
    (((12288, 64, 128),), ()),
    (((16384, 64, 128),), ()),
    (((1024, 64, 64),), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((2048, 64, 128),), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((8192, 64, 128),), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((8192, 64, 256),), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((12288, 64, 128),), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((16384, 64, 128),), ('None', 'None', 'None', 'False', 'False', 'False')),
    (((1024, 64, 64), (8,), (8, 64)), ('True', 'True', 'True')),
    (((2048, 64, 128), (16,), (16, 128)), ('True', 'True', 'True')),
    (((8192, 64, 128), (16,), (16, 128)), ('True', 'True', 'True')),
    (((8192, 64, 256), (32,), (32, 256)), ('True', 'True', 'True')),
    (((12288, 64, 128), (96,), (96, 128)), ('True', 'True', 'True')),
    (((16384, 64, 128), (64,), (64, 128)), ('True', 'True', 'True')),
]

_CONFIGS_chunk_cumsum_gc_helion = [
    {'block_sizes': [1, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': [''], 'num_warps': 2, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 1, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [True], 'load_eviction_policies': ['first'], 'num_warps': 2, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 8, 'maxnreg': 256},
    {'block_sizes': [1, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 7, 'indexing': ['pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last'], 'num_warps': 2, 'num_stages': 8, 'indexing': ['pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [2], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [True], 'range_flattens': [True], 'load_eviction_policies': ['', 'last', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 4},
    {'block_sizes': [1, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last'], 'num_warps': 2, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', 'last'], 'num_warps': 2, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', ''], 'num_warps': 2, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 32], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [1], 'range_warp_specializes': [], 'range_num_stages': [2], 'range_multi_buffers': [None], 'range_flattens': [True], 'load_eviction_policies': ['last', '', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 64, 'maxnreg': 256},
]


def key_chunk_cumsum_gc_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_cumsum_gc_helion, args)


def autotune_chunk_cumsum_gc_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_cumsum_gc_helion[key_chunk_cumsum_gc_helion(*args)]


_KEYS_chunk_cumsum_gc_varlen_helion = [
    (((8192, 64, 128), (128,), (128,), (8192, 64, 128), (64,), (64, 128)), ('True', 'True', 'True')),
    (((8192, 64, 128), (131,), (131,), (8384, 64, 128), (64,), (64, 128)), ('True', 'True', 'True')),
    (((8192, 96, 128), (128,), (128,), (12288, 64, 128), (96,), (96, 128)), ('True', 'True', 'True')),
    (((8192, 96, 128), (131,), (131,), (12576, 64, 128), (96,), (96, 128)), ('True', 'True', 'True')),
]

_CONFIGS_chunk_cumsum_gc_varlen_helion = [
    {'block_sizes': [128], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'last', '', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'last', '', ''], 'num_warps': 4, 'num_stages': 7, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'last', '', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last', '', 'last'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_cumsum_gc_varlen_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_cumsum_gc_varlen_helion, args)


def autotune_chunk_cumsum_gc_varlen_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_cumsum_gc_varlen_helion[key_chunk_cumsum_gc_varlen_helion(*args)]


_KEYS_chunk_fwd_A_diag_anchored_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ()),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128)), ()),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128)), ()),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256)), ()),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128)), ()),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128)), ()),
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ('True',)),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128)), ('True',)),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128)), ('True',)),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256)), ('True',)),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128)), ('True',)),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128)), ('True',)),
]

_CONFIGS_chunk_fwd_A_diag_anchored_helion = [
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'first', 'first', 'first', 'first', 'first', 'first', '', 'last', 'last', 'first', '', 'first', 'first', 'first', '', 'first', 'first', 'first', 'last'], 'num_warps': 2, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', '', '', 'last', 'last', '', '', 'first', 'last', 'last', '', '', '', 'first', 'first', 'first', 'first', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'last', 'last', 'first', 'first', 'last', '', '', '', 'first', 'last', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', '', '', '', 'first', '', 'last', 'last', '', '', '', '', '', 'last', 'last', 'first', 'first', 'first', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', '', 'first', 'first', '', 'first', 'last', '', 'last', '', '', 'first', '', 'first', 'first', '', '', 'first', '', '', 'first', '', 'last'], 'num_warps': 2, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', '', '', 'first', '', '', '', '', '', '', '', '', '', '', '', '', 'first', 'first', '', '', '', 'last', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'last'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'last', 'last', 'last', '', 'first', 'last', '', '', 'last', 'first', 'first', '', 'last', 'last', 'last', '', '', 'last', 'first', 'first', 'last', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_fwd_A_diag_anchored_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_A_diag_anchored_helion, args)


def autotune_chunk_fwd_A_diag_anchored_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_A_diag_anchored_helion[key_chunk_fwd_A_diag_anchored_helion(*args)]


_KEYS_chunk_fwd_A_diag_anchored_varlen_helion = [
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (128,), (128,), (8192, 64, 64), (8192, 64, 64)), ('True',)),
    (((8192, 64, 128), (8192, 64, 128), (8384, 64, 128), (131,), (131,), (8384, 64, 64), (8384, 64, 64)), ('True',)),
    (((8192, 96, 128), (8192, 96, 128), (12288, 64, 128), (128,), (128,), (12288, 64, 64), (12288, 64, 64)), ('True',)),
    (((8192, 96, 128), (8192, 96, 128), (12576, 64, 128), (131,), (131,), (12576, 64, 64), (12576, 64, 64)), ('True',)),
]

_CONFIGS_chunk_fwd_A_diag_anchored_varlen_helion = [
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'last', 'last', 'last', '', 'first', '', 'first', 'last', 'last', 'last', 'first', '', 'last', 'last', '', '', '', '', '', '', 'first', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_fwd_A_diag_anchored_varlen_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_A_diag_anchored_varlen_helion, args)


def autotune_chunk_fwd_A_diag_anchored_varlen_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_A_diag_anchored_varlen_helion[key_chunk_fwd_A_diag_anchored_varlen_helion(*args)]


_KEYS_chunk_fwd_h_delta_helion = [
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 128, 128), (32, 256, 64, 128), (32, 256, 128)), ('False', 'True', 'True')),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 64, 64), (64, 16, 64, 64), (64, 16, 64)), ('False', 'True', 'True')),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 128, 128), (64, 32, 64, 128), (64, 32, 128)), ('False', 'True', 'True')),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128), (96, 128, 64, 128), (96, 128, 128)), ('False', 'True', 'True')),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 256, 256), (256, 32, 64, 256), (256, 32, 256)), ('False', 'True', 'True')),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 128, 128), (256, 64, 64, 128), (256, 64, 128)), ('False', 'True', 'True')),
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 128, 128)), ('None', 'None', 'False', 'False', 'False')),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 64, 64)), ('None', 'None', 'False', 'False', 'False')),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 128, 128)), ('None', 'None', 'False', 'False', 'False')),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128)), ('None', 'None', 'False', 'False', 'False')),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 256, 256)), ('None', 'None', 'False', 'False', 'False')),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 128, 128)), ('None', 'None', 'False', 'False', 'False')),
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 64, 128), (32, 128, 128), (32, 256, 64), (32, 256)), ('True', 'False', 'False')),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64, 64), (64, 64, 64), (64, 16, 64), (64, 16)), ('True', 'False', 'False')),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 64, 128), (64, 128, 128), (64, 32, 64), (64, 32)), ('True', 'False', 'False')),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128), (96, 128, 64), (96, 128)), ('True', 'False', 'False')),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 64, 256), (256, 256, 256), (256, 32, 64), (256, 32)), ('True', 'False', 'False')),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 64, 128), (256, 128, 128), (256, 64, 64), (256, 64)), ('True', 'False', 'False')),
]

_CONFIGS_chunk_fwd_h_delta_helion = [
    {'block_sizes': [16], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', '', 'first', 'first', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'first', 'last', '', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['last', '', 'first', '', 'last'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', '', '', 'last', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last', 'first', 'last', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'first', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'last', 'last', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [True, True], 'range_flattens': [None, True], 'load_eviction_policies': ['last', '', '', 'first'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'last', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['first', '', '', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', '', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 4},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', '', '', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'last', '', 'first', 'last', 'first'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['last', '', 'last', '', '', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'last', 'last', '', 'last', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['first', '', '', '', 'last', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last', '', '', 'last', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_fwd_h_delta_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_h_delta_helion, args)


def autotune_chunk_fwd_h_delta_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_h_delta_helion[key_chunk_fwd_h_delta_helion(*args)]


_KEYS_chunk_fwd_h_delta_varlen_helion = [
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (64, 128, 128), (8192, 128), (2,)), ()),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (512, 128, 128), (8192, 128), (9,)), ()),
    (((8384, 64, 128), (8384, 64, 128), (8384, 64, 128), (384, 128, 128), (8384, 128), (7,)), ()),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (96, 128, 128), (12288, 128), (2,)), ()),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (768, 128, 128), (12288, 128), (9,)), ()),
    (((12576, 64, 128), (12576, 64, 128), (12576, 64, 128), (576, 128, 128), (12576, 128), (7,)), ()),
]

_CONFIGS_chunk_fwd_h_delta_varlen_helion = [
    {'block_sizes': [64], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', '', '', '', 'last', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [True, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'first', 'last', '', 'first', 'last', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 2},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last', '', 'first', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last', 'last', 'first', 'first', 'first', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last', 'first', '', 'last', 'last', ''], 'num_warps': 2, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last', '', '', 'first', 'last', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_fwd_h_delta_varlen_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_h_delta_varlen_helion, args)


def autotune_chunk_fwd_h_delta_varlen_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_h_delta_varlen_helion[key_chunk_fwd_h_delta_varlen_helion(*args)]


_KEYS_chunk_fwd_h_diag_fused = [
    (((32, 256, 64, 128), (32, 256, 64, 128)), ('None', 'None', 'None', 'False', 'False', 'False', 'False')),
    (((64, 16, 64, 64), (64, 16, 64, 64)), ('None', 'None', 'None', 'False', 'False', 'False', 'False')),
    (((64, 32, 64, 128), (64, 32, 64, 128)), ('None', 'None', 'None', 'False', 'False', 'False', 'False')),
    (((96, 128, 64, 128), (96, 128, 64, 128)), ('None', 'None', 'None', 'False', 'False', 'False', 'False')),
    (((256, 32, 64, 256), (256, 32, 64, 256)), ('None', 'None', 'None', 'False', 'False', 'False', 'False')),
    (((256, 64, 64, 128), (256, 64, 64, 128)), ('None', 'None', 'None', 'False', 'False', 'False', 'False')),
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256), (32, 256, 64)), ('None', 'True', 'True', 'False', 'False')),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16), (64, 16, 64)), ('None', 'True', 'True', 'False', 'False')),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32), (64, 32, 64)), ('None', 'True', 'True', 'False', 'False')),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128), (96, 128, 64)), ('None', 'True', 'True', 'False', 'False')),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32), (256, 32, 64)), ('None', 'True', 'True', 'False', 'False')),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64), (256, 64, 64)), ('None', 'True', 'True', 'False', 'False')),
    (((32, 256, 64, 128), (32, 256, 64, 128), (32, 256, 128), (32, 128, 128), (32, 256, 64, 128)), ('True', 'False', 'True', 'True')),
    (((64, 16, 64, 64), (64, 16, 64, 64), (64, 16, 64), (64, 64, 64), (64, 16, 64, 64)), ('True', 'False', 'True', 'True')),
    (((64, 32, 64, 128), (64, 32, 64, 128), (64, 32, 128), (64, 128, 128), (64, 32, 64, 128)), ('True', 'False', 'True', 'True')),
    (((96, 128, 64, 128), (96, 128, 64, 128), (96, 128, 128), (96, 128, 128), (96, 128, 64, 128)), ('True', 'False', 'True', 'True')),
    (((256, 32, 64, 256), (256, 32, 64, 256), (256, 32, 256), (256, 256, 256), (256, 32, 64, 256)), ('True', 'False', 'True', 'True')),
    (((256, 64, 64, 128), (256, 64, 64, 128), (256, 64, 128), (256, 128, 128), (256, 64, 64, 128)), ('True', 'False', 'True', 'True')),
]

_CONFIGS_chunk_fwd_h_diag_fused = [
    {'block_sizes': [64, 64], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 64], 'loop_orders': [[2, 0, 1]], 'l2_groupings': [4], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 64], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [4, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1, 'maxnreg': 256},
    {'block_sizes': [128, 128], 'loop_orders': [[0, 2, 1]], 'l2_groupings': [16], 'range_unroll_factors': [1, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [False, True], 'range_flattens': [None, False], 'load_eviction_policies': ['first', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 4},
    {'block_sizes': [128, 256], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['first', 'first'], 'num_warps': 16, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[2, 0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [4, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [True, False], 'range_flattens': [None, False], 'load_eviction_policies': ['last', ''], 'num_warps': 8, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [128, 32], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'first', '', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 32], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1, 'maxnreg': 128},
    {'block_sizes': [64, 64], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'first', '', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [1], 'range_unroll_factors': [1, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [True, None], 'range_flattens': [True, None], 'load_eviction_policies': ['last', '', 'first', 'first'], 'num_warps': 32, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 4, 'maxnreg': 64},
    {'block_sizes': [128, 128], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', '', 'first'], 'num_warps': 8, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'first', '', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    {'block_sizes': [64, 64], 'loop_orders': [[2, 0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'last', 'last', '', 'first'], 'num_warps': 2, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    {'block_sizes': [64, 32], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['last', '', 'first', '', ''], 'num_warps': 2, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 128], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'last', '', 'first'], 'num_warps': 32, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[0, 2, 1]], 'l2_groupings': [4], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['last', '', 'last', '', 'last'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 256], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'last', 'last', 'first', ''], 'num_warps': 16, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['last', '', 'last', 'last', 'last'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
]


def key_chunk_fwd_h_diag_fused(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_h_diag_fused, args)


def autotune_chunk_fwd_h_diag_fused(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_h_diag_fused[key_chunk_fwd_h_diag_fused(*args)]


_KEYS_chunk_fwd_o_diag_anchored_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ()),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 128, 128), (2048, 64, 64)), ()),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 128, 128), (8192, 64, 64)), ()),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 256, 256), (8192, 64, 64)), ()),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 128, 128), (12288, 64, 64)), ()),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 128, 128), (16384, 64, 64)), ()),
]

_CONFIGS_chunk_fwd_o_diag_anchored_helion = [
    {'block_sizes': [1, 64, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['first', 'first', 'first', 'first', ''], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'first', 'first', 'first'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 256, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'last', '', '', 'first'], 'num_warps': 8, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'first', 'first', 'first'], 'num_warps': 8, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'first', 'first', 'last', 'last'], 'num_warps': 8, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_fwd_o_diag_anchored_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_o_diag_anchored_helion, args)


def autotune_chunk_fwd_o_diag_anchored_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_o_diag_anchored_helion[key_chunk_fwd_o_diag_anchored_helion(*args)]


_KEYS_chunk_fwd_o_diag_anchored_varlen_helion = [
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 128, 128), (8192, 64, 64), (128,), (128,), (8192, 64, 128)), ('True',)),
    (((8192, 64, 128), (8384, 64, 128), (8384, 64, 128), (8384, 128, 128), (8384, 64, 64), (131,), (131,), (8192, 64, 128)), ('True',)),
    (((8192, 96, 128), (12288, 64, 128), (12288, 64, 128), (12288, 128, 128), (12288, 64, 64), (128,), (128,), (8192, 96, 128)), ('True',)),
    (((8192, 96, 128), (12576, 64, 128), (12576, 64, 128), (12576, 128, 128), (12576, 64, 64), (131,), (131,), (8192, 96, 128)), ('True',)),
]

_CONFIGS_chunk_fwd_o_diag_anchored_varlen_helion = [
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_fwd_o_diag_anchored_varlen_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_o_diag_anchored_varlen_helion, args)


def autotune_chunk_fwd_o_diag_anchored_varlen_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_o_diag_anchored_varlen_helion[key_chunk_fwd_o_diag_anchored_varlen_helion(*args)]


_KEYS_chunk_fwd_o_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ('None', 'False')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 128, 128)), ('None', 'False')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 128, 128)), ('None', 'False')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 256, 256)), ('None', 'False')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 128, 128)), ('None', 'False')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 128, 128)), ('None', 'False')),
    (((1024, 64, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64), (1024, 64, 64)), ('True',)),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64, 128), (2048, 64), (2048, 128, 128)), ('True',)),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64, 128), (8192, 64), (8192, 128, 128)), ('True',)),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64, 256), (8192, 64), (8192, 256, 256)), ('True',)),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64, 128), (12288, 64), (12288, 128, 128)), ('True',)),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64, 128), (16384, 64), (16384, 128, 128)), ('True',)),
]

_CONFIGS_chunk_fwd_o_helion = [
    {'block_sizes': [1, 64, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'first', 'last', 'first'], 'num_warps': 4, 'num_stages': 8, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'first', 'first', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'last', 'first', 'first'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'first', 'last'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 64, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'first', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', '', '', '', 'first'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['first', '', 'last', '', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['first', '', 'first', 'last', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [1, 128, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', 'last', 'first', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_fwd_o_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_o_helion, args)


def autotune_chunk_fwd_o_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_o_helion[key_chunk_fwd_o_helion(*args)]


_KEYS_chunk_fwd_wy_delta_helion = [
    (((1024, 64, 64), (1024, 64, 64), (1024, 64), (1024, 64, 64), (1024, 64, 64), (1024, 64, 64)), ('False', 'True')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64), (2048, 64, 128), (2048, 64, 64), (2048, 64, 128)), ('False', 'True')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64), (8192, 64, 128), (8192, 64, 64), (8192, 64, 128)), ('False', 'True')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64), (8192, 64, 256), (8192, 64, 64), (8192, 64, 256)), ('False', 'True')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64), (12288, 64, 128), (12288, 64, 64), (12288, 64, 128)), ('False', 'True')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64), (16384, 64, 128), (16384, 64, 64), (16384, 64, 128)), ('False', 'True')),
    (((1024, 64, 64), (1024, 64, 64), (1024, 64)), ('None', 'None', 'None', 'False', 'False')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64)), ('None', 'None', 'None', 'False', 'False')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64)), ('None', 'None', 'None', 'False', 'False')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64)), ('None', 'None', 'None', 'False', 'False')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64)), ('None', 'None', 'None', 'False', 'False')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64)), ('None', 'None', 'None', 'False', 'False')),
    (((1024, 64, 64), (1024, 64, 64), (1024, 64), (1024, 64)), ('None', 'None', 'True', 'False')),
    (((2048, 64, 128), (2048, 64, 128), (2048, 64), (2048, 64)), ('None', 'None', 'True', 'False')),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64), (8192, 64)), ('None', 'None', 'True', 'False')),
    (((8192, 64, 256), (8192, 64, 256), (8192, 64), (8192, 64)), ('None', 'None', 'True', 'False')),
    (((12288, 64, 128), (12288, 64, 128), (12288, 64), (12288, 64)), ('None', 'None', 'True', 'False')),
    (((16384, 64, 128), (16384, 64, 128), (16384, 64), (16384, 64)), ('None', 'None', 'True', 'False')),
]

_CONFIGS_chunk_fwd_wy_delta_helion = [
    {'block_sizes': [64, 64], 'range_unroll_factors': [4, 0, 0, 0], 'range_warp_specializes': [], 'range_multi_buffers': [True, True, None, True], 'range_flattens': [False, None, True, True], 'static_ranges': [False], 'load_eviction_policies': ['last', 'first', 'first', '', 'first', 'last'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 64, 'maxnreg': 128},
    {'block_sizes': [32, 32], 'range_unroll_factors': [0, 4, 4, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True, None, None], 'range_flattens': [None, False, True, True], 'static_ranges': [False], 'load_eviction_policies': ['first', '', 'last', '', '', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 32], 'range_unroll_factors': [0, 1, 1, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, True], 'range_flattens': [None, True, True, True], 'static_ranges': [False], 'load_eviction_policies': ['last', '', 'first', '', 'first', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 32], 'range_unroll_factors': [0, 0, 4, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None, False, False], 'range_flattens': [None, None, None, None], 'static_ranges': [False], 'load_eviction_policies': ['last', 'last', 'first', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 32], 'range_unroll_factors': [0, 0, 4, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, False], 'range_flattens': [None, True, True, False], 'static_ranges': [False], 'load_eviction_policies': ['', '', 'last', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 32], 'range_unroll_factors': [0, 2, 0, 1], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True, True, True], 'range_flattens': [None, None, True, True], 'static_ranges': [False], 'load_eviction_policies': ['last', '', 'last', 'first', 'first', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 32, 32], 'range_unroll_factors': [0, 3, 0, 2, 3], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False, None, None, None], 'range_flattens': [None, None, None, None, True], 'static_ranges': [True], 'load_eviction_policies': ['last', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 32, 32], 'range_unroll_factors': [0, 3, 3, 0, 4], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False, False, False, True], 'range_flattens': [None, True, True, True, None], 'static_ranges': [False], 'load_eviction_policies': ['last', 'last', 'first', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 32, 32], 'range_unroll_factors': [0, 2, 1, 0, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True, True, None, True], 'range_flattens': [None, None, True, True, True], 'static_ranges': [False], 'load_eviction_policies': ['', '', 'first', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 32, 32], 'range_unroll_factors': [0, 4, 0, 2, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, True, None, False, None], 'range_flattens': [None, None, None, None, True], 'static_ranges': [False], 'load_eviction_policies': ['last', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 32, 32], 'range_unroll_factors': [0, 2, 0, 0, 4], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False, None, False, False], 'range_flattens': [None, True, None, None, False], 'static_ranges': [True], 'load_eviction_policies': ['last', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 32, 32], 'range_unroll_factors': [0, 0, 4, 3, 4], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False, None, False, True], 'range_flattens': [None, False, None, None, False], 'static_ranges': [False], 'load_eviction_policies': ['last', '', '', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 16, 16], 'range_unroll_factors': [0, 0, 0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None, None, None, None], 'range_flattens': [None, None, None, None, None], 'static_ranges': [False], 'load_eviction_policies': ['', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 32, 32], 'range_unroll_factors': [2, 3, 0, 4, 0], 'range_warp_specializes': [], 'range_multi_buffers': [True, True, None, None, False], 'range_flattens': [True, True, None, None, None], 'static_ranges': [True], 'load_eviction_policies': ['last', 'last', 'last', '', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 16, 'maxnreg': 256},
    {'block_sizes': [128, 32, 32], 'range_unroll_factors': [0, 3, 0, 2, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False, None, None, True], 'range_flattens': [None, True, None, None, True], 'static_ranges': [True], 'load_eviction_policies': ['', 'last', 'first', 'first', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [16, 32, 32], 'range_unroll_factors': [0, 4, 0, 0, 3], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False, None, True, True], 'range_flattens': [None, False, None, False, True], 'static_ranges': [True], 'load_eviction_policies': ['first', 'last', 'last', '', 'first'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 32, 32], 'range_unroll_factors': [0, 3, 0, 0, 4], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False, None, True, True], 'range_flattens': [None, None, None, True, False], 'static_ranges': [True], 'load_eviction_policies': ['last', 'last', '', '', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 32, 32], 'range_unroll_factors': [0, 2, 0, 0, 2], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, False, None, True, True], 'range_flattens': [None, False, None, True, True], 'static_ranges': [True], 'load_eviction_policies': ['last', 'last', 'first', 'last', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_fwd_wy_delta_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_wy_delta_helion, args)


def autotune_chunk_fwd_wy_delta_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_wy_delta_helion[key_chunk_fwd_wy_delta_helion(*args)]


_KEYS_chunk_fwd_wy_delta_varlen_helion = [
    (((8192, 64, 128), (8192, 64, 128), (8192, 64), (8192, 64, 128), (8192, 64, 64), (128,), (128,), (8192, 64, 128), (8192, 64, 128), (8192, 64, 128)), ()),
    (((8192, 64, 128), (8192, 64, 128), (8192, 64), (8384, 64, 128), (8384, 64, 64), (131,), (131,), (8384, 64, 128), (8384, 64, 128), (8384, 64, 128)), ()),
    (((8192, 96, 128), (8192, 96, 128), (8192, 96), (12288, 64, 128), (12288, 64, 64), (128,), (128,), (12288, 64, 128), (12288, 64, 128), (12288, 64, 128)), ()),
    (((8192, 96, 128), (8192, 96, 128), (8192, 96), (12576, 64, 128), (12576, 64, 64), (131,), (131,), (12576, 64, 128), (12576, 64, 128), (12576, 64, 128)), ()),
]

_CONFIGS_chunk_fwd_wy_delta_varlen_helion = [
    {'block_sizes': [32], 'range_unroll_factors': [0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'static_ranges': [False], 'load_eviction_policies': ['', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'range_unroll_factors': [0, 0, 4], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, True, False], 'static_ranges': [False], 'load_eviction_policies': ['last', '', '', 'last', '', 'last', '', 'last'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'range_unroll_factors': [0, 0, 3], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'static_ranges': [False], 'load_eviction_policies': ['', 'last', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32], 'range_unroll_factors': [0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'static_ranges': [False], 'load_eviction_policies': ['', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_chunk_fwd_wy_delta_varlen_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_chunk_fwd_wy_delta_varlen_helion, args)


def autotune_chunk_fwd_wy_delta_varlen_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_chunk_fwd_wy_delta_varlen_helion[key_chunk_fwd_wy_delta_varlen_helion(*args)]


_KEYS_l2norm_fwd_helion = [
    (((524288, 128),), ()),
    (((786432, 128),), ()),
]

_CONFIGS_l2norm_fwd_helion = [
    {'block_sizes': [64], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'first'], 'num_warps': 32, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'first'], 'num_warps': 32, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def key_l2norm_fwd_helion(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_l2norm_fwd_helion, args)


def autotune_l2norm_fwd_helion(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_l2norm_fwd_helion[key_l2norm_fwd_helion(*args)]
