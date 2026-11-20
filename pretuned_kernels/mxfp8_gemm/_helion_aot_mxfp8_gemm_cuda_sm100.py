"""
Pretuned configs for the packed MXFP8 GEMM ``mxfp8_gemm`` (Helion Triton backend).
Backend: explicit per-shape table (exact (M, N, K) -> tuned config)

Tuned on NVIDIA GB300 (sm103, which uses sm100 heuristic files). For each
shape the candidates were the ``triton_b200_packed_mxfp8`` seed, a config
tuned for the generic (pointer) lowering, and a quick autotune (Triton 3.8,
CUDA-graph benchmarking) seeded with both.  Each candidate was timed under
CUDA-graph replay with the L2 cache flushed before every replay on Triton 3.7,
3.8 and 3.9; the checked-in config is the fastest on 3.7 and 3.8 (the
autotuned one for every shape except 1x1792x5120, where the seed wins).
Every config is checked against the FP32 dequantization reference on all
three Triton versions.

The M <= 32 shapes use 16-row tiles, which the native lowering issues as the
transposed MMA; 1x1792x5120, 16x2048x4096 and 128x1792x5120 also split K over
8 grid partitions so their few output tiles fill the GPU.  M = 128 and 1024
use 128-row persistent tiles.

Each tuned (M, N, K) maps to its own config; an unseen shape falls back to the
nearest tuned (M, N, K) (smallest sum of absolute log-ratios over M, N, K).

Provides, for kernel mxfp8_gemm:
- key_mxfp8_gemm(*args): config index (also the runtime cache key)
- autotune_mxfp8_gemm(*args): config dict for the given arguments
"""

import math


def _mnk(args):
    """(M, N, K) from raw kernel args: a[M, K], b[N, K]."""
    a, b = args[0], args[1]
    return int(a.shape[0]), int(b.shape[0]), int(a.shape[1])


def _select(keys, mnk):
    """Exact (M, N, K) match if tuned, else the nearest tuned shape."""
    for i, key in enumerate(keys):
        if tuple(key) == mnk:
            return i
    best_i, best_d = 0, float("inf")
    for i, key in enumerate(keys):
        d = sum(
            abs(math.log(max(x, 1)) - math.log(max(y, 1))) for x, y in zip(key, mnk)
        )
        if d < best_d:
            best_i, best_d = i, d
    return best_i


# Tuned (M, N, K) shapes, parallel to _CONFIGS_mxfp8_gemm.
_KEYS_mxfp8_gemm = [
    (1, 1792, 5120),
    (8, 8192, 1280),
    (16, 2048, 4096),
    (32, 5120, 2048),
    (128, 1792, 5120),
    (1024, 8192, 1280),
]

_CONFIGS_mxfp8_gemm = [
    # (M, N, K) = (1, 1792, 5120)
    {
        "block_sizes": [16, 128, 4],
        "l2_groupings": [8],
        "range_warp_specializes": [False, False],
        "range_num_stages": [5, 0],
        "range_multi_buffers": [False, False],
        "range_flattens": [False, False],
        "num_warps": 4,
        "num_stages": 5,
        "pid_type": "persistent_blocked",
        "num_sm_multiplier": 1,
        "split_k": 8,
        "maxnreg": None,
    },
    # (M, N, K) = (8, 8192, 1280)
    {
        "block_sizes": [16, 128, 8],
        "l2_groupings": [4],
        "range_warp_specializes": [False, False],
        "range_num_stages": [5, 0],
        "range_multi_buffers": [False, False],
        "range_flattens": [False, False],
        "load_eviction_policies": ["", "last", "", "last"],
        "num_warps": 4,
        "num_stages": 5,
        "indexing": [
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "pointer",
        ],
        "pid_type": "persistent_blocked",
        "num_sm_multiplier": 1,
        "split_k": 1,
        "maxnreg": None,
    },
    # (M, N, K) = (16, 2048, 4096)
    {
        "block_sizes": [16, 128, 8],
        "loop_orders": [[2, 1, 0]],
        "range_warp_specializes": [None, True],
        "range_num_stages": [0, 1],
        "range_multi_buffers": [None, True],
        "range_flattens": [None, False],
        "load_eviction_policies": ["first", "first", "first", "first"],
        "num_warps": 2,
        "num_stages": 5,
        "indexing": [
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        "pid_type": "flat",
        "split_k": 8,
    },
    # (M, N, K) = (32, 5120, 2048)
    {
        "block_sizes": [16, 128, 8],
        "l2_groupings": [8],
        "range_warp_specializes": [False, False],
        "range_num_stages": [5, 0],
        "range_multi_buffers": [None, False],
        "range_flattens": [False, False],
        "load_eviction_policies": ["first", "", "", ""],
        "num_warps": 4,
        "num_stages": 4,
        "pid_type": "persistent_blocked",
        "num_sm_multiplier": 1,
        "maxnreg": 128,
        "split_k": 1,
    },
    # (M, N, K) = (128, 1792, 5120)
    {
        "block_sizes": [128, 128, 8],
        "loop_orders": [[2, 0, 1]],
        "l2_groupings": [8],
        "range_warp_specializes": [False, False],
        "range_num_stages": [2, 0],
        "range_multi_buffers": [False, False],
        "range_flattens": [False, False],
        "load_eviction_policies": ["", "", "last", ""],
        "num_warps": 4,
        "num_stages": 3,
        "indexing": [
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        "pid_type": "persistent_blocked",
        "num_sm_multiplier": 1,
        "split_k": 8,
        "maxnreg": None,
    },
    # (M, N, K) = (1024, 8192, 1280)
    {
        "block_sizes": [128, 256, 8],
        "loop_orders": [[2, 1, 0]],
        "range_warp_specializes": [None, False],
        "range_multi_buffers": [False, False],
        "load_eviction_policies": ["", "last", "last", "last"],
        "num_warps": 4,
        "num_stages": 3,
        "pid_type": "persistent_interleaved",
        "num_sm_multiplier": 1,
        "maxnreg": 256,
        "split_k": 1,
    },
]


def key_mxfp8_gemm(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_mxfp8_gemm, _mnk(args))


def autotune_mxfp8_gemm(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_mxfp8_gemm[key_mxfp8_gemm(*args)]
