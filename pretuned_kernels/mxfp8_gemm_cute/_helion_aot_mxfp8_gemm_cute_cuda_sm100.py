"""
Pretuned configs for the packed MXFP8 GEMM ``mxfp8_gemm_cute`` (Helion CuTe backend).
Backend: explicit per-shape table (exact (M, N, K) -> tuned config)

Tuned on NVIDIA GB300 (sm103, which uses sm100 heuristic files). Every config
selects the tcgen05 block-scaled MMA path (``cute_scaled_mma``) with TMA
operand and scale-factor loads.  The M <= 32 shapes swap A and B so M becomes
the MMA's N (``cute_scaled_swap_ab``); 1024x8192x1280 uses a persistent
128x224 tile on a two-CTA cluster.  Each config is a quick autotune
(CUDA-graph benchmarking) seeded with a config tuned on an earlier Helion; it
was timed under CUDA-graph replay with the L2 cache flushed before every
replay and checked against the FP32 dequantization reference.

Each tuned (M, N, K) maps to its own config; an unseen shape falls back to the
nearest tuned (M, N, K) (smallest sum of absolute log-ratios over M, N, K).

Provides, for kernel mxfp8_gemm_cute:
- key_mxfp8_gemm_cute(*args): config index (also the runtime cache key)
- autotune_mxfp8_gemm_cute(*args): config dict for the given arguments
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


# Tuned (M, N, K) shapes, parallel to _CONFIGS_mxfp8_gemm_cute.
_KEYS_mxfp8_gemm_cute = [
    (1, 1792, 5120),
    (8, 8192, 1280),
    (16, 2048, 4096),
    (32, 5120, 2048),
    (128, 1792, 5120),
    (1024, 8192, 1280),
]

_CONFIGS_mxfp8_gemm_cute = [
    # (M, N, K) = (1, 1792, 5120)
    {
        "block_sizes": [16, 16, 16],
        "loop_orders": [[0, 1, 2]],
        "load_eviction_policies": ["", "", "", ""],
        "cute_store_policy": "default",
        "cute_compiler_pipeline": "opt2",
        "cute_pdl": False,
        "cute_register_chain": False,
        "cute_scaled_mma": True,
        "cute_scaled_tile_n": 8,
        "cute_scaled_tile_k": 256,
        "cute_scaled_stages": 8,
        "cute_scaled_cluster_m": 1,
        "cute_scaled_persistent": False,
        "cute_scaled_prepare": "tiled",
        "cute_scaled_prepare_rows": 32,
        "cute_scaled_epilogue_n": 64,
        "cute_scaled_fused_scales": True,
        "cute_scaled_tile_m": 32,
        "cute_scaled_swap_ab": True,
        "cute_scaled_compact_m": True,
        "cute_scaled_sf_tma": True,
        "cute_scaled_sf_inplace": True,
        "cute_scaled_epilogue_mode": "default",
        "split_k": 1,
    },
    # (M, N, K) = (8, 8192, 1280)
    {
        "block_sizes": [16, 16, 16],
        "loop_orders": [[0, 1, 2]],
        "load_eviction_policies": ["", "", "", ""],
        "cute_store_policy": "default",
        "cute_compiler_pipeline": "default",
        "cute_pdl": False,
        "cute_register_chain": False,
        "cute_scaled_mma": True,
        "cute_scaled_tile_n": 8,
        "cute_scaled_tile_k": 256,
        "cute_scaled_stages": 5,
        "cute_scaled_cluster_m": 1,
        "cute_scaled_persistent": True,
        "cute_scaled_prepare": "tiled",
        "cute_scaled_prepare_rows": 32,
        "cute_scaled_epilogue_n": 64,
        "cute_scaled_fused_scales": True,
        "cute_scaled_tile_m": 64,
        "cute_scaled_swap_ab": True,
        "cute_scaled_compact_m": True,
        "cute_scaled_sf_tma": True,
        "cute_scaled_sf_inplace": True,
        "cute_scaled_epilogue_mode": "default",
        "split_k": 1,
    },
    # (M, N, K) = (16, 2048, 4096)
    {
        "block_sizes": [16, 16, 16],
        "loop_orders": [[0, 1, 2]],
        "load_eviction_policies": ["", "", "", ""],
        "cute_store_policy": "default",
        "cute_compiler_pipeline": "default",
        "cute_pdl": False,
        "cute_register_chain": False,
        "cute_scaled_mma": True,
        "cute_scaled_tile_n": 8,
        "cute_scaled_tile_k": 256,
        "cute_scaled_stages": 6,
        "cute_scaled_cluster_m": 1,
        "cute_scaled_persistent": False,
        "cute_scaled_prepare": "tiled",
        "cute_scaled_prepare_rows": 32,
        "cute_scaled_epilogue_n": 64,
        "cute_scaled_fused_scales": True,
        "cute_scaled_tile_m": 64,
        "cute_scaled_swap_ab": True,
        "cute_scaled_compact_m": True,
        "cute_scaled_sf_tma": True,
        "cute_scaled_sf_inplace": True,
        "cute_scaled_epilogue_mode": "default",
        "split_k": 1,
    },
    # (M, N, K) = (32, 5120, 2048)
    {
        "block_sizes": [16, 16, 16],
        "loop_orders": [[0, 1, 2]],
        "load_eviction_policies": ["", "", "", ""],
        "cute_store_policy": "default",
        "cute_compiler_pipeline": "default",
        "cute_pdl": False,
        "cute_register_chain": False,
        "cute_scaled_mma": True,
        "cute_scaled_tile_n": 32,
        "cute_scaled_tile_k": 256,
        "cute_scaled_stages": 6,
        "cute_scaled_cluster_m": 1,
        "cute_scaled_persistent": False,
        "cute_scaled_prepare": "tiled",
        "cute_scaled_prepare_rows": 32,
        "cute_scaled_epilogue_n": 64,
        "cute_scaled_fused_scales": True,
        "cute_scaled_tile_m": 64,
        "cute_scaled_swap_ab": True,
        "cute_scaled_compact_m": True,
        "cute_scaled_sf_tma": True,
        "cute_scaled_sf_inplace": True,
        "cute_scaled_epilogue_mode": "default",
        "split_k": 1,
    },
    # (M, N, K) = (128, 1792, 5120)
    {
        "block_sizes": [16, 16, 16],
        "loop_orders": [[0, 1, 2]],
        "load_eviction_policies": ["", "", "", ""],
        "cute_store_policy": "default",
        "cute_compiler_pipeline": "default",
        "cute_pdl": False,
        "cute_register_chain": False,
        "cute_scaled_mma": True,
        "cute_scaled_tile_n": 32,
        "cute_scaled_tile_k": 256,
        "cute_scaled_stages": 8,
        "cute_scaled_cluster_m": 1,
        "cute_scaled_persistent": False,
        "cute_scaled_prepare": "tiled",
        "cute_scaled_prepare_rows": 32,
        "cute_scaled_epilogue_n": 64,
        "cute_scaled_fused_scales": True,
        "cute_scaled_tile_m": 64,
        "cute_scaled_swap_ab": False,
        "cute_scaled_compact_m": True,
        "cute_scaled_sf_tma": True,
        "cute_scaled_sf_inplace": True,
        "cute_scaled_epilogue_mode": "default",
        "split_k": 1,
    },
    # (M, N, K) = (1024, 8192, 1280)
    {
        "block_sizes": [16, 16, 16],
        "loop_orders": [[0, 1, 2]],
        "load_eviction_policies": ["", "", "", ""],
        "cute_store_policy": "default",
        "cute_compiler_pipeline": "default",
        "cute_pdl": False,
        "cute_register_chain": False,
        "cute_scaled_mma": True,
        "cute_scaled_tile_n": 224,
        "cute_scaled_tile_k": 128,
        "cute_scaled_stages": 6,
        "cute_scaled_cluster_m": 2,
        "cute_scaled_persistent": True,
        "cute_scaled_prepare": "tiled",
        "cute_scaled_prepare_rows": 32,
        "cute_scaled_epilogue_n": 64,
        "cute_scaled_fused_scales": True,
        "cute_scaled_tile_m": 128,
        "cute_scaled_swap_ab": False,
        "cute_scaled_compact_m": False,
        "cute_scaled_sf_tma": True,
        "cute_scaled_sf_inplace": True,
        "cute_scaled_epilogue_mode": "tma_prefetch_2",
        "split_k": 1,
    },
]


def key_mxfp8_gemm_cute(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_mxfp8_gemm_cute, _mnk(args))


def autotune_mxfp8_gemm_cute(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_mxfp8_gemm_cute[key_mxfp8_gemm_cute(*args)]
