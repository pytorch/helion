"""
Auto-generated heuristic for kernels: scale_mm_cute, scale_mm_cute_skinny_m
Backend: explicit per-shape table (exact (M, K, N) -> tuned config)

RowWise-scaled FP8 GEMM pretuned on NVIDIA B200 (sm100) for the Helion CuTe
(tcgen05) backend. Configs were autotuned under CUDA-graph benchmarking (how
these decode / small-batch GEMMs are actually invoked); each shape keeps its
own best config (best of the wall-clock and cudagraph autotune sweeps, chosen
by a per-shape cudagraph microbenchmark). Benchmarks clear the L2 cache before
every replay (cold L2), matching pretuned_kernels/_bench.py.

The two 512-M shapes were re-tuned (ncu-guided, not autotune) onto the fp8
small-grid 2-CTA cluster (bm=128, cluster_m=2, deep ab=12): NCU showed the old
cluster_m=1 bm=128 tile ran at 0.86 waves (128 tiles on 148 SMs) and was
barrier/occupancy bound; cluster_m=2 doubles the CTA count and multicasts A
across the CTA pair (halving its cold DRAM read), lifting 512x2048x4096 from
0.73x to 0.80x and 512x2048x2048 from 0.86x to 0.91x vs the best baseline.

Each tuned (M, K, N) maps to its own config; an unseen shape falls back to the
nearest tuned (M, K, N) (smallest sum of absolute log-ratios over M, K, N).

Provides, for each kernel <k>:
- key_<k>(*args): config index (also the runtime cache key)
- autotune_<k>(*args): config dict for the given arguments
"""

import math

import torch


def _mkn(args):
    """(M, K, N) from raw kernel args: x[M, K], y[K, N]."""
    x, y = args[0], args[1]
    return int(x.shape[0]), int(x.shape[1]), int(y.shape[1])


def _select(keys, mkn):
    """Exact (M, K, N) match if tuned, else the nearest tuned shape."""
    for i, k in enumerate(keys):
        if tuple(k) == mkn:
            return i
    best_i, best_d = 0, float('inf')
    for i, k in enumerate(keys):
        d = sum(abs(math.log(max(a, 1)) - math.log(max(b, 1))) for a, b in zip(k, mkn))
        if d < best_d:
            best_i, best_d = i, d
    return best_i


# === Kernel: scale_mm_cute_skinny_m ===
# Tuned (M, K, N) shapes, parallel to _CONFIGS_scale_mm_cute_skinny_m.
_KEYS_scale_mm_cute_skinny_m = [
    (1, 4096, 4096),
    (1, 4096, 256),
]

_CONFIGS_scale_mm_cute_skinny_m = [
    # (M, K, N) = (1, 4096, 4096)
    {'block_sizes': [1, 512], 'num_threads': [0, 32], 'cute_vector_widths': [1, 8]},
    # (M, K, N) = (1, 4096, 256)
    {'block_sizes': [1, 256], 'num_threads': [0, 32], 'cute_vector_widths': [8, 8]},
]


def key_scale_mm_cute_skinny_m(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_scale_mm_cute_skinny_m, _mkn(args))


def autotune_scale_mm_cute_skinny_m(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_scale_mm_cute_skinny_m[key_scale_mm_cute_skinny_m(*args)]


# === Kernel: scale_mm_cute ===
# Tuned (M, K, N) shapes, parallel to _CONFIGS_scale_mm_cute.
_KEYS_scale_mm_cute = [
    (4096, 4096, 4096),
    (512, 2048, 4096),
    (512, 2048, 2048),
    (16, 2048, 4096),
    (16, 2048, 12288),
    (16, 4096, 6144),
    (16, 4096, 24576),
]

_CONFIGS_scale_mm_cute = [
    # (M, K, N) = (4096, 4096, 4096)
    {'block_sizes': [256, 256, 128], 'l2_groupings': [4], 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_interleaved', 'tcgen05_cluster_m': 2, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 6, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 8, 'tcgen05_strategy': 'role_local_monolithic', 'tcgen05_layout_strategy': 'default', 'tcgen05_warp_spec_mma_warps': 1, 'tcgen05_warp_spec_ab_load_warps': 1, 'tcgen05_warp_spec_epi_load_warps': 0, 'tcgen05_warp_spec_scheduler_warps': 0, 'tcgen05_warp_spec_c_input_warps': 0, 'tcgen05_warp_spec_store_warps': 0, 'tcgen05_warp_spec_register_decrease': 120, 'tcgen05_warp_spec_register_increase': 256, 'cute_vector_widths': [1, 1, 1], 'tcgen05_persistence_model': 'static_persistent', 'tcgen05_layout_overrides_epi_tile_m': None, 'tcgen05_layout_overrides_epi_tile_n': None, 'tcgen05_layout_overrides_smem_swizzle_a': None, 'tcgen05_layout_overrides_smem_swizzle_b': None, 'tcgen05_layout_overrides_d_store_box_n': None},
    # (M, K, N) = (512, 2048, 4096)
    # Re-tuned to the fp8 small-grid 2-CTA cluster (bm=128, cluster_m=2, per-CTA
    # 64xbn) with a deep ab=12 prefetch. cluster_m=2 doubles the CTA count to 256
    # (fills the 148-SM B200; the old cluster_m=1 bm=128 tile ran at 0.86 waves)
    # and multicasts A across the CTA pair, halving its cold DRAM read. Cold-L2
    # cudagraph vs best baseline (cutlass): 0.73x -> 0.80x on B200.
    {'block_sizes': [128, 128, 128], 'l2_groupings': [1], 'num_warps': 8, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_interleaved', 'tcgen05_cluster_m': 2, 'tcgen05_cluster_n': 1, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_ab_stages': 12, 'tcgen05_num_epi_warps': 4, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (512, 2048, 2048)
    # Same fp8 small-grid 2-CTA cluster as (512, 2048, 4096): device-filling
    # cluster_m=2 + A-multicast + deep ab=12 beats the old cluster_m=1 bn=64 tile.
    # Cold-L2 cudagraph vs best baseline (torch): 0.86x -> 0.91x on B200.
    {'block_sizes': [128, 128, 128], 'l2_groupings': [1], 'num_warps': 8, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_interleaved', 'tcgen05_cluster_m': 2, 'tcgen05_cluster_n': 1, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_ab_stages': 12, 'tcgen05_num_epi_warps': 4, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (16, 2048, 4096)
    # Padded-M tcgen05 (stack #2840-#2854): block_m=64 padded from real M=16,
    # block_n=128 (CTA-count sweet spot), N-axis cluster_n=2 A-multicast. Tuned
    # cold-L2 cudagraph: block_m=64/bn=128 (11.1us) beats the block_m=128 seed
    # (14.7us) and the bn=256 variant (18us).
    {'block_sizes': [64, 128, 128], 'cute_vector_widths': [1, 1, 1], 'loop_orders': [[0, 1]], 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 2, 'tcgen05_ab_stages': 6, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4},
    # (M, K, N) = (16, 2048, 12288)  hand-tuned padded-M tcgen05; ~15.9us (0.67x
    # vs torch, 1.6 TB/s). block_n=128 is the CTA-count sweet spot (96 CTAs);
    # smaller bn adds CTAs but loses tile efficiency, larger bn under-fills SMs.
    {'block_sizes': [64, 128, 128], 'cute_vector_widths': [1, 1, 1], 'loop_orders': [[0, 1]], 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 2, 'tcgen05_ab_stages': 6, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4},
    # (M, K, N) = (16, 4096, 6144)  tuned: block_m=64/bn=128 (16.4us) beats the
    # bn=256 seed (26.5us) and block_m=128 (25.8us) cold-L2.
    {'block_sizes': [64, 128, 128], 'cute_vector_widths': [1, 1, 1], 'loop_orders': [[0, 1]], 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 2, 'tcgen05_ab_stages': 6, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4},
    # (M, K, N) = (16, 4096, 24576)  bn=256 wins at the largest N (39.5us) vs
    # bn=128 (51.8us) cold-L2 -- kept.
    {'block_sizes': [64, 256, 128], 'cute_vector_widths': [1, 1, 1], 'loop_orders': [[0, 1]], 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 2, 'tcgen05_ab_stages': 3, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4},
]


def key_scale_mm_cute(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_scale_mm_cute, _mkn(args))


def autotune_scale_mm_cute(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_scale_mm_cute[key_scale_mm_cute(*args)]
