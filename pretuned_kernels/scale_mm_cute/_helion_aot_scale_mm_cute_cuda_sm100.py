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

The twelve M=64 shapes (vLLM Qwen3 FP8 (K, N) sweep) use cluster_m=1 bm=64 tiles.
ncu showed the main loop dominates the gap to cutlass and was barrier-bound (5 of
6 warps idle at the post-K-loop sync while 1 MMA warp runs, feeding DRAM at ~47%
vs cutlass ~62%). PERSISTENT scheduling (pid_type=persistent_blocked +
static_persistent) keeps each CTA streaming across output tiles: barrier stalls
collapse (~620 -> ~35 pcsamp), the kernel goes cleanly DRAM-bound (47% -> 57%),
and a deep A/B pipeline keeps the TMA ahead of the MMA -- lifting the slow shapes
from 0.69-0.85x to 0.86-0.97x vs cutlass with no codegen change. See the per-shape
block below for which (bn, bk, ab) each shape uses.

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
    (64, 2048, 4096),
    (64, 2048, 2048),
    (64, 2048, 12288),
    (64, 6144, 2048),
    (64, 4096, 6144),
    (64, 4096, 4096),
    (64, 4096, 24576),
    (64, 12288, 4096),
    (64, 5120, 10240),
    (64, 5120, 5120),
    (64, 5120, 51200),
    (64, 25600, 5120),
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
    # M=64 decode / small-batch shapes (vLLM Qwen3 FP8 (K, N) weight sweep). With
    # M=64 there is only one M-tile, so the N-tile count == the grid; the cluster_m=2
    # 512-M configs (bm>=128) waste most of the tile (nearest-neighbor fallback
    # 0.24-0.41x). A cluster_m=1 bm=64 tile with a small N tile fixes that.
    #
    # KEY LEVER -- PERSISTENT scheduling. ncu on the non-persistent (flat) configs
    # showed the main loop (~89% of the gap to cutlass) barrier-bound: 5 of the 6
    # warps (4 epilogue + 1 TMA) idle at the post-K-loop sync while the single MMA
    # warp grinds all K tiles, so DRAM was fed at only ~47% of peak (cutlass ~62%).
    # ``pid_type=persistent_blocked`` + ``static_persistent`` keeps each CTA resident
    # and streaming across output tiles: barrier stalls collapse (~620 -> ~35 pcsamp
    # samples), the kernel becomes cleanly DRAM-bound like cutlass (DRAM 47% -> 57%),
    # and a DEEP A/B pipeline (ab=12 at bn=64, or ab=8 at bn=32/bk=256) keeps the TMA
    # ahead of the MMA. This lifted the slow shapes 0.69-0.85x -> 0.86-0.97x with no
    # codegen change. Per shape: bn=32/bk=256 for medium K/N (more device fill),
    # bn=64/bk=128/ab=12 for large-K or very-wide-N (deepest prefetch that fits SMEM).
    # (M, K, N) = (64, 2048, 4096) -- persistent bn=32 bk=256; 0.97x vs cutlass.
    {'block_sizes': [64, 32, 256], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 8, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 2048, 2048) -- non-persistent bn=16 bk=256 still wins here; 0.97x.
    {'block_sizes': [64, 16, 256], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'pid_type': 'flat', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 8, 'tcgen05_acc_stages': 1, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'non_persistent'},
    # (M, K, N) = (64, 2048, 12288) -- persistent bn=32 bk=256; 0.92x vs cutlass (was 0.82).
    {'block_sizes': [64, 32, 256], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 8, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 6144, 2048) -- persistent bn=32 bk=256; 1.00x vs cutlass (was 0.96).
    {'block_sizes': [64, 32, 256], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 8, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 4096, 6144) -- persistent bn=64 bk=128 ab=12; 0.96x vs cutlass
    # (was 0.86) after the epilogue scale-load pre-wait register hoist landed in
    # codegen (memory_ops.py): both the rowvec scale_b and colvec scale_a whole-tile
    # fragments now load into registers BEFORE the accumulator consumer_wait, hiding
    # their GMEM latency under the MMA wait instead of paying it per subtile.
    {'block_sizes': [64, 64, 128], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 12, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 4096, 4096) -- persistent bn=32 bk=256; 0.94x vs cutlass (was 0.91).
    {'block_sizes': [64, 32, 256], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 8, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 4096, 24576) -- persistent bn=64 bk=128 ab=12; 0.89x vs cutlass (was 0.75).
    {'block_sizes': [64, 64, 128], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 12, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 12288, 4096) -- persistent bn=32 bk=256; 1.02x vs cutlass (was 0.99).
    {'block_sizes': [64, 32, 256], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 8, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 5120, 10240) -- persistent bn=128 bk=128 ab=8; 0.92x vs cutlass (was 0.91).
    {'block_sizes': [64, 128, 128], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 8, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 5120, 5120) -- persistent bn=64 bk=128 ab=12; 0.94x vs cutlass
    # (was 0.86) via the same epilogue scale-load pre-wait hoist as (64, 4096, 6144).
    {'block_sizes': [64, 64, 128], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 12, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 5120, 51200) -- persistent bn=64 bk=128 ab=12; 0.97x vs cutlass (was 0.85).
    {'block_sizes': [64, 64, 128], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 12, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
    # (M, K, N) = (64, 25600, 5120) -- persistent bn=64 bk=128 ab=12; 0.91x vs cutlass (was 0.82).
    {'block_sizes': [64, 64, 128], 'l2_groupings': [1], 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'pid_type': 'persistent_blocked', 'tcgen05_cluster_m': 1, 'tcgen05_cluster_n': 1, 'tcgen05_ab_stages': 12, 'tcgen05_acc_stages': 2, 'tcgen05_c_stages': 2, 'tcgen05_num_epi_warps': 4, 'tcgen05_l2_swizzle_size': 1, 'tcgen05_persistence_model': 'static_persistent'},
]


def key_scale_mm_cute(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_scale_mm_cute, _mkn(args))


def autotune_scale_mm_cute(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_scale_mm_cute[key_scale_mm_cute(*args)]
