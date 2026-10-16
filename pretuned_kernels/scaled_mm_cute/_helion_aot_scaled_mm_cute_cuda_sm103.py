"""
Pretuned heuristic for kernels: scaled_mm_cute_sm103, scaled_mm_transposed_cute,
scaled_mm_skinny_cute
Backend: explicit per-shape table (exact (M, K, N, scale modes) -> tuned config)

FP8 scaled GEMM pretuned on NVIDIA GB300 (sm103) for the Helion CuTe (tcgen05)
backend. The 16 multi-row configs were selected by cold-L2 batched CUDA-graph
timing and the single-row config by CUPTI kernel time. The configs rely on the
CuTe backend's one-CTA FP8 pipelines, aligned-N physical tiles, and generic
column-major TMA output support. Four configs are mapped to an explicit
frontend-transposed implementation with physical (N, M, K) tile coordinates;
this frontend refactor has not yet been timed on the GPU.

Only these 20 cases are pretuned. The public scaled_mm() wrapper dispatches
M=1 to the skinny kernel and (M, K, N) = (256, 4096, 2048) to the transposed
kernel for all four scale modes. Other shapes use scaled_mm_cute_sm103: an unseen
shape falls back to the nearest of its 12 configs with the same scale modes
(smallest sum of absolute log-ratios over M, N, K). Those schedules are
shape-specific, so the fallback config can be rejected; retune for other
shapes. 1 < M < 64 needs a separate small-M implementation.

Provides, for each kernel <k>:
- key_<k>(*args): config index (also the runtime cache key)
- autotune_<k>(*args): config dict for the given arguments
"""

import math


def _key(args):
    """(M, K, N, scale_a mode, scale_b mode) from a[M, K], b[K, N] and scales."""
    a, b, scale_a, scale_b = args
    return (
        int(a.shape[0]),
        int(a.shape[1]),
        int(b.shape[1]),
        "tensor" if scale_a.ndim == 0 else "token",
        "tensor" if scale_b.ndim == 0 else "channel",
    )


def _select(keys, key):
    """Exact match if tuned, else the nearest tuned shape with the same scale modes."""
    if key in keys:
        return keys.index(key)
    m, k, n, *modes = key
    best_i, best_d = None, math.inf
    for i, (tm, tk, tn, *tuned_modes) in enumerate(keys):
        if tuned_modes != modes:
            continue
        d = sum(
            abs(math.log(max(tuned, 1) / max(query, 1)))
            for tuned, query in ((tm, m), (tn, n), (tk, k))
        )
        if d < best_d:
            best_i, best_d = i, d
    if best_i is None:
        raise KeyError(f"No CuTe scale-mode configuration for {key}")
    return best_i


# === Kernel: scaled_mm_skinny_cute ===
# The single-row (M=1) kernel uses one config for every scale mode; scaled_mm()
# passes a single-element token scale as a scalar.
_CONFIG_scaled_mm_skinny_cute = {"block_sizes": [1, 256], "cute_vector_widths": [1, 8], "num_threads": [0, 32]}


def key_scaled_mm_skinny_cute(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return 0


def autotune_scaled_mm_skinny_cute(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIG_scaled_mm_skinny_cute


# === Kernel: scaled_mm_cute_sm103 ===
# Settings shared by ordinary and frontend-transposed configs.
_BASE = {
    "cute_collective_compute": "warp",
    "cute_collective_copy": "scalar",
    "cute_collective_mma": False,
    "cute_collective_native_seeded": False,
    "cute_collective_tmem_a": False,
    "cute_collective_tmem_seed": False,
    "cute_lane_layouts": ["blocked", "blocked", "blocked"],
    "cute_vector_widths": [1, 1, 1],
    "tcgen05_cluster_n": 1,
    "tcgen05_layout_overrides_d_store_box_n": None,
    "tcgen05_layout_overrides_epi_tile_m": None,
    "tcgen05_layout_overrides_epi_tile_n": None,
    "tcgen05_layout_overrides_smem_swizzle_a": None,
    "tcgen05_layout_overrides_smem_swizzle_b": None,
    "tcgen05_layout_strategy": "default",
    "tcgen05_num_epi_warps": 4,
    "tcgen05_persistence_model": "static_persistent",
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_warp_spec_ab_load_warps": 1,
    "tcgen05_warp_spec_c_input_warps": 0,
    "tcgen05_warp_spec_epi_load_warps": 0,
    "tcgen05_warp_spec_mma_warps": 1,
    "tcgen05_warp_spec_register_decrease": 120,
    "tcgen05_warp_spec_register_increase": 256,
    "tcgen05_warp_spec_scheduler_warps": 0,
    "tcgen05_warp_spec_store_warps": 0,
}
# M=4096 aligned-N schedule: 128x256 public tile over a 192-wide physical tile.
_ALIGNED_N = {
    **_BASE,
    "block_sizes": [128, 256, 128],
    "cute_matmul_family": "tcgen05",
    "cute_split_k_schedule": "legacy",
    "l2_groupings": [1],
    "loop_orders": [[0, 1]],
    "num_sm_multiplier": 1,
    "pid_type": "persistent_interleaved",
    "tcgen05_ab_stages": 10,
    "tcgen05_acc_stages": 2,
    "tcgen05_aux_load_mode": "simt",
    "tcgen05_aux_load_placement": "post_acc_wait",
    "tcgen05_c_stages": 2,
    "tcgen05_c_store_mode": "normal",
    "tcgen05_cluster_m": 2,
    "tcgen05_cta_group": "two",
    "tcgen05_epilogue_layout": "normal",
    "tcgen05_flat_role_coordinates": False,
    "tcgen05_l2_swizzle_size": 1,
    "tcgen05_tile_n": 192,
    "tcgen05_tvm_ffi_launch": False,
}

# Tuned (M, K, N, scale_a mode, scale_b mode), parallel to _CONFIGS_scaled_mm_cute_sm103.
_KEYS_scaled_mm_cute_sm103 = [
    (64, 4096, 4096, "tensor", "tensor"),
    (128, 2880, 5120, "tensor", "tensor"),
    (4096, 4096, 2880, "tensor", "tensor"),
    (64, 4096, 4096, "tensor", "channel"),
    (128, 2880, 5120, "tensor", "channel"),
    (4096, 4096, 2880, "tensor", "channel"),
    (64, 4096, 4096, "token", "tensor"),
    (128, 2880, 5120, "token", "tensor"),
    (4096, 4096, 2880, "token", "tensor"),
    (64, 4096, 4096, "token", "channel"),
    (128, 2880, 5120, "token", "channel"),
    (4096, 4096, 2880, "token", "channel"),
]

_CONFIGS_scaled_mm_cute_sm103 = [
    # (M, K, N) = (64, 4096, 4096), scale_a=tensor, scale_b=tensor
    {
        **_BASE,
        "block_sizes": [64, 32, 256],
        "cute_min_blocks_per_mp": 1,
        "indexing": ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor", "pointer", "pointer"],
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_blocked",
        "tcgen05_ab_stages": 9,
        "tcgen05_acc_stages": 1,
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 1,
        "tcgen05_cta_group": "auto",
        "tcgen05_l2_swizzle_size": 4,
    },
    # (M, K, N) = (128, 2880, 5120), scale_a=tensor, scale_b=tensor
    {
        **_BASE,
        "block_sizes": [128, 128, 64],
        "cute_min_blocks_per_mp": 0,
        "indexing": ["pointer", "tensor_descriptor", "pointer", "tensor_descriptor", "pointer"],
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_blocked",
        "tcgen05_ab_stages": 16,
        "tcgen05_acc_stages": 1,
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 2,
        "tcgen05_cta_group": "two",
        "tcgen05_l2_swizzle_size": 1,
        "tcgen05_tile_n": 0,
    },
    # (M, K, N) = (4096, 4096, 2880), scale_a=tensor, scale_b=tensor
    {
        **_ALIGNED_N,
        "cute_min_blocks_per_mp": 0,
        "indexing": ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
    },
    # (M, K, N) = (64, 4096, 4096), scale_a=tensor, scale_b=channel
    {
        **_BASE,
        "block_sizes": [64, 32, 256],
        "cute_min_blocks_per_mp": 1,
        "indexing": ["pointer", "pointer", "pointer", "pointer", "pointer"],
        "l2_groupings": [1],
        "loop_orders": [[1, 0]],
        "pid_type": "persistent_blocked",
        "tcgen05_ab_stages": 8,
        "tcgen05_acc_stages": 1,
        "tcgen05_aux_load_placement": "pre_acc_wait",
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "direct",
        "tcgen05_cluster_m": 1,
        "tcgen05_cta_group": "auto",
        "tcgen05_l2_swizzle_size": 2,
    },
    # (M, K, N) = (128, 2880, 5120), scale_a=tensor, scale_b=channel
    {
        **_BASE,
        "block_sizes": [128, 128, 128],
        "cute_min_blocks_per_mp": 0,
        "indexing": ["pointer", "pointer", "pointer", "pointer", "pointer"],
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_blocked",
        "tcgen05_ab_stages": 12,
        "tcgen05_acc_stages": 2,
        "tcgen05_aux_load_placement": "pre_acc_wait",
        "tcgen05_c_stages": 4,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 2,
        "tcgen05_cta_group": "two",
        "tcgen05_l2_swizzle_size": 1,
        "tcgen05_tile_n": 0,
    },
    # (M, K, N) = (4096, 4096, 2880), scale_a=tensor, scale_b=channel
    {
        **_ALIGNED_N,
        "cute_min_blocks_per_mp": 1,
        "indexing": ["tensor_descriptor", "pointer", "tensor_descriptor", "pointer", "pointer"],
    },
    # (M, K, N) = (64, 4096, 4096), scale_a=token, scale_b=tensor
    {
        **_BASE,
        "block_sizes": [64, 32, 256],
        "cute_min_blocks_per_mp": 0,
        "indexing": ["pointer", "pointer", "tensor_descriptor", "tensor_descriptor", "pointer"],
        "l2_groupings": [32],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_interleaved",
        "tcgen05_ab_stages": 9,
        "tcgen05_acc_stages": 1,
        "tcgen05_aux_load_placement": "post_acc_wait",
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 1,
        "tcgen05_cta_group": "auto",
        "tcgen05_l2_swizzle_size": 2,
    },
    # (M, K, N) = (128, 2880, 5120), scale_a=token, scale_b=tensor
    {
        **_BASE,
        "block_sizes": [128, 128, 64],
        "cute_min_blocks_per_mp": 0,
        "indexing": ["pointer", "tensor_descriptor", "tensor_descriptor", "tensor_descriptor", "pointer"],
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_blocked",
        "tcgen05_ab_stages": 16,
        "tcgen05_acc_stages": 1,
        "tcgen05_aux_load_placement": "pre_acc_wait",
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 2,
        "tcgen05_cta_group": "two",
        "tcgen05_l2_swizzle_size": 1,
        "tcgen05_tile_n": 0,
    },
    # (M, K, N) = (4096, 4096, 2880), scale_a=token, scale_b=tensor
    {
        **_ALIGNED_N,
        "cute_min_blocks_per_mp": 1,
        "indexing": ["pointer", "tensor_descriptor", "tensor_descriptor", "tensor_descriptor", "pointer"],
    },
    # (M, K, N) = (64, 4096, 4096), scale_a=token, scale_b=channel
    {
        **_BASE,
        "block_sizes": [64, 32, 256],
        "cute_min_blocks_per_mp": 0,
        "indexing": ["pointer", "pointer", "tensor_descriptor", "pointer", "tensor_descriptor"],
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_blocked",
        "tcgen05_ab_stages": 8,
        "tcgen05_acc_stages": 1,
        "tcgen05_aux_load_placement": "pre_acc_wait",
        "tcgen05_c_stages": 4,
        "tcgen05_c_store_mode": "direct",
        "tcgen05_cluster_m": 1,
        "tcgen05_cta_group": "auto",
        "tcgen05_l2_swizzle_size": 1,
    },
    # (M, K, N) = (128, 2880, 5120), scale_a=token, scale_b=channel
    {
        **_BASE,
        "block_sizes": [128, 128, 64],
        "cute_min_blocks_per_mp": 0,
        "indexing": ["pointer", "pointer", "pointer", "pointer", "tensor_descriptor"],
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_blocked",
        "tcgen05_ab_stages": 12,
        "tcgen05_acc_stages": 1,
        "tcgen05_aux_load_placement": "pre_acc_wait",
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 2,
        "tcgen05_cta_group": "two",
        "tcgen05_l2_swizzle_size": 1,
        "tcgen05_tile_n": 0,
    },
    # (M, K, N) = (4096, 4096, 2880), scale_a=token, scale_b=channel
    {
        **_ALIGNED_N,
        "cute_min_blocks_per_mp": 1,
        "indexing": ["pointer", "tensor_descriptor", "pointer", "pointer", "pointer"],
    },
]


def key_scaled_mm_cute_sm103(*args) -> int:
    """Config index for the given args (also the cache key)."""
    key = _key(args)
    if key[0] < 64:
        raise ValueError(
            f"scaled_mm_cute_sm103 has no GB300 config for M={key[0]}: every pretuned "
            "multi-row schedule needs M >= 64. Use scaled_mm_skinny_cute for M=1 "
            "or a swapped (N, M) GEMM for 1 < M < 64."
        )
    return _select(_KEYS_scaled_mm_cute_sm103, key)


def autotune_scaled_mm_cute_sm103(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_scaled_mm_cute_sm103[key_scaled_mm_cute_sm103(*args)]


# === Kernel: scaled_mm_transposed_cute ===
# Keys retain logical (M, K, N) and the original scale order.
# Config blocks use the frontend kernel's physical (N, M, K) axes.
_KEYS_scaled_mm_transposed_cute = [
    (256, 4096, 2048, "tensor", "tensor"),
    (256, 4096, 2048, "tensor", "channel"),
    (256, 4096, 2048, "token", "tensor"),
    (256, 4096, 2048, "token", "channel"),
]

_CONFIGS_scaled_mm_transposed_cute = [
    # Logical (M, K, N) = (256, 4096, 2048), scale_a=tensor, scale_b=tensor
    {
        **_BASE,
        "block_sizes": [128, 64, 256],
        "cute_min_blocks_per_mp": 0,
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_interleaved",
        "tcgen05_ab_stages": 8,
        "tcgen05_acc_stages": 2,
        "tcgen05_aux_load_placement": "post_acc_wait",
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 2,
        "tcgen05_cta_group": "two",
        "tcgen05_l2_swizzle_size": 1,
    },
    # Logical (M, K, N) = (256, 4096, 2048), scale_a=tensor, scale_b=channel
    {
        **_BASE,
        "block_sizes": [128, 64, 256],
        "cute_min_blocks_per_mp": 0,
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_interleaved",
        "tcgen05_ab_stages": 8,
        "tcgen05_acc_stages": 1,
        "tcgen05_aux_load_placement": "post_acc_wait",
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 2,
        "tcgen05_cta_group": "two",
        "tcgen05_l2_swizzle_size": 1,
    },
    # Logical (M, K, N) = (256, 4096, 2048), scale_a=token, scale_b=tensor
    {
        **_BASE,
        "block_sizes": [128, 64, 256],
        "cute_min_blocks_per_mp": 0,
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_interleaved",
        "tcgen05_ab_stages": 8,
        "tcgen05_acc_stages": 1,
        "tcgen05_aux_load_placement": "post_acc_wait",
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 2,
        "tcgen05_cta_group": "two",
        "tcgen05_l2_swizzle_size": 1,
    },
    # Logical (M, K, N) = (256, 4096, 2048), scale_a=token, scale_b=channel
    {
        **_BASE,
        "block_sizes": [128, 64, 256],
        "cute_min_blocks_per_mp": 0,
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "pid_type": "persistent_interleaved",
        "tcgen05_ab_stages": 8,
        "tcgen05_acc_stages": 1,
        "tcgen05_aux_load_placement": "post_acc_wait",
        "tcgen05_c_stages": 2,
        "tcgen05_c_store_mode": "normal",
        "tcgen05_cluster_m": 2,
        "tcgen05_cta_group": "two",
        "tcgen05_l2_swizzle_size": 1,
    },
]


def _transposed_key(args):
    """Logical key from b_t[N, K], a_t[K, M], scale_a and scale_b."""
    b_t, a_t, scale_a, scale_b = args
    return (
        int(a_t.shape[1]),
        int(a_t.shape[0]),
        int(b_t.shape[0]),
        "tensor" if scale_a.ndim == 0 else "token",
        "tensor" if scale_b.ndim == 0 else "channel",
    )


def key_scaled_mm_transposed_cute(*args) -> int:
    """Exact config index for the frontend-transposed implementation."""
    key = _transposed_key(args)
    if key not in _KEYS_scaled_mm_transposed_cute:
        raise ValueError(f"scaled_mm_transposed_cute has no GB300 config for {key}")
    return _KEYS_scaled_mm_transposed_cute.index(key)


def autotune_scaled_mm_transposed_cute(*args) -> dict:
    """Config dict for the given transposed input views and original scales."""
    return _CONFIGS_scaled_mm_transposed_cute[key_scaled_mm_transposed_cute(*args)]
