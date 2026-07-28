"""
Auto-generated heuristic for kernels: nvfp4_gemv_fp4in_kernel, nvfp4_gemv_bf16in_kernel
Backend: single tuned config per kernel (shape-independent)

NVFP4 decode GEMV pretuned on NVIDIA B200 (sm100) for Helion's Triton backend:
FP16-decode multi-row bodies from pytorch/helion#3079 that tile over both M
(out[tile_m]) and the K scale-group dim, accumulating K tiles into a per-row
fp32 acc. Packed FP4 groups are loaded through
hl.load_float4_e2m1fn_x16_to_float16. Configs match the PR's tuned Triton
configs, validated against the dequant reference under cold-L2 cudagraph.

Provides, for each kernel <k>:
- key_<k>(*args): config index (also the runtime cache key)
- autotune_<k>(*args): config dict for the given arguments
"""

# W4A4 (NVFP4 weight * NVFP4 activation). Autotuned on N=8192, K=28672 ("down"):
# block_m=4 + inner block_g=256, num_stages=3 pipelining, K-loop multi-buffering.
# Validated against the dequant reference (38.8us cold-L2 cudagraph).
_CONFIG_nvfp4_gemv_fp4in_kernel = {
    "block_sizes": [4, 256],
    "num_warps": 2,
    "num_stages": 3,
    "range_multi_buffers": [None, True],
}


def key_nvfp4_gemv_fp4in_kernel(*args) -> int:
    """Config index for the given args (also the cache key). One config."""
    return 0


def autotune_nvfp4_gemv_fp4in_kernel(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIG_nvfp4_gemv_fp4in_kernel


# W4A16 (NVFP4 weight * BF16 activation). PR #3079's tuned coalesced-load
# config: block_m=16, inner block_g=128, with three pipeline stages.
_CONFIG_nvfp4_gemv_bf16in_kernel = {
    "block_sizes": [16, 128],
    "num_warps": 4,
    "num_stages": 3,
}


def key_nvfp4_gemv_bf16in_kernel(*args) -> int:
    """Config index for the given args (also the cache key). One config."""
    return 0


def autotune_nvfp4_gemv_bf16in_kernel(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIG_nvfp4_gemv_bf16in_kernel
