"""
Auto-generated heuristic for kernels: nvfp4_gemv_fp4in_kernel, nvfp4_gemv_bf16in_kernel
Backend: single tuned config per kernel (shape-independent)

NVFP4 decode GEMV pretuned on NVIDIA B200 (sm100) for Helion's Triton backend:
FP16-decode multi-row bodies that tile over both M (out[tile_m]) and the K
scale-group dim, accumulating K tiles into a per-row fp32 acc. Configs were
autotuned (curated grid, validated against the dequant reference under cold-L2
cudagraph) on the nvfp4_backend_comparison shapes -- fp4in "down" (N=8192,
K=28672) and bf16in across N=8192 with K in {8192, 28672}. The GEMV is memory
bound on streaming the NVFP4 weight from DRAM, so one config per kernel
transfers across the decode weight shapes in the sweep.

Provides, for each kernel <k>:
- key_<k>(*args): config index (also the runtime cache key)
- autotune_<k>(*args): config dict for the given arguments
"""


# W4A4 (NVFP4 weight * NVFP4 activation). Autotuned on N=8192, K=28672 ("down"):
# block_m=4 + inner block_g=256, num_stages=3 pipelining, K-loop multi-buffering.
# These multi-row bodies tile over M (out[tile_m]), so block_m>1 is correct and
# reused across output rows. Validated against the dequant reference (38.8us cold-L2
# cudagraph); beat the deeper num_stages=6 the gist baked for this shape.
_CONFIG_nvfp4_gemv_fp4in_kernel = {
    "block_sizes": [4, 256],
    "num_warps": 2,
    "num_stages": 3,
    "range_multi_buffers": [None, True],
}

# W4A16 (NVFP4 weight * BF16 activation). block_m=16 (activation reuse across the
# 16 output rows) + inner K tile block_g=128 so register pressure is bounded by the
# tile, not the full K -- the untiled body spilled catastrophically at large K
# (1298us at K=28672). Tuned across N=8192 K in {8192, 28672} and validated against
# the dequant reference: 27us at K=8192, 83us at K=28672 (cold-L2 cudagraph).
_CONFIG_nvfp4_gemv_bf16in_kernel = {
    "block_sizes": [16, 128],
    "num_warps": 2,
    "num_stages": 3,
}


def key_nvfp4_gemv_fp4in_kernel(*args) -> int:
    """Config index for the given args (also the cache key). One config."""
    return 0


def autotune_nvfp4_gemv_fp4in_kernel(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIG_nvfp4_gemv_fp4in_kernel


def key_nvfp4_gemv_bf16in_kernel(*args) -> int:
    """Config index for the given args (also the cache key). One config."""
    return 0


def autotune_nvfp4_gemv_bf16in_kernel(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIG_nvfp4_gemv_bf16in_kernel
