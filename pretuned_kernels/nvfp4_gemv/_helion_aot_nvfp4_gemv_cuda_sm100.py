"""
Auto-generated heuristic for kernels: nvfp4_gemv_fp4in_kernel, nvfp4_gemv_bf16in_kernel
Backend: single tuned config per kernel (shape-independent)

NVFP4 decode (M=1) GEMV pretuned on NVIDIA B200 (sm100) for Helion's Triton
backend. These are the tuned Triton configs from PR #2738 (the "improved Triton
nvfp4_gemv"): a block over N with a wide K reduction tile. The GEMV is memory
bound on streaming the NVFP4 weight from DRAM, so one config transfers across
the decode weight shapes in the sweep -- every (N, K) maps to the same config.

Provides, for each kernel <k>:
- key_<k>(*args): config index (also the runtime cache key)
- autotune_<k>(*args): config dict for the given arguments
"""


# W4A4 (NVFP4 weight * NVFP4 activation) -- PR #2738 FP4IN_TRITON_CONFIG.
_CONFIG_nvfp4_gemv_fp4in_kernel = {
    "block_sizes": [1, 128],
    "num_warps": 4,
    "num_stages": 3,
}

# W4A16 (NVFP4 weight * BF16 activation) -- PR #2738 BF16IN_TRITON_CONFIG.
_CONFIG_nvfp4_gemv_bf16in_kernel = {
    "block_sizes": [1, 512],
    "num_warps": 1,
    "num_stages": 2,
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
