"""
Auto-generated heuristic for kernels: nvfp4_gemv_fp4in_kernel, nvfp4_gemv_bf16in_kernel
Backend: PR defaults with exact-shape overrides for CUTLASS losses

NVFP4 decode GEMV pretuned on NVIDIA B200 (sm100) for Helion's Triton backend:
FP16-decode multi-row bodies from pytorch/helion#3079 that tile over both M
(out[tile_m]) and the K scale-group dim, accumulating K tiles into a per-row
fp32 acc. Packed FP4 groups are loaded through
hl.load_float4_e2m1fn_x16_to_float16. The sweep-wide defaults match the PR's
tuned Triton configs. Full AOT searches added exact-shape overrides for the
three shapes where those defaults lost to CUTLASS. All configs were validated
against the dequant reference and remeasured under cold-L2 cudagraph.

Provides, for each kernel <k>:
- key_<k>(*args): config index (also the runtime cache key)
- autotune_<k>(*args): config dict for the given arguments
"""

# W4A4 (NVFP4 weight * NVFP4 activation). Autotuned on N=8192, K=28672 ("down"):
# block_m=4 + inner block_g=256, num_stages=3 pipelining, K-loop multi-buffering.
# Validated against the dequant reference (38.8us cold-L2 cudagraph).
_CONFIG_FP4IN_DEFAULT = {
    "block_sizes": [4, 256],
    "num_warps": 2,
    "num_stages": 3,
    "range_multi_buffers": [None, True],
}

# N=15360, K=5120: a focused post-autotune grid found this simple config faster
# than both the AOT finalist and CUTLASS (15.32us vs 16.72us and 17.45us).
_CONFIG_FP4IN_15360_5120 = {
    "block_sizes": [16, 64],
    "num_warps": 4,
    "num_stages": 4,
}


def _nk(args):
    """Return logical (N, K); packed byte storage has shape [N, K // 2]."""
    weight = args[0]
    return int(weight.shape[0]), int(weight.shape[1]) * 2


def key_nvfp4_gemv_fp4in_kernel(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return 1 if _nk(args) == (15360, 5120) else 0


def autotune_nvfp4_gemv_fp4in_kernel(*args) -> dict:
    """Config dict for the given args."""
    return [_CONFIG_FP4IN_DEFAULT, _CONFIG_FP4IN_15360_5120][
        key_nvfp4_gemv_fp4in_kernel(*args)
    ]


# W4A16 (NVFP4 weight * BF16 activation). PR #3079's tuned coalesced-load
# config: block_m=16, inner block_g=128, with three pipeline stages.
_CONFIG_BF16IN_DEFAULT = {
    "block_sizes": [16, 128],
    "num_warps": 4,
    "num_stages": 3,
}

# N=28672, K=4096: 28.12us vs 30.53us default and 29.13us CUTLASS.
_CONFIG_BF16IN_28672_4096 = {
    "block_sizes": [16, 64],
    "range_unroll_factors": [0, 3],
    "range_warp_specializes": [None, False],
    "range_num_stages": [],
    "range_multi_buffers": [None, True],
    "range_flattens": [None, None],
    "load_eviction_policies": [
        "last", "first", "last", "", "last", "last", "last", "first",
        "first", "first", "first", "last", "", "first", "first", "first",
        "", "", "last", "first", "first",
    ],
    "num_warps": 2,
    "num_stages": 3,
    "indexing": [
        "pointer", "pointer", "pointer", "tensor_descriptor",
        "tensor_descriptor", "tensor_descriptor", "pointer",
        "tensor_descriptor", "tensor_descriptor", "tensor_descriptor",
        "tensor_descriptor", "tensor_descriptor", "tensor_descriptor",
        "pointer", "tensor_descriptor", "tensor_descriptor",
        "tensor_descriptor", "pointer", "pointer", "pointer", "pointer",
        "pointer",
    ],
    "atomic_indexing": [],
    "pid_type": "flat",
}

# N=8192, K=28672: 58.04us vs 59.13us default and 58.16us CUTLASS.
_CONFIG_BF16IN_8192_28672 = {
    "block_sizes": [16, 128],
    "range_unroll_factors": [3, 4],
    "range_warp_specializes": [False, False],
    "range_multi_buffers": [None, False],
    "range_flattens": [False, None],
    "load_eviction_policies": [
        "first", "last", "", "", "", "first", "", "first", "", "last", "",
        "", "", "", "", "", "", "first", "first", "first", "last",
    ],
    "num_warps": 4,
    "num_stages": 5,
    "indexing": [
        "pointer", "tensor_descriptor", "pointer", "pointer", "pointer",
        "tensor_descriptor", "pointer", "tensor_descriptor",
        "tensor_descriptor", "tensor_descriptor", "tensor_descriptor",
        "tensor_descriptor", "pointer", "pointer", "pointer",
        "tensor_descriptor", "tensor_descriptor", "pointer", "pointer",
        "pointer", "tensor_descriptor", "pointer",
    ],
    "atomic_indexing": [],
    "pid_type": "persistent_interleaved",
    "num_sm_multiplier": 64,
}

# N=15360, K=5120 uses the default config because its warp-specialized override
# triggers triton-lang/triton#10901.
_BF16IN_EXACT = {
    (28672, 4096): 1,
    (8192, 28672): 2,
}

_BF16IN_CONFIGS = [
    _CONFIG_BF16IN_DEFAULT,
    _CONFIG_BF16IN_28672_4096,
    _CONFIG_BF16IN_8192_28672,
]


def key_nvfp4_gemv_bf16in_kernel(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _BF16IN_EXACT.get(_nk(args), 0)


def autotune_nvfp4_gemv_bf16in_kernel(*args) -> dict:
    """Config dict for the given args."""
    return _BF16IN_CONFIGS[key_nvfp4_gemv_bf16in_kernel(*args)]
