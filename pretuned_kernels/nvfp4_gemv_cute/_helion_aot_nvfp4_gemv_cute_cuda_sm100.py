"""
Pretuned B200 configs for the Helion CuTe NVFP4 GEMV kernels.

These are actual ``@helion.aot_kernel(backend="cute")`` kernels. The defaults
use 32 groups per block; W4A4 ``(N=4096, K=14336)`` uses 64 groups because its
long reduction benefits more from fewer loop iterations than from occupancy.
"""

_CONFIG_BF16IN = {
    "block_sizes": [32],
    "indexing": ["pointer"] * 8,
    "load_eviction_policies": [
        "first",
        "last",
        "last",
        "last",
        "last",
        "first",
    ],
    "num_threads": [32],
    "num_warps": 4,
    "num_stages": 1,
    "pid_type": "flat",
    "range_warp_specializes": [None],
}

_CONFIG_FP4IN = {
    "block_sizes": [32],
    "indexing": ["pointer"] * 5,
    "load_eviction_policies": ["first", "last", "", "last"],
    "num_threads": [32],
    "num_warps": 2,
    "num_stages": 3,
    "pid_type": "flat",
    "range_warp_specializes": [None],
}

_CONFIG_FP4IN_LONG_K = {**_CONFIG_FP4IN, "block_sizes": [64]}


def key_nvfp4_gemv_bf16in_rows2_kernel(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return 0


def autotune_nvfp4_gemv_bf16in_rows2_kernel(*args) -> dict:
    """Return the checked-in W4A16 CuTe config."""
    return _CONFIG_BF16IN


def key_nvfp4_gemv_bf16in_rows4_kernel(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return 0


def autotune_nvfp4_gemv_bf16in_rows4_kernel(*args) -> dict:
    """Return the checked-in W4A16 CuTe config."""
    return _CONFIG_BF16IN


def key_nvfp4_gemv_fp4in_rows2_kernel(*args) -> int:
    """Config index for the given args (also the cache key)."""
    weight_bytes = args[0]
    return int(weight_bytes.shape == (4096, 14336 // 2))


def autotune_nvfp4_gemv_fp4in_rows2_kernel(*args) -> dict:
    """Return the checked-in W4A4 CuTe config."""
    if key_nvfp4_gemv_fp4in_rows2_kernel(*args):
        return _CONFIG_FP4IN_LONG_K
    return _CONFIG_FP4IN


def key_nvfp4_gemv_fp4in_rows4_kernel(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return 0


def autotune_nvfp4_gemv_fp4in_rows4_kernel(*args) -> dict:
    """Return the checked-in W4A4 CuTe config."""
    return _CONFIG_FP4IN
