"""
Pretuned B200 configs for the Helion CuTe NVFP4 GEMV kernels.

These are actual ``@helion.aot_kernel(backend="cute")`` kernels. The configs
come from the portable CuTe paths in ``examples/nvfp4_gemv.py`` and are used as
shape-independent defaults for the decode sweep.
"""

_CONFIG_BF16IN = {
    "block_sizes": [1, 128],
    "indexing": ["pointer"] * 8,
    "load_eviction_policies": [
        "first",
        "last",
        "last",
        "last",
        "last",
        "first",
    ],
    "num_threads": [1, 128],
    "num_warps": 4,
    "num_stages": 1,
    "pid_type": "flat",
    "range_warp_specializes": [None],
}

_CONFIG_FP4IN = {
    "block_sizes": [1, 128],
    "indexing": ["pointer"] * 5,
    "load_eviction_policies": ["first", "last", "", "last"],
    "num_threads": [1, 64],
    "num_warps": 2,
    "num_stages": 3,
    "pid_type": "flat",
    "range_warp_specializes": [None],
}


def key_nvfp4_gemv_bf16in_kernel(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return 0


def autotune_nvfp4_gemv_bf16in_kernel(*args) -> dict:
    """Return the checked-in W4A16 CuTe config."""
    return _CONFIG_BF16IN


def key_nvfp4_gemv_fp4in_kernel(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return 0


def autotune_nvfp4_gemv_fp4in_kernel(*args) -> dict:
    """Return the checked-in W4A4 CuTe config."""
    return _CONFIG_FP4IN
