"""
Auto-generated heuristic for kernels: nvfp4_gemv_fp4in_kernel, nvfp4_gemv_bf16in_kernel
Backend: fp4in single config; bf16in per-shape (N, K) table with nearest fallback

NVFP4 decode GEMV pretuned on NVIDIA B200 (sm100) for Helion's Triton backend:
FP16-decode multi-row bodies that tile over both M (out[tile_m]) and the K
scale-group dim, accumulating K tiles into a per-row fp32 acc. The weight is
loaded as one coalesced int64 per scale group (16 packed fp4). Configs were
autotuned (curated grid, validated against the dequant reference under cold-L2
cudagraph) over the decode weight shapes in the sweep.

fp4in: weight *and* activation are packed 4-bit, so DRAM traffic dominates and
one config transfers across shapes.

bf16in: the best (block_m, block_g) trades off with the (N, K) shape -- small-N /
large-K shapes want block_g=256 (bm=16), mid shapes want bm=16/bg=128, and the
device-filling shapes want bm=32/bg=128 -- so it uses a per-shape table with a
nearest-(N, K) fallback (smallest sum of absolute log-ratios).

Provides, for each kernel <k>:
- key_<k>(*args): config index (also the runtime cache key)
- autotune_<k>(*args): config dict for the given arguments
"""

import math


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


# W4A16 (NVFP4 weight * BF16 activation). Per-shape best-of grid (warmed, cold-L2
# cudagraph, validated vs the dequant reference). weight_i64 is [N, K/16], so
# (N, K) = (shape[0], shape[1] * 16). Three configs split the sweep:
#   [16, 256] w4 s3 -- small-N / large-K (more K per tile, fewer M tiles)
#   [16, 128] w4 s2 -- mid shapes
#   [32, 128] w4 s2 -- device-filling shapes (K=4096 wide-N, K>=8192 square)
# Tuned (N, K) shapes, parallel to _CONFIGS_bf16in.
_KEYS_bf16in = [
    (4096, 4096),
    (6144, 4096),
    (28672, 4096),
    (4096, 14336),
    (5120, 5120),
    (15360, 5120),
    (8192, 8192),
    (10240, 8192),
    (8192, 28672),
]

_CFG_16_256 = {"block_sizes": [16, 256], "num_warps": 4, "num_stages": 3}
_CFG_16_128 = {"block_sizes": [16, 128], "num_warps": 4, "num_stages": 2}
_CFG_32_128 = {"block_sizes": [32, 128], "num_warps": 4, "num_stages": 2}

_CONFIGS_bf16in = [
    _CFG_16_256,  # (4096, 4096)   7.79us
    _CFG_16_256,  # (6144, 4096)   9.53us
    _CFG_32_128,  # (28672, 4096)  40.45us
    _CFG_16_256,  # (4096, 14336)  22.73us
    _CFG_16_128,  # (5120, 5120)   13.06us
    _CFG_16_128,  # (15360, 5120)  26.04us
    _CFG_32_128,  # (8192, 8192)   20.16us
    _CFG_16_128,  # (10240, 8192)  24.20us
    _CFG_32_128,  # (8192, 28672)  63.11us
]


def _nk_bf16in(args):
    """(N, K) from bf16in args: weight_i64 is [N, K // 16]."""
    w = args[0]
    return int(w.shape[0]), int(w.shape[1]) * 16


def key_nvfp4_gemv_bf16in_kernel(*args) -> int:
    """Config index: exact (N, K) match if tuned, else the nearest tuned shape."""
    nk = _nk_bf16in(args)
    for i, k in enumerate(_KEYS_bf16in):
        if tuple(k) == nk:
            return i
    best_i, best_d = 0, float("inf")
    for i, k in enumerate(_KEYS_bf16in):
        d = sum(abs(math.log(max(a, 1)) - math.log(max(b, 1))) for a, b in zip(k, nk))
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def autotune_nvfp4_gemv_bf16in_kernel(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_bf16in[key_nvfp4_gemv_bf16in_kernel(*args)]
