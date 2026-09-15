"""Checked-in B200 config for the DeepSeek-V3 FP8 MoE megakernel."""

from __future__ import annotations

from copy import deepcopy

import torch

CONFIG = {
    "atomic_indexing": [],
    "block_sizes": [2, 512, 2, 2, 16, 16, 1, 1, 8, 16, 64, 256],
    "cross_loop_pipeline": "dynamic",
    "flatten_loops": [False],
    "indexing": ["pointer"] * 73,
    "l2_groupings": [1] * 9,
    "load_eviction_policies": [""] * 42,
    "loop_orders": [
        [0, 1],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1],
        [1, 0],
        [0, 1],
    ],
    "maxnreg": None,
    "num_sm_multiplier": 1,
    "num_stages": 1,
    "num_warps": 1,
    "pid_type": "persistent_blocked",
    "range_flattens": [None] * 17,
    "range_multi_buffers": [None] * 17,
    "range_num_stages": [0, 6, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 3, 0, 5, 0, 0],
    "range_unroll_factors": [0] * 17,
    "range_warp_specializes": [None] * 17,
}


_TENSOR_SIGNATURES = (
    ((1, 7168), torch.bfloat16),
    ((256, 7168), torch.bfloat16),
    ((256,), torch.float32),
    ((256, 4096, 7168), torch.float8_e4m3fn),
    ((256, 32, 56), torch.float32),
    ((256, 7168, 2048), torch.float8_e4m3fn),
    ((256, 56, 16), torch.float32),
    ((4096, 7168), torch.float8_e4m3fn),
    ((32, 56), torch.float32),
    ((7168, 2048), torch.float8_e4m3fn),
    ((56, 16), torch.float32),
)
_STATIC_ARGS = (8, 8, 4, 2.5)


def key_deepseek_v3_moe_fp8(*args) -> int:
    """This pretuned module intentionally supports one production shape."""
    tensor_count = len(_TENSOR_SIGNATURES)
    if (
        len(args) != tensor_count + len(_STATIC_ARGS)
        or tuple((tuple(arg.shape), arg.dtype) for arg in args[:tensor_count])
        != _TENSOR_SIGNATURES
        or args[tensor_count:] != _STATIC_ARGS
    ):
        raise ValueError("deepseek_v3_moe_fp8 is pretuned only for B1 DeepSeek-V3")
    return 0


def autotune_deepseek_v3_moe_fp8(*args) -> dict[str, object]:
    """Return the validated B200 dynamic-dispatch configuration."""
    key_deepseek_v3_moe_fp8(*args)
    return deepcopy(CONFIG)
