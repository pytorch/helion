"""Checked-in B200 config for the DeepSeek-V3 NVFP4 MoE megakernel."""

from __future__ import annotations

from copy import deepcopy

import torch


CONFIG = {
    "atomic_indexing": [],
    "block_sizes": [8, 512, 32, 4, 32, 256, 16, 16, 64, 32, 16, 32, 32],
    "cross_loop_pipeline": "dynamic",
    "host_tensor_descriptors": True,
    "indexing": [
        "tensor_descriptor" if index in (39, 56, 58) else "pointer"
        for index in range(74)
    ],
    "l2_groupings": [1, 1, 1, 1, 4],
    "load_eviction_policies": [""] * 46,
    "loop_orders": [[0, 1], [0, 1], [0, 1, 2], [0, 1], [1, 0]],
    "maxnreg": None,
    "num_sm_multiplier": 1,
    "num_stages": 1,
    "num_warps": 4,
    "pid_type": "persistent_blocked",
    "range_flattens": [
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        None,
        False,
        None,
        None,
        False,
        None,
        False,
        None,
    ],
    "range_multi_buffers": [
        None,
        None,
        None,
        None,
        True,
        None,
        None,
        False,
        True,
        None,
        None,
        True,
        None,
        True,
        None,
    ],
    "range_num_stages": [0, 4, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0],
    "range_unroll_factors": [0] * 15,
    "range_warp_specializes": [
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        None,
        False,
        None,
        None,
        False,
        None,
        False,
        None,
    ],
}


_TENSOR_SIGNATURES = (
    ((1, 7168), torch.bfloat16),
    ((256, 7168), torch.bfloat16),
    ((256,), torch.float32),
    ((256, 4096, 3584), torch.uint8),
    ((256, 4096, 448), torch.float8_e4m3fn),
    ((256, 7168, 1024), torch.uint8),
    ((256, 7168, 128), torch.float8_e4m3fn),
    ((4096, 3584), torch.uint8),
    ((4096, 448), torch.float8_e4m3fn),
    ((7168, 1024), torch.uint8),
    ((7168, 128), torch.float8_e4m3fn),
    ((256,), torch.float32),
    ((256,), torch.float32),
    ((1,), torch.float32),
    ((1,), torch.float32),
    ((1,), torch.float32),
    ((1,), torch.float32),
)
_STATIC_ARGS = (8, 8, 4, 2.5, 7)


def key_deepseek_v3_moe_nvfp4(*args) -> int:
    """This pretuned module intentionally supports one production shape."""
    tensor_count = len(_TENSOR_SIGNATURES)
    if (
        len(args) != tensor_count + len(_STATIC_ARGS)
        or tuple((tuple(arg.shape), arg.dtype) for arg in args[:tensor_count])
        != _TENSOR_SIGNATURES
        or args[tensor_count:] != _STATIC_ARGS
    ):
        raise ValueError("deepseek_v3_moe_nvfp4 is pretuned only for B1 DeepSeek-V3")
    return 0


def autotune_deepseek_v3_moe_nvfp4(*args) -> dict[str, object]:
    """Return the validated B200 host-TMA dynamic configuration."""
    key_deepseek_v3_moe_nvfp4(*args)
    return deepcopy(CONFIG)
