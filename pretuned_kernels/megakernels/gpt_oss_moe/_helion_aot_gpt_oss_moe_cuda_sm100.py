"""Checked-in B200 config for the GPT-OSS 120B MXFP4 MoE megakernel."""

from __future__ import annotations

from copy import deepcopy

import torch


CONFIG = {
    "block_sizes": [32, 8, 32, 256],
    "loop_orders": [[0, 1, 2, 3], [0, 1]],
    "l2_groupings": [1, 1],
    "reduction_loops": [None],
    "range_unroll_factors": [0, 0, 0, 0, 0, 0],
    "range_warp_specializes": [None, None, None, None, None, None],
    "range_num_stages": [0, 0, 0, 0, 0, 0],
    "range_multi_buffers": [None, None, None, None, None, None],
    "range_flattens": [None, None, None, None, None, None],
    "load_eviction_policies": ["last"] * 19,
    "num_warps": 1,
    "num_stages": 1,
    "indexing": ["pointer"] * 26,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "static",
    "num_sm_multiplier": 16,
    "maxnreg": 128,
    "atomic_indexing": [],
}


_TENSOR_SIGNATURES = (
    ((1, 128), torch.bfloat16),
    ((1, 3072), torch.bfloat16),
    ((128, 5888, 1536), torch.float4_e2m1fn_x2),
    ((128, 5888, 96), torch.uint8),
    ((128, 5888), torch.float32),
    ((128, 3072, 1472), torch.float4_e2m1fn_x2),
    ((128, 3072, 92), torch.uint8),
    ((128, 3072), torch.float32),
)
_STATIC_ARGS = (2880,)


def key_gpt_oss_moe(*args) -> int:
    """This pretuned module intentionally supports one production shape."""
    tensor_count = len(_TENSOR_SIGNATURES)
    if (
        len(args) != tensor_count + len(_STATIC_ARGS)
        or tuple((tuple(arg.shape), arg.dtype) for arg in args[:tensor_count])
        != _TENSOR_SIGNATURES
        or args[tensor_count:] != _STATIC_ARGS
    ):
        raise ValueError("gpt_oss_moe is pretuned only for GPT-OSS 120B batch one")
    return 0


def autotune_gpt_oss_moe(*args) -> dict[str, object]:
    """Return the validated B200 static-pipeline configuration."""
    key_gpt_oss_moe(*args)
    return deepcopy(CONFIG)
