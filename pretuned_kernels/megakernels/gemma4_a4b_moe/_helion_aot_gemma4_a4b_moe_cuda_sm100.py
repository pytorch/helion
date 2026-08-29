"""Checked-in B200 config for the Gemma 4 A4B MoE megakernel."""

from __future__ import annotations

from copy import deepcopy

import torch


CONFIG = {'block_sizes': [8, 16, 256, 128, 64, 64, 256], 'loop_orders': [[0, 1], [0, 1], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1]], 'l2_groupings': [1, 1, 1, 1, 1, 1], 'reduction_loops': [None, None], 'range_unroll_factors': [0, 0, 0, 0, 0, 0, 0, 0, 0, 4], 'range_warp_specializes': [None, None, None, None, None, None, None, None, None, None], 'range_num_stages': [0, 0, 0, 0, 3, 0, 0, 5, 1, 0], 'range_multi_buffers': [None, None, None, None, True, None, None, False, None, None], 'range_flattens': [None, None, None, None, False, None, None, False, None, None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'cross_loop_schedule': 'static_pipeline', 'num_sm_multiplier': 4, 'maxnreg': 128, 'atomic_indexing': []}


_TENSOR_SIGNATURES = (
    ((1, 2816), torch.bfloat16),
    ((2816,), torch.bfloat16),
    ((2816,), torch.bfloat16),
    ((), torch.bfloat16),
    ((128, 2816), torch.bfloat16),
    ((128,), torch.bfloat16),
    ((128, 1408, 2816), torch.bfloat16),
    ((128, 2816, 704), torch.bfloat16),
    ((2816,), torch.bfloat16),
)
_STATIC_ARGS = (8, 1e-6)


def key_gemma4_a4b_moe(*args) -> int:
    """This pretuned module intentionally supports one production shape."""
    tensor_count = len(_TENSOR_SIGNATURES)
    if (
        len(args) != tensor_count + len(_STATIC_ARGS)
        or tuple((tuple(arg.shape), arg.dtype) for arg in args[:tensor_count])
        != _TENSOR_SIGNATURES
        or args[tensor_count:] != _STATIC_ARGS
    ):
        raise ValueError("gemma4_a4b_moe is pretuned only for batch-one 26B-A4B")
    return 0


def autotune_gemma4_a4b_moe(*args) -> dict[str, object]:
    """Return the validated B200 static-pipeline configuration."""
    key_gemma4_a4b_moe(*args)
    return deepcopy(CONFIG)
