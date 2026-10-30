"""Checked-in B200 config for the Qwen3-30B-A3B MoE block megakernel (batch 1)."""

from __future__ import annotations

from copy import deepcopy

import torch


CONFIG = {'block_sizes': [256, 256, 8, 16, 32, 256], 'loop_orders': [[0, 1], [0, 1, 2], [0, 1, 2], [0, 1]], 'l2_groupings': [1, 1, 1, 1], 'reduction_loops': [None], 'range_unroll_factors': [0, 0, 0, 0, 0, 0], 'range_warp_specializes': [None, None, None, None, None, None], 'range_num_stages': [0, 0, 3, 0, 3, 0], 'range_multi_buffers': [None, None, None, None, None, None], 'range_flattens': [None, None, None, None, None, None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'cross_loop_pipeline': 'static', 'num_sm_multiplier': 4, 'host_tensor_descriptors': False, 'reduction_fusion': False, 'atomic_indexing': [], 'maxnreg': 128}


_TENSOR_SIGNATURES = (
    ((1, 2048), torch.bfloat16),
    ((2048,), torch.bfloat16),
    ((128, 2048), torch.bfloat16),
    ((128, 1536, 2048), torch.bfloat16),
    ((128, 2048, 768), torch.bfloat16),
)
_STATIC_ARGS = (8, 1e-6)


def key_qwen3_moe(*args) -> int:
    """This pretuned module intentionally supports one production shape."""
    tensor_count = len(_TENSOR_SIGNATURES)
    if (
        len(args) != tensor_count + len(_STATIC_ARGS)
        or tuple((tuple(arg.shape), arg.dtype) for arg in args[:tensor_count])
        != _TENSOR_SIGNATURES
        or args[tensor_count:] != _STATIC_ARGS
    ):
        raise ValueError("qwen3_moe is pretuned only for batch-one Qwen3-30B-A3B")
    return 0


def autotune_qwen3_moe(*args) -> dict[str, object]:
    """Return the validated B200 static-pipeline configuration."""
    key_qwen3_moe(*args)
    return deepcopy(CONFIG)
