"""Checked-in B200 config for the Qwen3-8B decode-layer megakernel."""

from __future__ import annotations

from copy import deepcopy

import torch


CONFIG = {'block_sizes': [8, 4, 4, 32, 1, 8, 16, 8], 'loop_orders': [[0, 1], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1, 2], [2, 1, 0], [0, 1], [0, 1, 2], [1, 0], [0, 1], [0, 1, 2], [0, 1], [0, 1], [0, 1]], 'l2_groupings': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'range_unroll_factors': [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 4], 'range_warp_specializes': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_num_stages': [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 4], 'range_multi_buffers': [None, None, None, None, True, None, None, None, True, None, None, None, None, False, None, None, None, None, True, None, None, False], 'range_flattens': [None, None, None, None, False, None, None, None, True, None, None, None, None, False, None, None, None, None, False, None, None, True], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'cross_loop_schedule': 'static_pipeline', 'num_sm_multiplier': 8, 'atomic_indexing': [], 'maxnreg': None}


_TENSOR_SIGNATURES = (
    ((1, 4096), torch.bfloat16),
    ((1, 4096), torch.bfloat16),
    ((4096,), torch.bfloat16),
    ((1, 4096), torch.float8_e4m3fn),
    ((1, 32), torch.float32),
    ((6144, 4096), torch.float8_e4m3fn),
    ((48, 32), torch.float32),
    ((128,), torch.bfloat16),
    ((128,), torch.bfloat16),
    ((8192, 128), torch.bfloat16),
    ((1,), torch.int64),
    ((640, 16, 8, 256), torch.bfloat16),
    ((1, 512), torch.int32),
    ((1,), torch.int64),
    ((4096, 4096), torch.float8_e4m3fn),
    ((32, 32), torch.float32),
    ((1, 4096), torch.float8_e4m3fn),
    ((1, 32), torch.float32),
    ((4096,), torch.bfloat16),
    ((1, 4096), torch.float8_e4m3fn),
    ((1, 32), torch.float32),
    ((24576, 4096), torch.float8_e4m3fn),
    ((192, 32), torch.float32),
    ((4096, 12288), torch.float8_e4m3fn),
    ((32, 96), torch.float32),
)
_STATIC_ARGS = (4096, 12288, 32, 8, 128, 8192, 16, 128, 128, 1e-6)


def key_qwen3_decode_layer(*args) -> int:
    """This pretuned module intentionally supports one production shape."""
    tensor_count = len(_TENSOR_SIGNATURES)
    if (
        len(args) != tensor_count + len(_STATIC_ARGS)
        or tuple((tuple(arg.shape), arg.dtype) for arg in args[:tensor_count])
        != _TENSOR_SIGNATURES
        or args[tensor_count:] != _STATIC_ARGS
    ):
        raise ValueError("qwen3_decode_layer is pretuned only for Qwen3-8B decode")
    return 0


def autotune_qwen3_decode_layer(*args) -> dict[str, object]:
    """Return the validated B200 static-pipeline configuration."""
    key_qwen3_decode_layer(*args)
    return deepcopy(CONFIG)
