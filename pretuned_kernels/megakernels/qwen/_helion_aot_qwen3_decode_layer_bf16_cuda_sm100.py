"""Checked-in B200 configs for the Qwen3-8B BF16 decode-layer megakernel.

One bucket per batch size (decode tokens); context lengths up to 8192 rows are
runtime metadata.
"""

from __future__ import annotations

from copy import deepcopy

import torch


CONFIGS = {
    1: {'block_sizes': [1, 8, 512, 2, 1, 8, 1024, 1, 8, 1024, 1, 8, 512], 'loop_orders': [[0, 1], [0, 1], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1]], 'l2_groupings': [1, 1, 1, 1, 1, 1, 1, 1, 1], 'range_unroll_factors': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_warp_specializes': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_num_stages': [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0], 'range_multi_buffers': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_flattens': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'cross_loop_pipeline': 'static', 'num_sm_multiplier': 7, 'host_tensor_descriptors': False, 'atomic_indexing': [], 'maxnreg': None},
    2: {'block_sizes': [2, 8, 1024, 4, 2, 8, 1024, 2, 8, 1024, 2, 8, 1024], 'loop_orders': [[0, 1], [0, 1], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1]], 'l2_groupings': [1, 1, 1, 1, 1, 1, 1, 1, 1], 'range_unroll_factors': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_warp_specializes': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_num_stages': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_multi_buffers': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_flattens': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'cross_loop_pipeline': 'static', 'num_sm_multiplier': 8, 'host_tensor_descriptors': False, 'atomic_indexing': [], 'maxnreg': None},
    4: {'block_sizes': [4, 8, 1024, 4, 4, 8, 1024, 4, 8, 1024, 4, 8, 1024], 'loop_orders': [[0, 1], [0, 1], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1]], 'l2_groupings': [1, 1, 1, 1, 1, 1, 1, 1, 1], 'range_unroll_factors': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_warp_specializes': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_num_stages': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_multi_buffers': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_flattens': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'cross_loop_pipeline': 'static', 'num_sm_multiplier': 8, 'host_tensor_descriptors': False, 'atomic_indexing': [], 'maxnreg': None},
}

_BATCH_FREE_SIGNATURES = (
    ((4096,), torch.bfloat16),
    ((6144, 4096), torch.bfloat16),
    ((128,), torch.bfloat16),
    ((128,), torch.bfloat16),
    ((8192, 128), torch.bfloat16),
    ((4096, 4096), torch.bfloat16),
    ((4096,), torch.bfloat16),
    ((24576, 4096), torch.bfloat16),
    ((4096, 12288), torch.bfloat16),
)
_STATIC_ARGS = (4096, 12288, 32, 8, 128, 8192, 16, 256, 128, 1e-06)


def key_qwen3_decode_layer_bf16(*args) -> int:
    """The batch-size bucket; rejects anything but the Qwen3-8B geometry."""
    batch = args[0].size(0)
    weights = (args[2], args[4], args[5], args[6], args[7], args[12], args[13], args[15], args[16])
    if (
        len(args) != 18 + len(_STATIC_ARGS)
        or batch not in CONFIGS
        or tuple((tuple(arg.shape), arg.dtype) for arg in weights) != _BATCH_FREE_SIGNATURES
        or args[18:] != _STATIC_ARGS
    ):
        raise ValueError(
            "qwen3_decode_layer_bf16 is pretuned only for Qwen3-8B decode at "
            f"batch {sorted(CONFIGS)}"
        )
    return batch


def autotune_qwen3_decode_layer_bf16(*args) -> dict[str, object]:
    """Return the validated B200 static-pipeline configuration for this batch."""
    return deepcopy(CONFIGS[key_qwen3_decode_layer_bf16(*args)])
