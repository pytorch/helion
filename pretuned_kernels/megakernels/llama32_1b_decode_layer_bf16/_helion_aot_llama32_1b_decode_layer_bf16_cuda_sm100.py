"""Checked-in B200 configs for the Llama-3.2-1B BF16 decode-layer megakernel.

One bucket per batch size (decode tokens); context lengths up to 8192 rows are
runtime metadata.
"""

from __future__ import annotations

from copy import deepcopy

import torch


CONFIGS = {
    1: {'block_sizes': [1, 8, 1024, 4, 32, 1, 8, 512, 1, 8, 512, 1, 8, 1024], 'loop_orders': [[0, 1], [0, 1], [0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1]], 'l2_groupings': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'range_unroll_factors': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_warp_specializes': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_num_stages': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_multi_buffers': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_flattens': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'cross_loop_pipeline': 'static', 'num_sm_multiplier': 8, 'host_tensor_descriptors': False, 'atomic_indexing': [], 'maxnreg': None},
    2: {'block_sizes': [2, 8, 1024, 4, 32, 2, 8, 512, 2, 8, 512, 2, 8, 1024], 'loop_orders': [[0, 1], [0, 1], [0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1]], 'l2_groupings': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'range_unroll_factors': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_warp_specializes': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_num_stages': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_multi_buffers': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_flattens': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'cross_loop_pipeline': 'static', 'num_sm_multiplier': 8, 'host_tensor_descriptors': False, 'atomic_indexing': [], 'maxnreg': None},
    4: {'block_sizes': [4, 8, 1024, 4, 32, 4, 8, 512, 4, 8, 512, 4, 8, 1024], 'loop_orders': [[0, 1], [0, 1], [0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1]], 'l2_groupings': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'range_unroll_factors': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_warp_specializes': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_num_stages': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'range_multi_buffers': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'range_flattens': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 'load_eviction_policies': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'pid_type': 'persistent_blocked', 'cross_loop_pipeline': 'static', 'num_sm_multiplier': 8, 'host_tensor_descriptors': False, 'atomic_indexing': [], 'maxnreg': None},
}

_BATCH_FREE_SIGNATURES = (
    ((2048,), torch.bfloat16),
    ((3072, 2048), torch.bfloat16),
    ((8192, 64), torch.bfloat16),
    ((2048, 2048), torch.bfloat16),
    ((2048,), torch.bfloat16),
    ((16384, 2048), torch.bfloat16),
    ((2048, 8192), torch.bfloat16),
)
_STATIC_ARGS = (2048, 8192, 32, 8, 64, 8192, 16, 128, 128, 1e-05)


def key_llama32_1b_decode_layer_bf16(*args) -> int:
    """The batch-size bucket; rejects anything but the Llama-3.2-1B geometry."""
    batch = args[0].size(0)
    weights = (args[2], args[4], args[5], args[10], args[11], args[13], args[14])
    if (
        len(args) != 16 + len(_STATIC_ARGS)
        or batch not in CONFIGS
        or tuple((tuple(arg.shape), arg.dtype) for arg in weights) != _BATCH_FREE_SIGNATURES
        or args[16:] != _STATIC_ARGS
    ):
        raise ValueError(
            "llama32_1b_decode_layer_bf16 is pretuned only for Llama-3.2-1B decode at "
            f"batch {sorted(CONFIGS)}"
        )
    return batch


def autotune_llama32_1b_decode_layer_bf16(*args) -> dict[str, object]:
    """Return the validated B200 static-pipeline configuration for this batch."""
    return deepcopy(CONFIGS[key_llama32_1b_decode_layer_bf16(*args)])
