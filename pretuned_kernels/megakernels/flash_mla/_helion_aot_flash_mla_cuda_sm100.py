"""Checked-in B200 config for the runtime-topology FlashMLA megakernel."""

from __future__ import annotations

from copy import deepcopy

import torch


CONFIG = {
    "block_sizes": [],
    "loop_orders": [
        [0, 1],
        [1, 0, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ],
    "l2_groupings": [1, 8, 32],
    "range_unroll_factors": [0, 0, 0, 0, 0, 0, 0],
    "range_warp_specializes": [None, None, None, None, None, None, None],
    "range_num_stages": [0, 0, 0, 0, 0, 0, 0],
    "range_multi_buffers": [None, None, None, None, None, None, None],
    "range_flattens": [None, None, None, None, None, None, None],
    "static_ranges": [False, False],
    "load_eviction_policies": ["", "", "", "", "", "", "", "", "", "", "", ""],
    "num_warps": 4,
    "num_stages": 1,
    "indexing": ["pointer"] * 20,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
    "num_sm_multiplier": 1,
    "maxnreg": None,
    "atomic_indexing": [],
}


_TENSOR_SIGNATURES = (
    ((4, 4, 16, 576), torch.bfloat16),
    ((1111, 1, 64, 576), torch.bfloat16),
    ((4, 802), torch.int32),
    ((4,), torch.int32),
    ((608,), torch.int32),
    ((608,), torch.int32),
    ((38,), torch.int32),
    ((5,), torch.int32),
)
_STATIC_ARGS = (1.0 / 24.0, 128)


def key_flash_mla(*args) -> int:
    """Validate the fixed B4 envelope; topology and lengths remain runtime."""
    tensor_count = len(_TENSOR_SIGNATURES)
    if (
        len(args) != tensor_count + len(_STATIC_ARGS)
        or tuple((tuple(arg.shape), arg.dtype) for arg in args[:tensor_count])
        != _TENSOR_SIGNATURES
        or args[tensor_count:] != _STATIC_ARGS
    ):
        raise ValueError(
            "flash_mla is pretuned for the B4/Q4/H16 runtime-topology envelope"
        )
    return 0


def autotune_flash_mla(*args) -> dict[str, object]:
    """Return the validated B200 dynamic-pipeline configuration."""
    key_flash_mla(*args)
    return deepcopy(CONFIG)
