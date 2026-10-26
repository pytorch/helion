"""Checked-in GB200 configs for Kimi-K3 KDA decode."""

from __future__ import annotations

from copy import deepcopy

import torch


_BASE_CONFIG: dict[str, object] = {
    "num_warps": 1,
    "num_stages": 4,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
    "num_sm_multiplier": 1,
    "maxnreg": 240,
    "indexing": "block_ptr",
    "load_eviction_policies": "",
}


def _config(
    block_sizes: list[int], *, num_warps: int = 1, num_stages: int = 4
) -> dict[str, object]:
    return {
        **_BASE_CONFIG,
        "block_sizes": block_sizes,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


# H12 is Kimi-K3 TP8. H6 is the TP16 envelope and has a distinct
# recurrence/output balance because it exposes half as many local heads.
CONFIGS: dict[tuple[int, int], dict[str, object]] = {
    (1, 12): _config(
        [1, 16, 1, 16, 1, 32, 128, 8, 1, 1, 16, 512, 256, 128, 128],
        num_stages=5,
    ),
    (2, 12): _config(
        [2, 16, 2, 16, 1, 32, 128, 8, 1, 2, 16, 512, 256, 128, 128],
        num_stages=5,
    ),
    (4, 12): _config(
        [4, 16, 4, 16, 1, 32, 128, 16, 1, 4, 32, 512, 256, 128, 128],
        num_warps=2,
    ),
    (8, 12): _config(
        [8, 16, 8, 16, 1, 32, 128, 32, 1, 8, 32, 512, 256, 128, 128],
        num_warps=2,
    ),
    (16, 12): _config(
        [2, 8, 16, 16, 1, 64, 128, 32, 4, 16, 16, 256, 256, 128, 256],
        num_warps=2,
    ),
    (1, 6): _config(
        [1, 16, 1, 16, 1, 32, 128, 8, 1, 1, 16, 512, 256, 128, 128]
    ),
    (2, 6): _config(
        [2, 16, 2, 16, 1, 32, 128, 8, 1, 2, 16, 512, 256, 128, 128]
    ),
    (4, 6): _config(
        [4, 16, 4, 16, 1, 32, 128, 8, 1, 4, 32, 512, 256, 128, 128]
    ),
    (8, 6): _config(
        [8, 16, 8, 16, 1, 32, 128, 16, 1, 8, 32, 512, 256, 128, 128]
    ),
    (16, 6): _config(
        [4, 16, 16, 16, 1, 32, 128, 16, 1, 8, 32, 512, 256, 128, 128]
    ),
}

_SCALE = 128**-0.5
_EPS = 1e-5
_LOWER_BOUND = -5.0
_HIDDEN = 7168
_HEAD_DIM = 128
_POOL_SIZE = 32


def _projection_width(heads: int) -> int:
    logical_width = 4 * heads * _HEAD_DIM + _HEAD_DIM + heads
    return (logical_width + 15) // 16 * 16


def _signature(
    batch: int, heads: int
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    segment = heads * _HEAD_DIM
    return (
        ((batch, _HIDDEN), torch.bfloat16),
        ((_projection_width(heads), _HIDDEN), torch.bfloat16),
        ((segment, _HEAD_DIM), torch.bfloat16),
        ((3, 4, segment), torch.float32),
        ((heads,), torch.float32),
        ((segment,), torch.float32),
        ((_POOL_SIZE, 3, 3 * segment), torch.bfloat16),
        ((_POOL_SIZE, heads, _HEAD_DIM, _HEAD_DIM), torch.float32),
        ((batch,), torch.int32),
        ((_HEAD_DIM,), torch.float32),
        ((_HIDDEN, segment), torch.bfloat16),
    )


def _static_args(heads: int) -> tuple[object, ...]:
    return (
        _SCALE,
        _EPS,
        _LOWER_BOUND,
        _projection_width(heads),
        _HEAD_DIM + heads,
        heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )


_SUPPORTED = tuple(
    (_signature(batch, heads), _static_args(heads), batch, heads)
    for heads in (12, 6)
    for batch in (1, 2, 4, 8, 16)
)

_TENSOR_SIGNATURES = _SUPPORTED[0][0]
_STATIC_ARGS = _SUPPORTED[0][1]


def key_kda_decode(*args) -> int:
    """Validate one of the ten Kimi-K3 TP8/TP16 physical envelopes."""
    tensor_count = len(_TENSOR_SIGNATURES)
    if len(args) != tensor_count + len(_STATIC_ARGS):
        raise ValueError("kda_decode expects eleven tensors and eight static args")
    tensor_signature = tuple(
        (tuple(arg.shape), arg.dtype) for arg in args[:tensor_count]
    )
    static_args = args[tensor_count:]
    for index, (supported_tensors, supported_static, _batch, _heads) in enumerate(
        _SUPPORTED
    ):
        if tensor_signature == supported_tensors and static_args == supported_static:
            return index
    raise ValueError(
        "kda_decode is pretuned only for Kimi-K3 "
        "B1/B2/B4/B8/B16 x TP8/H12 or TP16/H6"
    )


def autotune_kda_decode(*args) -> dict[str, object]:
    """Return the checked-in Kimi-K3 configuration."""
    index = key_kda_decode(*args)
    _signature_value, _static_value, batch, heads = _SUPPORTED[index]
    return deepcopy(CONFIGS[batch, heads])
