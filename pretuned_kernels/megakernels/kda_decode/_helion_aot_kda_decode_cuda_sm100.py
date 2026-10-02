"""Checked-in B200 configs for the Kimi-Linear KDA decode megakernel."""

from __future__ import annotations

from copy import deepcopy

import torch


# Tuned on GB200 with cold L2 against production vLLM and the matched
# six-launch PDL pipeline.  One configuration covers every supported B/H
# envelope; the source contains no batch- or head-dependent algorithm branch.
_SHARED_CONFIG: dict[str, object] = {
    "block_sizes": [2, 8, 2, 32, 32, 128, 32, 8, 2, 8, 512, 128, 128, 128],
    "num_warps": 1,
    "num_stages": 6,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
    "num_sm_multiplier": 1,
    "maxnreg": 240,
    "indexing": "block_ptr",
    "load_eviction_policies": "",
}
CONFIGS: dict[tuple[int, int], dict[str, object]] = {
    (batch, heads): _SHARED_CONFIG
    for batch in (1, 2, 4)
    for heads in (12, 16)
}

_SCALE = 128**-0.5
_EPS = 1e-5
_HIDDEN = 2304
_HEAD_DIM = 128
_POOL_SIZE = 32


def _signature(batch: int, heads: int) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    qkv_width = 3 * heads * _HEAD_DIM
    projection_width = qkv_width + heads + 2 * _HEAD_DIM
    return (
        ((batch, _HIDDEN), torch.bfloat16),
        ((projection_width, _HIDDEN), torch.bfloat16),
        ((2, heads * _HEAD_DIM, _HEAD_DIM), torch.bfloat16),
        ((qkv_width, 4), torch.float32),
        ((heads,), torch.float32),
        ((heads * _HEAD_DIM,), torch.float32),
        ((_POOL_SIZE, 3, qkv_width), torch.bfloat16),
        ((_POOL_SIZE, heads, _HEAD_DIM, _HEAD_DIM), torch.float32),
        ((batch,), torch.int32),
        ((_HEAD_DIM,), torch.bfloat16),
        ((_HIDDEN, heads * _HEAD_DIM), torch.bfloat16),
    )


def _static_args(heads: int) -> tuple[object, ...]:
    qkv_width = 3 * heads * _HEAD_DIM
    projection_width = qkv_width + heads + 2 * _HEAD_DIM
    return (
        _SCALE,
        _EPS,
        1,
        projection_width,
        qkv_width,
        heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )


_SUPPORTED = tuple(
    (_signature(batch, heads), _static_args(heads), batch, heads)
    for batch in (1, 2, 4)
    for heads in (12, 16)
)

# The canonical signature is exposed for the common fixed-shape heuristic test.
_TENSOR_SIGNATURES = _SUPPORTED[0][0]
_STATIC_ARGS = _SUPPORTED[0][1]


def key_kda_decode(*args) -> int:
    """Validate one of the six pretuned B/H physical envelopes."""
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
    raise ValueError("kda_decode is pretuned only for B1/B2/B4 x H12/H16 KDA decode")


def autotune_kda_decode(*args) -> dict[str, object]:
    """Return the validated B200 dynamic-pipeline configuration."""
    index = key_kda_decode(*args)
    _signature_value, _static_value, batch, heads = _SUPPORTED[index]
    return deepcopy(CONFIGS[batch, heads])
