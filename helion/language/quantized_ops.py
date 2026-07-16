from __future__ import annotations

import torch

from .. import exc
from . import _decorators

__all__ = [
    "float4_e2m1fn_x2_to_float32",
    "load_float4_e2m1fn_x16_to_float16",
]


@_decorators.api(is_device_only=True)
def float4_e2m1fn_x2_to_float32(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a ``torch.float4_e2m1fn_x2`` scalar tensor into FP32 lanes."""
    raise exc.NotInsideKernel


@_decorators.api(is_device_only=True, allow_host_tensor=True)
def load_float4_e2m1fn_x16_to_float16(
    storage: torch.Tensor,
    group_offsets: torch.Tensor,
    extra_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Load 8 packed FP4 bytes and unpack them into 16 FP16 lanes.

    ``group_offsets`` indexes 8-byte groups relative to the start of contiguous
    ``torch.uint8`` or ``torch.float4_e2m1fn_x2`` storage.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(float4_e2m1fn_x2_to_float32)
def _(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value.dtype is not torch.float4_e2m1fn_x2:
        raise exc.InvalidAPIUsage(
            "hl.float4_e2m1fn_x2_to_float32 expects a "
            f"torch.float4_e2m1fn_x2 tensor, got {value.dtype}"
        )
    return (
        torch.empty(value.shape, dtype=torch.float32, device=value.device),
        torch.empty(value.shape, dtype=torch.float32, device=value.device),
    )


@_decorators.register_fake(load_float4_e2m1fn_x16_to_float16)
def _(
    storage: torch.Tensor,
    group_offsets: torch.Tensor,
    extra_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if storage.dtype not in (torch.uint8, torch.float4_e2m1fn_x2):
        raise exc.InvalidAPIUsage(
            "hl.load_float4_e2m1fn_x16_to_float16 expects a "
            f"torch.uint8 or torch.float4_e2m1fn_x2 tensor, got {storage.dtype}"
        )
    return tuple(
        torch.empty(
            group_offsets.shape,
            dtype=torch.float16,
            device=group_offsets.device,
        )
        for _ in range(16)
    )
