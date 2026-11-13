"""Bound-layout fixtures for the explicit paged-attention schedule."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def paged_attention_args(
    *, rows: int = 64, width: int = 128, device: torch.device | str = "cpu"
) -> tuple[object, ...]:
    pages, heads, dim = 13, 2, 128
    raw_bytes = width * heads * dim
    scale_bytes = width * heads * 4
    page_stride = raw_bytes + scale_bytes
    storage_k = torch.empty(pages * page_stride, dtype=torch.int8, device=device)
    storage_v = torch.empty_like(storage_k)
    shape = (pages, 1, width, heads, dim)
    strides = (page_stride, raw_bytes, heads * dim, dim, 1)
    scale_shape = (pages, 1, width, heads, 4)
    scale_strides = (page_stride, scale_bytes, 4, width * 4, 1)
    return (
        torch.empty(4, rows, dim, dtype=torch.bfloat16, device=device),
        storage_k.as_strided(shape, strides),
        storage_v.as_strided(shape, strides),
        storage_k.view(torch.uint8).as_strided(scale_shape, scale_strides, raw_bytes),
        storage_v.view(torch.uint8).as_strided(scale_shape, scale_strides, raw_bytes),
        torch.empty(2, 17, dtype=torch.int32, device=device),
        torch.empty(2, dtype=torch.int32, device=device),
        5,
        8,
        31,
    )


def paged_attention_reference(
    args: Sequence[object], *, denominator_per_key: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve reduced or per-key denominator accumulation before final division."""
    q, k, v, sk, sv, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    query_len, group, window = (cast("int", arg) for arg in args[7:10])
    output = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], -float("inf"), device=q.device)
    width, heads = k.size(2), k.size(3)
    for item in range(q.size(0)):
        request, head = divmod(item, heads)
        length = cast("int", lengths[request].item())
        begin = max(length - query_len - window, 0) // width if window >= 0 else 0
        end = (length + width - 1) // width
        maximum = torch.full((q.size(1),), -float("inf"), device=q.device)
        if denominator_per_key:
            denominator = torch.zeros((q.size(1), width), device=q.device)
        else:
            denominator = torch.zeros_like(maximum)
        accumulator = torch.zeros(q.shape[1:], device=q.device)
        positions = (
            length - query_len + torch.arange(q.size(1), device=q.device) // group
        )
        for tile in range(begin, end):
            page = cast("int", pages[request, tile].item())
            key = (
                k[page, 0, :, head].float()
                * torch.exp2(sk[page, 0, :, head].float() - 127).repeat_interleave(
                    32, -1
                )
            ).to(q.dtype)
            value = (
                v[page, 0, :, head].float()
                * torch.exp2(sv[page, 0, :, head].float() - 127).repeat_interleave(
                    32, -1
                )
            ).to(q.dtype)
            score = q[item].float() @ key.float().T * q.size(2) ** -0.5
            keys = torch.arange(width, device=q.device) + tile * width
            visible = keys[None, :] <= positions[:, None]
            if window >= 0:
                visible &= keys[None, :] >= positions[:, None] - window
            score = torch.where(visible, score, -float("inf"))
            next_maximum = torch.maximum(maximum, score.amax(1))
            safe_maximum = torch.where(next_maximum == -float("inf"), 0, next_maximum)
            alpha = torch.exp(maximum - safe_maximum)
            probability = torch.exp(score - safe_maximum[:, None])
            if denominator_per_key:
                denominator = denominator * alpha[:, None] + probability
            else:
                denominator = denominator * alpha + probability.sum(1)
            accumulator = (
                accumulator * alpha[:, None]
                + probability.to(q.dtype).float() @ value.float()
            )
            maximum = next_maximum
        if denominator_per_key:
            denominator = denominator.sum(1)
        output[item] = (
            accumulator / torch.where(denominator == 0, 1, denominator)[:, None]
        )
        lse[item] = maximum + torch.log(denominator)
    return output, lse
