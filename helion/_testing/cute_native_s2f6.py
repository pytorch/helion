"""Portable integer decoder used to test native S2F6 recognition and lowering."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def integer_s2f6_decode(raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    signed = raw.to(torch.int32)
    exponent = scale.to(torch.int32)
    magnitude = torch.abs(signed)
    top = torch.zeros_like(magnitude)
    for bit in range(1, 8):
        top = torch.where(magnitude >= 2**bit, bit, top)
    biased = top + exponent - 6
    mantissa = (magnitude << (7 - top)) - 128
    normal = (biased << 7) + mantissa
    subnormal = magnitude << torch.clamp(exponent, max=6)
    magnitude_bits = torch.where(biased <= 0, subnormal, normal)
    magnitude_bits = torch.clamp(magnitude_bits, min=0, max=0x7F7F)
    magnitude_bits = torch.where(magnitude == 0, 0, magnitude_bits)
    bits = magnitude_bits | torch.where(signed < 0, 0x8000, 0)
    bits = torch.where(exponent == 255, 0x7FC0, bits)
    return bits.to(torch.int16).view(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True)
def decode_kernel(raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty(raw.shape, device=raw.device, dtype=torch.bfloat16)
    for tile in hl.tile(raw.numel()):
        out[tile] = integer_s2f6_decode(raw[tile], scale[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True)
def grouped_decode_kernel(raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty(raw.shape, device=raw.device, dtype=torch.bfloat16)
    for row, column in hl.tile(raw.size()):
        out[row, column] = integer_s2f6_decode(raw[row, column], scale[row, :])
    return out


@helion.kernel(backend="cute", static_shapes=True)
def grouped_decode_sum(raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty((raw.size(0),), device=raw.device, dtype=torch.float32)
    for row in hl.tile(raw.size(0)):
        decoded = integer_s2f6_decode(raw[row, :], scale[row, :])
        out[row] = decoded.float().sum(-1)
    return out


def decode_domain() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All byte/scale combinations and an independent exact BF16 reference."""
    codes = torch.arange(65536, dtype=torch.int32)
    raw = (codes % 256).to(torch.uint8).view(torch.int8)
    scale = (codes // 256).to(torch.uint8)
    limit = float(torch.finfo(torch.bfloat16).max)
    expected = (
        torch.ldexp(raw.double(), scale.int() - 133).clamp(-limit, limit).bfloat16()
    )
    expected.view(torch.int16)[scale == 255] = 0x7FC0
    return raw, scale, expected
