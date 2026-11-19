"""Helpers of the Qwen3.5 TP4 MoE megakernels (the MoE and the merged MoE + GDN
layer): NVFP4 rounding, the swizzled scale tiles and the BF16 routing keys."""

from __future__ import annotations

import torch

FP4_MAX = 6.0
FP8_MAX = 448.0
NORM_EPS = 1e-6
# A routing key no logit has: below every bf16_key (a picked or padding lane).
PICKED32 = -2147483647


def fp4_nibble(value: torch.Tensor) -> torch.Tensor:
    """Round scaled values to E2M1 with FlashInfer's tie convention."""
    magnitude = torch.abs(value)
    code = (magnitude > 0.25).to(torch.int32)
    code = code + (magnitude >= 0.75).to(torch.int32)
    code = code + (magnitude > 1.25).to(torch.int32)
    code = code + (magnitude >= 1.75).to(torch.int32)
    code = code + (magnitude > 2.5).to(torch.int32)
    code = code + (magnitude >= 3.5).to(torch.int32)
    code = code + (magnitude > 5.0).to(torch.int32)
    return code | ((value < 0).to(torch.int32) << 3)


def scale_tile(chunk: torch.Tensor) -> torch.Tensor:
    """[16, 256] swizzled scale bytes of a 128-row tile and 32 groups -> [128, 32]."""
    return chunk.reshape(8, 32, 4, 4).permute(2, 1, 0, 3).reshape(128, 32)


def scale_tiles_16(chunk: torch.Tensor) -> torch.Tensor:
    """[16, 256] swizzled scale bytes of two 128-row tiles and 16 groups -> [256, 16]."""
    return chunk.reshape(2, 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(256, 16)


def bf16_key(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """int32 keys ordered by the BF16 value, then by the lower index (< 65536
    indices): the value's order-preserving 16 bits above the index's."""
    bits = logits.view(torch.int16).to(torch.int32)
    ordered = torch.where(bits < 0, -32768 - bits, bits)
    return ordered * 65536 + (65535 - index)


def key_value(key: torch.Tensor) -> torch.Tensor:
    """The float32 value of a bf16_key's BF16 logit."""
    ordered = key >> 16
    bits = torch.where(ordered < 0, -32768 - ordered, ordered)
    return bits.to(torch.int16).view(torch.bfloat16).to(torch.float32)
