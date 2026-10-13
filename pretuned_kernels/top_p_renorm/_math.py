"""AIR key encoding and explicit FP32 rounding/FTZ operations."""

from __future__ import annotations

import torch

import helion.language as hl


def descending_key(value: torch.Tensor) -> torch.Tensor:
    bits = value.view(torch.int32)
    return torch.where(bits < 0, bits, bits ^ 2147483647)


def from_descending_key(key: torch.Tensor) -> torch.Tensor:
    return torch.where(key < 0, key, key ^ 2147483647).view(torch.float32)


def ftz(value: torch.Tensor) -> torch.Tensor:
    bits = value.view(torch.int32)
    return torch.where((bits & 2139095040) == 0, bits & -2147483648, bits).view(
        torch.float32
    )


def add_ftz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "add.rn.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def sub_ftz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "sub.rn.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def mul_ftz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "mul.rn.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def add_rn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )
