"""Canonical K10 input generator and storage containers, with local shape metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    dtype: str = "bfloat16"
    row_padding: int = 0
    distribution: str = "normal"
    normalize_rounded_residual: bool = False


@dataclass
class Case:
    x: torch.Tensor
    residual: torch.Tensor
    weight: torch.Tensor
    x_storage: torch.Tensor
    residual_storage: torch.Tensor
    normalize_rounded_residual: bool = False
    eps: float = 1e-6

    def tensor_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "x_storage": self.x_storage,
            "residual_storage": self.residual_storage,
            "weight": self.weight,
        }

    def kernel_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x, self.residual, self.weight


def make_case(shape: Shape, *, seed: int = 0, device: str = "cuda") -> Case:
    import torch

    if min(shape.m, shape.n) <= 0 or shape.row_padding < 0:
        raise ValueError("Use positive dimensions and nonnegative row padding")
    if shape.dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("Expected BF16, FP16 or FP32 inputs")
    if type(shape.normalize_rounded_residual) is not bool:
        raise ValueError("Residual normalization mode must be a bool")
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = getattr(torch, shape.dtype)

    def normal(dims: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            dims, device=device, generator=generator, dtype=torch.float32
        ).to(dtype)

    storage = normal((shape.m, shape.n + shape.row_padding))
    residual_storage = normal((shape.m, shape.n + shape.row_padding))
    x, residual = storage[:, : shape.n], residual_storage[:, : shape.n]
    weight = normal((shape.n,))
    if shape.distribution == "zero":
        x.zero_()
        residual.zero_()
    elif shape.distribution == "tiny":
        x.mul_(1e-4)
        residual.mul_(1e-4)
    elif shape.distribution == "impulse_tail":
        x.zero_()
        residual.zero_()
        x[:, -1] = 64
        residual[:, -1] = -32
    elif shape.distribution == "cancellation":
        x.mul_(32)
        residual.copy_(-x.float() + residual.float() * 0.01)
    elif shape.distribution == "rounding_sensitive":
        x.fill_(1)
        residual.fill_(torch.finfo(dtype).eps * 0.49)
        residual[:, ::2].neg_()
    elif shape.distribution != "normal":
        raise ValueError("Unknown residual RMSNorm distribution")
    return Case(
        x,
        residual,
        weight,
        storage,
        residual_storage,
        normalize_rounded_residual=shape.normalize_rounded_residual,
    )
