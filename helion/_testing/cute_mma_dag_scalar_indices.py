"""Shared fixtures for typed, per-iteration scalar table indices."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def wrap_scalar_index(value: int, bits: int) -> int:
    """Independent signed-width model for generated-address and native checks."""
    sign = 1 << (bits - 1)
    return (value + sign) % (2 * sign) - sign


def scalar_index_matmul(
    a: torch.Tensor,
    pages: torch.Tensor,
    order: torch.Tensor,
    bounds: torch.Tensor,
    offsets: torch.Tensor,
    variant: hl.constexpr,
    divisor: hl.constexpr,
    comparison: hl.constexpr,
    scalar_rhs: hl.constexpr,
    literal: hl.constexpr,
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(pages.size(2))
    hl.specialize(a.size(1))
    order_size = hl.specialize(order.size(0))
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows in hl.tile(m):
        begin, end = bounds[0], bounds[1]
        acc = hl.zeros((rows, n), dtype=torch.float32)
        for step in hl.grid(begin, end):
            if variant == "reverse":
                index = (begin + end) - 1 - step
            elif variant == "floor":
                index = (offsets[0] + step) // int(divisor) % order_size
            elif variant == "clamped_narrow":
                index = (offsets[0] + step).to(torch.int32).clamp(0, order_size - 1)
            elif variant in ("extrema", "extrema_narrow"):
                added = (offsets[0] + step).to(
                    torch.int32 if variant == "extrema_narrow" else offsets.dtype
                )
                index = (
                    torch.minimum(
                        torch.maximum(added, offsets[1]),
                        offsets[0] - step,
                    )
                    % order_size
                )
            elif variant == "float":
                index = (offsets[0].float() + step).to(torch.int32) % order_size
            elif variant == "power":
                index = offsets[0] ** 3 % order_size
            elif variant == "dynamic_div":
                index = offsets[0] // offsets[1] % order_size
            elif variant in (
                "compare",
                "compare_float_scalar",
                "bare_bool",
                "bare_bool_load",
            ):
                left = offsets[0] + step
                if scalar_rhs:
                    right = 1.5 if variant == "compare_float_scalar" else int(literal)
                else:
                    right = offsets[1]
                if comparison == "eq":
                    predicate = left == right
                elif comparison == "ne":
                    predicate = left != right
                elif comparison == "lt":
                    predicate = left < right
                elif comparison == "le":
                    predicate = left <= right
                elif comparison == "gt":
                    predicate = left > right
                elif comparison == "ge":
                    predicate = left >= right
                else:
                    raise ValueError(f"unknown comparison: {comparison}")
                index = (
                    predicate
                    if variant in ("bare_bool", "bare_bool_load")
                    # PyTorch wheel stubs omit this supported SymInt/Tensor sum.
                    # pyrefly: ignore[unsupported-operation]
                    else (step + predicate) % order_size
                )
            else:
                index = ((offsets[0] + step) * 3 - offsets[1]) % order_size
            if variant == "bare_bool_load":
                page = hl.load(order, [index])
            else:
                page = order[index]
            if variant == "nested":
                page = order[(page + 1) % order_size]
            acc = hl.dot(a[rows, :], pages[page, :, :], acc=acc * 0.5)
        out[rows, :] = acc
    return out


def scalar_index_args(
    dtype: torch.dtype,
    variant: str,
    device: torch.device | str = "cpu",
    divisor: int = 3,
    comparison: str = "gt",
    scalar_rhs: bool = False,
    literal: int = 1,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    str,
    int,
    str,
    bool,
    int,
]:
    return (
        torch.empty((64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((5, 64, 64), dtype=torch.bfloat16, device=device),
        torch.empty((5,), dtype=dtype, device=device),
        torch.empty((2,), dtype=dtype, device=device),
        torch.empty((2,), dtype=dtype, device=device),
        variant,
        divisor,
        comparison,
        scalar_rhs,
        literal,
    )


def scalar_index_config(mode: str, stages: int) -> helion.Config:
    return helion.Config(
        block_sizes=[64],
        cute_mma_dag=mode,
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=stages,
        cute_mma_dag_loads="tma",
    )
