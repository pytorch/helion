"""
FP8 Group Quantization
======================

This example implements per-group FP8 quantization in Helion.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


@helion.kernel(static_shapes=True)
def fp8_group_quant_kernel(
    grouped_x: torch.Tensor, grouped_q: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    nrows = grouped_x.size(0)
    group_size = hl.specialize(grouped_x.size(1))

    for row in hl.tile(nrows):
        x_row = grouped_x[row, :].to(torch.float32)
        amax = torch.amax(torch.abs(x_row), -1)
        amax = torch.clamp(amax, min=FP8_EPS)
        scale = amax / FP8_MAX
        q_row = torch.clamp(x_row / scale[:, None], min=FP8_MIN, max=FP8_MAX)
        grouped_q[row, :] = q_row.to(grouped_q.dtype)
        scales[row] = scale.to(scales.dtype)
    return grouped_q


def helion_fp8_group_quant(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hidden_dim = x.shape
    num_groups = hidden_dim // group_size
    grouped_rows = num_tokens * num_groups

    x_q = torch.empty_like(x)
    x_s = torch.empty(num_tokens, num_groups, dtype=torch.float32, device=x.device)
    fp8_group_quant_kernel(
        x.reshape(grouped_rows, group_size),
        x_q.reshape(grouped_rows, group_size),
        x_s.reshape(grouped_rows),
    )
    return x_q, x_s


def ref_fp8_group_quant(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hidden_dim = x.shape
    num_groups = hidden_dim // group_size
    grouped = x.float().reshape(num_tokens, num_groups, group_size)
    absmax = grouped.abs().amax(dim=-1).clamp(min=FP8_EPS)
    scales = absmax / FP8_MAX
    quantized = torch.clamp(grouped / scales.unsqueeze(-1), min=FP8_MIN, max=FP8_MAX)
    return quantized.reshape_as(x).to(x.dtype), scales


def test(num_tokens: int, hidden_dim: int, group_size: int) -> None:
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32, device=DEVICE)
    expected_q, expected_s = ref_fp8_group_quant(x, group_size)
    got_q, got_s = helion_fp8_group_quant(x, group_size)
    torch.testing.assert_close(got_q.to(torch.float32), expected_q.to(torch.float32))
    torch.testing.assert_close(got_s.to(torch.float32), expected_s.to(torch.float32))


def main() -> None:
    test(64, 1024, 128)
    test(128, 4096, 128)


if __name__ == "__main__":
    main()
