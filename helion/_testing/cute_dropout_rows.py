"""A dropout-residual-LayerNorm row: a masked persistent reduction row.

``dropout_residual_norm`` is the kernel of the CuTe persistent-row tests
(``test/test_cute_persistent_lane_unroll.py``,
``test/test_cute_unpadded_rows.py``), after better-benchmark's
dual_dropout_residual_layernorm: a Philox mask, ``residual + keep * x * c``
(a multiply-add LLVM contracts once the masked loads leave the arithmetic)
and a LayerNorm over the row.  A row whose width is not a power of two is
masked.  ``dropout_residual_norm_args`` builds its inputs.
"""

from __future__ import annotations

import torch

import helion.language as hl


def dropout_residual_norm(
    x: torch.Tensor, residual: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, n = x.shape
    keep_out = torch.empty([rows, n], dtype=torch.bool, device=x.device)
    centered_out = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    norm_out = torch.empty([rows, n], dtype=torch.bfloat16, device=x.device)
    for row in hl.tile(rows):
        keep = hl.rand([row, n], seed=seed) > 0.1
        added = (
            residual[row, :]
            + keep.to(torch.float32) * x[row, :].to(torch.float32) * 1.1111111111111112
        )
        centered = added - torch.mean(added, dim=-1, keepdim=True)
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        keep_out[row, :] = keep
        centered_out[row, :] = centered
        norm_out[row, :] = (centered * torch.rsqrt(var + 1e-5)).to(torch.bfloat16)
    return keep_out, centered_out, norm_out


def dropout_residual_norm_args(
    width: int, device: str | torch.device = "cpu"
) -> tuple[object, ...]:
    """19 rows of ``width`` columns: bf16 ``x``, fp32 ``residual`` and the
    Philox seed."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((19, width), generator=generator).to(device, torch.bfloat16)
    residual = torch.randn((19, width), generator=generator).to(device)
    return x, residual, 123
