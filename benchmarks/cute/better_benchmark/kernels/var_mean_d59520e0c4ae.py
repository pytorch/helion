"""better-benchmark repro var_mean_d59520e0c4ae (hf GoogleFnet).

An FNet layer's residual add feeding its output LayerNorm, all in fp32: the
``[batch * seq, hidden]`` projection output viewed as ``[batch, seq, hidden]``
plus the residual stream, normalized over the hidden dim (population
``var_mean``, eps 1e-12, affine), returned once in fp32 and once converted to
complex64 (the input of the next layer's Fourier transform). Inductor emits
one kernel and finishes with an ATen copy:

- ``triton_per_fused_add_mul_rsqrt_sub_var_mean_view_0``: a persistent
  reduction over ``hidden`` for every row. The sum is not stored; each row's
  statistics are computed in two passes (``mean = sum(x) / hidden``,
  ``var = sum((x - mean)**2) / hidden``) and
  ``(x - mean) * rsqrt(var + 1e-12) * weight + bias`` is the only store.
- Inductor's wrapper then allocates the complex64 output and fills it with an
  ATen ``copy_`` of the fp32 result (real part the value, imaginary part
  zero); the port does that conversion in a second kernel instead of on the
  host. The kernel reads the fp32 result as one flat dimension and writes the
  interleaved (real, imaginary) fp32 pairs of the complex64 storage, a
  ``[numel, 2]`` buffer that the host views as complex64.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def residual_add_layernorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    rows, hidden = branch.shape
    out = torch.empty([rows, hidden], dtype=branch.dtype, device=branch.device)
    for tile_r in hl.tile(rows):
        summed = branch[tile_r, :].to(torch.float32) + residual[tile_r, :].to(
            torch.float32
        )
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(summed, dim=-1, keepdim=True)
        centered = summed - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-12)
        y = y * weight[None, :].to(torch.float32) + bias[None, :].to(torch.float32)
        out[tile_r, :] = y.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def real_to_complex_pairs(x_flat: torch.Tensor) -> torch.Tensor:
    # out: the (real, imaginary) pairs of a complex64 tensor of x_flat's size.
    # A kernel cannot allocate or store complex64 (a real -> complex cast does
    # not lower on either backend), so it writes that storage as fp32 pairs,
    # the bytes the ATen copy writes, and forward() views them as complex64.
    n = x_flat.size(0)
    out = torch.empty([n, 2], dtype=x_flat.dtype, device=x_flat.device)
    for tile_n in hl.tile(n):
        real = x_flat[tile_n]
        out[tile_n, 0] = real
        out[tile_n, 1] = torch.zeros_like(real)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    branch, residual, weight, bias = inputs[:4]
    out_shape = inputs[4]  # _shape_param_0 = [batch, seq, hidden]
    batch, seq, hidden = residual.shape
    norm_out = residual_add_layernorm(
        branch, residual.view(batch * seq, hidden), weight, bias
    )
    pairs = real_to_complex_pairs(norm_out.view(-1))
    return (
        norm_out.view(out_shape),
        torch.view_as_complex(pairs).view(out_shape),
    )
