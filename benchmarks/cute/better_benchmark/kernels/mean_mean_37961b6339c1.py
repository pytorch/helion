"""better-benchmark repro mean_mean_37961b6339c1 (hf Qwen/Qwen3-0.6B inference).

The attention prologue of a Qwen3 layer: per-head RMSNorm of the q and k
projections (``q_norm`` / ``k_norm``: fp32 ``mean(x**2)`` over the head dim,
eps 1e-6, times a bf16 weight), the rotary embedding ``x * cos +
rotate_half(x) * sin`` with ``rotate_half(x) = cat([-x[..., D/2:],
x[..., :D/2]])``, and ``repeat_kv`` of the rotated k for grouped-query
attention. Inductor emits three kernels:

- ``triton_per_fused_add_cat_convert_element_type_mean_mul_neg_permute_pow_rsqrt_slice_unsqueeze_view_0``
  and ``..._1``: the same persistent reduction, once over the ``B*S*H`` q rows
  and once over the ``B*S*KVH`` k rows (head dim 128): the sum of squares of
  each row, ``rsqrt(sum / 128 + 1e-6)``, then the normalized row times the
  weight, rotated with the cos/sin row of its token, stored in bf16 in the
  ``[B, S, H, D]`` order of the projection (returned permuted to
  ``[B, H, S, D]``). Like Inductor (no ``emulate_precision_casts``), the
  graph's bf16 rounding of the normalized row and of the RoPE products stays
  in fp32; only the stores round. Each launch gets its own Helion kernel
  (same body) so each is tuned for its own shape.
- ``triton_poi_fused_clone_expand_unsqueeze_2``: the contiguous clone of the
  rotated k expanded to ``[B, KVH, n_rep, S, D]``, viewed as
  ``[B, KVH * n_rep, S, D]``. The host takes the ``unsqueeze``/``expand``
  view (stride 0 over ``n_rep``) and the kernel copies it.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def _rmsnorm_rope(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: [B*S, H, D] bf16, a view of the projection output; weight: [D];
    # cos, sin: [B*S, D], views of the [B, S, D] rotary tables.
    rows, heads, head_dim = x.shape
    half = head_dim // 2
    out = torch.empty([rows, heads, head_dim], dtype=x.dtype, device=x.device)
    for tile_r, tile_h in hl.tile([rows, heads]):
        xf = x[tile_r, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + 1e-06)
        # weight * (x * rstd) for each half of the head dim.
        lo = weight[:half].to(torch.float32) * (
            x[tile_r, tile_h, :half].to(torch.float32) * rstd
        )
        hi = weight[half:].to(torch.float32) * (
            x[tile_r, tile_h, half:].to(torch.float32) * rstd
        )
        cos_lo = cos[tile_r, :half].to(torch.float32).unsqueeze(1)
        cos_hi = cos[tile_r, half:].to(torch.float32).unsqueeze(1)
        sin_lo = sin[tile_r, :half].to(torch.float32).unsqueeze(1)
        sin_hi = sin[tile_r, half:].to(torch.float32).unsqueeze(1)
        # x * cos + cat([-hi, lo]) * sin
        out[tile_r, tile_h, :half] = (lo * cos_lo + (-hi) * sin_lo).to(out.dtype)
        out[tile_r, tile_h, half:] = (hi * cos_hi + lo * sin_hi).to(out.dtype)
    return out


q_rmsnorm_rope = helion.kernel(static_shapes=True)(_rmsnorm_rope)
k_rmsnorm_rope = helion.kernel(static_shapes=True)(_rmsnorm_rope)


@helion.kernel(static_shapes=True)
def repeat_kv_clone(expanded: torch.Tensor) -> torch.Tensor:
    # expanded: [B, KVH, n_rep, S, D], a stride-0 expand of the rotated k.
    out = torch.empty(expanded.shape, dtype=expanded.dtype, device=expanded.device)
    for tile in hl.tile(out.size()):
        out[tile] = expanded[tile]
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    q_proj, q_weight, cos, sin, k_proj, k_weight = inputs[:6]
    # [B, S, H*D], [B, S, -1, D], [B, S, KVH*D], [B, S, -1, D],
    # [B, KVH, n_rep, S, D], [B, H, S, D]
    q_shape0, q_shape1, k_shape0, k_shape1, expand_shape, repeat_shape = inputs[6:12]
    q = q_proj.view(q_shape0).view(q_shape1)
    k = k_proj.view(k_shape0).view(k_shape1)
    batch, seq, heads, head_dim = q.shape
    kv_heads = k.size(2)
    cos_rows = cos.flatten(0, 1)
    sin_rows = sin.flatten(0, 1)
    q_out = q_rmsnorm_rope(q.flatten(0, 1), q_weight, cos_rows, sin_rows)
    k_out = k_rmsnorm_rope(k.flatten(0, 1), k_weight, cos_rows, sin_rows)
    q_out = q_out.view(batch, seq, heads, head_dim).permute(0, 2, 1, 3)
    k_out = k_out.view(batch, seq, kv_heads, head_dim).permute(0, 2, 1, 3)
    repeated = repeat_kv_clone(k_out.unsqueeze(2).expand(expand_shape))
    return q_out, k_out, repeated.view(repeat_shape)
