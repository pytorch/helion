"""better-benchmark repro mean_mean_7291b09200a4 (hf/infer/google/gemma-3-4b-it).

The Gemma 3 attention prologue: per-head RMSNorm of the query and key
projections (fp32 mean of squares over ``head_dim``, eps 1e-6, Gemma's
``(1 + weight)`` scale), rotary embeddings ``y * cos + rotate_half(y) * sin``
with ``rotate_half(y) = cat(-y[D/2:], y[:D/2])``, and the grouped-query
``repeat_kv`` of the keys. Inductor emits three kernels, and the port one
Helion kernel per launch:

- ``triton_per_fused_add_cat_convert_element_type_mean_mul_neg_permute_pow_rsqrt_slice_unsqueeze_view_0``
  (``q_rmsnorm_rope``): a persistent reduction per ``(token, query head)`` row
  of the ``[T, Hq * D]`` projection. It reads the row, the norm weight and the
  token's cos/sin rows, and stores the bf16 result in the projection's
  ``[T, Hq, D]`` order, returned as the permuted ``[1, Hq, S, D]`` view.
- ``..._1`` (``k_rmsnorm_rope``): the same for the key heads, storing the
  result twice: heads-major ``[1, Hk, S, D]`` (the input of ``repeat_kv``)
  and token-major ``[T, Hk, D]`` (returned through the permuted view and the
  graph's ``[-1023:]`` sliding-window slice, a view).
- ``triton_poi_fused_clone_expand_unsqueeze_2`` (``repeat_kv``): the
  contiguous clone of the heads-major keys expanded to ``[1, Hk, n_rep, S, D]``
  (one load per output element), returned viewed as ``[1, Hk * n_rep, S, D]``.

Like Inductor, the norm and rotation stay in fp32 up to the single bf16 store
(the graph's intermediate bf16 rounding is not materialized). Inductor builds
``rotate_half`` from loads of the row and weight at ``r +- D/2``; the port
reads the two halves of the row and weight and writes the two output halves.
Inductor flattens ``(token, head)`` into one row index; the port tiles both.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def q_rmsnorm_rope(
    q: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # q: [T, Hq, D] view of the projection; cos, sin: [T, D].
    tokens, heads, head_dim = q.shape
    half = head_dim // 2
    out = torch.empty([tokens, heads, head_dim], dtype=q.dtype, device=q.device)
    for tile_t, tile_h in hl.tile([tokens, heads]):
        x = q[tile_t, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + 1e-06)
        lo = (
            q[tile_t, tile_h, :half].to(torch.float32)
            * rstd
            * (weight[:half].to(torch.float32) + 1.0)
        )
        hi = (
            q[tile_t, tile_h, half:].to(torch.float32)
            * rstd
            * (weight[half:].to(torch.float32) + 1.0)
        )
        cos_lo = cos[tile_t, :half].to(torch.float32)[:, None, :]
        cos_hi = cos[tile_t, half:].to(torch.float32)[:, None, :]
        sin_lo = sin[tile_t, :half].to(torch.float32)[:, None, :]
        sin_hi = sin[tile_t, half:].to(torch.float32)[:, None, :]
        out[tile_t, tile_h, :half] = (lo * cos_lo + (-hi) * sin_lo).to(out.dtype)
        out[tile_t, tile_h, half:] = (hi * cos_hi + lo * sin_hi).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def k_rmsnorm_rope(
    k: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # k: [T, Hk, D] view of the projection; cos, sin: [T, D].
    tokens, heads, head_dim = k.shape
    half = head_dim // 2
    heads_major = torch.empty([heads, tokens, head_dim], dtype=k.dtype, device=k.device)
    token_major = torch.empty([tokens, heads, head_dim], dtype=k.dtype, device=k.device)
    heads_major_t = heads_major.permute(1, 0, 2)  # [T, Hk, D] view
    for tile_t, tile_h in hl.tile([tokens, heads]):
        x = k[tile_t, tile_h, :].to(torch.float32)
        rstd = torch.rsqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + 1e-06)
        lo = (
            k[tile_t, tile_h, :half].to(torch.float32)
            * rstd
            * (weight[:half].to(torch.float32) + 1.0)
        )
        hi = (
            k[tile_t, tile_h, half:].to(torch.float32)
            * rstd
            * (weight[half:].to(torch.float32) + 1.0)
        )
        cos_lo = cos[tile_t, :half].to(torch.float32)[:, None, :]
        cos_hi = cos[tile_t, half:].to(torch.float32)[:, None, :]
        sin_lo = sin[tile_t, :half].to(torch.float32)[:, None, :]
        sin_hi = sin[tile_t, half:].to(torch.float32)[:, None, :]
        y_lo = (lo * cos_lo + (-hi) * sin_lo).to(k.dtype)
        y_hi = (hi * cos_hi + lo * sin_hi).to(k.dtype)
        heads_major_t[tile_t, tile_h, :half] = y_lo
        heads_major_t[tile_t, tile_h, half:] = y_hi
        token_major[tile_t, tile_h, :half] = y_lo
        token_major[tile_t, tile_h, half:] = y_hi
    return heads_major, token_major


@helion.kernel(static_shapes=True)
def repeat_kv(src: torch.Tensor) -> torch.Tensor:
    # src: [Hk, n_rep, S * D], the heads-major keys expanded (stride 0) over n_rep.
    heads, n_rep, n = src.shape
    out = torch.empty([heads, n_rep, n], dtype=src.dtype, device=src.device)
    for tile_h, tile_r, tile_n in hl.tile([heads, n_rep, n]):
        out[tile_h, tile_r, tile_n] = src[tile_h, tile_r, tile_n]
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    q_proj, q_weight, cos, sin, k_proj, k_weight = inputs[:6]
    batch, seq, _ = inputs[6]  # _shape_param_0 = [B, S, Hq * D]
    head_dim = inputs[7][-1]  # _shape_param_1 = [B, S, -1, D]
    n_rep = inputs[10][2]  # _shape_param_4 = [B, Hk, n_rep, S, D]
    if batch != 1:
        # The heads-major keys [B, Hk, S, D] are viewed as [Hk, T, D].
        raise ValueError("this port handles batch size 1")
    tokens = batch * seq
    q_heads = q_proj.size(1) // head_dim
    k_heads = k_proj.size(1) // head_dim
    cos_t = cos.view(tokens, head_dim)
    sin_t = sin.view(tokens, head_dim)
    q_out = q_rmsnorm_rope(
        q_proj.view(tokens, q_heads, head_dim), q_weight, cos_t, sin_t
    )
    k_heads_major, k_token_major = k_rmsnorm_rope(
        k_proj.view(tokens, k_heads, head_dim), k_weight, cos_t, sin_t
    )
    k_repeated = repeat_kv(
        k_heads_major.view(k_heads, 1, seq * head_dim).expand(
            k_heads, n_rep, seq * head_dim
        )
    )
    q_out = q_out.view(batch, seq, q_heads, head_dim).permute(0, 2, 1, 3)
    k_out = k_token_major.view(batch, seq, k_heads, head_dim).permute(0, 2, 1, 3)
    return (
        q_out,
        k_repeated.view(batch, k_heads * n_rep, seq, head_dim),
        k_out[:, :, -1023:],
    )
