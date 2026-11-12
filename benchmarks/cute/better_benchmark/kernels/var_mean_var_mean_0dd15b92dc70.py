"""better-benchmark repro var_mean_var_mean_0dd15b92dc70 (hf MBartForCausalLM inference).

Inductor emits one persistent reduction,
``triton_per_fused_add_convert_element_type_embedding_mul_rsqrt_sub_unsqueeze_var_mean_0``,
over the ``B*S`` token rows of the MBart decoder input: the token embedding
gather (``embedding(table, ids)``; the graph's ``* 1.0`` embed scale is an
identity Inductor drops) plus the learned positional embedding of position
``s + 2`` (``iota(S) + 2``, broadcast over the batch), then
``layernorm_embedding`` (fp32 two-pass ``var_mean``, eps 1e-5, affine) stored
in bf16 as the first output, then the first decoder layer's
``self_attn_layer_norm`` of that result stored in bf16 and returned as three
``[B*S, D]`` views (the q, k and v projection inputs). Like Inductor (no
``emulate_precision_casts``), the graph's bf16 rounding of the embedding sum
and of the first LayerNorm's output stays in fp32 inside the kernel; only the
stores round.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def embed_pos_layernorm_layernorm(
    ids: torch.Tensor,
    token_table: torch.Tensor,
    pos_table: torch.Tensor,
    ln1_weight: torch.Tensor,
    ln1_bias: torch.Tensor,
    ln2_weight: torch.Tensor,
    ln2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ids: [B, S] int64; token_table: [V, D]; pos_table: [S + 2, D];
    # ln*_weight, ln*_bias: [D].
    batch, seq = ids.shape
    dim = token_table.size(1)
    rows = batch * seq
    flat_ids = ids.view(rows)
    ln1_out = torch.empty([rows, dim], dtype=token_table.dtype, device=ids.device)
    ln2_out = torch.empty([rows, dim], dtype=token_table.dtype, device=ids.device)
    for tile_r in hl.tile(rows):
        token = flat_ids[tile_r]
        # embedding(pos_table, iota(S) + 2), broadcast over the batch.
        position = tile_r.index % seq + 2
        x = token_table[token, :].to(torch.float32) + pos_table[position, :].to(
            torch.float32
        )
        # torch.mean divides by the count, as Inductor does (a constant
        # divisor would become a reciprocal multiply).
        mean = torch.mean(x, dim=-1, keepdim=True)
        centered = x - mean
        var = torch.mean(centered * centered, dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + 1e-05)
        y = y * ln1_weight[None, :].to(torch.float32) + ln1_bias[None, :].to(
            torch.float32
        )
        ln1_out[tile_r, :] = y.to(ln1_out.dtype)
        mean2 = torch.mean(y, dim=-1, keepdim=True)
        centered2 = y - mean2
        var2 = torch.mean(centered2 * centered2, dim=-1, keepdim=True)
        z = centered2 * torch.rsqrt(var2 + 1e-05)
        z = z * ln2_weight[None, :].to(torch.float32) + ln2_bias[None, :].to(
            torch.float32
        )
        ln2_out[tile_r, :] = z.to(ln2_out.dtype)
    return ln1_out, ln2_out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    token_table, ids, pos_table = inputs[:3]
    ln1_weight, ln1_bias, ln2_weight, ln2_bias = inputs[3:7]
    q_shape, k_shape, v_shape = inputs[7:10]  # each [B*S, D]
    batch, seq = ids.shape
    ln1_out, ln2_out = embed_pos_layernorm_layernorm(
        ids, token_table, pos_table, ln1_weight, ln1_bias, ln2_weight, ln2_bias
    )
    return (
        ln1_out.view(batch, seq, token_table.size(1)),
        ln2_out.view(q_shape),
        ln2_out.view(k_shape),
        ln2_out.view(v_shape),
    )
