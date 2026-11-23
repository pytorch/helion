"""better-benchmark repro pointwise_7551a58e5d34 (hf DistillGPT2 training and inference).

The LM head's bf16 ``[batch, seq, vocab]`` activation, viewed as
``[rows, vocab]`` and zero-padded from V = 50257 to 50264 vocabulary entries (a
multiple of 8) twice: transposed (``[50264, rows]``) and as is
(``[rows, 50264]``). Inductor emits one pointwise kernel,
``triton_poi_fused_constant_pad_nd_permute_view_0``: a 2-D grid over
``[rows, 50264]`` that loads the activation once per element (masked to the
first V columns, the masked lanes reading as the padding's zero, widened to
fp32) and stores it in bf16 to both padded outputs, at ``[col, row]`` of the
transposed one and at ``[row, col]`` of the other. The graph's view and
permute are free. The port tiles the same two dimensions with the same masked
load and the same two stores.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def pad_vocab_and_transpose(
    x: torch.Tensor, padded_vocab: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [rows, vocab] bf16, zero-padded to padded_vocab columns and stored
    # transposed ([padded_vocab, rows]) and as is ([rows, padded_vocab]).
    rows, vocab = x.shape
    out_t = torch.empty([padded_vocab, rows], dtype=x.dtype, device=x.device)
    out_padded = torch.empty([rows, padded_vocab], dtype=x.dtype, device=x.device)
    for tile_r, tile_v in hl.tile([rows, padded_vocab]):
        in_vocab = tile_v.index < vocab
        v = hl.load(x, [tile_r, tile_v], extra_mask=in_vocab[None, :]).to(torch.float32)
        v = torch.where(in_vocab[None, :], v, 0.0)
        out_t[tile_v, tile_r] = v.permute(1, 0).to(out_t.dtype)
        out_padded[tile_r, tile_v] = v.to(out_padded.dtype)
    return out_t, out_padded


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    rows, vocab = inputs[1]  # _shape_param_0 = [batch * seq, vocab]
    pad_t = inputs[2]  # _shape_param_1 = [0, 0, 0, pad]: the permuted dim 0
    pad = inputs[3]  # _shape_param_2 = [0, pad, 0, 0]: dim 1
    if list(pad_t) != [0, 0, 0, pad[1]] or list(pad) != [0, pad[1], 0, 0]:
        raise ValueError(f"unexpected pads {pad_t} and {pad}")
    out_t, out_padded = pad_vocab_and_transpose(
        arg0_1.view(rows, vocab), vocab + pad[1]
    )
    return out_t, out_padded
