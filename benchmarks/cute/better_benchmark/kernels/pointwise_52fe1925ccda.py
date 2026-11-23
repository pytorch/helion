"""better-benchmark repro pointwise_52fe1925ccda (hf BartForCausalLM training and others).

The tied LM-head weight of a causal language model, prepared for the head's
matmuls: the fp32 ``[vocab, hidden]`` embedding is cast to bf16 and returned
both as is and transposed with its vocabulary dim zero-padded to a multiple of
8 (``[hidden, vocab + pad]``). Inductor emits two pointwise kernels:

- ``triton_poi_fused_constant_pad_nd_convert_element_type_permute_0``: a 2-D
  grid over ``[vocab, hidden]`` that loads the fp32 weight once and stores it
  in bf16 twice: transposed into the first ``vocab`` columns of the padded
  output (a ``[hidden, vocab]`` alias with row stride ``vocab + pad``) and
  untransposed into the ``[vocab, hidden]`` output.
- ``triton_poi_fused_constant_pad_nd_convert_element_type_permute_1``:
  zero-fills the ``pad`` trailing columns of the padded output (a
  ``[hidden, pad]`` alias of the same buffer); it loads nothing.

The port allocates the padded output on the host and passes each kernel the
alias Inductor passes it. The zero fill tiles its ``[hidden, pad]`` alias
(Inductor indexes it as one flat dimension and recovers the row and column
with a division and a remainder).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cast_and_transpose(x: torch.Tensor, out_t: torch.Tensor) -> torch.Tensor:
    # x: [vocab, hidden] fp32; out_t: [hidden, vocab] bf16, the unpadded
    # columns of the padded output (row stride vocab + pad).
    vocab, hidden = x.shape
    out = torch.empty([vocab, hidden], dtype=out_t.dtype, device=x.device)
    for tile_v, tile_h in hl.tile([vocab, hidden]):
        value = x[tile_v, tile_h]
        out_t[tile_h, tile_v] = value.permute(1, 0).to(out_t.dtype)
        out[tile_v, tile_h] = value.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def zero_fill(out: torch.Tensor) -> torch.Tensor:
    # out: [hidden, pad] bf16, the padded columns (row stride vocab + pad).
    rows, cols = out.shape
    for tile_r, tile_c in hl.tile([rows, cols]):
        out[tile_r, tile_c] = hl.zeros([tile_r, tile_c], dtype=out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    pad = inputs[1]  # _shape_param_0 = [0, pad, 0, 0]: the permuted dim 1
    if list(pad) != [0, pad[1], 0, 0]:
        raise ValueError(f"unexpected pad {pad}")
    vocab, hidden = arg0_1.shape
    padded = torch.empty(
        [hidden, vocab + pad[1]], dtype=torch.bfloat16, device=arg0_1.device
    )
    out = cast_and_transpose(arg0_1, padded[:, :vocab])
    zero_fill(padded[:, vocab:])
    return padded, out
