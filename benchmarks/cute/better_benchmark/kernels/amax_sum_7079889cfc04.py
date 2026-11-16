"""better-benchmark repro amax_sum_7079889cfc04 (genai/static/SoftmaxForward).

Inductor emits one looped reduction,
``triton_red_fused_convert_element_type_div_exp_prepare_softmax_online_sub_0``:
a bf16 softmax over the last dim of a huge ``[8192, 262144]`` input (2**31
elements, so all index math is 64-bit). The first loop over column blocks
folds each block into a per-row fp32 (max, sum) state the way Inductor's
``online_softmax_reduce_scalar_combine`` does: reduce the block to its own max
and its sum of ``exp(x - block_max)``, then combine with the running state,
rescaling whichever side has the smaller max by ``exp(-|max_a - max_b|)``. The
second loop reloads the input and stores ``exp(x - max) / sum`` in bf16.

The two column loops tile independently (Inductor uses one ``R0_BLOCK`` for
both).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def online_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty([rows, cols], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(rows):
        row_max = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(cols):
            vals = x[tile_m, tile_n].to(torch.float32)
            # online_softmax_reduce: the block's max and sum of exp(x - max);
            # an all -inf block uses 0 instead of -inf - -inf = nan.
            block_max = torch.amax(vals, dim=-1)
            delta = torch.where(
                block_max[:, None] == float("-inf"), 0.0, vals - block_max[:, None]
            )
            block_sum = torch.sum(torch.exp(delta), dim=-1)
            # online_softmax_combine_with_sum
            new_max = torch.maximum(row_max, block_max)
            scale = torch.exp(-torch.abs(row_max - block_max))
            row_scale = torch.where(row_max == new_max, 1.0, scale)
            block_scale = torch.where(block_max == new_max, 1.0, scale)
            new_sum = row_sum * row_scale + block_sum * block_scale
            row_sum = torch.where(new_max == float("inf"), float("nan"), new_sum)
            row_max = new_max
        for tile_n in hl.tile(cols):
            vals = x[tile_m, tile_n].to(torch.float32)
            probs = torch.exp(vals - row_max[:, None]) / row_sum[:, None]
            out[tile_m, tile_n] = probs.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    x = inputs[0]
    return (online_softmax(x),)
