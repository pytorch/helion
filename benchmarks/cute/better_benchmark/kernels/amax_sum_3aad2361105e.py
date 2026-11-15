"""better-benchmark repro amax_sum_3aad2361105e (genai/static/CrossEntropyForward).

Inductor emits one looped reduction,
``triton_red_fused_convert_element_type_full_gather_log_ne_neg_prepare_softmax_online_squeeze_sub_unsqueeze_where_0``:
the per-row cross-entropy loss (no reduction, ``ignore_index=-100``) of bf16
logits ``[N, V]`` against int64 targets ``[N]``. One loop over V blocks folds
each block into a per-row fp32 (max, sum) state the way Inductor's
``online_softmax_reduce_scalar_combine`` does: reduce the block to its own max
and its sum of ``exp(x - block_max)``, then combine with the running state,
rescaling whichever side has the smaller max by ``exp(-|max_a - max_b|)``.
After the loop the kernel gathers the target logit at
``where(target != -100, target, 0)``, computes
``-((x[target] - max) - log(sum))``, zeroes the ignored rows and stores the
bf16 ``[N]`` losses. The graph's bf16 ``log_softmax`` ``[N, V]`` is never
materialized; like Inductor, the gathered value stays in fp32 until the store.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # logits: [N, V] bf16; target: [N] int64 class indices (-100 = ignored).
    n, v = logits.shape
    out = torch.empty([n], dtype=logits.dtype, device=logits.device)
    for tile_n in hl.tile(n):
        row_max = hl.full([tile_n], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_n], dtype=torch.float32)
        for tile_v in hl.tile(v):
            x = logits[tile_n, tile_v].to(torch.float32)
            # online_softmax_reduce: the block's max and sum of exp(x - max);
            # an all -inf block uses 0 instead of -inf - -inf = nan.
            block_max = torch.amax(x, dim=-1)
            delta = torch.where(
                block_max[:, None] == float("-inf"), 0.0, x - block_max[:, None]
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
        label = target[tile_n]
        valid = label != -100
        safe_label = torch.where(valid, label, 0)
        # gather(log_softmax, 1, safe_label[:, None]).squeeze(1), fused: read
        # the one target logit of each row.
        picked = logits[tile_n.index[:, None], safe_label[:, None]].squeeze(1)
        picked = picked.to(torch.float32)
        loss = -((picked - row_max) - torch.log(row_sum))
        out[tile_n] = torch.where(valid, loss, 0.0).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    target, logits = inputs[:2]
    return (cross_entropy_loss(logits, target),)
