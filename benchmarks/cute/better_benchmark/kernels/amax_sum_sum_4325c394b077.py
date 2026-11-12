"""better-benchmark repro amax_sum_sum_4325c394b077 (genai SoftmaxBackward, static).

The softmax statistics of a bf16 ``[M, N]`` input in fp32 (the row ``amax``
and ``sum(exp(x - amax))``, both returned as fp32 ``[M, 1]``) and the bf16 sum
of all the softmax probabilities. Inductor emits two kernels:

- ``triton_red_fused_convert_element_type_div_exp_prepare_softmax_online_sub_sum_0``:
  per row, a first loop over N blocks runs the online softmax
  (``prepare_softmax_online``) the way Inductor's
  ``online_softmax_reduce_scalar_combine`` does: reduce the block to its own
  max and its sum of ``exp(x - block_max)``, then fold that into the running
  (max, sum), rescaling whichever side has the smaller max by
  ``exp(-|max_a - max_b|)``. The final state is stored as the ``amax`` and
  ``sum_1`` outputs. A second loop over the same ``R0_BLOCK`` reloads the row
  and adds each block's sum of the probabilities ``exp(x - max) / sum`` to a
  per-row accumulator (``[XBLOCK, 1]``: Inductor's scalar accumulators are
  on by default), stored after the loop into an fp32 ``[M]`` workspace; it
  adds them in fp32 without first rounding them to bf16 (the graph's
  ``convert_element_type`` before ``sum_2``), as the port does. The port
  spells that loop as a second ``hl.tile`` over ``block_n`` with a
  ``[tile_m, block_n]`` accumulator summed once after the loop (the same
  terms, added in another fp32 order): a full-row ``torch.sum`` would give
  the second pass a reduction block of its own, which on CuTe also claims a
  thread axis of its own beside the first loop's, so that each pass runs
  repeated across the other's threads.
- ``triton_red_fused_convert_element_type_div_exp_sub_sum_1``: sums the ``M``
  per-row partials into the 0-d bf16 ``sum_2`` (a ``[1, M]`` view of the
  workspace reduced over its single row).

The online softmax adds one block sum to each running row sum per iteration,
so its column block is at least a warp (32) wide: one column per iteration
turns ``sum_1`` into a 262144-term sequential fp32 sum, about 5e-5 relative
error against the anchored gate's 1e-5 (Inductor's reduction heuristic
starts this kernel at ``R0_BLOCK=1024``).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def softmax_stats_row_prob_sums(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.shape
    row_max_out = torch.empty([m], dtype=torch.float32, device=x.device)
    row_sum_out = torch.empty([m], dtype=torch.float32, device=x.device)
    prob_sums = torch.empty([m], dtype=torch.float32, device=x.device)
    # At least a warp of columns per online-softmax step (module docstring).
    block_n = hl.register_block_size(32, n)
    for tile_m in hl.tile(m):
        row_max = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            vals = x[tile_m, tile_n].to(torch.float32)
            # online_softmax_reduce: the block's max and sum of exp(x - max),
            # with an all -inf block summing exp(0) instead of nan.
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
        row_max_out[tile_m] = row_max
        row_sum_out[tile_m] = row_sum
        acc = hl.zeros([tile_m, block_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + (
                torch.exp(x[tile_m, tile_n].to(torch.float32) - row_max[:, None])
                / row_sum[:, None]
            )
        prob_sums[tile_m] = torch.sum(acc, dim=-1)
    return row_max_out, row_sum_out, prob_sums


@helion.kernel(static_shapes=True)
def total_prob_sum(prob_sums: torch.Tensor) -> torch.Tensor:
    # prob_sums: [1, M] view of the per-row probability sums.
    rows, _ = prob_sums.shape
    out = torch.empty([rows], dtype=torch.bfloat16, device=prob_sums.device)
    for tile_r in hl.tile(rows):
        out[tile_r] = torch.sum(prob_sums[tile_r, :], dim=-1).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    (x,) = inputs[:1]
    m = x.shape[0]
    row_max, row_sum, prob_sums = softmax_stats_row_prob_sums(x)
    total = total_prob_sum(prob_sums.view(1, m))
    return row_max.view(m, 1), row_sum.view(m, 1), total.view([])
