# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""Device helpers of the GLM-5.3 Flash MoE megakernels (``moe_tp4``, ``dense_tp4``,
``mtp_moe_tp4``): the noaux_tc routing, the trtllm-gen W13 tile's up / gate halves,
SGLang's clamped SwiGLU and the NVFP4 rounding of the activations."""

from __future__ import annotations

from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import FP4_MAX
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import fp4_nibble
import torch

import helion.language as hl


def _top2(first_a, second_a, first_b, second_b):
    """Combine of a (largest, second largest) pair reduction."""
    return (
        torch.maximum(first_a, first_b),
        torch.maximum(
            torch.minimum(first_a, first_b), torch.maximum(second_a, second_b)
        ),
    )


def route(logits, bias, live, experts, lanes, routed_scale):
    """A token's top 8 from its FP32 router logits [lanes] (``lanes`` >= experts;
    lanes past the experts never picked): sigmoid scores, the largest score +
    bias in full FP32, ties to the lower expert. One int64 max per pick: the
    key packs the order-preserving bits of score + bias above the reversed
    expert index (two picks per reduction: a (largest, second) pair). The
    weights: the picks' scores over their sum times ``routed_scale``. Returns
    the ids in pick order (``experts`` where ``live`` fails), the FP32 weights
    and the picks as a bitmap of ``lanes // 32`` int32 words (bit b of word w:
    expert 32 w + b)."""
    scores = torch.sigmoid(logits)
    bits = (scores + bias).view(torch.int32)
    indices = hl.arange(lanes).to(torch.int32)
    ordered = torch.where(bits < 0, -2147483648 - bits, bits)
    keys = (ordered.to(torch.int64) << 32) | (lanes - 1 - indices).to(torch.int64)
    keys = torch.where(indices < experts, keys, -9223372036854775807)
    picks = hl.arange(8)
    ids = hl.full([8], 0, dtype=torch.int32)
    # two picks per reduction (keys are distinct: the index bits)
    floor = hl.full([lanes], -9223372036854775807, dtype=torch.int64)
    for pair in hl.static_range(4):
        first, second = hl.reduce(_top2, (keys, floor), dim=0)
        keys = torch.where(
            (keys == first) | (keys == second), -9223372036854775807, keys
        )
        ids = torch.where(
            picks == 2 * pair, lanes - 1 - (first & 4294967295).to(torch.int32), ids
        )
        ids = torch.where(
            picks == 2 * pair + 1,
            lanes - 1 - (second & 4294967295).to(torch.int32),
            ids,
        )
    # every pick's score in one reduction (one-hot sums: exact)
    weights = torch.sum(
        torch.where(indices[None, :] == ids[:, None], scores[None, :], 0.0), dim=1
    )
    total = torch.sum(weights)
    ids = torch.where(live, ids, experts)
    words = hl.arange(lanes // 32)
    union = torch.sum(
        torch.where(
            ((ids >> 5)[None, :] == words[:, None]) & live[None, :],
            1 << (ids & 31)[None, :],
            0,
        ),
        dim=1,
    )
    return ids, weights / total * routed_scale, union


def up_gate(acc):
    """A W13 tile's accumulator [128 trtllm-gen rows, T] -> its up and gate halves
    [T, 4, 8, 2] in intermediate order (64 intermediates: 4 groups of 16).
    Physical row ``b * 32 + 16 h + 8 u + m`` of the tile holds intermediate
    ``16 b + 2 m + h``, up (u = 0) or gate (u = 1)."""
    tokens = acc.size(1)
    # [T, 4 blocks, half, up/gate, m] -> [T, 4, m, half, up/gate]
    return hl.split(acc.T.reshape(tokens, 4, 2, 2, 8).permute(0, 1, 4, 2, 3))


def nvfp4_pairs(low, high, amax, global_scale):
    """NVFP4 of [R, G] groups of 16 values given as their even / odd halves ``low`` /
    ``high`` [R, G, 8] (FP32) and each group's amax [R, G]: packed bytes [R, G, 8]
    (even values in the low nibble) and E4M3 block scales [R, G]."""
    block_scale = (amax * global_scale / FP4_MAX).to(torch.float8_e4m3fn)
    actual = block_scale.to(torch.float32)
    # (unsqueeze: a [:, :, None] slice of a computed tile fails to lower in some
    # roots)
    divisor = torch.unsqueeze(torch.where(actual > 0, actual, 1.0), 2)
    packed = (
        fp4_nibble(low * global_scale / divisor)
        | fp4_nibble(high * global_scale / divisor) << 4
    ).to(torch.uint8)
    return packed, block_scale


def clamped_swiglu_nvfp4(acc, alpha, global_scale):
    """A W13 tile's FP32 accumulator [128 trtllm-gen rows, T] (``up_gate``) -> SGLang's
    NVFP4 linear SwiGLU (the shared expert's, the dense MLP's): the alpha, then the
    clamps (|up| <= 10, gate <= 10: ``_setup.SWIGLU_LIMIT``), SwiGLU and NVFP4
    (``swiglu_nvfp4``): packed [T, 4, 8], E4M3 scales [T, 4]."""
    up, gate = up_gate(acc)
    return swiglu_nvfp4(
        torch.clamp(up * alpha, min=-10.0, max=10.0),
        torch.clamp(gate * alpha, max=10.0),
        global_scale,
    )


def swiglu_nvfp4(up, gate, global_scale):
    """silu(gate) * up [T, G, 8, 2] as NVFP4 (``nvfp4_pairs``): packed bytes [T, G, 8]
    (G groups of 16 intermediates) and E4M3 scales [T, G]."""
    tokens, groups = up.size(0), up.size(1)
    act = gate * torch.sigmoid(gate) * up
    low, high = hl.split(act)
    amax = torch.amax(torch.abs(act.reshape(tokens, groups, 16)), dim=-1)
    return nvfp4_pairs(low, high, amax, global_scale)
