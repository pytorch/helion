"""Helpers of the Qwen3.5 TP4 MoE megakernels (the MoE and the merged MoE + GDN
layer): the 16-group scale tiles, the BF16 routing keys and the routing itself (a
token's top-k and the window's distinct experts) and a token's picks as an expert
bitmap (``pick_bits``, also GLM-5.3's); the NVFP4 rounding and the 128-row scale
tiles are DeepSeek's (``deepseek_v3_moe_nvfp4/_common.py``)."""

from __future__ import annotations

import operator

from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import popcount
import torch

import helion.language as hl

# A routing key no logit has: below every bf16_key (a picked or padding lane).
PICKED32 = -2147483647


def scale_tiles_16(chunk: torch.Tensor) -> torch.Tensor:
    """[16, 256] swizzled scale bytes of two 128-row tiles and 16 groups -> [256, 16]."""
    return chunk.reshape(2, 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(256, 16)


def bf16_key(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """int32 keys ordered by the BF16 value, then by the lower index (< 65536
    indices): the value's order-preserving 16 bits above the index's."""
    bits = logits.view(torch.int16).to(torch.int32)
    ordered = torch.where(bits < 0, -32768 - bits, bits)
    return ordered * 65536 + (65535 - index)


def key_value(key: torch.Tensor) -> torch.Tensor:
    """The float32 value of a bf16_key's BF16 logit."""
    ordered = key >> 16
    bits = torch.where(ordered < 0, -32768 - ordered, ordered)
    return bits.to(torch.int16).view(torch.bfloat16).to(torch.float32)


def imax(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """int32 max: a reduction with this one combiner lowers to redux.sync across a warp."""
    return torch.maximum(a, b)


def expert_of(
    words: torch.Tensor, counts: torch.Tensor, before: torch.Tensor, j: object
) -> torch.Tensor:
    """Expert of the j-th set bit (ascending) of a bitmap of 16 words (word w's bit
    b is expert 32 w + b), given each word's popcount and exclusive prefix count."""
    hit = (before <= j) & (before + counts > j)
    word = hl.reduce(operator.or_, torch.where(hit, words, 0), dim=0)
    word_index = torch.sum(torch.where(hit, hl.arange(16), 0))
    rank = j - torch.sum(torch.where(hit, before, 0))
    # int32 lanes (an int64 index would widen the popcount); set bits at or below
    # each position (2 << 31 wraps to 0: all 32)
    bits = hl.arange(32).to(torch.int32)
    upto = popcount(word & ((2 << bits) - 1))
    return word_index * 32 + torch.sum((upto <= rank).to(torch.int32))


def route(
    logits: torch.Tensor,
    live: torch.Tensor,
    top_k: int,
    experts: int,
    picks: torch.Tensor,
    groups: int,
    segment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A token's top-k from its BF16 logits [groups, group_n] (``bf16_key``
    order): each group's top-k candidates, then the top-k of those.  Its picks'
    ids (``experts`` where ``live`` fails), their softmax weights, and the picks
    as a bitmap of 16 words (bit b of word w: expert 32 w + b)."""
    tk_groups = hl.arange(groups)
    tk_slots = hl.arange(logits.size(1))
    keys = bf16_key(
        logits,
        (tk_groups[:, None] * logits.size(1) + tk_slots[None, :]).to(torch.int32),
    )
    tk_lists = torch.full_like(keys, PICKED32)
    for rnd in hl.static_range(top_k):
        tk_group_max = hl.reduce(imax, keys, dim=1)
        keys = torch.where(keys == tk_group_max[:, None], PICKED32, keys)
        tk_lists = torch.where(
            tk_slots[None, :] == tk_groups[:, None] * segment + rnd,
            tk_group_max[:, None],
            tk_lists,
        )
    tk_cands = hl.reduce(imax, tk_lists, dim=0)
    tops = torch.full_like(picks, PICKED32)
    for rnd in hl.static_range(top_k):
        tk_top = hl.reduce(imax, tk_cands, dim=0)
        tk_cands = torch.where(tk_cands == tk_top, PICKED32, tk_cands)
        tops = torch.where(picks == rnd, tk_top, tops)
    picked = (picks < top_k) & live
    ids = torch.where(picked, 65535 - (tops & 65535), experts)
    top_values = key_value(tops)
    top_max = torch.amax(torch.where(picks < top_k, top_values, float("-inf")))
    exps = torch.where(picks < top_k, torch.exp(top_values - top_max), 0.0)
    # a token's picks are distinct: the sum of their bits is their union
    words = hl.arange(16)
    bits = torch.where(
        ((ids >> 5)[None, :] == words[:, None]) & picked[None, :],
        1 << (ids & 31)[None, :],
        0,
    )
    return ids, exps / torch.sum(exps), torch.sum(bits, dim=1)


def pick_bits(ids: torch.Tensor, picked: torch.Tensor, words: int) -> torch.Tensor:
    """A token's picks as a bitmap of ``words`` int32 words (bit b of word w:
    expert 32 w + b), the bitmap of ``route`` for ids any router picks; the
    picks are distinct, so the sum of their bits is their union."""
    lanes = hl.arange(words)
    bits = torch.where(
        ((ids >> 5)[None, :] == lanes[:, None]) & picked[None, :],
        1 << (ids & 31)[None, :],
        0,
    )
    return torch.sum(bits, dim=1)
