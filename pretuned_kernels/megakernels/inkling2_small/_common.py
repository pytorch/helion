"""Pieces shared by the Inkling-2-Small megakernels: device helpers of the attention roots (approximate rsqrt and
sigmoid, the q / k / r L2 clip, the verify rows' short conv) and of the MoE roots (route selection and weights, block
scale tiles, MXFP8 blocks), and the host views of vLLM's weights the kernels take."""

from __future__ import annotations

import math

from pretuned_kernels.megakernels.inkling2_small._fp4 import rcp_rn
from pretuned_kernels.megakernels.inkling2_small._fp4 import rdiv_rn
import torch

import helion.language as hl

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
E4M3 = torch.float8_e4m3fn
# Pages a window of 1024 keys before 8 new rows can touch (10), padded to a power of two (aranges).
PAGES = 1 << (math.ceil((1023 + 8) / 128)).bit_length()


def _rsqrt_approx(x: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "rsqrt.approx.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    """attention.py _sigmoid: rcp.approx(1 + ex2.approx(-x * log2e))."""
    return hl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; ex2.approx.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.f32 $0, t; }",
        "=f,f",
        [x],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _l2_clip(x: torch.Tensor, threshold: float, eps: float) -> torch.Tensor:
    """qk_l2_clip on rows of x [rows, d] (fp32): x * min(threshold / (eps + |x|), 1)."""
    total = torch.sum(x * x, dim=-1)
    norm = torch.where(total > 0.0, total * _rsqrt_approx(total), 0.0)
    coeff = rdiv_rn(threshold, eps + norm)
    coeff = torch.where(coeff >= 1.0, 1.0, coeff)
    return x * coeff[:, None]


def _verify_conv(
    part: torch.Tensor,
    col0: int,
    cols: torch.Tensor,
    row: int,
    conv_w: torch.Tensor,
    conv_cache: torch.Tensor,
    head: int,
    n_heads: int,
    conv_pages: torch.Tensor,
    conv_loc: torch.Tensor,
    conv_block: int,
    cache_end: torch.Tensor,
    taps: int,
) -> torch.Tensor:
    """One verify row of attention_prologue's static-weight sconv over qkvr columns col0 + cols (residual,
    (W + 1)^-0.5; the taps before the verify rows from the paged conv cache, zero before position 0), storing the
    row's conv input in the cache (row slot * n_heads + head of the stream); bf16 like the stock conv output."""
    width = cols.size(0)
    dims = hl.arange(width)
    tap = hl.arange(taps)
    src = row - (taps - 1) + tap
    in_window = src >= 0
    # The taps' history and weights (nothing in this launch writes them before) and every tap's partial rows, one
    # load each, all issued before the arithmetic; the taps then come out exactly (a one-hot sum) in tap order.
    pos = cache_end + src
    hist = (~in_window) & (pos >= 0)
    safe = torch.clamp_min(pos, 0)
    blk = hl.load(
        conv_pages, [torch.clamp_max(safe // conv_block, conv_pages.size(0) - 1)]
    )
    prior = hl.load(
        conv_cache,
        [
            ((blk * conv_block + safe % conv_block) * n_heads + head)[:, None],
            dims[None, :],
        ],
        extra_mask=hist[:, None],
    ).to(torch.float32)
    w = hl.load(conv_w, [cols[None, :], tap[:, None]]).to(torch.float32)
    loc = hl.load(conv_loc, [row])
    raw = torch.sum(
        hl.load(
            part,
            [slice(None), torch.clamp_min(src, 0)[:, None], (col0 + cols)[None, :]],
            extra_mask=in_window[None, :, None],
        ),
        dim=0,
    )
    raw = raw.to(torch.bfloat16).to(torch.float32)
    value = torch.where(in_window[:, None], raw, prior)
    acc = hl.zeros([width], dtype=torch.float32)
    for t in hl.static_range(taps):
        pick = (tap == t)[:, None]
        acc = acc + torch.sum(torch.where(pick, value, 0.0), dim=0) * torch.sum(
            torch.where(pick, w, 0.0), dim=0
        )
    current = torch.sum(torch.where((tap == taps - 1)[:, None], value, 0.0), dim=0)
    hl.store(
        conv_cache,
        [loc * n_heads + head, dims],
        current.to(torch.bfloat16),
        extra_mask=(hl.zeros([width], dtype=torch.int64) + loc) >= 0,
    )
    return ((acc + current) * (taps + 1) ** -0.5).to(torch.bfloat16).to(torch.float32)


def _or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a | b


def _add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def _popcount(v: torch.Tensor) -> torch.Tensor:
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    return (((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24


def _distinct_table(
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local expert ids [n] (< 128; -1 = not local) -> bitmap words [4], popcounts, exclusive prefix, distinct count."""
    lanes = hl.arange(4)
    row = ids.unsqueeze(0)
    bits = torch.where((row >> 5) == lanes.unsqueeze(1), 1 << (row & 31), 0)
    words = hl.reduce(_or, bits, dim=1)
    counts = _popcount(words)
    before = hl.cumsum(counts, dim=0) - counts
    return words, counts, before, hl.reduce(_add, counts, dim=0)


def _expert_of(
    words: torch.Tensor, counts: torch.Tensor, before: torch.Tensor, j: object
) -> torch.Tensor:
    """The j-th distinct local expert (ascending)."""
    hit = (before <= j) & (before + counts > j)
    word = hl.reduce(_or, torch.where(hit, words, 0), dim=0)
    key = hl.reduce(_or, torch.where(hit, before * 8 + hl.arange(4), 0), dim=0)
    upto = _popcount(word & ((2 << hl.arange(32)) - 1))
    bit = hl.reduce(_add, (upto <= j - (key >> 3)).to(torch.int32), dim=0)
    return (key & 7) * 32 + bit


def _local_ids(ids: torch.Tensor, base: int, experts: int) -> torch.Tensor:
    local = ids - base
    return torch.where((local >= 0) & (local < experts), local, -1)


def _select(
    logits: torch.Tensor, bias: torch.Tensor, n: int, n_pad: int
) -> torch.Tensor:
    """route_topk at top-8: the biased sigmoid's stable top-8 (key desc, id asc: the sort's order) as eight rounds of
    a max over the ordered keys, then the lowest remaining id at that key."""
    experts = hl.arange(n_pad)
    selection = rcp_rn(1.0 + torch.exp(0.0 - logits)) + bias
    kbits = (selection + 0.0).view(dtype=torch.int32)
    ordered = torch.where(kbits >= 0, kbits, kbits ^ 0x7FFFFFFF)
    lowest = -2147483647 - 1
    ordered = torch.where(experts < n, ordered, lowest)
    rank8 = hl.arange(8)
    best = hl.zeros([8], dtype=torch.int32)
    for kk in hl.static_range(8):
        top = torch.amax(ordered, dim=0)
        pick = torch.amin(torch.where(ordered == top, experts, n_pad), dim=0)
        best = torch.where(rank8 == kk, pick, best)
        ordered = torch.where(experts == pick, lowest, ordered)
    return best


def _renorm(
    logits: torch.Tensor, sel_ids: torch.Tensor, n_pad: int, route_scale: float
) -> torch.Tensor:
    """renorm_topk_logits_scaled's logsigmoid weights of the selected ids (its summation order)."""
    experts = hl.arange(n_pad)
    picked = torch.sum(
        torch.where(experts[None, :] == sel_ids[:, None], logits[None, :], 0.0), dim=-1
    )
    log_probs = torch.minimum(picked, hl.zeros([8], dtype=torch.float32)) - torch.log1p(
        torch.exp(0.0 - torch.abs(picked))
    )
    top_lp = torch.amax(log_probs, dim=0)
    shifted = torch.exp(log_probs - top_lp)
    total = (
        (_pick(shifted, 0) + _pick(shifted, 2))
        + (_pick(shifted, 1) + _pick(shifted, 3))
    ) + (
        (_pick(shifted, 4) + _pick(shifted, 6))
        + (_pick(shifted, 5) + _pick(shifted, 7))
    )
    return torch.exp(log_probs - (top_lp + torch.log(total))) * route_scale


def _pick(values: torch.Tensor, i: int) -> torch.Tensor:
    return torch.sum(torch.where(hl.arange(8) == i, values, 0.0), dim=0)


def _scale_tile(chunks: torch.Tensor, col_blocks: int) -> torch.Tensor:
    """Swizzled scale bytes of one 128-row tile over one K step ([col_blocks * 2, 256]: CUTLASS's 512-byte
    (32 row, 4 row group, 4 column) atoms, K-major) -> [128 rows, col_blocks * 4]."""
    return (
        chunks.reshape(1, col_blocks, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(128, col_blocks * 4)
    )


def _mxfp8_block(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """moe.py _mxfp8_block over rows of 32 values ([n, 32] fp32): e8m0 exponents (amax / 448 rounded up to a power of
    two) and the clamped e4m3 values."""
    amax = torch.amax(torch.abs(groups), dim=-1)
    raw = amax * (1.0 / 448.0)
    exponent = ((raw.view(dtype=torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(
        raw <= 2.0**-127,
        0,
        torch.minimum(exponent, hl.full([1], 254, dtype=torch.int32)),
    )
    exponent = torch.where(raw != raw, 255, exponent)
    scale = torch.maximum(
        (exponent << 23).view(dtype=torch.float32),
        hl.full([1], 2.0**-127, dtype=torch.float32),
    )
    inverse = torch.where(exponent == 0, 2.0**127, rcp_rn(scale))
    inverse = torch.where(exponent == 255, 0.0, inverse)
    values = torch.clamp(groups * inverse[:, None], -448.0, 448.0)
    return values.to(E4M3), exponent.to(torch.uint8)


def attn_views(
    w_qkvr: torch.Tensor, wo: torch.Tensor, k_splits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's qkvr [N, H] and wo_ud [H, 1536] weights as the kernel's (N, K split, K) / (H, K quarter, K)."""
    n, h = w_qkvr.shape
    return w_qkvr.view(n, k_splits, h // k_splits), wo.view(
        wo.shape[0], 4, wo.shape[1] // 4
    )


def moe_views(
    w13: torch.Tensor, w13_s: torch.Tensor, w2: torch.Tensor, w2_s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLLM's expert weights (rows shuffled in blocks of 32) and R128c4 block scales ([E, N, K / 16] E4M3, as
    uint8) as the kernel's (expert, 128-row tile, row, K step, byte) / (expert, 128-row tile, K step, chunk) views."""
    e, n13, kb13 = w13.shape
    _, h, kb2 = w2.shape
    return (
        w13.view(e, n13 // 128, 128, kb13 // 256, 256),
        w13_s.view(e, n13 // 128, kb13 // 256, 16, 256),
        w2.view(e, h // 128, 128, kb2 // 128, 128),
        w2_s.view(e, h // 128, kb2 // 128, 8, 256),
    )


def mx_views(
    w13: torch.Tensor, w13_s: torch.Tensor, w2: torch.Tensor, w2_s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLLM's MXFP8 expert weights [E, N, K] (e4m3, rows shuffled in blocks of 32) and their R128c4 e8m0 scales
    (uint8) as the kernel's views: weights (expert, 128-row tile, row, 256-value K step, value), scales (expert,
    128-row tile, 256-value K step, the step's two 512-byte (32 row, 4 row group, 4 column) atoms as [8, 128]: a
    tensor-descriptor box needs 8 rows)."""
    e, n13, k13 = w13.shape
    _, h, k2 = w2.shape
    return (
        w13.view(e, n13 // 128, 128, k13 // 256, 256),
        w13_s.reshape(e, n13 // 128, k13 // 256, 8, 128),
        w2.view(e, h // 128, 128, k2 // 256, 256),
        w2_s.reshape(e, h // 128, k2 // 256, 8, 128),
    )


def unswizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """The scales as the kernel reads them: vLLM's R128c4 layout as is (the harnesses' hook)."""
    return scales.contiguous()
