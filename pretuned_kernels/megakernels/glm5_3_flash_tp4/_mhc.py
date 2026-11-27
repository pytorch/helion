# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""Helion device helpers of GLM-5.3 Flash's mHC layer boundary (``_setup.boundary``):
the all-reduce of a sublayer's per-rank partials, hc_post and the next sublayer's
hc_pre, as the roots of the KDA, MLA-out, MoE and dense megakernels' tails; and
their host-side operands, scratch and outputs (``boundary_operands``,
``tail_args``, ``new_outputs``).

- ``chunk`` (one task per hidden chunk, all window tokens): x = the partials summed
  in list order (FP32, BF16 out, as an all-reduce rounds), R'_o = bf16(post_o x +
  sum_i comb[i, o] R_i) for the 4 streams, and this chunk's partials of the 24
  pre-norm mixes (R' . fn), and of the streams' Gram matrix G_ik = R'_i . R'_k
  (10 values): sum(R'^2) is its trace and sum(y^2) = pre^T G pre, so the layer
  input needs no second full-row reduction after pre.
- ``finalize`` (one task per hidden chunk): pre from the summed mixes, the y chunk
  from R' (read back), the RMSNorm -> the next sublayer's input (BF16).
- ``comb_logits`` (one task per 2 tokens): post' and comb's logits for the next
  boundary, then in the same task its 20 Sinkhorn iterations (``sinkhorn_init`` +
  ``sinkhorn_step`` on 16 per-entry token vectors: every row / column sum is an
  elementwise add, no cross-lane reduction), off the layer input's path. Or the
  *raw* form: the logits alone, the iterations left to the next boundary's
  megakernel: in a head root long before its chunk root reads comb, or in each
  chunk task before its all-reduce wait (``chunk``'s ``comb_tile``), whichever its
  placement hides.

The tails keep the per-split sums token-major ([8, S] tiles: ``mix_t``,
``gram_part``, ``sq_part``), so the split sums are warp-local reductions. Every
split sum runs in chunk order (deterministic). Window lanes past T (tiles of 8
tokens) are masked by the tile; their partial rows are never read.
"""

from __future__ import annotations

import functools

import torch

import helion.language as hl

RMS_EPS = 1e-5
HC_EPS = 1e-6
NORM_EPS = 1e-5
ITERS = 20
POST_MULT = 2.0
# Gram entries (i, k), i <= k, are stored in the order (0,0) (0,1) (0,2) (0,3)
# (1,1) (1,2) (1,3) (2,2) (2,3) (3,3): the trace is entries 0, 4, 7, 9.


def chunk(
    parts,
    tt,
    tc,
    r,
    post,
    comb,
    fn_pad,
    r_new,
    mix_part,
    gram_part,
    sq_part,
    ksplit=1,
    *,
    mix_t,
    comb_tile=False,
):
    """Root body for the tile (tokens ``tt``, hidden chunk ``tc``): see the module
    docstring. ``parts``: the tensors [T, H] whose sum is the sublayer output (the
    ranks' symmetric partials in rank order, or one AR'd output). ``fn_pad`` [32, 4, H]
    BF16: the hc fn rows (pre 0:4, post 4:8, comb 8:24, zero 24:32) per stream; the 24
    mixes are 4 MMAs (BF16 x BF16 products are exact, FP32 sums) into ``mix_part``
    [S, 8, 32]. ``ksplit`` > 1: each part is [ksplit, T, H] (a K-split producer's
    partials), summed split-major over the ranks (a fixed order). ``mix_t`` [8, 32, S]:
    the mixes token-major too (``finalize`` reads them), ``gram_part`` [8, 4, 4, S]
    and ``sq_part`` [8, S] token-major. ``comb_tile``: ``comb`` is the
    tile's final comb itself, an [8, 4, 4] value (the caller's), else the [T, 4, 4]
    tensor. Every load is issued before any arithmetic or store."""
    if ksplit > 1:
        xs = [p[s, tt, tc] for s in range(ksplit) for p in parts]
    else:
        xs = [p[tt, tc] for p in parts]
    rs = [r[tt, i, tc].to(torch.float32) for i in range(4)]
    pv = [post[tt, i] for i in range(4)]
    if comb_tile:
        # its 16 entries as token vectors: masked sums over the 4 x 4 tile, exact
        # (one nonzero term; a Sinkhorn comb is positive, so the zeros' sign
        # never matters)
        i4 = hl.arange(4)
        cv = [
            torch.sum(
                torch.sum(
                    torch.where(
                        ((i4[:, None] == k // 4) & (i4[None, :] == k % 4))[None, :, :],
                        comb,
                        0.0,
                    ),
                    dim=2,
                ),
                dim=1,
            )
            for k in range(16)
        ]
    else:
        cv = [comb[tt, k // 4, k % 4] for k in range(16)]  # [i, o] row-major
    fw = [fn_pad[:, i, tc] for i in range(4)]  # [32, C] bf16
    total = xs[0].to(torch.float32)
    for j in hl.static_range(len(xs) - 1):
        total = total + xs[j + 1].to(torch.float32)
    xv = total.to(torch.bfloat16).to(torch.float32)
    vb = [
        (
            pv[o][:, None] * xv
            + cv[o][:, None] * rs[0]
            + cv[4 + o][:, None] * rs[1]
            + cv[8 + o][:, None] * rs[2]
            + cv[12 + o][:, None] * rs[3]
        ).to(torch.bfloat16)
        for o in range(4)
    ]
    acc = hl.dot(fw[0], vb[0].T)  # [32, 8] fp32
    acc = hl.dot(fw[1], vb[1].T, acc=acc)
    acc = hl.dot(fw[2], vb[2].T, acc=acc)
    acc = hl.dot(fw[3], vb[3].T, acc=acc)
    f = [v.to(torch.float32) for v in vb]
    r_new[tt, 0, tc] = vb[0]
    r_new[tt, 1, tc] = vb[1]
    r_new[tt, 2, tc] = vb[2]
    r_new[tt, 3, tc] = vb[3]
    mix_part[tc.id, tt, :] = acc.T
    mix_t[tt, :, tc.id] = acc.T
    s00 = torch.sum(f[0] * f[0], dim=-1)
    s01 = torch.sum(f[0] * f[1], dim=-1)
    s02 = torch.sum(f[0] * f[2], dim=-1)
    s03 = torch.sum(f[0] * f[3], dim=-1)
    s11 = torch.sum(f[1] * f[1], dim=-1)
    s12 = torch.sum(f[1] * f[2], dim=-1)
    s13 = torch.sum(f[1] * f[3], dim=-1)
    s22 = torch.sum(f[2] * f[2], dim=-1)
    s23 = torch.sum(f[2] * f[3], dim=-1)
    s33 = torch.sum(f[3] * f[3], dim=-1)
    gram = torch.stack(
        [
            torch.stack([s00, s01, s02, s03], dim=-1),
            torch.stack([s01, s11, s12, s13], dim=-1),
            torch.stack([s02, s12, s22, s23], dim=-1),
            torch.stack([s03, s13, s23, s33], dim=-1),
        ],
        dim=-2,
    )
    sq_part[tt, tc.id] = s00 + s11 + s22 + s33
    gram_part[tt, :, :, tc.id] = gram


def finalize(
    tb,
    tn,
    hidden,
    r_new,
    norm_w,
    mix_part,
    gram_part,
    sq_part,
    scale,
    base_pre,
    li,
    *,
    mix_t,
):
    """Root body for the tile (tokens ``tb``, hidden chunk ``tn``): the next
    sublayer's input li = bf16(bf16(y) * rsqrt(sum(y^2) / H + eps) * w), y = sum_i
    pre_i R'_i, sum(y^2) = pre^T G pre (per-entry [8] vectors: no register selects).
    The 15 split sums run along the contiguous split axis of token-major [8, S]
    tiles (``mix_t``, ``gram_part``, ``sq_part``), warp-local reductions (split-major
    [S, 8] tiles reduce across warps through shared memory: 4.5 vs 2.0 us a task,
    GB300)."""
    # every load first: a reduction's CTA barrier would hold the loads after it
    ik10 = (0, 1, 2, 3, 5, 6, 7, 10, 11, 15)
    sq_l = sq_part[tb, :]
    g_l = [gram_part[tb, ik // 4, ik % 4, :] for ik in ik10]
    p_l = [mix_t[tb, i, :] for i in range(4)]
    red = 1
    rr = [r_new[tb, i, tn] for i in range(4)]
    wn = norm_w[tn].to(torch.float32)
    s0 = hl.load(scale, [0])
    bp = [hl.load(base_pre, [i]) for i in range(4)]
    sq = torch.sum(sq_l, dim=red)  # [8]: sum(R'^2), the Gram trace
    g = [torch.sum(x, dim=red) for x in g_l]
    praw = [torch.sum(x, dim=red) for x in p_l]
    rr = [x.to(torch.float32) for x in rr]
    rs = torch.rsqrt(sq / (4 * hidden) + RMS_EPS)
    p = [torch.sigmoid(praw[i] * rs * s0 + bp[i]) + HC_EPS for i in range(4)]
    # g: (0,0) (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (2,2) (2,3) (3,3)
    var = (
        p[0] * p[0] * g[0]
        + p[1] * p[1] * g[4]
        + p[2] * p[2] * g[7]
        + p[3] * p[3] * g[9]
        + 2.0
        * (
            p[0] * (p[1] * g[1] + p[2] * g[2] + p[3] * g[3])
            + p[1] * (p[2] * g[5] + p[3] * g[6])
            + p[2] * p[3] * g[8]
        )
    ) / hidden
    y = (
        p[0][:, None] * rr[0]
        + p[1][:, None] * rr[1]
        + p[2][:, None] * rr[2]
        + p[3][:, None] * rr[3]
    )
    yb = y.to(torch.bfloat16).to(torch.float32)
    li[tb, tn] = (yb * torch.rsqrt(var + NORM_EPS)[:, None] * wn).to(torch.bfloat16)


def comb_logits(
    ts,
    hidden,
    mix_part,
    sq_part,
    scale,
    base_post,
    base_comb,
    post_out,
):
    """Root body (first part) for the tokens ``ts``: stores post' = 2 sigmoid(mix_post s1 +
    b) of the next boundary and returns comb' as its logits, an [8, 4, 4] tile (mix_comb
    s2 + b, before the row softmax). A boundary may hand these on as its *raw* comb':
    the next boundary's megakernel makes them final (``sinkhorn_init``, then
    ``sinkhorn_step`` ITERS - 1 times) off its critical path (module docstring); the
    stored FP32 logits give the same bits as the fused form."""
    sp = hl.arange(mix_part.size(0))
    i4 = hl.arange(4)
    sq_l = sq_part[ts, :]
    q_l = hl.load(
        mix_part, [sp[:, None, None], ts.index[None, :, None], 4 + i4[None, None, :]]
    )
    c_l = hl.load(
        mix_part,
        [
            sp[:, None, None, None],
            ts.index[None, :, None, None],
            8 + 4 * i4[None, None, :, None] + i4[None, None, None, :],
        ],
    )
    bq = hl.load(base_post, [hl.arange(4)])
    bc = hl.load(base_comb, [hl.arange(4)[:, None], hl.arange(4)[None, :]])
    s1 = hl.load(scale, [1])
    s2 = hl.load(scale, [2])
    sq = torch.sum(sq_l, dim=1)
    qraw = torch.sum(q_l, dim=0)  # [8, 4]
    craw = torch.sum(c_l, dim=0)  # [8, 4, 4]
    rs = torch.rsqrt(sq / (4 * hidden) + RMS_EPS)
    post_out[ts, :] = POST_MULT * torch.sigmoid(qraw * rs[:, None] * s1 + bq)
    return craw * rs[:, None, None] * s2 + bc


def sinkhorn_init(c):
    """comb' logits [8, 4, 4] -> the row softmax (+ eps) and the first column
    normalization (Sinkhorn's first iteration)."""
    mx = torch.amax(c, dim=-1, keepdim=True)
    e = torch.exp(c - mx)
    c = e / torch.sum(e, dim=-1, keepdim=True) + HC_EPS
    return c / (torch.sum(c, dim=-2, keepdim=True) + HC_EPS)


def sinkhorn_step(c):
    """One Sinkhorn (row, column) normalization of the [8, 4, 4] tile ``c``."""
    c = c / (torch.sum(c, dim=-1, keepdim=True) + HC_EPS)
    return c / (torch.sum(c, dim=-2, keepdim=True) + HC_EPS)


# -------------------------------------------------------------------- host side

CHUNK = 128  # ``chunk``'s hidden columns per task (also set in the kernels' bodies)


@functools.cache
def scratch(device: torch.device, hidden: int) -> dict[str, torch.Tensor]:
    """The tails' scratch on ``device`` (the split sums), one set for every kernel
    and call."""
    s = hidden // CHUNK
    f32 = torch.float32
    return {
        "mix_part": torch.zeros(s, 8, 32, device=device, dtype=f32),
        "mix_t": torch.zeros(8, 32, s, device=device, dtype=f32),
        "gram_t": torch.zeros(8, 4, 4, s, device=device, dtype=f32),
        "sq_t": torch.zeros(8, s, device=device, dtype=f32),
    }


def fn_pad(fn: torch.Tensor) -> torch.Tensor:
    """A layer's hc fn [24, 4H] (BF16, as the checkpoint stores it) as the kernels'
    [32, 4, H] operand: rows padded with zeros to the MMA tile (prepared once)."""
    out = torch.zeros(32, 4, fn.size(1) // 4, device=fn.device, dtype=torch.bfloat16)
    out[:24] = fn.view(24, 4, -1)
    return out


def boundary_operands(t: dict) -> tuple:
    """A boundary's operands for the megakernels with this tail folded in: (r
    [T, 4, H] BF16, post [T, 4] / comb [T, 4, 4] FP32, the next sublayer's
    ``fn_pad``, scale [3], base [24], input norm weight) of a boundary's inputs
    ``t`` (``_setup.boundary_inputs``' names)."""
    keys = ("r", "post", "comb", "fn", "scale", "base", "norm_w")
    return tuple(fn_pad(t[k]) if k == "fn" else t[k] for k in keys)


def tail_args(mhc, device, hidden) -> tuple:
    """The tail's kernel arguments: ``mhc`` (``boundary_operands``) with the base
    split per mix, and the scratch."""
    r, post, comb, fpad, hc_scale, base, norm_w = mhc
    sc = scratch(device, hidden)
    return (
        r,
        post,
        comb,
        fpad,
        hc_scale,
        base[0:4],
        base[4:8],
        base[8:24].view(4, 4),
        norm_w,
        *sc.values(),
    )


def new_outputs(r: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """A boundary's outputs for the residual streams ``r`` [T, 4, H]: (li [T, H]
    BF16, r_new [T, 4, H] BF16, post_out [T, 4] FP32, comb_out [T, 4, 4] FP32)."""
    tokens, hc, hidden = r.shape
    f32 = torch.float32
    return (
        r.new_empty(tokens, hidden),
        torch.empty_like(r),
        r.new_empty(tokens, hc, dtype=f32),
        r.new_empty(tokens, hc, hc, dtype=f32),
    )
