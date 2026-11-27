# ruff: noqa: ANN001, ANN002, ANN202
# pyrefly: ignore-errors
"""Shared pieces of the GPT-OSS-120B decoder-layer megakernels.

In-kernel helpers (MXFP4 activation quantization, RoPE), synthetic layer
weights in the layout SGLang serves the checkpoint in, and an FP32 reference
of one decoder layer over ``T`` consecutive tokens.

Weight layout (SGLang main, ``moe_runner_backend=flashinfer_mxfp4``, after
``process_weights_after_loading``):

* QKV: one fused ``[5120, 2880]`` bf16 weight.  The megakernels take it with
  the input RMSNorm weight folded in and re-tiled so each (row tile, K split)
  task's three ``[128, 64]`` K steps are one contiguous 48 KB block.
* Experts: FlashInfer TRT-LLM MXFP4.  Hidden padded 2880 -> 3072, intermediate
  2880 -> 2944; gate/up row pairs swapped; rows shuffled within blocks of 32
  for TRT-LLM's epilogue; E8M0 scales in the 128x4 swizzle (each 128-row x
  4-column tile one 512-byte block, which is exactly the operand layout
  ``tcgen05.cp`` loads).
"""

from __future__ import annotations

import dataclasses
import math

from pretuned_kernels.megakernels.gpt_oss_moe._standalone import _dequant_e2m1
from pretuned_kernels.megakernels.gpt_oss_moe._standalone import _to_production_w2
from pretuned_kernels.megakernels.gpt_oss_moe._standalone import _to_production_w13
import torch

import helion.language as hl

HIDDEN = 2880
HIDDEN_PAD = 3072
INTER = 2880
INTER_PAD = 2944
EXPERTS = 128
TOP_K = 4
Q_HEADS = 64
KV_HEADS = 8
HEAD_DIM = 64
QKV_ROWS = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM  # 5120
EPS = 1e-5
INV_HIDDEN = (
    1.0 / HIDDEN
)  # the RMSNorm mean (a float: a module int would trace as a SymInt)
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
SLIDING_WINDOW = 127  # even layers; odd layers attend to the full context
PAGE_SIZE = 64
POOL_PAGES = 96
MAX_POSITION = 8192

# Loads that stream through the gluon TMA ring (projection and expert operands,
# their scales and biases); every other memory access is a pointer access.
RING_LOADS = frozenset(
    [
        "qkv_w",
        "bq_in",
        "o_w",
        "attn_b",
        "w13",
        "w13_sp",
        "y_q",
        "y_sp",
        "w13_b",
        "w2",
        "w2_sp",
        "a_q",
        "a_s",
        "w2_b",
    ]
)


# ---------------------------------------------------------------------------
# In-kernel helpers
# ---------------------------------------------------------------------------


def pack_nibbles(code, lead: int):
    """[lead, 32] e2m1 codes -> [lead, 16] bytes (element 2k in the low nibble)."""
    lo, hi = hl.split(code.reshape(lead, 16, 2))
    return (lo + hi * 16).to(torch.uint8)


def e2m1_term(r, e):
    """One MXFP4 term of ``r`` [..., 32] at block exponents ``e`` [...]: the
    e2m1 codes [..., 32] and the residual."""
    v = r * torch.exp2(-e)[..., None]
    a = torch.abs(v)
    m = (
        (a >= 0.25).to(torch.int32)
        + (a >= 0.75).to(torch.int32)
        + (a >= 1.25).to(torch.int32)
        + (a >= 1.75).to(torch.int32)
        + (a >= 2.5).to(torch.int32)
        + (a >= 3.5).to(torch.int32)
        + (a >= 5.0).to(torch.int32)
    )
    mag = torch.where(
        m < 5,
        m.to(torch.float32) * 0.5,
        torch.where(m == 5, 3.0, torch.where(m == 6, 4.0, 6.0)),
    )
    neg = v < 0
    deq = torch.where(neg, -mag, mag) * torch.exp2(e)[..., None]
    return m + neg.to(torch.int32) * 8, r - deq


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """GPT-OSS's expert activation: SwiGLU with gate capped at SWIGLU_LIMIT and
    up clamped to +-SWIGLU_LIMIT (the kernels' and the reference's one copy; in
    a kernel, called as a helper, its constants trace as literals)."""
    gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    up = torch.clamp(up, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (up + 1.0) * gate * torch.sigmoid(SWIGLU_ALPHA * gate)


def scale_tiles(chunk: torch.Tensor) -> torch.Tensor:
    """``n`` packed (tcgen05.cp) scale tiles of 32 x 4 x 4 bytes -> the
    [128, 4 * n] scale operand of a 128-row ``hl.dot_scaled`` K step of
    64 * n bytes."""
    n = chunk.size(0)
    return chunk.reshape(n, 32, 4, 4).permute(2, 1, 0, 3).reshape(128, 4 * n)


def e2m1_exponent(r):
    """The first term's block exponents of ``r`` [..., 32].  After a term at
    exponent e the residual is at most 2**e (the widest e2m1 gap is 4..6), so
    term t takes e - 2t (6 * 2**(e - 2) >= 2**e) and one block max serves all
    four terms."""
    amax = torch.amax(torch.abs(r), dim=-1)
    return torch.clamp(torch.ceil(torch.log2(amax * (1.0 / 6.0))), -120.0, 126.0)


def rope(v, cs, heads: int):
    """Neox RoPE of ``v`` [heads * 64] with ``cs`` = cos | sin [64] -> [heads, 64]."""
    x1, x2 = hl.split(v.reshape(heads, 2, 32).permute(0, 2, 1))
    cos, sin = hl.split(cs.reshape(2, 32).permute(1, 0))
    o1 = x1 * cos[None, :] - x2 * sin[None, :]
    o2 = x2 * cos[None, :] + x1 * sin[None, :]
    return hl.join(o1, o2).permute(0, 2, 1).reshape(heads, 64)


# ---------------------------------------------------------------------------
# Synthetic layer (checkpoint-free) in SGLang's served layout
# ---------------------------------------------------------------------------


def _served_experts(blocks, scales, bias, rows_pad, k_pad, *, swap_pairs):
    """Pad, then lay out as TRT-LLM serves them (gate/up rows also pair-swapped)."""
    experts, rows, k_bytes = blocks.shape
    w = torch.zeros(
        experts, rows_pad, k_pad // 2, dtype=torch.uint8, device=blocks.device
    )
    w[:, :rows, :k_bytes] = blocks
    s = torch.zeros(
        experts, rows_pad, k_pad // 32, dtype=torch.uint8, device=blocks.device
    )
    s[:, :rows, : scales.size(-1)] = scales
    b = torch.zeros(experts, rows_pad, dtype=torch.float32, device=blocks.device)
    b[:, :rows] = bias
    return (_to_production_w13 if swap_pairs else _to_production_w2)(w, s, b)


def rope_cos_sin(device: torch.device, theta: float = 150000.0) -> torch.Tensor:
    """[MAX_POSITION, 64] fp32 cos | sin table (plain RoPE; the kernels only
    read the table, so its frequencies do not matter for the check)."""
    inv_freq = theta ** (-torch.arange(0, HEAD_DIM, 2, device=device) / HEAD_DIM)
    angles = torch.arange(MAX_POSITION, device=device)[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1).float()


def make_layer(seed: int, device: torch.device, init: bool = True) -> tuple[dict, dict]:
    """Random GPT-OSS-120B layer weights: the logical tensors (for the
    reference) and the megakernels' operands.  With ``init=False`` both are
    uninitialized, laid out by tracing this on the meta device (binding needs
    only the operands' shapes, strides and dtypes)."""
    if not init:
        return tuple(
            {k: torch.empty_like(v, device=device) for k, v in tensors.items()}
            for tensors in make_layer(seed, torch.device("meta"))
        )
    g = torch.Generator("cpu" if device.type == "meta" else device).manual_seed(seed)

    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, device=device) * scale).to(dtype)

    def randint(lo, hi, *shape):
        return torch.randint(
            lo, hi, shape, generator=g, device=device, dtype=torch.uint8
        )

    raw = {
        "ln1": randn(HIDDEN, scale=0.1) + 1.0,
        "qkv_w": randn(QKV_ROWS, HIDDEN, scale=0.02),
        "qkv_b": randn(QKV_ROWS, scale=0.1),
        "o_w": randn(HIDDEN, Q_HEADS * HEAD_DIM, scale=0.02),
        "o_b": randn(HIDDEN, scale=0.1),
        "sinks": randn(Q_HEADS, dtype=torch.float32),
        "ln2": randn(HIDDEN, scale=0.1) + 1.0,
        "router_w": randn(EXPERTS, HIDDEN, scale=0.02),
        "router_b": randn(EXPERTS, scale=0.1),
        # gate/up rows interleaved (gate 2i, up 2i + 1), as in the checkpoint
        "gu_blocks": randint(0, 256, EXPERTS, 2 * INTER, HIDDEN // 2),
        "gu_scales": randint(119, 122, EXPERTS, 2 * INTER, HIDDEN // 32),
        "gu_bias": randn(EXPERTS, 2 * INTER, scale=0.1, dtype=torch.float32),
        "dn_blocks": randint(0, 256, EXPERTS, HIDDEN, INTER // 2),
        "dn_scales": randint(119, 122, EXPERTS, HIDDEN, INTER // 32),
        "dn_bias": randn(EXPERTS, HIDDEN, scale=0.1, dtype=torch.float32),
    }
    w13, w13_s, w13_b = _served_experts(
        raw["gu_blocks"],
        raw["gu_scales"],
        raw["gu_bias"],
        2 * INTER_PAD,
        HIDDEN_PAD,
        swap_pairs=True,
    )
    w2, w2_s, w2_b = _served_experts(
        raw["dn_blocks"],
        raw["dn_scales"],
        raw["dn_bias"],
        HIDDEN_PAD,
        INTER_PAD,
        swap_pairs=False,
    )
    qkv_ln1 = (raw["qkv_w"].float() * raw["ln1"].float()[None, :]).to(torch.bfloat16)
    served = {
        # (row tile, K split, K step) [128, 64] blocks, contiguous per task
        "qkv_w": qkv_ln1.view(40, 128, 15, 3, 64)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(600 * 3 * 128, 64),
        "qkv_b": raw["qkv_b"],
        "sinks": raw["sinks"],
        "o_w": raw["o_w"].view(HIDDEN, 16, 256),
        "o_b": raw["o_b"],
        "ln2": raw["ln2"],
        "router_w": raw["router_w"],
        "router_b": raw["router_b"],
        "w13": w13.view(EXPERTS * 2 * INTER_PAD, HIDDEN_PAD // 2),
        "w13_sp": w13_s.view(EXPERTS * 46, 24, 32, 16),
        "w13_b": w13_b,
        "w2": w2.view(EXPERTS * HIDDEN_PAD, INTER_PAD // 2),
        "w2_sp": w2_s.view(EXPERTS * 24, 23, 32, 16),
        "w2_b": w2_b,
    }
    return raw, served


@dataclasses.dataclass
class Context:
    """A paged bf16 KV pool ([slots, 8, 64]) holding ``pos`` cached tokens and
    the slots of the ``tokens`` new ones."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor  # [4, 2048] int32, row 0 is the request
    k_hist: torch.Tensor  # [pos, 8, 64] the cached keys in position order
    v_hist: torch.Tensor
    positions: torch.Tensor  # [tokens] int64
    out_loc: torch.Tensor  # [tokens] int64
    window: torch.Tensor  # [1] int32: SLIDING_WINDOW or -1
    pos: int
    win: int


def make_context(pos: int, win: int, tokens: int, seed: int, device) -> Context:
    # The page table maps logical pages 0 .. POOL_PAGES - 2 (the rest stay on
    # physical page 0): positions past that would alias page 0.
    assert pos + tokens <= (POOL_PAGES - 1) * PAGE_SIZE, (pos, tokens)
    g = torch.Generator(device=device).manual_seed(seed)
    slots = POOL_PAGES * PAGE_SIZE
    k_cache = torch.randn(slots, KV_HEADS, HEAD_DIM, generator=g, device=device)
    v_cache = torch.randn(slots, KV_HEADS, HEAD_DIM, generator=g, device=device)
    page_table = torch.zeros(4, 2048, dtype=torch.int32, device=device)
    page_table[0, : POOL_PAGES - 1] = (
        torch.randperm(POOL_PAGES - 1, generator=g, device=device) + 1
    ).to(torch.int32)
    order = torch.arange(pos + tokens, device=device)
    loc = page_table[0, order // PAGE_SIZE].long() * PAGE_SIZE + order % PAGE_SIZE
    k_cache, v_cache = k_cache.to(torch.bfloat16), v_cache.to(torch.bfloat16)
    return Context(
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        k_hist=k_cache[loc[:pos]].clone(),
        v_hist=v_cache[loc[:pos]].clone(),
        positions=torch.arange(pos, pos + tokens, dtype=torch.int64, device=device),
        out_loc=loc[pos:].clone(),
        window=torch.tensor([win], dtype=torch.int32, device=device),
        pos=pos,
        win=win,
    )


# ---------------------------------------------------------------------------
# FP32 reference
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * w.float()


def reference_attention(
    raw: dict, ctx: Context, cos_sin: torch.Tensor, x: torch.Tensor
):
    """The attention half of one decoder layer over the ``T`` new tokens (a
    causal chain after the cached ones); ``x`` [T, 2880] is the input
    residual.  Returns the post-attention residual [T, 2880] and the new
    tokens' K and V [T, 8, 64]."""
    tokens = x.size(0)
    qkv = _rms_norm(x, raw["ln1"]) @ raw["qkv_w"].float().T + raw["qkv_b"].float()
    q = qkv[:, :4096].view(tokens, Q_HEADS, HEAD_DIM)
    k = qkv[:, 4096:4608].view(tokens, KV_HEADS, HEAD_DIM)
    v = qkv[:, 4608:].view(tokens, KV_HEADS, HEAD_DIM)
    cs = cos_sin[ctx.pos : ctx.pos + tokens]
    cos, sin = cs[:, None, :32], cs[:, None, 32:]

    def rot(t):
        t1, t2 = t[..., :32], t[..., 32:]
        return torch.cat([t1 * cos - t2 * sin, t2 * cos + t1 * sin], -1)

    q, k = rot(q), rot(k)
    group = torch.arange(Q_HEADS, device=x.device) // (Q_HEADS // KV_HEADS)
    h = []
    for t in range(tokens):
        lo = max(0, ctx.pos + t - ctx.win) if ctx.win >= 0 else 0
        keys = torch.cat([ctx.k_hist[lo:].float(), k[: t + 1].bfloat16().float()])
        values = torch.cat([ctx.v_hist[lo:].float(), v[: t + 1].bfloat16().float()])
        scores = torch.einsum("hd,nhd->hn", q[t], keys[:, group]) / math.sqrt(HEAD_DIM)
        probs = torch.softmax(torch.cat([scores, raw["sinks"][:, None]], 1), 1)[:, :-1]
        o = torch.einsum("hn,nhd->hd", probs, values[:, group]).reshape(-1)
        h.append(x[t] + raw["o_w"].float() @ o + raw["o_b"].float())
    return torch.stack(h), k, v


def reference_moe(raw: dict, h: torch.Tensor):
    """The MoE half: the expert output [T, 2880] of the post-attention
    residual ``h`` [T, 2880] and the routed experts [T, 4]."""
    y = _rms_norm(h.float(), raw["ln2"])
    logits = y @ raw["router_w"].float().T + raw["router_b"].float()
    top, ids = torch.topk(logits, TOP_K, dim=-1)
    weights = torch.softmax(top, dim=-1)
    moe = torch.zeros_like(y)
    for e in ids.unique().tolist():
        w13 = _dequant_e2m1(raw["gu_blocks"][e], raw["gu_scales"][e])
        w2 = _dequant_e2m1(raw["dn_blocks"][e], raw["dn_scales"][e])
        for t, rank in (ids == e).nonzero().tolist():
            pre = w13 @ y[t] + raw["gu_bias"][e]
            act = clamped_swiglu(pre[0::2], pre[1::2])
            moe[t] += weights[t, rank] * (w2 @ act + raw["dn_bias"][e])
    return moe, ids


def rel_max_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (actual.float() - expected.float()).abs().max() / expected.float().abs().max()
    )


def check(name: str, actual, expected, tol: float) -> float:
    err = rel_max_err(actual, expected)
    if not err <= tol:
        raise AssertionError(f"{name}: relative max error {err:.3e} > {tol:.0e}")
    return err
