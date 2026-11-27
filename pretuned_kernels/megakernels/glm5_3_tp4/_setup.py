# ruff: noqa: ANN001, ANN002, ANN202, C408
"""Synthetic GLM-5.3 TP4 weights and inputs and the torch references of the
megakernels' checks: the MoE (``moe_*``, ``window_router``), the MLA attention
front and output side (``front_*``, ``out_*``), the DSA indexer (``indexer_*``)
and the MTP draft's input (``eh_*``) and LM head (``lm_head_inputs``,
``check_greedy``, ``stock_lm_head``). GLM-5.3 Flash's checks
(``glm5_3_flash_tp4/_setup.py``) reuse them for the layouts the two models
share, at Flash's shapes (``MoeShape``, ``AttnShape``).

Layouts are SGLang's (main, trtllm DSA decode backend, ``flashinfer_trtllm``
MoE runner):

- routed experts in FlashInfer's trtllm-gen NVFP4 layout; the routed layout, the
  E2M1 decode and the NVFP4 rounding are Qwen3.5's (``qwen3_5_moe_tp4/_setup.py``);
  the shared expert and the router in BF16;
- the MLA cache: one unscaled E4M3 row of 576 per slot (the 512 normed latent,
  then the 64 RoPE'd key dims); the sparse attention reads, per query row, the
  top-k slots the indexer picked (-1: none);
- the index-K cache: one row of 132 * 64 bytes per page of 64 keys, the 64 E4M3
  keys of 128 dims first, then their 64 FP32 scales. A verify window of T rows:
  row i sees the request's first ``lengths[i]`` keys (the context and the window
  up to itself), its keys at ``page_table[p // 64] * 64 + p % 64``.
"""

from __future__ import annotations

import dataclasses

from pretuned_kernels.megakernels.qwen3_5_moe_tp4._setup import _nvfp4_quant
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._setup import _row_maps
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._setup import e2m1_decode
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._setup import swizzle_scales
import torch

HIDDEN = 6144
VOCAB = 154880
# rms_norm_eps.
NORM_EPS = 1e-5
# GLM-5.3's default RoPE: interleaved pairs, theta 8e6, no mscale.
ROPE_THETA = 8.0e6


def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """SGLang's RMSNorm: FP32, rounded once to BF16."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + NORM_EPS)
    return (out * w.float()).to(torch.bfloat16)


def cos_sin_cache(max_position: int, rope: int = 64, device="cuda") -> torch.Tensor:
    """SGLang's rotary cache [max_position, rope] FP32: cos of the rope / 2
    frequencies, then their sin."""
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, rope, 2, device=device).float() / rope))
    freqs = torch.arange(max_position, device=device).float()[:, None] * inv[None, :]
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Pairs (2i, 2i + 1) rotated by frequency i (SGLang's non-NeoX RoPE), FP32."""
    x = x.float()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x0 * cos - x1 * sin, x1 * cos + x0 * sin), dim=-1)
    return out.flatten(-2)


# -------------------------------------------------------------------------- MoE


@dataclasses.dataclass(frozen=True)
class MoeShape:
    hidden: int = HIDDEN
    intermediate: int = 512  # per rank (TP4 of 2048)
    num_experts: int = 256
    top_k: int = 8
    shared_intermediate: int = 512  # per rank
    routed_scale: float = 2.5


MOE_SHAPE = MoeShape()


def nvfp4_bytes(g: torch.Generator, *size: int) -> torch.Tensor:
    """Random packed NVFP4 values (two E2M1 codes a byte)."""
    return torch.randint(0, 256, size, generator=g, device=g.device, dtype=torch.uint8)


def nvfp4_scales(g: torch.Generator, *size: int) -> torch.Tensor:
    """Random E4M3 block scales in [1/8, 1/2)."""
    v = torch.exp2(torch.randint(-3, 0, size, generator=g, device=g.device).float())
    v = v * (1 + torch.randint(0, 8, size, generator=g, device=g.device).float() / 8)
    return v.to(torch.float8_e4m3fn)


def trtllm_gen_w13(w13_log: torch.Tensor, w13s_log: torch.Tensor):
    """Logical ModelOpt W13 [E, 2N, K / 2] (rows [gate; up]) and its block scales
    [E, 2N, K / 16] -> FlashInfer's trtllm-gen layout (rows [up; gate], the
    gated-act interleave and the epilogue-tile shuffle; the scales likewise, then
    128x4-interleaved), shaped like the input."""
    n = w13_log.shape[1] // 2
    phys13, _ = _row_maps(w13_log.device, n, 0)
    upgate = torch.cat((w13_log[:, n:], w13_log[:, :n]), dim=1)
    upgate_s = torch.cat((w13s_log[:, n:], w13s_log[:, :n]), dim=1)
    return upgate[:, phys13].contiguous(), swizzle_scales(
        upgate_s[:, phys13].contiguous()
    )


def trtllm_gen_w2(w2_log: torch.Tensor, w2s_log: torch.Tensor):
    """Logical W2 [E, H, N / 2] and its scales -> the trtllm-gen layout (the
    epilogue-tile row shuffle, the scales 128x4-interleaved)."""
    _, phys2 = _row_maps(w2_log.device, 0, w2_log.shape[1])
    return w2_log[:, phys2].contiguous(), swizzle_scales(w2s_log[:, phys2].contiguous())


def nvfp4_experts(g: torch.Generator, experts: int, intermediate: int, hidden: int):
    """Random NVFP4 experts: the logical ModelOpt copies (``*_log``, W13 rows
    [gate; up]) and their trtllm-gen layout (``w13``, ``w13s``, ``w2``, ``w2s``)."""
    E, N, H = experts, intermediate, hidden
    w = dict(
        w13_log=nvfp4_bytes(g, E, 2 * N, H // 2),
        w13s_log=nvfp4_scales(g, E, 2 * N, H // 16),
        w2_log=nvfp4_bytes(g, E, H, N // 2),
        w2s_log=nvfp4_scales(g, E, H, N // 16),
    )
    w["w13"], w["w13s"] = trtllm_gen_w13(w["w13_log"], w["w13s_log"])
    w["w2"], w["w2s"] = trtllm_gen_w2(w["w2_log"], w["w2s_log"])
    return w


def nvfp4_dequant(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[..., K / 2] packed values and [..., K / 16] block scales -> [..., K] FP32."""
    return e2m1_decode(packed) * scales.float().repeat_interleave(16, -1)


def moe_weights(shape: MoeShape = MOE_SHAPE, seed: int = 0, device="cuda") -> dict:
    """One rank's MoE weights: the routed experts (``nvfp4_experts``), their
    alphas and global scales, the router and its correction bias, and the BF16
    shared expert."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    E, H, SHI = shape.num_experts, shape.hidden, shape.shared_intermediate
    w = nvfp4_experts(g, E, shape.intermediate, H)
    input_global = torch.tensor([64.0], device=device)
    act_global = torch.tensor([64.0], device=device)
    alpha1 = (torch.rand(E, generator=g, device=device) * 0.5 + 0.75) / (
        input_global.item() * 64
    )
    alpha2 = (torch.rand(E, generator=g, device=device) * 0.5 + 0.75) / (
        act_global.item() * 16
    )
    router = (torch.randn(E, H, generator=g, device=device) * 0.02).to(torch.bfloat16)
    bias = torch.randn(E, generator=g, device=device) * 0.05
    # BF16 shared expert: gate rows [0, SHI), up rows [SHI, 2 SHI) (SGLang's
    # merged gate_up shard), and its down projection.
    sw13 = (torch.randn(2 * SHI, H, generator=g, device=device) * 0.02).to(
        torch.bfloat16
    )
    sw2 = (torch.randn(H, SHI, generator=g, device=device) * 0.02).to(torch.bfloat16)
    return dict(
        w,
        alpha1=alpha1,
        alpha2=alpha2,
        input_global=input_global,
        act_global=act_global,
        router=router,
        bias=bias,
        sw13=sw13,
        sw2=sw2,
    )


def moe_weight_args(w: dict, shape: MoeShape = MOE_SHAPE) -> tuple:
    """The MoE kernel's weight arguments (``router_weight`` .. ``routed_scale``)."""
    E, H, N = shape.num_experts, shape.hidden, shape.intermediate
    return (
        w["router"],
        w["bias"],
        w["w13"],
        w["w13s"].view(E, (2 * N // 128) * (H // 512), 16, 256),
        w["w2"],
        w["w2s"].view(E, H // 128, (N // 512) * 16, 256),
        w["alpha1"],
        w["alpha2"],
        w["input_global"],
        w["act_global"],
        w["sw13"],
        w["sw2"],
        shape.top_k,
        shape.routed_scale,
    )


def moe_benchmark_weights(
    layer: int, rank: int, device, shape: MoeShape = MOE_SHAPE
) -> dict:
    """Layer ``layer``'s weights on ``rank`` without the logical copies; the router
    and its bias seeded by the layer alone (TP replicates them)."""
    w = moe_weights(shape, seed=1000 * layer + rank, device=device)
    for key in ("w13_log", "w13s_log", "w2_log", "w2s_log"):
        del w[key]
    g = torch.Generator(device=device)
    g.manual_seed(1000 * layer + 777)
    w["router"] = (
        torch.randn(shape.num_experts, shape.hidden, generator=g, device=device) * 0.02
    ).to(torch.bfloat16)
    w["bias"] = torch.randn(shape.num_experts, generator=g, device=device) * 0.05
    return w


def window_router(
    x: torch.Tensor, distinct: int, shape: MoeShape = MOE_SHAPE, seed: int = 0
) -> torch.Tensor:
    """A BF16 router whose logits on the window ``x`` [T, H] route it to
    ``distinct`` experts in all (at most top_k * T), like a served window with
    that many: token t picks top_k consecutive experts of a random pool of
    ``distinct``, from t * top_k on (wrapping). The target logits (picks near 3,
    the rest near -3) are solved for exactly through ``x`` (T <= 8 rows of
    6144: x x^T is invertible), so the margins dwarf the BF16 rounding."""
    T, E, K = x.size(0), shape.num_experts, shape.top_k
    distinct = min(distinct, K * T)
    assert distinct >= K, distinct
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    pool = torch.randperm(E, generator=g, device=x.device)[:distinct]
    target = torch.randn(T, E, generator=g, device=x.device, dtype=torch.float64)
    target = target * 0.3 - 3.0
    for t in range(T):
        picks = pool[(t * K + torch.arange(K, device=x.device)) % distinct]
        target[t, picks] += 6.0
    xd = x.double()
    return (target.T @ torch.linalg.solve(xd @ xd.T, xd)).to(torch.bfloat16)


def route_reference(x: torch.Tensor, w: dict, shape: MoeShape = MOE_SHAPE):
    """GLM-5.3's routing (DeepSeek-V3's noaux_tc with one group): FP32 logits,
    sigmoid scores, the top-k of score + bias (higher first, then the lower
    index), the picks' scores renormalized times the routed scale."""
    logits = x.float() @ w["router"].float().T
    scores = torch.sigmoid(logits)
    keyed = scores + w["bias"][None, :]
    _, ids = torch.sort(keyed, dim=-1, descending=True, stable=True)
    ids = ids[:, : shape.top_k]
    picked = torch.gather(scores, 1, ids)
    wts = picked / picked.sum(-1, keepdim=True) * shape.routed_scale
    return logits, ids, wts


def routed_rows(
    x_q, ids, w: dict, alpha1, alpha2, kernel_act: tuple, act_global: float, clamp=None
):
    """Each pick's W2 row [T, K, H] (BF16: the GEMM's output) on the logical
    experts ``w`` (``nvfp4_experts``) with per-expert alphas. Each GEMM reads
    the kernel's copy of its input (``qwen3_5_moe_tp4``'s reference convention):
    ``x_q`` the dequantized NVFP4 input [T, H], ``kernel_act`` = (packed, scales)
    [distinct, T, N / 2 | N / 16] the activations (row j: the window's j-th
    distinct expert, ascending). Also the activations as (the kernel's, their
    exact NVFP4 at ``act_global``) [T * K, N]. ``clamp``: per-expert limits on
    GEMM1's raw accumulators (FlashInfer's gatedSilu: gate <= limit, |up| <=
    limit) before the alpha."""
    act_q, act_scale = kernel_act
    T, K = ids.shape
    N = w["w13_log"].shape[1] // 2
    distinct = sorted({int(e) for e in ids.flatten()})
    y = torch.empty(T, K, w["w2_log"].shape[1], device=x_q.device, dtype=torch.bfloat16)
    acts, exacts = [], []
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            h = nvfp4_dequant(w["w13_log"][e], w["w13s_log"][e]) @ x_q[t]
            if clamp is not None:
                h = torch.cat(
                    (torch.minimum(h[:N], clamp[e]), h[N:].clamp(-clamp[e], clamp[e]))
                )
            gate, up = (h * alpha1[e]).split(N)
            j = distinct.index(e)
            aq = nvfp4_dequant(act_q[j, t], act_scale[j, t])
            acts.append(aq)
            exacts.append(_nvfp4_quant(gate * torch.sigmoid(gate) * up, act_global, aq))
            w2 = nvfp4_dequant(w["w2_log"][e], w["w2s_log"][e])
            y[t, k] = ((w2 @ aq) * alpha2[e]).to(torch.bfloat16)
    return y, (torch.stack(acts), torch.stack(exacts))


def moe_reference(x: torch.Tensor, w: dict, kernel_quant: tuple, shape=MOE_SHAPE):
    """This rank's MoE output (BF16 [T, H]) and top-k ids on the logical weights,
    and the kernel's NVFP4 input and activations with their exact quantizations
    (``routed_rows``): the picks' rows weighted by BF16 weights in pick order,
    plus the BF16 shared expert."""
    xf = x.float()
    _, ids, wts = route_reference(x, w, shape)
    input_q, input_scale, act_q, act_scale = kernel_quant
    xq = nvfp4_dequant(input_q, input_scale)
    y, acts = routed_rows(
        xq, ids, w, w["alpha1"], w["alpha2"], (act_q, act_scale), w["act_global"].item()
    )
    wb = wts.to(torch.bfloat16).float()
    routed = sum(wb[:, k, None] * y[:, k].float() for k in range(shape.top_k))
    SHI = shape.shared_intermediate
    gu = (xf @ w["sw13"].float().T).to(torch.bfloat16).float()
    g, u = gu[:, :SHI], gu[:, SHI:]
    a = (g * torch.sigmoid(g) * u).to(torch.bfloat16).float()
    sh = (a @ w["sw2"].float().T).to(torch.bfloat16).float()
    quant = {
        "input (NVFP4)": (xq, _nvfp4_quant(xf, w["input_global"].item(), xq)),
        "activation (NVFP4)": acts,
    }
    return (routed + sh).to(torch.bfloat16), ids.int(), quant


# ---------------------------------------------------------------- MLA attention


@dataclasses.dataclass(frozen=True)
class AttnShape:
    hidden: int = HIDDEN
    heads: int = 16  # per rank (TP4 of 64)
    kv_lora: int = 512
    rope: int = 64
    nope: int = 192
    v_head: int = 256
    q_lora: int = 2048
    top_k: int = 2048
    page: int = 64
    # Tokens a pooled index key covers (GLM-5.3 Flash's DSA: 4); a row's top-k
    # list holds top_k / kpool closed pools, then its open pool's tokens.
    kpool: int = 1


ATTN_SHAPE = AttnShape()
# 1 / sqrt(qk_head_dim).
SOFTMAX_SCALE = (ATTN_SHAPE.nope + ATTN_SHAPE.rope) ** -0.5


def front_weights(shape: AttnShape = ATTN_SHAPE, seed: int = 0, device="cuda") -> dict:
    """The attention front of one rank: the fused q_a | kv_a projection
    [2048 + 512 + 64, hidden] (replicated), the q_a / kv_a RMSNorm weights, q_b's
    [heads * 256, 2048] shard and W_UK per head as [head, latent, nope] (kv_b_proj's
    key rows, transposed: K contiguous)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    H, L, R, N = shape.heads, shape.kv_lora, shape.rope, shape.nope

    def rn(*size, std):
        return (torch.randn(*size, generator=g, device=device) * std).to(torch.bfloat16)

    return dict(
        qkv_a=rn(shape.q_lora + L + R, shape.hidden, std=shape.hidden**-0.5),
        q_a_norm=(1 + 0.1 * torch.randn(shape.q_lora, generator=g, device=device)).to(
            torch.bfloat16
        ),
        kv_a_norm=(1 + 0.1 * torch.randn(L, generator=g, device=device)).to(
            torch.bfloat16
        ),
        q_b=rn(H * (N + R), shape.q_lora, std=shape.q_lora**-0.5),
        w_uk=rn(H, L, N, std=N**-0.5),
    )


def front_reference(x, w: dict, positions, cs, shape: AttnShape = ATTN_SHAPE):
    """SGLang's MLA prologue on one rank: q_lora (the normed q_a, BF16, the
    indexer's input), the FP8 queries [T, heads, kv_lora + rope] (absorbed nope |
    RoPE'd pe) and the window's FP8 MLA cache rows [T, kv_lora + rope] (normed
    latent | RoPE'd k_pe; ``positions`` / ``cs`` unused without RoPE). Rounding
    points: each GEMM's BF16 output, the norms' BF16 outputs."""
    H, L, R, N = shape.heads, shape.kv_lora, shape.rope, shape.nope
    a = (x.float() @ w["qkv_a"].float().T).to(torch.bfloat16)
    q_a = a[:, : shape.q_lora]
    kv_a = a[:, shape.q_lora : shape.q_lora + L]
    k_pe = a[:, shape.q_lora + L :]
    q_lora = rms(q_a, w["q_a_norm"])
    kv_n = rms(kv_a, w["kv_a_norm"])
    q = (q_lora.float() @ w["q_b"].float().T).to(torch.bfloat16).view(-1, H, N + R)
    q_nope, q_pe = q[..., :N], q[..., N:]
    q_abs = torch.einsum("thd,hld->thl", q_nope.float(), w["w_uk"].float()).to(
        torch.bfloat16
    )
    q8, kv8 = q_abs.float(), kv_n.float()
    if R:
        cos = cs[positions, : R // 2]
        sin = cs[positions, R // 2 :]
        q_rot = _rope_interleaved(q_pe, cos[:, None, :], sin[:, None, :])
        q8 = torch.cat((q8, q_rot), dim=-1)
        kv8 = torch.cat((kv8, _rope_interleaved(k_pe, cos, sin)), dim=-1)
    return q_lora, q8.to(torch.float8_e4m3fn), kv8.to(torch.float8_e4m3fn)


def out_weights(shape: AttnShape = ATTN_SHAPE, seed: int = 0, device="cuda") -> dict:
    """The attention output side of one rank: W_UV per head as [head, v, latent]
    (kv_b_proj's value rows, K contiguous), o_proj's [hidden, heads * v] shard."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    H, V, L = shape.heads, shape.v_head, shape.kv_lora
    w_uv = (torch.randn(H, V, L, generator=g, device=device) * L**-0.5).to(
        torch.bfloat16
    )
    o_proj = (
        torch.randn(shape.hidden, H * V, generator=g, device=device) * (H * V) ** -0.5
    ).to(torch.bfloat16)
    return dict(w_uv=w_uv, o_proj=o_proj)


def make_cache(
    tokens: int,
    context: int,
    shape: AttnShape = ATTN_SHAPE,
    seed: int = 1,
    device="cuda",
) -> dict:
    """An MLA cache of ``context`` committed slots scattered over a pool (pages
    shuffled), FP8 query rows of the window and each row's top-k slots (width
    top_k + kpool - 1): the closed pools of kpool slots below the row's causal
    limit, all when they fit top_k, else a random top_k / kpool of them, then
    the open pool's slots; -1 past the picks."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    D = shape.kv_lora + shape.rope
    pages = (context + tokens + shape.page - 1) // shape.page
    pool_pages = pages * 2 + 4
    perm = torch.randperm(pool_pages, generator=g, device=device)[:pages]
    slots = (
        perm[:, None] * shape.page + torch.arange(shape.page, device=device)
    ).reshape(-1)
    kv = (torch.randn(pool_pages * shape.page, D, generator=g, device=device)).to(
        torch.float8_e4m3fn
    )
    q = (torch.randn(tokens, shape.heads, D, generator=g, device=device) * 2).to(
        torch.float8_e4m3fn
    )
    P = shape.kpool
    width, budget = shape.top_k + P - 1, shape.top_k // P
    topk = torch.full((tokens, width), -1, dtype=torch.int32, device=device)
    for t in range(tokens):
        limit = context + t + 1  # the row's own slot and those before it
        pools = limit // P
        if pools <= budget:
            pick = torch.arange(pools, device=device)
        else:
            pick = torch.randperm(pools, generator=g, device=device)[:budget]
        closed = (pick[:, None] * P + torch.arange(P, device=device)).reshape(-1)
        ids = torch.cat((closed, torch.arange(pools * P, limit, device=device)))
        topk[t, : ids.numel()] = slots[ids].int()
    return dict(kv_cache=kv, q=q, topk=topk)


def sparse_mla_reference(q, kv_cache, topk, scale=SOFTMAX_SCALE, kv_lora=512):
    """[T, heads, kv_lora] FP32: each row's softmax over its top-k slots of
    q . k * scale, times the slots' latent (the MLA value)."""
    T = q.size(0)
    out = []
    for t in range(T):
        idx = topk[t][topk[t] >= 0].long()
        k = kv_cache[idx].float()
        s = (q[t].float() @ k.T) * scale
        p = torch.softmax(s, dim=-1)
        out.append(p @ k[:, :kv_lora])
    return torch.stack(out)


def out_reference(q, kv_cache, topk, w: dict, shape: AttnShape = ATTN_SHAPE):
    """The sparse MLA (BF16, as the trtllm kernel returns it), the value
    up-projection (BF16 bmm) and this rank's o_proj partial (BF16)."""
    attn = sparse_mla_reference(q, kv_cache, topk, kv_lora=shape.kv_lora).to(
        torch.bfloat16
    )
    v = torch.einsum("thl,hvl->thv", attn.float(), w["w_uv"].float()).to(torch.bfloat16)
    partial = (v.reshape(v.size(0), -1).float() @ w["o_proj"].float().T).to(
        torch.bfloat16
    )
    return attn, v, partial


# ------------------------------------------------------------------ DSA indexer


@dataclasses.dataclass(frozen=True)
class IndexerShape:
    hidden: int = HIDDEN
    q_lora: int = 2048
    heads: int = 32
    head_dim: int = 128
    rope: int = 64
    page: int = 64
    top_k: int = 2048
    k_norm_eps: float = 1e-6


INDEXER_SHAPE = IndexerShape()
# weights_out = gate * softmax_scale * heads ** -0.5 * q_scale (SGLang's
# fused_q_indexer_rope_first_quant).
INDEXER_WEIGHT_SCALE = INDEXER_SHAPE.head_dim**-0.5 * INDEXER_SHAPE.heads**-0.5


def indexer_weights(shape: IndexerShape = INDEXER_SHAPE, seed: int = 0, device="cuda"):
    """wk | weights_proj as one [192, hidden] BF16 matrix (32 zero rows pad it
    to 64-row tiles), wq_b and the key LayerNorm."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    D, H = shape.head_dim, shape.heads
    wkw = torch.zeros(192, shape.hidden, device=device, dtype=torch.bfloat16)
    wkw[: D + H] = (
        torch.randn(D + H, shape.hidden, generator=g, device=device) * 0.02
    ).to(torch.bfloat16)
    wq_b = (torch.randn(H * D, shape.q_lora, generator=g, device=device) * 0.02).to(
        torch.bfloat16
    )
    return dict(
        wkw=wkw,
        wq_b=wq_b,
        k_norm_weight=1 + 0.1 * torch.randn(D, generator=g, device=device),
        k_norm_bias=0.1 * torch.randn(D, generator=g, device=device),
    )


def indexer_inputs(
    tokens: int,
    context: int,
    shape: IndexerShape = INDEXER_SHAPE,
    seed: int = 1,
    device="cuda",
):
    """A window of ``tokens`` rows after ``context`` cached keys: the layer
    input and q_lora, positions, the request's pages (shuffled in a pool), the
    index-K cache (random keys and scales) and the window's slots / lengths."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    P, T, page = context, tokens, shape.page
    pages = (P + T + page - 1) // page
    pool = max(2 * pages, 64)
    cache = torch.empty(pool, 132 * page, dtype=torch.uint8, device=device)
    keys = torch.randn(pool * page, shape.head_dim, generator=g, device=device)
    scales = torch.rand(pool * page, generator=g, device=device) * 0.01 + 0.001
    cache[:, : 128 * page] = (
        keys.to(torch.float8_e4m3fn).view(torch.uint8).view(pool, 128 * page)
    )
    cache[:, 128 * page :] = scales.view(pool, page).view(torch.uint8)
    page_table = torch.randperm(pool, generator=g, device=device)[:pages].to(
        torch.int32
    )
    pos = torch.arange(P, P + T, device=device)
    return dict(
        x=torch.randn(T, shape.hidden, generator=g, device=device).to(torch.bfloat16),
        q_lora=torch.randn(T, shape.q_lora, generator=g, device=device).to(
            torch.bfloat16
        ),
        positions=pos.to(torch.int64),
        cache=cache,
        page_table=page_table,
        cache_slot=(page_table[pos // page].to(torch.int64) * page + pos % page),
        lengths=torch.arange(P + 1, P + T + 1, device=device, dtype=torch.int32),
        cos_sin=cos_sin_cache(P + T + 64, shape.rope, device),
    )


def cache_views(cache: torch.Tensor, shape: IndexerShape = INDEXER_SHAPE):
    """The kernel's views of the [pages, 132 * 64] uint8 index-K cache: keys as
    [pages, 66, 128] E4M3 (rows 0-63), scales as [pages, 2112] FP32 (2048 + j).
    Flash's pooled cache has the same layout (a pool a row)."""
    pages = cache.size(0)
    return (
        cache.view(torch.float8_e4m3fn).view(pages, 66, shape.head_dim),
        cache.view(torch.float32).view(pages, 33 * shape.page),
    )


def _rope_first(x, cos, sin):
    """Interleaved RoPE of the leading 64 dims of x [..., 128]."""
    return torch.cat([_rope_interleaved(x[..., :64], cos, sin), x[..., 64:]], dim=-1)


def _fp8_rows(x):
    """FP8 per-row quantization: scale max(1e-4, absmax) / 448."""
    scale = x.abs().amax(-1).clamp_min(1e-4) / 448.0
    return (x / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def indexer_reference(inp: dict, w: dict, shape: IndexerShape = INDEXER_SHAPE):
    """FP32 torch reference: the window's FP8 keys and scales, the FP8 queries
    and head weights, and each row's logits over its keys (with the window's
    keys in place); the kernel's rounding points (BF16 GEMM outputs, BF16 keys
    before quantization)."""
    T, D, H = inp["x"].shape[0], shape.head_dim, shape.heads
    kw = (inp["x"].float() @ w["wkw"].float().T).to(torch.bfloat16).float()
    key, gate = kw[:, :D], kw[:, D : D + H]
    cs = inp["cos_sin"][inp["positions"]]
    cos, sin = cs[:, :32], cs[:, 32:]
    mean = key.mean(-1, keepdim=True)
    var = ((key - mean) ** 2).mean(-1, keepdim=True)
    k = (key - mean) * torch.rsqrt(var + shape.k_norm_eps) * w["k_norm_weight"] + w[
        "k_norm_bias"
    ]
    k = _rope_first(k, cos, sin).to(torch.bfloat16).float()
    k8, k_scale = _fp8_rows(k)
    q = (inp["q_lora"].float() @ w["wq_b"].float().T).to(torch.bfloat16).float()
    q = _rope_first(q.view(T, H, D), cos[:, None], sin[:, None])
    q8, q_scale = _fp8_rows(q)
    weights_out = gate * INDEXER_WEIGHT_SCALE * q_scale
    keys8, scales = cache_views(inp["cache"].clone(), shape)
    page, pt = shape.page, inp["page_table"].long()
    n = pt.numel() * page
    kk = keys8[pt, :page].reshape(n, D).float().clone()
    ks = scales[pt, 32 * page :].reshape(n).clone()
    P = int(inp["lengths"][0]) - 1
    kk[P : P + T] = k8.float()
    ks[P : P + T] = k_scale
    s = torch.einsum("thd,kd->thk", q8.float(), kk)
    logits = (torch.relu(s) * weights_out[:, :, None]).sum(1) * ks[None, :]
    return dict(k8=k8, k_scale=k_scale, q8=q8, weights=weights_out, logits=logits)


# -------------------------------------------------------------------- MTP draft


def eh_weights(device, seed: int = 0, hidden: int = HIDDEN) -> dict:
    """The MTP draft's embedding and eh weights (replicated on every rank) at
    checkpoint-like scales."""
    g = torch.Generator(device=device).manual_seed(seed)

    def norm_weight():
        return (1 + 0.1 * torch.randn(hidden, generator=g, device=device)).bfloat16()

    return {
        "embed": (
            0.02 * torch.randn(VOCAB, hidden, generator=g, device=device)
        ).bfloat16(),
        "enorm": norm_weight(),
        "hnorm": norm_weight(),
        "eh": (
            0.01 * torch.randn(hidden, 2 * hidden, generator=g, device=device)
        ).bfloat16(),
        "input_norm": norm_weight(),
    }


def eh_inputs(
    rows: int, device, seed: int = 1, hidden: int = HIDDEN
) -> tuple[torch.Tensor, torch.Tensor]:
    """A window's token ids and previous hidden states."""
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.randint(0, VOCAB, (rows,), generator=g, device=device)
    prev = (3 * torch.randn(rows, hidden, generator=g, device=device)).bfloat16()
    return ids, prev


def eh_reference(ids, prev, w) -> tuple[torch.Tensor, torch.Tensor]:
    """The stock chain: the embedding, fused_eh_norm (enorm / hnorm), the BF16
    eh_proj GEMM and the decoder's input RMSNorm: (normed output, residual)."""
    eh_in = torch.cat([rms(w["embed"][ids], w["enorm"]), rms(prev, w["hnorm"])], dim=-1)
    residual = torch.mm(eh_in, w["eh"].T)
    return rms(residual, w["input_norm"]), residual


def lm_head_inputs(
    seed: int, rank: int, tokens: int, ties: bool = False, hidden: int = HIDDEN
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicated final-norm rows and this rank's LM head shard [VOCAB / 4,
    hidden] (``ties``: a coarse grid, so many logits tie exactly and the
    tie-break decides)."""
    x = torch.Generator(device="cuda").manual_seed(seed)
    g = torch.Generator(device="cuda").manual_seed(seed * 16 + rank)
    x_rows = torch.randn(tokens, hidden, device="cuda", generator=x)
    weight = 0.15 * torch.randn(VOCAB // 4, hidden, device="cuda", generator=g)
    if ties:
        x_rows = (x_rows * 2).round() / 2
        weight = torch.randint(-1, 2, weight.shape, device="cuda", generator=g).float()
    return x_rows.bfloat16(), weight.bfloat16()


def check_greedy(logits, hidden, weight, greedy_ids, group, exact_ties: bool):
    """This rank's failures (none: it passed): every rank holds identical
    tokens; this rank's logits (when given) are BF16 roundings of the FP64 ones
    and the tokens are the gathered logits' first argmax; with ``exact_ties``
    (integer-valued inputs: FP32 sums exact) the tokens are the first argmax of
    the BF16-rounded exact logits. Every rank runs the same collectives."""
    import torch.distributed as dist

    rank, tokens, world = group.rank(), hidden.size(0), group.size()
    exact = hidden.double() @ weight.double().T
    ids = [torch.empty_like(greedy_ids) for _ in range(world)]
    dist.all_gather(ids, greedy_ids, group=group)
    failures = []
    if not all(torch.equal(i, greedy_ids) for i in ids):
        failures.append("ranks disagree on the greedy tokens")
    if logits is not None:
        mine = logits.view(tokens, world, -1)[:, rank].double()
        bad = (mine - exact).abs() > exact.abs() * 2.0**-8 + 1e-3
        if bool(bad.any()):
            failures.append(f"{int(bad.sum())} logits off their FP64 values")
        if not torch.equal(greedy_ids.long(), torch.argmax(logits, dim=-1)):
            failures.append("greedy tokens are not the logits' first argmax")
    if exact_ties:
        full = [torch.empty_like(exact) for _ in range(world)]
        dist.all_gather(full, exact.contiguous(), group=group)
        expected = torch.argmax(torch.cat(full, dim=1).bfloat16().float(), dim=-1)
        if not torch.equal(greedy_ids.long(), expected):
            failures.append(
                f"greedy tokens {greedy_ids.tolist()} != first argmax "
                f"{expected.tolist()}"
            )
    return failures


def stock_lm_head(hidden, weight, group):
    """The stock path: the BF16 lm head GEMM, NCCL all-gather of the BF16 vocab
    shards, rank-major copy, FP32 cast and argmax."""
    import torch.distributed as dist

    tokens, world = hidden.size(0), group.size()
    gathered = torch.empty(
        world, tokens, weight.size(0), dtype=torch.bfloat16, device="cuda"
    )
    greedy_ids = torch.empty(tokens, dtype=torch.int64, device="cuda")

    def call() -> torch.Tensor:
        logits = torch.mm(hidden, weight.T)
        dist.all_gather_into_tensor(gathered, logits, group=group)
        full = gathered.permute(1, 0, 2).reshape(tokens, -1).float()
        torch.argmax(full, dim=-1, out=greedy_ids)
        return full

    return call
