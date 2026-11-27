# ruff: noqa: ANN001, ANN002, ANN202
"""Synthetic GLM-5.3 Flash TP4 weights and inputs and the torch references of the
megakernels' checks, in SGLang's served layouts (``flashinfer_trtllm`` MoE runner,
trtllm DSA decode backend, ModelOpt NVFP4 linears on the generic FP4 GEMM branch).

GLM-5.3's generators (``glm5_3_tp4/_setup.py``) serve every layout the two models
share, at Flash's shapes (``MOE_SHAPE``, ``ATTN_SHAPE``): the trtllm-gen NVFP4
routed experts, the router and its window routing, the routed rows' reference, the
MLA front and output weights and references, the MLA cache with its top-k lists
(Flash pools the index keys: a row's list holds 512 closed pools of 4 tokens, then
its open pool), and the MTP draft's eh and LM head inputs. Flash's own: the NVFP4
shared expert and dense MLP, the pooled DSA indexer, the mHC layer boundary, KDA
and the MTP draft's BF16 MoE. ``compiled`` compiles every module's kernel with the
config its ``mk_config`` derives at bind.
"""

from __future__ import annotations

import functools
from itertools import starmap
import math

from pretuned_kernels.megakernels._qwen3_5_tp4 import all_reduce
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import same
from pretuned_kernels.megakernels.glm5_3_flash_tp4._mhc import HC_EPS
from pretuned_kernels.megakernels.glm5_3_flash_tp4._mhc import ITERS
from pretuned_kernels.megakernels.glm5_3_flash_tp4._mhc import POST_MULT
from pretuned_kernels.megakernels.glm5_3_tp4._setup import AttnShape
from pretuned_kernels.megakernels.glm5_3_tp4._setup import MoeShape
from pretuned_kernels.megakernels.glm5_3_tp4._setup import cache_views
from pretuned_kernels.megakernels.glm5_3_tp4._setup import nvfp4_dequant
from pretuned_kernels.megakernels.glm5_3_tp4._setup import nvfp4_experts
from pretuned_kernels.megakernels.glm5_3_tp4._setup import route_reference
from pretuned_kernels.megakernels.glm5_3_tp4._setup import routed_rows
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._setup import _nvfp4_quant
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._setup import _row_maps
from pretuned_kernels.megakernels.qwen3_5_moe_tp4._setup import swizzle_scales
import torch
import torch.nn.functional as F

HIDDEN = 4096
# The verify window (EAGLE 5/1/6).
T = 6
NORM_EPS = 1e-5
SWIGLU_LIMIT = 10.0
MOE_SHAPE = MoeShape(hidden=HIDDEN, num_experts=288)
DENSE_INTERMEDIATE = 3072  # per rank (TP4 of 12288)
# NoPE MLA (qk_rope 0), pooled DSA index keys.
ATTN_SHAPE = AttnShape(hidden=HIDDEN, rope=0, nope=256, q_lora=1536, kpool=4)
SCALE_LOG2 = ATTN_SHAPE.nope**-0.5 * math.log2(math.e)


def poison(buffers) -> None:
    """Fill ``buffers`` with NaN (BF16 / FP32) or 0xFF bytes (NaN in E4M3)."""
    for buffer in buffers:
        if buffer.dtype.is_floating_point and buffer.element_size() >= 2:
            buffer.fill_(float("nan"))
        else:
            buffer.view(torch.uint8).fill_(0xFF)


def repeat_bitwise(launches, poisoned) -> tuple[bool, bool]:
    """``launches``: (call, outputs) pairs, each output holding its call's result:
    every call again back to back (no host synchronization between them) must
    give the same bits, and then the last call again with ``poisoned`` (its
    scratch and outputs) filled with NaN first."""
    import torch.distributed as dist

    first = [[o.clone() for o in outputs] for _, outputs in launches]
    for call, _ in launches:
        call()
    torch.cuda.synchronize()
    back_to_back = all(
        same(o, f)
        for (_, outputs), firsts in zip(launches, first, strict=True)
        for o, f in zip(outputs, firsts, strict=True)
    )
    poison(poisoned)
    dist.barrier()
    launches[-1][0]()
    torch.cuda.synchronize()
    poisoned_ok = all(starmap(same, zip(launches[-1][1], first[-1], strict=True)))
    return back_to_back, poisoned_ok


def compiled(kernel, mk_config, args: tuple):
    """``kernel`` compiled for ``args`` with the config ``mk_config`` derives from
    its binding (Flash's configs are derived at bind, as ``kda_mtp``'s and the
    Inkling-2-Small kernels' are): derived and compiled once per binding, which
    the kernel caches by its specialization key."""
    return _compiled(kernel.bind(args), mk_config)


@functools.cache
def _compiled(bound, mk_config):
    return bound.compile_config(mk_config(bound))


# -------------------------------------------------------------------------- mHC

HC = 4  # hc_mult: residual streams


def sinkhorn(c: torch.Tensor) -> torch.Tensor:
    """comb logits [..., 4 (i), 4 (o)] FP32 -> SGLang's Sinkhorn: a row softmax
    (+ eps), a column normalization, then 19 (row, column) normalizations."""
    c = c.softmax(-1) + HC_EPS
    c = c / (c.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        c = c / (c.sum(-1, keepdim=True) + HC_EPS)
        c = c / (c.sum(-2, keepdim=True) + HC_EPS)
    return c


def hc_post(x, r, post, comb) -> torch.Tensor:
    """R'_o = bf16(post_o x + sum_i comb[i, o] R_i): x [T, H] BF16, r [T, 4, H]."""
    out = post.float().unsqueeze(-1) * x.float().unsqueeze(1)
    out = out + (comb.float().unsqueeze(-1) * r.float().unsqueeze(2)).sum(dim=1)
    return out.to(torch.bfloat16)


def hc_pre(r, fn, scale, base, norm_w):
    """The next sublayer's input (BF16 [T, H]), post [T, 4] and comb [T, 4, 4]
    (FP32): rs = rsqrt(mean(R^2) + eps) over the 4 streams, mix = (R @ fn^T) rs
    [T, 24] (fn [24, 4H]), pre = sigmoid(mix_pre s0 + b) + eps, post = 2
    sigmoid(mix_post s1 + b), comb = sinkhorn(mix_comb s2 + b), y = sum_i pre_i R_i
    (FP32), input = bf16(bf16(y) rsqrt(mean(y^2) + eps) w) (SGLang's TileLang
    convention: the norm of the FP32 y)."""
    t, n, h = r.shape
    x = r.reshape(t, n * h).float()
    rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)
    mix = (x.double() @ fn.double().T).float() * rs
    pre = torch.sigmoid(mix[:, :n] * scale[0] + base[:n]) + HC_EPS
    post = POST_MULT * torch.sigmoid(mix[:, n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = sinkhorn(mix[:, 2 * n :].view(t, n, n) * scale[2] + base[2 * n :].view(n, n))
    y = (pre.unsqueeze(-1) * r.float()).sum(dim=1)
    norm = torch.rsqrt(y.square().mean(-1, keepdim=True) + NORM_EPS)
    li = (y.to(torch.bfloat16).float() * norm * norm_w.float()).to(torch.bfloat16)
    return li, post, comb


def boundary_inputs(seed: int = 0, tokens: int = T, device="cuda") -> dict:
    """A layer boundary's inputs: the next sublayer's hc fn [24, 4H] (BF16), scale,
    base and input norm weight, residual streams of unequal scale, an all-reduced
    sublayer output ``x`` and the sublayer's post / comb."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(shape, device=device, generator=g) * scale

    streams = torch.tensor([1.0, 0.7, 1.3, 0.5], device=device)[None, :, None]
    return {
        "fn": randn(24, HC * HIDDEN, scale=0.02).to(torch.bfloat16),
        "base": randn(24, scale=0.5),
        "scale": 0.5 + randn(3, scale=0.05),
        "norm_w": (1.0 + randn(HIDDEN, scale=0.05)).to(torch.bfloat16),
        "r": (randn(tokens, HC, HIDDEN) * streams * 2).to(torch.bfloat16),
        "x": randn(tokens, HIDDEN, scale=0.8).to(torch.bfloat16),
        "post": POST_MULT * torch.sigmoid(randn(tokens, HC)),
        "comb": sinkhorn(randn(tokens, HC, HC)),
    }


def comb_logits(seed: int, tokens: int = T, device="cuda") -> torch.Tensor:
    """Random comb logits [T, 4, 4] (a raw comb: ``sinkhorn`` makes it final)."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(tokens, HC, HC, generator=g, device=device)


def boundary(x, t: dict, comb=None):
    """hc_post of the all-reduced sublayer output ``x`` with ``t``'s streams and
    mixes (``comb``: the final comb, default ``t["comb"]``), then the next
    sublayer's hc_pre: (input, R', post', comb')."""
    r_new = hc_post(x, t["r"], t["post"], t["comb"] if comb is None else comb)
    li, post, comb_new = hc_pre(r_new, t["fn"], t["scale"], t["base"], t["norm_w"])
    return li, r_new, post, comb_new


def report_boundary(rows, out, partial, t: dict, group, raw=False, tol=None) -> bool:
    """A kernel's mHC tail outputs ``out`` = (li, R', post', comb'): R' against
    ``hc_post`` of the all-reduce of every rank's reference ``partial``, li /
    post' / comb' against ``hc_pre`` of the kernel's R' (BF16: a one-ulp
    difference in R' moves post' by ~1e-5). ``raw``: the kernel took the comb as
    logits (``t["comb"]``, Sinkhorned here) and gave comb' as logits. ``tol``:
    relative RMS tolerances, default 2e-2 for li / R' and 1e-3 for post' /
    comb'."""
    tol = tol or (2e-2, 2e-2, 1e-3, 1e-3)
    comb = sinkhorn(t["comb"]) if raw else t["comb"]
    r_new = hc_post(
        all_reduce(partial, group).to(torch.bfloat16), t["r"], t["post"], comb
    )
    li, post, comb_new = hc_pre(out[1], t["fn"], t["scale"], t["base"], t["norm_w"])
    expected = (li, r_new, post, comb_new)
    got = (*out[:3], sinkhorn(out[3]) if raw else out[3])
    ok = True
    for label, a, b, tolerance in zip(
        ("li", "r_new", "post", "comb"), got, expected, tol, strict=True
    ):
        ok &= report(rows, label, a, b, tolerance)
    return ok


# -------------------------------------------------------------------------- KDA

KDA_HEADS = 16  # per rank (TP4 of 64)
KDA_HEAD_DIM = 128
SEGMENT = KDA_HEADS * KDA_HEAD_DIM
LOWER_BOUND = -5.0
# fused_qkvbfg_a_proj's row offsets: q | k | v | b | f_a | g_a.
B_ROW = 3 * SEGMENT
FA_ROW = B_ROW + KDA_HEADS
GA_ROW = FA_ROW + KDA_HEAD_DIM
FUSED_ROWS = GA_ROW + KDA_HEAD_DIM
# The buffers a verify step writes (besides its output).
KDA_WRITES = ("intermediate_ssm", "intermediate_conv")


def kda_inputs(seed: int = 0, padded: bool = False, device="cuda") -> dict:
    """A KDA layer's per-rank weights, states and verify window in SGLang's layouts
    (``Glm5NextLinearAttention``): the fused a projection [6416, 4096] (q | k | v
    by head, b, f_a | g_a replicated), F_B / G_B [2, 2048, 128], the causal conv
    weights [6144, 4] FP32, A_log, dt_bias, the gated norm weight, o_proj
    [4096, 2048]; the conv [slots, 3, 6144] and recurrent [slots, 16, 128, 128]
    states (read only) and the step's per-token writes (filled with 42). The
    request uses state slot 5 and scratch row 1; ``padded``: a padding slot (-1)."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16, scale=0.1):
        return (torch.randn(shape, device=device, generator=g) * scale).to(dtype)

    f32, D = torch.float32, KDA_HEAD_DIM
    slots, rows = 8, 3
    return {
        "hidden_states": randn(T, HIDDEN, scale=1.0),
        "fused_weight": randn(FUSED_ROWS, HIDDEN, scale=0.02),
        "fg_b": randn(2, SEGMENT, D, scale=0.05),
        "conv_weight": randn(3 * SEGMENT, 4, dtype=f32, scale=0.15),
        "a_log": randn(KDA_HEADS, dtype=f32, scale=0.2),
        "dt_bias": randn(SEGMENT, dtype=f32, scale=0.1),
        "conv_state": randn(slots, 3, 3 * SEGMENT),
        "recurrent_state": randn(slots, KDA_HEADS, D, D, dtype=f32, scale=0.02),
        "state_indices": torch.tensor([-1 if padded else 5], device=device).int(),
        "intermediate_ssm": torch.full((rows, T, KDA_HEADS, D, D), 42.0, device=device),
        "intermediate_conv": torch.full(
            (rows, T, 3, 3 * SEGMENT), 42.0, device=device, dtype=torch.bfloat16
        ),
        "intermediate_indices": torch.tensor([1], device=device).int(),
        "norm_weight": (1 + randn(D, dtype=f32, scale=0.02)).to(torch.bfloat16),
        "output_weight": randn(HIDDEN, SEGMENT, scale=0.02),
    }


def kda_reference(t: dict) -> torch.Tensor:
    """The verify step token by token in FP32 after SGLang's BF16 projections
    (q | k | v, beta = sigmoid(b), f = f_a F_B^T, g = g_a G_B^T), the conv output
    and the recurrence in FP32, the recurrence output rounded to BF16 before the
    gated norm rms(o) w sigmoid(g) (as SGLang's verify). Writes the step's caches
    into ``t``; returns the o_proj partial (BF16 [T, 4096])."""
    S, H, D = SEGMENT, KDA_HEADS, KDA_HEAD_DIM
    fused = (t["hidden_states"].float() @ t["fused_weight"].float().T).to(
        torch.bfloat16
    )
    qkv, b, fa, ga = torch.split(fused, [3 * S, H, D, D], dim=-1)
    f = (fa.float() @ t["fg_b"][0].float().T).to(torch.bfloat16)
    gate = (ga.float() @ t["fg_b"][1].float().T).to(torch.bfloat16)
    qkv, beta = qkv.float(), torch.sigmoid(b.float())
    decay = torch.exp(t["a_log"])[None, :, None] * (f.float() + t["dt_bias"]).view(
        T, H, D
    )
    log_decay = LOWER_BOUND * torch.sigmoid(decay)
    gate = torch.sigmoid(gate.float()).view(T, H, D)
    out = torch.zeros(T, H, D, device=qkv.device, dtype=torch.bfloat16)
    slot = int(t["state_indices"][0])
    if slot >= 0:
        row = int(t["intermediate_indices"][0])
        window = t["conv_state"][slot].float().clone()  # [3, 3S]
        state = t["recurrent_state"][slot].clone()  # [H, v, k]
        for step in range(T):
            four = torch.cat((window, qkv[step][None]), dim=0)  # tap 3: this token
            conv = F.silu((four.T * t["conv_weight"]).sum(dim=1)).view(3, H, D)
            window = four[1:].clone()
            t["intermediate_conv"][row, step] = window.to(torch.bfloat16)
            q, k, v = conv.unbind(0)
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D**-0.5
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            state = state * torch.exp(log_decay[step])[:, None, :]
            delta = v - (state * k[:, None, :]).sum(-1)
            state = state + (delta * beta[step][:, None])[:, :, None] * k[:, None, :]
            t["intermediate_ssm"][row, step] = state
            core = (state * q[:, None, :]).sum(-1).to(torch.bfloat16).float()
            norm = torch.rsqrt((core * core).mean(-1, keepdim=True) + NORM_EPS)
            out[step] = (core * norm * t["norm_weight"].float() * gate[step]).to(
                torch.bfloat16
            )
    return F.linear(out.view(T, S).float(), t["output_weight"].float()).to(
        torch.bfloat16
    )


def kda_writes_correct(candidate: dict, reference: dict) -> bool:
    """The step's cache writes against the reference's: the rows it must leave
    alone unchanged, the request's row within tolerance (the conv windows hold
    BF16 values: compared at their ulp)."""
    slot, row = (
        int(candidate["state_indices"][0]),
        int(candidate["intermediate_indices"][0]),
    )
    ok = True
    for name in KDA_WRITES:
        value, want = candidate[name], reference[name]
        untouched = torch.ones(value.shape[0], dtype=torch.bool, device=value.device)
        untouched[row] = slot < 0
        ok &= torch.equal(value[untouched], want[untouched])
        atol, rtol = (2e-3, 8e-3) if name == "intermediate_conv" else (3e-4, 2e-2)
        ok &= torch.allclose(value.float(), want.float(), atol=atol, rtol=rtol)
    return ok


# ------------------------------------------------------------ MoE and dense MLP


def nvfp4_mlp(x_q, mlp: dict, kernel_act: tuple):
    """An NVFP4 MLP (SGLang's ModelOpt linears: the MoE's shared expert, the dense
    MLP) on its logical weights (``mlp``: ``w13_log`` rows [gate; up], ``w13s_log``,
    ``w2_log``, ``w2s_log``, ``alpha1``, ``alpha2``, ``act_global``), in FP32 up to
    the activation's NVFP4 (the megakernels'; SGLang rounds gate|up and the
    activation to BF16): gate|up times alpha1, SGLang's clamps (gate <= 10, |up|
    <= 10), SwiGLU; down on the kernel's activation ``kernel_act`` (packed [T, N /
    2], scales [T, N / 16]) times alpha2, BF16. ``x_q``: the kernel's dequantized
    NVFP4 input [T, H]. Also the activation as (the kernel's, its exact NVFP4)."""
    n = mlp["w13_log"].shape[0] // 2
    gate_up = (x_q @ nvfp4_dequant(mlp["w13_log"], mlp["w13s_log"]).T) * mlp["alpha1"]
    gate = gate_up[:, :n].clamp(max=SWIGLU_LIMIT)
    up = gate_up[:, n:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    act_q = nvfp4_dequant(*kernel_act)
    exact = _nvfp4_quant(gate * torch.sigmoid(gate) * up, mlp["act_global"], act_q)
    w2 = nvfp4_dequant(mlp["w2_log"], mlp["w2s_log"])
    return ((act_q @ w2.T) * mlp["alpha2"]).to(torch.bfloat16), (act_q, exact)


def nvfp4_linears(g, intermediate: int) -> dict:
    """An NVFP4 MLP's logical weights (``nvfp4_mlp``), input global scale 64."""
    experts = nvfp4_experts(g, 1, intermediate, HIDDEN)
    mlp = {k: v[0] for k, v in experts.items() if k.endswith("_log")}
    alpha1, alpha2 = _alphas(g, 1, 64.0, 64.0)
    return dict(mlp, alpha1=alpha1[0], alpha2=alpha2[0], act_global=64.0)


def _alphas(g, count: int, input_global: float, act_global: float):
    """Per-output GEMM1 / GEMM2 alphas: 1 / (global scale x the weights' typical
    block scale), with a +-25% spread."""
    a1 = (torch.rand(count, generator=g, device=g.device) * 0.5 + 0.75) / (
        input_global * 64
    )
    a2 = (torch.rand(count, generator=g, device=g.device) * 0.5 + 0.75) / (
        act_global * 16
    )
    return a1, a2


def moe_weights(seed: int = 0, device="cuda", shape: MoeShape = MOE_SHAPE) -> dict:
    """One rank's MoE weights under the names ``moe_tp4.prepared_weights`` reads
    (SGLang's processed ``DeepseekV2MoE``): the routed experts (``nvfp4_experts``:
    the trtllm-gen layout, read in place, and the logical copies), their GEMM1 /
    GEMM2 alphas (``g1_alphas`` = ``g1_alphas_up``, ``g2_alphas``), the SwiGLU
    clamp on GEMM1's raw accumulators (``gemm1_clamp_limit`` = 10 / g1_alphas) and
    the global scales of the routed input and activation; the router and its
    correction bias; the NVFP4 shared expert (``shared``: ``nvfp4_linears``; its
    W13 and block scales logical, W2 as SGLang holds it with 128x4-interleaved
    scales, its alphas and global scales)."""
    g = torch.Generator(device=device).manual_seed(seed)
    E, H, N = shape.num_experts, shape.hidden, shape.intermediate
    w = nvfp4_experts(g, E, N, H)
    g1, g2 = _alphas(g, E, 64.0, 64.0)
    shared = nvfp4_linears(g, shape.shared_intermediate)
    one = torch.ones((), device=device)
    return dict(
        w,
        router=(torch.randn(E, H, generator=g, device=device) * 0.02).to(
            torch.bfloat16
        ),
        bias=torch.randn(E, generator=g, device=device) * 0.05,
        g1_alphas=g1,
        g1_alphas_up=g1,
        g2_alphas=g2,
        gemm1_clamp_limit=SWIGLU_LIMIT / g1,
        w13_input_scale_quant=64.0 * one,
        w2_input_scale_quant=64.0 * one,
        shared=shared,
        sw13_log=shared["w13_log"],
        sw13s_log=shared["w13s_log"],
        sw2=shared["w2_log"],
        sw2s_sw=swizzle_scales(shared["w2s_log"]),
        s13_alpha=shared["alpha1"],
        s13_input_scale_inv=64.0 * one,
        s2_alpha=shared["alpha2"],
        s2_input_scale_inv=shared["act_global"] * one,
    )


def moe_reference(x, w: dict, kernel: dict, shape: MoeShape = MOE_SHAPE):
    """This rank's MoE partial (BF16 [T, H]: the picks' W2 rows weighted by FP32
    routing weights in pick order, plus the shared expert, one rounding), the
    top-k ids, each pick's W2 row (``routed_rows``: the clamped gatedSilu) and
    the shared output, on the logical weights; each GEMM reads the kernel's copy
    of its input (``kernel``: its scratch, ``moe_tp4.scratch``'s names, the
    activation rows for the window's distinct experts). ``quant``: the kernel's
    NVFP4 copies against their exact quantizations."""
    _, ids, wts = route_reference(x, w, shape)
    distinct = len({int(e) for e in ids.flatten()})
    x_q = nvfp4_dequant(kernel["input_q"], kernel["input_scale"])
    acts = (kernel["activation_q"][:distinct], kernel["activation_scale"][:distinct])
    y, act = routed_rows(
        x_q,
        ids,
        w,
        w["g1_alphas"],
        w["g2_alphas"],
        acts,
        w["w2_input_scale_quant"].item(),
        clamp=w["gemm1_clamp_limit"],
    )
    s_x_q = nvfp4_dequant(kernel["s_input_q"], kernel["s_input_scale"])
    shared_act = (kernel["shared_act_q"], kernel["shared_act_scale"])
    shared, s_act = nvfp4_mlp(s_x_q, w["shared"], shared_act)
    routed = sum(wts[:, k, None] * y[:, k].float() for k in range(shape.top_k))
    xf = x.float()
    quant = {
        "input (NVFP4)": (
            x_q,
            _nvfp4_quant(xf, w["w13_input_scale_quant"].item(), x_q),
        ),
        "shared input (NVFP4)": (
            s_x_q,
            _nvfp4_quant(xf, w["s13_input_scale_inv"].item(), s_x_q),
        ),
        "activation (NVFP4)": act,
        "shared act (NVFP4)": s_act,
    }
    partial = (routed + shared.float()).to(torch.bfloat16)
    return partial, ids, y, shared, quant


def dense_weights(seed: int = 0, device="cuda") -> dict:
    """One rank's dense MLP (layers 0-2; ``nvfp4_linears``) under the names
    ``dense_tp4.prepared_weights`` reads: gate|up as SGLang loads it (rows [gate;
    up]) with its logical block scales, down as SGLang holds it with its
    128x4-interleaved scales, the alphas and the input global scales."""
    w = nvfp4_linears(
        torch.Generator(device=device).manual_seed(seed), DENSE_INTERMEDIATE
    )
    one = torch.ones((), device=device)
    return dict(
        w,
        w13=w["w13_log"],
        w2=w["w2_log"],
        w2s_sw=swizzle_scales(w["w2s_log"]),
        in_inv1=64.0 * one,
        in_inv2=w["act_global"] * one,
    )


# ------------------------------------------------------------ MLA + DSA indexer

IDX_HEADS = 32
IDX_DIM = 128
KPOOL = ATTN_SHAPE.kpool
TAIL_SLOTS = KPOOL + T  # index_kpool + tail_extra_slots (the draft tokens)
POOL_PAGE = (
    64  # pools a pooled index-K page holds (132 B a pool: 128 E4M3 + an FP32 scale)
)


def indexer_layer_weights(seed: int = 0, device="cuda") -> dict:
    """An MLA layer's DSA indexer (replicated): wq_b [32 x 128, 1536], wk and the
    compress gate [128, 4096] (BF16), weights_proj [32, 4096] (FP32 holding BF16
    values, as the checkpoint), the key LayerNorm and the pool compression's ape
    [4, 128] (FP32)."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale):
        return torch.randn(shape, generator=g, device=device) * scale

    return {
        "wq_b": randn(IDX_HEADS * IDX_DIM, ATTN_SHAPE.q_lora, scale=0.02).bfloat16(),
        "wk": randn(IDX_DIM, HIDDEN, scale=0.02).bfloat16(),
        "weights_proj": randn(IDX_HEADS, HIDDEN, scale=0.02).bfloat16().float(),
        "k_norm_w": 1 + randn(IDX_DIM, scale=0.1),
        "k_norm_b": randn(IDX_DIM, scale=0.1),
        "compress_gate": randn(IDX_DIM, HIDDEN, scale=0.02).bfloat16(),
        "ape": randn(KPOOL, IDX_DIM, scale=0.5),
    }


def hadamard_matrix(n: int = IDX_DIM, device="cuda") -> torch.Tensor:
    """Sylvester's +-1 Hadamard matrix [n, n] (exact in BF16): y = H x is the
    unnormalized fast Walsh-Hadamard transform."""
    h = torch.ones(1, 1, device=device)
    while h.size(0) < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def pow2_quant(x: torch.Tensor):
    """act_quant's ue8m0 branch over the last dim: scale = 2^ceil(log2(max(amax,
    1e-4) / 448)), the E4M3 of clamp(x / scale)."""
    amax = x.abs().amax(-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax * (1.0 / 448.0))))
    return (x / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def index_state(
    context: int, seed: int = 5, device="cuda", wide_pages: int = 0
) -> dict:
    """A request's indexer state before a verify step: ``context`` committed tokens
    (write_start), its committed pools (random keys / pow2 scales) in shuffled
    pooled pages, the 10-slot tail rings of 4 request slots (random: the open
    pool's committed tokens and stale entries), the pooled page table (padded to
    ``wide_pages`` entries with page 0, as a CUDA graph's), the write plan
    (SGLang's update_kpool_write_plan: the <= 2 pools the window closes) and each
    row's pool count."""
    g = torch.Generator(device=device).manual_seed(seed)
    groups = ((context + T) // KPOOL + 1 + POOL_PAGE - 1) // POOL_PAGE
    n_pages = max(2 * groups + 4, 16)
    # page 0: the padding page
    perm = torch.randperm(n_pages - 1, generator=g, device=device)[:groups] + 1
    cache = torch.zeros(n_pages, 132 * POOL_PAGE, dtype=torch.uint8, device=device)
    keys, scales = cache_views(cache)
    pool = torch.arange(context // KPOOL, device=device)
    kv = torch.randn(pool.numel(), IDX_DIM, generator=g, device=device)
    page, at = perm[pool // POOL_PAGE], pool % POOL_PAGE
    keys[page, at] = (kv * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    exponent = torch.randint(-12, -6, (pool.numel(),), generator=g, device=device)
    scales[page, 2048 + at] = torch.exp2(exponent.float())
    table = perm.to(torch.int32)
    if wide_pages:
        padding = torch.zeros(
            wide_pages - table.numel(), dtype=torch.int32, device=device
        )
        table = torch.cat([table, padding])
    base = context // KPOOL
    closing = base + torch.arange((T + KPOOL - 1) // KPOOL, device=device)
    write_loc = perm[closing // POOL_PAGE].long() * POOL_PAGE + closing % POOL_PAGE
    rows = context + 1 + torch.arange(T, device=device)
    return {
        "ws": context,
        "cache": cache,
        "page_table": table,
        "tail_k": torch.randn(
            4, TAIL_SLOTS, IDX_DIM, generator=g, device=device
        ).bfloat16(),
        "tail_s": torch.randn(
            4, TAIL_SLOTS, IDX_DIM, generator=g, device=device
        ).bfloat16(),
        "req": torch.tensor([2], dtype=torch.int64, device=device),
        "write_start": torch.tensor([context], dtype=torch.int32, device=device),
        "tail_start": torch.tensor([base * KPOOL], dtype=torch.int32, device=device),
        "write_loc": write_loc.view(1, -1),
        "pool_lens": (rows // KPOOL).to(torch.int32),
    }


def clone_state(state: dict) -> dict:
    """``state`` with private copies of the buffers a step writes."""
    return {**state, **{k: state[k].clone() for k in ("cache", "tail_k", "tail_s")}}


def indexer_reference(x, q_lora, w: dict, st: dict):
    """FP32 reference of SGLang's indexer step (its rounding points): the key
    (wk, LayerNorm, BF16) and gate score into the request's tail ring; each pool
    the window closes compressed (a per-dim softmax over its 4 slots of score +
    ape, the keys' weighted sum, BF16, Hadamard, BF16, pow2 FP8) into the pooled
    cache at its write_loc; the queries (wq_b, Hadamard, pow2 FP8), the head
    weights and each row's logits over the pools the last row sees (DeepGEMM's
    paged MQA logits). Writes ``st``'s cache and rings; returns (q8 [T, 32,
    128], weights [T, 32], logits [T, pools])."""
    dev = x.device
    hm = hadamard_matrix(IDX_DIM, dev)
    key = (x.float() @ w["wk"].float().T).to(torch.bfloat16).float()
    mean = key.mean(-1, keepdim=True)
    var = (key - mean).square().mean(-1, keepdim=True)
    key = (key - mean) * torch.rsqrt(var + 1e-6) * w["k_norm_w"] + w["k_norm_b"]
    key = key.to(torch.bfloat16)
    score = (x.float() @ w["compress_gate"].float().T).to(torch.bfloat16)
    ws, r = st["ws"], int(st["req"][0])
    for t in range(x.size(0)):
        st["tail_k"][r, (ws + t) % TAIL_SLOTS] = key[t]
        st["tail_s"][r, (ws + t) % TAIL_SLOTS] = score[t]
    keys, scales = cache_views(st["cache"])
    start = (ws // KPOOL) * KPOOL
    for p in range((ws + x.size(0)) // KPOOL - ws // KPOOL):
        ring = [(start + KPOOL * p + s) % TAIL_SLOTS for s in range(KPOOL)]
        sc = torch.stack(
            [st["tail_s"][r, i].float() + w["ape"][s] for s, i in enumerate(ring)]
        )
        kk = torch.stack([st["tail_k"][r, i].float() for i in ring])
        pooled = (torch.softmax(sc, dim=0) * kk).sum(0).to(torch.bfloat16).float()
        q8p, sp = pow2_quant(((hm @ pooled) * IDX_DIM**-0.5).to(torch.bfloat16).float())
        loc = int(st["write_loc"][0, p])
        keys[loc // POOL_PAGE, loc % POOL_PAGE] = q8p
        scales[loc // POOL_PAGE, 2048 + loc % POOL_PAGE] = sp
    q = (q_lora.float() @ w["wq_b"].float().T).to(torch.bfloat16).float()
    q = ((q.view(-1, IDX_HEADS, IDX_DIM) @ hm.T) * IDX_DIM**-0.5).to(torch.bfloat16)
    q8, q_scale = pow2_quant(q.float())
    gates = (x.float() @ w["weights_proj"].float().T) * IDX_HEADS**-0.5
    weights = gates * q_scale * IDX_DIM**-0.5
    pool = torch.arange(int(st["pool_lens"][-1]), device=dev)
    page = st["page_table"][pool // POOL_PAGE].long()
    kk = keys[page, pool % POOL_PAGE].float()
    ks = scales[page, 2048 + pool % POOL_PAGE]
    s = torch.einsum("thd,kd->thk", q8.float(), kk)
    logits = (torch.relu(s) * weights[:, :, None]).sum(1) * ks[None, :]
    return q8, weights, logits


# ---------------------------------------------------------------- MTP draft MoE


def bf16_block_layout(w13: torch.Tensor, w2: torch.Tensor):
    """SGLang's TRT-LLM BF16 block layout (``_apply_trtllm_bf16_block_layout``) of
    canonical [E, up | gate, H] / [E, H, I] experts: the trtllm-gen row maps of
    the NVFP4 layout (``qwen3_5_moe_tp4._row_maps``), then K blocks of 64:
    [E, H / 64, 2I, 64] / [E, I / 64, H, 64]."""
    E, two_i, H = w13.shape
    phys13, phys2 = _row_maps(w13.device, two_i // 2, H)
    w13 = w13[:, phys13].view(E, two_i, H // 64, 64).transpose(1, 2).contiguous()
    w2 = w2[:, phys2].view(E, H, -1, 64).transpose(1, 2).contiguous()
    return w13, w2


def mtp_moe_weights(seed: int = 0, device="cuda", shape: MoeShape = MOE_SHAPE) -> dict:
    """One rank's draft MoE (the unquantized NextN layer) at checkpoint-like
    scales: the router and its correction bias, the BF16 routed experts in the
    TRT-LLM block layout (``mtp_moe_tp4.kernel_args``' operands) and their
    canonical rows, the BF16 shared expert (rows [gate; up])."""
    g = torch.Generator(device=device).manual_seed(seed)
    E, H, N = shape.num_experts, shape.hidden, shape.intermediate

    def randn(*size, scale=0.02):
        return (scale * torch.randn(*size, generator=g, device=device)).bfloat16()

    router = randn(E, H)
    bias = 0.05 * torch.randn(E, generator=g, device=device) + 10.0
    canon = {"up": randn(E, N, H), "gate": randn(E, N, H), "down": randn(E, H, N)}
    w13, w2 = bf16_block_layout(
        torch.cat([canon["up"], canon["gate"]], dim=1), canon["down"]
    )
    sw13 = randn(2 * shape.shared_intermediate, H)
    return {
        "router": router,
        "bias": bias,
        "w13": w13,
        "w2": w2,
        "sw13": sw13,
        "sw2": randn(H, shape.shared_intermediate),
        "canon": canon,
    }


def mtp_moe_reference(x, w: dict, shape: MoeShape = MOE_SHAPE):
    """This rank's draft MoE partial (BF16 [T, H]) and top-k ids at SGLang's
    rounding points: the routed SwiGLU (no clamp) and each W2 row in BF16, the
    rows weighted by BF16 weights; the shared expert's gate|up and activation in
    BF16 with SGLang's clamps; routed + shared in BF16."""
    _, ids, wts = route_reference(x, w, shape)
    c = w["canon"]
    out = torch.zeros(x.shape[0], shape.hidden, device=x.device)
    for t in range(x.shape[0]):
        for k in range(shape.top_k):
            e = int(ids[t, k])
            gate = x[t].float() @ c["gate"][e].float().T
            up = x[t].float() @ c["up"][e].float().T
            act = (gate * torch.sigmoid(gate) * up).bfloat16().float()
            y = (act @ c["down"][e].float().T).bfloat16().float()
            out[t] += y * wts[t, k].bfloat16().float()
    N = shape.shared_intermediate
    gate_up = (x.float() @ w["sw13"].float().T).bfloat16().float()
    gate = gate_up[:, :N].clamp(max=SWIGLU_LIMIT)
    up = gate_up[:, N:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    act = (gate * torch.sigmoid(gate) * up).bfloat16().float()
    shared = (act @ w["sw2"].float().T).bfloat16().float()
    return (out.bfloat16().float() + shared).bfloat16(), ids
