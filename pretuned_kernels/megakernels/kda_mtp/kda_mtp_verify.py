# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""Kimi-K3 KDA speculative verification as one Gluon megakernel: the TP8 shard
(12 heads), one request, T = 8 draft tokens, ReplaySSM cache ring, on GB200.

Roots, in launch order (one persistent CTA per SM, 8-deep TMA ring):
  1. q/k/v projection: [8, 7168] x [4608, 7168]^T, 4 K splits of 1792 (144
     tasks), split-K atomics into ``pq``.
  2. f_a projection, 8 K splits of 896 (8 tasks); each task also computes its
     split of beta with a register dot from a plain-load alias of the hidden
     states (``hidp``, not a ring tensor); atomics into ``pfb``.
  3. output-gate projection: 4 K splits (48 tasks), atomics into ``pg``.
  4. recurrence per (head, 16 value rows) (96 tasks): the F_B decay first
     (it waits only for f_a), then the conv taps straight from ``pq`` (the
     q/k/v wait), q/k L2 norm, and the delta rule over the 8 tokens in
     order through task-private scratch; per-token ReplaySSM outputs.
  5. gated RMSNorm per head (reads the gate before the recurrence output:
     a task waits for what it reads in the order it reads it).
  6. output projection [8, 1536] x [7168, 1536]^T (112 tasks); task 0
     re-zeroes the split-K scratch for the next launch.

Comms are left out (TP all-reduce of the output projection).  Needs the
compiler commit that orders a task's waits by its reads and proves more
stores apart (no CTA barriers between the recurrence's stores).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

D = 128
H = 12
S = H * D
T = 8


def _tap(pq, conv_state, sl, ok, col, j: int):
    """Conv input at token t - j [8, C]: the bf16-rounded projection sum, or the
    committed window row 3 + t - j where t < j."""
    tt = hl.arange(8)[:, None]
    zt = tt * 0
    p = hl.load(pq, [torch.clamp(tt - j, min=0), col[None, :]], extra_mask=tt >= j)
    if j == 0:
        return p.to(torch.bfloat16).to(torch.float32)
    w = hl.load(
        conv_state,
        [sl + zt, torch.clamp(tt + 3 - j, max=2), col[None, :]],
        extra_mask=(tt < j) & ((zt + ok) != 0),
    )
    return torch.where(tt >= j, p.to(torch.bfloat16).to(torch.float32), w.to(torch.float32))


def _win3(iconv, scr, ok, col, x0, x1, x2, cmask):
    """Token t's conv window (inputs at t-2, t-1, t) -> iconv[scr, t], channels
    where ``cmask``."""
    tt = hl.arange(8)[:, None]
    zt = tt * 0
    m = ((zt + ok) != 0) & cmask[None, :]
    hl.store(iconv, [scr + zt, tt, zt, col[None, :]], x2.to(torch.bfloat16), extra_mask=m)
    hl.store(iconv, [scr + zt, tt, zt + 1, col[None, :]], x1.to(torch.bfloat16), extra_mask=m)
    hl.store(iconv, [scr + zt, tt, zt + 2, col[None, :]], x0.to(torch.bfloat16), extra_mask=m)


def _cw(conv_w, kind: int, ch):
    """The conv weights of channels ``ch`` broadcast over the 8 tokens [8, C]
    (the taps' shape: a [C] vector broadcast at the multiply costs a layout
    conversion through shared memory)."""
    z = hl.arange(8)[:, None] * 0
    return (
        hl.load(conv_w, [z + kind, z, ch[None, :]]),
        hl.load(conv_w, [z + kind, z + 1, ch[None, :]]),
        hl.load(conv_w, [z + kind, z + 2, ch[None, :]]),
        hl.load(conv_w, [z + kind, z + 3, ch[None, :]]),
    )


def _convy(x0, x1, x2, x3, w0, w1, w2, w3):
    """SiLU(causal conv) [8, C]: y[t] = sum_k w[k] row[t + k]."""
    y = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3
    return y * torch.sigmoid(y)


def _token(stg, sdec, sbeta, task, d, t: int):
    """Token ``t``'s recurrence operands: q, k, decay [1, 128], v [16, 1], beta."""
    zd = d[None, :] * 0
    j = hl.arange(16)[:, None]
    zj = j * 0
    q = hl.load(stg, [task + zd, zd, zd + t, d[None, :]])
    k = hl.load(stg, [task + zd, zd + 1, zd + t, d[None, :]])
    e = hl.load(sdec, [task + zd, zd + t, d[None, :]])
    v = hl.load(stg, [task + zj, zj + 2, zj + t, j])
    b = hl.load(sbeta, [task, t])
    return q, k, e, v, b


def _step(st, q, k, e, v, b):
    """One delta-rule step of state rows ``st`` [16, 128] -> (state, o [16, 1])."""
    st = st * e
    r = v - torch.sum(st * k, dim=1, keepdim=True)
    st = st + (b * r) * k
    return st, torch.sum(st * q, dim=1, keepdim=True)


@helion.kernel(
    static_shapes=True,
    backend="gluon",
    dot_precision="tf32",
    ignore_warnings=[helion.exc.FastMathUnused],
)
def kda_mtp(
    hid4,  # [8, 4, 1792] bf16: hidden_states (4 K splits)
    w_qkv,  # [4608, 4, 1792] bf16: input_weight rows q k v
    hid8,  # [8, 8, 896] bf16: hidden_states (8 K splits)
    w_fa,  # [128, 8, 896] bf16: input_weight rows f_a
    hidp,  # [8, 8, 896] bf16: hidden_states, plain loads (not the ring)
    w_beta,  # [16, 8, 896] bf16: input_weight rows beta (12), pad (4)
    w_g,  # [1536, 4, 1792] bf16: input_weight rows of the output gate
    f_b,  # [1536, 128] bf16
    conv_w,  # [3, 4, 1536] fp32 (contiguous)
    a_log,  # [12] fp32
    dt_bias,  # [1536] fp32
    conv_state,  # [P, 3, 4608] bf16
    rec_state,  # [P, 12, 128, 128] fp32
    state_idx,  # [1] int32
    iconv,  # [R, C, 3, 4608] bf16
    iconv_idx,  # [1] int32
    norm_w,  # [128] fp32
    w_out,  # [7168, 1536] bf16
    rawv,  # [P, 12, C, 128] bf16
    rawk,  # [P, 12, C, 128] bf16
    rg,  # [P, 12, C, 128] fp32
    rbeta,  # [P, 12, C] fp32
    pq,  # [8, 4608] fp32 scratch: the q/k/v projection's split-K sum (zero between launches)
    pfb,  # [8, 256] fp32 scratch: the f_a/beta projection's split-K sum (zero between launches)
    pg,  # [8, 1536] fp32 scratch: the output-gate projection's split-K sum (zero between launches)
    stg,  # [96, 3, 8, 128] fp32 scratch: per recurrence task q, k (normalized), v
    sdec,  # [96, 8, 128] fp32 scratch: per recurrence task decay
    sbeta,  # [96, 8] fp32 scratch: per recurrence task beta
    opre,  # [8, 1536] fp32 scratch: core output before the gated RMSNorm
    ob,  # [8, 1536] bf16 scratch: normalized core
    out,  # [8, 7168] bf16
):
    pass
    hl.pdl_launch_dependents()
    hl.pdl_wait()
    # 1. projection rows the recurrence needs first: q k v (4608) and f_a/beta
    # (140), 4 K splits of 1792 (152 tasks, split-K atomics)
    for tile_m, tile_n, tile_s in hl.tile([4608, 8, 4], block_size=[128, 16, 1]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(1792, block_size=64):
            acc = hl.dot(w_qkv[tile_m, tile_s.begin, tile_k], hid4[tile_n, tile_s.begin, tile_k].T, acc=acc)
        hl.atomic_add(pq, [tile_n, tile_m], acc.T)
    # f_a (128) and beta (12) rows, 8 K splits of 896, split-fastest: on the 8
    # workers the q/k/v tasks leave idle; the beta tile's rows past 16 are TMA
    # zero fill
    for tile_s, tile_m, tile_n in hl.tile([8, 128, 8], block_size=[1, 128, 16]):
        # (body loads take ~5 us under the projection's HBM load: issue the
        # slot and beta operand loads together, before anything waits)
        t8 = hl.arange(8)
        j16 = hl.arange(16)
        kk = hl.arange(896)
        hx = hl.load(hidp, [t8[:, None], tile_s.begin + t8[:, None] * 0, kk[None, :]])
        wb = hl.load(w_beta, [j16[:, None], tile_s.begin + j16[:, None] * 0, kk[None, :]])
        # beta rows: this split's [16, 896] x [8, 896] in registers (off the ring)
        pb = hl.dot(hx, wb.T)  # [8, 16] fp32
        hl.atomic_add(pfb, [t8, 128 + j16], pb)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(896, block_size=64):
            acc = hl.dot(w_fa[tile_m, tile_s.begin, tile_k], hid8[tile_n, tile_s.begin, tile_k].T, acc=acc)
        hl.atomic_add(pfb, [tile_n, tile_m], acc.T)
    # 2. output-gate projection rows (1536), 4 K splits (48 tasks on the
    # workers after the q/k/v tasks); each task then warms its share of the
    # output projection's weights
    for tile_m, tile_n, tile_s in hl.tile([1536, 8, 4], block_size=[128, 16, 1]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(1792, block_size=64):
            acc = hl.dot(w_g[tile_m, tile_s.begin, tile_k], hid4[tile_n, tile_s.begin, tile_k].T, acc=acc)
        hl.atomic_add(pg, [tile_n, tile_m], acc.T)

    # 3. per (head, 16 value rows): conv (q, k all channels; v this task's 16),
    # q/k L2 norm, F_B decay, then the delta rule over the 8 tokens in order.
    # The operands go through task-private scratch (a CTA barrier between) so
    # the chain runs in the state's layout: key vectors [1, 128], value columns
    # [16, 1].
    for tile_h, tile_v in hl.tile([12, 128], block_size=[1, 16]):
        h = tile_h.begin
        v0 = tile_v.begin
        tt = hl.arange(8)
        d = hl.arange(128)
        vr = v0 + hl.arange(16)
        zt = tt[:, None] * 0
        z2 = vr[:, None] * 0
        slot = hl.load(state_idx, [0])
        scr = hl.load(iconv_idx, [0])
        ok = slot >= 0
        sl = torch.where(ok, slot, 0)
        # (the address depends on a loaded value: no cp.async pipelining of
        # the scratch reloads above the barrier)
        task = h * 8 + tile_v.id + slot * 0
        mine = (d >= v0) & (d < v0 + 16)
        mt = (zt + ok) != 0
        # every global load first: layout conversions (smem round trips with
        # a CTA barrier) between loads would serialize their latencies
        fa = hl.load(pfb, [tt[:, None], slot * 0 + d[None, :]])
        braw = hl.load(pfb, [tt[:, None], 128 + h + zt])  # [8, 1]
        fb = hl.load(f_b, [h * 128 + d[:, None], d[None, :]])  # [d, j]
        dtb = hl.load(dt_bias, [h * 128 + zt + d[None, :]])
        al = hl.load(a_log, [h])
        wq0, wq1, wq2, wq3 = _cw(conv_w, 0, h * 128 + d)
        wk0, wk1, wk2, wk3 = _cw(conv_w, 1, h * 128 + d)
        wv0, wv1, wv2, wv3 = _cw(conv_w, 2, h * 128 + vr)
        st = hl.load(
            rec_state, [sl + z2, h + z2, vr[:, None], d[None, :]], extra_mask=(z2 + ok) != 0
        )  # [16 v, 128 k]
        beta = torch.sigmoid(braw.to(torch.bfloat16).to(torch.float32))  # [8, 1]
        gate = hl.dot(fa.to(torch.bfloat16), fb.T)  # [8, 128] fp32
        graw = gate.to(torch.bfloat16).to(torch.float32) + dtb
        logd = -5.0 * torch.sigmoid(torch.exp(al) * graw)  # [8, 128]
        hl.store(rg, [sl + zt, h + zt, tt[:, None], d[None, :]], logd, extra_mask=mt & mine[None, :])
        hl.store(rbeta, [sl + zt, h + zt, tt[:, None]], beta, extra_mask=mt & ((zt + v0) == 0))
        hl.store(sdec, [task + zt, tt[:, None], d[None, :]], torch.exp(logd))
        hl.store(sbeta, [task + zt, tt[:, None]], beta)
        cq = h * 128 + d
        q_0 = _tap(pq, conv_state, sl, ok, cq, 0)
        q_1 = _tap(pq, conv_state, sl, ok, cq, 1)
        q_2 = _tap(pq, conv_state, sl, ok, cq, 2)
        q_3 = _tap(pq, conv_state, sl, ok, cq, 3)
        k_0 = _tap(pq, conv_state, sl, ok, 1536 + cq, 0)
        k_1 = _tap(pq, conv_state, sl, ok, 1536 + cq, 1)
        k_2 = _tap(pq, conv_state, sl, ok, 1536 + cq, 2)
        k_3 = _tap(pq, conv_state, sl, ok, 1536 + cq, 3)
        cv = 3072 + h * 128 + vr
        v_0 = _tap(pq, conv_state, sl, ok, cv, 0)
        v_1 = _tap(pq, conv_state, sl, ok, cv, 1)
        v_2 = _tap(pq, conv_state, sl, ok, cv, 2)
        v_3 = _tap(pq, conv_state, sl, ok, cv, 3)
        j16 = hl.arange(16)
        _win3(iconv, scr, ok, cq, q_0, q_1, q_2, mine)
        _win3(iconv, scr, ok, 1536 + cq, k_0, k_1, k_2, mine)
        _win3(iconv, scr, ok, cv, v_0, v_1, v_2, vr >= 0)
        yq = _convy(q_3, q_2, q_1, q_0, wq0, wq1, wq2, wq3)
        yk = _convy(k_3, k_2, k_1, k_0, wk0, wk1, wk2, wk3)
        yv = _convy(v_3, v_2, v_1, v_0, wv0, wv1, wv2, wv3)  # [8, 16]
        hl.store(rawk, [sl + zt, h + zt, tt[:, None], d[None, :]], yk.to(torch.bfloat16), extra_mask=mt & mine[None, :])
        hl.store(rawv, [sl + zt, h + zt, tt[:, None], vr[None, :]], yv.to(torch.bfloat16), extra_mask=mt)
        hl.store(stg, [task + zt, zt, tt[:, None], d[None, :]], yq * (torch.rsqrt(torch.sum(yq * yq, dim=1) + 1e-6) * 0.08838834764831845)[:, None])
        hl.store(stg, [task + zt, zt + 1, tt[:, None], d[None, :]], yk * torch.rsqrt(torch.sum(yk * yk, dim=1) + 1e-6)[:, None])
        hl.store(stg, [task + zt, zt + 2, tt[:, None], (vr - v0)[None, :]], yv)
        hl.inline_triton("tl.debug_barrier()", args=(), output_like=None)
        q0, k0, e0, v0_, b0 = _token(stg, sdec, sbeta, task, d, 0)
        q1, k1, e1, v1, b1 = _token(stg, sdec, sbeta, task, d, 1)
        q2, k2, e2, v2, b2 = _token(stg, sdec, sbeta, task, d, 2)
        q3, k3, e3, v3, b3 = _token(stg, sdec, sbeta, task, d, 3)
        q4, k4, e4, v4, b4 = _token(stg, sdec, sbeta, task, d, 4)
        q5, k5, e5, v5, b5 = _token(stg, sdec, sbeta, task, d, 5)
        q6, k6, e6, v6, b6 = _token(stg, sdec, sbeta, task, d, 6)
        q7, k7, e7, v7, b7 = _token(stg, sdec, sbeta, task, d, 7)
        st, o0 = _step(st, q0, k0, e0, v0_, b0)
        st, o1 = _step(st, q1, k1, e1, v1, b1)
        st, o2 = _step(st, q2, k2, e2, v2, b2)
        st, o3 = _step(st, q3, k3, e3, v3, b3)
        st, o4 = _step(st, q4, k4, e4, v4, b4)
        st, o5 = _step(st, q5, k5, e5, v5, b5)
        st, o6 = _step(st, q6, k6, e6, v6, b6)
        st, o7 = _step(st, q7, k7, e7, v7, b7)
        hl.store(opre, [0 + z2, h * 128 + vr[:, None]], o0)
        hl.store(opre, [1 + z2, h * 128 + vr[:, None]], o1)
        hl.store(opre, [2 + z2, h * 128 + vr[:, None]], o2)
        hl.store(opre, [3 + z2, h * 128 + vr[:, None]], o3)
        hl.store(opre, [4 + z2, h * 128 + vr[:, None]], o4)
        hl.store(opre, [5 + z2, h * 128 + vr[:, None]], o5)
        hl.store(opre, [6 + z2, h * 128 + vr[:, None]], o6)
        hl.store(opre, [7 + z2, h * 128 + vr[:, None]], o7)
    # 2d. per head: gated RMSNorm -> bf16 operand of the output projection
    for tile_h in hl.tile(12, block_size=1):
        h = tile_h.begin
        tt = hl.arange(8)
        d = hl.arange(128)
        slot = hl.load(state_idx, [0])
        # the gate first: g proj finishes long before the recurrence (a task
        # waits for what it reads in the order it reads it)
        g = torch.sigmoid(
            hl.load(pg, [tt[:, None], h * 128 + d[None, :]]).to(torch.bfloat16).to(torch.float32)
        )
        o = hl.load(opre, [tt[:, None], h * 128 + d[None, :]])
        nw = hl.load(norm_w, [tt[:, None] * 0 + d[None, :]])  # [8, 128]: no broadcast conversion
        rinv = torch.rsqrt(torch.sum(o * o, dim=1) * (1.0 / 128) + 1e-5)
        on = o * rinv[:, None] * nw * g
        hl.store(ob, [tt[:, None], h * 128 + d[None, :]], torch.where((tt[:, None] * 0 + slot) >= 0, on, 0.0).to(torch.bfloat16))
    # 3. output projection; task 0 re-zeroes the f_a/beta sums (every head read them)
    for tile_m, tile_n in hl.tile([7168, 8], block_size=[64, 16]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(1536, block_size=128):
            acc = hl.dot(w_out[tile_m, tile_k], ob[tile_n, tile_k].T, acc=acc)
        out[tile_n, tile_m] = acc.T.to(torch.bfloat16)
        # re-zero the projection sums for the next launch (64 columns per task)
        zr = tile_m.begin + hl.arange(64)
        t8 = hl.arange(8)
        z = hl.zeros([8, 64], dtype=torch.float32)
        hl.store(pq, [t8[:, None], zr[None, :]], z, extra_mask=(zr < 4608)[None, :])
        hl.store(pg, [t8[:, None], zr[None, :] - 4608], z, extra_mask=((zr >= 4608) & (zr < 6144))[None, :])
        hl.store(pfb, [t8[:, None], zr[None, :] - 6144], z, extra_mask=((zr >= 6144) & (zr < 6400))[None, :])


RING_LOADS = frozenset(["w_qkv", "w_fa", "w_g", "hid4", "hid8", "w_out", "ob"])


def _k_loops(spec) -> list[bool | None]:
    groups = [list(g.block_ids) for g in spec.range_warp_specialize]
    return [
        True if len(g) == 1 and i > 0 and len(groups[i - 1]) >= 2 else None
        for i, g in enumerate(groups)
    ]


def mk_config(bound, **overrides) -> helion.Config:
    """The measured config: ring loads of ``RING_LOADS`` through TMA, the rest
    plain pointers; the K loops of the dot roots warp-specialized."""
    spec = bound.config_spec
    n = len(spec.default_config().config["indexing"])
    td = {
        f.indexing_index
        for f in spec.memory_op_facts
        if f.kind == "load" and f.tensor_name in RING_LOADS
    }
    cfg = dict(
        indexing=["tensor_descriptor" if i in td else "pointer" for i in range(n)],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=8,
        num_warps=8,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=_k_loops(spec),
    )
    cfg.update(overrides)
    return helion.Config(**cfg)


_SCRATCH: dict = {}


def scratch(device="cuda"):
    if not _SCRATCH:
        _SCRATCH["pq"] = torch.zeros(T, 4608, device=device)
        _SCRATCH["pfb"] = torch.zeros(T, 256, device=device)
        _SCRATCH["pg"] = torch.zeros(T, 1536, device=device)
        _SCRATCH["stg"] = torch.zeros(96, 3, T, D, device=device)
        _SCRATCH["sdec"] = torch.zeros(96, T, D, device=device)
        _SCRATCH["sbeta"] = torch.zeros(96, T, device=device)
        _SCRATCH["opre"] = torch.zeros(T, S, device=device)
        _SCRATCH["ob"] = torch.zeros(T, S, device=device, dtype=torch.bfloat16)
    return _SCRATCH


def kernel_args(t: dict, out: torch.Tensor):
    sc = scratch()
    hs, w = t["hidden_states"], t["input_weight"]
    return (
        hs.view(8, 4, 1792),
        w[:4608].view(4608, 4, 1792),
        hs.view(8, 8, 896),
        w[6144:6272].view(128, 8, 896),
        hs.view(8, 8, 896)[:],
        w[6272:6288].view(16, 8, 896),
        w[4608:6144].view(1536, 4, 1792),
        t["f_b_weight"],
        t["conv_weight"].contiguous(),  # (weight layout, prepared once: taps-major rows)
        t["a_log"],
        t["dt_bias"],
        t["conv_state"],
        t["recurrent_state"],
        t["state_indices"],
        t["intermediate_conv"],
        t["intermediate_indices"],
        t["norm_weight"],
        t["output_weight"],
        t["replayssm_rawv"],
        t["replayssm_rawk"],
        t["replayssm_g"],
        t["replayssm_beta"],
        sc["pq"],
        sc["pfb"],
        sc["pg"],
        sc["stg"],
        sc["sdec"],
        sc["sbeta"],
        sc["opre"],
        sc["ob"],
        out,
    )


def compile_kernel(args: tuple) -> object:
    """``kda_mtp`` bound to ``args`` and compiled with ``mk_config``."""
    bound = kda_mtp.bind(args)
    return bound.compile_config(mk_config(bound))


def correctness_check(seed: int = 0) -> None:
    """One verify step against the FP32 PyTorch reference: the output and every
    ReplaySSM cache write (``_reference.assert_correct``)."""
    from pretuned_kernels.megakernels.kda_mtp import _reference as ref

    base = ref.make_inputs(1, H, T, cache_ring=True, seed=seed)
    candidate = ref.clone_outputs(base)
    out = torch.empty(T, 7168, device="cuda", dtype=torch.bfloat16)
    args = kernel_args(candidate, out)
    compile_kernel(args)(*args)
    torch.cuda.synchronize()
    reference = ref.clone_outputs(base)
    expected = ref.torch_reference(reference, T, cache_ring=True)
    print(ref.assert_correct(out, candidate, expected, reference))


def main(rounds: int = 3) -> None:
    """Read-flush single CUDA-graph replay and steady state (20 x flush + call
    in one graph, minus the flushes), against SGLang's verify path when SGLang
    is installed."""
    from pretuned_kernels._bench import capture_cuda_graph
    from pretuned_kernels.megakernels.kda_mtp import _cold as cold
    from pretuned_kernels.megakernels.kda_mtp import _reference as ref

    base = ref.make_inputs(1, H, T, cache_ring=True, seed=0)
    candidate = ref.clone_outputs(base)
    out = torch.empty(T, 7168, device="cuda", dtype=torch.bfloat16)
    args = kernel_args(candidate, out)
    fn = compile_kernel(args)
    calls = {"helion": lambda: fn(*args)}
    try:
        calls["sglang"] = ref.make_sglang_call(
            ref.clone_outputs(base), T, side_stream=torch.cuda.Stream()
        )
    except ImportError:
        print("SGLang not installed: Helion alone")
    with torch.inference_mode():
        graphs = {name: capture_cuda_graph(call)[0] for name, call in calls.items()}
        for _ in range(rounds):
            single = cold.bench_clean([g.replay for g in graphs.values()], rep=400)
            steady, _ = cold.bench_steady(list(calls.values()), 20)
            print(
                "read-flush single replay: "
                + "  ".join(f"{n} {v:.2f} us" for n, v in zip(graphs, single))
                + "   steady: "
                + "  ".join(f"{n} {v:.2f} us" for n, v in zip(calls, steady))
            )


if __name__ == "__main__":
    correctness_check()
    main()
