# pyrefly: ignore-errors
"""Kimi-K3 KDA speculative verification as one Gluon megakernel: the KDA layer of
the TP8 shard (12 heads of 128) for one request of T = 8 draft tokens, writing
the ReplaySSM cache ring (``_reference`` has the PyTorch reference and SGLang's
verify path). The output projection's TP all-reduce is left out.

Roots, in launch order (420 tasks on one persistent CTA per SM, 8-deep TMA ring):
  1. q/k/v projection: [8, 7168] x [4608, 7168]^T in 2 K splits of 3584 over
     64-row tiles (144 tasks, K steps of 128), each split's partial into ``pq``.
  2. f_a projection in 8 K splits of 896 (8 tasks); each task also computes its
     split of the beta projection by a register dot from a plain-load alias of
     the hidden states (``hidp``, off the ring); partials into ``pfa`` and
     ``pbeta``.
  3. output-gate projection in 4 K splits of 1792 (48 tasks), partials into
     ``pg``.
  4. recurrence per (head, 16 value rows) (96 tasks): the decay (the F_B
     projection of f_a), then the causal conv of q, k (all 128 channels) and v
     (the task's 16 rows), the q/k L2 norm and the delta rule over the 8
     tokens in order; per-token conv windows and ReplaySSM writes (k, v, log
     decay, beta).
  5. gated RMSNorm per head (12 tasks), the BF16 operand of
  6. the output projection [8, 1536] x [7168, 1536]^T (112 tasks).

The split-K partials are summed by their readers in split order (no atomics),
so the kernel is deterministic run to run.

A task waits for a producer root at its first read of that root's output, in
source order, so the order of the reads in the source is the order of the
waits: the recurrence reads f_a's partials (and computes the decay) before the
q/k/v projection's, which finishes later, and the norm reads the gate before
the recurrence output. Each of the recurrence's conv taps sums the 2 q/k/v
partials at its shifted rows (the shifted reads of the same rows hit L1), so
there is no scratch round trip before the conv; the conv output goes through
task-private scratch (the compiler places the CTA barriers between) so the
delta rule runs in the state's layout: key vectors [1, 128], value columns
[16, 1]. Measurements and the alternatives tried: README.md.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.kda_decode.kda_decode import EPS
from pretuned_kernels.megakernels.kda_decode.kda_decode import LOWER_BOUND
from pretuned_kernels.megakernels.kda_decode.kda_decode import SCALE
from pretuned_kernels.megakernels.kda_mtp import _cold as cold
from pretuned_kernels.megakernels.kda_mtp import _reference as ref
from pretuned_kernels.megakernels.kda_mtp._common import conv_weights
from pretuned_kernels.megakernels.kda_mtp._common import conv_window
from pretuned_kernels.megakernels.kda_mtp._common import delta_step
from pretuned_kernels.megakernels.kda_mtp._common import silu_conv
from pretuned_kernels.megakernels.kda_mtp._common import token_operands
import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel

D = 128
H = 12
S = H * D
T = 8


def _tap(
    pq: torch.Tensor, col: torch.Tensor, j: int, w: torch.Tensor | None = None
) -> torch.Tensor:
    """Conv input at token t - j [8, C], channels ``col``: the q/k/v projection's
    output at row t - j (its 2 K-split partials summed, rounded to bf16, the
    projection's dtype), or the committed window ``w`` where t < j. The two
    partials are separate [8, C] loads (a [2, 8, C] load summed over its first
    dimension reduces across warps)."""
    tt = hl.arange(8)[:, None]
    zt = tt * 0
    row = torch.clamp(tt - j, min=0)
    live = tt >= j
    p0 = hl.load(pq, [zt, row, col[None, :]], extra_mask=live)
    p1 = hl.load(pq, [zt + 1, row, col[None, :]], extra_mask=live)
    p = (p0 + p1).to(torch.bfloat16).to(torch.float32)
    if w is None:
        return p
    return torch.where(live, p, w)


def _win3(
    iconv: torch.Tensor,
    scr: torch.Tensor,
    ok: torch.Tensor,
    col: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    cmask: torch.Tensor,
) -> None:
    """Token t's conv window (inputs at t-2, t-1, t) -> iconv[scr, t], channels
    where ``cmask``."""
    tt = hl.arange(8)[:, None]
    zt = tt * 0
    m = ((zt + ok) != 0) & cmask[None, :]
    hl.store(
        iconv, [scr + zt, tt, zt, col[None, :]], x2.to(torch.bfloat16), extra_mask=m
    )
    hl.store(
        iconv, [scr + zt, tt, zt + 1, col[None, :]], x1.to(torch.bfloat16), extra_mask=m
    )
    hl.store(
        iconv, [scr + zt, tt, zt + 2, col[None, :]], x0.to(torch.bfloat16), extra_mask=m
    )


@helion.kernel(
    static_shapes=True,
    backend="gluon",
    ignore_warnings=[helion.exc.FastMathUnused],
)
def kda_mtp(
    hid4: torch.Tensor,  # [8, 4, 1792] bf16: hidden_states (4 K splits)
    w_qkv: torch.Tensor,  # [4608, 2, 3584] bf16: input_weight rows q k v
    hid2: torch.Tensor,  # [8, 2, 3584] bf16: hidden_states (2 K splits)
    hid8: torch.Tensor,  # [8, 8, 896] bf16: hidden_states (8 K splits)
    w_fa: torch.Tensor,  # [128, 8, 896] bf16: input_weight rows f_a
    hidp: torch.Tensor,  # [8, 8, 896] bf16: hidden_states, plain loads (off the ring)
    w_beta: torch.Tensor,  # [16, 8, 896] bf16: input_weight rows beta (12), pad (4)
    w_g: torch.Tensor,  # [1536, 4, 1792] bf16: input_weight rows output gate
    f_b: torch.Tensor,  # [1536, 128] bf16
    conv_w: torch.Tensor,  # [3, 4, 1536] fp32 (contiguous)
    a_log: torch.Tensor,  # [12] fp32
    dt_bias: torch.Tensor,  # [1536] fp32
    # P state slots, R scratch rows, W >= 8 window steps, C >= 8 ring positions
    conv_state: torch.Tensor,  # [P, 3, 4608] bf16
    rec_state: torch.Tensor,  # [P, 12, 128, 128] fp32
    state_idx: torch.Tensor,  # [1] int32
    iconv: torch.Tensor,  # [R, W, 3, 4608] bf16
    iconv_idx: torch.Tensor,  # [1] int32
    norm_w: torch.Tensor,  # [128] fp32
    w_out: torch.Tensor,  # [7168, 1536] bf16
    rawv: torch.Tensor,  # [P, 12, C, 128] bf16
    rawk: torch.Tensor,  # [P, 12, C, 128] bf16
    rg: torch.Tensor,  # [P, 12, C, 128] fp32
    rbeta: torch.Tensor,  # [P, 12, C] fp32
    # scratch, fp32 but ob
    pq: torch.Tensor,  # [2, 8, 4608] q/k/v projection's split-K partials
    pfa: torch.Tensor,  # [8, 8, 128] f_a projection's split-K partials
    pbeta: torch.Tensor,  # [8, 8, 16] beta projection's split-K partials
    pg: torch.Tensor,  # [4, 8, 1536] output-gate projection's split-K partials
    stg: torch.Tensor,  # [96, 3, 8, 128] per recurrence task q, k (normalized), v
    sdec: torch.Tensor,  # [96, 8, 128] per recurrence task decay
    sbeta: torch.Tensor,  # [96, 8] per recurrence task beta
    opre: torch.Tensor,  # [8, 1536] core output before the gated RMSNorm
    ob: torch.Tensor,  # [8, 1536] bf16: normalized core
    # output
    out: torch.Tensor,  # [8, 7168] bf16
    # model constants: kda_decode's SCALE, EPS, LOWER_BOUND
    scale: hl.constexpr,
    eps: hl.constexpr,
    lower_bound: hl.constexpr,
) -> None:
    hl.pdl_launch_dependents()
    hl.pdl_wait()
    # 1. q/k/v projection (4608 rows), 2 K splits of 3584 (144 tasks of 64
    # rows), a partial per split. K steps of 128 keep a ring slot's weight
    # tile at 16 KB (the bytes in flight per SM that the 128-row, 4-split
    # tiling had)
    for tile_m, tile_n, tile_s in hl.tile([4608, 8, 2], block_size=[64, 16, 1]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(3584, block_size=128):
            acc = hl.dot(
                w_qkv[tile_m, tile_s.begin, tile_k],
                hid2[tile_n, tile_s.begin, tile_k].T,
                acc=acc,
            )
        pq[tile_s.begin, tile_n, tile_m] = acc.T
    # 2. f_a (128 rows) and beta (12 rows and 4 of padding), 8 K splits of 896
    # (8 tasks)
    for tile_s, tile_m, tile_n in hl.tile([8, 128, 8], block_size=[1, 128, 16]):
        # (plain loads take ~5 us under the q/k/v projection's HBM traffic:
        # both beta operands' loads are issued together, before anything waits)
        t8 = hl.arange(8)
        j16 = hl.arange(16)
        kk = hl.arange(896)
        hx = hl.load(hidp, [t8[:, None], tile_s.begin + t8[:, None] * 0, kk[None, :]])
        wb = hl.load(
            w_beta, [j16[:, None], tile_s.begin + j16[:, None] * 0, kk[None, :]]
        )
        # beta: this split's [8, 896] x [16, 896]^T in registers (off the ring)
        pb = hl.dot(hx, wb.T)  # [8, 16] fp32
        hl.store(pbeta, [tile_s.begin + t8[:, None] * 0, t8[:, None], j16[None, :]], pb)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(896, block_size=64):
            acc = hl.dot(
                w_fa[tile_m, tile_s.begin, tile_k],
                hid8[tile_n, tile_s.begin, tile_k].T,
                acc=acc,
            )
        pfa[tile_s.begin, tile_n, tile_m] = acc.T
    # 3. output-gate projection (1536 rows), 4 K splits of 1792 (48 tasks)
    for tile_m, tile_n, tile_s in hl.tile([1536, 8, 4], block_size=[128, 16, 1]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(1792, block_size=64):
            acc = hl.dot(
                w_g[tile_m, tile_s.begin, tile_k],
                hid4[tile_n, tile_s.begin, tile_k].T,
                acc=acc,
            )
        pg[tile_s.begin, tile_n, tile_m] = acc.T

    # 4. per (head, 16 value rows): the F_B decay and beta, the conv (q, k all
    # channels; v this task's 16), q/k L2 norm, then the delta rule over the 8
    # tokens in order. The operands go through task-private scratch (the
    # compiler places the CTA barriers between) so the chain runs in the
    # state's layout: key vectors [1, 128], value columns [16, 1].
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
        # In int64 through the loaded slot: the scratch addresses' 64-bit
        # arithmetic then needs no per-use widening (24 fewer SASS instructions,
        # about 0.2 us in the steady state on GB300).
        task = h * 8 + tile_v.id + slot * 0
        mine = (d >= v0) & (d < v0 + 16)
        mt = (zt + ok) != 0
        # every global load first: layout conversions (smem round trips with
        # a CTA barrier) between loads would serialize their latencies. The
        # f_a and beta sums (their 8 K-split partials in order) are this
        # task's first reads of a projection: it waits for f_a here and for
        # q/k/v at the first _tap below.
        s8 = hl.arange(8)[:, None, None]
        fa = torch.sum(hl.load(pfa, [s8, tt[None, :, None], d[None, None, :]]), dim=0)
        braw = torch.sum(
            hl.load(pbeta, [s8, tt[None, :, None], h + zt[None, :, :]]), dim=0
        )  # [8, 1]
        fb = hl.load(f_b, [h * 128 + d[:, None], d[None, :]])  # [d, j]
        dtb = hl.load(dt_bias, [h * 128 + zt + d[None, :]])
        al = hl.load(a_log, [h])
        wq0, wq1, wq2, wq3 = conv_weights(conv_w, 0, h * 128 + d)
        wk0, wk1, wk2, wk3 = conv_weights(conv_w, 1, h * 128 + d)
        wv0, wv1, wv2, wv3 = conv_weights(conv_w, 2, h * 128 + vr)
        st = hl.load(
            rec_state,
            [sl + z2, h + z2, vr[:, None], d[None, :]],
            extra_mask=(z2 + ok) != 0,
        )  # [16 v, 128 k]
        beta = torch.sigmoid(braw.to(torch.bfloat16).to(torch.float32))  # [8, 1]
        gate = hl.dot(fa.to(torch.bfloat16), fb.T)  # [8, 128] fp32
        graw = gate.to(torch.bfloat16).to(torch.float32) + dtb
        logd = lower_bound * torch.sigmoid(torch.exp(al) * graw)  # [8, 128]
        hl.store(
            rg,
            [sl + zt, h + zt, tt[:, None], d[None, :]],
            logd,
            extra_mask=mt & mine[None, :],
        )
        hl.store(
            rbeta,
            [sl + zt, h + zt, tt[:, None]],
            beta,
            extra_mask=mt & ((zt + v0) == 0),
        )
        hl.store(sdec, [task + zt, tt[:, None], d[None, :]], torch.exp(logd))
        hl.store(sbeta, [task + zt, tt[:, None]], beta)
        cq = h * 128 + d
        cv = 3072 + h * 128 + vr
        # the committed windows (inputs only), then the conv taps from the
        # projection's partials (the q/k/v wait)
        hq1 = conv_window(conv_state, sl, ok, cq, 1)
        hq2 = conv_window(conv_state, sl, ok, cq, 2)
        hq3 = conv_window(conv_state, sl, ok, cq, 3)
        hk1 = conv_window(conv_state, sl, ok, 1536 + cq, 1)
        hk2 = conv_window(conv_state, sl, ok, 1536 + cq, 2)
        hk3 = conv_window(conv_state, sl, ok, 1536 + cq, 3)
        hv1 = conv_window(conv_state, sl, ok, cv, 1)
        hv2 = conv_window(conv_state, sl, ok, cv, 2)
        hv3 = conv_window(conv_state, sl, ok, cv, 3)
        q_0 = _tap(pq, cq, 0)
        k_0 = _tap(pq, 1536 + cq, 0)
        v_0 = _tap(pq, cv, 0)
        q_1 = _tap(pq, cq, 1, hq1)
        q_2 = _tap(pq, cq, 2, hq2)
        q_3 = _tap(pq, cq, 3, hq3)
        k_1 = _tap(pq, 1536 + cq, 1, hk1)
        k_2 = _tap(pq, 1536 + cq, 2, hk2)
        k_3 = _tap(pq, 1536 + cq, 3, hk3)
        v_1 = _tap(pq, cv, 1, hv1)
        v_2 = _tap(pq, cv, 2, hv2)
        v_3 = _tap(pq, cv, 3, hv3)
        _win3(iconv, scr, ok, cq, q_0, q_1, q_2, mine)
        _win3(iconv, scr, ok, 1536 + cq, k_0, k_1, k_2, mine)
        _win3(iconv, scr, ok, cv, v_0, v_1, v_2, vr >= 0)
        yq = silu_conv(q_3, q_2, q_1, q_0, wq0, wq1, wq2, wq3)
        yk = silu_conv(k_3, k_2, k_1, k_0, wk0, wk1, wk2, wk3)
        yv = silu_conv(v_3, v_2, v_1, v_0, wv0, wv1, wv2, wv3)  # [8, 16]
        hl.store(
            rawk,
            [sl + zt, h + zt, tt[:, None], d[None, :]],
            yk.to(torch.bfloat16),
            extra_mask=mt & mine[None, :],
        )
        hl.store(
            rawv,
            [sl + zt, h + zt, tt[:, None], vr[None, :]],
            yv.to(torch.bfloat16),
            extra_mask=mt,
        )
        hl.store(
            stg,
            [task + zt, zt, tt[:, None], d[None, :]],
            yq * (torch.rsqrt(torch.sum(yq * yq, dim=1) + 1e-6) * scale)[:, None],
        )
        hl.store(
            stg,
            [task + zt, zt + 1, tt[:, None], d[None, :]],
            yk * torch.rsqrt(torch.sum(yk * yk, dim=1) + 1e-6)[:, None],
        )
        hl.store(stg, [task + zt, zt + 2, tt[:, None], (vr - v0)[None, :]], yv)
        q0, k0, e0, v0_, b0 = token_operands(stg, sdec, sbeta, task, d, 0)
        q1, k1, e1, v1, b1 = token_operands(stg, sdec, sbeta, task, d, 1)
        q2, k2, e2, v2, b2 = token_operands(stg, sdec, sbeta, task, d, 2)
        q3, k3, e3, v3, b3 = token_operands(stg, sdec, sbeta, task, d, 3)
        q4, k4, e4, v4, b4 = token_operands(stg, sdec, sbeta, task, d, 4)
        q5, k5, e5, v5, b5 = token_operands(stg, sdec, sbeta, task, d, 5)
        q6, k6, e6, v6, b6 = token_operands(stg, sdec, sbeta, task, d, 6)
        q7, k7, e7, v7, b7 = token_operands(stg, sdec, sbeta, task, d, 7)
        st, o0 = delta_step(st, q0, k0, e0, v0_, b0)
        st, o1 = delta_step(st, q1, k1, e1, v1, b1)
        st, o2 = delta_step(st, q2, k2, e2, v2, b2)
        st, o3 = delta_step(st, q3, k3, e3, v3, b3)
        st, o4 = delta_step(st, q4, k4, e4, v4, b4)
        st, o5 = delta_step(st, q5, k5, e5, v5, b5)
        st, o6 = delta_step(st, q6, k6, e6, v6, b6)
        st, o7 = delta_step(st, q7, k7, e7, v7, b7)
        hl.store(opre, [0 + z2, h * 128 + vr[:, None]], o0)
        hl.store(opre, [1 + z2, h * 128 + vr[:, None]], o1)
        hl.store(opre, [2 + z2, h * 128 + vr[:, None]], o2)
        hl.store(opre, [3 + z2, h * 128 + vr[:, None]], o3)
        hl.store(opre, [4 + z2, h * 128 + vr[:, None]], o4)
        hl.store(opre, [5 + z2, h * 128 + vr[:, None]], o5)
        hl.store(opre, [6 + z2, h * 128 + vr[:, None]], o6)
        hl.store(opre, [7 + z2, h * 128 + vr[:, None]], o7)
    # 5. per head: gated RMSNorm -> bf16 operand of the output projection
    for tile_h in hl.tile(12, block_size=1):
        h = tile_h.begin
        tt = hl.arange(8)
        d = hl.arange(128)
        slot = hl.load(state_idx, [0])
        # the gate first: its projection finishes long before the recurrence
        g = torch.sigmoid(
            torch.sum(
                hl.load(
                    pg,
                    [
                        hl.arange(4)[:, None, None],
                        tt[None, :, None],
                        h * 128 + d[None, None, :],
                    ],
                ),
                dim=0,
            )
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        o = hl.load(opre, [tt[:, None], h * 128 + d[None, :]])
        nw = hl.load(
            norm_w, [tt[:, None] * 0 + d[None, :]]
        )  # [8, 128]: no broadcast conversion
        rinv = torch.rsqrt(torch.sum(o * o, dim=1) * (1.0 / 128) + eps)
        on = o * rinv[:, None] * nw * g
        hl.store(
            ob,
            [tt[:, None], h * 128 + d[None, :]],
            torch.where((tt[:, None] * 0 + slot) >= 0, on, 0.0).to(torch.bfloat16),
        )
    # 6. output projection
    for tile_m, tile_n in hl.tile([7168, 8], block_size=[64, 16]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(1536, block_size=128):
            acc = hl.dot(w_out[tile_m, tile_k], ob[tile_n, tile_k].T, acc=acc)
        out[tile_n, tile_m] = acc.T.to(torch.bfloat16)


RING_LOADS = frozenset(["w_qkv", "hid2", "w_fa", "w_g", "hid4", "hid8", "w_out", "ob"])
# Each root's task loop in source order, followed by its K loop if it has one:
# q/k/v, f_a and gate projections, recurrence, norm, output projection.
WARP_SPECIALIZED_K_LOOPS = [None, True, None, True, None, True, None, None, None, True]


def mk_config(bound: BoundKernel) -> helion.Config:
    """The measured config: the loads of ``RING_LOADS`` through the TMA ring (8
    deep), every other access by pointer; one persistent CTA per SM, claiming
    the tasks root by root; the projections' K loops warp-specialized."""
    return helion.Config(
        indexing=ring_config_fields(bound, RING_LOADS)["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=8,
        num_warps=8,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=WARP_SPECIALIZED_K_LOOPS,
    )


@functools.cache
def scratch(device: torch.device) -> dict[str, torch.Tensor]:
    """The kernel's scratch buffers on ``device``, one set for every call."""
    f32 = torch.float32
    return {
        "pq": torch.zeros(2, T, 3 * S, device=device, dtype=f32),
        "pfa": torch.zeros(8, T, D, device=device, dtype=f32),
        "pbeta": torch.zeros(8, T, 16, device=device, dtype=f32),
        "pg": torch.zeros(4, T, S, device=device, dtype=f32),
        "stg": torch.zeros(96, 3, T, D, device=device, dtype=f32),
        "sdec": torch.zeros(96, T, D, device=device, dtype=f32),
        "sbeta": torch.zeros(96, T, device=device, dtype=f32),
        "opre": torch.zeros(T, S, device=device, dtype=f32),
        "ob": torch.zeros(T, S, device=device, dtype=torch.bfloat16),
    }


def kernel_args(t: dict, out: torch.Tensor) -> tuple[object, ...]:
    """``kda_mtp``'s arguments for the tensors ``t`` of
    ``_reference.make_inputs(1, H, T)`` and the output ``out``
    [T, 7168]: K-split views of the hidden states and of the input projection's
    row blocks, the module's scratch buffers, the model constants."""
    sc = scratch(out.device)
    hs, w = t["hidden_states"], t["input_weight"]
    return (
        hs.view(8, 4, 1792),
        w[:4608].view(4608, 2, 3584),
        hs.view(8, 2, 3584),
        hs.view(8, 8, 896),
        w[6144:6272].view(128, 8, 896),
        hs.view(8, 8, 896),
        w[6272:6288].view(16, 8, 896),
        w[4608:6144].view(1536, 4, 1792),
        t["f_b_weight"],
        # taps-major rows; the model's weight is channel-major (prepared once)
        t["conv_weight"].contiguous(),
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
        sc["pfa"],
        sc["pbeta"],
        sc["pg"],
        sc["stg"],
        sc["sdec"],
        sc["sbeta"],
        sc["opre"],
        sc["ob"],
        out,
        SCALE,
        EPS,
        LOWER_BOUND,
    )


def compile_kernel(args: tuple) -> object:
    """``kda_mtp`` bound to ``args`` and compiled with ``mk_config``."""
    bound = kda_mtp.bind(args)
    return bound.compile_config(mk_config(bound))


def prepare(base: dict) -> tuple[dict, torch.Tensor, Callable[[], object]]:
    """The compiled kernel's call on ``base`` with private copies of the buffers
    it writes (``_reference.clone_outputs``): those copies, its output, the
    call."""
    candidate = ref.clone_outputs(base)
    device = base["hidden_states"].device
    out = torch.empty(T, ref.HIDDEN, device=device, dtype=torch.bfloat16)
    args = kernel_args(candidate, out)
    kernel = compile_kernel(args)
    return candidate, out, lambda: kernel(*args)


def correctness_check(seed: int = 0) -> None:
    """One verify step against the FP32 PyTorch reference
    (``_reference.assert_correct``: the output and every cache write), for a
    request and for a padding slot (state index -1: a zero output, no cache
    writes). Then the step again, from the inputs restored and the scratch and
    output filled with NaN, must give the same bits (deterministic, and every
    value read is written first)."""
    for padded in (False, True):
        base = ref.make_inputs(1, H, T, seed=seed, padded=padded)
        candidate, out, call = prepare(base)
        call()
        reference = ref.clone_outputs(base)
        expected = ref.torch_reference(reference, T)
        print(ref.assert_correct(out, candidate, expected, reference))
        names = ref.WRITES
        first = [out.clone(), *(candidate[name].clone() for name in names)]
        for name in names:
            candidate[name].copy_(base[name])
        for buffer in [out, *scratch(out.device).values()]:
            buffer.fill_(float("nan"))
        call()
        again = [out, *(candidate[name] for name in names)]
        for before, after in zip(first, again, strict=True):
            torch.testing.assert_close(
                after, before, atol=0, rtol=0, msg="a repeated step differs"
            )


def main(rounds: int = 3) -> None:
    """Time the kernel against SGLang's verify path (alone without SGLang), as
    ``_cold.compare`` does."""
    base = ref.make_inputs(1, H, T)
    calls = {"helion": prepare(base)[2]}
    try:
        calls["sglang"] = ref.make_sglang_call(
            ref.clone_outputs(base), T, side_stream=torch.cuda.Stream()
        )
    except ImportError:
        print("SGLang is not installed: timing Helion alone")
    cold.compare(calls, rounds)


if __name__ == "__main__":
    correctness_check()
    main()
