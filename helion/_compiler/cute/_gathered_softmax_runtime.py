"""Warp-native BF16 page transitions and bounded FP32 split-weight reductions."""

from __future__ import annotations

from typing import cast

import cutlass
import cutlass.cute as cute

from . import _gathered_panel_runtime as panel
from ._flash_row_mma_runtime import mma_bf16
from ._flash_row_mma_runtime import pack_bf16x2
from .reduce_helpers import _cute_nan_max
from .reduce_helpers import _warp_reduce_max
from .reduce_helpers import _warp_reduce_sum


@cute.jit
def _pair(ptr: cute.Pointer, index: cute.Numeric, step: cute.Numeric) -> cutlass.Uint32:
    return pack_bf16x2(
        cutlass.Float32((ptr + index).load()),
        cutlass.Float32((ptr + index + step).load()),
    )


@cute.jit
def _qk_pair(
    ptr: cute.Pointer,
    index: cute.Numeric,
    step: cute.Numeric,
    packed: cutlass.Constexpr,
) -> cutlass.Uint32:
    # MMA expects the increasing-K BF16 pair in low/high order. A naturally
    # aligned unit-stride word has that representation already.
    if cutlass.const_expr(packed):
        return cutlass.Uint32(cute.recast_ptr(ptr + index, dtype=cutlass.Uint32).load())
    return _pair(ptr, index, step)


@cute.jit
def _key_pair(
    ptr: cute.Pointer,
    index: cute.Numeric,
    step: cute.Numeric,
    valid: cutlass.Boolean,
    packed: cutlass.Constexpr,
) -> cutlass.Uint32:
    pair = cutlass.Uint32(0)
    if valid:
        pair = _qk_pair(ptr, index, step, packed)
    return pair


@cute.jit
def _zero_pair(
    ptr: cute.Pointer, index: cute.Numeric, step: cute.Numeric, valid: cutlass.Boolean
) -> cutlass.Uint32:
    pair = cutlass.Uint32(0)
    if valid:
        pair = _pair(ptr, index, step)
    return pair


@cute.jit
def gathered_softmax_page(
    q: cute.Tensor,
    keys: cute.Tensor,
    vals: cute.Tensor,
    limit: cute.Tensor,
    prev_m: cute.Tensor,
    prev_l: cute.Tensor,
    prev_a: cute.Tensor,
    out_m: cute.Tensor,
    out_l: cute.Tensor,
    out_a: cute.Tensor,
    probs: cute.Tensor,
    scratch: cute.Tensor,
    tid: cute.Numeric,
    physical: cute.Numeric,
    logical: cute.Numeric,
    dim0: cute.Numeric,
    scale: cute.Numeric,
    nkeys: cute.Numeric,
    nvals: cute.Numeric,
    vwidth: cute.Numeric,
    qr: cute.Numeric,
    qk: cute.Numeric,
    kp: cute.Numeric,
    kr: cute.Numeric,
    kk: cute.Numeric,
    vp: cute.Numeric,
    vr: cute.Numeric,
    vd: cute.Numeric,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    D: cutlass.Constexpr,
    W: cutlass.Constexpr,
    A_IN: cutlass.Constexpr,
    A_OUT: cutlass.Constexpr,
    P_ROW: cutlass.Constexpr,
    PACKED_Q: cutlass.Constexpr = False,
    PACKED_K: cutlass.Constexpr = False,
    panel_storage: cute.Tensor | None = None,
    panel_stages: cutlass.Constexpr = 2,
    fast_math: cutlass.Constexpr = False,
) -> None:
    warp = tid // 32
    lane = tid % 32
    g = lane // 4
    t = lane % 4
    # The unmasked K and V lookups each wrap once against their own bank.
    raw = cutlass.Int64(physical)
    k_index = raw + cutlass.Int64(nkeys) if raw < 0 else raw
    v_index = raw + cutlass.Int64(nvals) if raw < 0 else raw
    valid_k = (k_index >= 0) & (k_index < nkeys)
    valid_v = (v_index >= 0) & (v_index < nvals)
    # Pointer arithmetic is 64-bit, including for dynamically strided views.
    q_row = cutlass.Int64(qr)
    q_col = cutlass.Int64(qk)
    k_page = k_index * cutlass.Int64(kp)
    k_row = cutlass.Int64(kr)
    k_col = cutlass.Int64(kk)
    v_page = v_index * cutlass.Int64(vp)
    v_row = cutlass.Int64(vr)
    v_col = cutlass.Int64(vd)
    slab = cutlass.Int32(0)
    if cutlass.const_expr(panel_storage is not None):
        assert panel_storage is not None
        slab = (
            cutlass.Int32(cast("cute.Pointer", panel_storage.iterator).toint()) + 127
        ) & -128
    for group in cutlass.range_constexpr(M // 16):
        h0 = group * 16 + g
        h1 = h0 + 8
        a_lim = limit[h0]
        b_lim = limit[h1]
        scores = cute.make_rmem_tensor((N // (8 * W) * 4,), cutlass.Float32)
        scores.fill(cutlass.Float32(0))
        if cutlass.const_expr(panel_storage is not None):
            panel.query_key(
                scores,
                q,
                keys,
                k_index,
                kp,
                q_row,
                k_row,
                group,
                slab,
                tid,
                valid_k,
                K,
                panel_stages,
            )
            panel.issue_values(
                slab, vals, v_index, vp, v_row, dim0, tid, valid_v, vwidth
            )
        else:
            for ks in range(K // 16):
                k = ks * 16 + 2 * t
                a0 = _qk_pair(q.iterator, h0 * q_row + k * q_col, q_col, PACKED_Q)
                a1 = _qk_pair(q.iterator, h1 * q_row + k * q_col, q_col, PACKED_Q)
                a2 = _qk_pair(q.iterator, h0 * q_row + (k + 8) * q_col, q_col, PACKED_Q)
                a3 = _qk_pair(q.iterator, h1 * q_row + (k + 8) * q_col, q_col, PACKED_Q)
                for nt in cutlass.range_constexpr(N // (8 * W)):
                    base = k_page + (8 * (nt * W + warp) + g) * k_row
                    b0 = _key_pair(
                        keys.iterator, base + k * k_col, k_col, valid_k, PACKED_K
                    )
                    b1 = _key_pair(
                        keys.iterator, base + (k + 8) * k_col, k_col, valid_k, PACKED_K
                    )
                    i = 4 * nt
                    s0, s1, s2, s3 = mma_bf16(
                        a0,
                        a1,
                        a2,
                        a3,
                        b0,
                        b1,
                        scores[i],
                        scores[i + 1],
                        scores[i + 2],
                        scores[i + 3],
                    )
                    scores[i], scores[i + 1], scores[i + 2], scores[i + 3] = (
                        s0,
                        s1,
                        s2,
                        s3,
                    )
        top = cutlass.Float32(float("-inf"))
        bottom = cutlass.Float32(float("-inf"))
        for nt in cutlass.range_constexpr(N // (8 * W)):
            i = 4 * nt
            p0 = cutlass.Int32(logical * N + 8 * (nt * W + warp) + 2 * t)
            p1 = cutlass.Int32(p0 + 1)
            scores[i] = (
                scores[i] * scale if p0 < a_lim else cutlass.Float32(float("-inf"))
            )
            scores[i + 1] = (
                scores[i + 1] * scale if p1 < a_lim else cutlass.Float32(float("-inf"))
            )
            scores[i + 2] = (
                scores[i + 2] * scale if p0 < b_lim else cutlass.Float32(float("-inf"))
            )
            scores[i + 3] = (
                scores[i + 3] * scale if p1 < b_lim else cutlass.Float32(float("-inf"))
            )
            top = _cute_nan_max(top, _cute_nan_max(scores[i], scores[i + 1]))
            bottom = _cute_nan_max(bottom, _cute_nan_max(scores[i + 2], scores[i + 3]))
        top = _warp_reduce_max(top, threads_in_group=4)
        bottom = _warp_reduce_max(bottom, threads_in_group=4)
        if t == 0:
            scratch[warp * 16 + g] = top
            scratch[warp * 16 + g + 8] = bottom
        cute.arch.sync_threads()
        if tid < 16:
            h = 16 * group + tid
            incoming_max = cutlass.Float32(prev_m[h])
            mm = incoming_max
            for w in cutlass.range_constexpr(W):
                mm = _cute_nan_max(mm, cutlass.Float32(scratch[16 * w + tid]))
            safe = cutlass.Float32(0) if mm == cutlass.Float32(float("-inf")) else mm
            out_m[h] = mm
            scratch[16 * W + tid] = safe
            scratch[16 * (W + 1) + tid] = cute.math.exp2(
                incoming_max - safe, fastmath=fast_math
            )
        cute.arch.sync_threads()
        m0, m1 = scratch[16 * W + g], scratch[16 * W + g + 8]
        sum0 = cutlass.Float32(0)
        sum1 = cutlass.Float32(0)
        for nt in cutlass.range_constexpr(N // (8 * W)):
            i = 4 * nt
            col = 8 * (nt * W + warp) + 2 * t
            p0, p1 = (
                cute.math.exp2(scores[i] - m0, fastmath=fast_math),
                cute.math.exp2(scores[i + 1] - m0, fastmath=fast_math),
            )
            p2, p3 = (
                cute.math.exp2(scores[i + 2] - m1, fastmath=fast_math),
                cute.math.exp2(scores[i + 3] - m1, fastmath=fast_math),
            )
            sum0 = sum0 + (p0 + p1)
            sum1 = sum1 + (p2 + p3)
            # Preserve the graph's round-to-nearest conversion immediately
            # before operand A of PV. The denominator consumed FP32 above.
            probs[g * P_ROW + col], probs[g * P_ROW + col + 1] = (
                cutlass.BFloat16(p0),
                cutlass.BFloat16(p1),
            )
            probs[(g + 8) * P_ROW + col], probs[(g + 8) * P_ROW + col + 1] = (
                cutlass.BFloat16(p2),
                cutlass.BFloat16(p3),
            )
        sum0 = _warp_reduce_sum(sum0, threads_in_group=4)
        sum1 = _warp_reduce_sum(sum1, threads_in_group=4)
        if t == 0:
            scratch[16 * warp + g] = sum0
            scratch[16 * warp + g + 8] = sum1
        cute.arch.sync_threads()
        if tid < 16:
            total = cutlass.Float32(0)
            for w in cutlass.range_constexpr(W):
                total = total + scratch[16 * w + tid]
            h = group * 16 + tid
            out_l[h] = (
                cutlass.Float32(prev_l[h])
                * cutlass.Float32(scratch[16 * (W + 1) + tid])
                + total
            )
        if cutlass.const_expr(panel_storage is not None):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        alpha0 = cutlass.Float32(scratch[16 * (W + 1) + g])
        alpha1 = cutlass.Float32(scratch[16 * (W + 1) + g + 8])
        for nc in cutlass.range_constexpr(D // (8 * W)):
            base = 8 * (nc * W + warp)
            col = base + 2 * t
            c0, c1 = (
                prev_a[h0 * A_IN + col] * alpha0,
                prev_a[h0 * A_IN + col + 1] * alpha0,
            )
            c2, c3 = (
                prev_a[h1 * A_IN + col] * alpha1,
                prev_a[h1 * A_IN + col + 1] * alpha1,
            )
            if cutlass.const_expr(panel_storage is not None):
                c0, c1, c2, c3 = panel.probability_value(
                    slab, probs, tid, nc, P_ROW, c0, c1, c2, c3
                )
            else:
                vindex = cutlass.Int64(dim0) + base + g
                valid_col = valid_v & (vindex >= 0) & (vindex < vwidth)
                for ks in cutlass.range_constexpr(N // 16):
                    k = ks * 16 + 2 * t
                    a0 = _pair(probs.iterator, g * P_ROW + k, 1)
                    a1 = _pair(probs.iterator, (g + 8) * P_ROW + k, 1)
                    a2 = _pair(probs.iterator, g * P_ROW + k + 8, 1)
                    a3 = _pair(probs.iterator, (g + 8) * P_ROW + k + 8, 1)
                    basev = v_page + vindex * v_col
                    b0 = _zero_pair(vals.iterator, basev + k * v_row, v_row, valid_col)
                    b1 = _zero_pair(
                        vals.iterator, basev + (k + 8) * v_row, v_row, valid_col
                    )
                    c0, c1, c2, c3 = mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)
            out_a[h0 * A_OUT + col], out_a[h0 * A_OUT + col + 1] = c0, c1
            out_a[h1 * A_OUT + col], out_a[h1 * A_OUT + col + 1] = c2, c3
        # All users of P and the row scratch must finish before another
        # iteration/row group can recycle either allocation.
        cute.arch.sync_threads()


@cute.jit
def split_weighted_merge(
    lse: cute.Tensor,
    values: cute.Tensor,
    output: cute.Tensor,
    row: cute.Numeric,
    col_base: cute.Numeric,
    live_raw: cute.Numeric,
    fast_math: cutlass.Constexpr = False,
) -> None:
    nsplit = 64  # Two split slots per lane of the first warp.
    x, y, z = cute.arch.thread_idx()
    dx, dy, _dz = cute.arch.block_dim()
    tid = cutlass.Int32(x + dx * (y + dy * z))
    buf = cutlass.utils.SmemAllocator().allocate(4 * (nsplit + 1), 16)
    weights = cute.recast_ptr(buf, dtype=cutlass.Float32)
    denominator = weights + nsplit
    live = cutlass.Int32(live_raw)
    if live < 0:
        live = cutlass.Int32(0)
    if live > nsplit:
        live = cutlass.Int32(nsplit)
    if tid < 32:
        z0 = cutlass.Float32(float("-inf"))
        z1 = cutlass.Float32(float("-inf"))
        if tid < live:
            z0 = lse[cutlass.Int64(tid), cutlass.Int64(row)]
        if tid + 32 < live:
            z1 = lse[cutlass.Int64(tid + 32), cutlass.Int64(row)]
        top = _warp_reduce_max(
            _cute_nan_max(cutlass.Float32(z0), cutlass.Float32(z1)), threads_in_group=32
        )
        if top == cutlass.Float32(float("-inf")):
            top = cutlass.Float32(0)
        a = cutlass.Float32(0)
        b = cutlass.Float32(0)
        if tid < live:
            a = cute.math.exp2(z0 - top, fastmath=fast_math)
        if tid + 32 < live:
            b = cute.math.exp2(z1 - top, fastmath=fast_math)
        (weights + tid).store(a)
        (weights + tid + 32).store(b)
    cute.arch.sync_threads()
    if tid == 0:
        den = cutlass.Float32(0)
        for split in range(cutlass.Int32(0), live):
            den = den + (weights + split).load()
        denominator.store(den)
    cute.arch.sync_threads()
    acc = cutlass.Float32(0)
    col = col_base + tid
    for split in range(cutlass.Int32(0), live):
        acc = (
            acc
            + (weights + split).load()
            * values[cutlass.Int64(split), cutlass.Int64(row), cutlass.Int64(col)]
        )
    den = denominator.load()
    if cutlass.const_expr(fast_math):
        quotient = cute.math.div(acc, den, approx=True, ftz=True)
    else:
        quotient = acc / den
    output[cutlass.Int64(row), cutlass.Int64(col)] = cutlass.BFloat16(quotient)
    cute.arch.sync_threads()
