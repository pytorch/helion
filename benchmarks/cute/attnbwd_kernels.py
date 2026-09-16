"""Helion attention-backward kernels for the attnbwd hillclimb.

Backend-agnostic definitions (no pinned configs) so each backend autotunes
its own configuration. Two designs:

- split (default): dKdV kernel (KV-major) + dQ kernel (Q-major), no
  atomics, grads written once in the input dtype. This is the shape
  FlexAttention's backward uses.
- fused: single KV-major kernel accumulating dQ into an fp32 buffer with
  atomics (the examples/blackwell_attention.py design). The CuTe backend
  cannot lower tile atomics yet, so this is triton-only for now.

Conventions:
- Kernels take 3-D/4-D group views: q/do [G, MM, D] (or [G, group, M, D]
  for causal), k/v [G, N, D], lse/delta matching q's leading dims, where
  G = B*H_kv and MM = group*M (group = H_q // H_kv; 1 without GQA).
- K is raw (not pre-scaled); the score scale sm_scale is folded into the
  exp2 argument (qk * sm_scale * RCP_LN2 - lse2).
- lse is the base-2 log-sum-exp of (q @ k^T * sm_scale) rows.
- delta = rowsum(o.float() * do.float()) from attention_bwd_preprocess.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

RCP_LN2 = 1.4426950408889634


@helion.kernel(static_shapes=True)
def attention_bwd_preprocess(o_in: torch.Tensor, do_in: torch.Tensor) -> torch.Tensor:
    """delta = rowsum(o * do) in fp32, shape = o_in shape minus last dim."""
    head_dim = hl.specialize(o_in.size(-1))
    o = o_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    total = o.size(0)
    delta = torch.empty(total, device=o.device, dtype=torch.float32)
    for tile in hl.tile(total):
        delta[tile] = torch.sum(
            o[tile, :].to(torch.float32) * do[tile, :].to(torch.float32), dim=-1
        )
    return delta.reshape(o_in.size()[:-1])


@helion.kernel(static_shapes=True, autotune_accuracy_check=False)
def attention_bwd_dkdv(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dK/dV half of the split backward (non-causal)."""
    mm_dim = q_in.size(1)
    n_dim = k_in.size(1)
    head_dim = hl.specialize(q_in.size(-1))
    assert k_in.size(-1) == head_dim and v_in.size(-1) == head_dim
    qk_scale2 = sm_scale * RCP_LN2
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_kv_rows = k.size(0)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(mm_dim)
    block_n = hl.register_block_size(n_dim)
    assert mm_dim % block_m == 0 and n_dim % block_n == 0
    for tile_n in hl.tile(total_kv_rows, block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        dk_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        start_m = tile_n.begin // n_dim * mm_dim
        end_m = start_m + mm_dim
        for tile_m in hl.tile(start_m, end_m, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            m_i = lse[tile_m]
            di = delta[tile_m]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            p_t = torch.exp2(qk_t * qk_scale2 - m_i[None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - di[None, :])).to(q.dtype)
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
        dv[tile_n, :] = dv_acc.to(v.dtype)
        dk[tile_n, :] = (dk_acc * sm_scale).to(k.dtype)
    return dk.reshape(k_in.size()), dv.reshape(v_in.size())


@helion.kernel(static_shapes=True, autotune_accuracy_check=False)
def attention_bwd_dq(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """dQ half of the split backward (non-causal)."""
    mm_dim = q_in.size(1)
    n_dim = k_in.size(1)
    head_dim = hl.specialize(q_in.size(-1))
    assert k_in.size(-1) == head_dim and v_in.size(-1) == head_dim
    qk_scale2 = sm_scale * RCP_LN2
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_q_rows = q.size(0)
    dq = torch.empty_like(q)
    block_m = hl.register_block_size(mm_dim)
    block_n = hl.register_block_size(n_dim)
    assert mm_dim % block_m == 0 and n_dim % block_n == 0
    for tile_m in hl.tile(total_q_rows, block_size=block_m):
        q_i = q[tile_m, :]
        do_i = do[tile_m, :]
        m_i = lse[tile_m]
        di = delta[tile_m]
        dq_acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        start_n = tile_m.begin // mm_dim * n_dim
        end_n = start_n + n_dim
        for tile_n in hl.tile(start_n, end_n, block_size=block_n):
            k_j = k[tile_n, :]
            v_j = v[tile_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            p = torch.exp2(qk * qk_scale2 - m_i[:, None])
            dp = hl.dot(do_i, v_j.T, out_dtype=torch.float32)
            ds = (p * (dp - di[:, None])).to(q.dtype)
            dq_acc = hl.dot(ds, k_j, acc=dq_acc)
        dq[tile_m, :] = (dq_acc * sm_scale).to(q.dtype)
    return dq.reshape(q_in.size())


@helion.kernel(static_shapes=True, autotune_accuracy_check=False)
def attention_bwd_dkdv_causal(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dK/dV half of the split causal backward.

    q_in/do_in: [G, group, M, D]; k_in/v_in: [G, N, D] with M == N. Q rows
    for a KV group are ordered (g, m), so the causal diagonal repeats per
    g; the inner loop starts at the first contributing row of g == 0 and
    masks the rest.
    """
    group = q_in.size(1)
    m_dim = q_in.size(2)
    n_dim = k_in.size(1)
    head_dim = hl.specialize(q_in.size(-1))
    assert k_in.size(-1) == head_dim and v_in.size(-1) == head_dim
    qk_scale2 = sm_scale * RCP_LN2
    mm_dim = group * m_dim
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_kv_rows = k.size(0)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(mm_dim)
    block_n = hl.register_block_size(n_dim)
    assert m_dim % block_m == 0 and n_dim % block_n == 0
    for tile_n in hl.tile(total_kv_rows, block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        dk_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        start_m = tile_n.begin // n_dim * mm_dim
        end_m = start_m + mm_dim
        diag_m = start_m + tile_n.begin % n_dim // block_m * block_m
        for tile_m in hl.tile(diag_m, end_m, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            m_i = lse[tile_m]
            di = delta[tile_m]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            q_pos = tile_m.index % m_dim
            kv_pos = tile_n.index % n_dim
            qk_t = torch.where(q_pos[None, :] >= kv_pos[:, None], qk_t, float("-inf"))
            p_t = torch.exp2(qk_t * qk_scale2 - m_i[None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - di[None, :])).to(q.dtype)
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
        dv[tile_n, :] = dv_acc.to(v.dtype)
        dk[tile_n, :] = (dk_acc * sm_scale).to(k.dtype)
    return dk.reshape(k_in.size()), dv.reshape(v_in.size())


@helion.kernel(static_shapes=True, autotune_accuracy_check=False)
def attention_bwd_dq_causal(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """dQ half of the split causal backward.

    Q tiles never cross a group stripe (m_dim % block_m == 0), so the KV
    loop can stop at the tile's diagonal regardless of the group index.
    """
    group = q_in.size(1)
    m_dim = q_in.size(2)
    n_dim = k_in.size(1)
    head_dim = hl.specialize(q_in.size(-1))
    assert k_in.size(-1) == head_dim and v_in.size(-1) == head_dim
    qk_scale2 = sm_scale * RCP_LN2
    mm_dim = group * m_dim
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_q_rows = q.size(0)
    dq = torch.empty_like(q)
    block_m = hl.register_block_size(mm_dim)
    block_n = hl.register_block_size(n_dim)
    assert m_dim % block_m == 0 and n_dim % block_n == 0
    for tile_m in hl.tile(total_q_rows, block_size=block_m):
        q_i = q[tile_m, :]
        do_i = do[tile_m, :]
        m_i = lse[tile_m]
        di = delta[tile_m]
        dq_acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        start_n = tile_m.begin // mm_dim * n_dim
        end_n = start_n + tile_m.begin % m_dim + block_m
        for tile_n in hl.tile(start_n, end_n, block_size=block_n):
            k_j = k[tile_n, :]
            v_j = v[tile_n, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            q_pos = tile_m.index % m_dim
            kv_pos = tile_n.index % n_dim
            qk = torch.where(q_pos[:, None] >= kv_pos[None, :], qk, float("-inf"))
            p = torch.exp2(qk * qk_scale2 - m_i[:, None])
            dp = hl.dot(do_i, v_j.T, out_dtype=torch.float32)
            ds = (p * (dp - di[:, None])).to(q.dtype)
            dq_acc = hl.dot(ds, k_j, acc=dq_acc)
        dq[tile_m, :] = (dq_acc * sm_scale).to(q.dtype)
    return dq.reshape(q_in.size())


@helion.kernel(static_shapes=True, autotune_accuracy_check=False)
def attention_bwd(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused non-causal backward: dK/dV in registers, dQ via fp32 atomics."""
    mm_dim = q_in.size(1)
    n_dim = k_in.size(1)
    head_dim = hl.specialize(q_in.size(-1))
    assert k_in.size(-1) == head_dim and v_in.size(-1) == head_dim
    qk_scale2 = sm_scale * RCP_LN2
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_kv_rows = k.size(0)
    dq = torch.zeros((q.size(0), head_dim), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(mm_dim)
    block_n = hl.register_block_size(n_dim)
    assert mm_dim % block_m == 0 and n_dim % block_n == 0
    for tile_n in hl.tile(total_kv_rows, block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        dk_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        start_m = tile_n.begin // n_dim * mm_dim
        end_m = start_m + mm_dim
        for tile_m in hl.tile(start_m, end_m, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            m_i = lse[tile_m]
            di = delta[tile_m]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            p_t = torch.exp2(qk_t * qk_scale2 - m_i[None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - di[None, :])).to(q.dtype)
            dq_acc = hl.dot(ds_t.T, k_j, out_dtype=torch.float32)
            hl.atomic_add(dq, [tile_m, slice(None)], dq_acc * sm_scale)
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
        dv[tile_n, :] = dv_acc.to(v.dtype)
        dk[tile_n, :] = (dk_acc * sm_scale).to(k.dtype)
    return (
        dq.reshape(q_in.size()),
        dk.reshape(k_in.size()),
        dv.reshape(v_in.size()),
    )


@helion.kernel(static_shapes=True, autotune_accuracy_check=False)
def attention_bwd_causal(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused causal backward (shapes as attention_bwd_dkdv_causal)."""
    group = q_in.size(1)
    m_dim = q_in.size(2)
    n_dim = k_in.size(1)
    head_dim = hl.specialize(q_in.size(-1))
    assert k_in.size(-1) == head_dim and v_in.size(-1) == head_dim
    qk_scale2 = sm_scale * RCP_LN2
    mm_dim = group * m_dim
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_kv_rows = k.size(0)
    dq = torch.zeros((q.size(0), head_dim), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(mm_dim)
    block_n = hl.register_block_size(n_dim)
    assert m_dim % block_m == 0 and n_dim % block_n == 0
    for tile_n in hl.tile(total_kv_rows, block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        dk_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        start_m = tile_n.begin // n_dim * mm_dim
        end_m = start_m + mm_dim
        diag_m = start_m + tile_n.begin % n_dim // block_m * block_m
        for tile_m in hl.tile(diag_m, end_m, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            m_i = lse[tile_m]
            di = delta[tile_m]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            q_pos = tile_m.index % m_dim
            kv_pos = tile_n.index % n_dim
            qk_t = torch.where(q_pos[None, :] >= kv_pos[:, None], qk_t, float("-inf"))
            p_t = torch.exp2(qk_t * qk_scale2 - m_i[None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - di[None, :])).to(q.dtype)
            dq_acc = hl.dot(ds_t.T, k_j, out_dtype=torch.float32)
            hl.atomic_add(dq, [tile_m, slice(None)], dq_acc * sm_scale)
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
        dv[tile_n, :] = dv_acc.to(v.dtype)
        dk[tile_n, :] = (dk_acc * sm_scale).to(k.dtype)
    return (
        dq.reshape(q_in.size()),
        dk.reshape(k_in.size()),
        dv.reshape(v_in.size()),
    )


def bwd_kernel_calls(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse2: torch.Tensor,
    do: torch.Tensor,
    sm_scale: float,
    causal: bool,
    mode: str = "split",
) -> list[tuple[str, object, tuple[object, ...]]]:
    """(name, kernel, args) for each kernel helion_attention_backward runs.

    Used by the benchmark harness to bind, autotune, and pin configs on the
    exact argument views the wrapper will pass at timing time.
    """
    b, hq, m_dim, head_dim = q.shape
    hkv, n_dim = k.shape[1], k.shape[2]
    group = hq // hkv
    delta = (o.float() * do.float()).sum(-1)
    kg = k.reshape(b * hkv, n_dim, head_dim)
    vg = v.reshape(b * hkv, n_dim, head_dim)
    calls: list[tuple[str, object, tuple[object, ...]]] = [
        ("preprocess", attention_bwd_preprocess, (o, do))
    ]
    if causal:
        qg = q.reshape(b * hkv, group, m_dim, head_dim)
        dog = do.reshape(b * hkv, group, m_dim, head_dim)
        lseg = lse2.reshape(b * hkv, group, m_dim)
        deltag = delta.reshape(b * hkv, group, m_dim)
        base = (qg, kg, vg, lseg, dog, deltag, sm_scale)
        if mode == "split":
            calls.extend(
                [
                    ("dkdv", attention_bwd_dkdv_causal, base),
                    ("dq", attention_bwd_dq_causal, base),
                ]
            )
        else:
            calls.append(("fused", attention_bwd_causal, base))
    else:
        qg = q.reshape(b * hkv, group * m_dim, head_dim)
        dog = do.reshape(b * hkv, group * m_dim, head_dim)
        lseg = lse2.reshape(b * hkv, group * m_dim)
        deltag = delta.reshape(b * hkv, group * m_dim)
        base = (qg, kg, vg, lseg, dog, deltag, sm_scale)
        if mode == "split":
            calls.extend(
                [
                    ("dkdv", attention_bwd_dkdv, base),
                    ("dq", attention_bwd_dq, base),
                ]
            )
        else:
            calls.append(("fused", attention_bwd, base))
    return calls


def helion_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse2: torch.Tensor,
    do: torch.Tensor,
    sm_scale: float,
    causal: bool,
    mode: str = "split",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host wrapper: 4-D [B, H, S, D] tensors in, grads in input dtype out.

    lse2 is the base-2 LSE of shape [B, H_q, M]. Supports GQA (H_q a
    multiple of H_kv). Timed end-to-end by the benchmark harness,
    including the preprocess kernel. mode="split" runs the no-atomics
    dKdV + dQ kernel pair; mode="fused" runs the single kernel with fp32
    atomic dQ accumulation (plus the fp32 -> input-dtype convert).
    """
    b, hq, m_dim, head_dim = q.shape
    hkv, n_dim = k.shape[1], k.shape[2]
    group = hq // hkv
    assert hkv * group == hq
    delta = attention_bwd_preprocess(o, do)
    kg = k.reshape(b * hkv, n_dim, head_dim)
    vg = v.reshape(b * hkv, n_dim, head_dim)
    if causal:
        assert m_dim == n_dim
        qg = q.reshape(b * hkv, group, m_dim, head_dim)
        dog = do.reshape(b * hkv, group, m_dim, head_dim)
        lseg = lse2.reshape(b * hkv, group, m_dim)
        deltag = delta.reshape(b * hkv, group, m_dim)
        if mode == "split":
            dk, dv = attention_bwd_dkdv_causal(qg, kg, vg, lseg, dog, deltag, sm_scale)
            dq = attention_bwd_dq_causal(qg, kg, vg, lseg, dog, deltag, sm_scale)
        else:
            dq, dk, dv = attention_bwd_causal(qg, kg, vg, lseg, dog, deltag, sm_scale)
            dq = dq.to(q.dtype)
    else:
        qg = q.reshape(b * hkv, group * m_dim, head_dim)
        dog = do.reshape(b * hkv, group * m_dim, head_dim)
        lseg = lse2.reshape(b * hkv, group * m_dim)
        deltag = delta.reshape(b * hkv, group * m_dim)
        if mode == "split":
            dk, dv = attention_bwd_dkdv(qg, kg, vg, lseg, dog, deltag, sm_scale)
            dq = attention_bwd_dq(qg, kg, vg, lseg, dog, deltag, sm_scale)
        else:
            dq, dk, dv = attention_bwd(qg, kg, vg, lseg, dog, deltag, sm_scale)
            dq = dq.to(q.dtype)
    return (
        dq.reshape(q.shape),
        dk.reshape(k.shape),
        dv.reshape(v.shape),
    )
