"""
Flex Attention Example
========================

This code implements a custom attention kernel using Helion and PyTorch for efficient computation of scaled dot-product attention,
with support for both static and dynamic input shapes.
"""

# %%
# Imports
# -------
from __future__ import annotations

import functools
import math
from typing import Any
from typing import Callable
from typing import cast

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import _create_empty_block_mask
from torch.nn.attention.flex_attention import _identity
from torch.nn.attention.flex_attention import _score_mod_signature
from torch.nn.attention.flex_attention import flex_attention

import helion
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl


# %%
# Flex Attention Kernel Implementation
# ----------------------------
@helion.kernel(
    autotune_accuracy_check=False,
    static_shapes=True,
    # fast_math: the CuTe flash backend offers its polynomial (degree-2) exp2
    # packets, an approximation of ex2.approx, only under this setting; the
    # Triton backend takes the approximate divide (fast_dividef,
    # div.approx.ftz.f32) for the output normalization (the LSE rescale
    # divides by a constant, which Inductor folds into a multiply) and
    # Inductor's use_fast_math (flush-to-zero exp; exp2 keeps libdevice, or
    # tl.math.exp2 on XPU).
    fast_math=True,
)
def helion_flex_attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask_kv_num_blocks: torch.Tensor,
    block_mask_kv_indices: torch.Tensor,
    block_mask_full_kv_num_blocks: torch.Tensor | None,
    block_mask_full_kv_indices: torch.Tensor | None,
    block_mask_mask_mod: Callable,
    block_mask_m: int,
    block_mask_n: int,
    scale: float,
    enable_gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, M, D = query.size()
    D = hl.specialize(D)
    # The V/O head dim may differ from the Q/K head dim (the MLA prefill pair
    # 192/128): the accumulator and the output take V's width.
    Dv = hl.specialize(value.size(-1))
    Bk, Hk, N, Dk = key.size()
    assert Bk == B
    assert Dk == D
    assert value.size()[:3] == key.size()[:3]
    if enable_gqa:
        assert H % Hk == 0
    else:
        assert Hk == H
    # Computed outside the branch so it stays a static int (a branch on the
    # runtime ``enable_gqa`` flag would merge it into a symbolic value).
    num_groups = H // Hk
    # The BlockMask block sizes are compile-time constants of the walk (the
    # fused CuTe plan reads them off the IR); a mask with other block sizes
    # binds a different specialization.
    block_mask_m = hl.specialize(block_mask_m)
    block_mask_n = hl.specialize(block_mask_n)
    # BlockMask tables may broadcast over batch and/or heads (size-1 leading
    # dims): index them modulo their own sizes, like FlexAttention's
    # ``sparse_idx_z = off_z % SPARSE_Z``.
    table_batch, table_heads, _ = block_mask_kv_num_blocks.size()
    # The output keeps the query's memory layout (a transposed or packed
    # query gives a transposed output, like ``empty_like``); a V/O width
    # other than the Q/K width gets a dense output of its own.
    if Dv == D:
        out = torch.empty_like(query)
    else:
        out = torch.empty((B, H, M, Dv), dtype=query.dtype, device=query.device)
    lse = torch.empty((B, H, M), dtype=torch.float32, device=out.device)
    log_2_e = 1.44269504
    block_m = hl.register_block_size(min(256, block_mask_m))
    block_n = hl.register_block_size(min(256, block_mask_n))
    assert (block_mask_full_kv_indices is None) == (
        block_mask_full_kv_num_blocks is None
    )
    for tile_b, tile_h, tile_m in hl.tile([B, H, M], block_size=[1, 1, block_m]):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_m, Dv], dtype=torch.float32)
        q_i = query[tile_b.begin, tile_h.begin, tile_m, :]

        sparse_row = tile_m.begin // block_mask_m
        b_idx = tile_b.begin
        h_idx = tile_h.begin
        h_kv_idx = h_idx // num_groups
        tb_idx = b_idx % table_batch
        th_idx = h_idx % table_heads
        # The batch and head tiles are one element wide (``block_size=[1, 1,
        # block_m]``), so ``b`` / ``h`` are ``[1, 1]`` vectors that broadcast
        # against every ``[tile_m, tile_n]`` score; built from a literal
        # one-element range rather than the tiles' own (symbolic) extent so
        # that the shape is a concrete 1 for the broadcast.
        bcast_b = (tile_b.begin + hl.arange(1))[:, None]
        bcast_h = (tile_h.begin + hl.arange(1))[:, None]
        bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[:, None]

        # iterate through full tiles (no mask needed)
        if block_mask_full_kv_indices is not None:
            sparse_num_blocks = block_mask_full_kv_num_blocks[  # pyrefly: ignore[unsupported-operation]
                tb_idx, th_idx, sparse_row
            ]

            for block_idx in hl.tile(sparse_num_blocks, block_size=1):
                # kv_indices holds KV *block* indices; convert to a token offset.
                start_n = (
                    block_mask_full_kv_indices[tb_idx, th_idx, sparse_row, block_idx.id]
                    * block_mask_n
                )
                end_n = start_n + block_mask_n
                end_N = end_n.new_full([], N)
                end_n = torch.minimum(end_n, end_N)

                for tile_n in hl.tile(start_n, end_n, block_size=block_n):
                    k = key[b_idx, h_kv_idx, tile_n, :]
                    bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[None, :]
                    qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    qk = hl.dot(q_i, k.T, acc=qk)
                    # Apply score_mod (score is in sm_scale space for API compat)
                    qk = qk * scale
                    bcast_qk = qk
                    score = score_mod(bcast_qk, bcast_b, bcast_h, bcast_m, bcast_n)
                    qk = score
                    qk *= log_2_e
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    # A row with every score masked so far has m_ij == -inf;
                    # rescale against 0 instead so -inf - -inf never produces
                    # NaN, but keep the true running max in m_i.
                    m_safe = torch.where(m_ij == float("-inf"), 0.0, m_ij)
                    qk = qk - m_safe[:, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_safe)
                    m_i = m_ij
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, None]
                    v = value[b_idx, h_kv_idx, tile_n, :]
                    p = p.to(v.dtype)
                    acc = hl.dot(p, v, acc=acc)

        # iterate through partial tiles (mask needed)
        sparse_num_blocks = block_mask_kv_num_blocks[tb_idx, th_idx, sparse_row]

        for block_idx in hl.tile(sparse_num_blocks, block_size=1):
            start_n = (
                block_mask_kv_indices[tb_idx, th_idx, sparse_row, block_idx.id]
                * block_mask_n
            )
            end_n = start_n + block_mask_n
            end_N = end_n.new_full([], N)
            end_n = torch.minimum(end_n, end_N)

            for tile_n in hl.tile(start_n, end_n, block_size=block_n):
                k = key[b_idx, h_kv_idx, tile_n, :]
                bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[None, :]
                qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                qk = hl.dot(q_i, k.T, acc=qk)
                # Apply score_mod and mask
                qk = qk * scale
                bcast_qk = qk
                score = score_mod(bcast_qk, bcast_b, bcast_h, bcast_m, bcast_n)
                mask = block_mask_mask_mod(bcast_b, bcast_h, bcast_m, bcast_n)
                score = torch.where(mask, score, -float("inf"))
                qk = score
                qk *= log_2_e
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                m_safe = torch.where(m_ij == float("-inf"), 0.0, m_ij)
                qk = qk - m_safe[:, None]
                p = torch.exp2(qk)
                l_ij = torch.sum(p, -1)
                alpha = torch.exp2(m_i - m_safe)
                m_i = m_ij
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v = value[b_idx, h_kv_idx, tile_n, :]
                p = p.to(v.dtype)
                acc = hl.dot(p, v, acc=acc)

        # Rows with no visible key: output 0 and lse -inf, like FlexAttention.
        l_i = torch.where(l_i == 0.0, 1.0, l_i)
        lse_row = (m_i + torch.log2(l_i)) / log_2_e  # Convert back to natural log
        acc = acc / l_i[:, None]
        out[tile_b, tile_h, tile_m, :] = acc[None, None, :, :].to(out.dtype)
        lse[tile_b, tile_h, tile_m] = lse_row[None, None, :]
    return out, lse


@functools.cache
def _zero_block_table(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """An all-zero BlockMask table (block counts or indices) for a mask that
    has no full blocks.

    The kernel only reads it, so one tensor per shape, dtype and device serves
    every call: allocating and zero-filling fresh tables per call would put
    that work on the launch path (and into CUDA-graph replays) and hand the
    fused CuTe plan a new table identity on every call.  The cache holds the
    tables for the life of the process.
    """
    return torch.zeros(shape, dtype=dtype, device=device)


def _zero_block_table_like(table: torch.Tensor) -> torch.Tensor:
    return _zero_block_table(tuple(table.shape), table.dtype, table.device)


def helion_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable | None = None,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    B, H, M, D = query.size()
    if score_mod is None:
        score_mod = _identity
    if block_mask is None:
        block_mask = _create_empty_block_mask(query, key)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    # The fused CuTe plan folds ``scale`` into the softmax multiplier of a
    # mask-only kernel, which is only an identity for a positive scale
    # (``-inf * 0`` and ``-inf * negative`` differ from the source).
    assert scale > 0, "flex attention requires a positive scale"
    # Always pass both block lists so one kernel specialization serves every
    # mask: a BlockMask without full blocks gets cached zero-count full tables
    # (``_zero_block_table``), so the launch path allocates nothing.
    full_kv_num_blocks = block_mask.full_kv_num_blocks
    full_kv_indices = block_mask.full_kv_indices
    if full_kv_num_blocks is None or full_kv_indices is None:
        full_kv_num_blocks = _zero_block_table_like(block_mask.kv_num_blocks)
        full_kv_indices = _zero_block_table_like(block_mask.kv_indices)
    out, lse = helion_flex_attention_kernel(
        query,
        key,
        value,
        score_mod,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        block_mask.mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        scale,
        enable_gqa,
    )

    if return_lse:
        return out, lse
    return out


def flex_attention_tritonbench(
    tb_op: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    score_mod: _score_mod_signature | None,
    block_mask: BlockMask | None,
    mod_type: str,
    kernel_options: dict[str, Any],
) -> Callable | None:
    return lambda: helion_flex_attention(q, k, v, score_mod, block_mask)


# %%
# Flex Attention Backward
# -----------------------


@helion.kernel(static_shapes=True)
def _flex_attention_bwd_preprocess(
    o_in: torch.Tensor, do_in: torch.Tensor, lse_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-row terms of the softmax backward: ``delta[b, h, m] = sum_d o * do``
    (fp32) and the forward's natural-log LSE in the exp2 base, ``lse2 = lse *
    log2(e)``, with a row that saw no key (``lse == -inf``) mapped to 0 so the
    main kernel's ``score * log2(e) - lse2`` stays finite (its P is 0 anyway:
    every score of such a row is masked).

    ``o_in`` / ``do_in`` are indexed as the ``[B, H, M, D]`` views they are
    (a ``[B, S, H, D]`` allocation's transpose included): a row-flattening
    reshape would copy such views on the host before every backward."""
    B, H, M, _ = o_in.size()
    delta = torch.empty((B, H, M), device=o_in.device, dtype=torch.float32)
    lse2 = torch.empty((B, H, M), device=o_in.device, dtype=torch.float32)
    log_2_e = 1.44269504
    for tile_b, tile_h, tile_m in hl.tile([B, H, M]):
        o_rows = o_in[tile_b, tile_h, tile_m, :].to(torch.float32)
        do_rows = do_in[tile_b, tile_h, tile_m, :].to(torch.float32)
        delta[tile_b, tile_h, tile_m] = torch.sum(o_rows * do_rows, dim=-1)
        lse_rows = lse_in[tile_b, tile_h, tile_m]
        lse2[tile_b, tile_h, tile_m] = (
            torch.where(lse_rows == float("-inf"), 0.0, lse_rows) * log_2_e
        )
    return delta, lse2


def _gradient_accuracy_check(actual: object, expected: object) -> None:
    """Autotuning accuracy check of :func:`helion_flex_attention_backward_kernel`.

    bf16 attention gradients carry rounding error proportional to each
    output's peak magnitude (P and dS round to bf16 before the dV / dK / dQ
    MMAs; dQ accumulates through fp32 atomics), and gradient tensors are
    heavy-tailed (their peak is ~30-70x their rms).  The default check's
    absolute tolerance (1e-2, scaled only when the rms exceeds 1) rejects
    correct configs over a single mid-magnitude element (e.g. every polynomial
    exp2 config of a causal mask).  This check bounds every element's
    difference by 2% of the output's peak instead: correct configs differ from
    each other by at most ~0.8% of the peak, broken ones (a missing tile, a
    wrong scale, a race) by far more.
    """
    for got, ref in zip(
        cast("tuple[torch.Tensor, ...]", actual),
        cast("tuple[torch.Tensor, ...]", expected),
        strict=True,
    ):
        ref = ref.float()
        peak = ref.abs().max().item()
        torch.testing.assert_close(got.float(), ref, atol=2e-2 * peak, rtol=0.0)


@helion.kernel(
    static_shapes=True,
    fast_math=True,
    autotune_baseline_accuracy_check_fn=_gradient_accuracy_check,
)
def helion_flex_attention_backward_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
    lse2: torch.Tensor,
    delta: torch.Tensor,
    score_mod: Callable,
    score_mod_bwd: Callable,
    block_mask_q_num_blocks: torch.Tensor,
    block_mask_q_indices: torch.Tensor,
    block_mask_full_q_num_blocks: torch.Tensor,
    block_mask_full_q_indices: torch.Tensor,
    block_mask_mask_mod: Callable,
    block_mask_m: int,
    block_mask_n: int,
    scale: float,
    enable_gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlexAttention-style fused attention backward (the twin of
    :func:`helion_flex_attention_kernel`).

    One work item per KV tile of one (batch, kv head): dK / dV accumulate in
    fp32 over the Q blocks the BlockMask lists for that KV block
    (``full_q_*``: no mask; ``q_*``: ``mask_mod`` applied), for every q head of
    the GQA group; dQ accumulates into an fp32 buffer with atomics (the
    wrapper converts).  ``lse2`` and ``delta`` are the ``_flex_attention_bwd_preprocess``
    row terms: the forward's natural-log logsumexp (its second output) in the
    exp2 base with no-visible-key rows mapped to 0, and the ``o * do`` row
    sums.  ``score_mod`` acts on the natural-unit scores like the forward;
    ``score_mod_bwd(grad, score, b, h, q_idx, kv_idx)`` maps the gradient
    w.r.t. the modified score back to the raw one (FlashAttention-4's
    interface; the identity for additive biases).  The CuTe backend lowers
    this form to its fused tcgen05 backward.
    """
    B, H, M, D = query.size()
    D = hl.specialize(D)
    Bk, Hk, N, Dk = key.size()
    Dv = hl.specialize(value.size(-1))
    assert Bk == B
    assert Dk == D
    assert grad_out.size(-1) == Dv
    if enable_gqa:
        assert H % Hk == 0
    else:
        assert Hk == H
    num_groups = H // Hk
    block_mask_m = hl.specialize(block_mask_m)
    block_mask_n = hl.specialize(block_mask_n)
    table_batch, table_heads, _ = block_mask_q_num_blocks.size()
    dq = torch.zeros((B, H, M, D), dtype=torch.float32, device=query.device)
    dk = torch.empty_like(key)
    dv = torch.empty_like(value)
    log_2_e = 1.44269504
    block_m = hl.register_block_size(min(256, block_mask_m))
    block_n = hl.register_block_size(min(256, block_mask_n))
    for tile_b, tile_hk, tile_n in hl.tile([B, Hk, N], block_size=[1, 1, block_n]):
        k_j = key[tile_b.begin, tile_hk.begin, tile_n, :]
        v_j = value[tile_b.begin, tile_hk.begin, tile_n, :]
        dk_acc = hl.zeros([tile_n, D], dtype=torch.float32)
        dv_acc = hl.zeros([tile_n, Dv], dtype=torch.float32)
        sparse_col = tile_n.begin // block_mask_n
        b_idx = tile_b.begin
        hk_idx = tile_hk.begin
        tb_idx = b_idx % table_batch
        bcast_b = (tile_b.begin + hl.arange(1))[:, None]
        bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[:, None]
        # The q heads sharing this KV head (one for MHA).
        for tile_h in hl.tile(
            hk_idx * num_groups, hk_idx * num_groups + num_groups, block_size=1
        ):
            h_idx = tile_h.begin
            th_idx = h_idx % table_heads
            bcast_h = (tile_h.begin + hl.arange(1))[:, None]

            # Q blocks that need no mask.
            full_num_blocks = block_mask_full_q_num_blocks[tb_idx, th_idx, sparse_col]
            for block_idx in hl.tile(full_num_blocks, block_size=1):
                start_m = (
                    block_mask_full_q_indices[tb_idx, th_idx, sparse_col, block_idx.id]
                    * block_mask_m
                )
                end_m = start_m + block_mask_m
                end_M = end_m.new_full([], M)
                end_m = torch.minimum(end_m, end_M)
                for tile_m in hl.tile(start_m, end_m, block_size=block_m):
                    q_i = query[b_idx, h_idx, tile_m, :]
                    do_i = grad_out[b_idx, h_idx, tile_m, :]
                    lse2_i = lse2[b_idx, h_idx, tile_m]
                    delta_i = delta[b_idx, h_idx, tile_m]
                    bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[None, :]
                    s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
                    s_t = s_t * scale
                    score = score_mod(s_t, bcast_b, bcast_h, bcast_m, bcast_n)
                    p_t = torch.exp2(score * log_2_e - lse2_i[None, :])
                    dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
                    dv_acc = hl.dot(p_t.to(value.dtype), do_i, acc=dv_acc)
                    ds_t = p_t * (dp_t - delta_i[None, :])
                    ds_t = score_mod_bwd(ds_t, s_t, bcast_b, bcast_h, bcast_m, bcast_n)
                    ds_c = ds_t.to(query.dtype)
                    dq_i = hl.dot(ds_c.T, k_j, out_dtype=torch.float32)
                    hl.atomic_add(dq, [b_idx, h_idx, tile_m, slice(None)], dq_i * scale)
                    dk_acc = hl.dot(ds_c, q_i, acc=dk_acc)

            # Q blocks with the mask applied.
            num_blocks = block_mask_q_num_blocks[tb_idx, th_idx, sparse_col]
            for block_idx in hl.tile(num_blocks, block_size=1):
                start_m = (
                    block_mask_q_indices[tb_idx, th_idx, sparse_col, block_idx.id]
                    * block_mask_m
                )
                end_m = start_m + block_mask_m
                end_M = end_m.new_full([], M)
                end_m = torch.minimum(end_m, end_M)
                for tile_m in hl.tile(start_m, end_m, block_size=block_m):
                    q_i = query[b_idx, h_idx, tile_m, :]
                    do_i = grad_out[b_idx, h_idx, tile_m, :]
                    lse2_i = lse2[b_idx, h_idx, tile_m]
                    delta_i = delta[b_idx, h_idx, tile_m]
                    bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[None, :]
                    s_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
                    s_t = s_t * scale
                    score = score_mod(s_t, bcast_b, bcast_h, bcast_m, bcast_n)
                    mask = block_mask_mask_mod(bcast_b, bcast_h, bcast_m, bcast_n)
                    score = torch.where(mask, score, -float("inf"))
                    p_t = torch.exp2(score * log_2_e - lse2_i[None, :])
                    dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
                    dv_acc = hl.dot(p_t.to(value.dtype), do_i, acc=dv_acc)
                    ds_t = p_t * (dp_t - delta_i[None, :])
                    ds_t = score_mod_bwd(ds_t, s_t, bcast_b, bcast_h, bcast_m, bcast_n)
                    ds_c = ds_t.to(query.dtype)
                    dq_i = hl.dot(ds_c.T, k_j, out_dtype=torch.float32)
                    hl.atomic_add(dq, [b_idx, h_idx, tile_m, slice(None)], dq_i * scale)
                    dk_acc = hl.dot(ds_c, q_i, acc=dk_acc)

        dv[tile_b, tile_hk, tile_n, :] = dv_acc[None, None, :, :].to(dv.dtype)
        dk[tile_b, tile_hk, tile_n, :] = (dk_acc * scale)[None, None, :, :].to(dk.dtype)
    return dq, dk, dv


def _identity_grad(
    grad: torch.Tensor,
    score: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    return grad


def _q_block_tables(block_mask: BlockMask) -> tuple[torch.Tensor, ...]:
    """The BlockMask's Q-side tables (``q_num_blocks``, ``q_indices`` and the
    full-block variants; zero tables when the mask has no full blocks)."""
    q_num_blocks = block_mask.q_num_blocks
    q_indices = block_mask.q_indices
    assert q_num_blocks is not None and q_indices is not None
    full_q_num_blocks = block_mask.full_q_num_blocks
    full_q_indices = block_mask.full_q_indices
    if full_q_num_blocks is None or full_q_indices is None:
        full_q_num_blocks = _zero_block_table_like(q_num_blocks)
        full_q_indices = _zero_block_table_like(q_indices)
    return (q_num_blocks, q_indices, full_q_num_blocks, full_q_indices)


def flex_attention_backward_args(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    out: torch.Tensor,
    score_mod: Callable | None,
    score_mod_bwd: Callable | None,
    block_mask: BlockMask,
    scale: float,
    enable_gqa: bool,
) -> tuple[object, ...]:
    """The argument tuple :func:`helion_flex_attention_backward` passes to the
    kernel (for binding / autotuning on the exact argument views)."""
    delta = torch.empty(lse.shape, dtype=torch.float32, device=query.device)
    lse2 = torch.empty(lse.shape, dtype=torch.float32, device=query.device)
    return (
        query,
        key,
        value,
        grad_out,
        lse2,
        delta,
        score_mod if score_mod is not None else _identity,
        score_mod_bwd if score_mod_bwd is not None else _identity_grad,
        *_q_block_tables(block_mask),
        block_mask.mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        scale,
        enable_gqa,
    )


def helion_flex_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    score_mod: Callable | None = None,
    score_mod_bwd: Callable | None = None,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients (dq, dk, dv) of :func:`helion_flex_attention` given its output,
    its natural-log LSE (``return_lse=True``) and the upstream gradient.

    ``score_mod_bwd(grad, score, b, h, q_idx, kv_idx)`` is the vector-Jacobian
    product of ``score_mod`` with respect to its score input (the identity when
    omitted, which is exact for additive biases).
    """
    B, H, M, D = query.size()
    if block_mask is None:
        block_mask = _create_empty_block_mask(query, key)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    assert scale > 0, "flex attention requires a positive scale"
    delta, lse2 = _flex_attention_bwd_preprocess(out, grad_out, lse)
    args = flex_attention_backward_args(
        query,
        key,
        value,
        grad_out,
        lse,
        out,
        score_mod,
        score_mod_bwd,
        block_mask,
        scale,
        enable_gqa,
    )
    dq, dk, dv = helion_flex_attention_backward_kernel(
        *args[:4], lse2, delta, *args[6:]
    )
    return dq.to(query.dtype), dk, dv


# %%
# Testing Function
# -------------
def test(
    z: int,
    h: int,
    n_ctx: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    score_mod: Callable | None = None,
    device: torch.device | str = "cuda",
) -> None:
    """
    Test the attention kernel implementation against PyTorch's native attention functions.

    Args:
        z: Batch size
        h: Number of attention heads
        n_ctx: Sequence length (context size)
        head_dim: Dimension of each attention head
        dtype: Data type for the tensors
        device: Device to run the test on
    """
    q, k, v = [
        torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=device)
        for _ in range(3)
    ]

    # Pre-compute block_mask to avoid overhead in benchmark loop
    block_mask = _create_empty_block_mask(q, k)

    flex_compiled = cast(
        "Callable[..., torch.Tensor]", torch.compile(flex_attention, fullgraph=True)
    )

    def helion_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ret = helion_flex_attention(q, k, v, score_mod, block_mask)
        assert isinstance(ret, torch.Tensor)
        return ret

    def flex_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flex_compiled(q, k, v, score_mod, block_mask)

    baselines = {
        "flex": flex_fn,
    }

    run_example(helion_fn, baselines, (q, k, v))  # pyright: ignore[reportArgumentType]


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the attention kernel test with specific parameters.
    Tests with batch size 2, 32 heads, 1024 sequence length, and 64-dimensional heads using float16.
    """
    test(2, 32, 1024, 64, HALF_DTYPE)
    test(2, 4, 1024, 64, HALF_DTYPE, lambda score, *_: torch.tanh(score))


if __name__ == "__main__":
    main()
