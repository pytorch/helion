"""
Standalone Triton GDPA (Generalized Dot Product Attention)
Simplified version extracted from mkl/ops/triton/triton_generalized_dot_product_attention.py
Removed all complex features: TMA, warp specialization, MTIA, ensemble activations, etc.
"""

import math
import torch
import triton
import triton.language as tl

# Import local triton math functions
from triton_math import (
    fast_gelu,
    fast_gelu_grad,
    tanh_approx_fp32,
    raw,
    raw_grad,
)


# ============================================================================
# Activation helpers
# ============================================================================

def activation_string_to_int(s: str) -> int:
    """Convert activation string to integer enum"""
    activation_map = {
        "raw": 0,
        "gelu": 1,
        "gelu_approx": 2,
        "fast_gelu": 3,
        "leaky_relu": 4,
        "relu": 5,
        "fast_gelu_bf16": 6,
        "silu": 7,
        "fast_silu": 8,
        "hardswish": 9,
        "relu_square": 10,
    }
    if s not in activation_map:
        raise ValueError(f"Unsupported activation function: {s}")
    return activation_map[s]


def next_power_of_2(x: int) -> int:
    return 2 ** (math.ceil(math.log(x, 2)))


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if x is not None and not x.is_contiguous():
        return x.contiguous()
    return x


# ============================================================================
# Triton Kernels - Forward
# ============================================================================

@triton.jit
def _gdpa_fwd_inner(
    acc,
    q,
    K_block_ptr,
    V_block_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    offs_d: tl.constexpr,
    qlen,
    klen,
    v_dtype,
    activation_enum_int,
    qk_scale,
):
    """Inner loop for forward pass"""
    lo, hi = 0, klen
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # Loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K and V
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Compute QK
        q = q.to(k.dtype)
        qk = tl.dot(q, k)

        # Apply activation
        if activation_enum_int == 3:  # fast_gelu
            p = fast_gelu(qk)
        else:
            p = qk.to(tl.float32)

        p *= qk_scale
        p = p.to(v_dtype)

        # Accumulate
        acc = tl.dot(p, v, acc)

        # Advance pointers
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc


@triton.jit
def _gdpa_fwd_kernel(
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    Out,
    stride_qm,
    stride_qh,
    stride_qk,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,
    stride_oh,
    stride_ok,
    H,
    G,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    qk_scale,
    activation_enum_int,
):
    # Get program IDs
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // G
    start_m = tl.program_id(0)

    # Get offsets
    q_offset = off_h.to(tl.int64) * stride_qh
    kv_offset = off_h_kv.to(tl.int64) * stride_kh
    out_offset = off_h.to(tl.int64) * stride_oh

    begin_q = tl.load(Q_offsets + off_z)
    end_q = tl.load(Q_offsets + off_z + 1)
    qlen = end_q - begin_q
    qlen = tl.minimum(qlen, N_CTX)

    begin_k = tl.load(K_offsets + off_z)
    end_k = tl.load(K_offsets + off_z + 1)
    klen = end_k - begin_k

    if start_m * BLOCK_M < qlen:
        # Create block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset + begin_q * stride_qm,
            shape=(qlen, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset + begin_k * stride_vn,
            shape=(klen, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset + begin_k * stride_kn,
            shape=(HEAD_DIM, klen),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_D, BLOCK_N),
            order=(0, 1),
        )

        # Initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        # Output pointers
        o_ptrs = (
            Out
            + off_h.to(tl.int64) * stride_oh
            + begin_q * stride_om
            + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
        )

        # Initialize accumulator
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Load Q
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Compute forward
        acc = _gdpa_fwd_inner(
            acc,
            q,
            K_block_ptr,
            V_block_ptr,
            BLOCK_M,
            BLOCK_D,
            BLOCK_N,
            offs_m,
            offs_n,
            offs_d,
            qlen,
            klen,
            V.dtype.element_ty,
            activation_enum_int,
            qk_scale,
        )

        # Store output
        o_mask = (offs_m[:, None] < qlen) & (offs_d[None, :] < HEAD_DIM)
        tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=o_mask)


# ============================================================================
# Triton Kernels - Backward
# ============================================================================

@triton.jit
def _gdpa_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    Q_offsets,
    K_offsets,
    stride_qm,
    stride_qh,
    stride_d,
    stride_km,
    stride_kh,
    stride_dom,
    H,
    G,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    qk_scale,
    activation_enum_int,
):
    # Get program IDs
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // G
    pid = tl.program_id(0)

    # Get offsets
    begin_q = tl.load(Q_offsets + off_z)
    end_q = tl.load(Q_offsets + off_z + 1)
    qlen = end_q - begin_q

    begin_k = tl.load(K_offsets + off_z)
    end_k = tl.load(K_offsets + off_z + 1)
    klen = end_k - begin_k

    start_n = pid * BLOCK_N1

    if start_n < klen:
        # Adjust pointers for batch/head
        off_h2 = off_h.to(tl.int64)
        qadj = off_h2 * stride_qh + begin_q * stride_qm
        kadj = off_h_kv * stride_kh + begin_k * stride_km
        doadj = off_h2 * stride_dom + begin_q * stride_dom

        Q = Q + qadj
        K = K + kadj
        V = V + kadj
        DO = DO + doadj
        DK = DK + kadj
        DV = DV + kadj

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        offs_k = tl.arange(0, BLOCK_D)
        kmask = (offs_k[None, :] < HEAD_DIM) & (offs_n[:, None] < klen)

        # Load K and V
        k = tl.load(
            K + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d,
            mask=kmask,
        )
        v = tl.load(
            V + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d,
            mask=kmask,
        )

        dv = tl.zeros([BLOCK_N1, BLOCK_D], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, BLOCK_D], dtype=tl.float32)

        # Loop over Q blocks
        num_steps = tl.cdiv(qlen, BLOCK_M1)
        for step in range(num_steps):
            start_m = step * BLOCK_M1
            offs_m = start_m + tl.arange(0, BLOCK_M1)

            qmask = (offs_k[:, None] < HEAD_DIM) & (offs_m[None, :] < qlen)
            omask = (offs_m[:, None] < qlen) & (offs_k[None, :] < HEAD_DIM)

            # Load Q and DO
            qT = tl.load(
                Q + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_d,
                mask=qmask,
            )
            do = tl.load(
                DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_d,
                mask=omask,
            )

            qT = qT.to(k.dtype)
            qkT = tl.dot(k, qT)
            dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
            pT = qkT

            # Compute dV
            if activation_enum_int == 3:  # fast_gelu
                tanh_out = tanh_approx_fp32(
                    0.7978845608 * pT * (1 + 0.044715 * pT * pT)
                )
                ppT = 0.5 * pT * (1 + tanh_out)
                ppT *= qk_scale
                ppT = ppT.to(Q.dtype.element_ty)
                dv = dv + tl.dot(ppT, do)

                pT = (
                    0.5
                    * pT
                    * (1 - tanh_out * tanh_out)
                    * (0.7978845608 + 0.1070322243 * pT * pT)
                ) + 0.5 * (1 + tanh_out)
            else:
                ppT = pT
                ppT *= qk_scale
                ppT = ppT.to(Q.dtype.element_ty)
                dv = dv + tl.dot(ppT, do)
                pT = pT

            pT *= qk_scale
            dsT = pT * dpT
            dsT = dsT.to(Q.dtype.element_ty)
            dk = dk + tl.dot(dsT, tl.trans(qT))

        # Store DK and DV
        tl.store(
            DV + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d,
            dv,
            mask=kmask,
        )
        tl.store(
            DK + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d,
            dk,
            mask=kmask,
        )


@triton.jit
def _gdpa_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    Q_offsets,
    K_offsets,
    stride_qm,
    stride_qh,
    stride_d,
    stride_km,
    stride_kh,
    stride_dom,
    H,
    G,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_D: tl.constexpr,
    qk_scale,
    activation_enum_int,
):
    # Get program IDs
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // G
    pid = tl.program_id(0)

    # Get offsets
    begin_q = tl.load(Q_offsets + off_z)
    end_q = tl.load(Q_offsets + off_z + 1)
    qlen = end_q - begin_q

    begin_k = tl.load(K_offsets + off_z)
    end_k = tl.load(K_offsets + off_z + 1)
    klen = end_k - begin_k

    start_m = pid * BLOCK_M2

    if start_m < qlen:
        # Adjust pointers for batch/head
        off_h2 = off_h.to(tl.int64)
        qadj = off_h2 * stride_qh + begin_q * stride_qm
        kadj = off_h_kv * stride_kh + begin_k * stride_km
        doadj = off_h2 * stride_dom + begin_q * stride_dom

        Q = Q + qadj
        K = K + kadj
        V = V + kadj
        DO = DO + doadj
        DQ = DQ + qadj

        offs_m = start_m + tl.arange(0, BLOCK_M2)
        offs_k = tl.arange(0, BLOCK_D)

        qmask = (offs_k[None, :] < HEAD_DIM) & (offs_m[:, None] < qlen)

        # Load Q and DO
        q = tl.load(
            Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_d,
            mask=qmask,
        )
        do = tl.load(
            DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_d,
            mask=qmask,
        )

        dq = tl.zeros([BLOCK_M2, BLOCK_D], dtype=tl.float32)

        # Loop over K blocks
        num_steps = tl.cdiv(klen, BLOCK_N2)
        for step in range(num_steps):
            start_n = step * BLOCK_N2
            offs_n = start_n + tl.arange(0, BLOCK_N2)

            kmask = (offs_k[:, None] < HEAD_DIM) & (offs_n[None, :] < klen)

            # Load K and V
            kT = tl.load(
                K + offs_n[None, :] * stride_km + offs_k[:, None] * stride_d,
                mask=kmask,
            )
            vT = tl.load(
                V + offs_n[None, :] * stride_km + offs_k[:, None] * stride_d,
                mask=kmask,
            )

            qk = tl.dot(q, kT)

            if activation_enum_int == 3:  # fast_gelu_grad
                p = fast_gelu_grad(qk)
            else:
                p = qk.to(tl.float32)

            p *= qk_scale

            # Compute dP and dS
            dp = tl.dot(do, vT).to(tl.float32)
            ds = p * dp
            ds = ds.to(K.dtype.element_ty)

            # Compute dQ
            dq = dq + tl.dot(ds, tl.trans(kT))

        # Store DQ
        tl.store(
            DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_d,
            dq,
            mask=qmask,
        )


# ============================================================================
# Main Python Wrapper
# ============================================================================

def generalized_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    activation: str = "raw",
    broadcast_q: bool = False,
    qk_scale: float | None = None,
    **kwargs,  # Ignore other kwargs
) -> torch.Tensor:
    """
    Simplified triton GDPA forward pass
    """
    if qk_scale is None:
        qk_scale = 1.0

    HEAD_DIM = query.shape[-1]
    BLOCK_D = max(next_power_of_2(HEAD_DIM), 16)

    if broadcast_q:
        BATCH = key_offset.size(0) - 1
    else:
        BATCH = query_offset.size(0) - 1

    # Create output tensor
    o = torch.empty_like(query)

    # Get dimensions
    nheads = query.shape[1]
    G = query.shape[1] // key.shape[1]
    batch_size = BATCH * nheads

    # Autotune configs
    configs = [
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN},
            num_stages=s,
            num_warps=w,
        )
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
        for s in [1, 3]
        for w in [4, 8]
    ]

    grid = lambda args: (
        triton.cdiv(max_seq_len_q, args["BLOCK_M"]),
        batch_size,
        1,
    )

    activation_enum_int = activation_string_to_int(activation)

    # Launch kernel
    _gdpa_fwd_kernel[grid](
        query,
        query_offset,
        key,
        key_offset,
        value,
        o,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        nheads,
        G,
        max_seq_len_q,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_D=BLOCK_D,
        qk_scale=qk_scale,
        activation_enum_int=activation_enum_int,
    )

    return o


def generalized_dot_product_attention_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    BATCH: int,
    N_HEAD: int,
    Q_GROUP: int,
    N_CTX: int,
    N_CTX_KV: int,
    HEAD_DIM: int,
    qk_scale: float | None = None,
    activation_enum_int: int = 0,
    **kwargs,  # Ignore other kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simplified triton GDPA backward pass
    """
    if qk_scale is None:
        qk_scale = 1.0

    BLOCK_D = max(next_power_of_2(HEAD_DIM), 32)

    # Initialize gradient tensors
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    do = expect_contiguous(do)

    # Grid for backward
    grid_dkdv = lambda args: (
        N_HEAD * triton.cdiv(N_CTX_KV, args["BLOCK_N1"]),
        1,
        BATCH,
    )

    grid_dq = lambda args: (
        N_HEAD * triton.cdiv(N_CTX, args["BLOCK_M2"]),
        1,
        BATCH,
    )

    # Launch dK/dV kernel
    _gdpa_bwd_dkdv_kernel[grid_dkdv](
        q,
        k,
        v,
        do,
        dk,
        dv,
        q_offsets,
        k_offsets,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        do.stride(0),
        N_HEAD,
        Q_GROUP,
        N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M1=32,
        BLOCK_N1=64,
        BLOCK_D=BLOCK_D,
        qk_scale=qk_scale,
        activation_enum_int=activation_enum_int,
    )

    # Launch dQ kernel
    _gdpa_bwd_dq_kernel[grid_dq](
        q,
        k,
        v,
        do,
        dq,
        q_offsets,
        k_offsets,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        do.stride(0),
        N_HEAD,
        Q_GROUP,
        N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M2=64,
        BLOCK_N2=32,
        BLOCK_D=BLOCK_D,
        qk_scale=qk_scale,
        activation_enum_int=activation_enum_int,
    )

    return dq, dk, dv


# ============================================================================
# Autograd Function
# ============================================================================

class GDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, query_offset, key_offset, max_seq_len_q, max_seq_len_kv, activation, qk_scale, broadcast_q):
        output = generalized_dot_product_attention(
            query, key, value, query_offset, key_offset,
            max_seq_len_q, max_seq_len_kv, activation=activation, qk_scale=qk_scale, broadcast_q=broadcast_q,
        )

        ctx.save_for_backward(query, key, value, output, query_offset, key_offset)
        ctx.max_seq_len_q = max_seq_len_q
        ctx.max_seq_len_kv = max_seq_len_kv
        ctx.activation = activation
        ctx.qk_scale = qk_scale
        ctx.broadcast_q = broadcast_q
        if broadcast_q:
            ctx.BATCH = key_offset.size(0) - 1
        else:
            ctx.BATCH = query_offset.size(0) - 1
        ctx.N_HEAD = query.shape[1]
        ctx.G = query.shape[1] // key.shape[1]
        ctx.HEAD_DIM = key.shape[-1]

        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, q_offsets, k_offsets = ctx.saved_tensors

        dq, dk, dv = generalized_dot_product_attention_backward(
            do, q, k, v,
            q_offsets, k_offsets,
            ctx.BATCH,
            ctx.N_HEAD,
            ctx.G,
            ctx.max_seq_len_q,
            ctx.max_seq_len_kv,
            ctx.HEAD_DIM,
            qk_scale=ctx.qk_scale,
            activation_enum_int=activation_string_to_int(ctx.activation),
        )

        return dq, dk, dv, None, None, None, None, None, None, None


def gdpa_autograd(query, key, value, query_offset, key_offset, max_seq_len_q, max_seq_len_kv, activation="fast_gelu", qk_scale=1.0, broadcast_q=False, **kwargs):
    """Wrapper with autograd support (ignores extra kwargs like is_causal for compatibility)"""
    return GDPA.apply(query, key, value, query_offset, key_offset, max_seq_len_q, max_seq_len_kv, activation, qk_scale, broadcast_q)
