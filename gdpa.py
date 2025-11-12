"""
Helion GDPA (Generalized Dot Product Attention)
Extracted from mkl/ops/helion/gdpa.py
"""

import types
import torch
import helion
import helion.language as hl


# ============================================================================
# Activation Functions
# ============================================================================

def _tanh_approx_fp32(x: torch.Tensor) -> torch.Tensor:
    """Vectorized F32 PTX tanh approximation"""
    output = hl.inline_asm_elementwise(
        asm="""
            tanh.approx.f32 $0, $1;
            """,
        constraints="=r,r",
        args=[x],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )
    return output


def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU activation using tanh approximation"""
    return x * 0.5 * (1 + _tanh_approx_fp32(0.7978845608 * x * (1.0 + 0.044715 * x * x)))


def fast_gelu_grad(x):
    """Gradient of fast GELU"""
    tanh_out = _tanh_approx_fp32(0.7978845608 * x * (1.0 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


# ============================================================================
# Helion GDPA Forward Kernel
# ============================================================================

@helion.kernel(
    config=helion.Config(
        block_sizes=[4, 16, 16],
        indexing=[
            "pointer",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        l2_groupings=[1],
        load_eviction_policies=["last", "first", "", "last", "first", "last", "last"],
        loop_orders=[[0, 1, 2]],
        num_stages=3,
        num_warps=4,
        pid_type="flat",
        range_flattens=[None, False, None],
        range_multi_buffers=[None, None, True],
        range_num_stages=[0, 3, 2],
        range_unroll_factors=[0, 2, 1],
        range_warp_specializes=[],
    ),
    static_shapes=True,
)
def helion_gdpa_kernel(
    q: torch.Tensor,
    q_offsets: torch.Tensor,
    k: torch.Tensor,
    k_offsets: torch.Tensor,
    v: torch.Tensor,
    max_seq_len_q: hl.constexpr,
    max_seq_len_kv: hl.constexpr,
    activation_enum_int: hl.constexpr,
    qk_scale: hl.constexpr,
    broadcast_q: hl.constexpr,
) -> torch.Tensor:
    if broadcast_q:
        BATCH = k_offsets.size(0) - 1
    else:
        BATCH = q_offsets.size(0) - 1

    out = torch.zeros_like(q)
    h = q.size(1)
    dimV = hl.specialize(v.size(2))

    # Tile over batch, head, sequence
    for tile_b, tile_h, tile_d in hl.tile([BATCH, h, dimV], block_size=[1, 1, None]):
        q_starts = q_offsets[tile_b.begin]
        q_ends = q_offsets[tile_b.begin + 1]
        q_seq_len = q_ends - q_starts
        k_starts = k_offsets[tile_b.begin]
        k_ends = k_offsets[tile_b.begin + 1]
        k_seq_len = k_ends - k_starts

        for tile_q in hl.tile(q_seq_len, block_size=None):
            q_blk = q[tile_q.index + q_starts, tile_h.begin, tile_d]
            acc = hl.zeros([tile_q, tile_d], dtype=torch.float32)

            for tile_kv in hl.tile(k_seq_len, block_size=None):
                k_blk = k[tile_kv.index + k_starts, tile_h.begin, tile_d]
                v_blk = v[tile_kv.index + k_starts, tile_h.begin, tile_d]
                qk = torch.matmul(q_blk, k_blk.T)

                # Apply activation
                if activation_enum_int == 0:
                    p = qk.to(torch.float32)
                elif activation_enum_int == 3:
                    p = fast_gelu(qk)
                else:
                    p = qk.to(torch.float32)

                p *= qk_scale
                p = p.to(v.dtype)
                acc = torch.addmm(acc, p.to(v.dtype), v_blk)

            # Store result
            out[tile_q.index + q_starts, tile_h.begin, tile_d] = acc.to(out.dtype)

    return out


# ============================================================================
# Helion GDPA Backward Kernels
# ============================================================================

@helion.kernel(
    config=helion.Config(
        block_sizes=[128, 32],
        indexing=[
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "tensor_descriptor",
        ],
        l2_groupings=[4],
        load_eviction_policies=["", "last", "last", "last", "", "", "", ""],
        loop_orders=[[1, 0]],
        num_stages=3,
        num_warps=8,
        pid_type="flat",
        range_flattens=[None, False, False],
        range_multi_buffers=[None, None, None],
        range_num_stages=[0, 1, 0],
        range_unroll_factors=[0, 0, 1],
        range_warp_specializes=[],
    ),
    static_shapes=True,
)
def _gdpa_bwd_dkdv(
    Q,
    k,
    v,
    DO,
    Q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    max_seq_len_q: hl.constexpr,
    max_seq_len_kv: hl.constexpr,
    broadcast_q: hl.constexpr,
    qk_scale: hl.constexpr,
    activation_enum_int: hl.constexpr,
):
    if broadcast_q:
        BATCH = k_offsets.size(0) - 1
    else:
        BATCH = Q_offsets.size(0) - 1

    h = Q.size(1)
    dimV = hl.specialize(v.size(2))

    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # Tile over batch, head, sequence
    for tile_b, tile_h in hl.tile([BATCH, h], block_size=[1, 1]):
        q_starts = Q_offsets[tile_b.begin]
        q_ends = Q_offsets[tile_b.begin + 1]
        k_starts = k_offsets[tile_b.begin]
        k_ends = k_offsets[tile_b.begin + 1]
        q_seq_len = q_ends - q_starts
        k_seq_len = k_ends - k_starts

        for tile_kv in hl.tile(k_seq_len, block_size=None):
            k_blk = k[tile_kv.index + k_starts, tile_h.begin, :]
            v_blk = v[tile_kv.index + k_starts, tile_h.begin, :]

            acc_dk = hl.zeros([tile_kv, dimV], dtype=torch.float32)
            acc_dv = hl.zeros([tile_kv, dimV], dtype=torch.float32)

            for tile_q in hl.tile(q_seq_len, block_size=None):
                q_blk = Q[tile_q.index + q_starts, tile_h.begin, :].to(k.dtype)
                qT_blk = q_blk.transpose(0, 1)
                do_blk = DO[tile_q.index + q_starts, tile_h.begin, :]

                qkT = hl.dot(k_blk, qT_blk)
                dpT = hl.dot(v_blk, do_blk.transpose(0, 1)).to(torch.float32)
                pT = qkT

                # Compute dV
                if activation_enum_int == 3:
                    tanh_out = _tanh_approx_fp32(0.7978845608 * pT * (1 + 0.044715 * pT * pT))
                    ppT = 0.5 * pT * (1 + tanh_out)
                    ppT *= qk_scale
                    ppT = ppT.to(Q.dtype)
                    acc_dv = acc_dv + hl.dot(ppT, do_blk)

                    pT = (
                        0.5 * pT * (1 - tanh_out * tanh_out) *
                        (0.7978845608 + 0.1070322243 * pT * pT)
                    ) + 0.5 * (1 + tanh_out)
                else:
                    ppT = pT
                    ppT *= qk_scale
                    ppT = ppT.to(Q.dtype)
                    acc_dv = acc_dv + hl.dot(ppT, do_blk)
                    pT = pT

                pT *= qk_scale
                dsT = pT * dpT
                dsT = dsT.to(Q.dtype)
                acc_dk = acc_dk + hl.dot(dsT, qT_blk.transpose(0, 1))

            dk[tile_kv.index + k_starts, tile_h.begin, :] = acc_dk.to(k.dtype)
            dv[tile_kv.index + k_starts, tile_h.begin, :] = acc_dv.to(v.dtype)

    return dk, dv


@helion.kernel(
    config=helion.Config(
        block_sizes=[64, 16],
        indexing=[
            "pointer",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
        ],
        l2_groupings=[8],
        load_eviction_policies=["", "last", "last", "last", "first", "first", "", ""],
        loop_orders=[[0, 1]],
        num_stages=7,
        num_warps=4,
        pid_type="flat",
        range_flattens=[None, False, True],
        range_multi_buffers=[None, None, True],
        range_num_stages=[0, 0, 3],
        range_unroll_factors=[0, 1, 2],
        range_warp_specializes=[],
    ),
    static_shapes=True,
)
def _gdpa_bwd_dq(
    q,
    K,
    V,
    DO,
    q_offsets: torch.Tensor,
    K_offsets: torch.Tensor,
    max_seq_len_q: hl.constexpr,
    max_seq_len_kv: hl.constexpr,
    broadcast_q: hl.constexpr,
    qk_scale: hl.constexpr,
    activation_enum_int: hl.constexpr,
):
    if broadcast_q:
        BATCH = K_offsets.size(0) - 1
    else:
        BATCH = q_offsets.size(0) - 1

    h = q.size(1)
    dimV = hl.specialize(V.size(2))

    dq = torch.zeros_like(q)

    # Tile over batch, head, sequence
    for tile_b, tile_h in hl.tile([BATCH, h], block_size=[1, 1]):
        q_starts = q_offsets[tile_b.begin]
        q_ends = q_offsets[tile_b.begin + 1]
        q_seq_len = q_ends - q_starts
        k_starts = K_offsets[tile_b.begin]
        k_ends = K_offsets[tile_b.begin + 1]
        k_seq_len = k_ends - k_starts

        for tile_q in hl.tile(q_seq_len, block_size=None):
            q_blk = q[tile_q.index + q_starts, tile_h.begin, :]
            do_blk = DO[tile_q.index + q_starts, tile_h.begin, :]

            acc_dq = hl.zeros([tile_q, dimV], dtype=torch.float32)

            for tile_kv in hl.tile(k_seq_len, block_size=None):
                k_blk = K[tile_kv.index + k_starts, tile_h.begin, :]
                v_blk = V[tile_kv.index + k_starts, tile_h.begin, :]

                qk = hl.dot(q_blk, k_blk.transpose(0, 1))

                if activation_enum_int == 0:
                    p = qk.to(torch.float32)
                elif activation_enum_int == 3:
                    p = fast_gelu_grad(qk)
                else:
                    p = qk.to(torch.float32)

                p *= qk_scale

                dp = hl.dot(do_blk, v_blk.transpose(0, 1)).to(torch.float32)
                ds = p * dp
                ds = ds.to(K.dtype)
                acc_dq = acc_dq + hl.dot(ds, k_blk)

            dq[tile_q.index + q_starts, tile_h.begin, :] = acc_dq.to(q.dtype)

    return dq


# ============================================================================
# PyTorch Custom Op Registration
# ============================================================================

@torch.library.custom_op("mkl::helion_gdpa", mutates_args=())
def helion_gdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    activation: str,
    broadcast_q: bool = False,
    qk_scale: float | None = None,
) -> torch.Tensor:
    return helion_gdpa_kernel(
        q=q,
        q_offsets=q_offsets,
        k=k,
        k_offsets=kv_offsets,
        v=v,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_kv=max_seq_len_kv,
        activation_enum_int=3,  # fast_gelu
        qk_scale=1.0,
        broadcast_q=broadcast_q,
    )


def helion_gdpa_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    activation_enum_int: int,
    broadcast_q: bool = False,
    qk_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dk, dv = _gdpa_bwd_dkdv(
        q, k, v, do,
        q_offsets, k_offsets,
        max_seq_len_q, max_seq_len_kv,
        broadcast_q, qk_scale, activation_enum_int,
    )

    dq = _gdpa_bwd_dq(
        q, k, v, do,
        q_offsets, k_offsets,
        max_seq_len_q, max_seq_len_kv,
        broadcast_q, qk_scale, activation_enum_int,
    )

    return dq, dk, dv


@torch.library.custom_op("mkl::helion_gdpa_backward", mutates_args=())
def helion_gdpa_backward_wrap(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    activation: str,
    broadcast_q: bool = False,
    qk_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return helion_gdpa_backward(
        do, q, k, v, o,
        q_offsets, k_offsets,
        max_seq_len_q, max_seq_len_kv,
        3,  # fast_gelu
        broadcast_q, qk_scale,
    )


def _gdpa_setup_context(ctx, inputs, output):
    (
        query, key, value,
        query_offset, key_offset,
        max_seq_len_q, max_seq_len_kv,
        activation, broadcast_q, qk_scale,
    ) = inputs

    if broadcast_q:
        BATCH = key_offset.size(0) - 1
    else:
        BATCH = query_offset.size(0) - 1

    nheads = query.shape[1]
    nheads_k = key.shape[1]

    ctx.save_for_backward(query, key, value, output, query_offset, key_offset)
    ctx.HEAD_DIM = key.shape[-1]
    ctx.max_seq_len_q = max_seq_len_q
    ctx.max_seq_len_kv = max_seq_len_kv
    ctx.BATCH = BATCH
    ctx.N_HEAD = nheads
    ctx.G = nheads // nheads_k
    ctx.broadcast_q = broadcast_q
    ctx.activation = activation
    ctx.qk_scale = qk_scale


def _gdpa_backward(ctx, do):
    q, k, v, o, q_offsets, k_offsets = ctx.saved_tensors
    dq, dk, dv = helion_gdpa_backward_wrap(
        do, q, k, v, o,
        q_offsets, k_offsets,
        ctx.max_seq_len_q, ctx.max_seq_len_kv,
        ctx.activation, ctx.broadcast_q, ctx.qk_scale,
    )
    return (
        dq, dk, dv,
        None, None, None, None, None, None, None,
    )


# Register autograd
if not isinstance(helion_gdpa, types.FunctionType):
    helion_gdpa.register_autograd(
        _gdpa_backward,
        setup_context=_gdpa_setup_context,
    )


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if x is not None and not x.is_contiguous():
        return x.contiguous()
    return x


@torch.fx.wrap
def helion_gdpa_wrap(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Q_offsets: torch.Tensor,
    K_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    activation: str,
    broadcast_q: bool = False,
    qk_scale: float | None = None,
) -> torch.Tensor:
    Q = expect_contiguous(Q)
    K = expect_contiguous(K)
    V = expect_contiguous(V)

    return torch.ops.mkl.helion_gdpa(
        q=Q, k=K, v=V,
        q_offsets=Q_offsets,
        kv_offsets=K_offsets,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_kv=max_seq_len_kv,
        activation=activation,
        qk_scale=qk_scale,
        broadcast_q=broadcast_q,
    )
