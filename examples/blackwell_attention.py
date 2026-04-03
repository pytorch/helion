"""
BLackwell Attention Example
=================

This code implements a custom attention kernel using Helion and PyTorch for efficient computation of scaled dot-product attention,
specifically tuned for Blackwell.
"""
# %%
# Imports
# -------

# %%
from __future__ import annotations

import math
from typing import Callable

import torch
from triton.testing import do_bench

import helion
from helion._testing import run_example
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl

# %%
# Utility Functions
# -------------------------------


# %%
def _mul_f32x2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized F32 PTX MUL"""
    return hl.inline_asm_elementwise(
        """
            {
                .reg .b64 ra, rb, rc;
                mov.b64 ra, { $2, $3 };
                mov.b64 rb, { $4, $5 };
                mul.f32x2 rc, ra, rb;
                mov.b64 { $0, $1 }, rc;
            }
            """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=torch.float32,
        is_pure=True,
        pack=2,
    )


# %%
def _fma_f32x2(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Vectorized F32 PTX FMA"""
    return hl.inline_asm_elementwise(
        """
            {
                .reg .b64 ra, rb, rc;
                mov.b64 ra, { $2, $3 };
                mov.b64 rb, { $4, $5 };
                mul.f32x2 rc, ra, rb;
                mov.b64 { $0, $1 }, rc;
            }
            """,
        "=r,=r,r,r,r,r",
        [a, b, c],
        dtype=torch.float32,
        is_pure=True,
        pack=2,
    )


# %%
# Attention Kernel Implementation
# -------------------------------


# %%
# pyrefly: ignore [no-matching-overload]
@helion.kernel(
    config=helion.Config(
        block_sizes=[256, 128],
        range_warp_specializes=[True, None],
        range_multi_buffers=[None, None],
        pid_type="persistent_interleaved",
        indexing=[
            "tensor_descriptor",  # q load
            "tensor_descriptor",  # k load
            "tensor_descriptor",  # v load
            "block_ptr",  # lse store — block_ptr is correctly partitioned
            "tensor_descriptor",  # o store   under WS dpf=2; pointer stores are not
        ],
        num_warps=4,
        num_stages=3,
        _triton_range_id_data_partition_factor=0,
        _triton_range_value_data_partition_factor=2,
        _triton_config_maxRegAutoWS=152,
    ),
    static_shapes=True,
    autotune_accuracy_check=False,
)
def blackwell_attention_kernel(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot-product attention.

    Implements the attention mechanism: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        q_in: Query tensor of shape [..., seq_len_q, head_dim]
        k_in: Key tensor of shape [..., seq_len_k, head_dim]
        v_in: Value tensor of shape [..., seq_len_k, head_dim]

    Returns:
        Output tensor of shape [..., seq_len_q, head_dim]
    """
    B, H, M, D = q_in.shape
    Bk, Hk, N, Dk = k_in.shape
    assert Dk == D
    assert Bk == B
    assert Hk == H
    Bv, Hv, Nv, Dv = v_in.shape
    assert Bv == B
    assert Hv == Hk
    assert Nv == N
    D = hl.specialize(D)
    Dv = hl.specialize(Dv)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, Dv)
    MM = q.shape[0]
    assert v.shape[0] == k.shape[0]
    o = q.new_empty(MM, Dv)
    lse = q.new_empty(MM, dtype=torch.float32)
    block_m = hl.register_block_size(M)
    block_n = hl.register_block_size(N)
    assert M % block_m == 0
    assert N % block_n == 0
    hl.register_tunable(
        "_triton_range_id_data_partition_factor", EnumFragment(choices=(0,))
    )
    hl.register_tunable(
        "_triton_range_value_data_partition_factor", EnumFragment(choices=(2,))
    )
    hl.register_tunable("_triton_config_maxRegAutoWS", EnumFragment(choices=(152, 192)))
    SUBTILING = True
    VECT_MUL = 1
    qk_scale = qk_scale * 1.44269504  # 1/log(2)
    for tile_m in hl.tile(MM, block_size=block_m):
        m_i = hl.zeros([tile_m]) - float("inf")
        l_i = hl.zeros([tile_m]) + 1.0
        acc = hl.zeros([tile_m, Dv])
        q_i = q[tile_m, :]

        start_N = tile_m.begin // M * N
        for tile_n in hl.tile(N, block_size=block_n):
            k_j = k[tile_n + start_N, :]
            v_j = v[tile_n + start_N, :]
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            if VECT_MUL == 2 or VECT_MUL == 3:
                # pyrefly: ignore [bad-argument-type]
                qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
            else:
                qk = qk * qk_scale - m_ij[:, None]

            p = torch.exp2(qk)
            # -- compute correction factor
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)

            if SUBTILING:
                acc0, acc1 = hl.split(
                    # pyrefly: ignore [no-matching-overload]
                    acc.reshape([tile_m, 2, Dv // 2]).permute(0, 2, 1)
                )
                if VECT_MUL == 1 or VECT_MUL == 3:
                    acc0 = _mul_f32x2(acc0, alpha[:, None])
                    acc1 = _mul_f32x2(acc1, alpha[:, None])
                else:
                    acc0 = acc0 * alpha[:, None]
                    acc1 = acc1 * alpha[:, None]
                acc = (
                    hl.join(acc0, acc1)
                    .permute(0, 2, 1)
                    .reshape(acc.size(0), acc.size(1))
                )
            else:
                acc = acc * alpha[:, None]

            # update m_i and l_i

            # We can potentially move these to be before updating l_ij, so the dot
            # is not blocked.
            # prepare p and v for the dot
            p = p.to(v.dtype)
            # note that this non transposed v for FP8 is only supported on Blackwell
            acc = hl.dot(p, v_j, acc=acc)

            l_i = l_i * alpha + l_ij
            m_i = m_ij

        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc

    return o.reshape(B, H, M, Dv), lse.reshape(B, H, M)


# %%
# Backward Kernel Implementation
# -------------------------------

BWD_EPILOGUE_SUBTILE = 4


@helion.kernel(
    config=helion.Config(block_sizes=[128], num_warps=4),
    static_shapes=True,
)
def _bwd_preprocess(o_in: torch.Tensor, do_in: torch.Tensor) -> torch.Tensor:
    B, H, N, D = o_in.shape
    D = hl.specialize(D)
    o = o_in.reshape(-1, D)
    do = do_in.reshape(-1, D)
    total = o.size(0)
    delta = torch.empty(total, device=o.device, dtype=torch.float32)
    for tile in hl.tile(total):
        delta[tile] = torch.sum(
            o[tile, :].to(torch.float32) * do[tile, :].to(torch.float32), dim=-1
        )
    return delta.reshape(B, H, N)


# %%
# pyrefly: ignore [no-matching-overload]
@helion.kernel(
    config=helion.Config(
        block_sizes=[128, 128],
        range_warp_specializes=[True, False],
        range_merge_epilogues=[True, False],
        range_tmem_alloc_algos=[2, 0],
        range_smem_alloc_algos=[1, 0],
        range_smem_budgets=[200000, 0],
        range_multi_buffers=[False, False],
        pid_type="persistent_interleaved",
        indexing=[
            "tensor_descriptor",
            "tensor_descriptor",  # k, v loads
            "tensor_descriptor",
            "tensor_descriptor",  # q, do loads
            "pointer",
            "pointer",  # lse, delta loads
            "pointer",
            "pointer",  # dv, dk stores
        ],
        num_warps=4,
        num_stages=2,
        dot_stages=[0, 0, 0, 1, 1],
        dot_orders=[0, 2, 2, 1, 1],
        dot_channels=[
            ["opndA,smem,1,0", "opndB,smem,2,1", "opndD,tmem,1,2"],
            ["opndA,smem,1,3", "opndB,smem,1,4", "opndD,tmem,1,5"],
            ["opndA,tmem,1,2", "opndD,tmem,1,7"],
            ["opndA,smem,1,8", "opndD,tmem,1,5"],
            ["opndD,tmem,1,10"],
        ],
        _triton_config_maxRegAutoWS=192,
    ),
    static_shapes=True,
    autotune_accuracy_check=False,
)
def blackwell_attention_backward_kernel(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    o_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    qk_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes backward pass of scaled dot-product attention.

    Args:
        q_in: Query tensor [B, H, N, D]
        k_in: Key tensor [B, H, N, D], pre-scaled by (sm_scale * RCP_LN2)
        v_in: Value tensor [B, H, N, D]
        o_in: Forward output tensor [B, H, N, D]
        lse_in: Log-sum-exp from forward [B, H, N] (base-2)
        do_in: Gradient of output [B, H, N, D]
        delta_in: Precomputed (o * do).sum(-1) [B, H, N]
        qk_scale: Scaling factor sqrt(1/D)

    Returns:
        (dq, dk, dv) gradient tensors, each [B, H, N, D]
    """
    B, H, N, D = q_in.shape
    Bk, Hk, Nk, Dk = k_in.shape
    assert Bk == B and Hk == H and Nk == N and Dk == D
    Bv, Hv, Nv, Dv = v_in.shape
    assert Bv == B and Hv == H and Nv == N and Dv == D
    Bo, Ho, No, Do = o_in.shape
    assert Bo == B and Ho == H and No == N and Do == D
    Bd, Hd, Nd, Ddo = do_in.shape
    assert Bd == B and Hd == H and Nd == N and Ddo == D
    D = hl.specialize(D)
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, D)
    do = do_in.reshape(-1, D)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_rows = q.size(0)
    LN2: float = 0.6931471824645996
    dq = torch.zeros((total_rows, D), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(N)
    block_n = hl.register_block_size(N)
    assert N % block_m == 0 and N % block_n == 0
    hl.register_tunable("_triton_config_maxRegAutoWS", EnumFragment(choices=(192,)))
    for tile_n in hl.tile(total_rows, block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.zeros([tile_n, D], dtype=torch.float32)
        dk_acc = hl.zeros([tile_n, D], dtype=torch.float32)
        start_m = tile_n.begin // N * N
        end_m = start_m + N
        for tile_m in hl.tile(start_m, end_m, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            m_i = lse[tile_m]
            di = delta[tile_m]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            p_t = torch.exp2(qk_t - m_i[None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - di[None, :])).to(q.dtype)
            dq_acc = hl.dot(ds_t.T, k_j, out_dtype=torch.float32)
            hl.descriptor_atomic_add(
                dq,
                [tile_m, slice(None)],
                dq_acc * LN2,
                subtile=BWD_EPILOGUE_SUBTILE,
            )
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
        dv[tile_n, :] = dv_acc.to(v.dtype)
        dk[tile_n, :] = (dk_acc * qk_scale).to(k.dtype)
    return (
        dq.reshape(B, H, N, D),
        dk.reshape(B, H, N, D),
        dv.reshape(B, H, N, D),
    )


# %%
# Autograd Function
# -----------------


# %%
class _BlackwellAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        o, lse = blackwell_attention_kernel(q, k, v, qk_scale=sm_scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale  # type: ignore[attr-defined]
        return o

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        do: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale  # type: ignore[attr-defined]
        RCP_LN2 = 1.4426950408889634
        k_scaled = k * (sm_scale * RCP_LN2)
        delta = _bwd_preprocess(o, do)
        dq, dk, dv = blackwell_attention_backward_kernel(
            q, k_scaled, v, o, lse, do.contiguous(), delta, sm_scale
        )
        return dq, dk, dv, None


def blackwell_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return blackwell_attention_kernel(q, k, v, qk_scale=math.sqrt(1.0 / q.shape[-1]))


def blackwell_attention_fwd_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return _BlackwellAttentionFunc.apply(q, k, v, math.sqrt(1.0 / q.shape[-1]))


def blackwell_attention_tritonbench(
    tb_mod: object, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Callable:
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)

    def fn() -> list[torch.Tensor]:
        return [blackwell_attention_fwd_bwd(q, k, v)]

    fn._grad_inputs = [q, k, v]  # type: ignore[attr-defined]
    return fn


# %%
# Testing Function
# ----------------


# %%
def test(
    z: int,
    h: int,
    n_ctx: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
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

    def ref_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Reference manual attention implementation"""
        p = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        p = torch.softmax(p.float(), dim=-1).to(dtype)
        return torch.matmul(p, v)

    baselines = {
        "torch": torch.nn.functional.scaled_dot_product_attention,
        "ref": ref_attention,
    }

    run_example(
        lambda *args: blackwell_attention(*args)[0],
        baselines,
        (q, k, v),
        atol=0.1,
        rtol=0.1,
    )
    # pyrefly: ignore [bad-assignment]
    dur: float = do_bench(lambda: blackwell_attention(q, k, v))
    print(
        f"{z=} {h=} {n_ctx=} {head_dim=} tflops={z * h * n_ctx * n_ctx * head_dim * 4 / dur * 1e-9:.2f}"
    )


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the attention kernel test with specific parameters.
    Tests with batch size 2, 32 heads, 1024 sequence length, and 64-dimensional heads using float16.
    """
    test(4, 32, 8192, 64, torch.bfloat16)
    test(4, 32, 8192, 128, torch.bfloat16)


if __name__ == "__main__":
    main()
