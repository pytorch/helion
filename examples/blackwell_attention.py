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
@helion.kernel(
    configs=[
        helion.Config(
            block_sizes=[256, N],
            range_warp_specializes=[OUTER_LOOP or None, None if OUTER_LOOP else True],
            range_multi_buffers=[None, False],
            pid_type="persistent_interleaved",
            indexing="tensor_descriptor",
            num_warps=4,
            num_stages=3,
            _triton_range_id_data_partition_factor=0,
            _triton_range_value_data_partition_factor=2,
            _triton_config_maxRegAutoWS=maxreg,
        )
        for N in [64, 128]
        for OUTER_LOOP in [True]
        for maxreg in [152, 192]
    ],
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
                qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])  # pyright: ignore[reportArgumentType]
            else:
                qk = qk * qk_scale - m_ij[:, None]

            p = torch.exp2(qk)
            # -- compute correction factor
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)

            if SUBTILING:
                acc0, acc1 = hl.split(
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


def blackwell_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return blackwell_attention_kernel(q, k, v, qk_scale=math.sqrt(1.0 / q.shape[-1]))


def blackwell_attention_tritonbench(
    tb_mod: object, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Callable:
    return lambda: blackwell_attention(q, k, v)


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
    dur: float = do_bench(lambda: blackwell_attention(q, k, v))  # pyright: ignore[reportArgumentType, reportAssignmentType]
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
