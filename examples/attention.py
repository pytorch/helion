"""
Attention Example
=================

This code implements a custom attention kernel using Helion and PyTorch for efficient computation of scaled dot-product attention,
with support for both static and dynamic input shapes.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import math
from typing import Callable
from typing import cast

import torch
from torch.nn.attention.flex_attention import flex_attention

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Attention Kernel Implementation
# -------------------------------


# Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

def torch_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # q: [batch, heads, M, D]
    # k: [batch, heads, N, D]
    # v: [batch, heads, N, D]
    # Out: [batch, heads, M, D]

    dtype = q.dtype
    head_dim = q.size(-1)

    # Compute raw QK^T attention scores -> [batch, heads, M, N]
    logits = torch.matmul(q.float(), k.float().transpose(2, 3))

    # Softmax (over last dim) keeps [batch, heads, M, N]
    logits = logits / math.sqrt(head_dim)
    probs = torch.softmax(logits, dim=-1)

    # Weighted sum of values using the normalized probabilities -> [batch, heads, M, D]
    return torch.matmul(probs, v.float()).to(dtype)


def torch_attention_softmax_decomp(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # q: [batch, heads, M, D]
    # k: [batch, heads, N, D]
    # v: [batch, heads, N, D]
    # Out: [batch, heads, M, D]

    dtype = q.dtype
    head_dim = q.size(-1)

    # Compute raw QK^T attention scores -> [batch, heads, M, N]
    logits = torch.matmul(q.float(), k.float().transpose(2, 3))

    # Softmax (over last dim) keeps [batch, heads, M, N]
    logits = logits / math.sqrt(head_dim)
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)  # -> [batch, heads, M, 1]
    shifted = logits - max_logits
    exp_logits = torch.exp(shifted)
    denom = torch.sum(exp_logits, dim=-1, keepdim=True)  # -> [batch, heads, M, 1]
    probs = exp_logits / denom

    # Weighted sum of values using the normalized probabilities -> [batch, heads, M, D]
    return torch.matmul(probs, v.float()).to(dtype)


@helion.kernel(static_shapes=True)
def attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    # q_in: [batch, heads, M, D]
    # k_in: [batch, heads, N, D]
    # v_in: [batch, heads, N, D]
    # Out: [batch, heads, M, D]

    m_dim = q_in.size(-2)  # query length M
    n_dim = k_in.size(-2)  # key/value length N
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])  # [batch*heads, M, D]
    v_view = v_in.reshape([-1, n_dim, head_dim])  # [batch*heads, N, D]
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)  # [batch*heads, D, N]
    out = torch.empty_like(q_view)  # [batch*heads, M, D]
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    # Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    # The outer hl.tile loop runs in parallel across (batch*heads) tiles and across M tiles.
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):

        # Track running row max (max_logits), row sum (denom), and partial attention output.
        max_logits = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        denom = torch.full_like(max_logits, 1.0)
        value_accum = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q_tile = q_view[tile_b, tile_m, :]

        # The inner hl.tile loop processes N tiles sequentially.
        for tile_n in hl.tile(v_view.size(1)):

            # Q @ K^T
            k_tile = k_view[tile_b, :, tile_n]
            logits = torch.bmm(q_tile, k_tile)

            # softmax(... / sqrt(d_k))
            new_max_logits = torch.maximum(max_logits, torch.amax(logits, -1) * qk_scale)
            shifted_logits = logits * qk_scale - new_max_logits[:, :, None]
            exp_logits = torch.exp2(shifted_logits)
            denom_tile = torch.sum(exp_logits, -1)
            alpha_correction = torch.exp2(max_logits - new_max_logits)
            denom = denom * alpha_correction + denom_tile
            value_accum = value_accum * alpha_correction[:, :, None]
            v_tile = v_view[tile_b, tile_n, :]
            probs = exp_logits.to(v_tile.dtype)

            # ... @ V
            value_accum = torch.baddbmm(value_accum, probs, v_tile)

            # Update running max_logits
            max_logits = new_max_logits

        max_logits += torch.log2(denom)
        value_accum = value_accum / denom[:, :, None]
        out[tile_b, tile_m, :] = value_accum.to(out.dtype)

    return out.view(q_in.size())


@helion.kernel(static_shapes=True)
def blackwell_attention_kernel(
    q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # q_in: [batch, heads, M, D]
    # k_in: [batch, heads, N, D]
    # v_in: [batch, heads, N, D]
    # Out: [batch, heads, M, D]

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
                qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
            else:
                qk = qk * qk_scale - m_ij[:, None]

            p = torch.exp2(qk)
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

            p = p.to(v.dtype)
            acc = hl.dot(p, v_j, acc=acc)

            l_i = l_i * alpha + l_ij
            m_i = m_ij

        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc

    return o.reshape(B, H, M, Dv), lse.reshape(B, H, M)

















# %%
# Dynamic Shape Version
# ---------------------

# %%
attention_dynamic: object = helion.kernel(  # pyright: ignore[reportCallIssue]
    attention.fn,
    configs=attention.configs,  # pyright: ignore[reportArgumentType]
    static_shapes=False,
)
"""
Dynamic shape version of the attention kernel.
This version allows for variable input shapes at runtime.
"""


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

    flex_compiled = cast(
        "Callable[..., torch.Tensor]", torch.compile(flex_attention, fullgraph=True)
    )
    baselines = {
        "torch": torch_attention,
        "torch_softmax_decomp": torch_attention_softmax_decomp,
        "flex": flex_compiled,
    }

    run_example(attention, baselines, (q, k, v))


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the attention kernel test with specific parameters.
    Tests with batch size 2, 32 heads, 1024 sequence length, and 64-dimensional heads using float16.
    """
    test(2, 32, 1024, 64, torch.float16, device=DEVICE)


if __name__ == "__main__":
    main()
