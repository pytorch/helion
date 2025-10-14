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


# %%
@helion.kernel(
    # Static shapes provides a speedup for attention
    static_shapes=True,
)
def attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
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
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


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

    def ref_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Reference manual attention implementation"""
        p = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        p = torch.softmax(p.float(), dim=-1).to(dtype)
        return torch.matmul(p, v)

    flex_compiled = cast(
        "Callable[..., torch.Tensor]", torch.compile(flex_attention, fullgraph=True)
    )
    baselines = {
        "torch": torch.nn.functional.scaled_dot_product_attention,
        "flex": flex_compiled,
        "ref": ref_attention,
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
