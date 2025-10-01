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
from helion._testing import run_example
import helion.language as hl


# %%
# Flex Attention Kernel Implementation
# ----------------------------
@helion.kernel(autotune_accuracy_check=False)
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
    assert key.size() == value.size()
    Bk, Hk, N, Dk = key.size()
    assert Bk == B
    assert Dk == D
    if enable_gqa:
        assert H % Hk == 0
        num_groups = H // Hk
    else:
        assert Hk == H
        num_groups = 1
    out = torch.empty_like(query)
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
        acc = hl.zeros([tile_m, D], dtype=torch.float32)
        q_i = query[tile_b.begin, tile_h.begin, tile_m, :]

        # iterate through full tiles
        if block_mask_full_kv_indices is not None:
            sparse_row = tile_m.begin // block_mask_m
            sparse_num_blocks = block_mask_full_kv_num_blocks[  # pyright: ignore[reportOptionalSubscript]
                tile_b.begin, tile_h.begin, sparse_row
            ]

            for block_idx in hl.tile(sparse_num_blocks, block_size=1):
                start_n = block_mask_full_kv_indices[
                    tile_b.begin, tile_h.begin, sparse_row, block_idx.id
                ]
                end_n = start_n + block_mask_n
                end_N = end_n.new_full([], N)
                end_n = torch.minimum(end_n, end_N)

                # figure out how many tiles there are here
                for tile_n in hl.tile(start_n, end_n, block_size=block_n):
                    k = key[tile_b.begin, tile_h.begin // num_groups, tile_n, :]
                    bcast_b = (tile_b.begin + hl.arange(tile_b.block_size))[
                        :, None, None, None
                    ]
                    bcast_h = (tile_h.begin + hl.arange(tile_h.block_size))[
                        None, :, None, None
                    ]
                    bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[
                        None, None, :, None
                    ]
                    bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[
                        None, None, None, :
                    ]
                    qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    qk = hl.dot(q_i, k.T, acc=qk)
                    qk = qk * scale
                    bcast_qk = qk[None, None, :, :]
                    score = score_mod(bcast_qk, bcast_b, bcast_h, bcast_m, bcast_n)
                    # mask = block_mask_mask_mod(bcast_b, bcast_h, bcast_m, bcast_n)
                    # score = torch.where(mask, score, -float("inf"))
                    qk = score.squeeze(0).squeeze(0)

                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    qk = qk - m_ij[:, None]
                    p = torch.exp2(log_2_e * qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    m_i = m_ij
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, None]
                    v = value[tile_b.begin, tile_h.begin // num_groups, tile_n, :]
                    p = p.to(v.dtype)
                    acc = hl.dot(p, v, acc=acc)

        # iterate through partial tiles

        if True:
            sparse_row = tile_m.begin // block_mask_m
            sparse_num_blocks = block_mask_kv_num_blocks[
                tile_b.begin, tile_h.begin, sparse_row
            ]

            for block_idx in hl.tile(sparse_num_blocks, block_size=1):
                start_n = block_mask_kv_indices[
                    tile_b.begin, tile_h.begin, sparse_row, block_idx.id
                ]
                end_n = start_n + block_mask_n
                end_N = end_n.new_full([], N)
                end_n = torch.minimum(end_n, end_N)

                # figure out how many tiles there are here
                for tile_n in hl.tile(start_n, end_n, block_size=block_n):
                    k = key[tile_b.begin, tile_h.begin // num_groups, tile_n, :]
                    bcast_b = (tile_b.begin + hl.arange(tile_b.block_size))[
                        :, None, None, None
                    ]
                    bcast_h = (tile_h.begin + hl.arange(tile_h.block_size))[
                        None, :, None, None
                    ]
                    bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[
                        None, None, :, None
                    ]
                    bcast_n = (tile_n.begin + hl.arange(tile_n.block_size))[
                        None, None, None, :
                    ]
                    qk = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    qk = hl.dot(q_i, k.T, acc=qk)
                    qk = qk * scale
                    bcast_qk = qk[None, None, :, :]
                    score = score_mod(bcast_qk, bcast_b, bcast_h, bcast_m, bcast_n)
                    mask = block_mask_mask_mod(bcast_b, bcast_h, bcast_m, bcast_n)
                    score = torch.where(mask, score, -float("inf"))
                    qk = score.squeeze(0).squeeze(0)

                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    qk = qk - m_ij[:, None]
                    p = torch.exp2(log_2_e * qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    m_i = m_ij
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, None]
                    v = value[tile_b.begin, tile_h.begin // num_groups, tile_n, :]
                    p = p.to(v.dtype)
                    acc = hl.dot(p, v, acc=acc)

        m_i += torch.log(l_i)
        acc = acc / l_i[:, None]
        out[tile_b, tile_h, tile_m, :] = acc[None, None, :, :].to(out.dtype)
        lse[tile_b, tile_h, tile_m] = m_i[None, None, :]
    return out, lse


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
    out, lse = helion_flex_attention_kernel(
        query,
        key,
        value,
        score_mod,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
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

    flex_compiled = cast(
        "Callable[..., torch.Tensor]", torch.compile(flex_attention, fullgraph=True)
    )
    baselines = {
        "flex": flex_compiled,
    }

    run_example(helion_flex_attention, baselines, (q, k, v), atol=0.2)  # pyright: ignore[reportArgumentType]


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the attention kernel test with specific parameters.
    Tests with batch size 2, 32 heads, 1024 sequence length, and 64-dimensional heads using float16.
    """
    test(2, 32, 1024, 64, torch.float16)
    test(2, 4, 1024, 64, torch.float16, torch.tanh)


if __name__ == "__main__":
    main()
