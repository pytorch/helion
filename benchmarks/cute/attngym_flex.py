"""Output-only attention kernels for attention-gym benchmark cases.

Plain dense and causal cases use ``examples/attention.py``. Other cases keep
the BlockMask walk from ``examples/flex_attention.py`` for fused FlexAttention.
The value dimension may differ from the query/key dimension, and BlockMask
tables may vary over batch and query heads.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Callable

from benchmarks.cute import attngym_variants as agv
from examples.attention import attention_output
from examples.attention import causal_attention_output
import torch
from torch.nn.attention.flex_attention import _identity
from torch.nn.attention.flex_attention import noop_mask

import helion
import helion.language as hl


def attention_gym_flex_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask_kv_num_blocks: torch.Tensor,
    block_mask_kv_indices: torch.Tensor,
    block_mask_full_kv_num_blocks: torch.Tensor,
    block_mask_full_kv_indices: torch.Tensor,
    block_mask_mask_mod: Callable,
    block_mask_m: int,
    block_mask_n: int,
    scale: float,
    enable_gqa: bool = False,
) -> torch.Tensor:
    B, H, M, D = query.size()
    D = hl.specialize(D)
    Bk, Hk, N, Dk = key.size()
    Bv, Hv, Nv, Dv = value.size()
    Dv = hl.specialize(Dv)
    assert Bk == B
    assert (Bv, Hv, Nv) == (Bk, Hk, N)
    assert Dk == D
    assert query.dtype == key.dtype == value.dtype
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    if enable_gqa:
        assert H % Hk == 0
    else:
        assert Hk == H
    num_groups = H // Hk
    block_mask_m = hl.specialize(block_mask_m)
    block_mask_n = hl.specialize(block_mask_n)
    assert block_mask_m in (128, 256)
    assert block_mask_n == 128
    table_batch, table_heads, _ = block_mask_kv_num_blocks.size()
    out = torch.empty((B, H, M, Dv), dtype=query.dtype, device=query.device)
    log_2_e = 1.44269504
    block_m = hl.register_block_size(min(256, block_mask_m))
    block_n = hl.register_block_size(min(256, block_mask_n))
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
        bcast_b = (tile_b.begin + hl.arange(1))[:, None]
        bcast_h = (tile_h.begin + hl.arange(1))[:, None]
        bcast_m = (tile_m.begin + hl.arange(tile_m.block_size))[:, None]

        sparse_num_blocks = block_mask_full_kv_num_blocks[tb_idx, th_idx, sparse_row]
        for block_idx in hl.tile(sparse_num_blocks, block_size=1):
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
                qk = qk * scale
                score = score_mod(qk, bcast_b, bcast_h, bcast_m, bcast_n)
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
                qk = qk * scale
                score = score_mod(qk, bcast_b, bcast_h, bcast_m, bcast_n)
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

        # Empty BlockMask rows and fully masked partial rows both produce zero.
        l_i = torch.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]
        out[tile_b, tile_h, tile_m, :] = acc[None, None, :, :].to(out.dtype)
    return out


def _autotune_reference(
    variant: agv.Variant,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *kernel_args: object,
) -> torch.Tensor:
    score_mod, mask_mod = variant.make_mods(query.device)
    return agv.reference_output(variant, query, key, value, score_mod, mask_mod)


_KERNELS: dict[tuple[int, str], helion.Kernel[torch.Tensor]] = {}


def _kernel_for_case(variant: agv.Variant) -> helion.Kernel[torch.Tensor]:
    backend = os.environ.get("HELION_BACKEND", "triton")
    cache_key = (id(variant), backend)
    if cache_key not in _KERNELS:
        _KERNELS[cache_key] = helion.kernel(
            backend=backend,
            static_shapes=True,
            fast_math=True,
            autotune_baseline_fn=functools.partial(_autotune_reference, variant),
            autotune_baseline_atol=5e-2,
            autotune_baseline_rtol=2e-2,
        )(attention_gym_flex_kernel)
    return _KERNELS[cache_key]


def kernel_call_args(
    variant: agv.Variant,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
) -> tuple[helion.Kernel[torch.Tensor], tuple[object, ...]]:
    from benchmarks.cute import attngym_figure

    if variant.name == "noop":
        return attention_output, (q, k, v)
    if variant.name == "causal":
        return causal_attention_output, (q, k, v)

    _, torch_mask = variant.make_mods(device)
    score_mod, mask_mod = attngym_figure.helion_mods(variant, device)
    block_mask = agv.build_block_mask(variant, torch_mask, device)
    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    if full_num is None or full_idx is None:
        full_num = torch.zeros_like(block_mask.kv_num_blocks)
        full_idx = torch.zeros_like(block_mask.kv_indices)
    args = (
        q,
        k,
        v,
        _identity if score_mod is None else score_mod,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        full_num,
        full_idx,
        noop_mask if mask_mod is None else mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        1.0 / math.sqrt(variant.d),
        variant.enable_gqa,
    )
    return _kernel_for_case(variant), args
