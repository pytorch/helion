"""An independent paged NT contraction for CuTe source and native checks."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def routed_transposed_product(
    x: torch.Tensor,
    bank: torch.Tensor,
    order: torch.Tensor,
    nonzero_seed: hl.constexpr,
    shift_page: hl.constexpr,
    check_page: hl.constexpr,
) -> torch.Tensor:
    m, k = x.shape
    p = bank.size(1)
    out = torch.empty((order.numel(), m, p), device=x.device, dtype=torch.float32)
    for slot in hl.tile(order.numel(), block_size=1):
        address = order[slot.begin]
        if check_page:
            torch._assert_async(address >= 0, "nonnegative page")
        if shift_page:
            address = (address + 1) % bank.size(0)
        accumulator = hl.full([m, p], 1.0 if nonzero_seed else 0.0, torch.float32)
        for kk in hl.tile(k):
            accumulator = hl.dot(x[:, kk], bank[address, :, kk].T, acc=accumulator)
        out[slot.begin, :, :] = accumulator
    return out


def split_paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fixture: a causal gathered-dot producer and a split-weighted merge share a grid."""
    _, query_len, num_heads, query_dim = query.shape
    query_len = hl.specialize(query_len)
    num_heads = hl.specialize(num_heads)
    query_dim = hl.specialize(query_dim)
    page = hl.specialize(kv_cache.size(2))
    value_dim = hl.specialize(query_dim - 64)
    # A literal: a module-level int reaches the kernel as a runtime scalar.
    split_capacity = hl.specialize(64)
    rows = query_len * num_heads
    q2d = query.view(rows, query_dim)
    kv3 = kv_cache.view(kv_cache.size(0), page, query_dim)
    k_lat3 = kv3[:, :, :value_dim]
    partial = torch.empty(
        (split_capacity, rows, value_dim), dtype=torch.float32, device=query.device
    )
    partial_lse = torch.empty(
        (split_capacity, rows), dtype=torch.float32, device=query.device
    )
    output = torch.empty((rows, value_dim), dtype=torch.bfloat16, device=query.device)
    qk_scale = scale * 1.44269504
    block_k = hl.register_block_size(query_dim)
    block_c = hl.register_block_size(split_capacity)

    for tile_s, tile_d in hl.tile([split_capacity, value_dim], block_size=[1, None]):
        seq_len = seq_lens[0]
        num_pages = (seq_len + page - 1) // page
        per_split = (num_pages + split_capacity - 1) // split_capacity
        begin = tile_s.begin * per_split
        end = torch.minimum(begin + per_split, num_pages)
        if begin < end:
            # Row q * H + h attends to positions < seq_len - (Q - 1) + q (causal
            # over the query tokens).
            limit = seq_len - (query_len - 1) + hl.arange(rows) // num_heads
            m_i = hl.full([rows], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([rows], dtype=torch.float32)
            acc = hl.zeros([rows, tile_d], dtype=torch.float32)
            for tile_j in hl.tile(begin, end, block_size=1):
                phys = block_tables[0, tile_j.begin]
                scores = hl.zeros([rows, page], dtype=torch.float32)
                for tile_k in hl.tile(query_dim, block_size=block_k):
                    scores = hl.dot(q2d[:, tile_k], kv3[phys, :, tile_k].T, acc=scores)
                pos = tile_j.begin * page + hl.arange(page)
                scores = torch.where(
                    pos[None, :] < limit[:, None], scores * qk_scale, float("-inf")
                )
                m_new = torch.maximum(m_i, torch.amax(scores, dim=-1))
                m_safe = torch.where(m_new == float("-inf"), 0.0, m_new)
                alpha = torch.exp2(m_i - m_safe)
                p = torch.exp2(scores - m_safe[:, None])
                l_i = l_i * alpha + torch.sum(p, dim=-1)
                v = k_lat3[phys, :, tile_d]
                acc = hl.dot(p.to(torch.bfloat16), v, acc=acc * alpha[:, None])
                m_i = m_new
            empty = l_i == 0
            safe_l = torch.where(empty, 1.0, l_i)
            partial[tile_s.begin, :, tile_d] = torch.where(
                empty[:, None], 0.0, acc / safe_l[:, None]
            )
            if tile_d.begin == 0:
                partial_lse[tile_s.begin, :] = torch.where(
                    empty, float("-inf"), m_i + torch.log2(safe_l)
                )

    # Combine over the live splits (masked loads skip the envelope's dead
    # splits): one pass for the maximum log-sum-exp, one for the weighted sum.
    for tile_r, tile_e in hl.tile([rows, value_dim]):
        seq_len = seq_lens[0]
        num_pages = (seq_len + page - 1) // page
        per_split = (num_pages + split_capacity - 1) // split_capacity
        live = (num_pages + per_split - 1) // per_split
        best = hl.full([tile_r], float("-inf"), dtype=torch.float32)
        for tile_c in hl.tile(split_capacity, block_size=block_c):
            if tile_c.begin < live:
                ok = tile_c.index < live
                lse = hl.load(partial_lse, [tile_c, tile_r], extra_mask=ok[:, None])
                lse = torch.where(ok[:, None], lse, float("-inf"))
                best = torch.maximum(best, torch.amax(lse, dim=0))
        best = torch.where(best == float("-inf"), 0.0, best)
        denom = hl.zeros([tile_r], dtype=torch.float32)
        out = hl.zeros([tile_r, tile_e], dtype=torch.float32)
        for tile_c in hl.tile(split_capacity, block_size=block_c):
            if tile_c.begin < live:
                ok = tile_c.index < live
                lse = hl.load(partial_lse, [tile_c, tile_r], extra_mask=ok[:, None])
                w = torch.where(ok[:, None], torch.exp2(lse - best[None, :]), 0.0)
                vals = hl.load(
                    partial, [tile_c, tile_r, tile_e], extra_mask=ok[:, None, None]
                )
                denom = denom + torch.sum(w, dim=0)
                out = out + torch.sum(w[:, :, None] * vals, dim=0)
        output[tile_r, tile_e] = (out / denom[:, None]).to(torch.bfloat16)

    return output.view(1, query_len, num_heads, value_dim)
