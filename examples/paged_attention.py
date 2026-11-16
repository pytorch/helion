"""
Paged Attention Example
=======================

Causal attention over a paged KV cache with variable-length queries, as used
by serving engines such as vLLM: keys and values live in a pool of fixed-size
pages, each sequence addresses its pages through a block table, and a batch
of decode and prefill requests is packed into one ``[total_q_tokens, ...]``
query tensor.

Based on the PyTorch blog post "Portable paged attention in Helion"
(https://pytorch.org/blog/portable-paged-attention-in-helion/) and on the
vLLM Triton attention backend ``triton_unified_attention.py``:
https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py

Structure of the kernel (GQA supported; a query tile covers all query heads
of one KV head, so a tile never straddles KV heads):

- **Flat q-block grid**: programs are indexed by a global q-block id and
  resolve their sequence through a vectorized rank reduction over
  ``cu_seqlens_q`` instead of a dense ``[num_seqs, heads, max_query_len]``
  grid, which over-provisions for skewed batches. Because the sequence index
  is computed on the device, the block table is passed flattened with an
  explicit row stride.
- **Online (flash-style) softmax** with ``exp2`` (``log2(e)`` is folded into
  the scale host-side) over KV tiles gathered page by page.
- **KV loop split** at the causal diagonal: page tiles entirely below the
  diagonal are visible to every query row of the block, so they run without
  the causal mask and the ``seq_len`` zero-fill; only the boundary tiles
  around the diagonal carry masks. Each loop gets its own block size.
- **Ladder specialization**: ``q_block_padded_size`` is the query length
  rounded up to a power of two (the ceiling of the q-block block size), so
  decode and prefill batches of different lengths autotune separately.
  At the same time, the ladder is required to not compile & autotune every
  different shape that happens to be scheduled by vLLM.
- **Default configs**: a small list of tile shapes covering the serving
  regimes. The autotuner measures them per specialization key and keeps
  the fastest, which keeps the example quick and deterministic — but the
  shapes are informed by measurements of the source kernel on sm_100 GPUs
  and are not optimal on other targets or workloads. Remove ``configs=``
  from the decorator to let the autotuner search the full space.

"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
import helion.language as hl

# %%
# Constants and Default Configs
# -----------------------------

# %%
# log2(e), folded into the softmax scale so the kernel can use exp2
LOG2E = 1.4426950408889634

# ceiling of the q_block_padded_size ladder
MAX_Q_BLOCK_PADDED = 8192


def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


# Default tile shapes, measured on the source kernel: block_sizes are
# (q_block, boundary pages, full pages), block_m is
# num_queries_per_kv * q_block and block_n is pages * page_size. The
# autotuner measures these per specialization key and keeps the fastest;
# configs whose q_block exceeds the ladder ceiling of a key (the prefill
# shapes under a decode key) are dropped automatically.
# Measured on sm_100 GPUs; not optimal for other targets or workloads —
# see the module docstring for the caveat and how to enable a full search.
DEFAULT_CONFIGS = [
    # prefill / mixed: wide query tile
    helion.Config(block_sizes=[32, 4, 4], num_warps=4, num_stages=2, pid_type="flat"),
    # short prefill: narrower KV tile
    helion.Config(block_sizes=[32, 2, 2], num_warps=4, num_stages=2, pid_type="flat"),
    # decode: minimal query tile, moderate KV tile
    helion.Config(block_sizes=[1, 4, 4], num_warps=4, num_stages=2, pid_type="flat"),
    # decode with a longer KV tile
    helion.Config(block_sizes=[1, 8, 8], num_warps=4, num_stages=2, pid_type="flat"),
]

# %%
# Paged Attention Kernel
# ----------------------


# %%
# Based on the kernel presented in the PyTorch blog post "Portable paged attention
# in Helion" (https://pytorch.org/blog/portable-paged-attention-in-helion/).
# Unlike a naive implementation, which materializes each sequence's full
# [q, kv] score matrix, this kernel runs an online softmax over one KV tile
# at a time and drops masks on tiles entirely below the causal diagonal.
@helion.kernel(
    configs=DEFAULT_CONFIGS,
    static_shapes=False,
    index_dtype=torch.int64,
)
def paged_attention_kernel(
    q: torch.Tensor,  # [total_q_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    block_tables_flat: torch.Tensor,  # [num_seqs * block_table_stride]
    block_table_stride: int,
    seq_lens: torch.Tensor,  # [num_seqs]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1] query-token offsets
    scale: float,  # sm_scale * LOG2E
    num_seqs: int,
    q_block_padded_size: hl.constexpr,
    num_seqs_padded: hl.constexpr,
) -> torch.Tensor:
    """
    Computes causal attention over a paged KV cache; see the module docstring.

    Args:
        q: Packed queries, where token ``cu_seqlens_q[s] + i`` is query
            position ``i`` of sequence ``s``.
        key_cache: Pooled keys [num_pages, page_size, num_kv_heads, head_dim].
        value_cache: Pooled values, same layout as ``key_cache``.
        block_tables_flat: The block table flattened to one row per sequence
            of length ``block_table_stride``.
        seq_lens: Total (context + query) length of each sequence.
        cu_seqlens_q: Exclusive scan of the per-sequence query lengths.
        scale: Softmax scale pre-multiplied by ``log2(e)``.
        num_seqs: Number of sequences in the batch.
        q_block_padded_size: Ladder bucket, ``next_power_of_2(max_query_len)``.
        num_seqs_padded: Width of the ``gid -> seq_idx`` reduction vector.

    Returns:
        Output tensor of shape [total_q_tokens, num_heads, head_dim].
    """
    num_heads = hl.specialize(q.size(1))
    head_dim = hl.specialize(q.size(2))
    num_kv_heads = hl.specialize(key_cache.size(2))
    page_size = hl.specialize(key_cache.size(1))
    num_queries_per_kv = hl.specialize(num_heads // num_kv_heads)
    assert page_size == value_cache.size(1)
    assert head_dim == key_cache.size(3) == value_cache.size(3)

    out = torch.empty_like(q)

    # pyrefly: ignore [bad-argument-type]
    q_block_size = hl.register_block_size(1, q_block_padded_size)
    num_pages_at_once = hl.register_block_size(1, 32)
    num_pages_at_once_full = hl.register_block_size(1, 32)

    # upper bound on the total number of q blocks: each sequence contributes
    # at most total_q_tokens // q_block_size + 1 blocks
    num_q_blocks = q.size(0) // q_block_size + num_seqs

    for q_blk_tile, tile_h in hl.tile(
        [num_q_blocks, num_heads], block_size=[1, num_queries_per_kv]
    ):
        gid = q_blk_tile.begin

        # vectorized gid -> seq_idx: cu_seqlens_q // q_block + s is monotone
        # in s, so the sequence owning this block is the rank of gid among
        # the per-sequence block starts
        # pyrefly: ignore [bad-argument-type]
        s_idx = hl.arange(num_seqs_padded)
        valid = s_idx < num_seqs
        cu = hl.load(cu_seqlens_q, [s_idx], extra_mask=valid)
        starts = cu // q_block_size + s_idx
        seq_idx = torch.sum(((starts <= gid) & valid).to(torch.int32)) - 1

        seq_len = seq_lens[seq_idx]
        query_start = cu_seqlens_q[seq_idx]
        query_end = cu_seqlens_q[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len
        q_block_offset = (gid - (query_start // q_block_size + seq_idx)) * q_block_size
        # flat row offset into the block table; the 2-D form cannot be
        # lowered when seq_idx is a device-computed scalar
        block_table_offset = seq_idx * block_table_stride

        # blocks past this sequence's last q block are padding
        if q_block_offset < query_len:
            block_m = num_queries_per_kv * q_block_size
            kv_head = tile_h.begin // num_queries_per_kv

            # q tile [q_block, num_queries_per_kv, head_dim] -> [block_m, head_dim]
            q_idx = query_start + q_block_offset + hl.arange(q_block_size)
            head_idx = tile_h.begin + hl.arange(num_queries_per_kv)
            q_mask = q_idx[:, None, None] < query_end
            qt = hl.load(q, [q_idx, head_idx, hl.arange(head_dim)], extra_mask=q_mask)
            qt = qt.flatten(start_dim=0, end_dim=1)

            m_i = hl.full([block_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([block_m], 1.0, dtype=torch.float32)
            acc = hl.zeros([block_m, head_dim], dtype=torch.float32)

            # causal bound: no row of this q block sees kv beyond
            # context_len + q_block_offset + q_block_size
            kv_bound = torch.minimum(
                context_len + q_block_offset + q_block_size, seq_len
            )
            num_blocks = (kv_bound + page_size - 1) // page_size

            # first kv position not visible to *every* row of the q block;
            # page tiles entirely below it need neither causal mask nor
            # seq_len zero-fill, so the KV loop splits in two. No NaN hazard:
            # if full_pages > 0 every row enters the boundary loop with
            # finite m_i; otherwise the first boundary tile contains kv
            # position 0, which is visible to every row.
            diag_start = context_len + q_block_offset + 1
            full_pages = (
                diag_start // (num_pages_at_once_full * page_size)
            ) * num_pages_at_once_full
            full_pages = torch.minimum(full_pages, num_blocks)

            # ── full tiles: no causal mask, no seq_len zero-fill ──────────
            for tile_p in hl.tile(full_pages, block_size=num_pages_at_once_full):
                page_ids = hl.load(
                    block_tables_flat,
                    [
                        block_table_offset
                        + tile_p.begin
                        + hl.arange(num_pages_at_once_full)
                    ],
                )
                page_ids = page_ids.view([num_pages_at_once_full]).to(torch.int64)
                block_n = num_pages_at_once_full * page_size

                k = key_cache[page_ids, :, kv_head, :]
                k = k.flatten(start_dim=0, end_dim=1).transpose(0, 1)
                v = value_cache[page_ids, :, kv_head, :]
                v = v.flatten(start_dim=0, end_dim=1)

                scores = hl.dot(qt, k, out_dtype=torch.float32) * scale
                m_new = torch.maximum(m_i, torch.amax(scores, 1))
                p = torch.exp2(scores - m_new[:, None])
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + torch.sum(p, 1)
                acc = acc * alpha[:, None]
                acc = hl.dot(p.to(v.dtype), v, out_dtype=torch.float32, acc=acc)
                m_i = m_new

            # ── boundary tiles around the diagonal: masked body ──────────
            for tile_p in hl.tile(full_pages, num_blocks, block_size=num_pages_at_once):
                page_ids = hl.load(
                    block_tables_flat,
                    [block_table_offset + tile_p.begin + hl.arange(num_pages_at_once)],
                )
                page_ids = page_ids.view([num_pages_at_once]).to(torch.int64)
                block_n = num_pages_at_once * page_size

                k = key_cache[page_ids, :, kv_head, :]
                k = k.flatten(start_dim=0, end_dim=1)
                v = value_cache[page_ids, :, kv_head, :]
                v = v.flatten(start_dim=0, end_dim=1)
                # absolute kv position of every column; zero-fill past
                # seq_len so the partial last page and the zero-padded tail
                # of the block table are harmless
                kv_pos = tile_p.begin * page_size + hl.arange(block_n)
                pad = hl.zeros([block_n, head_dim], dtype=k.dtype)
                k = torch.where(kv_pos[:, None] < seq_len, k, pad)
                v = torch.where(kv_pos[:, None] < seq_len, v, pad)
                k = k.transpose(0, 1)

                scores = hl.dot(qt, k, out_dtype=torch.float32) * scale
                # causal mask: query position i attends to kv < context_len + i + 1
                q_row = (q_block_offset + hl.arange(q_block_size)).repeat_interleave(
                    num_queries_per_kv, dim=0
                )
                scores = torch.where(
                    kv_pos[None, :] < context_len + q_row[:, None] + 1,
                    scores,
                    float("-inf"),
                )

                m_new = torch.maximum(m_i, torch.amax(scores, 1))
                p = torch.exp2(scores - m_new[:, None])
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + torch.sum(p, 1)
                acc = acc * alpha[:, None]
                acc = hl.dot(p.to(v.dtype), v, out_dtype=torch.float32, acc=acc)
                m_i = m_new

            acc = acc / l_i[:, None]
            hl.store(
                out,
                [q_idx, tile_h.index, hl.arange(head_dim)],
                acc.view([q_block_size, num_queries_per_kv, head_dim]),
                extra_mask=q_mask,
            )
    return out


# %%
def paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """
    Computes causal attention over a paged KV cache.

    Same argument contract as :func:`paged_attention_reference`; derives the
    specialization key and launches the kernel.
    """
    return paged_attention_kernel(
        q,
        key_cache,
        value_cache,
        # seq_idx is device-computed, so the block table travels flattened
        block_table.reshape(-1),
        block_table.stride(0),
        seq_lens,
        cu_seqlens_q,
        sm_scale * LOG2E,
        seq_lens.size(0),
        # ladder: one autotune bucket per power-of-two query length. It is
        # necessary because decode and prefill batches want opposite tile
        # shapes, so a single shared config would fit neither, and because
        # the bucket is also the ceiling of the q_block block size — without
        # raising it, long prefills cannot reach wide query tiles.
        q_block_padded_size=min(MAX_Q_BLOCK_PADDED, _next_power_of_2(max_query_len)),
        num_seqs_padded=_next_power_of_2(seq_lens.size(0)),
    )


# %%
# Reference Implementation
# ------------------------


# %%
def paged_attention_reference(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """Gathers each sequence's KV from its pages and runs SDPA per sequence."""
    page_size = key_cache.size(1)
    num_queries_per_kv = q.size(1) // key_cache.size(2)
    out = torch.empty_like(q)
    for s in range(seq_lens.numel()):
        start = int(cu_seqlens_q[s])
        end = int(cu_seqlens_q[s + 1])
        seq_len = int(seq_lens[s])
        num_pages = (seq_len + page_size - 1) // page_size
        pages = block_table[s, :num_pages].long()
        # [num_kv_heads, seq_len, head_dim], expanded to query heads for GQA
        k = key_cache[pages].flatten(0, 1)[:seq_len].transpose(0, 1).float()
        v = value_cache[pages].flatten(0, 1)[:seq_len].transpose(0, 1).float()
        k = k.repeat_interleave(num_queries_per_kv, dim=0)
        v = v.repeat_interleave(num_queries_per_kv, dim=0)
        qs = q[start:end].transpose(0, 1).float()  # [num_heads, q_len, head_dim]
        # query position i attends to kv positions <= context_len + i
        q_len = end - start
        q_pos = torch.arange(q_len, device=q.device)
        kv_pos = torch.arange(seq_len, device=q.device)
        mask = kv_pos[None, :] <= (seq_len - q_len) + q_pos[:, None]
        o = torch.nn.functional.scaled_dot_product_attention(
            qs, k, v, attn_mask=mask, scale=sm_scale
        )
        out[start:end] = o.transpose(0, 1).to(q.dtype)
    return out


# %%
# Testing
# -------


# %%
def test(
    query_lens: list[int] | None = None,
    seq_lens: list[int] | None = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> None:
    """Runs the kernel against the reference on the given batch."""
    # helion._testing hard-imports pytest; importing it lazily keeps this
    # module (and the kernel) importable without a test framework installed,
    # e.g. when a benchmark harness loads the kernel by file path.
    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE
    from helion._testing import run_example

    dtype = HALF_DTYPE if dtype is None else dtype
    device = DEVICE if device is None else device
    torch.manual_seed(0)
    # default batch: decodes with context, prefills with and without
    # context, and a ragged length
    if query_lens is None or seq_lens is None:
        query_lens = [1, 1, 1, 128, 512, 2048, 333, 1]
        seq_lens = [4096, 1023, 77, 512, 1024, 2048, 1000, 3]

    cu_seqlens_q = [0]
    for n in query_lens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + n)
    q = torch.randn(cu_seqlens_q[-1], num_heads, head_dim, dtype=dtype, device=device)

    # page pool with a disjoint page range per sequence
    pages_per_seq = [(n + page_size - 1) // page_size for n in seq_lens]
    key_cache = torch.randn(
        sum(pages_per_seq),
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros(
        len(seq_lens), max(pages_per_seq), dtype=torch.int32, device=device
    )
    page_start = 0
    for row, n in enumerate(pages_per_seq):
        block_table[row, :n] = torch.arange(page_start, page_start + n, device=device)
        page_start += n

    args = (
        q,
        key_cache,
        value_cache,
        block_table,
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device),
        max(query_lens),
        head_dim**-0.5,
    )
    run_example(paged_attention, paged_attention_reference, args)


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """Runs the example on a mixed batch and a pure-decode batch."""
    test()
    # pure decode: the ladder keys this batch at q_block_padded_size == 1,
    # so it autotunes separately from the prefill batch above
    test(query_lens=[1] * 32, seq_lens=[1024 + 7 * i for i in range(32)])


if __name__ == "__main__":
    main()
