"""Grid lane loops around device loops, for ``cute_grid_lane_jam``.

Shared by the CuTe codegen tests (``test/test_cute_jam_grid_lanes_codegen.py``)
and their GPU twins (``test/test_cute_jam_grid_lanes_cuda.py``).  Under
``heads_config`` the eight heads of a tile are a lane loop of one thread
around the kernel's device loops over the keys.
"""

from __future__ import annotations

import torch

import helion.language as hl


def heads_config(key_loops: int, keys: int = 256) -> dict[str, object]:
    """One CTA per (batch, query): the eight heads as a thread's lane loop
    and, in each of ``key_loops`` device loops, ``keys`` keys on one thread
    per eight-wide packet (a warp for 256 keys, a two-warp shared-memory
    reduce for 512)."""
    return {
        "block_sizes": [1, 8, 1, *[keys] * key_loops],
        "num_threads": [1, 1, 1, *[keys // 8] * key_loops],
        "cute_vector_widths": [1, 1, 1, *[8] * key_loops],
        "cute_lane_layouts": ["blocked"] * (3 + key_loops),
    }


def bias_softmax(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """T5's relative-position softmax: an online max and sum over the keys,
    then the normalized probabilities.  ``bias`` is laid out heads-innermost
    (``heads_innermost_bias``), so the heads of one key are adjacent."""
    batch, heads, q_len, k_len = bias.shape
    out = torch.empty_like(scores)
    for tile_b, tile_h, tile_q in hl.tile([batch, heads, q_len]):
        row_max = hl.full([tile_b, tile_h, tile_q], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_b, tile_h, tile_q], dtype=torch.float32)
        for tile_k in hl.tile(k_len):
            x = scores[tile_b, tile_h, tile_q, tile_k].to(torch.float32) + bias[
                tile_b, tile_h, tile_q, tile_k
            ].to(torch.float32)
            new_max = torch.maximum(row_max, torch.amax(x, dim=-1))
            row_sum = row_sum * torch.exp(row_max - new_max) + torch.sum(
                torch.exp(x - new_max[:, :, :, None]), dim=-1
            )
            row_max = new_max
        for tile_k in hl.tile(k_len):
            x = scores[tile_b, tile_h, tile_q, tile_k].to(torch.float32) + bias[
                tile_b, tile_h, tile_q, tile_k
            ].to(torch.float32)
            probs = torch.exp(x - row_max[:, :, :, None]) / row_sum[:, :, :, None]
            out[tile_b, tile_h, tile_q, tile_k] = probs.to(out.dtype)
    return out


def nested_bias_exp(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``exp(scores + bias)`` with ``bias_softmax``'s heads-innermost bias, in
    a device loop over the keys nested in one over the queries."""
    batch, heads, q_len, k_len = bias.shape
    out = torch.empty_like(scores)
    for tile_b, tile_h in hl.tile([batch, heads]):
        for tile_q in hl.tile(q_len):
            for tile_k in hl.tile(k_len):
                x = scores[tile_b, tile_h, tile_q, tile_k].to(torch.float32) + bias[
                    tile_b, tile_h, tile_q, tile_k
                ].to(torch.float32)
                out[tile_b, tile_h, tile_q, tile_k] = torch.exp(x).to(out.dtype)
    return out


def leading_batch_bias_exp(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``exp(scores + bias)`` over a device loop of keys, with HF T5's
    batch-broadcast ``[1, heads, q, k]`` heads-innermost bias read as
    ``bias[0, ...]``."""
    batch, heads, q_len, k_len = scores.shape
    out = torch.empty_like(scores)
    for tile_b, tile_h, tile_q in hl.tile([batch, heads, q_len]):
        for tile_k in hl.tile(k_len):
            x = scores[tile_b, tile_h, tile_q, tile_k].to(torch.float32) + bias[
                0, tile_h, tile_q, tile_k
            ].to(torch.float32)
            out[tile_b, tile_h, tile_q, tile_k] = torch.exp(x).to(out.dtype)
    return out


def row_norm(x: torch.Tensor) -> torch.Tensor:
    """Two device loops over each row: its mean square, then the scaled row.
    Rows are ``x``'s outer dimension, so a row lane's elements are not
    adjacent to the next lane's: the jam interleaves the rows and batches
    their reduces, and loads nothing as a vector across them."""
    rows, n = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        total = hl.zeros([tile_r], dtype=torch.float32)
        for tile_n in hl.tile(n):
            v = x[tile_r, tile_n].to(torch.float32)
            total = total + torch.sum(v * v, dim=-1)
        scale = torch.rsqrt(total / n + 1e-6)
        for tile_n in hl.tile(n):
            v = x[tile_r, tile_n].to(torch.float32)
            out[tile_r, tile_n] = (v * scale[:, None]).to(out.dtype)
    return out


def kept_row_norm(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """``row_norm`` of the elements a boolean mask keeps: the mask's loads
    compare each loaded byte with zero, which compares no index."""
    rows, n = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        total = hl.zeros([tile_r], dtype=torch.float32)
        for tile_n in hl.tile(n):
            v = x[tile_r, tile_n].to(torch.float32) * keep[tile_r, tile_n].to(
                torch.float32
            )
            total = total + torch.sum(v * v, dim=-1)
        scale = torch.rsqrt(total / n + 1e-6)
        for tile_n in hl.tile(n):
            v = x[tile_r, tile_n].to(torch.float32) * keep[tile_r, tile_n].to(
                torch.float32
            )
            out[tile_r, tile_n] = (v * scale[:, None]).to(out.dtype)
    return out


ROW_NORM_CONFIG = {
    # Four rows as one thread's lanes, each loop's 512 columns on 64 threads.
    "block_sizes": [4, 512, 512],
    "num_threads": [1, 64, 64],
    "cute_vector_widths": [1, 8, 8],
    "cute_lane_layouts": ["blocked"] * 3,
}


def scale_rows_in_place(x: torch.Tensor) -> torch.Tensor:
    """Each row read back from the tensor it updates: every access names the
    row's head, so each head lane updates its own row."""
    batch, heads, q_len, k_len = x.shape
    for tile_b, tile_h, tile_q in hl.tile([batch, heads, q_len]):
        for tile_k in hl.tile(k_len):
            x[tile_b, tile_h, tile_q, tile_k] = x[tile_b, tile_h, tile_q, tile_k] * 2
    return x


def rows_into_two(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    """Each row of ``x`` stored twice, scaled, into ``a`` and into ``b``,
    every store naming the row's head."""
    batch, heads, q_len, k_len = x.shape
    for tile_b, tile_h, tile_q in hl.tile([batch, heads, q_len]):
        for tile_k in hl.tile(k_len):
            v = x[tile_b, tile_h, tile_q, tile_k]
            a[tile_b, tile_h, tile_q, tile_k] = v * 2
            b[tile_b, tile_h, tile_q, tile_k] = v * 3


def invariant_rows_over_heads(x: torch.Tensor, out: torch.Tensor) -> None:
    """``x[b, q, :]``, doubled, stored to every head's row of
    ``out[b, :, q, :]``: the stored values do not vary with the head."""
    batch, heads, q_len, k_len = out.shape
    for tile_b, tile_h, tile_q in hl.tile([batch, heads, q_len]):
        for tile_k in hl.tile(k_len):
            out[tile_b, tile_h, tile_q, tile_k] = (x[tile_b, tile_q, tile_k] * 2)[
                :, None, :, :
            ]


def heads_innermost_bias(
    batch: int, heads: int, q_len: int, k_len: int, device: torch.device | str
) -> torch.Tensor:
    """A ``[batch, heads, q, k]`` view of a ``[batch, q, k, heads]`` tensor."""
    return torch.randn(
        batch, q_len, k_len, heads, device=device, dtype=torch.bfloat16
    ).permute(0, 3, 1, 2)


SCALAR_ROWS_CONFIG = {
    # Four rows as one thread's lanes, each loop's 512 columns on 512 threads:
    # every load is one element, so both loops load ``x`` in the same form.
    "block_sizes": [4, 512, 512],
    "num_threads": [1, 512, 512],
    "cute_vector_widths": [1, 1, 1],
    "cute_lane_layouts": ["blocked"] * 3,
}


def doubled_rows_plus_one(x: torch.Tensor) -> torch.Tensor:
    """Each row doubled in place, then read back: the second loop reads what
    the first stored, not what the first loaded (``SCALAR_ROWS_CONFIG``)."""
    rows, n = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        for tile_n in hl.tile(n):
            x[tile_r, tile_n] = x[tile_r, tile_n] * 2
        for tile_n in hl.tile(n):
            out[tile_r, tile_n] = x[tile_r, tile_n] + 1
    return out


def head_bias_softmax_bwd(
    grad: torch.Tensor,
    scores: torch.Tensor,
    bias: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
) -> torch.Tensor:
    """T5's softmax backward over each (batch, head) row: the probabilities
    from the scores and the head's bias row, stored to a workspace that the
    row's second sweep reads back.  ``bias`` is ``[heads, q, k]`` laid out
    heads-innermost (``head_bias``), and a tile's rows ``bh`` are heads
    ``bh % heads`` of one batch, so the heads of one key are adjacent."""
    bh, q_len, k_len = scores.shape
    heads = bias.size(0)
    probs_ws = torch.empty(
        [bh, q_len, k_len], dtype=torch.float32, device=scores.device
    )
    out = torch.empty_like(scores)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        logits = scores[tile_bh, tile_q, :].to(torch.float32) + bias[head, tile_q, :]
        probs = (
            torch.exp(logits - row_max[tile_bh, tile_q][:, :, None])
            / row_sum[tile_bh, tile_q][:, :, None]
        )
        probs_ws[tile_bh, tile_q, :] = probs
        g = grad[tile_bh, tile_q, :].to(torch.float32)
        total = torch.sum(g * probs, dim=-1, keepdim=True)
        out[tile_bh, tile_q, :] = (probs_ws[tile_bh, tile_q, :] * (g - total)).to(
            out.dtype
        )
    return out


HEAD_ROWS_CONFIG = {
    "block_sizes": [4, 1],
    "num_threads": [1, 0, 128],
    "reduction_loops": [512],
    "cute_vector_widths": [1, 1, 1],
    "cute_reduction_reloads": ["gmem"],
}
"""For ``head_bias_softmax_bwd``: four heads as a thread's lane loop around a
512-key looped reduction on 128 threads, the second sweep reloading the
gradient from memory."""


def head_bias(
    heads: int, q_len: int, k_len: int, device: torch.device | str
) -> torch.Tensor:
    """A ``[heads, q, k]`` view of a ``[q, k, heads]`` fp32 tensor."""
    return torch.randn(q_len, k_len, heads, device=device).permute(2, 0, 1)


def hinted_head_rows(scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Each (batch * head) row of ``scores`` plus its head's ``bias`` row
    (``head_bias``), loaded with a cache hint, minus the row maximum."""
    bh, q_len, k_len = scores.shape
    heads = bias.size(0)
    out = torch.empty_like(scores)
    for tile_bh, tile_q in hl.tile([bh, q_len]):
        head = tile_bh.index % heads
        row = hl.load(bias, [head, tile_q, slice(None)], eviction_policy="evict_last")
        logits = scores[tile_bh, tile_q, :].to(torch.float32) + row
        out[tile_bh, tile_q, :] = (
            logits - torch.amax(logits, dim=-1, keepdim=True)
        ).to(out.dtype)
    return out
