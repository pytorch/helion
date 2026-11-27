# ruff: noqa: ANN001, ANN202
"""Helpers of the Qwen3.5 TP4 Gated-DeltaNet mixer megakernels (the mixer and the
merged MoE + GDN layer): in_proj's window rows from its K-chunk partials,
SGLang's ``causal_conv1d_update`` (state, window row, post-window state) and the
core's causal products.

A window token's results never read a later token's values: those terms are
selected out, not multiplied by zero, so a non-finite later (rejected) draft
cannot reach an earlier token's output or ring record (as in SGLang's kernels)."""

from __future__ import annotations

import torch

import helion.language as hl


def _shift(x, lanes, d):
    """``x[t + d]`` over the window lanes (0 off the window), exactly: one term."""
    pick = lanes[None, :] == lanes[:, None] + d
    return torch.sum(torch.where(pick[:, :, None], x[None, :, :], 0.0), dim=1)


def finite(x):
    """``x`` with its non-finite entries as 0, so a later product's zero terms stay
    zero: ``0 * NaN`` would carry a rejected draft's non-finite values into an
    earlier token's sums."""
    return torch.where(torch.abs(x) < float("inf"), x, 0.0)


def proj_rows(parts, chunks, scale, lanes, live, channels):
    """An in_proj output's window rows for ``channels``: BF16(sum over the K
    chunks' FP32 partials in chunk order * scale), as [lanes, channels] FP32
    holding the BF16 values (0 past the window)."""
    lane = torch.where(live, lanes, 0)
    acc = hl.zeros([lanes.size(0), channels.size(0)], dtype=torch.float32)
    for c in hl.static_range(chunks):
        acc = acc + hl.load(
            parts, [c, lane[:, None], channels[None, :]], extra_mask=live[:, None]
        )
    return (acc * scale).to(torch.bfloat16).to(torch.float32)


def conv_silu(x, state2, conv_weight, state_row0, channels, lanes, live):
    """SGLang's ``causal_conv1d_update`` (width 4, SiLU, no bias) of the window rows
    ``x`` [lanes, channels]: token t sums bf16(w_j * E[t + j]) in FP32 over E =
    [state0, state1, state2, x_0 ..], the output rounds to BF16. Returns [lanes,
    channels] FP32 holding those BF16 values."""
    acc = hl.zeros([lanes.size(0), channels.size(0)], dtype=torch.float32)
    for j in hl.static_range(4):
        src = lanes + j - 3
        from_state = live & (src < 0)
        s = hl.load(
            state2,
            [
                (state_row0 + channels)[None, :],
                torch.where(from_state, src + 3, 0)[:, None],
            ],
            extra_mask=from_state[:, None],
        ).to(torch.float32)
        e = torch.where(from_state[:, None], s, _shift(x, lanes, j - 3))
        w = hl.load(conv_weight, [channels, j]).to(torch.float32)
        acc = acc + (e * w[None, :]).to(torch.bfloat16).to(torch.float32)
    return (acc / (1.0 + torch.exp(-acc))).to(torch.bfloat16).to(torch.float32)


def store_conv_window(
    x, state2, window2, state_row0, window_row0, channels, lanes, live
):
    """The deduplicated verify window row ``[state[1], state[2], x_0 .. x_{T-1}]``
    of ``channels`` (SGLang's commit restores the accepted step's state from it)."""
    two = hl.arange(2)
    hl.store(
        window2,
        [(window_row0 + channels)[:, None], two[None, :]],
        hl.load(state2, [(state_row0 + channels)[:, None], (two + 1)[None, :]]),
    )
    hl.store(
        window2,
        [(window_row0 + channels)[None, :], (lanes + 2)[:, None]],
        x.to(torch.bfloat16),
        extra_mask=live[:, None],
    )


def store_conv_state(x, state2, state_row0, channels, lanes, live, tokens, state_len):
    """The post-window conv state ``[x_{T-3}, x_{T-2}, x_{T-1}]`` of ``channels``
    (the kernels assert ``tokens >= state_len``: no old column stays)."""
    tail = live & (lanes >= tokens - state_len)
    hl.store(
        state2,
        [
            (state_row0 + channels)[None, :],
            torch.where(tail, lanes - (tokens - state_len), 0)[:, None],
        ],
        x.to(torch.bfloat16),
        extra_mask=tail[:, None],
    )
