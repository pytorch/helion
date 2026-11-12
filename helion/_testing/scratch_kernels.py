"""Kernels with host workspaces, for the kernel-local scratch tests.

Shared by ``test/test_scratch_tensors.py`` (which allocations are scratch)
and the CuTe tests of what is done with them.
"""

from __future__ import annotations

import torch

import helion.language as hl


def unread_workspace(x: torch.Tensor) -> torch.Tensor:
    """Each row stored to a workspace nothing reads back: its stores, its
    kernel parameter and its allocation are dead."""
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].float() * 2
        ws[tile_r, :] = v
        out[tile_r, :] = (v + 1).to(x.dtype)
    return out


def two_sweep_workspace(x: torch.Tensor) -> torch.Tensor:
    """Each row stored to a workspace and read back."""
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = x[tile_r, :].float() * 2
        out[tile_r, :] = (ws[tile_r, :] + 1).to(x.dtype)
    return out


def returned_unread_workspace(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``unread_workspace`` returning its workspace: the caller reads it."""
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].float() * 2
        ws[tile_r, :] = v
        out[tile_r, :] = (v + 1).to(x.dtype)
    return out, ws


def row_softmax_bwd(
    grad: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    workspace_like: torch.Tensor,
) -> torch.Tensor:
    """A softmax backward over rows: the first sweep stores each row's
    probabilities to a workspace of ``workspace_like``'s dtype and sums
    ``grad * probs``, the second reads them back (Inductor's looped-reduction
    shape)."""
    rows, n = logits.shape
    probs_ws = torch.empty([rows, n], dtype=workspace_like.dtype, device=logits.device)
    out = torch.empty_like(logits)
    for tile_r in hl.tile(rows):
        probs = (
            torch.exp(logits[tile_r, :].to(torch.float32) - row_max[tile_r][:, None])
            / row_sum[tile_r][:, None]
        )
        probs_ws[tile_r, :] = probs
        g = grad[tile_r, :].to(torch.float32)
        total = torch.sum(g * probs, dim=-1, keepdim=True)
        out[tile_r, :] = (probs_ws[tile_r, :].to(torch.float32) * (g - total)).to(
            out.dtype
        )
    return out


def row_softmax_bwd_keeping_workspace(
    grad: torch.Tensor,
    logits: torch.Tensor,
    row_max: torch.Tensor,
    row_sum: torch.Tensor,
    workspace_like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``row_softmax_bwd`` returning its workspace: the caller reads it, so
    the second sweep keeps reloading it."""
    rows, n = logits.shape
    probs_ws = torch.empty([rows, n], dtype=workspace_like.dtype, device=logits.device)
    out = torch.empty_like(logits)
    for tile_r in hl.tile(rows):
        probs = (
            torch.exp(logits[tile_r, :].to(torch.float32) - row_max[tile_r][:, None])
            / row_sum[tile_r][:, None]
        )
        probs_ws[tile_r, :] = probs
        g = grad[tile_r, :].to(torch.float32)
        total = torch.sum(g * probs, dim=-1, keepdim=True)
        out[tile_r, :] = (probs_ws[tile_r, :].to(torch.float32) * (g - total)).to(
            out.dtype
        )
    return out, probs_ws


def relu_spatial_mean(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """A BatchNorm-ReLU epilogue's shape whose ReLU output nothing reads
    back: each element is stored to ``act`` inside the rolled reduction that
    pools it, and only the pool is returned (the cast names ``x.dtype``, not
    ``act.dtype``, so that ``act`` is only subscripted)."""
    batch, rows, channels = x.shape
    act = torch.empty_like(x)
    pooled = torch.empty([batch, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c in hl.tile([batch, channels]):
        y = torch.relu(
            x[tile_n, :, tile_c].to(torch.float32) - shift[tile_c][None, None, :]
        )
        act[tile_n, :, tile_c] = y.to(x.dtype)
        pooled[tile_n, tile_c] = torch.mean(y, dim=1).to(pooled.dtype)
    return pooled


ROW_SWEEPS_CONFIG = {
    "block_sizes": [1],
    "num_threads": [1, 128],
    "reduction_loops": [1024],
}
"""For ``row_softmax_bwd``: each row's 1024 elements in one chunk of a
looped reduction on 128 threads (two sweeps)."""

RELU_SINK_CONFIG = {
    "block_sizes": [1, 64],
    "num_threads": [1, 8, 16],
    "cute_vector_widths": [1, 1, 8],
    "reduction_loops": [32],
    "cute_vloop_sink": True,
}
"""For ``relu_spatial_mean`` over ``[8, 784, 120]``: a ``[1, 64]`` tile of
eight vectors of eight channels and sixteen reduction threads over 32-row
chunks, the V-loop sunk into the reduction, so that ``act``'s eight lanes
leave as one packet."""
