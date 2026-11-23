"""Sliding-window (unfold) kernels for the tests of their packet forms:
``out[n, k] = flat[n + channels * (k - pad)]`` for source positions inside
the sequence, zero in the padding."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def sliding_windows(x: torch.Tensor, window: int, pad: int) -> torch.Tensor:
    """The windows of a contiguous ``[batch, seq, channels]`` activation, into
    a fresh ``[batch * seq * channels, window]`` output."""
    window, pad = hl.specialize(window), hl.specialize(pad)
    _, seq, channels = x.shape
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n, window], dtype=x.dtype, device=x.device)
    for tile_n, tile_k in hl.tile([n, window]):
        i = tile_n.index
        k = tile_k.index
        src = ((i // channels) % seq)[:, None] + k[None, :] - pad
        in_seq = (src >= 0) & (src < seq)
        offset = i[:, None] + channels * (k[None, :] - pad)
        out[tile_n, tile_k] = hl.load(flat, [offset], extra_mask=in_seq)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def sliding_windows_into(
    flat: torch.Tensor, out: torch.Tensor, seq: int, channels: int, pad: int
) -> torch.Tensor:
    """The same windows of a 1-D ``flat`` into the caller's ``[n, window]``
    ``out``: either argument may be a view (an offset, a stride, a slice of a
    larger buffer, the other argument's storage)."""
    seq, channels, pad = (
        hl.specialize(seq),
        hl.specialize(channels),
        hl.specialize(pad),
    )
    n, window = out.shape
    for tile_n, tile_k in hl.tile([n, window]):
        i = tile_n.index
        k = tile_k.index
        src = ((i // channels) % seq)[:, None] + k[None, :] - pad
        in_seq = (src >= 0) & (src < seq)
        offset = i[:, None] + channels * (k[None, :] - pad)
        out[tile_n, tile_k] = hl.load(flat, [offset], extra_mask=in_seq)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def sliding_windows_and_copy(
    x: torch.Tensor, window: int, pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``sliding_windows`` with a second output: the kernel writes two
    tensors."""
    window, pad = hl.specialize(window), hl.specialize(pad)
    _, seq, channels = x.shape
    flat = x.view(-1)
    n = flat.size(0)
    out = torch.empty([n, window], dtype=x.dtype, device=x.device)
    first = torch.empty([n, window], dtype=x.dtype, device=x.device)
    for tile_n, tile_k in hl.tile([n, window]):
        i = tile_n.index
        k = tile_k.index
        src = ((i // channels) % seq)[:, None] + k[None, :] - pad
        in_seq = (src >= 0) & (src < seq)
        offset = i[:, None] + channels * (k[None, :] - pad)
        value = hl.load(flat, [offset], extra_mask=in_seq)
        out[tile_n, tile_k] = value
        first[tile_n, tile_k] = value
    return out, first


def sliding_windows_reference(
    flat: torch.Tensor, seq: int, channels: int, window: int, pad: int
) -> torch.Tensor:
    """What the kernels compute, in eager PyTorch: a window slot whose source
    position lies outside the sequence, or whose element lies outside
    ``flat``, is zero."""
    n = flat.numel()
    i = torch.arange(n, device=flat.device)
    k = torch.arange(window, device=flat.device)
    src = ((i // channels) % seq)[:, None] + k[None, :] - pad
    offset = i[:, None] + channels * (k[None, :] - pad)
    valid = (src >= 0) & (src < seq) & (offset >= 0) & (offset < n)
    gathered = flat[offset.clamp(0, n - 1)]
    return torch.where(valid, gathered, torch.zeros((), dtype=flat.dtype).to(flat))


def register_tile_config(
    *,
    threads: int,
    window_block: int,
    vec: int = 8,
    staged: bool = True,
    **overrides: object,
) -> dict[str, object]:
    """The register-tile layout of the gather: ``threads`` threads of one
    ``vec``-lane packet each along the gathered tile, every window slot a
    per-thread lane (one tile of ``window_block`` slots), the V-loop sunk
    into the slot loop and unrolled; with ``cute_staged_store`` unless
    ``staged`` is False."""
    return {
        "block_sizes": [vec * threads, window_block],
        "num_threads": [threads, 1],
        "cute_vector_widths": [vec, 1],
        "cute_vloop_sink": True,
        "cute_lane_unroll": 16,
        **({"cute_staged_store": True} if staged else {}),
        **overrides,
    }
