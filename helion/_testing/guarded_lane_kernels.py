"""Kernels shared by the ``cute_guarded_lanes`` / ``cute_fmnmx_extrema`` tests
(``test_cute_guarded_lanes.py`` on CPU, ``test_cute_guarded_lanes_cuda.py``
on the GPU): windows over a channels-last activation, whose loads go through
computed coordinates or a padding ``extra_mask``."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def window_max(x: torch.Tensor) -> torch.Tensor:
    """2x2 max pooling, stride 2, of a ``[H, W, C]`` activation."""
    height, width, channels = x.shape
    out = torch.empty(
        [height // 2, width // 2, channels], dtype=x.dtype, device=x.device
    )
    for tile_h, tile_w, tile_c in hl.tile([height // 2, width // 2, channels]):
        row = (tile_h.index * 2)[:, None]
        col = (tile_w.index * 2)[None, :]
        best = x[row, col, tile_c].to(torch.float32)
        for offset in hl.static_range(1, 4):
            value = x[row + offset // 2, col + offset % 2, tile_c].to(torch.float32)
            best = torch.maximum(best, value)
        out[tile_h, tile_w, tile_c] = best.to(out.dtype)
    return out


def window_max_reference(x: torch.Tensor) -> torch.Tensor:
    """``window_max`` in eager PyTorch (fp32 maxima of the same elements)."""
    height, width, _ = x.shape
    value = x[: height // 2 * 2, : width // 2 * 2].to(torch.float32)
    best = value[0::2, 0::2]
    for offset in range(1, 4):
        best = torch.maximum(best, value[offset // 2 :: 2, offset % 2 :: 2])
    return best.to(x.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def padded_sum(x: torch.Tensor) -> torch.Tensor:
    """Each element of a ``[H, W, C]`` activation plus its upper and lower
    neighbours, the rows past the image reading as zero."""
    height, width, channels = x.shape
    out = torch.empty_like(x)
    for tile_h, tile_w, tile_c in hl.tile([height, width, channels]):
        h = tile_h.index
        above = (h >= 1)[:, None, None]
        below = (h + 1 < height)[:, None, None]
        up = hl.load(x, [h - 1, tile_w, tile_c], extra_mask=above)
        down = hl.load(x, [h + 1, tile_w, tile_c], extra_mask=below)
        total = (
            up.to(torch.float32)
            + x[tile_h, tile_w, tile_c].to(torch.float32)
            + down.to(torch.float32)
        )
        out[tile_h, tile_w, tile_c] = total.to(out.dtype)
    return out


def padded_sum_reference(x: torch.Tensor) -> torch.Tensor:
    """``padded_sum`` in eager PyTorch, adding in the kernel's order."""
    value = x.to(torch.float32)
    zeros = torch.zeros_like(value[:1])
    up = torch.cat([zeros, value[:-1]])
    down = torch.cat([value[1:], zeros])
    return ((up + value) + down).to(x.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def masked_accumulate(x: torch.Tensor, keep: torch.Tensor, y: torch.Tensor) -> None:
    """``x[r, c] = (x[r, c] if keep[r] else 0) + y[r, c]`` in place: the
    store writes the tensor the masked load reads."""
    rows, columns = x.shape
    for tile_r, tile_c in hl.tile([rows, columns]):
        kept = hl.load(x, [tile_r, tile_c], extra_mask=keep[tile_r][:, None])
        x[tile_r, tile_c] = (
            kept.to(torch.float32) + y[tile_r, tile_c].to(torch.float32)
        ).to(x.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def padded_and_transposed(
    x: torch.Tensor, out: torch.Tensor, out_t: torch.Tensor
) -> None:
    """``x`` padded with columns of zeros to the width of ``out``, written
    to ``out`` and, transposed, to ``out_t``: a lane's value goes to the
    packet stored along the columns and to a store of its own."""
    rows, padded = out.shape
    columns = x.size(1)
    for tile_r, tile_c in hl.tile([rows, padded]):
        in_range = (tile_c.index < columns)[None, :]
        value = hl.load(x, [tile_r, tile_c], extra_mask=in_range)
        value = torch.where(in_range, value, 0.0)
        out[tile_r, tile_c] = value
        out_t[tile_c, tile_r] = value.permute(1, 0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.maximum`` and ``torch.minimum`` of two fp32 vectors."""
    high = torch.empty_like(x)
    low = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        high[tile] = torch.maximum(x[tile], y[tile])
        low[tile] = torch.minimum(x[tile], y[tile])
    return high, low


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema_relu_clamp(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``torch.maximum`` of two fp32 vectors beside a ``relu`` and a
    ``clamp`` of one of them, which lower through the same extremum."""
    high = torch.empty_like(x)
    rectified = torch.empty_like(x)
    clamped = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        high[tile] = torch.maximum(x[tile], y[tile])
        rectified[tile] = torch.relu(x[tile])
        clamped[tile] = torch.clamp(x[tile], -1.0, 6.0)
    return high, rectified, clamped


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def sorted_kept_rows(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Each row of ``x`` that ``keep`` names, sorted; the other rows read
    as zeros: a padded load whose value a ``torch.sort`` takes."""
    out = torch.empty_like(x)
    for tile_r, tile_c in hl.tile(x.shape):
        kept = hl.load(x, [tile_r, tile_c], extra_mask=keep[tile_r][:, None])
        out[tile_r, tile_c] = torch.sort(kept, dim=-1).values
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def flags_reread_after_a_store(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverts ``flags`` in place and returns the inverted flags as ``x``'s
    dtype, and the elements of ``x`` whose flag is set before and after the
    inversion (none, so zeros): a load masked by the flag as read after the
    store, stored under the flag as read before it."""
    inverted = torch.empty_like(x)
    both = torch.zeros_like(x)
    for tile in hl.tile(x.size(0)):
        old = flags[tile]
        flags[tile] = ~old
        new = flags[tile]
        inverted[tile] = new.to(x.dtype)
        value = hl.load(x, [tile], extra_mask=new)
        hl.store(both, [tile], value, extra_mask=old)
    return inverted, both


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema_chains(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four chains of elementwise extrema of negated integers, each over an
    input of its own: two nested maxima, two nested minima, three minima
    through a value a static loop renames, and two maxima through a named
    value."""
    rows, columns = a.shape
    nested_max = torch.empty_like(a)
    nested_min = torch.empty_like(b)
    deep_min = torch.empty_like(c)
    named_max = torch.empty_like(d)
    for tile_r in hl.tile(rows):
        for tile_c in hl.tile(columns):
            value = a[tile_r, tile_c]
            nested_max[tile_r, tile_c] = torch.maximum(
                torch.maximum(-value, -(value * 2)), -(value + 1)
            )
            value = b[tile_r, tile_c]
            nested_min[tile_r, tile_c] = torch.minimum(
                torch.minimum(-value, -(value * 2)), -(value + 1)
            )
            value = c[tile_r, tile_c]
            lowest = -value
            for factor in hl.static_range(2, 5):
                lowest = torch.minimum(lowest, -(value * factor))
            deep_min[tile_r, tile_c] = lowest
            value = d[tile_r, tile_c]
            first = torch.maximum(-value, -(value * 2))
            named_max[tile_r, tile_c] = torch.maximum(first, -(value + 1))
    return nested_max, nested_min, deep_min, named_max


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema_outside_chains(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Elementwise extrema of negated integers that form no chain of one
    kind: a single maximum, a ``clamp`` between constants, a ``clamp``
    between negated values, and a maximum of a minimum."""
    rows, columns = x.shape
    single = torch.empty_like(x)
    clamped = torch.empty_like(x)
    bounded = torch.empty_like(x)
    mixed = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        for tile_c in hl.tile(columns):
            value = x[tile_r, tile_c]
            single[tile_r, tile_c] = torch.maximum(-value, -(value * 2))
            clamped[tile_r, tile_c] = torch.clamp(-value, -5, 7)
            bounded[tile_r, tile_c] = torch.clamp(
                -value, min=-(value * 2), max=-(value + 1)
            )
            mixed[tile_r, tile_c] = torch.maximum(
                torch.minimum(-value, -(value * 2)), -(value + 1)
            )
    return single, clamped, bounded, mixed


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema_bound_chains(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Chains of maxima without two ``torch.maximum``: a lower ``clamp``
    bound over a lower ``clamp`` bound, and a ``relu`` over a maximum."""
    rows, columns = x.shape
    clamped = torch.empty_like(x)
    rectified = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        for tile_c in hl.tile(columns):
            value = x[tile_r, tile_c]
            clamped[tile_r, tile_c] = torch.clamp(
                torch.clamp(-value, min=-(value * 2)), min=-(value + 1)
            )
            rectified[tile_r, tile_c] = torch.relu(torch.maximum(-value, -(value * 2)))
    return clamped, rectified


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema_carried_max(x: torch.Tensor, lowest: hl.constexpr) -> torch.Tensor:
    """Row maxima of ``-x`` accumulated by a ``torch.maximum`` per column of
    a serial loop: a chain across the loop's trips, with no reduction."""
    rows, columns = x.shape
    out = torch.empty([rows], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            best = torch.maximum(best, -x[tile_r, column])
        out[tile_r] = best
    return out
