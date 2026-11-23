"""Kernels shared by the ``cute_predicated_element_loads`` tests
(``test_cute_predicated_element_loads.py`` on CPU,
``test_cute_predicated_element_loads_cuda.py`` on the GPU): loads under an
``extra_mask`` that gather along their tensor's last dim, so their lanes are
single elements under a mask at every config."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def padded_window_max(x: torch.Tensor, out: torch.Tensor) -> None:
    """3x3 max pooling, stride 2, padding 1, of a ``[P, H, W]`` image into
    ``out`` (``[P, (H - 1) // 2 + 1, (W - 1) // 2 + 1]``): every tap is
    masked to the image and reads as ``-inf`` in the padding."""
    planes, height, width = x.shape
    out_h = (height - 1) // 2 + 1
    out_w = (width - 1) // 2 + 1
    for tile_p, tile_h, tile_w in hl.tile([planes, out_h, out_w]):
        top = tile_h.index * 2 - 1
        left = tile_w.index * 2 - 1
        inside = ((top >= 0) & (top < height))[None, :, None] & (
            (left >= 0) & (left < width)
        )[None, None, :]
        value = hl.load(x, [tile_p, top[:, None], left[None, :]], extra_mask=inside).to(
            torch.float32
        )
        best = torch.where(inside, value, float("-inf"))
        for offset in hl.static_range(1, 9):
            row = top + offset // 3
            col = left + offset % 3
            inside = ((row >= 0) & (row < height))[None, :, None] & (
                (col >= 0) & (col < width)
            )[None, None, :]
            value = hl.load(
                x, [tile_p, row[:, None], col[None, :]], extra_mask=inside
            ).to(torch.float32)
            best = torch.maximum(best, torch.where(inside, value, float("-inf")))
        out[tile_p, tile_h, tile_w] = best.to(out.dtype)


def padded_window_shape(x: torch.Tensor) -> list[int]:
    """The shape of ``padded_window_max``'s output for the image ``x``."""
    planes, height, width = x.shape
    return [planes, (height - 1) // 2 + 1, (width - 1) // 2 + 1]


def padded_window_max_reference(x: torch.Tensor) -> torch.Tensor:
    """``padded_window_max`` in eager PyTorch: fp32 maxima of the same taps
    in the same order, the padding as ``-inf``."""
    height, width = x.shape[1:]
    out_h = (height - 1) // 2 + 1
    out_w = (width - 1) // 2 + 1
    padded = torch.nn.functional.pad(
        x.to(torch.float32), (1, 2, 1, 2), value=float("-inf")
    )
    best = None
    for offset in range(9):
        i, j = divmod(offset, 3)
        tap = padded[:, i : i + 2 * out_h : 2, j : j + 2 * out_w : 2]
        best = tap if best is None else torch.maximum(best, tap)
    assert best is not None
    return best.to(x.dtype)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def strided_pair_sum(x: torch.Tensor) -> torch.Tensor:
    """``x[r, c] + x[r, 2 * c]``, the element past the row reading as zero:
    a masked gather whose masked-off zero is added, not selected."""
    rows, columns = x.shape
    out = torch.empty([rows, columns], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([rows, columns]):
        col = tile_c.index * 2
        inside = col < columns
        far = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
        out[tile_r, tile_c] = x[tile_r, tile_c].to(torch.float32) + far.to(
            torch.float32
        )
    return out


def strided_pair_sum_reference(x: torch.Tensor) -> torch.Tensor:
    """``strided_pair_sum`` in eager PyTorch."""
    value = x.to(torch.float32)
    far = torch.zeros_like(value)
    far[:, : (value.size(1) + 1) // 2] = value[:, ::2]
    return value + far


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def gathered_or_fill(
    x: torch.Tensor, index: torch.Tensor, fill: hl.constexpr
) -> torch.Tensor:
    """``x[r, index[c]]`` where ``index[c]`` is a column of ``x`` and
    ``fill`` elsewhere, in ``x``'s dtype: a gather through a loaded index
    under a mask, selected against a constant without a cast."""
    rows, columns = x.shape
    out = torch.empty([rows, index.size(0)], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, index.size(0)]):
        col = index[tile_c]
        inside = (col >= 0) & (col < columns)
        value = hl.load(x, [tile_r, col], extra_mask=inside[None, :])
        out[tile_r, tile_c] = torch.where(  # pyrefly: ignore[no-matching-overload]
            inside[None, :], value, fill
        )
    return out


def gathered_or_fill_reference(
    x: torch.Tensor, index: torch.Tensor, fill: float
) -> torch.Tensor:
    """``gathered_or_fill`` in eager PyTorch."""
    inside = (index >= 0) & (index < x.size(1))
    value = x[:, index.clamp(0, x.size(1) - 1).long()]
    return torch.where(inside[None, :], value, torch.full_like(value, fill))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def gathered_under_row_mask(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``out[a, b] = x[a, index[b]]`` where ``index[a]``, the index of the
    row's lane, is a column of ``x``, and ``-1`` elsewhere, for a square
    ``[n, n]`` output: the mask's ``col < columns`` is about another lane
    than the one the load gathers through (``ok[:, None]`` puts the term
    over ``tile_b`` on ``tile_a``'s axis, which Helion accepts for tiles of
    one extent), so the load's own ``index < size`` bound is all that keeps
    a column past the row from being read.  A mask is bound to the
    positions of its tile, so the result is this one only where one tile
    holds the whole output; ``index`` holds no negative value (nothing
    checks ``index >= 0``)."""
    n, columns = x.shape
    out = torch.empty([n, n], dtype=x.dtype, device=x.device)
    for tile_a, tile_b in hl.tile([n, n]):
        col = index[tile_b]
        ok = (col >= 0) & (col < columns)
        value = hl.load(x, [tile_a, col], extra_mask=ok[:, None])
        out[tile_a, tile_b] = torch.where(ok[:, None], value, -1.0)
    return out


def gathered_under_row_mask_reference(
    x: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """``gathered_under_row_mask`` in eager PyTorch: a column past the row
    reads as the masked load's zero."""
    columns = x.size(1)
    ok = (index >= 0) & (index < columns)
    value = x[:, index.clamp(0, columns - 1).long()]
    value = torch.where((index < columns)[None, :], value, torch.zeros_like(value))
    return torch.where(ok[:, None], value, torch.full_like(value, -1.0))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def interior_or_fill_in_place(x: torch.Tensor, y: torch.Tensor) -> None:
    """``x[p, h, w] = (x[p, h, w] if (h, w) is inside the border else -1) +
    y[p, h, w]`` in place, each element read through computed coordinates
    that are no affine function of the tile (``2 * h // 2``): the store
    writes the tensor the masked gather reads."""
    planes, height, width = x.shape
    for tile_p, tile_h, tile_w in hl.tile([planes, height, width]):
        row = tile_h.index * 2 // 2
        col = tile_w.index * 2 // 2
        inside = ((row >= 1) & (row < height - 1))[None, :, None] & (
            (col >= 1) & (col < width - 1)
        )[None, None, :]
        value = hl.load(x, [tile_p, row[:, None], col[None, :]], extra_mask=inside).to(
            torch.float32
        )
        value = torch.where(inside, value, -1.0)
        x[tile_p, tile_h, tile_w] = (
            value + y[tile_p, tile_h, tile_w].to(torch.float32)
        ).to(x.dtype)


def interior_or_fill_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """What ``interior_or_fill_in_place`` leaves in ``x``, in eager PyTorch."""
    value = torch.full_like(x, -1.0, dtype=torch.float32)
    value[:, 1:-1, 1:-1] = x[:, 1:-1, 1:-1].to(torch.float32)
    return (value + y.to(torch.float32)).to(x.dtype)
