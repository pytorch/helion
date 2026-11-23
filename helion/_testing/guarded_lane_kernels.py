"""Kernels shared by the ``cute_guarded_lanes`` / ``cute_fmnmx_extrema`` tests
(``test_cute_guarded_lanes.py`` on CPU, ``test_cute_guarded_lanes_cuda.py``
on the GPU): windows over a channels-last activation, whose loads go through
computed coordinates or a padding ``extra_mask``, and the filter for the
packet tile seeds of their searches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.config_spec import ConfigSpec


def packet_tile_seeds(spec: ConfigSpec) -> list[dict[str, object]]:
    """The compiler seeds of ``spec`` that turn ``cute_guarded_lanes`` on, as
    config dicts: the N-D packet tile seeds of the kernels these tests
    bind."""
    return [
        seed.config
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_guarded_lanes") is True
    ]


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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def window_max_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``window_max`` of ``x`` written to ``out``, a ``[H // 2, W // 2, C]``
    view in the caller's layout: a permuted view of a channel-first buffer
    has another dim than the channels contiguous."""
    out_h, out_w, channels = out.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        row = (tile_h.index * 2)[:, None]
        col = (tile_w.index * 2)[None, :]
        best = x[row, col, tile_c].to(torch.float32)
        for offset in hl.static_range(1, 4):
            value = x[row + offset // 2, col + offset % 2, tile_c].to(torch.float32)
            best = torch.maximum(best, value)
        out[tile_h, tile_w, tile_c] = best.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def neighbor_max_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The maxima of each element of a ``[H, W + 1, C]`` activation and its
    neighbor along W, written to ``out``, a ``[H, W, C]`` view in the
    caller's layout: a stride-1 window whose second tap is the tile's index
    plus one, a subscript that keeps its tile."""
    out_h, out_w, channels = out.shape
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        left = x[tile_h, tile_w, tile_c].to(torch.float32)
        right = x[tile_h, tile_w.index + 1, tile_c].to(torch.float32)
        out[tile_h, tile_w, tile_c] = torch.maximum(left, right).to(out.dtype)
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
def pooled_max(x: torch.Tensor, kernel_h: int, kernel_w: int) -> torch.Tensor:
    """``kernel_h`` x ``kernel_w`` max pooling, stride 2 and no padding, of a
    ``[H, W, C]`` activation: one fp32 maximum of ``kernel_h * kernel_w``
    loads per output element."""
    kernel_h = hl.specialize(kernel_h)
    kernel_w = hl.specialize(kernel_w)
    height, width, channels = x.shape
    out_h = (height - kernel_h) // 2 + 1
    out_w = (width - kernel_w) // 2 + 1
    out = torch.empty([out_h, out_w, channels], dtype=x.dtype, device=x.device)
    for tile_h, tile_w, tile_c in hl.tile([out_h, out_w, channels]):
        row = (tile_h.index * 2)[:, None]
        col = (tile_w.index * 2)[None, :]
        best = x[row, col, tile_c].to(torch.float32)
        for offset in hl.static_range(1, kernel_h * kernel_w):
            i = offset // kernel_w
            j = offset % kernel_w
            value = x[row + i, col + j, tile_c].to(torch.float32)
            best = torch.maximum(best, value)
        out[tile_h, tile_w, tile_c] = best.to(out.dtype)
    return out


def pooled_max_reference(x: torch.Tensor, kernel_h: int, kernel_w: int) -> torch.Tensor:
    """``pooled_max`` in eager PyTorch (fp32 maxima of the same elements)."""
    height, width, _ = x.shape
    out_h = (height - kernel_h) // 2 + 1
    out_w = (width - kernel_w) // 2 + 1
    value = x.to(torch.float32)
    best = value[0 : 2 * out_h - 1 : 2, 0 : 2 * out_w - 1 : 2]
    for offset in range(1, kernel_h * kernel_w):
        i, j = divmod(offset, kernel_w)
        tap = value[i : i + 2 * out_h - 1 : 2, j : j + 2 * out_w - 1 : 2]
        best = torch.maximum(best, tap)
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
def extrema_view_chains(
    a: torch.Tensor, c: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Chains of two extrema of negated integers whose links only a view or
    a copy separates, over ``a`` [rows], ``c`` [columns], ``x`` [rows,
    columns] and ``y`` [columns, rows]: a maximum broadcast by ``[:, None]``,
    a minimum by ``unsqueeze`` and ``expand``, a maximum through a full
    slice, a minimum through a ``permute``, and a minimum through a
    ``clone``.  Last, a maximum through a power of one: a value identity,
    which emits no statement either."""
    rows, columns = x.shape
    broadcast = torch.empty_like(x)
    expanded = torch.empty_like(x)
    sliced = torch.empty_like(x)
    permuted = torch.empty_like(x)
    copied = torch.empty_like(x)
    raised = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([rows, columns]):
        row = a[tile_r]
        column = c[tile_c]
        value = x[tile_r, tile_c]
        turned = y[tile_c, tile_r]
        inner = torch.maximum(-row, -(row * 2))
        broadcast[tile_r, tile_c] = torch.maximum(inner[:, None], -column[None, :])
        inner = torch.minimum(-column, -(column * 2))
        expanded[tile_r, tile_c] = torch.minimum(
            inner.unsqueeze(0).expand(row.size(0), -1), (-row).unsqueeze(1)
        )
        inner = torch.maximum(-value, -(value * 2))
        sliced[tile_r, tile_c] = torch.maximum(inner[:, :], -(value + 1))
        inner = torch.minimum(-turned, -(turned * 2))
        permuted[tile_r, tile_c] = torch.minimum(inner.permute(1, 0), -value)
        inner = torch.minimum(-value, -(value * 3))
        copied[tile_r, tile_c] = torch.minimum(inner.clone(), -(value - 1))
        inner = torch.maximum(-value, -(value * 3))
        raised[tile_r, tile_c] = torch.maximum(inner**1, -(value - 1))
    return broadcast, expanded, sliced, permuted, copied, raised


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema_carried_max_copies(
    x: torch.Tensor, lowest: hl.constexpr
) -> tuple[torch.Tensor, ...]:
    """``extrema_carried_max`` five times, with a copy at the loop's
    boundary: a ``clone`` of the value the body takes in and of the one it
    hands back, a ``repeat`` of one of each, and a flip of no dim of the one
    it hands back.  Then two loops with a maximum of a maximum of negated
    values in their body.  The first stores it per column, with a ``repeat``
    of one and a plain copy of the variable between the two maxima: nothing
    is carried.  The second hands the last column's on, with the sum of the
    first maximum and a difference that is zero and a plain copy between
    them: the later maximum reads a copy of a value computed in the body."""
    rows, columns = x.shape
    taken = torch.empty([rows], dtype=x.dtype, device=x.device)
    handed = torch.empty([rows], dtype=x.dtype, device=x.device)
    repeated_in = torch.empty([rows], dtype=x.dtype, device=x.device)
    repeated_out = torch.empty([rows], dtype=x.dtype, device=x.device)
    flipped_out = torch.empty([rows], dtype=x.dtype, device=x.device)
    chained = torch.empty_like(x)
    moved_last = torch.empty([rows], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            best = torch.maximum(best.clone(), -x[tile_r, column])
        taken[tile_r] = best
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            best = torch.maximum(best, -x[tile_r, column]).clone()
        handed[tile_r] = best
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            best = torch.maximum(best.repeat(1), -x[tile_r, column])
        repeated_in[tile_r] = best
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            best = torch.maximum(best, -x[tile_r, column]).repeat(1)
        repeated_out[tile_r] = best
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            best = torch.maximum(best, -x[tile_r, column]).flip(())
        flipped_out[tile_r] = best
        for column in hl.grid(columns):
            value = x[tile_r, column]
            inner = torch.maximum(-value, -(value * 2))
            same = inner.repeat(1)
            kept = same
            chained[tile_r, column] = torch.maximum(kept, -(value + 1))
        last = hl.zeros([tile_r], dtype=x.dtype)
        for column in hl.grid(columns):
            value = x[tile_r, column]
            inner = torch.maximum(-value, -(value * 2))
            moved = inner + (value - value)
            kept = moved
            last = torch.maximum(kept, -(value + 1))
        moved_last[tile_r] = last
    return (
        taken,
        handed,
        repeated_in,
        repeated_out,
        flipped_out,
        chained,
        moved_last,
    )


# The outputs of ``extrema_literal_identity_chains`` and of
# ``extrema_identity_chains`` in order, each with its extremum.  Chain ``i``
# of a kernel is ``op(op(-x, -(x * (i + 2))) <identity>, -(x + 1))``.
LITERAL_IDENTITY_CHAINS = (
    ("plus_zero", "max"),
    ("minus_zero", "min"),
    ("times_one", "max"),
    ("floor_divided_by_one", "min"),
    ("or_zero", "max"),
    ("xor_zero", "min"),
    ("and_all_ones", "max"),
    ("shifted_left_by_zero", "min"),
    ("shifted_right_by_zero", "max"),
)
IDENTITY_CHAINS = (
    ("rounded", "max"),
    ("floored", "min"),
    ("truncated", "max"),
    ("ceiled", "min"),
    ("selected", "max"),
    ("negated_twice", "min"),
    ("widened", "max"),
    ("negated_around_a_view", "min"),
    ("negated_around_a_copy", "max"),
    ("widened_around_a_view", "min"),
    ("widened_around_a_copy", "max"),
    ("negated_around_a_sum_with_zero", "min"),
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema_literal_identity_chains(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Chains of two extrema of negated integers with a value identity
    between the links: an operation whose literal operand leaves the other
    one as it is (``LITERAL_IDENTITY_CHAINS``, maxima and minima in turn).
    Every chain has operands of its own, so no link is shared."""
    rows, columns = x.shape
    plus = torch.empty_like(x)
    minus = torch.empty_like(x)
    times = torch.empty_like(x)
    divided = torch.empty_like(x)
    ored = torch.empty_like(x)
    xored = torch.empty_like(x)
    anded = torch.empty_like(x)
    left = torch.empty_like(x)
    right = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([rows, columns]):
        value = x[tile_r, tile_c]
        last = -(value + 1)
        inner = torch.maximum(-value, -(value * 2))
        plus[tile_r, tile_c] = torch.maximum(inner + 0, last)
        inner = torch.minimum(-value, -(value * 3))
        minus[tile_r, tile_c] = torch.minimum(inner - 0, last)
        inner = torch.maximum(-value, -(value * 4))
        times[tile_r, tile_c] = torch.maximum(inner * 1, last)
        inner = torch.minimum(-value, -(value * 5))
        divided[tile_r, tile_c] = torch.minimum(inner // 1, last)
        inner = torch.maximum(-value, -(value * 6))
        ored[tile_r, tile_c] = torch.maximum(inner | 0, last)
        inner = torch.minimum(-value, -(value * 7))
        xored[tile_r, tile_c] = torch.minimum(inner ^ 0, last)
        inner = torch.maximum(-value, -(value * 8))
        anded[tile_r, tile_c] = torch.maximum(inner & -1, last)
        inner = torch.minimum(-value, -(value * 9))
        left[tile_r, tile_c] = torch.minimum(inner << 0, last)
        inner = torch.maximum(-value, -(value * 10))
        right[tile_r, tile_c] = torch.maximum(inner >> 0, last)
    return plus, minus, times, divided, ored, xored, anded, left, right


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema_identity_chains(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Chains of two extrema of negated integers with a value identity
    between the links (``IDENTITY_CHAINS``, maxima and minima in turn): the
    four roundings, ``torch.where`` with one value for both branches, the
    negation of a negation, and a conversion to int64 and back; then the
    last two with a view (a full slice) and with a copy (a ``clone``)
    between the two negations and between the two conversions, and two
    negations with another identity (``+ 0``) between them."""
    rows, columns = x.shape
    rounded = torch.empty_like(x)
    floored = torch.empty_like(x)
    truncated = torch.empty_like(x)
    ceiled = torch.empty_like(x)
    selected = torch.empty_like(x)
    negated = torch.empty_like(x)
    widened = torch.empty_like(x)
    negated_view = torch.empty_like(x)
    negated_copy = torch.empty_like(x)
    widened_view = torch.empty_like(x)
    widened_copy = torch.empty_like(x)
    negated_sum = torch.empty_like(x)
    for tile_r, tile_c in hl.tile([rows, columns]):
        value = x[tile_r, tile_c]
        last = -(value + 1)
        inner = torch.maximum(-value, -(value * 2))
        rounded[tile_r, tile_c] = torch.maximum(torch.round(inner), last)
        inner = torch.minimum(-value, -(value * 3))
        floored[tile_r, tile_c] = torch.minimum(torch.floor(inner), last)
        inner = torch.maximum(-value, -(value * 4))
        truncated[tile_r, tile_c] = torch.maximum(torch.trunc(inner), last)
        inner = torch.minimum(-value, -(value * 5))
        ceiled[tile_r, tile_c] = torch.minimum(torch.ceil(inner), last)
        inner = torch.maximum(-value, -(value * 6))
        selected[tile_r, tile_c] = torch.maximum(
            torch.where(value > 0, inner, inner), last
        )
        inner = torch.minimum(-value, -(value * 7))
        negated[tile_r, tile_c] = torch.minimum(torch.neg(torch.neg(inner)), last)
        inner = torch.maximum(-value, -(value * 8))
        widened[tile_r, tile_c] = torch.maximum(inner.to(torch.int64).to(x.dtype), last)
        inner = torch.minimum(-value, -(value * 9))
        negated_view[tile_r, tile_c] = torch.minimum(
            torch.neg(torch.neg(inner)[:, :]), last
        )
        inner = torch.maximum(-value, -(value * 10))
        negated_copy[tile_r, tile_c] = torch.maximum(
            torch.neg(torch.neg(inner).clone()), last
        )
        inner = torch.minimum(-value, -(value * 11))
        widened_view[tile_r, tile_c] = torch.minimum(
            inner.to(torch.int64)[:, :].to(x.dtype), last
        )
        inner = torch.maximum(-value, -(value * 12))
        widened_copy[tile_r, tile_c] = torch.maximum(
            inner.to(torch.int64).clone().to(x.dtype), last
        )
        inner = torch.minimum(-value, -(value * 13))
        negated_sum[tile_r, tile_c] = torch.minimum(
            torch.neg(torch.neg(inner) + 0), last
        )
    return (
        rounded,
        floored,
        truncated,
        ceiled,
        selected,
        negated,
        widened,
        negated_view,
        negated_copy,
        widened_view,
        widened_copy,
        negated_sum,
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extrema_carried_max_identities(
    x: torch.Tensor, lowest: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    """``extrema_carried_max`` twice, with a value identity at the loop's
    boundary: a rounding of the value the body takes in, and a power of one
    of the value it hands back."""
    rows, columns = x.shape
    rounded = torch.empty([rows], dtype=x.dtype, device=x.device)
    raised = torch.empty([rows], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            best = torch.maximum(torch.round(best), -x[tile_r, column])
        rounded[tile_r] = best
        # pyrefly: ignore [bad-argument-type]
        best = hl.full([tile_r], lowest, dtype=x.dtype)
        for column in hl.grid(columns):
            best = torch.maximum(best, -x[tile_r, column]) ** 1
        raised[tile_r] = best
    return rounded, raised


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
