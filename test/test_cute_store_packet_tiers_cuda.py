"""GPU test: ``cute_store_packet_tiers`` stores what the scalar code stores
(``test_cute_store_packet_tiers.py`` pins the generated tiers on CPU).  A
device loop that only stores along its tile, bf16 and fp32, at row lengths
that put the output rows at every 16-byte phase (even) and at every element
(odd: the narrowest tier and the scalar loop behind it), with one and two
packets per lane and a partial last tile; and into a caller's view whose
base is on an 8-byte boundary, on none, or whose columns are every other
element, with the elements around the view left alone; a loop that begins
at a caller's column and stores a value it loads per row (the narrower
tiers' own packet offsets); a loop that stores bf16 and fp32 rows at once,
the fp32 rows aligned or as unaligned as the bf16 ones; and a Philox
packet's stores."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.stored_row_kernels import doubled_rows
from helion._testing.stored_row_kernels import doubled_rows_into
from helion._testing.stored_row_kernels import doubled_rows_into_two_dtypes
from helion._testing.stored_row_kernels import planted_rows
from helion._testing.stored_row_kernels import random_rows_into
from helion._testing.stored_row_kernels import row_fill_from
from helion._testing.stored_row_kernels import row_view


@pytest.mark.parametrize(
    ("dtype", "columns", "width"),
    [
        (torch.bfloat16, 1021, 8),
        (torch.bfloat16, 1018, 16),
        (torch.float32, 1021, 4),
        (torch.float32, 1022, 8),
    ],
)
def test_rows_stored_in_tiers_match_eager(
    dtype: torch.dtype, columns: int, width: int
) -> None:
    """Every tier (and the scalar loop at an odd row length) writes the
    element the scalar code writes, and nothing beside the row; so does the
    code without the knob."""
    x = planted_rows(19, columns, dtype, DEVICE)
    config = {
        "block_sizes": [4, 32 * width],
        "num_threads": [0, 32],
        "cute_vector_widths": [1, width],
    }
    integer = torch.int16 if dtype is torch.bfloat16 else torch.int32
    for tiers in (True, False):
        kernel = helion.kernel(
            doubled_rows.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config(**config, cute_store_packet_tiers=tiers),
        )
        out = kernel(x)
        assert out.shape == (19, columns) and out.is_contiguous()
        assert torch.equal(out.view(integer), (x * 2.0).view(integer))


@pytest.mark.parametrize(
    ("dtype", "view", "width"),
    [
        # Eight bytes into the storage: four-lane packets and their two-lane
        # tier (the knob stays on), no eight-lane packet (it reads off).
        (torch.bfloat16, "4", 4),
        (torch.bfloat16, "4", 8),
        # Off every packet boundary, and columns that are not adjacent: the
        # knob reads off and every store is scalar.
        (torch.bfloat16, "1", 8),
        (torch.float32, "3", 4),
        (torch.float32, "strided", 4),
    ],
)
def test_rows_stored_into_a_view_match_eager(
    dtype: torch.dtype, view: str, width: int
) -> None:
    """The stored rows are the caller's view of rows 2042 elements apart (a
    multiple of no packet, so the rows start at changing phases): every
    element of the view is written as the scalar code writes it, and no
    element beside it, with the knob and without."""
    columns = 1018
    x = planted_rows(19, columns, dtype, DEVICE)
    wide = torch.empty(19, 2042, dtype=dtype, device=DEVICE)
    out = row_view(wide, view, columns)
    inside = torch.zeros(19, 2042, dtype=torch.bool, device=DEVICE)
    row_view(inside, view, columns).fill_(True)
    integer = torch.int16 if dtype is torch.bfloat16 else torch.int32
    for tiers in (True, False):
        wide.fill_(7.0)
        kernel = helion.kernel(
            doubled_rows_into.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                block_sizes=[4, 32 * width],
                num_threads=[0, 32],
                cute_vector_widths=[1, width],
                cute_store_packet_tiers=tiers,
            ),
        )
        kernel(x, out)
        assert torch.equal(out.contiguous().view(integer), (x * 2.0).view(integer)), (
            tiers
        )
        assert bool((wide[~inside] == 7.0).all()), tiers


@pytest.mark.parametrize("start", [2, 3, 4])
def test_rows_filled_from_a_callers_column_match_eager(start: int) -> None:
    """The tiles begin at a column the caller chooses and the stored value
    is loaded per row, so the store's lane-zero offset is the only thing
    checked at run time (the generated code has no packet load): every
    fragment is ``start`` past a multiple of eight lanes, so from column 2
    the two-lane tier runs, from column 4 the four-lane one (the two-lane
    one on the odd rows, 1018 elements apart) and from column 3 the scalar
    loop.  A narrower tier's packets each take the offset of their own first
    lane.  The columns ahead of ``start`` are left alone."""
    scale = torch.randn(19, device=DEVICE, dtype=torch.bfloat16)
    expected = torch.full((19, 1018), 7.0, device=DEVICE, dtype=torch.bfloat16)
    expected[:, start:] = (scale * 2.0)[:, None]
    for tiers in (True, False):
        kernel = helion.kernel(
            row_fill_from.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                block_sizes=[4, 256],
                num_threads=[0, 32],
                cute_vector_widths=[1, 8],
                cute_store_packet_tiers=tiers,
            ),
        )
        out = torch.full((19, 1018), 7.0, device=DEVICE, dtype=torch.bfloat16)
        code = kernel.bind((scale, out, start)).to_code()
        assert "cute.arch.load(" not in code
        assert ("% 2 == 0" in code) is tiers
        kernel(scale, out, start)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), tiers


@pytest.mark.parametrize("stride", [1024, 1018])
def test_rows_stored_in_two_element_sizes_match_eager(stride: int) -> None:
    """One device loop stores sixteen lanes as bf16 into rows 1018 elements
    apart and as fp32 into a view of rows ``stride`` apart; the two stores'
    tier chains differ in length (four tiers of bf16, three of fp32).  At
    1024 the fp32 rows are proven aligned and that store keeps its four-lane
    packets in every arm while the bf16 store narrows.  At 1018 both stores
    are guarded: the fp32 store reaches the end of its chain first and stays
    at its two-lane packets beside the bf16 store's two-lane ones in the
    last arm.  Every arm writes what the scalar code writes and nothing
    beside the rows."""
    columns = 1018
    x = planted_rows(19, columns, torch.float32, DEVICE)
    wide = torch.empty(19, stride, dtype=torch.float32, device=DEVICE)
    full = wide[:, :columns]
    for tiers in (True, False):
        half = torch.full((19, columns), 7.0, dtype=torch.bfloat16, device=DEVICE)
        wide.fill_(7.0)
        kernel = helion.kernel(
            doubled_rows_into_two_dtypes.fn,
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                block_sizes=[4, 512],
                num_threads=[0, 32],
                cute_vector_widths=[1, 16],
                cute_store_packet_tiers=tiers,
            ),
        )
        kernel(x, half, full)
        assert torch.equal(
            half.view(torch.int16), (x * 2.0).to(torch.bfloat16).view(torch.int16)
        ), tiers
        assert torch.equal(
            full.contiguous().view(torch.int32), (x + 1.0).view(torch.int32)
        ), tiers
        assert bool((wide[:, columns:] == 7.0).all()), tiers


@pytest.mark.parametrize("width", [4, 8])
def test_random_rows_stored_in_tiers_are_the_same_stream(width: int) -> None:
    """A Philox packet's stores (``cute_rng_packet``; no packet load beside
    them) into 1016 columns of fp32 rows 1018 elements apart: the packet is
    drawn on every row (its own guard is on the position in the stream,
    ``row * 1016 + column``), and the odd rows, off the 16-byte boundary,
    store 8-byte packets with the knob and scalars without.  Every element
    holds the value the code without the knob stores, which is the stream of
    the scalar code (no packets), and the elements beside the view are left
    alone."""
    outs = []
    for knobs in (
        {"cute_rng_packet": True, "cute_store_packet_tiers": True},
        {"cute_rng_packet": True, "cute_store_packet_tiers": False},
        {"cute_rng_packet": False},
    ):
        kernel = helion.kernel(
            random_rows_into.fn,
            backend="cute",
            static_shapes=True,
            cute_rng_stream="philox4",
            config=helion.Config(
                block_sizes=[4, 32 * width],
                num_threads=[0, 32],
                cute_vector_widths=[1, width],
                **knobs,
            ),
        )
        wide = torch.full((19, 1018), 7.0, device=DEVICE)
        out = wide[:, :1016]
        code = kernel.bind((out, 123)).to_code()
        assert ("philox_packet_" in code) is knobs["cute_rng_packet"]
        assert ("% 2 == 0" in code) is knobs.get("cute_store_packet_tiers", False)
        kernel(out, 123)
        assert bool((wide[:, 1016:] == 7.0).all()), knobs
        outs.append(out.contiguous())
    tiered, packed, scalar = outs
    assert bool(((scalar >= 0) & (scalar < 1)).all()) and scalar.unique().numel() > 1000
    assert torch.equal(tiered.view(torch.int32), packed.view(torch.int32))
    assert torch.equal(tiered.view(torch.int32), scalar.view(torch.int32))
