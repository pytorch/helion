"""GPU numerics for lane-invariant atomics (see the GPU-free companion)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from test.test_cute_lane_invariant_atomics import _FLAT
from test.test_cute_lane_invariant_atomics import _GRID_ROW
from test.test_cute_lane_invariant_atomics import _GRID_ROW_WIDE
from test.test_cute_lane_invariant_atomics import _LANES
from test.test_cute_lane_invariant_atomics import _NESTED
from test.test_cute_lane_invariant_atomics import _NESTED_3D
from test.test_cute_lane_invariant_atomics import _NESTED_INNER_SERIAL
from test.test_cute_lane_invariant_atomics import _NESTED_SCALAR
from test.test_cute_lane_invariant_atomics import _ROW
from test.test_cute_lane_invariant_atomics import _begin_count_then_copy
from test.test_cute_lane_invariant_atomics import _col_count_then_copy
from test.test_cute_lane_invariant_atomics import _convert_bytes_and_count
from test.test_cute_lane_invariant_atomics import _copy_count_copy_back
from test.test_cute_lane_invariant_atomics import _copy_then_release_count
from test.test_cute_lane_invariant_atomics import _copy_then_row_increment
from test.test_cute_lane_invariant_atomics import (
    _count_and_copy_first_column_in_inner_tile,
)
from test.test_cute_lane_invariant_atomics import _count_and_per_lane_add_in_inner_tile
from test.test_cute_lane_invariant_atomics import _count_in_inner_tile
from test.test_cute_lane_invariant_atomics import _count_then_copy
from test.test_cute_lane_invariant_atomics import _count_then_copy_3d
from test.test_cute_lane_invariant_atomics import _count_then_offset_the_copy
from test.test_cute_lane_invariant_atomics import _first_column_then_copy
from test.test_cute_lane_invariant_atomics import _first_element_then_copy
from test.test_cute_lane_invariant_atomics import _first_row_then_copy
from test.test_cute_lane_invariant_atomics import _flag_col_count_copy
from test.test_cute_lane_invariant_atomics import _flagged_col_count_then_copy
from test.test_cute_lane_invariant_atomics import _grid_count_then_copy
from test.test_cute_lane_invariant_atomics import _guarded_col_count_and_flag_then_copy
from test.test_cute_lane_invariant_atomics import _guarded_col_count_then_copy
from test.test_cute_lane_invariant_atomics import _loaded_index_count_then_copy
from test.test_cute_lane_invariant_atomics import _max_then_copy
from test.test_cute_lane_invariant_atomics import _per_lane_atomic
from test.test_cute_lane_invariant_atomics import _read_then_row_increment
from test.test_cute_lane_invariant_atomics import _row_count_then_copy
from test.test_cute_lane_invariant_atomics import _row_count_then_copy_3d
from test.test_cute_lane_invariant_atomics import _row_increment_then_read
from test.test_cute_lane_invariant_atomics import _segment_sums
from test.test_cute_lane_invariant_atomics import _sum_into_scalar
from test.test_cute_lane_loop_distribution import _TWO_SLICE_CONFIG
from test.test_cute_lane_loop_distribution import _atomic_into_the_copied_tensor
from test.test_cute_lane_loop_distribution import _atomic_then_copy

import helion
from helion import exc
from helion._testing import skipUnlessBackends

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

CUDA_DEVICE = "cuda"


def _run(kernel: object, args: tuple[object, ...], **config: object) -> Any:
    bound = kernel.bind(args)  # pyrefly: ignore [missing-attribute]
    return bound.compile_config(helion.Config.from_dict(config))(*args)


def _tiles(shape: tuple[int, ...], config: dict[str, object]) -> int:
    block_sizes = config["block_sizes"]
    assert isinstance(block_sizes, list)
    return math.prod(
        -(-size // block) for size, block in zip(shape, block_sizes, strict=True)
    )


_UNIFORM_CASES = [
    pytest.param(_count_then_copy, _LANES, (8, 256), id="lanes"),
    pytest.param(_count_then_copy, _LANES, (8, 250), id="lanes-masked"),
    pytest.param(_count_then_copy, _ROW, (8, 1024), id="vector_lane"),
    pytest.param(_count_then_copy, _NESTED, (8, 256), id="nested"),
    pytest.param(_count_then_copy, _NESTED, (6, 250), id="nested-masked"),
    pytest.param(_count_then_copy_3d, _NESTED_3D, (4, 8, 256), id="nested_3d"),
    pytest.param(_count_then_copy, _FLAT, (8, 2048), id="flattened"),
]


@pytest.mark.parametrize(("kernel", "config", "shape"), _UNIFORM_CASES)
def test_tile_uniform_count_is_once_per_tile(
    kernel: object, config: dict[str, object], shape: tuple[int, ...]
) -> None:
    x = torch.randn(shape, device=CUDA_DEVICE)
    counter = torch.zeros((1,), dtype=torch.int32, device=CUDA_DEVICE)
    out = _run(kernel, (x, counter), **config)
    assert counter.item() == _tiles(shape, config)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize(
    ("kernel", "config", "shape"),
    [
        pytest.param(_row_count_then_copy, _NESTED, (8, 256), id="rows"),
        pytest.param(_row_count_then_copy, _NESTED, (6, 250), id="rows-masked"),
        pytest.param(_row_count_then_copy_3d, _NESTED_3D, (4, 8, 256), id="rows_3d"),
    ],
)
def test_row_count_is_once_per_row_per_column_tile(
    kernel: object, config: dict[str, object], shape: tuple[int, ...]
) -> None:
    x = torch.randn(shape, device=CUDA_DEVICE)
    dims = len(shape) - 1
    counts = torch.zeros(shape[:dims], device=CUDA_DEVICE)
    out = _run(kernel, (x, counts), **config)
    block_sizes = config["block_sizes"]
    assert isinstance(block_sizes, list)
    column_tiles = -(-shape[-1] // block_sizes[-1])
    torch.testing.assert_close(counts, torch.full_like(counts, float(column_tiles)))
    torch.testing.assert_close(out, x)


def test_tile_begin_count_is_once_per_tile() -> None:
    x = torch.randn((8, 512), device=CUDA_DEVICE)
    counts = torch.zeros((8,), device=CUDA_DEVICE)
    out = _run(_begin_count_then_copy, (x, counts), **_NESTED)
    expected = torch.zeros_like(counts)
    expected[::4] = 2.0  # two column tiles per row tile of four
    torch.testing.assert_close(counts, expected)
    torch.testing.assert_close(out, x)


def test_float_max_matches_reference() -> None:
    x = torch.randn((8, 256), device=CUDA_DEVICE)
    best = torch.zeros((1,), device=CUDA_DEVICE)
    out = _run(_max_then_copy, (x, best), **_LANES)
    assert best.item() == 3.0
    torch.testing.assert_close(out, x)


def test_release_count_is_once_per_tile() -> None:
    x = torch.randn((8, 256), device=CUDA_DEVICE)
    counter = torch.zeros((1,), dtype=torch.int32, device=CUDA_DEVICE)
    out = _run(_copy_then_release_count, (x, counter), **_LANES)
    assert counter.item() == 8
    torch.testing.assert_close(out, x)


def test_per_element_atomics_match_reference() -> None:
    x = torch.randint(0, 5, (8, 256), device=CUDA_DEVICE).float()
    total = torch.zeros((1,), device=CUDA_DEVICE)
    out = _run(_sum_into_scalar, (x, total), **_LANES)
    assert total.item() == x.sum().item()
    torch.testing.assert_close(out, x)
    acc = torch.zeros_like(x)
    out = _run(_per_lane_atomic, (x, acc), **_LANES)
    torch.testing.assert_close(acc, x)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("packet_flush", [False, True], ids=["values", "packet"])
def test_byte_conversion_and_count_match_reference(packet_flush: bool) -> None:
    packed = torch.randint(-128, 128, (8, 256), dtype=torch.int8, device=CUDA_DEVICE)
    counter = torch.zeros((1,), dtype=torch.int32, device=CUDA_DEVICE)
    out = _run(
        _convert_bytes_and_count,
        (packed, counter),
        block_sizes=[1, 256],
        num_threads=[0, 32],
        cute_vector_widths=[1, 4],
        cute_signed_bitfield_bf16=packet_flush,
    )
    assert counter.item() == 8
    torch.testing.assert_close(out, packed.to(torch.bfloat16))


def test_per_lane_atomic_slice_next_to_a_copy_matches_reference() -> None:
    # The atomic's slice and the copy's slice run in sibling lane loops:
    # each element of x is accumulated exactly once.
    x = torch.randn((64, 512), device=CUDA_DEVICE)
    y = torch.randn((64, 768), device=CUDA_DEVICE)
    acc, out = _run(_atomic_then_copy, (x, y), **_TWO_SLICE_CONFIG)
    torch.testing.assert_close(acc, x)
    torch.testing.assert_close(out, y)
    out = _run(_atomic_into_the_copied_tensor, (x, y), **_TWO_SLICE_CONFIG)
    torch.testing.assert_close(out, torch.cat([x, y], dim=1))


def test_unplaceable_lane_invariant_atomics_reject_the_config() -> None:
    x = torch.randn((8, 256), device=CUDA_DEVICE)
    with pytest.raises(exc.BackendUnsupported, match="lane-invariant atomic on out"):
        _run(_copy_count_copy_back, (x,), **_LANES)
    counter = torch.zeros((1,), dtype=torch.int32, device=CUDA_DEVICE)
    with pytest.raises(exc.BackendUnsupported, match="leader thread"):
        _run(_count_then_offset_the_copy, (x, counter), **_LANES)


def _small_integers(shape: tuple[int, ...]) -> torch.Tensor:
    """Integer-valued floats: every partial sum below is exact."""
    return torch.randint(0, 5, shape, device=CUDA_DEVICE).float()


@pytest.mark.parametrize(
    ("kernel", "config", "shape", "rows", "columns"),
    [
        # value ``x[tile0.begin, tile1.begin]`` under a partial column tile:
        # no mask ties it to the lane loop, once per tile
        pytest.param(
            _first_element_then_copy, _LANES, (8, 250), 1, 256, id="masked_element"
        ),
        # value ``x[tile0, tile1.begin]``: per row, in the row loop only
        pytest.param(
            _first_column_then_copy, _NESTED, (6, 250), 1, 256, id="masked_column"
        ),
        # value ``x[tile0.begin, tile1]``: per column, nested in the row loop
        # and pinned to its first lane
        pytest.param(_first_row_then_copy, _NESTED, (8, 256), 4, 1, id="row_packet"),
    ],
)
def test_uniform_atomic_adds_each_tile_value_once(
    kernel: object,
    config: dict[str, object],
    shape: tuple[int, ...],
    rows: int,
    columns: int,
) -> None:
    x = _small_integers(shape)
    total = torch.zeros((1,), device=CUDA_DEVICE)
    out = _run(kernel, (x, total), **config)
    assert total.item() == x[::rows, ::columns].sum().item()
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize(
    ("kernel", "shape"),
    [
        pytest.param(_col_count_then_copy, (8, 256), id="column_count"),
        pytest.param(_col_count_then_copy, (6, 250), id="column_count-masked"),
        pytest.param(_flag_col_count_copy, (8, 256), id="column_count-placed"),
    ],
)
def test_column_count_is_once_per_row_tile(
    kernel: object, shape: tuple[int, ...]
) -> None:
    x = torch.randn(shape, device=CUDA_DEVICE)
    counts = torch.zeros((shape[1],), device=CUDA_DEVICE)
    flag = torch.zeros((1,), device=CUDA_DEVICE)
    args = (x, counts, flag) if kernel is _flag_col_count_copy else (x, counts)
    out = _run(kernel, args, **_NESTED)
    # Two row tiles of four count every column twice.
    torch.testing.assert_close(counts, torch.full_like(counts, 2.0))
    torch.testing.assert_close(out, x)
    if kernel is _flag_col_count_copy:
        assert flag.item() == 1.0


@pytest.mark.parametrize("config", [_NESTED_SCALAR, _NESTED], ids=["scalar", "vector"])
def test_pinned_row_increment_precedes_every_row_read(
    config: dict[str, object],
) -> None:
    out = torch.randn((4, 256), device=CUDA_DEVICE)
    expected = out.clone()
    expected[1] += 1.0
    out2 = torch.empty_like(out)
    result = _run(_row_increment_then_read, (out, out2), **config)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(out, expected)


def test_pinned_row_increment_after_a_per_lane_access_rejects_the_config() -> None:
    out = torch.randn((4, 256), device=CUDA_DEVICE)
    other = torch.empty_like(out)
    with pytest.raises(exc.BackendUnsupported, match="lane-invariant atomic on out"):
        _run(_read_then_row_increment, (out, other), **_NESTED_SCALAR)
    with pytest.raises(exc.BackendUnsupported, match="lane-invariant atomic on out"):
        _run(_copy_then_row_increment, (other, out), **_NESTED_SCALAR)


@pytest.mark.parametrize(
    ("config", "shape"),
    [
        pytest.param(_LANES, (8, 256), id="lanes"),
        pytest.param(_LANES, (8, 250), id="lanes-masked"),
        pytest.param(_NESTED, (8, 256), id="nested"),
        pytest.param(_NESTED, (6, 250), id="nested-masked"),
    ],
)
def test_loaded_scalar_index_count_is_once_per_tile(
    config: dict[str, object], shape: tuple[int, int]
) -> None:
    x = torch.randn(shape, device=CUDA_DEVICE)
    idx = torch.zeros((shape[0],), dtype=torch.int32, device=CUDA_DEVICE)
    counts = torch.zeros((shape[0],), device=CUDA_DEVICE)
    out = _run(_loaded_index_count_then_copy, (x, idx, counts), **config)
    assert counts[0].item() == _tiles(shape, config)
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize(
    ("config", "shape"),
    [
        pytest.param(_LANES, (8, 512), id="lanes"),
        pytest.param(_LANES, (8, 500), id="lanes-masked"),
        pytest.param(_NESTED, (8, 512), id="nested"),
        pytest.param(_NESTED, (6, 500), id="nested-masked"),
    ],
)
def test_row_count_in_an_inner_tile_loop_is_once_per_column_tile(
    config: dict[str, object], shape: tuple[int, int]
) -> None:
    x = torch.randn(shape, device=CUDA_DEVICE)
    counts = torch.zeros((shape[0],), device=CUDA_DEVICE)
    out = _run(_count_in_inner_tile, (x, counts), **config)
    block_sizes = config["block_sizes"]
    assert isinstance(block_sizes, list)
    column_tiles = -(-shape[1] // block_sizes[1])
    torch.testing.assert_close(counts, torch.full_like(counts, float(column_tiles)))
    torch.testing.assert_close(out, x)


def test_atomic_in_a_user_branch_is_once_per_tile() -> None:
    x = torch.randn((6, 250), device=CUDA_DEVICE)
    counts = torch.zeros((250,), device=CUDA_DEVICE)
    out = _run(_guarded_col_count_then_copy, (x, counts), **_NESTED)
    torch.testing.assert_close(counts, torch.ones_like(counts))
    torch.testing.assert_close(out, x)
    flag = torch.zeros((1,), device=CUDA_DEVICE)
    with pytest.raises(exc.BackendUnsupported, match="skip the rest of its statement"):
        _run(_guarded_col_count_and_flag_then_copy, (x, counts, flag), **_NESTED)


def test_uniform_atomic_beside_a_per_lane_atomic_rejects_the_config() -> None:
    x = torch.randn((8, 256), device=CUDA_DEVICE)
    y = torch.randn((8,), device=CUDA_DEVICE)
    counter = torch.zeros((1,), dtype=torch.int32, device=CUDA_DEVICE)
    out = torch.zeros((8, 256), device=CUDA_DEVICE)
    with pytest.raises(exc.BackendUnsupported, match="skip the rest of its statement"):
        _run(
            _count_and_per_lane_add_in_inner_tile,
            (x, y, counter, out),
            **_NESTED_INNER_SERIAL,
        )


@pytest.mark.parametrize("shape", [(8, 512), (8, 500)], ids=["full", "partial"])
def test_count_in_an_inner_tile_indexed_by_begin_is_once_per_column_tile(
    shape: tuple[int, int],
) -> None:
    x = torch.randn(shape, device=CUDA_DEVICE)
    counts = torch.zeros((shape[0],), device=CUDA_DEVICE)
    out = torch.full(shape, -7.0, device=CUDA_DEVICE)
    _run(_count_and_copy_first_column_in_inner_tile, (x, counts, out), **_LANES)
    torch.testing.assert_close(counts, torch.full_like(counts, 2.0))
    expected = torch.full(shape, -7.0, device=CUDA_DEVICE)
    expected[:, ::256] = x[:, ::256]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("config", [_NESTED, _NESTED_SCALAR], ids=["vector", "scalar"])
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_flagged_column_count_is_once_per_flagged_row_tile(
    config: dict[str, object], shape: tuple[int, int]
) -> None:
    x = torch.randn(shape, device=CUDA_DEVICE)
    flags = torch.zeros((shape[0],), device=CUDA_DEVICE)
    flags[0] = 1.0
    flags[4] = 1.0
    counts = torch.zeros((shape[1],), device=CUDA_DEVICE)
    out = _run(_flagged_col_count_then_copy, (x, flags, counts), **config)
    # Both row tiles of four start at a flagged row.
    torch.testing.assert_close(counts, torch.full_like(counts, 2.0))
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("config", [_GRID_ROW, _GRID_ROW_WIDE], ids=["warp", "cta"])
def test_a_grid_indexed_count_is_once_per_column_tile(
    config: dict[str, object],
) -> None:
    x = torch.randn((6, 100), device=CUDA_DEVICE)
    counts = torch.zeros((6,), device=CUDA_DEVICE)
    out = _run(_grid_count_then_copy, (x, counts), **config)
    block_sizes = config["block_sizes"]
    assert isinstance(block_sizes, list)
    assert counts.tolist() == [-(-100 // block_sizes[0])] * 6
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("config", [_GRID_ROW, _GRID_ROW_WIDE], ids=["warp", "cta"])
def test_segment_sums_under_a_grid_index_match_reference(
    config: dict[str, object],
) -> None:
    """Every thread used to add the warp's (or CTA's) total: 20 to 38 times the segment sum."""
    offsets = torch.tensor([0, 50, 50, 57, 89, 189, 192])
    x = torch.randn((192,), device=CUDA_DEVICE)
    out = _run(_segment_sums, (x, offsets.to(CUDA_DEVICE)), **config)
    expected = torch.stack(
        [x[int(offsets[i]) : int(offsets[i + 1])].sum() for i in range(6)]
    )
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
