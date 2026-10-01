from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_pointwise_unroll import PointwiseUnroll
from helion._compiler.cute.chained_vector_ownership import plan_vector_ownership
from helion.exc import BackendUnsupported


@pytest.mark.parametrize("height", [1, 7, 32, 65])
@pytest.mark.parametrize("width", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("threads", [32, 128, 384])
def test_default_preserves_exact_source(height: int, width: int, threads: int) -> None:
    ownership = plan_vector_ownership((height, width), threads)
    assert ownership is not None
    columns = width // 8
    rows = threads // columns
    assert ownership.row_expression("thread", "step") == (
        f"thread // {columns} + step * {rows}"
    )
    assert ownership.base_expression("thread", "step") == f"thread % {columns} * 8"
    assert ownership.copy_indices("step") == "None, step, 0"
    assert ownership.trips == (height + rows - 1) // rows
    assert ownership.column_tiles == 1
    assert not ownership.changed
    assert ownership == plan_vector_ownership(
        (height, width), threads, tile_columns=width
    )


def test_explicit_column_retile_matches_reviewed_source() -> None:
    ownership = plan_vector_ownership((32, 128), 128, tile_columns=32)
    assert ownership is not None
    assert ownership.row_expression("thread", "step") == "thread // 4"
    assert ownership.base_expression("thread", "step") == "thread % 4 * 8 + step * 32"
    assert ownership.copy_indices("step") == "None, 0, step"
    assert ownership.thread_rows == 32
    assert ownership.thread_columns == 4
    assert ownership.row_tiles == 1
    assert ownership.column_tiles == ownership.trips == 4
    assert ownership.changed


@pytest.mark.parametrize("height", [1, 7, 32, 65])
@pytest.mark.parametrize(
    "width,tile", [(8, 8), (32, 8), (48, 16), (128, 32), (160, 32)]
)
@pytest.mark.parametrize("threads", [32, 128, 256, 384])
def test_expression_and_copy_coordinates_cover_every_cell_once(
    height: int, width: int, tile: int, threads: int
) -> None:
    ownership = plan_vector_ownership((height, width), threads, tile_columns=tile)
    assert ownership is not None
    row_code = compile(ownership.row_expression("thread", "step"), "<row>", "eval")
    base_code = compile(ownership.base_expression("thread", "step"), "<base>", "eval")
    copy_code = compile(ownership.copy_indices("step"), "<copy>", "eval")
    cells: Counter[tuple[int, int]] = Counter()
    padding: Counter[tuple[int, int]] = Counter()
    # The logical shape need not fill its physical allocation. Ownership must
    # preserve every logical cell and every separately zero-filled padded cell.
    logical_height, logical_width = max(1, height - 1), max(1, width - 3)
    logical: Counter[tuple[int, int]] = Counter()
    for step in range(ownership.trips):
        for thread in range(threads):
            variables = {"thread": thread, "step": step}
            row = eval(row_code, {"__builtins__": {}}, variables)
            base = eval(base_code, {"__builtins__": {}}, variables)
            vector, copy_row, copy_column = eval(
                copy_code, {"__builtins__": {}}, variables
            )
            assert vector is None
            assert 0 <= copy_row < ownership.row_tiles
            assert 0 <= copy_column < ownership.column_tiles
            # CopyTV's row-major thread tile and contiguous eight-value tile
            # must use exactly the same tile coordinates as scalar stores.
            copy_origin = (
                copy_row * ownership.thread_rows,
                copy_column * ownership.tile_columns,
            )
            thread_row, thread_column = divmod(thread, ownership.thread_columns)
            assert row == copy_origin[0] + thread_row
            assert base == copy_origin[1] + 8 * thread_column
            assert base % 8 == 0 and 0 <= base <= width - 8
            if row >= height:
                continue
            for element in range(8):
                cell = row, base + element
                cells[cell] += 1
                if row < logical_height and base + element < logical_width:
                    logical[cell] += 1
                else:
                    padding[cell] += 1
    expected = {(row, column) for row in range(height) for column in range(width)}
    assert set(cells) == expected
    assert set(cells.values()) == {1}
    expected_logical = {
        (row, column)
        for row in range(logical_height)
        for column in range(logical_width)
    }
    assert set(logical) == expected_logical
    assert set(padding) == expected - expected_logical


@pytest.mark.parametrize(
    "shape,threads,tile",
    [
        ([], 128, 0),
        ([32, 128], 128, 0),
        ((32,), 128, 0),
        ((32, 128, 1), 128, 0),
        ((True, 128), 128, 0),
        ((32, False), 128, 0),
        ((32.0, 128), 128, 0),
        ((32, "128"), 128, 0),
        ((0, 128), 128, 0),
        ((-1, 128), 128, 0),
        ((32, 0), 128, 0),
        ((32, -128), 128, 0),
        ((32, 128), True, 0),
        ((32, 128), 128.0, 0),
        ((32, 128), "128", 0),
        ((32, 128), 0, 0),
        ((32, 128), -128, 0),
        ((32, 128), 128, False),
        ((32, 128), 128, True),
        ((32, 128), 128, None),
        ((32, 128), 128, []),
        ((32, 128), 128, {}),
        ((32, 128), 128, 32.0),
        ((32, 128), 128, "32"),
        ((32, 128), 128, -32),
        ((32, 128), 128, 4),
        ((32, 128), 128, 24),
        ((32, 72), 128, 24),
        ((32, 128), 128, 256),
        ((32, 127), 128, 0),
        ((32, 48), 128, 0),
        ((32, 48), 128, 32),
        ((32, 128), 6, 32),
        ((32, 128), 8, 0),
    ],
)
def test_invalid_geometry_fails_closed(
    shape: object, threads: object, tile: object
) -> None:
    assert (
        plan_vector_ownership(
            cast("tuple[int, int]", shape),
            cast("int", threads),
            tile_columns=cast("int", tile),
        )
        is None
    )


def test_two_axis_iteration_and_original_unroll_contracts() -> None:
    ownership = plan_vector_ownership((65, 128), 128, tile_columns=32)
    assert ownership is not None
    assert (
        ownership.row_expression("thread", "step") == "thread // 4 + (step // 4) * 32"
    )
    assert (
        ownership.base_expression("thread", "step")
        == "thread % 4 * 8 + (step % 4) * 32"
    )
    assert ownership.copy_indices("step") == "None, step // 4, step % 4"
    assert ownership.trips == 12
    bounded = BoundedProducerUnroll(16)
    assert bounded.loop_factor(ownership.trips) == 12
    assert bounded.activated
    exact = PointwiseUnroll(4)
    assert exact.loop_factor(ownership.trips) == 4
    assert exact.activated
    incompatible = PointwiseUnroll(8)
    with pytest.raises(BackendUnsupported, match="whole number of unroll groups"):
        incompatible.loop_factor(ownership.trips)
    assert not incompatible.activated
    # Ownership itself neither mutates activation nor replaces the policy.
    assert ownership.trips == 12


def test_record_is_immutable_and_scalar_team_need_not_be_power_of_two() -> None:
    ownership = plan_vector_ownership((17, 128), 384, tile_columns=32)
    assert ownership is not None
    assert ownership.thread_rows == 96
    with pytest.raises(FrozenInstanceError):
        ownership.threads = 128  # pyrefly: ignore [read-only]
