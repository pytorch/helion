from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.chained_tcgen_stage import allocate_stages
from helion._compiler.cute.chained_tcgen_stage import shared_bytes
from helion._compiler.cute.chained_tcgen_stage import stage_geometry


@pytest.mark.parametrize(
    ("shape", "physical", "transpose"),
    [
        ((16, 128, 128), (128, 16, 128), True),
        ((32, 128, 64), (128, 32, 64), True),
        ((64, 128, 32), (128, 64, 32), True),
        ((128, 128, 16), (128, 128, 16), False),
        ((128, 160, 32), (128, 160, 32), False),
        ((16, 16, 16), (128, 16, 16), False),
        ((48, 72, 48), (128, 80, 48), False),
    ],
)
def test_physical_geometry_retains_logical_domains(
    shape: tuple[int, int, int], physical: tuple[int, int, int], transpose: bool
) -> None:
    geometry = stage_geometry(shape)
    assert geometry is not None
    assert geometry.logical == shape
    assert geometry.physical == physical
    assert geometry.transpose is transpose
    logical_m, logical_n, k = shape
    for row in range(logical_n if transpose else logical_m):
        for column in range(logical_m if transpose else logical_n):
            logical_row, logical_column = geometry.result_coordinates(
                str(row), str(column)
            )
            assert 0 <= int(logical_row) < logical_m
            assert 0 <= int(logical_column) < logical_n
            for reduction in (0, k - 1):
                a_index, a_coords = geometry.operand("a", str(row), str(reduction))
                b_index, b_coords = geometry.operand("b", str(column), str(reduction))
                lhs, rhs = (b_coords, a_coords) if transpose else (a_coords, b_coords)
                assert (a_index, b_index) == ((1, 0) if transpose else (0, 1))
                assert lhs == (logical_row, str(reduction))
                assert rhs == (str(reduction), logical_column)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 128, 16),
        (16, 0, 16),
        (16, 16, 0),
        (15, 128, 16),
        (16, 15, 16),
        (16, 128, 15),
        (256, 128, 16),
        (128, 264, 32),
    ],
)
def test_unsupported_physical_geometry_rejected(shape: tuple[int, int, int]) -> None:
    assert stage_geometry(shape) is None


def test_resident_allocations_are_per_region_not_per_iteration() -> None:
    shapes = ((16, 128, 128), (16, 128, 16), (128, 128, 16))
    geometries = tuple(stage_geometry(shape) for shape in shapes)
    assert all(item is not None for item in geometries)
    admitted = tuple(item for item in geometries if item is not None)
    source = "\n".join(allocate_stages(admitted))
    tree = ast.parse(source)
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    assert source.count("chain_allocator.allocate(") == 1
    assert source.count("mbarrier_init(chain_bars +") == len(shapes)
    # A=128x128 halves, B=16x128 halves; three logical FP32 results,
    # plus independently aligned completion barriers and allocator mailbox.
    assert shared_bytes(admitted) == 32768 + 4096 + 8192 + 8192 + 65536 + 256
