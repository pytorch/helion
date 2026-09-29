from __future__ import annotations

import ast
from math import prod
from types import SimpleNamespace

import pytest

from helion._compiler.cute.thread_block_projection import flatten_thread_coordinates
from helion._compiler.cute.thread_block_projection import update_launch_block


@pytest.mark.parametrize(
    "dimensions",
    [(1, 4, 16), (4, 16, 1), (16, 1, 4), (2, 4, 8), (8, 4, 2), (64, 1, 1), (1, 1, 64)],
)
def test_every_coordinate_and_warp_lane_owner_is_preserved(
    dimensions: tuple[int, int, int],
) -> None:
    source = (
        "x, y, z = cute.arch.thread_idx()\n"
        "dimensions = cute.arch.block_dim()\n"
        "coordinates = tuple(cute.arch.thread_idx()[axis] for axis in range(3))\n"
        "literal = (cute.arch.thread_idx()[0], cute.arch.thread_idx()[1], cute.arch.thread_idx()[2])\n"
        "rank = x + dimensions[0] * (y + dimensions[1] * z)\n"
        "record((coordinates, literal, dimensions, rank, cute.arch.warp_idx(), rank % 32))\n"
    )
    body = ast.parse(source).body
    assert flatten_thread_coordinates(body, dimensions)
    transformed = ast.unparse(ast.fix_missing_locations(ast.Module(body, [])))
    results = [[], []]
    for rank in range(prod(dimensions)):
        logical = (
            rank % dimensions[0],
            rank // dimensions[0] % dimensions[1],
            rank // (dimensions[0] * dimensions[1]),
        )
        for index, (program, coordinates, block) in enumerate(
            ((source, logical, dimensions), (transformed, (rank, 0, 0), (64, 1, 1)))
        ):
            arch = SimpleNamespace(
                thread_idx=lambda values=coordinates: values,
                block_dim=lambda values=block: values,
                warp_idx=lambda current=rank: current // 32,
            )
            exec(
                program,
                {
                    "cute": SimpleNamespace(arch=arch),
                    "cutlass": SimpleNamespace(Int32=int),
                    "record": results[index].append,
                },
            )
    assert results[0] == results[1]
    assert [row[3] for row in results[1]] == list(range(prod(dimensions)))


@pytest.mark.parametrize(
    "source",
    [
        "alias = cute.arch.thread_idx",
        "alias = cute.arch.block_dim",
        "index = cute.arch.thread_idx(loc=location)",
        "index = cute.arch.thread_idx(argument)",
    ],
)
def test_noncanonical_coordinate_calls_decline_before_mutation(source: str) -> None:
    body = ast.parse("index = cute.arch.thread_idx()[2]\n" + source).body
    original = ast.dump(ast.Module(body, []))
    assert not flatten_thread_coordinates(body, (1, 4, 16))
    assert ast.dump(ast.Module(body, [])) == original


def test_projection_preserves_unrelated_node_annotations() -> None:
    body = ast.parse("offset = 2\nindex = offset + cute.arch.thread_idx()[2]").body
    marker = object()
    vars(body[1])["_ownership_annotation"] = marker
    assert flatten_thread_coordinates(body, (1, 4, 16))
    assert vars(body[1])["_ownership_annotation"] is marker
    assert ast.unparse(body[1]) == "index = offset + cute.arch.thread_idx()[0] // 4"


def test_late_block_refresh_preserves_other_host_arguments_and_calls() -> None:
    module = ast.parse(
        "output = allocate()\n"
        "_launcher(kernel, grid, left, right, output, block=(1, 4, 16), option=option)\n"
        "if condition:\n"
        "    output = _launcher(kernel, grid, left, right, output, block=(1, 4, 16), option=option)\n"
        "_launcher(other_kernel, grid, output, block=(1, 4, 16))\n"
        "_launcher(kernel, grid, output, block=(1, 4, 16))\n"
    )
    conditional = module.body[2]
    assert isinstance(conditional, ast.If)
    for statement in (module.body[1], conditional.body[0], module.body[3]):
        vars(statement)["_is_kernel_call"] = True
    original = ast.dump(module)
    host_statements: list[ast.AST] = list(module.body)
    assert update_launch_block(host_statements, "kernel", (64, 1, 1)) == 2
    source = ast.unparse(module)
    assert source.count("block=(64, 1, 1)") == 2
    assert source.count("block=(1, 4, 16)") == 2
    assert (
        ast.dump(ast.parse(source.replace("block=(64, 1, 1)", "block=(1, 4, 16)")))
        == original
    )
