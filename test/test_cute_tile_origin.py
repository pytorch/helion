"""CPU coordinate invariants for tiled CuTe scalar and vector lane layouts."""

from __future__ import annotations

import ast
from collections import Counter
from collections import defaultdict
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.int4_gemm import matmul_bf16_int4
import pytest
import torch

from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _decorators
from helion.language import tile_ops

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.inductor_lowering import CodegenState


@pytest.fixture(autouse=True)
def _cpu_b200() -> Iterator[None]:
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


@pytest.fixture
def emitted_origins() -> Iterator[dict[int, set[str]]]:
    """Observe the actual backend result, including an inlined tile offset."""
    origins: dict[int, set[str]] = defaultdict(set)
    api = tile_ops.tile_begin
    assert _decorators.is_api_func(api)
    handler = api._codegen["cute"]

    def record(state: CodegenState) -> ast.AST:
        result = handler(state)
        assert isinstance(result, ast.Name)
        tile = state.proxy_arg(0)
        assert isinstance(tile, torch.SymInt)
        block_id = tile_ops._resolve_tile_block_id(tile, state)
        assert block_id is not None
        origins[block_id].add(result.id)
        return result

    with patch.dict(api._codegen, {"cute": record}):
        yield origins


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grid_origins(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.int32)
    for row, column in hl.tile(x.shape):
        out[row, column] = row.begin * x.size(1) + column.begin
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _loop_origins(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.int32)
    for row in hl.tile(x.size(0)):
        for column in hl.tile(3, x.size(1)):
            out[row, column] = column.begin
    return out


def _region(source: str, index: int) -> str:
    tree = ast.parse(source)
    regions = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "PyCodeCache.load"
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]
    return regions[index]


def _block(tree: ast.Module) -> tuple[int, int, int]:
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    return ast.literal_eval(
        next(kw.value for kw in launch.keywords if kw.arg == "block")
    )


def _evaluate(
    name: str, assignments: dict[str, ast.expr], scope: dict[str, object]
) -> int:
    if name in scope:
        value = scope[name]
        assert isinstance(value, int)
        return value
    expression = assignments[name]
    for node in ast.walk(expression):
        if (
            isinstance(node, ast.Name)
            and node.id in assignments
            and node.id not in scope
        ):
            scope[node.id] = _evaluate(node.id, assignments, scope)
    value = eval(ast.unparse(expression), scope)
    assert isinstance(value, int)
    scope[name] = value
    return value


def _axis_coordinates(
    source: str, block_id: int, offset: int, emitted_origins: dict[int, set[str]]
) -> list[tuple[int, int]]:
    """Evaluate actual emitted index/origin assignments over their lane loops."""
    tree = ast.parse(source)
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(device)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    index_name = f"indices_{block_id}"
    offset_name = f"tile_offset_{block_id}"
    origins = emitted_origins[block_id]
    assert len(origins) == 1, (block_id, origins, source)
    origin_name = next(iter(origins))
    assert any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == origin_name
        for node in ast.walk(device)
    )
    # The emitted expression determines its physical axis, not the config's
    # logical order. Inline the index's prerequisites to find that axis.
    needed = {index_name}
    while True:
        expanded = needed | {
            node.id
            for name in needed
            if name in assignments and name != offset_name
            for node in ast.walk(assignments[name])
            if isinstance(node, ast.Name)
        }
        if expanded == needed:
            break
        needed = expanded
    thread_axes = {
        ast.literal_eval(node.slice)
        for name in needed
        if name in assignments and name != offset_name
        for node in ast.walk(assignments[name])
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.arch.thread_idx"
    }
    assert len(thread_axes) == 1, thread_axes
    axis = thread_axes.pop()
    ranges = {}
    for node in ast.walk(device):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id in {f"lane_{block_id}", f"vec_lane_{block_id}"}
        ):
            assert isinstance(node.iter, ast.Call)
            ranges[node.target.id] = ast.literal_eval(node.iter.args[0])
    results = []
    for thread in range(_block(tree)[axis]):
        thread_idx = [0, 0, 0]
        thread_idx[axis] = thread
        for lane in range(ranges.get(f"lane_{block_id}", 1)):
            for vector_lane in range(ranges.get(f"vec_lane_{block_id}", 1)):
                scope: dict[str, object] = {
                    "cutlass": SimpleNamespace(Int32=int, Int64=int),
                    "cute": SimpleNamespace(
                        arch=SimpleNamespace(thread_idx=lambda value=thread_idx: value)
                    ),
                    offset_name: offset,
                    f"lane_{block_id}": lane,
                    f"vec_lane_{block_id}": vector_lane,
                }

                index = _evaluate(index_name, assignments, scope)
                origin = _evaluate(origin_name, assignments, scope)
                results.append((index, origin))
    return results


@pytest.mark.parametrize("layout", ("blocked", "strided"))
@pytest.mark.parametrize("vector", (1, 2, 4, 8))
@pytest.mark.parametrize("order", ([0, 1], [1, 0]))
def test_grid_tile_origins_match_every_emitted_lane(
    layout: str, vector: int, order: list[int], emitted_origins: dict[int, set[str]]
) -> None:
    bound = _grid_origins._bind_isolated((torch.empty((137, 47)),))
    source = bound.to_code(
        helion.Config(
            block_sizes=[64, 32],
            num_threads=[4, 2],
            cute_vector_widths=[vector, vector],
            cute_lane_layouts=[layout, layout],
            loop_orders=[order],
        )
    )
    for block_id, block_size in enumerate((64, 32)):
        for offset in (0, block_size, block_size * 2):
            records = _axis_coordinates(source, block_id, offset, emitted_origins)
            assert Counter(index for index, _ in records) == Counter(
                range(offset, offset + block_size)
            )
            assert {origin for _, origin in records} == {offset}


@pytest.mark.parametrize("layout", ("blocked", "strided"))
@pytest.mark.parametrize("vector", (1, 2, 4, 8))
def test_device_tile_origin_keeps_its_nonzero_begin(
    layout: str, vector: int, emitted_origins: dict[int, set[str]]
) -> None:
    bound = _loop_origins._bind_isolated((torch.empty((3, 139)),))
    source = bound.to_code(
        helion.Config(
            block_sizes=[1, 64],
            num_threads=[1, 4],
            cute_vector_widths=[1, vector],
            cute_lane_layouts=["blocked", layout],
        )
    )
    for offset in (3, 67, 131):
        records = _axis_coordinates(source, 1, offset, emitted_origins)
        assert Counter(index for index, _ in records) == Counter(
            range(offset, offset + 64)
        )
        assert {origin for _, origin in records} == {offset}


@pytest.mark.parametrize(
    ("blocks", "threads", "layouts", "vectors", "order", "pid"),
    (
        (
            [256, 128, 128, 256, 16],
            [64, 0, 0, 0, 0],
            ["strided", "strided"],
            [4, 4],
            [0, 1],
            "flat",
        ),
        (
            [16, 512, 128, 128, 16],
            [2, 4, 0, 0, 0],
            ["blocked", "strided"],
            [4, 1],
            [1, 0],
            "persistent_blocked",
        ),
        (
            [64, 16, 128, 256, 32],
            [2, 16, 0, 0, 0],
            ["blocked", "strided"],
            [2, 4],
            [0, 1],
            "persistent_interleaved",
        ),
        (
            [512, 32, 128, 16, 64],
            [32, 32, 0, 0, 0],
            ["strided", "blocked"],
            [2, 4],
            [1, 0],
            "persistent_interleaved",
        ),
    ),
)
def test_materialized_interleave_keeps_every_source_row(
    blocks: list[int],
    threads: list[int],
    layouts: list[str],
    vectors: list[int],
    order: list[int],
    pid: str,
    emitted_origins: dict[int, set[str]],
) -> None:
    kernel = helion.kernel(
        matmul_bf16_int4.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_materialize_transformed_operands=True,
    )
    bound = kernel._bind_isolated(
        (
            torch.empty((512, 1024), dtype=torch.bfloat16),
            torch.empty((512, 512), dtype=torch.int8),
        )
    )
    config = helion.Config.from_dict(
        {
            "block_sizes": blocks,
            "num_threads": threads,
            "cute_lane_layouts": [*layouts, "blocked", "blocked", "blocked"],
            "cute_vector_widths": [*vectors, 1, 1, 1],
            "loop_orders": [order, [0, 1]],
            "pid_type": pid,
            "cute_collective_mma": False,
            "tcgen05_persistence_model": (
                "non_persistent" if pid == "flat" else "static_persistent"
            ),
        }
    )
    source = _region(bound.to_code(config), 0)
    for offset in range(0, 512, blocks[0]):
        records = _axis_coordinates(source, 0, offset, emitted_origins)
        assert Counter(index for index, _ in records) == Counter(
            range(offset, offset + blocks[0])
        )
        assert {origin for _, origin in records} == {offset}
        # The interleave's arange uses tile.begin plus the local coordinate;
        # both packed nibbles must return to this same logical input row.
        assert all(origin + index - offset == index for index, origin in records)
