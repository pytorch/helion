from __future__ import annotations

import ast
from unittest.mock import create_autospec
from unittest.mock import patch

import pytest
import torch

from .test_cute_grid_launch_extents import _code
from .test_cute_grid_launch_extents import _tile_coordinates
import helion
from helion._compiler.cute import tile_ops
from helion._compiler.cute.cute_reshape import _per_thread_nd_tile_offset
from helion._compiler.cute.memory_ops import _cute_block_tile_begin_expr
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.inductor_lowering import CodegenState
from helion._compiler.tile_strategy import DeviceLoopState
from helion._compiler.tile_strategy import PerThreadFlattenedTileStrategy
from helion._compiler.tile_strategy import PerThreadNDTileStrategy
from helion._compiler.tile_strategy import TileStrategy
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language._decorators import is_api_func

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_begins(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        value = x[row, col]
        for step in hl.tile(3, block_size=1):
            value = value + row.begin + col.begin + step.begin
        out[row, col] = value
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _multigrid_begins(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(x)
    second = torch.empty_like(y)
    for row, col in hl.tile(x.shape):
        first[row, col] = x[row, col] + row.begin + col.begin
    for row, col in hl.tile(y.shape):
        second[row, col] = y[row, col] + row.begin + col.begin
    return first, second


@pytest.mark.parametrize("kind", ["coordinates", "nested", "multigrid"])
@pytest.mark.parametrize("layout", ["blocked", "strided"])
@pytest.mark.parametrize("vector", [1, 4])
def test_nd_begin_source_uses_uniform_tile_offsets(
    kind: str, layout: str, vector: int
) -> None:
    args = (torch.empty((7, 259), dtype=torch.float32),)
    kernel = _tile_coordinates
    blocks, threads = [2, 128], [2, 32]
    widths, layouts = [1, vector], ["blocked", layout]
    if kind == "nested":
        kernel = _nested_begins
        widths.append(1)
        layouts.append("blocked")
    elif kind == "multigrid":
        kernel = _multigrid_begins
        args += (torch.empty((11, 387), dtype=torch.float32),)
        blocks *= 2
        threads *= 2
        widths *= 2
        layouts *= 2
    code = _code(
        kernel,
        args,
        helion.Config(
            block_sizes=blocks,
            num_threads=threads,
            cute_vector_widths=widths,
            cute_lane_layouts=layouts,
        ),
    )
    tree = ast.parse(code)
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id.startswith("tile_begin_")
        for node in ast.walk(tree)
    )
    # These shapes require multiple nonzero program IDs on both axes. The
    # emitted tile offsets, not per-thread indices, must remain live.
    assert "tile_offset_0" in code and "tile_offset_1" in code
    if kind == "coordinates":
        assert "indices_1 - tile_offset_1" in code


@pytest.mark.parametrize("broadcast", [False, True])
def test_begin_selects_innermost_index_owner(broadcast: bool) -> None:
    state = create_autospec(CodegenState, instance=True)
    state.codegen = create_autospec(GenerateAST, instance=True)
    outer = create_autospec(DeviceLoopState, instance=True)
    inner = create_autospec(DeviceLoopState, instance=True)
    for owner, name in ((outer, "outer"), (inner, "inner")):
        owner.strategy = create_autospec(PerThreadNDTileStrategy, instance=True)
        owner.strategy.block_ids = [7]
        owner.strategy.offset_var.return_value = f"{name}_offset"
        owner.strategy.index_var.return_value = f"{name}_index"
        owner.block_thread_axes = {7: 0}
    state.codegen.active_device_loops = {7: [outer, inner]}
    state.codegen.current_grid_state = outer
    state.codegen.index_var.return_value = "inner_index"
    if broadcast:
        assert _cute_block_tile_begin_expr(state, 7) == "inner_offset"
    else:
        with patch.object(tile_ops, "_disable_flatten_get_tile", return_value=7):
            assert is_api_func(tile_ops.tile_begin)
            result = tile_ops.tile_begin._codegen["cute"](state)
            assert isinstance(result, ast.AST)
            assert ast.unparse(result) == "inner_offset"
    state.codegen.lift.assert_not_called()
    outer.strategy.offset_var.assert_not_called()


@pytest.mark.parametrize(
    "strategy_type",
    [TileStrategy, PerThreadFlattenedTileStrategy, PerThreadNDTileStrategy],
)
def test_direct_offset_requires_owned_nd_dimension(
    strategy_type: type[TileStrategy],
) -> None:
    strategy = create_autospec(strategy_type, instance=True)
    strategy.block_ids = [1]
    assert _per_thread_nd_tile_offset(strategy, 7) is None
    if strategy_type is not PerThreadNDTileStrategy:
        assert _per_thread_nd_tile_offset(strategy, 1) is None
    strategy.offset_var.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["nested", "multigrid"])
@pytest.mark.parametrize("layout", ["blocked", "strided"])
@pytest.mark.parametrize("vector", [1, 4])
def test_nd_begin_nested_and_multigrid_runtime(
    kind: str, layout: str, vector: int
) -> None:
    x = torch.arange(7 * 259, device=DEVICE, dtype=torch.float32).view(7, 259)
    row = torch.arange(7, device=DEVICE)[:, None] // 2 * 2
    col = torch.arange(259, device=DEVICE)[None, :] // 128 * 128
    args = (x,)
    kernel = _nested_begins
    expected = x + 3 * row + 3 * col + 3
    blocks, threads = [2, 128], [2, 32]
    widths, layouts = [1, vector, 1], ["blocked", layout, "blocked"]
    if kind == "multigrid":
        kernel = _multigrid_begins
        y = torch.arange(11 * 387, device=DEVICE, dtype=torch.float32).view(11, 387)
        row_y = torch.arange(11, device=DEVICE)[:, None] // 2 * 2
        col_y = torch.arange(387, device=DEVICE)[None, :] // 128 * 128
        args += (y,)
        expected = (x + row + col, y + row_y + col_y)
        blocks *= 2
        threads *= 2
        widths, layouts = [1, vector] * 2, ["blocked", layout] * 2
    saved = tuple(value.clone() for value in args)
    config = helion.Config(
        block_sizes=blocks,
        num_threads=threads,
        cute_vector_widths=widths,
        cute_lane_layouts=layouts,
    )
    run = kernel._bind_isolated(args).compile_config(config)
    actual = run(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(run(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
