from __future__ import annotations

import ast
from unittest.mock import create_autospec
from unittest.mock import patch

import pytest
import torch

from .test_cute_grid_launch_extents import _code
from .test_cute_grid_launch_extents import _launch_block
from .test_cute_grid_launch_extents import _tile_coordinates
import helion
from helion import exc
from helion._compiler.cute import tile_ops
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.cute_reshape import _per_thread_nd_tile_offset
from helion._compiler.cute.memory_ops import _cute_block_tile_begin_expr
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.indexing_strategy import SubscriptIndexing
from helion._compiler.inductor_lowering import CodegenState
from helion._compiler.tile_strategy import DeviceLoopState
from helion._compiler.tile_strategy import PerThreadFlattenedTileStrategy
from helion._compiler.tile_strategy import PerThreadNDTileStrategy
from helion._compiler.tile_strategy import TileStrategy
from helion._testing import DEVICE
from helion._testing import _example_kernel
from helion._testing import patch_cute_mma_support
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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _serial_rows(
    x: torch.Tensor, row_scale: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    block = hl.register_block_size(x.shape[0])
    columns = hl.specialize(x.shape[1])
    for group in hl.tile(x.shape[0], block_size=block):
        for row in hl.tile(group.begin, group.end):
            value = x[row, :].float() * row_scale[row][:, None]
            total = torch.sum(value, dim=-1) / columns
            out[row, :] = (value + total[:, None]).to(x.dtype)
    return out


def _config() -> helion.Config:
    return helion.Config(block_sizes=[32, 1], num_warps=4, num_stages=3)


@pytest.mark.parametrize("columns", [8, 16, 32])
@pytest.mark.parametrize("rows", [160, 169])
def test_serial_row_root_uses_physical_launch_axes(rows: int, columns: int) -> None:
    args = (
        torch.empty((rows, columns)),
        torch.empty((rows,)),
        torch.empty((rows, columns)),
    )
    code = _code(_serial_rows, args, _config())
    assert _launch_block(code) == (columns, 1, 1)
    # The root row dimension has no remaining per-thread work; its logical
    # width32 must not enlarge the feature axis mapped onto threadIdx.x.
    thread_axes = {
        node.slice.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "thread_idx"
        and isinstance(node.slice, ast.Constant)
    }
    assert thread_axes == {0}


@pytest.mark.parametrize("rows", [160, 1152000])
def test_original_layernorm_backward_keeps_feature_extent(rows: int) -> None:
    kernel = _example_kernel("layer_norm", fn_name="layer_norm_bwd")
    args = (
        torch.empty((rows, 16), dtype=torch.float16),
        torch.empty((rows, 16), dtype=torch.float16),
        torch.empty((rows,), dtype=torch.float32),
        torch.empty((rows,), dtype=torch.float32),
        torch.empty((16,), dtype=torch.float16),
        True,
    )
    code = _code(kernel, args, _config())
    assert _launch_block(code) == (16, 1, 1)
    assert "threads_in_group=16" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("columns", [8, 16, 32])
@pytest.mark.parametrize("rows", [160, 169])
def test_serial_rows_preserve_boundaries_and_canaries(rows: int, columns: int) -> None:
    # Padding makes accidental feature>=N accesses deterministic and checks
    # the trailing OOB writes instead of relying on allocator contents.
    x_storage = torch.arange(
        (rows + 2) * columns, device=DEVICE, dtype=torch.float32
    ).view(rows + 2, columns)
    x = x_storage[1:-1]
    # Extra feature lanes would read the next row's x with this row's scale,
    # so every CTA boundary is numerically wrong even if another CTA writes
    # the same output location. The trailing canary also detects the OOB.
    row_scale = torch.arange(1, rows + 1, device=DEVICE, dtype=torch.float32)
    storage = torch.full_like(x_storage, -12345)
    out = storage[1:-1]
    saved = x_storage.clone()
    saved_scale = row_scale.clone()
    value = x * row_scale[:, None]
    expected = value + value.mean(dim=-1, keepdim=True)
    run = _serial_rows._bind_isolated((x, row_scale, out)).compile_config(_config())
    for _ in range(3):
        out.fill_(float("nan"))
        result = run(x, row_scale, out)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(result[::32], expected[::32], atol=0, rtol=0)
        assert torch.all(storage[0] == -12345)
        assert torch.all(storage[-1] == -12345)
        torch.testing.assert_close(x_storage, saved, atol=0, rtol=0)
        torch.testing.assert_close(row_scale, saved_scale, atol=0, rtol=0)


@pytest.mark.parametrize(
    "index,expected",
    [
        ([slice(None), slice(None)], [3, 5]),
        ([None, slice(None), slice(None)], [1, 3, 5]),
        ([slice(None), None, slice(None)], [3, 1, 5]),
        ([slice(None), slice(None), None], [3, 5, 1]),
        ([None, slice(None), None, slice(None), None], [1, 3, 1, 5, 1]),
    ],
)
def test_shape_only_views_do_not_allocate_reduction_dimensions(
    index: list[object], expected: list[int]
) -> None:
    tensor = torch.empty((3, 5))
    with patch.object(
        SubscriptIndexing,
        "compute_shape",
        side_effect=AssertionError("shape-only views must preserve dimensions"),
    ):
        assert CuteBackend().fake_subscript_shape(tensor, index) == expected


@pytest.mark.parametrize(
    "index",
    [
        [1, slice(None)],
        [slice(1, 3), slice(None)],
        [slice(None), slice(0, 4, 1)],
    ],
)
def test_narrowing_keeps_general_index_shape_path(index: list[object]) -> None:
    tensor = torch.empty((3, 5))
    with patch.object(
        SubscriptIndexing, "compute_shape", return_value=[2, 5]
    ) as compute:
        assert CuteBackend().fake_subscript_shape(tensor, index) == [2, 5]
    compute.assert_called_once_with(tensor, index)


@pytest.mark.parametrize(
    "index", [[slice(None), slice(None, None, 2)], [-1, slice(None)]]
)
def test_unsupported_narrowing_still_fails_closed(index: list[object]) -> None:
    with pytest.raises(exc.InvalidIndexingType):
        CuteBackend().fake_subscript_shape(torch.empty((3, 5)), index)


@pytest.mark.parametrize("topology", ["fa4", "ws_overlap"])
def test_causal_flash_host_has_no_undefined_block_size(topology: str) -> None:
    from .test_cute_backend import cute_causal_biased_attention

    values = [torch.empty((1, 2, 256, 64), dtype=torch.float16) for _ in range(3)]
    values.append(torch.empty((1, 2, 256, 256), dtype=torch.float16))
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        cute_causal_biased_attention.reset()
        code = cute_causal_biased_attention.bind(tuple(values)).to_code(
            helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_topology=topology,
                cute_flash_causal_kv_order="descending",
                cute_flash_causal_loop_split=True,
            )
        )
    assert "'kind': 'helion_flash'" in code
    tree = ast.parse(code)
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    read = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id.startswith("_BLOCK_SIZE_")
    }
    assert not read - assigned
