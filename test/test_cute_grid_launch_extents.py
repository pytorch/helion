from __future__ import annotations

import ast
from typing import Any

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute.backend import _live_grid_thread_dims
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _indexed_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + (tile.index % 3).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _indexed_copy_2d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col] + row.begin % 3
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _free_axis_copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.shape[0]):
        column = hl.arange(x.shape[1])
        out[row, column] = x[row, column] + row.index[:, None].to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_grid(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rows, reduction = a.shape
    columns = b.shape[1]
    out = torch.empty((rows, columns), dtype=a.dtype, device=a.device)
    for row, col in hl.tile([rows, columns]):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for kk in hl.tile(reduction):
            acc = hl.dot(a[row, kk], b[kk, col], acc=acc)
        out[row, col] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _merged_copy(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(x)
    second = torch.empty_like(y)
    for tile in hl.tile(x.numel()):
        first[tile] = x[tile] + (tile.index % 3).to(x.dtype)
    for tile in hl.tile(y.numel()):
        second[tile] = y[tile] - (tile.index % 5).to(y.dtype)
    return first, second


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_rank_copy(
    x: torch.Tensor, y: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    first = torch.empty_like(x)
    for row, col in hl.tile(x.shape, block_size=[1, None]):
        first[row, col] = x[row, col] + row.index[:, None].to(x.dtype)
    for tile in hl.tile(y.numel()):
        second[tile] = y[tile.index]
    return first


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _opposed_grids(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.empty_like(x)
    second = torch.empty_like(y)
    for row, col in hl.tile(x.shape):
        first[row, col] = x[row, col] + row.begin % 3
    for row, col in hl.tile(y.shape):
        second[row, col] = y[row, col] + row.begin % 5
    return first, second


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tile_coordinates(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gathered = torch.empty_like(x)
    begins = torch.empty_like(x)
    ends = torch.empty_like(x)
    for row, tile in hl.tile(x.shape):
        local = tile.index - tile.begin
        other = tile.begin + (local + 1) % (tile.end - tile.begin)
        gathered[row, tile] = x[row, other]
        begins[row, tile] = tile.begin
        ends[row, tile] = tile.end
    return gathered, begins, ends


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grid_scan(x: torch.Tensor, reverse: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile, col in hl.tile(x.shape):
        # pyrefly: ignore [bad-argument-type]
        out[tile, col] = hl.cumsum(x[tile, col], dim=0, reverse=reverse)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grid_argreduce(
    a: torch.Tensor, b: torch.Tensor, largest: hl.constexpr
) -> torch.Tensor:
    out = torch.empty(
        (a.shape[0], (b.shape[1] + 127) // 128), dtype=torch.int64, device=a.device
    )
    for row, tile in hl.tile([a.shape[0], b.shape[1]]):
        value = hl.dot(a[row, :], b[:, tile])
        if largest:
            result = torch.argmax(value, dim=1)
        else:
            result = torch.argmin(value, dim=1)
        out[row, tile.begin // 128] = result
    return out


def _config(kind: str, layout: str = "strided", vector: int = 1) -> helion.Config:
    if kind == "collective":
        blocks, threads = [2, 128], [2, 32]
    elif kind == "scan":
        blocks, threads = [128, 2], [32, 2]
    elif kind == "2d":
        blocks, threads = [4, 1024], [2, 128]
    elif kind == "merged":
        blocks, threads = [1024, 512], [128, 256]
    elif kind == "mixed":
        blocks, threads = [2048, 128], [256, 128]
    elif kind == "free":
        blocks, threads = [16], [8]
    else:
        blocks, threads = [2048], [128]
    config = helion.Config(
        block_sizes=blocks,
        num_threads=threads,
        cute_vector_widths=[vector] * len(threads),
        cute_lane_layouts=[layout] * len(threads),
    )
    if kind == "mixed":
        config.config["cute_vector_widths"] = [1, vector, 1]
        config.config["cute_lane_layouts"] = ["blocked", layout, "blocked"]
    return config


def _code(kernel: Any, args: tuple[Any, ...], config: helion.Config) -> str:
    with _cpu_codegen():
        return kernel._bind_isolated(args).to_code(config)


def _launch_block(code: str) -> tuple[int, int, int]:
    values = [
        ast.literal_eval(keyword.value)
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "block"
    ]
    assert len(values) == 1
    return values[0]


@pytest.mark.parametrize("layout", ["blocked", "strided"])
@pytest.mark.parametrize("vector", [1, 8])
@pytest.mark.parametrize("length", [4096, 4112])
def test_grid_layout_preserves_launch_extent(
    layout: str, vector: int, length: int
) -> None:
    code = _code(
        _indexed_copy,
        (torch.empty((length,), dtype=torch.bfloat16),),
        _config("1d", layout, vector),
    )
    assert _launch_block(code) == (128, 1, 1)
    if layout == "strided" and vector == 1:
        assert "cutlass.Int32(lane_0) * 128" in code
        assert "cute.arch.thread_idx()[0]) * 16" not in code


@pytest.mark.parametrize("kind", ["2d", "merged"])
def test_multiple_grid_requirements_combine_per_axis(kind: str) -> None:
    if kind == "2d":
        kernel, args = _indexed_copy_2d, (torch.empty((17, 2057)),)
        expected = [1, 2, 128]
    else:
        kernel, args = _merged_copy, (torch.empty((2057,)), torch.empty((1031,)))
        expected = [1, 1, 256]
    code = _code(kernel, args, _config(kind))
    assert sorted(_launch_block(code)) == expected
    if kind == "merged":
        # The larger root launches 256 threads; the 128-thread root retains
        # its own thread mask to prevent duplicate/out-of-tile accesses.
        assert "cute.arch.thread_idx()[0]) < 128" in code


def test_explicit_launch_metadata_ignores_expression_spelling_and_dead_names() -> None:
    body = ast.parse(
        "first = offset + tid + lane * 128\n"
        "if branch:\n"
        "    base = opaque_index_transform()\n"
        "else:\n"
        "    wider = other_expression()\n"
        "unregistered = 99\n"
    ).body
    assert _live_grid_thread_dims(
        body,
        {"first": (0, 128), "base": (1, 4), "wider": (0, 256), "dead": (2, 64)},
    ) == [256, 4, 1]


def test_live_free_axis_is_not_dropped() -> None:
    args = (torch.empty((65, 32)),)
    blocked = _code(_free_axis_copy, args, _config("free", "blocked"))
    strided = _code(_free_axis_copy, args, _config("free", "strided"))
    assert _launch_block(blocked) == _launch_block(strided)
    assert sorted(_launch_block(strided)) == [1, 2, 8]


@pytest.mark.parametrize(
    "first_threads,second_threads", [(256, 128), (128, 256), (128, 128), (256, 256)]
)
def test_mixed_rank_grid_keeps_surplus_thread_mask(
    first_threads: int, second_threads: int
) -> None:
    config = _config("mixed", vector=8)
    config.config["num_threads"] = [first_threads, second_threads]
    config.config["block_sizes"] = [2048, second_threads]
    code = _code(
        _mixed_rank_copy,
        (torch.empty((2, 4096)), torch.empty((1024,)), torch.empty((1024,))),
        config,
    )
    assert sorted(_launch_block(code)) == [1, 1, max(first_threads, second_threads)]
    if first_threads > second_threads:
        assert "_BLOCK_SIZE_2 = 128" in code
        assert "cute.arch.thread_idx()[1]) < _BLOCK_SIZE_2" in code
        assert "if mask_2:" in code
    elif second_threads > first_threads:
        assert "cute.arch.thread_idx()[1]) < 128" in code
        assert "if mask_1:" in code
    else:
        assert "mask_1 =" not in code
        assert "mask_2 =" not in code


def test_incompatible_grid_axes_fail_closed_at_thread_budget() -> None:
    config = helion.Config(
        block_sizes=[16, 2048, 2048, 16],
        num_threads=[16, 64, 64, 16],
        cute_vector_widths=[1, 1, 1, 1],
        cute_lane_layouts=["strided"] * 4,
    )
    with pytest.raises(exc.BackendUnsupported, match="thread"):
        _code(
            _opposed_grids, (torch.empty((32, 4096)), torch.empty((4096, 32))), config
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("vector", [1, 8])
@pytest.mark.parametrize("wide_first", [True, False])
def test_mixed_rank_grid_runtime_canary(vector: int, wide_first: bool) -> None:
    x = torch.arange(8192, device=DEVICE, dtype=torch.float32).reshape(2, 4096)
    y = torch.arange(1024, device=DEVICE, dtype=torch.float32)
    storage = torch.full((1280,), -98765.0, device=DEVICE)
    second = storage[128:1152]
    args = (x, y, second)
    saved = (x.clone(), y.clone())
    config = _config("mixed", vector=vector)
    if not wide_first:
        config.config["num_threads"] = [128, 256]
        config.config["block_sizes"] = [2048, 256]
    run = _mixed_rank_copy._bind_isolated(args).compile_config(config)
    expected = x + torch.arange(2, device=DEVICE)[:, None]
    torch.testing.assert_close(run(*args), expected, atol=0, rtol=0)
    torch.testing.assert_close(run(*args), expected, atol=0, rtol=0)
    torch.testing.assert_close(second, y, atol=0, rtol=0)
    torch.testing.assert_close((x, y), saved, atol=0, rtol=0)
    assert torch.all(storage[:128] == -98765.0)
    assert torch.all(storage[1152:] == -98765.0)


def test_matmul_grid_retains_its_collective_mapping() -> None:
    args = (
        torch.empty((16, 32), dtype=torch.bfloat16),
        torch.empty((32, 64), dtype=torch.bfloat16),
    )
    codes = []
    for layout in ("blocked", "strided"):
        config = helion.Config(
            block_sizes=[8, 64, 32],
            num_threads=[4, 16, 1],
            cute_vector_widths=[1, 1, 1],
            cute_lane_layouts=[layout, layout, "blocked"],
        )
        codes.append(_code(_matmul_grid, args, config))
    assert ast.dump(ast.parse(codes[0])) == ast.dump(ast.parse(codes[1]))


@pytest.mark.parametrize("kind", ["coordinates", "scan", "argreduce"])
@pytest.mark.parametrize("reverse_or_max", [False, True])
def test_grid_logical_coordinates_codegen(kind: str, reverse_or_max: bool) -> None:
    x = torch.empty((259, 3) if kind == "scan" else (3, 259), dtype=torch.float32)
    if kind == "coordinates":
        kernel, args = _tile_coordinates, (x,)
    elif kind == "scan":
        kernel, args = _grid_scan, (x, reverse_or_max)
    else:
        kernel, args = (
            _grid_argreduce,
            (torch.empty((3, 16)), torch.empty((16, 128)), reverse_or_max),
        )
    code = _code(kernel, args, _config("scan" if kind == "scan" else "collective"))
    if kind == "argreduce":
        # Matmul-derived argreductions use the supported shared-tile lowering;
        # ordinary argreduce lane folding is not supported yet. The collective
        # mapping is intentionally unchanged by a scalar-strided request.
        blocked = _code(kernel, args, _config("collective", "blocked"))
        assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(blocked))
        assert "argreduce_smem" in code
        return
    assert sorted(_launch_block(code)) == [1, 2, 32]
    axis = 0 if kind == "scan" else 1
    assert f"cutlass.Int32(lane_{axis}) * 32" in code
    if kind == "scan":
        assert "scan_warp_prefix" not in code
    else:
        assert "indices_1 - tile_offset_1" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", ["blocked", "strided"])
@pytest.mark.parametrize("kind", ["coordinates", "scan", "argreduce"])
@pytest.mark.parametrize("reverse_or_max", [False, True])
def test_grid_logical_coordinates_runtime(
    layout: str, kind: str, reverse_or_max: bool
) -> None:
    index = torch.arange(259, device=DEVICE)
    x = (torch.arange(3, device=DEVICE)[:, None] + index[None, :] % 7).float()
    if kind == "coordinates":
        kernel, args = _tile_coordinates, (x,)
        begin = index // 128 * 128
        end = torch.minimum(begin + 128, torch.full_like(begin, 259))
        other = begin + (index - begin + 1) % (end - begin)
        expected = (x[:, other], begin.float().expand_as(x), end.float().expand_as(x))
    elif kind == "scan":
        x = x.T.contiguous()
        kernel, args = _grid_scan, (x, reverse_or_max)
        parts = x.split(128)
        expected = torch.cat(
            [
                p.flip(0).cumsum(0).flip(0) if reverse_or_max else p.cumsum(0)
                for p in parts
            ]
        )
    else:
        # Equal extrema at logical 1 and 32 belong to different physical
        # lane iterations under blocked versus strided layouts.
        x = torch.ones((3, 16), device=DEVICE)
        b = torch.zeros((16, 128), device=DEVICE)
        b[:, [1, 32]] = 9 if reverse_or_max else -9
        kernel, args = _grid_argreduce, (x, b, reverse_or_max)
        fn = torch.argmax if reverse_or_max else torch.argmin
        expected = torch.stack([fn(p, dim=1) for p in (x @ b).split(128, dim=1)], dim=1)
    tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
    saved = tuple(arg.clone() for arg in tensor_args)
    run = kernel._bind_isolated(args).compile_config(
        _config("scan" if kind == "scan" else "collective", layout)
    )
    actual = run(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(run(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(tensor_args, saved, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["1d", "2d", "merged", "free"])
@pytest.mark.parametrize("layout", ["blocked", "strided"])
def test_grid_launch_runtime_exact(kind: str, layout: str) -> None:
    if kind == "free":
        x = torch.arange(65 * 32, device=DEVICE).reshape(65, 32).remainder(7)
        x = x.to(torch.bfloat16)
        kernel, args = _free_axis_copy, (x,)
        expected = x + torch.arange(65, device=DEVICE)[:, None].to(x.dtype)
    elif kind == "2d":
        x = torch.arange(17 * 2057, device=DEVICE).reshape(17, 2057).remainder(7)
        x = x.to(torch.bfloat16)
        kernel, args = _indexed_copy_2d, (x,)
        expected = x + ((torch.arange(17, device=DEVICE) // 4 * 4) % 3)[:, None].to(
            x.dtype
        )
    else:
        x = torch.arange(2057, device=DEVICE).remainder(7).to(torch.bfloat16)
        expected = x + (torch.arange(2057, device=DEVICE) % 3).to(x.dtype)
        kernel, args = _indexed_copy, (x,)
        if kind == "merged":
            y = torch.arange(1031, device=DEVICE).remainder(11).to(torch.bfloat16)
            kernel, args = _merged_copy, (x, y)
            expected = (
                expected,
                y - (torch.arange(1031, device=DEVICE) % 5).to(y.dtype),
            )
    saved = tuple(value.clone() for value in args)
    run = kernel._bind_isolated(args).compile_config(_config(kind, layout))
    actual = run(*args)
    repeated = run(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
