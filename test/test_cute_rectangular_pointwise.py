from __future__ import annotations

import ast
from collections import Counter
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from .test_cute_grid_launch_extents import _code
from .test_cute_grid_launch_extents import _launch_block
import helion
from helion import exc
from helion._compiler.cute.backend import _live_grid_thread_extents
from helion._compiler.cute.backend import _pointwise_grid_thread_dims
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rectangular_pointwise(
    x: torch.Tensor, row: torch.Tensor, col: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for bi, mi, ni in hl.tile(x.shape, block_size=[1, None, None]):
        b = bi.begin
        value = x[b, mi, ni] + row[b, mi][:, None] * col[b, ni][None, :]
        out[b, mi, ni] = torch.where(mi.index[:, None] >= ni.index[None, :], value, 0.0)
    return out


def _args(
    rows: int,
    columns: int,
    *,
    offset: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def allocate(shape: tuple[int, ...]) -> torch.Tensor:
        if offset:
            storage = torch.empty((*shape[:-1], shape[-1] + 8), device=device)
            return storage[..., 4:-4]
        return torch.empty(shape, device=device)

    return (
        allocate((2, rows, columns)),
        allocate((2, rows)),
        allocate((2, columns)),
        allocate((2, rows, columns)),
    )


def _config(rows: int, columns: int, *, swap: bool = True) -> helion.Config:
    return helion.Config(
        block_sizes=[rows, columns], loop_orders=[[0, 2, 1] if swap else [0, 1, 2]]
    )


def _write_coordinates(code: str) -> Counter[tuple[int, int, int]]:
    """Interpret only generated integer launch/index/mask AST, never tensor code."""
    module = ast.parse(code)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("_BLOCK_SIZE_")
    }
    device_fn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    prefix: list[ast.stmt] = []
    for node in device_fn.body:
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith(
                ("num_blocks_", "pid_", "tile_offset_", "indices_", "mask_")
            )
        ):
            break
        prefix.append(node)
    assert prefix
    executable = compile(ast.Module(body=prefix, type_ignores=[]), "<indices>", "exec")
    launch = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    grid = eval(compile(ast.Expression(launch.args[1]), "<grid>", "eval"), constants)
    assert len(grid) == 1
    block = _launch_block(code)
    result: Counter[tuple[int, int, int]] = Counter()
    for pid, tx, ty, tz in itertools.product(
        range(grid[0]), *(range(size) for size in block)
    ):
        namespace = {
            **constants,
            "cutlass": SimpleNamespace(Int32=int),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(
                    block_idx=lambda pid=pid: (pid, 0, 0),
                    thread_idx=lambda tx=tx, ty=ty, tz=tz: (tx, ty, tz),
                )
            ),
        }
        exec(executable, namespace)
        if all(value for name, value in namespace.items() if name.startswith("mask_")):
            batch = namespace["tile_offset_0"]
            row = namespace["indices_1"]
            column = namespace["indices_2"]
            assert (
                isinstance(batch, int)
                and isinstance(row, int)
                and isinstance(column, int)
            )
            result[batch, row, column] += 1
    return result


@pytest.mark.parametrize("tile", [(4, 64), (64, 4), (8, 128), (32, 32)])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_rectangular_grid_has_exactly_one_writer(
    tile: tuple[int, int], swap: bool, offset: bool
) -> None:
    initialized_before = torch.cuda.is_initialized()
    rows, columns = (65, 97) if offset else (64, 128)
    if tile == (8, 128) and not swap:
        # This mapping needs z=128. Integer-coordinate completeness alone
        # does not make that a valid CUDA launch, even at 1024 total threads.
        with pytest.raises(exc.BackendUnsupported, match="per-axis"):
            _code(
                _rectangular_pointwise,
                _args(rows, columns, offset=offset),
                _config(*tile, swap=swap),
            )
        assert torch.cuda.is_initialized() is initialized_before
        return
    code = _code(
        _rectangular_pointwise,
        _args(rows, columns, offset=offset),
        _config(*tile, swap=swap),
    )
    coordinates = _write_coordinates(code)
    assert coordinates == Counter(
        itertools.product(range(2), range(rows), range(columns))
    )
    assert torch.cuda.is_initialized() is initialized_before


@pytest.mark.parametrize(
    "pointwise,nested,free_axes,final_axes",
    [
        (False, False, False, {1, 2}),
        (True, True, False, {1, 2}),
        (True, False, True, {1, 2}),
        (True, False, False, {0, 1, 2}),
    ],
)
def test_pointwise_launch_requires_complete_producer_ownership(
    pointwise: bool, nested: bool, free_axes: bool, final_axes: set[int]
) -> None:
    assert (
        _pointwise_grid_thread_dims(
            {1: 64, 2: 4},
            final_axes,
            (1, 1, 1),
            has_pointwise_fact=pointwise,
            has_nested_device_loops=nested,
            has_synthetic_free_axes=free_axes,
        )
        is None
    )


def test_dead_producer_does_not_cover_an_independent_thread_axis() -> None:
    body = ast.parse("row = offset + tid\ncolumn = other_index\n").body
    live = _live_grid_thread_extents(
        body, {"row": (2, 4), "column": (1, 64), "dead": (0, 8)}
    )
    assert live == {1: 64, 2: 4}
    assert (
        _pointwise_grid_thread_dims(
            live,
            {0, 1, 2},
            (8, 64, 4),
            has_pointwise_fact=True,
            has_nested_device_loops=False,
            has_synthetic_free_axes=False,
        )
        is None
    )


def test_pointwise_launch_rejects_unproved_surplus_references() -> None:
    with pytest.raises(exc.BackendUnsupported, match="reference exceeds"):
        _pointwise_grid_thread_dims(
            {1: 64, 2: 4},
            {1, 2},
            (4, 128, 16),
            has_pointwise_fact=True,
            has_nested_device_loops=False,
            has_synthetic_free_axes=False,
        )


def test_pointwise_launch_keeps_total_thread_budget_guard() -> None:
    config = _config(32, 64)
    config.config["num_threads"] = [32, 64]
    with pytest.raises(exc.BackendUnsupported, match="thread"):
        _code(_rectangular_pointwise, _args(64, 128), config)


def test_unit_producer_is_distinct_from_a_missing_producer() -> None:
    live = _live_grid_thread_extents(
        ast.parse("unit = tid\nrow = other\n").body, {"unit": (0, 1), "row": (1, 32)}
    )
    assert _pointwise_grid_thread_dims(
        live,
        {0, 1},
        (1, 32, 64),
        has_pointwise_fact=True,
        has_nested_device_loops=False,
        has_synthetic_free_axes=False,
    ) == (1, 32, 1)


@pytest.mark.parametrize("axis,limit", [(0, 1024), (1, 1024), (2, 64)])
def test_pointwise_launch_cuda_axis_boundaries(axis: int, limit: int) -> None:
    expected = [1, 1, 1]
    expected[axis] = limit
    assert _pointwise_grid_thread_dims(
        {axis: limit},
        {axis},
        (1, 1, 1),
        has_pointwise_fact=True,
        has_nested_device_loops=False,
        has_synthetic_free_axes=False,
    ) == tuple(expected)
    with pytest.raises(exc.BackendUnsupported, match="per-axis"):
        _pointwise_grid_thread_dims(
            {axis: limit * 2},
            {axis},
            (1, 1, 1),
            has_pointwise_fact=True,
            has_nested_device_loops=False,
            has_synthetic_free_axes=False,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_pointwise_reduction(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    total = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col] + 1
    for row in hl.tile(x.shape[0]):
        total[row] = torch.sum(x[row, :], dim=1)
    return out, total


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_pointwise_matmul(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    product = torch.empty((x.shape[0], y.shape[1]), device=x.device, dtype=x.dtype)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col] + 1
    for row, col in hl.tile([x.shape[0], y.shape[1]]):
        kk = hl.arange(x.shape[1])
        product[row, col] = hl.dot(x[row, kk], y[kk, col]).to(x.dtype)
    return out, product


@pytest.mark.parametrize("kind", ["reduction", "matmul"])
def test_pointwise_fact_excludes_other_collective_roots(kind: str) -> None:
    before = torch.cuda.is_initialized()
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
    ):
        x = torch.empty((16, 32), dtype=torch.bfloat16)
        if kind == "reduction":
            bound = _mixed_pointwise_reduction._bind_isolated((x,))
            assert bound.env.config_spec.reduction_kernel_fact is not None
        else:
            bound = _mixed_pointwise_matmul._bind_isolated(
                (x, torch.empty((32, 64), dtype=x.dtype))
            )
            assert bound.env.config_spec.matmul_facts
        assert bound.host_function is not None
        assert len(bound.host_function.device_ir.grid_block_ids) == 2
        assert not bound.env.config_spec.pointwise_facts
    assert torch.cuda.is_initialized() is before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tile", [(4, 64), (64, 4)])
@pytest.mark.parametrize("swap", [False, True])
def test_rectangular_pointwise_runtime(tile: tuple[int, int], swap: bool) -> None:
    args = _args(65, 97, offset=True, device=DEVICE)
    x, row, col, output = args
    for value in args:
        backing = value._base
        assert isinstance(backing, torch.Tensor)
        backing.fill_(-12345.0)
    for value in args:
        value.fill_(float("nan"))
    x.copy_(torch.arange(x.numel(), device=DEVICE).reshape(x.shape) / 128)
    row.copy_(torch.arange(row.numel(), device=DEVICE).reshape(row.shape) / 16)
    col.copy_(torch.arange(col.numel(), device=DEVICE).reshape(col.shape) / 32)
    before = tuple(value.clone() for value in args[:3])
    expected = x + row[:, :, None] * col[:, None, :]
    expected = torch.where(
        torch.arange(65, device=DEVICE)[:, None]
        >= torch.arange(97, device=DEVICE)[None, :],
        expected,
        0.0,
    )
    run = _rectangular_pointwise._bind_isolated(args).compile_config(
        _config(*tile, swap=swap)
    )
    for _ in range(2):
        output.fill_(float("nan"))
        assert run(*args) is output
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(args[:3], before, atol=0, rtol=0)
        for value in args:
            backing = value._base
            assert isinstance(backing, torch.Tensor)
            assert torch.all(backing[..., :4] == -12345.0)
            assert torch.all(backing[..., -4:] == -12345.0)
