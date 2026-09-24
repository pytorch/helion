from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_pointwise import _code as _guarded_code
from .test_cute_tile_loop_vec_hoist import _reduction_kernel
import helion
from helion._compiler.tile_strategy import DeviceGridState
from helion._compiler.tile_strategy import TileStrategy
from helion._compiler.tile_strategy import VecLaneWrapper
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language.memory_ops import _cute_register_tile_unroll_vec_store

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _recurrent_tile(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    rows, chunks, columns = x.shape
    out = torch.empty_like(x)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        acc = hl.zeros([col], dtype=torch.float32)
        for chunk in hl.grid(chunks):
            out[row.begin, chunk, col] = acc.to(x.dtype)
            acc = acc * decay[row.begin, chunk] + x[row.begin, chunk, col].float()
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_tile(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        out[row.begin, col] = x[row.begin, col]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_tile_nested(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row in hl.tile(rows, block_size=1):
        for col in hl.tile(columns):
            out[row.begin, col] = x[row.begin, col]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _horizontal_tile_indices(
    x: torch.Tensor, auxiliary: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, columns = x.shape
    first = torch.empty_like(x)
    second = torch.empty_like(x)
    third = torch.empty_like(x)
    copied = torch.empty_like(auxiliary)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        index = row.index[:, None] + col.index[None, :]
        value = x[row, col].float()
        first[row, col] = (value + (index % 3).float()).to(x.dtype)
        second[row, col] = (value * 0.5).to(x.dtype)
        third[row, col] = (index % 5).to(x.dtype)
    for tile in hl.tile(auxiliary.numel()):
        copied[tile] = auxiliary[tile]
    return first, second, third, copied


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_axis_copy(x: torch.Tensor, transposed_input: hl.constexpr) -> torch.Tensor:
    if transposed_input:
        columns, rows = x.shape
    else:
        rows, columns = x.shape
    first = torch.empty((rows, columns), device=x.device, dtype=x.dtype)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        if transposed_input:
            value = x[col, row.begin]
        else:
            value = x[row.begin, col]
        first[row.begin, col] = value
    return first


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_after_write(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = x.shape
    out = torch.empty_like(x)
    second = torch.empty_like(x)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        out[row.begin, col] = x[row.begin, col]
        second[row.begin, col] = out[row.begin, col] + 1
    return out, second


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _conditional_tile(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        if row.begin % 2 == 0:
            out[row.begin, col] = x[row.begin, col]
        else:
            out[row.begin, col] = x[row.begin, col] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _local_index_tile(x: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty_like(x)
    for row, col in hl.tile([rows, columns], block_size=[1, None]):
        selected = x[row.begin, 0].to(torch.int64) % rows
        out[row.begin, col] = x[selected, col]
    return out


def _config() -> helion.Config:
    return helion.Config(
        block_sizes=[1024],
        num_threads=[128],
        cute_vector_widths=[1, 8],
        cute_lane_layouts=["blocked", "blocked"],
    )


def _code(args: tuple[Any, ...], kernel: Any) -> str:
    with patch("test.test_cute_chained_pointwise._config", return_value=_config()):
        return _guarded_code(args, kernel=kernel)


@pytest.mark.parametrize("chunks", [1, 3, 8])
def test_nested_recurrence_keeps_memory_inside_loop(chunks: int) -> None:
    args = (
        torch.empty((2, chunks, 1024), dtype=torch.bfloat16),
        torch.empty((2, chunks), dtype=torch.float32),
    )
    code = _code(args, _recurrent_tile)
    assert "_tile_unroll_vec_" not in code
    assert "_tile_store_vals_" not in code
    assert "cutlass.range_constexpr(8)" in code
    assert ".load()" in code and ".store(" in code


@pytest.mark.parametrize("kernel", [_copy_tile, _copy_tile_nested])
def test_direct_tile_scope_still_vectorizes(kernel: Any) -> None:
    code = _code((torch.empty((2, 2048), dtype=torch.bfloat16),), kernel)
    assert "_tile_unroll_vec_" in code
    assert "_tile_store_vals_" in code


def _horizontal_config() -> helion.Config:
    return helion.Config(
        block_sizes=[1024, 128],
        num_threads=[128, 128],
        cute_vector_widths=[1, 8, 1],
        cute_lane_layouts=["blocked"] * 3,
    )


def test_horizontal_tile_index_aliases_do_not_disable_vectorization() -> None:
    args = (
        torch.empty((2, 2048), dtype=torch.bfloat16),
        torch.empty((512,), dtype=torch.bfloat16),
    )
    with patch(
        "test.test_cute_chained_pointwise._config", return_value=_horizontal_config()
    ):
        code = _guarded_code(args, kernel=_horizontal_tile_indices)
    assert "synthetic_lane" not in code
    assert code.count(" = cute.arch.load(") == 1
    assert code.count("_cute_store_u16_vec(") == 3


@pytest.mark.parametrize("transposed_input", [False, True])
def test_deferred_sites_preserve_their_own_contiguous_axis(
    transposed_input: bool,
) -> None:
    x = torch.empty((2, 2048), dtype=torch.bfloat16)
    if transposed_input:
        x = x.T
    code = _code((x, transposed_input), _mixed_axis_copy)
    load = next(line for line in code.splitlines() if " = cute.arch.load(" in line)
    assert "indices_1" not in load
    expected_axis = 0 if transposed_input else 1
    assert f"lane_base_1) * cutlass.Int32(x.layout.stride[{expected_axis}])" in load
    stores = [line for line in code.splitlines() if "_cute_store_u16_vec(" in line]
    assert len(stores) == 1
    assert all("indices_1" not in store for store in stores)


@pytest.mark.parametrize("lane_axis", [0, 1])
def test_store_emitter_uses_captured_axis(lane_axis: int) -> None:
    vloop = ast.parse("for vi in range(8):\n    pass").body[0]
    strategy = SimpleNamespace(
        _cute_lane_base_index_var_by_block={1: "base"},
        _cute_lane_body_by_block={1: [vloop]},
        _cute_lane_vloop_by_block={1: vloop},
        _cute_lane_axis_pos_by_block={1: 1 - lane_axis},
    )
    state = Mock()
    state.device_function.new_var.return_value = "values"
    with (
        patch(
            "helion.language.memory_ops._cute_scalar_pointer_expr",
            return_value="pointer",
        ) as pointer,
        patch(
            "helion.language.memory_ops.statement_from_string",
            side_effect=lambda source: ast.parse(source).body[0],
        ),
    ):
        _cute_register_tile_unroll_vec_store(
            state,
            strategy,
            1,
            "out",
            ["row", "col"],
            "value",
            None,
            torch.bfloat16,
            lane_axis_pos=lane_axis,
        )
    expected = ["row", "col"]
    expected[lane_axis] = "base"
    pointer.assert_called_once_with("out", expected)


def test_read_after_write_barrier_keeps_memory_order() -> None:
    code = _code((torch.empty((2, 2048), dtype=torch.bfloat16),), _read_after_write)
    assert "cute.arch.sync_threads()" in code
    assert "_tile_unroll_vec_" not in code
    assert "_tile_store_vals_" not in code


@pytest.mark.parametrize("inner_live", [False, True])
def test_deferred_scope_uses_the_complete_body(inner_live: bool) -> None:
    outer = ast.parse(
        "for lane in range(1):\n    base = lane * 8\n    for vi in range(8):\n        pass"
    ).body[0]
    assert isinstance(outer, ast.For)
    vloop = outer.body[1]
    assert isinstance(vloop, ast.For)
    grid = DeviceGridState(
        strategy=Mock(spec=TileStrategy),
        block_id_to_info={},
        lane_loops=[("lane", 8), ("inner", 64)],
        lane_setup_statements=[*ast.parse("index = base + vi\nother = inner").body],
        vec_lane_wrappers={"lane": VecLaneWrapper(outer, vloop, "vi", "base")},
    )
    scalar = ast.parse("result = scalar_load(index)").body[0]
    vector = ast.parse("result = vector_load(vi)").body[0]
    emit = Mock(return_value=vector)
    grid.deferred_vector_ops.append((vloop, scalar, emit))
    body: list[ast.AST] = [scalar]
    # This use is added AFTER the candidate was queued. A prefix-only
    # liveness check would incorrectly hoist it over the now-live inner loop.
    if inner_live:
        body.extend(ast.parse("consume(other)").body)
    wrapped = grid.wrap_body(body)
    code = "\n".join(ast.unparse(stmt) for stmt in wrapped)
    assert ("vector_load" in code) is not inner_live
    assert ("scalar_load" in code) is inner_live
    assert emit.call_count == int(not inner_live)


def test_branch_sites_remain_inside_the_branch() -> None:
    code = _code((torch.empty((2, 2048), dtype=torch.bfloat16),), _conditional_tile)
    assert "_tile_unroll_vec_" not in code
    assert "_tile_store_vals_" not in code


def test_body_defined_index_is_not_hoisted() -> None:
    code = _code((torch.empty((2, 2048), dtype=torch.bfloat16),), _local_index_tile)
    assert "_tile_unroll_vec_" not in code
    # The output address is independent of the computed input selector.
    assert "_tile_store_vals_" in code


def test_two_pass_reduction_still_vectorizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")
    config = helion.Config(
        block_sizes=[1, 256],
        num_threads=[0, 32],
        cute_vector_widths=[1, 8],
    )
    with patch("test.test_cute_chained_pointwise._config", return_value=config):
        code = _guarded_code(
            (torch.empty((2, 2048), dtype=torch.float16),), kernel=_reduction_kernel
        )
    assert "_tile_unroll_vec_" in code
    assert "ir.VectorType.get([8]" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("transposed_input", [False, True])
def test_deferred_mixed_axes_runtime_exact(transposed_input: bool) -> None:
    values = (
        torch.arange(4096, device=DEVICE)
        .reshape(2, 2048)
        .remainder(13)
        .to(torch.bfloat16)
    )
    x = values.T if transposed_input else values
    saved = x.clone()
    run = _mixed_axis_copy._bind_isolated((x, transposed_input)).compile_config(
        _config()
    )
    actual = run(x, transposed_input)
    repeated = run(x, transposed_input)
    torch.testing.assert_close(actual, values, atol=0, rtol=0)
    torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    assert actual.data_ptr() != repeated.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_read_after_write_runtime_exact() -> None:
    x = (
        torch.arange(4096, device=DEVICE)
        .reshape(2, 2048)
        .remainder(13)
        .to(torch.bfloat16)
    )
    saved = x.clone()
    run = _read_after_write._bind_isolated((x,)).compile_config(_config())
    actual = run(x)
    repeated = run(x)
    torch.testing.assert_close(actual, (x, x + 1), atol=0, rtol=0)
    torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    assert all(
        a.data_ptr() != b.data_ptr() for a, b in zip(actual, repeated, strict=True)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_horizontal_tile_indices_runtime_exact() -> None:
    x = (
        torch.arange(4096, device=DEVICE)
        .reshape(2, 2048)
        .remainder(7)
        .to(torch.bfloat16)
    )
    auxiliary = torch.arange(512, device=DEVICE).to(torch.bfloat16)
    index = (
        torch.arange(2, device=DEVICE)[:, None]
        + torch.arange(2048, device=DEVICE)[None, :]
    )
    expected = (
        (x.float() + (index % 3).float()).to(x.dtype),
        (x.float() * 0.5).to(x.dtype),
        (index % 5).to(x.dtype),
        auxiliary,
    )
    saved = x.clone(), auxiliary.clone()
    run = _horizontal_tile_indices._bind_isolated((x, auxiliary)).compile_config(
        _horizontal_config()
    )
    actual = run(x, auxiliary)
    repeated = run(x, auxiliary)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
    torch.testing.assert_close((x, auxiliary), saved, atol=0, rtol=0)
    assert all(
        a.data_ptr() != b.data_ptr() for a, b in zip(actual, repeated, strict=True)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("chunks", [1, 3, 8])
def test_nested_recurrence_runtime_exact(chunks: int) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(592)
    x = torch.randint(-2, 3, (2, chunks, 1024), device=DEVICE, generator=generator).to(
        torch.bfloat16
    )
    decay = torch.full((2, chunks), 0.5, device=DEVICE, dtype=torch.float32)
    expected = torch.empty_like(x)
    carry = torch.zeros_like(x[:, 0], dtype=torch.float32)
    for chunk in range(chunks):
        expected[:, chunk] = carry.to(x.dtype)
        carry = carry * decay[:, chunk, None] + x[:, chunk].float()
    saved = x.clone(), decay.clone()
    run = _recurrent_tile._bind_isolated((x, decay)).compile_config(_config())
    actual = run(x, decay)
    repeated = run(x, decay)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
    assert actual.data_ptr() != repeated.data_ptr()
    torch.testing.assert_close((x, decay), saved, atol=0, rtol=0)
