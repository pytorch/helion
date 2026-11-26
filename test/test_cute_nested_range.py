"""CPU checks for the native CuTe induction type of nested grid and tile loops."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

import helion
from helion._testing import skipIfRefEagerMark
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_thread_simulator import execute_pointwise_thread_program
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render,
    skipIfRefEagerMark("checks native CuTe loop widths"),
]


def _render(kernel: helion.Kernel, *args: object) -> str:
    return cpu_bind(kernel, args).to_code(helion.Config(block_sizes=[]))


def _range_guard(code: str) -> Callable[[int | float], bool]:
    """Evaluate the single-bound trap predicate before the emitted range."""
    device = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
    guard = next(
        n for n in device.body if isinstance(n, ast.If) and "trap;" in ast.unparse(n)
    )
    assert guard.lineno < next(n.lineno for n in device.body if isinstance(n, ast.For))
    (operand,) = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)} - {
        "cutlass"
    }
    predicate = compile(ast.Expression(guard.test), "<device-range>", "eval")
    dsl = SimpleNamespace(Float64=float, Uint64=lambda v: int(v) % (2**64))
    return lambda value: eval(predicate, {operand: value, "cutlass": dsl})


@helion.kernel(backend="cute")
def loaded_bound(bounds, x, out):
    for row in hl.grid(2):
        for tile in hl.tile(bounds[0], block_size=4):
            out[row, tile] = torch.gather(
                x[row, tile], 0, (tile.index - tile.begin).to(torch.int64)
            )
    return out


def test_nested_range_uses_the_unsigned_value_of_a_loaded_bound() -> None:
    out = torch.empty(2, 4)
    x = torch.empty_like(out)
    code = _render(loaded_bound, torch.tensor([4], dtype=torch.uint64), x, out)
    fails = _range_guard(code)
    # The DSL loads PyTorch uint64 storage through its signed Int64 carrier.
    assert not fails(4) and fails(2**31) and fails(-(2**63)) and fails(-1)
    assert "trap;" not in _render(
        loaded_bound, torch.tensor([4], dtype=torch.int32), x, out
    )


def test_nested_range_keeps_float_conversion_toward_zero() -> None:
    x = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    out = torch.zeros_like(x)
    code = _render(loaded_bound, torch.tensor([4.0], dtype=torch.float64), x, out)
    fails = _range_guard(code)
    assert not any(fails(v) for v in (4.5, 2**31 - 0.5, -(2**31) - 0.5))
    assert all(fails(v) for v in (2**31, -(2**31) - 1, float("inf"), float("nan")))

    got, _ = execute_pointwise_thread_program(
        code, (torch.tensor([3.5], dtype=torch.float64), x, out), expected_writes=6
    )
    torch.testing.assert_close(
        got, torch.tensor([[1.0, 2.0, 3.0, 0.0], [5.0, 6.0, 7.0, 0.0]])
    )


def test_nested_range_proves_only_an_input_dimension_fits() -> None:
    @helion.kernel(backend="cute")
    def local_bound(x, out, n):
        local = x.new_empty((0, n + 1))
        for row in hl.grid(2):
            for tile in hl.tile(local.size(1), block_size=4):
                out[row, tile] = torch.gather(
                    x[row, tile], 0, (tile.index - tile.begin).to(torch.int64)
                )
        return out

    @helion.kernel(backend="cute")
    def input_bound(x, out):
        _, cols = out.size()
        for row in hl.grid(2):
            for tile in hl.tile(cols, block_size=4):
                out[row, tile] = torch.gather(
                    x[row, tile], 0, (3 - (tile.index - tile.begin)).to(torch.int64)
                )
        return out

    out = torch.empty(2, 4)
    code = _render(local_bound, out, out, 3)
    fails = _range_guard(code)
    assert not fails(4) and fails(2**31)
    x = torch.arange(8, dtype=torch.float32).reshape_as(out)
    input_code = _render(input_bound, x, out)
    assert "trap;" not in input_code
    got, _ = execute_pointwise_thread_program(input_code, (x, out))
    torch.testing.assert_close(got, x.flip(1))


def test_nested_step_requires_a_representable_magnitude() -> None:
    @helion.kernel(backend="cute")
    def literal(out, step: hl.constexpr):
        for row in hl.grid(2):
            for col in hl.grid(1, 0, step=step):
                out[row, col] = 9
        return out

    @helion.kernel(backend="cute")
    def dynamic(x, out, step):
        for row in hl.grid(2):
            for col in hl.grid(1, 0, step=step):
                for tile in hl.tile(4, block_size=4):
                    out[row, tile] = torch.gather(
                        x[col, tile], 0, (tile.index - tile.begin).to(torch.int64)
                    )
        return out

    out = torch.empty(2, 4)
    with pytest.raises(helion.exc.BackendUnsupported, match="magnitude of the step"):
        _render(literal, out, -(2**31))
    assert "trap;" not in _render(literal, out, -3)
    fails = _range_guard(_render(dynamic, out, out.clone(), -3))
    assert fails(0) and fails(-(2**31))
    assert not fails(-(2**31) + 1) and not fails(-3)


def test_plain_nested_dynamic_range_keeps_its_existing_lowering() -> None:
    @helion.kernel(backend="cute")
    def fill(out, lo, hi):
        for row in hl.grid(2):
            for tile in hl.tile(lo, hi, block_size=4):
                out[row, tile.index - tile.begin] = 7
        return out

    args = (torch.zeros(2, 4), 4, 8)
    code = _render(fill, *args)
    assert "trap;" not in code
    result, _ = execute_pointwise_thread_program(code, args)
    torch.testing.assert_close(result, torch.full_like(result, 7))


def test_gather_nested_steps_cannot_wrap_the_final_increment() -> None:
    @helion.kernel(backend="cute")
    def stepped(x, out, lo, hi, step):
        for row in hl.grid(2):
            for col in hl.grid(lo, hi, step=step):
                for t in hl.tile(4, block_size=4):
                    ids = (t.index - t.begin).to(torch.int64)
                    out[row, t] = torch.gather(x[row, t], 0, ids) + col
        return out

    limit = 2**31 - 1
    x = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    out = torch.full_like(x, -1)
    code = _render(stepped, x, out, 1, 3, limit)
    device = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
    guard = next(
        n for n in device.body if isinstance(n, ast.If) and "trap;" in ast.unparse(n)
    )
    counter = next(n for n in device.body if isinstance(n, ast.For))
    # Python's range does not model an overflowing CuTe backedge. Verify its
    # actual induction type as well as executing the emitted coordinates.
    assert guard.lineno < counter.lineno and "cutlass.Int64(0)" in ast.unparse(
        counter.iter
    )
    reject = compile(ast.Expression(guard.test), "<gather-bounds>", "eval")
    assert eval(reject, {"lo": 1, "hi": limit + 1, "step": limit})
    for lo, hi, step in ((1, 3, limit), (3, 1, -limit), (3, 1, limit), (1, 3, -limit)):
        nonempty = bool(range(lo, hi, step))
        got, _ = execute_pointwise_thread_program(
            code,
            (x, out.clone(), lo, hi, step),
            expected_writes=x.numel() if nonempty else 0,
        )
        torch.testing.assert_close(got, x + lo if nonempty else out)


def test_gather_tile_near_int32_stop_keeps_its_implicit_step() -> None:
    @helion.kernel(backend="cute")
    def tiled(x, out, sibling_out, lo, hi):
        for row in hl.grid(2):
            for t in hl.tile(lo, hi, block_size=4):
                position = t.index - t.begin
                out[row, position] = torch.gather(
                    x[row, t], 0, (position * 0).to(torch.int64)
                )
            # Widen the shared thread axis so each gather thread owns one lane.
            for sibling in hl.tile(128, block_size=128):
                sibling_out[row, sibling] = sibling.index + 1
        return out

    x = torch.tensor([[7], [9]], dtype=torch.int32)
    out = torch.zeros(2, 4, dtype=torch.int32)
    sibling_out = torch.empty(2, 128, dtype=torch.int32)
    args = (x, out, sibling_out, 2**31 - 3, 2**31 - 1)
    code = _render(tiled, *args)
    device = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
    counter = next(n for n in device.body if isinstance(n, ast.For))
    assert "cutlass.Int64(" in ast.unparse(counter.iter)
    got, _ = execute_pointwise_thread_program(code, args, expected_writes=260)
    torch.testing.assert_close(
        got, torch.tensor([[7, 7, 0, 0], [9, 9, 0, 0]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        sibling_out, torch.arange(1, 129, dtype=x.dtype).expand(2, -1)
    )


def test_loaded_nested_begin_uses_the_native_counter_for_its_mask() -> None:
    @helion.kernel(backend="cute", index_dtype=torch.int32)
    def fill(bounds, out):
        for row in hl.grid(2):
            for tile in hl.tile(bounds[0], bounds[1], block_size=4):
                out[row, 1 + (tile.index & 3)] = 7
        return out

    bounds = torch.tensor([2**32, 2**32 + 4])
    out = torch.full((2, 6), -11, dtype=torch.int32)
    got, _ = execute_pointwise_thread_program(
        _render(fill, bounds, out), (bounds, out), expected_writes=8
    )
    torch.testing.assert_close(
        got, torch.tensor([[-11, 7, 7, 7, 7, -11]] * 2, dtype=out.dtype)
    )


@pytest.mark.parametrize("threads", [1, 4])
def test_uint32_nested_tile_masks_lanes_at_the_signed_boundary(threads: int) -> None:
    @helion.kernel(backend="cute", index_dtype=torch.uint32)
    def fill(begin, out):
        for row in hl.grid(2):
            for tile in hl.tile(begin[0], 2**31 - 1):
                out[row, tile.index & 3] = 7
        return out

    args = (torch.tensor([2**31 - 2], dtype=torch.int32), torch.full((2, 4), -11))
    code = cpu_bind(fill, args).to_code(
        helion.Config(block_sizes=[4], num_threads=[threads])
    )
    # Python range visits just the first tile. Model its signed serial lane
    # addition at the native width; the cooperative layout adds a Uint32 tid.
    with np.errstate(over="ignore"):
        got, _ = execute_pointwise_thread_program(
            code, args, int32=np.int32, expected_writes=2
        )
    torch.testing.assert_close(got, torch.tensor([[-11, -11, 7, -11]] * 2))
