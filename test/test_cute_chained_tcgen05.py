from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import chained_matmul
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = skipUnlessBackends(["cute"])


# Memory.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _memory_chain(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, None, None]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, qq, col]).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _computed_memory_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, None, None]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        first = hl.dot(
            (a[bi, row, kk].float() * 0.5).to(a.dtype),
            (b[bi, qq, kk].float() * 0.75).to(a.dtype).T,
        )
        out[bi, row, col] = hl.dot(
            first.to(a.dtype), (v[bi, qq, col].float() * 0.25).to(a.dtype)
        ).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gathered_memory_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    permutation_k: torch.Tensor,
    permutation_q: torch.Tensor,
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, None, None]):
        bi = batch.begin
        kk = permutation_k[hl.arange(k)]
        qq = hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, permutation_q[qq], col]).to(
            a.dtype
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _both_operands_chain(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batches, m, k = a.shape
    out = torch.empty((batches, m, m), device=a.device, dtype=a.dtype)
    for batch in hl.tile(batches, block_size=1):
        bi = batch.begin
        qq = hl.arange(m)
        kk = hl.arange(k)
        first = hl.dot(a[bi, qq, kk], b[bi, qq, kk].T)
        converted = first.to(a.dtype)
        out[bi, :, :] = hl.dot(converted, converted).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transpose_bridge_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batches, m, k = a.shape
    n = v.shape[2]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch in hl.tile(batches, block_size=1):
        bi = batch.begin
        qq = hl.arange(m)
        kk = hl.arange(k)
        nn = hl.arange(n)
        first = hl.dot(a[bi, qq, kk], b[bi, qq, kk].T)
        out[bi, :, :] = hl.dot(first.T.to(a.dtype), v[bi, qq, nn]).to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _live_producer_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch in hl.tile(batches, block_size=1):
        bi = batch.begin
        mm = hl.arange(m)
        kk = hl.arange(k)
        qq = hl.arange(q)
        nn = hl.arange(n)
        first = hl.dot(a[bi, mm, kk], b[bi, qq, kk].T)
        result = hl.dot(first.to(a.dtype), v[bi, qq, nn])
        out[bi, :, :] = (result + first).to(a.dtype)
    return out


def _view(shape: tuple[int, ...], layout: str, device: Any) -> torch.Tensor:
    batches, rows, cols = shape
    if layout.startswith("offset"):
        offset = int(layout.removeprefix("offset"))
        storage = torch.randn(
            math.prod(shape) + offset, device=device, dtype=torch.bfloat16
        )
        value = storage[offset:].view(shape)
    elif layout.startswith("padded"):
        padding = int(layout.removeprefix("padded"))
        value = torch.randn(
            (batches, rows, cols + padding), device=device, dtype=torch.bfloat16
        )[..., :cols]
    elif layout == "inner_stride2":
        value = torch.randn(
            (batches, rows, cols * 2), device=device, dtype=torch.bfloat16
        )[..., ::2]
    elif layout == "transpose":
        value = torch.randn(
            (batches, cols, rows), device=device, dtype=torch.bfloat16
        ).transpose(1, 2)
    else:
        assert layout == "contiguous"
        value = torch.randn(shape, device=device, dtype=torch.bfloat16)
    return value.mul_(0.1)


def _memory_inputs(
    layout: str, device: Any, *, tails: bool = False
) -> tuple[torch.Tensor, ...]:
    m, k, q, n = (35, 64, 49, 24) if tails else (32, 64, 64, 32)
    return tuple(
        _view(shape, layout, device) for shape in ((2, m, k), (2, q, k), (2, q, n))
    )


def _memory_config(
    warps: int, schedule: str = "cp_async", *, full: bool = False
) -> helion.Config:
    return helion.Config(
        block_sizes=[] if full else [16, 32],
        num_warps=warps,
        cute_chained_mma_schedule=schedule,
    )


def _memory_reference(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    first = (a.double() @ b.double().transpose(-1, -2)).to(a.dtype)
    return (first.double() @ v.double()).to(a.dtype)


def _memory_check(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    expected: torch.Tensor,
) -> None:
    frozen = tuple(arg.clone() for arg in args)
    first = fn(*args)
    saved = first.clone()
    second = fn(*args)
    torch.testing.assert_close(first, expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(second, first, atol=0, rtol=0)
    torch.testing.assert_close(first, saved, atol=0, rtol=0)
    assert first.data_ptr() != second.data_ptr()
    for value, original in zip(args, frozen, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)


def _memory_cpu_code(
    kernel: Any,
    args: tuple[torch.Tensor, ...],
    warps: int,
    schedule: str = "cp_async",
    *,
    full: bool = False,
) -> str:
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
        kernel.reset()
        return kernel.bind(args).to_code(_memory_config(warps, schedule, full=full))


def test_async_memory_codegen_patches_already_imported_support() -> None:
    with patch.object(
        chained_matmul,
        "get_cute_mma_support",
        side_effect=AssertionError("unpatched previously imported binding"),
    ) as original:
        code = _memory_cpu_code(_memory_chain, _memory_inputs("contiguous", "cpu"), 4)
        assert "chain_0_mma" in code
        original.assert_not_called()
        assert chained_matmul.get_cute_mma_support is original


@pytest.mark.parametrize("warps", [1, 2, 4])
@pytest.mark.parametrize(
    "layout,expected_async",
    [
        ("contiguous", True),
        ("offset1", True),
        ("offset8", True),
        ("padded1", False),
        ("padded8", True),
        ("inner_stride2", False),
        ("transpose", True),
    ],
)
def test_async_memory_codegen_guards(
    layout: str, expected_async: bool, warps: int
) -> None:
    code = _memory_cpu_code(_memory_chain, _memory_inputs(layout, "cpu"), warps)
    assert "_helion_cute_pointer_alignment = 1" in code
    assert ("CopyG2SOp" in code) is expected_async
    guards = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.If) and ".toint() % 16" in ast.unparse(node.test)
    ]
    assert bool(guards) is expected_async
    for guard in guards:
        condition = ast.unparse(guard.test)
        assert "chain_thread" not in condition
        assert ".load()" not in condition
        assert "0 <=" in condition
        assert "CopyG2SOp" in ast.unparse(guard)
        assert ".load()" in "\n".join(ast.unparse(node) for node in guard.orelse)
    if expected_async:
        assert "cp_async_wait_group(0)" in code


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_codegen_tail_bounds(warps: int) -> None:
    code = _memory_cpu_code(
        _memory_chain, _memory_inputs("contiguous", "cpu", tails=True), warps
    )
    assert "CopyG2SOp" in code
    guards = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.If) and ".toint() % 16" in ast.unparse(node.test)
    ]
    conditions = "\n".join(ast.unparse(node.test) for node in guards)
    assert "< 35" in conditions
    assert "< 49" in conditions
    assert ".load()" in code


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_codegen_computed_and_gathered_fallback(warps: int) -> None:
    args = _memory_inputs("contiguous", "cpu")
    code = _memory_cpu_code(_computed_memory_chain, args, warps)
    assert "chain_0_mma" in code
    assert "CopyG2SOp" not in code
    gathered = (*args, torch.randperm(64), torch.randperm(64))
    code = _memory_cpu_code(_gathered_memory_chain, gathered, warps)
    assert "chain_0_mma" in code
    assert "CopyG2SOp" not in code


@pytest.mark.parametrize("warps", [1, 2, 4])
@pytest.mark.parametrize(
    "layout", ["offset1", "offset8", "padded1", "padded8", "inner_stride2", "transpose"]
)
def test_async_memory_views(layout: str, warps: int) -> None:
    args = _memory_inputs(layout, DEVICE)
    _memory_chain.reset()
    fn = _memory_chain.bind(args).compile_config(_memory_config(warps))
    _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_alignment_is_checked_on_every_call(warps: int) -> None:
    aligned = _memory_inputs("contiguous", DEVICE)
    _memory_chain.reset()
    fn = _memory_chain.bind(aligned).compile_config(_memory_config(warps))
    _memory_check(fn, aligned, _memory_reference(*aligned))
    for offset in (1, 8):
        args = _memory_inputs(f"offset{offset}", DEVICE)
        assert all(
            value.stride() == old.stride()
            for value, old in zip(args, aligned, strict=True)
        )
        _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_strides_are_checked_on_every_call(warps: int) -> None:
    aligned = _memory_inputs("contiguous", DEVICE)
    _memory_chain.reset()
    fn = _memory_chain.bind(aligned).compile_config(_memory_config(warps))
    _memory_check(fn, aligned, _memory_reference(*aligned))
    for layout in ("padded8", "transpose", "inner_stride2"):
        args = _memory_inputs(layout, DEVICE)
        assert all(
            value.shape == old.shape for value, old in zip(args, aligned, strict=True)
        )
        _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_register_bridge_codegen_exclusive_region(warps: int) -> None:
    args = tuple(_view((2, 32, 64), "contiguous", "cpu") for _ in range(2))
    square = _view((2, 32, 32), "contiguous", "cpu")
    for kernel, values in (
        (_both_operands_chain, args),
        (_transpose_bridge_chain, (*args, square)),
        (_live_producer_chain, (*args, square)),
    ):
        code = _memory_cpu_code(kernel, values, warps, "cp_async_register", full=True)
        assert "chain_0_c_ptr" in code
        assert "_bridge_values" not in code


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_register_bridge_codegen_narrow_producer(warps: int) -> None:
    args = tuple(
        _view(shape, "contiguous", "cpu")
        for shape in ((2, 32, 64), (2, 16, 64), (2, 16, 32))
    )
    code = _memory_cpu_code(_memory_chain, args, warps, "cp_async_register")
    assert ("chain_1_a_bridge_values" in code) is (warps != 4)
    assert ("chain_0_c_ptr" in code) is (warps == 4)


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_register_bridge_exclusive_region(warps: int) -> None:
    a, b = (_view((2, 32, 64), "contiguous", DEVICE) for _ in range(2))
    v = _view((2, 32, 32), "contiguous", DEVICE)
    first = a.double() @ b.double().transpose(-1, -2)
    converted = first.to(a.dtype).double()
    cases = (
        (_both_operands_chain, (a, b), (converted @ converted).to(a.dtype)),
        (
            _transpose_bridge_chain,
            (a, b, v),
            (converted.transpose(-1, -2) @ v.double()).to(a.dtype),
        ),
        (_live_producer_chain, (a, b, v), (converted @ v.double() + first).to(a.dtype)),
    )
    for kernel, args, expected in cases:
        kernel.reset()
        fn = kernel.bind(args).compile_config(
            _memory_config(warps, "cp_async_register", full=True)
        )
        _memory_check(fn, args, expected)


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_register_bridge_tail_tiles(warps: int) -> None:
    args = _memory_inputs("contiguous", DEVICE, tails=True)
    _memory_chain.reset()
    fn = _memory_chain.bind(args).compile_config(
        _memory_config(warps, "cp_async_register")
    )
    _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_tail_tiles(warps: int) -> None:
    args = _memory_inputs("contiguous", DEVICE, tails=True)
    _memory_chain.reset()
    fn = _memory_chain.bind(args).compile_config(_memory_config(warps))
    _memory_check(fn, args, _memory_reference(*args))


@pytest.mark.parametrize("warps", [1, 2, 4])
def test_async_memory_computed_and_gathered_fallback(warps: int) -> None:
    args = _memory_inputs("contiguous", DEVICE)
    _computed_memory_chain.reset()
    fn = _computed_memory_chain.bind(args).compile_config(_memory_config(warps))
    transformed = tuple(
        (value.float() * scale).to(value.dtype)
        for value, scale in zip(args, (0.5, 0.75, 0.25), strict=True)
    )
    _memory_check(fn, args, _memory_reference(*transformed))
    pk, pq = (torch.randperm(64, device=DEVICE) for _ in range(2))
    gathered = (*args, pk, pq)
    _gathered_memory_chain.reset()
    fn = _gathered_memory_chain.bind(gathered).compile_config(_memory_config(warps))
    a, b, v = args
    _memory_check(fn, gathered, _memory_reference(a[..., pk], b[..., pk], v[:, pq, :]))
