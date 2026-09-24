from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
_SCHEDULE = "cp_async_register_reuse_scan"
_MODES = (
    "same",
    "shift",
    "other_batch",
    "other_source",
    "gather",
    "multiple",
    "scan_only",
)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_cached_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    other: torch.Tensor,
    permutation: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batches, length, k = a.shape
    n = v.shape[2]
    out = torch.empty_like(v)
    for batch, row, col in hl.tile([batches, length, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(length)
        scan_value = delta[bi, qq].float()
        if mode == "multiple":
            scan_value *= other[bi, qq].float()
        decay = hl.cumsum(scan_value, dim=0)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        weights = first * torch.exp(decay[row][:, None] - decay[qq][None, :])
        if mode == "shift":
            scale = delta[bi, qq + 16]
            residual = delta[bi, row.index + 16]
        elif mode == "other_batch":
            scale = delta[(bi + 1) % batches, qq]
            residual = delta[(bi + 1) % batches, row]
        elif mode == "other_source":
            scale = other[bi, qq]
            residual = other[bi, row]
        elif mode == "gather":
            scale = delta[bi, permutation[qq]]
            residual = delta[bi, permutation[row]]
        elif mode == "multiple":
            scale = other[bi, qq]
            residual = delta[bi, row]
        else:
            scale = delta[bi, qq]
            residual = delta[bi, row]
        if mode != "scan_only":
            weights *= scale[None, :].float()
        weights = torch.where(row.index[:, None] >= qq[None, :], weights, 0.0)
        result = hl.dot(weights.to(v.dtype), v[bi, qq, col])
        if mode != "scan_only":
            result += residual[:, None].float()
        out[bi, row, col] = result.to(v.dtype)
    return out


def _inputs(device: Any, length: int, stride: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(481)
    operands = tuple(
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        * 0.1
        for shape in ((2, length, 64), (2, length, 64), (2, length, 32))
    )
    vectors = tuple(
        -torch.rand((2, (length + 32) * stride), device=device, generator=generator).to(
            torch.bfloat16
        )[:, ::stride]
        * 0.005
        for _ in range(2)
    )
    # Multiplication would make a contiguous tensor; retain noncontiguous views.
    if stride > 1:
        vectors = tuple(
            vector.repeat_interleave(stride, dim=1)[:, ::stride] for vector in vectors
        )
    permutation = torch.randperm(length, device=device, generator=generator)
    return (*operands, *vectors, permutation)


def _cpu_code(
    args: tuple[torch.Tensor, ...],
    mode: str,
    schedule: str = _SCHEDULE,
    capacity: int = 232448,
    warps: int = 4,
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
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
        ),
    ):
        _scan_cached_chain.reset()
        return _scan_cached_chain.bind((*args, mode)).to_triton_code(
            helion.Config(cute_chained_mma_schedule=schedule, num_warps=warps)
        )


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("length,stride", [(64, 1), (49, 2)])
def test_scan_input_cache_codegen(mode: str, length: int, stride: int) -> None:
    code = _cpu_code(_inputs("cpu", length, stride), mode)
    tree = ast.parse(code)
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id.startswith("chain_scan_0_input_")
            for target in node.targets
        )
    ]
    expected = (
        2 if mode == "multiple" else 0 if mode in ("scan_only", "other_source") else 1
    )
    assert len(stores) == expected
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.IfExp)
        and "chain_scan_0_input_" in ast.unparse(node.body)
    ]
    if mode in ("same", "shift", "multiple"):
        assert reads
        assert any("chain_store" in ast.unparse(node.test) for node in reads)
        assert any("bridge" in ast.unparse(node.test) for node in reads)
        if length == 49:
            assert all("< 49" in ast.unparse(node.test) for node in reads)
    else:
        assert not reads
    for store in stores:
        # Store the existing loaded register, never issue another global read.
        assert ".load()" not in ast.unparse(store.value)


def test_scan_input_cache_is_optional() -> None:
    code = _cpu_code(_inputs("cpu", 64, 1), "same", "cp_async_register_reuse")
    assert "chain_scan_0_input_" not in code


def test_scan_input_cache_shared_memory_admission() -> None:
    args = _inputs("cpu", 64, 1)
    # Two dot shapes (16,64,64), (16,32,64), four scan warp totals.
    allocations = (
        2 * (16 * 64 + 8 * 64),
        2 * (64 * 64 + 8 * 64),
        4 * 16 * (64 + 4),
        4 * 16 * (32 + 4),
        4 * (64 + 4),
    )
    capacity = sum((size + 127) // 128 * 128 for size in allocations)
    assert "chain_1_mma" in _cpu_code(args, "same", "cp_async_register_reuse", capacity)
    with pytest.raises(exc.BackendUnsupported, match="associative_scan input"):
        _cpu_code(args, "same", capacity=capacity)
    # The non-cache schedule must also count scan allocation alignment.
    with pytest.raises(exc.BackendUnsupported, match="associative_scan input"):
        _cpu_code(args, "same", "cp_async_register_reuse", sum(allocations))


def test_scan_input_cache_logical_tail_predicate() -> None:
    code = _cpu_code(_inputs("cpu", 49, 2), "shift")
    reads = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.IfExp)
        and "chain_scan_0_input_" in ast.unparse(node.body)
        and "chain_store" in ast.unparse(node.test)
    ]
    assert reads
    values = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "chain_origin_0": 0,
        "chain_origin_1": 32,
        "chain_origin_2": 0,
    }
    for read in reads:
        condition = compile(ast.Expression(read.test), "<scan-cache-domain>", "eval")
        # Global indices48 and49 both exist and fit shared64; logical49 is
        # outside the scan's original vector and must use global fallback.
        assert eval(condition, {}, {**values, "chain_store": 0})
        assert not eval(condition, {}, {**values, "chain_store": 32})


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scan_input_cache_dtype_and_eight_warps_codegen(dtype: torch.dtype) -> None:
    args = _inputs("cpu", 49, 2)
    args = (*args[:3], args[3].to(dtype), args[4].to(dtype), args[5])
    code = _cpu_code(args, "multiple", warps=8)
    assert "chain_scan_0_input_1" in code
    dtype_name = "cutlass.Float16" if dtype is torch.float16 else "cutlass.Float32"
    assert f"{dtype_name}(chain_scan_0_input_0[" in code
    assert "if chain_scan_0_index < 64:" in code
    assert "cute.make_layout(8)" in code


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("length,stride", [(64, 1), (49, 2)])
def test_scan_input_cache_runtime(mode: str, length: int, stride: int) -> None:
    args = _inputs(DEVICE, length, stride)
    frozen = tuple(arg.clone() for arg in args)
    fn = _scan_cached_chain.bind((*args, mode)).compile_config(
        helion.Config(cute_chained_mma_schedule=_SCHEDULE)
    )
    output = fn(*args, mode)
    a, b, v, delta, other, permutation = args
    scan = delta[:, :length].double()
    if mode == "multiple":
        scan *= other[:, :length].double()
    decay = scan.cumsum(-1)
    weights = (a.double() @ b.double().transpose(-1, -2)) * (
        decay[:, :, None] - decay[:, None, :]
    ).exp()
    selected = delta[:, :length]
    if mode == "shift":
        selected = delta[:, 16 : length + 16]
    elif mode == "other_batch":
        selected = delta.flip(0)[:, :length]
    elif mode in ("other_source", "multiple"):
        selected = other[:, :length]
    elif mode == "gather":
        selected = delta[:, permutation]
    if mode != "scan_only":
        weights *= selected[:, None, :].double()
    result = torch.tril(weights).to(v.dtype).double() @ v.double()
    if mode != "scan_only":
        residual = delta[:, :length] if mode == "multiple" else selected
        result += residual[:, :, None].double()
    torch.testing.assert_close(
        output,
        result.to(v.dtype),
        atol=0.003 if mode == "scan_only" else 0.0001,
        rtol=0.02,
    )
    torch.testing.assert_close(fn(*args, mode), output, atol=0, rtol=0)
    assert output.data_ptr() not in {arg.data_ptr() for arg in args}
    for actual, original in zip(args, frozen, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
