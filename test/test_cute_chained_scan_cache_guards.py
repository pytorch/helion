from __future__ import annotations

import ast
from types import SimpleNamespace
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
from helion.language import _tracing_ops
from helion.language import memory_ops
from helion.language import scan_ops

pytestmark = skipUnlessBackends(["cute"])
_MODES = ("scalar_last", "scalar_outside", "broadcast", "reverse", "nested")
_SCHEDULE = "cp_async_register_reuse_scan"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_guard_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batches, length, k = a.shape
    n = v.shape[2]
    out = torch.empty_like(v)
    for batch, row, col in hl.tile([batches, length, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(length)
        scan_value = delta[bi, qq].float()
        if mode == "reverse":
            scan_value = delta[bi, length - 1 - qq].float()
        decay = hl.cumsum(scan_value * 0.01, dim=0)
        if mode == "nested":
            decay = hl.cumsum(decay * 0.01 + delta[bi, qq].float() * 0.01, dim=0)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        weights = first * torch.exp(decay[row][:, None] - decay[qq][None, :])
        weights *= delta[bi, qq][None, :].float()
        weights = torch.where(row.index[:, None] >= qq[None, :], weights, 0.0)
        result = hl.dot(weights.to(v.dtype), v[bi, qq, col])
        if mode == "scalar_outside":
            result += delta[bi, length].float()
        elif mode == "broadcast":
            result += delta[bi, 0].float()
        else:
            result += delta[bi, length - 1].float()
        out[bi, row, col] = result.to(v.dtype)
    return out


def _inputs(device: Any, seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    operands = tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.1
        for shape in ((2, 49, 64), (2, 49, 64), (2, 49, 32))
    )
    delta = (
        torch.randn((2, 80), device=device, dtype=torch.float32, generator=generator)
        * 0.1
    )
    return (*operands, delta)


def _cpu_code(args: tuple[torch.Tensor, ...], mode: str) -> str:
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
        _scan_guard_chain.reset()
        return _scan_guard_chain.bind((*args, mode)).to_code(
            helion.Config(num_warps=8, cute_chained_mma_schedule=_SCHEDULE)
        )


@pytest.mark.parametrize("mode", _MODES)
def test_scan_cache_scalar_and_nested_codegen(mode: str) -> None:
    code = _cpu_code(_inputs("cpu"), mode)
    assert "chain_scan_0_input_0" in code
    assert ("chain_scan_1_input_0" in code) is (mode == "nested")
    # Scalar host loads can safely read the vector cache, including broadcast
    # to every output coordinate. An index49 request is globally valid in the
    # 80-element input but outside the original49-element scan vector.
    epilogue = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For) and ast.unparse(node.target) == "chain_store_step"
    )
    reads = [
        node
        for node in ast.walk(epilogue)
        if isinstance(node, ast.IfExp)
        and "chain_scan_0_input_" in ast.unparse(node.body)
    ]
    assert reads
    values = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "chain_origin_0": 0,
        "chain_origin_1": 0,
        "chain_origin_2": 0,
        "chain_store": 0,
    }
    for read in reads:
        assert ".load()" in ast.unparse(read.orelse)
        actual = bool(
            eval(
                compile(ast.Expression(read.test), "<scalar-cache>", "eval"), {}, values
            )
        )
        assert actual is (mode != "scalar_outside")


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int64,
        torch.bool,
    ],
)
def test_scan_cache_lossless_dtype_policy(dtype: torch.dtype) -> None:
    graph = torch.fx.Graph()
    source = graph.call_function(_tracing_ops._host_tensor, ("delta",))
    source.meta["val"] = torch.empty(49, dtype=dtype)
    leaf = graph.call_function(memory_ops.load, (source, [slice(None)], None))
    leaf.meta["val"] = torch.empty(49, dtype=dtype)
    scan = graph.call_function(scan_ops._associative_scan, (0, leaf, 0, False, False))
    later = graph.call_function(torch.add, (scan, leaf))
    output = graph.call_function(_tracing_ops._host_tensor, ("out",))
    store = graph.call_function(memory_ops.store, (output, [], later, None))
    # Isolate the cache admission policy from tile-shape configuration. The
    # supported floating leaf dtypes are cached without conversion.
    with patch.object(
        chained_matmul, "_shape", side_effect=lambda node: tuple(node.meta["val"].shape)
    ):
        candidates = chained_matmul._scan_cache_candidates(scan, (scan,), store)
    assert (leaf in candidates) is (
        dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _reference(args: tuple[torch.Tensor, ...], mode: str) -> torch.Tensor:
    a, b, v, delta = args
    scan = delta[:, :49].double()
    if mode == "reverse":
        scan = scan.flip(1)
    decay = (scan * 0.01).cumsum(-1)
    if mode == "nested":
        decay = (decay * 0.01 + delta[:, :49].double() * 0.01).cumsum(-1)
    first = a.double() @ b.double().transpose(-1, -2)
    weights = first * torch.exp(decay[:, :, None] - decay[:, None, :])
    weights = torch.tril(weights * delta[:, None, :49].double()).to(v.dtype)
    scalar = 49 if mode == "scalar_outside" else 0 if mode == "broadcast" else 48
    result = weights.double() @ v.double() + delta[:, scalar, None, None].double()
    return result.to(v.dtype)


@pytest.mark.parametrize("mode", _MODES)
def test_scan_cache_scalar_and_nested_runtime(mode: str) -> None:
    args = _inputs(DEVICE)
    _scan_guard_chain.reset()
    fn = _scan_guard_chain.bind((*args, mode)).compile_config(
        helion.Config(num_warps=8, cute_chained_mma_schedule=_SCHEDULE)
    )
    for seed in range(5):
        args = _inputs(DEVICE, seed)
        frozen = tuple(value.clone() for value in args)
        actual = fn(*args, mode)
        repeated = fn(*args, mode)
        torch.testing.assert_close(
            actual, _reference(args, mode), atol=0.003, rtol=0.02
        )
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        assert actual.data_ptr() != repeated.data_ptr()
        for value, original in zip(args, frozen, strict=True):
            torch.testing.assert_close(value, original, atol=0, rtol=0)
