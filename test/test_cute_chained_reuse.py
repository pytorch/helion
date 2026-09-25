from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
_MODES = (
    "same",
    "shift",
    "reverse",
    "other_batch",
    "other_source",
    "computed",
    "earlier",
    "gather",
    "constant_col",
)
_REUSES = {"same", "shift", "reverse", "constant_col"}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reuse_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    other: torch.Tensor,
    permutation: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = b.shape[1], v.shape[2]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk, qq = hl.arange(k), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        last = v[bi, qq, col]
        if mode == "computed":
            last = (last.float() * 0.5).to(v.dtype)
        if mode == "gather":
            last = v[bi, permutation[qq], col]
        result = hl.dot(first.to(a.dtype), last)
        if mode == "shift":
            residual = v[bi, row.index + 32, col]
        elif mode == "reverse":
            residual = v[bi, 79 - row.index, col]
        elif mode == "other_batch":
            residual = v[(bi + 1) % batches, row, col]
        elif mode == "other_source":
            residual = other[bi, row, col]
        elif mode == "earlier":
            residual = b[bi, row, col]
        elif mode == "constant_col":
            residual = v[bi, row, 0][:, None]
        else:
            residual = v[bi, row, col]
        out[bi, row, col] = (result + residual.float()).to(a.dtype)
    return out


def _config(warps: int) -> helion.Config:
    return helion.Config(
        num_warps=warps, cute_chained_mma_schedule="cp_async_register_reuse"
    )


def _inputs(device: Any, seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    # K=64, Q=49 and M=35 stress padded domains. V's physical length80 means
    # indices49..63 exist globally but must not read masked shared padding.
    values = tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.1
        for shape in ((2, 35, 64), (2, 49, 64), (2, 80, 64), (2, 80, 64))
    )
    return (*values, torch.randperm(49, device=device, generator=generator))


def _cpu_code(args: tuple[torch.Tensor, ...], mode: str, warps: int) -> str:
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
        _reuse_chain.reset()
        return _reuse_chain.bind((*args, mode)).to_code(_config(warps))


def _reuse_expressions(code: str) -> list[ast.IfExp]:
    epilogue = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "chain_store_step"
    )
    return [
        node
        for node in ast.walk(epilogue)
        if isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Subscript)
        and isinstance(node.body.value, ast.Name)
        and node.body.value.id in {"chain_1_a", "chain_1_b"}
    ]


def _predicate(
    condition: ast.expr, *, row_origin: int, col_origin: int, store: int = 0
) -> bool:
    values = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "chain_origin_0": 0,
        "chain_origin_1": row_origin,
        "chain_origin_2": col_origin,
        "chain_store": store,
    }
    return bool(
        eval(compile(ast.Expression(condition), "<reuse-domain>", "eval"), {}, values)
    )


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("mode", _MODES)
def test_staged_reuse_codegen_eligibility(mode: str, warps: int) -> None:
    code = _cpu_code(_inputs("cpu"), mode, warps)
    assert f"block=({warps * 32}, 1, 1)" in code
    expressions = _reuse_expressions(code)
    assert bool(expressions) is (mode in _REUSES)
    for expression in expressions:
        assert ".load()" in ast.unparse(expression.orelse)
        assert "chain_0_b[" not in ast.unparse(expression)
    if mode == "shift":
        assert len(expressions) == 1
        condition = expressions[0].test
        assert _predicate(condition, row_origin=0, col_origin=0)
        # Global V[49] is valid, but outside the 49 staged values. Padding
        # shared memory to64 must not replace a valid global value with zero.
        assert not _predicate(condition, row_origin=16, col_origin=0, store=32)
        assert "< 49" in ast.unparse(condition)
    if mode == "constant_col":
        assert len(expressions) == 1
        condition = expressions[0].test
        assert _predicate(condition, row_origin=0, col_origin=0)
        assert not _predicate(condition, row_origin=0, col_origin=32)


def _reference(args: tuple[torch.Tensor, ...], mode: str) -> torch.Tensor:
    a, b, v, other, permutation = args
    first = (a.double() @ b.double().transpose(-1, -2)).to(a.dtype)
    last = v[:, :49, :]
    if mode == "computed":
        last = (last.float() * 0.5).to(v.dtype)
    if mode == "gather":
        last = v[:, permutation, :]
    result = first.double() @ last.double()
    if mode == "shift":
        residual = v[:, 32:67, :]
    elif mode == "reverse":
        residual = v[:, 45:80, :].flip(1)
    elif mode == "other_batch":
        residual = v.flip(0)[:, :35, :]
    elif mode == "other_source":
        residual = other[:, :35, :]
    elif mode == "earlier":
        residual = b[:, :35, :]
    elif mode == "constant_col":
        residual = v[:, :35, :1]
    else:
        residual = v[:, :35, :]
    return (result + residual.double()).to(a.dtype)


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("mode", _MODES)
def test_staged_reuse_numerics(mode: str, warps: int) -> None:
    args = _inputs(DEVICE)
    _reuse_chain.reset()
    fn = _reuse_chain.bind((*args, mode)).compile_config(_config(warps))
    for seed in range(5):
        args = _inputs(DEVICE, seed)
        if mode == "other_source" and seed % 2:
            # Reusing the same compiled callable must also remain correct
            # when two previously distinct read-only arguments alias.
            args = (*args[:3], args[2], args[4])
        originals = tuple(value.clone() for value in args)
        result = fn(*args, mode)
        repeated = fn(*args, mode)
        assert result.dtype is torch.bfloat16
        assert result.data_ptr() != repeated.data_ptr()
        torch.testing.assert_close(result, _reference(args, mode), atol=0.01, rtol=0.01)
        torch.testing.assert_close(repeated, result, atol=0, rtol=0)
        for value, original in zip(args, originals, strict=True):
            torch.testing.assert_close(value, original, atol=0, rtol=0)
