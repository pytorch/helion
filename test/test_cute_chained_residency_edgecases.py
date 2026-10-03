from __future__ import annotations

import ast
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion import exc
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_seed(a, b, c, d, coefficients):
    m, k = a.shape
    n = b.size(1)
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows, cols in hl.tile([m, n], block_size=[128, 32]):
        kk = hl.arange(k)
        prefix = hl.cumsum(coefficients[rows], dim=0)
        seed = torch.exp(prefix)[:, None] + hl.zeros([rows, cols], dtype=torch.float32)
        first = hl.dot(a[rows, kk], b[kk, cols], acc=seed, out_dtype=torch.float32)
        output[rows, cols] = hl.dot(
            c[rows, kk], d[kk, cols], acc=seed + first, out_dtype=torch.float32
        )
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _bridge_coefficients(a, b, c, d, scale):
    m, k = a.shape
    output = torch.empty_like(a, dtype=torch.float32)
    for rows, cols in hl.tile([m, k], block_size=[128, 64]):
        kk, pp, qq = hl.arange(k), hl.arange(b.size(-1)), hl.arange(c.size(-1))
        coefficient = torch.exp(scale[rows])[:, None]
        first = hl.dot(a[rows, kk], b[kk, pp], out_dtype=torch.float32)
        second = hl.dot(
            (first * coefficient).to(a.dtype), c[pp, qq], out_dtype=torch.float32
        )
        output[rows, cols] = hl.dot(
            (second * coefficient).to(a.dtype), d[qq, cols], out_dtype=torch.float32
        )
    return output


def _args(width):
    return (
        *(
            torch.empty(shape, dtype=torch.bfloat16)
            for shape in ((128, width), (width, width), (128, width), (width, width))
        ),
        torch.empty((128,), dtype=torch.float32),
    )


def _config(schedule, budget):
    return helion.Config(
        num_warps=4,
        cute_chained_mma_schedule=schedule,
        cute_chained_pointwise_cache_bytes=budget,
    )


def _code(kernel, args, schedule, budget):
    with _cpu_codegen():
        return kernel._bind_isolated(args).to_code(_config(schedule, budget))


def _cache_reads(source):
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and ast.unparse(node.value) == "chain_pointwise_cache_0"
    ]


def test_scan_used_only_by_explicit_accumulators_is_published_before_cache_fill():
    source = _code(_scan_seed, _args(32), "tcgen05_tmem", 4096)
    statements = list(ast.walk(ast.parse(source)))
    cache_loop = next(
        node
        for node in statements
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "chain_pointwise_cache_0_step"
    )
    scan_reads = [
        node
        for node in ast.walk(cache_loop)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and ast.unparse(node.value).startswith("chain_scan_0")
    ]
    assert scan_reads
    scan_name = ast.unparse(scan_reads[0].value)
    declaration = next(
        node
        for node in statements
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == scan_name for target in node.targets)
    )
    stores = [
        node
        for node in statements
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript) and ast.unparse(target.value) == scan_name
            for target in node.targets
        )
    ]
    assert declaration.lineno < cache_loop.lineno
    assert stores and max(node.lineno for node in stores) < cache_loop.lineno
    assert len(_cache_reads(source)) == 2


@pytest.mark.parametrize("schedule", ["cp_async_register", "tcgen05_tmem"])
def test_bridged_operands_really_consume_resident_coefficient(schedule):
    args = (
        torch.empty((128, 64), dtype=torch.bfloat16),
        *(torch.empty((64, 64), dtype=torch.bfloat16) for _ in range(3)),
        torch.empty((128,), dtype=torch.float32),
    )
    ordinary = _code(_bridge_coefficients, args, schedule, 0)
    cached = _code(_bridge_coefficients, args, schedule, 4096)
    assert "bridge_values" in ordinary
    assert len(_cache_reads(cached)) == 2
    # Warp bridges precompute expression statements before publication, so
    # their affected operands must use ordinary cache-aware staging. TCgen
    # bridges emit a fresh expression at the consumption stage and remain safe.
    assert ("bridge_values" in cached) is (schedule == "tcgen05_tmem")


def test_resident_coefficient_counts_against_root_tcgen_capacity():
    footprints = []
    shared_memory_bytes = chained_tcgen05._shared_memory_bytes

    def record_footprint(plan, *, startup=False):
        total = shared_memory_bytes(plan, startup=startup)
        if plan.pointwise_cache is not None:
            base = shared_memory_bytes(
                replace(plan, pointwise_cache=None), startup=startup
            )
            footprints.append((base, total, plan.pointwise_cache.shared_bytes))
        return total

    args = _args(32)
    with patch.object(
        chained_tcgen05, "_shared_memory_bytes", side_effect=record_footprint
    ):
        _code(_scan_seed, args, "tcgen05_tmem", 4096)
    assert footprints
    base, total, cache_bytes = footprints[0]
    assert all(footprint == footprints[0] for footprint in footprints)
    assert cache_bytes > 0 and total == base + cache_bytes

    # Observe the real footprint above, then vary only the hardware resource
    # limit. No classifier or admission predicate is replaced by this test.
    with _cpu_codegen():
        with patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=base
        ):
            ordinary = _scan_seed._bind_isolated(args).to_code(
                _config("tcgen05_tmem", 0)
            )
            assert "chain_pointwise_cache_0" not in ordinary
            with pytest.raises(exc.BackendUnsupported):
                _scan_seed._bind_isolated(args).to_code(_config("tcgen05_tmem", 4096))
        with patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=total
        ):
            cached = _scan_seed._bind_isolated(args).to_code(
                _config("tcgen05_tmem", 4096)
            )
            assert len(_cache_reads(cached)) == 2


def _runtime_inputs(kind, dtype):
    generator = torch.Generator(device=DEVICE).manual_seed(943)
    width = 32 if kind == "scan_seed" else 64
    shapes = (
        (128, width),
        (width, width),
        (128 if kind == "scan_seed" else width, width),
        (width, width),
    )
    matrices = tuple(
        (
            torch.randn(shape, device=DEVICE, generator=generator, dtype=torch.float32)
            * 0.05
        ).to(dtype)
        for shape in shapes
    )
    coefficient = (
        torch.randn((128,), device=DEVICE, generator=generator, dtype=torch.float32)
        * 0.01
    )
    return (*matrices, coefficient)


def _reference(kind, args):
    a, b, c, d, coefficient = args
    if kind == "scan_seed":
        seed = torch.exp(torch.cumsum(coefficient, dim=0))[:, None]
        first = (a.double() @ b.double()).float() + seed
        return (c.double() @ d.double()).float() + (seed + first)
    scale = torch.exp(coefficient)[:, None]
    first = (a.double() @ b.double()).float()
    second = (((first * scale).to(a.dtype)).double() @ c.double()).float()
    return (((second * scale).to(a.dtype)).double() @ d.double()).float()


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "kind,schedule",
    [
        ("scan_seed", "tcgen05_tmem"),
        ("bridge", "cp_async_register"),
        ("bridge", "tcgen05_tmem"),
    ],
)
def test_residency_seed_and_bridge_gpu(kind, schedule, dtype):
    args = _runtime_inputs(kind, dtype)
    saved = tuple(arg.clone() for arg in args)
    kernel = _scan_seed if kind == "scan_seed" else _bridge_coefficients
    ordinary = kernel._bind_isolated(args).compile_config(_config(schedule, 0))
    cached = kernel._bind_isolated(args).compile_config(_config(schedule, 4096))
    expected, actual = ordinary(*args), cached(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cached(*args), actual, rtol=0, atol=0)
    torch.testing.assert_close(ordinary(*args), expected, rtol=0, atol=0)
    # FP64 products avoid TF32 reference variability. Source FP32 rounding and
    # reduced-precision casts remain at their original contraction boundaries.
    torch.testing.assert_close(actual, _reference(kind, args), rtol=3e-3, atol=3e-5)
    torch.testing.assert_close(args, saved, rtol=0, atol=0)
