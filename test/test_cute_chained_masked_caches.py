from __future__ import annotations

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _auxiliary_masked_read(a, b, v, delta, masked: hl.constexpr, hint: str | None):
    m, k = a.shape
    n = v.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for rows, cols in hl.tile([m, n]):
        kk = hl.arange(k)
        first = hl.dot(a[rows, kk], b[kk, :], out_dtype=torch.float32)
        if masked:
            factor = hl.load(delta, [kk], extra_mask=kk % 2 == 0, eviction_policy=hint)
        else:
            factor = hl.load(delta, [kk], eviction_policy=hint)
        second = hl.dot(
            (first * torch.exp(factor)[None, :]).to(a.dtype),
            v[kk, cols],
            out_dtype=torch.float32,
        )
        output[rows, cols] = second + delta[k - 1]
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _scan_masked_read(a, b, delta, masked: hl.constexpr, hint: str | None):
    m, k = a.shape
    n = b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    for rows, cols in hl.tile([m, n]):
        kk = hl.arange(k)
        if masked:
            factor = hl.load(delta, [kk], extra_mask=kk % 2 == 0, eviction_policy=hint)
        else:
            factor = hl.load(delta, [kk], eviction_policy=hint)
        prefix = hl.cumsum(factor, 0)
        result = hl.dot(
            (a[rows, kk].float() * torch.exp(prefix)[None, :]).to(a.dtype),
            b[kk, cols],
            out_dtype=torch.float32,
        )
        output[rows, cols] = result + delta[cols][None, :]
    return output


def _inputs(device: str | torch.device, scan: bool) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(1723)
    matrices = ((128, 128), (128, 64)) if scan else ((128, 128), (128, 128), (128, 64))
    return (
        *(
            torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
            * 0.05
            for shape in matrices
        ),
        torch.linspace(0.001, 0.01, 128, dtype=torch.float32, device=device),
    )


def _config(scan: bool, enabled: bool = True) -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64],
        num_warps=4,
        cute_chained_mma_schedule=(
            "cp_async_register_reuse_scan" if scan else "tcgen05_tmem"
        ),
        cute_chained_auxiliary_cache=enabled if not scan else False,
    )


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("masked,hint", [(False, None), (True, None), (False, "last")])
def test_masked_and_hinted_vectors_are_not_raw_cached(
    scan: bool, masked: bool, hint: str | None
) -> None:
    kernel = _scan_masked_read if scan else _auxiliary_masked_read
    with _cpu_codegen():
        bound = kernel._bind_isolated((*_inputs("cpu", scan), masked, hint))
        source = bound.to_code(_config(scan))
    assert "chain_0_mma" in source
    marker = "chain_scan_0_input_0 =" if scan else "chain_late_aux_0 ="
    assert (marker in source) == (not masked and hint is None)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("scan", [False, True])
def test_masked_cache_does_not_change_later_unmasked_reads(scan: bool) -> None:
    args = _inputs(DEVICE, scan)
    kernel = _scan_masked_read if scan else _auxiliary_masked_read
    bound = kernel._bind_isolated((*args, True, None))
    compiled = bound.compile_config(_config(scan))
    actual = compiled(*args, True, None)
    delta = args[-1]
    masked = torch.where(torch.arange(128, device=DEVICE) % 2 == 0, delta, 0.0)
    if scan:
        a, b, _ = args
        left = (a.float() * torch.exp(masked.cumsum(0))[None, :]).to(a.dtype)
        expected = left.float() @ b.float() + delta[:64][None, :]
    else:
        a, b, v, _ = args
        first = a.float() @ b.float()
        second = (first * torch.exp(masked)[None, :]).to(a.dtype)
        expected = second.float() @ v.float() + delta[-1]
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("pipeline", ["startup", "paired_leaf"])
@pytest.mark.parametrize("masked,hint", [(False, None), (True, None), (False, "last")])
def test_raw_transfer_seed_requires_an_unmasked_unhinted_leaf(
    pipeline: str, masked: bool, hint: str | None
) -> None:
    from torch.fx import Graph

    from .test_cute_chained_group_guards import _convert
    from .test_cute_chained_group_guards import _dot
    from .test_cute_chained_group_guards import _input
    from helion._compiler.cute.chained_leaf_pipeline import has_leaf_candidate
    from helion._compiler.cute.chained_startup import has_startup_leaf
    from helion._compiler.device_ir import RootGraphInfo
    from helion.language import _tracing_ops
    from helion.language import memory_ops

    graph = Graph()
    dtype = torch.bfloat16 if pipeline == "startup" else torch.float32
    source = graph.call_function(_tracing_ops._host_tensor, ("raw",))
    source.meta["val"] = torch.empty((128, 128), dtype=dtype)
    mask = _input(graph, "mask", (128, 128), torch.bool) if masked else None
    leaf = graph.call_function(
        memory_ops.load, (source, [slice(None), slice(None)], mask, hint)
    )
    leaf.meta["val"] = torch.empty((128, 128), dtype=dtype)
    operand = _convert(graph, leaf, torch.bfloat16)
    rhs = _input(graph, "rhs", (128, 128), torch.bfloat16)
    first = _dot(graph, operand, rhs)
    outputs = (first,)
    if pipeline == "paired_leaf":
        outputs = (first, _dot(graph, operand, rhs))
    graph.output(outputs)
    graph.lint()
    predicate = has_startup_leaf if pipeline == "startup" else has_leaf_candidate
    assert predicate([RootGraphInfo(0, graph)]) == (not masked and hint is None)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _startup_masked_read(a, b, masked: hl.constexpr, hint: str | None):
    m, k = a.shape
    n = b.shape[1]
    output = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows, cols in hl.tile([m, n]):
        kk = hl.arange(k)
        if masked:
            raw = hl.load(
                a, [rows, kk], extra_mask=(kk % 2 == 0)[None, :], eviction_policy=hint
            )
        else:
            raw = hl.load(a, [rows, kk], eviction_policy=hint)
        output[rows, cols] = hl.dot(
            (raw.float() * 0.25).to(a.dtype), b[kk, cols], out_dtype=torch.float32
        )
    return output


@pytest.mark.parametrize("masked,hint", [(False, None), (True, None), (False, "last")])
def test_startup_transfer_preserves_masked_or_hinted_original_load(
    masked: bool, hint: str | None
) -> None:
    from .test_cute_chained_pipeline import plans

    with _cpu_codegen():
        bound = _startup_masked_read._bind_isolated(
            (
                torch.empty((128, 128), dtype=torch.bfloat16),
                torch.empty((128, 64), dtype=torch.bfloat16),
                masked,
                hint,
            )
        )
        source = bound.to_code(
            helion.Config(
                block_sizes=[128, 64],
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
                cute_chained_startup_transfer="tma",
            )
        )
    descriptors = plans(source)
    assert len(descriptors) == (2 if not masked and hint is None else 1)
    assert ("chain_start_a_index" in source) == (not masked and hint is None)
    assert "chain_start_b_shared" in source
