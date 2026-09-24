from __future__ import annotations

import ast
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _coefficient_dot(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    rows, reduction = a.shape
    columns = b.shape[1]
    out = torch.empty((rows, columns), dtype=torch.float32, device=a.device)
    for row, col in hl.tile([rows, columns]):
        kk = hl.arange(reduction)
        factor = (
            torch.exp((scale[reduction - 1].float() - scale[kk].float()).clamp(max=0))
            * weight[kk].float()
            + 1.0
        )
        left = (a[row, kk].float() * factor[None, :]).to(a.dtype)
        out[row, col] = hl.dot(left, b[kk, col])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _external_output(
    a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for row, col in hl.tile([a.shape[0], b.shape[1]]):
        kk = hl.arange(a.shape[1])
        left = (a[row, kk].float() * scale[kk][None, :]).to(a.dtype)
        out[row, col] = hl.dot(left, b[kk, col])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_coordinate_maps(
    a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device)
    for row, col in hl.tile([a.shape[0], b.shape[1]]):
        kk = hl.arange(a.shape[1])
        factor = scale[row].float()[:, None] - scale[kk].float()[None, :]
        left = (a[row, kk].float() * factor).to(a.dtype)
        out[row, col] = hl.dot(left, b[kk, col])
    return out


def _config(enabled: bool = True, vectorize: bool = True) -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_pointwise_vectorize=vectorize,
        cute_chained_auxiliary_cache=enabled,
    )


def _args(
    kind: str = "dense",
    dtype: torch.dtype = torch.float32,
    device: Any = "cpu",
    *,
    aligned_offset: bool = False,
) -> tuple[torch.Tensor, ...]:
    reduction = 49 if kind == "tail" else 128
    generator = torch.Generator(device=device).manual_seed(823)
    a = (
        torch.randn(
            128, reduction, dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.1
    )
    b = (
        torch.randn(
            reduction, 64, dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.1
    )
    scale = (
        torch.randn(reduction, dtype=dtype, device=device, generator=generator) * 0.1
    )
    weight = (
        torch.randn(reduction, dtype=dtype, device=device, generator=generator) * 0.1
    )
    if kind == "offset":
        offset = 16 // scale.element_size() if aligned_offset else 1
        backing = torch.empty(reduction + offset, dtype=dtype, device=device)
        backing[offset:].copy_(scale)
        scale = backing[offset:]
    if kind == "stride":
        scale = torch.stack((scale, scale), dim=1).flatten()[::2]
        weight = torch.stack((weight, weight), dim=1).flatten()[::2]
    if kind == "alias":
        scale = a[0]
        weight = a[1]
    if kind == "short":
        scale = scale[:37]
        weight = weight[:61]
    return a, b, scale, weight


def _code(
    args: tuple[torch.Tensor, ...],
    enabled: bool = True,
    vectorize: bool = True,
    kernel: Any = _coefficient_dot,
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
        return kernel._bind_isolated(args).to_code(_config(enabled, vectorize))


def _cache_allocations(code: str) -> list[ast.Assign]:
    return [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.startswith("chain_early_aux_")
            for target in node.targets
        )
        and "alloc_smem" in ast.unparse(node.value)
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "kind", ["dense", "offset", "tail", "alias", "stride", "short"]
)
def test_early_aux_typed_codegen(kind: str, dtype: torch.dtype) -> None:
    code = _code(_args(kind, dtype))
    allocations = _cache_allocations(code)
    assert len(allocations) == (0 if kind == "stride" else 2)
    if allocations:
        actual_dtype = torch.bfloat16 if kind == "alias" else dtype
        name = {
            torch.float16: "Float16",
            torch.bfloat16: "BFloat16",
            torch.float32: "Float32",
        }[actual_dtype]
        assert all(
            f"alloc_smem(cutlass.{name}," in ast.unparse(node.value)
            for node in allocations
        )
        assert "chain_early_aux_0[" in code
        assert "chain_early_aux_1[" in code
        assert code.index("chain_early_aux_1[") < code.index("chain_0_mma =")
        assert "alignment=128" in code
    if kind == "tail":
        assert "< 49" in code  # Original masked-load/domain fallback remains.
    if kind == "short":
        assert "< 37" in code and "< 61" in code
    assert "cute.math.min(cutlass.Float32" in code
    assert "cute.math.exp2(cutlass.Float32" in code
    assert "chain_output_ptr = cute.arch.alloc_smem(cutlass.Float32" in code


def test_early_aux_disabled_does_not_call_helper() -> None:
    with patch(
        "helion._compiler.cute.chained_tcgen05.make_early_auxiliary_cache",
        side_effect=AssertionError("inactive"),
    ):
        assert not _cache_allocations(_code(_args(), False))


def test_early_aux_budget_falls_back_without_allocation() -> None:
    with patch(
        "helion._compiler.cute.chained_tcgen05._shared_memory_bytes",
        return_value=232448,
    ):
        assert not _cache_allocations(_code(_args()))


def test_early_aux_scalar_staging_uses_same_cache() -> None:
    code = _code(_args(), vectorize=False)
    assert len(_cache_allocations(code)) == 2
    assert "_pointwise_copy =" not in code
    assert "chain_early_aux_0[" in code


@pytest.mark.parametrize("rows", [128, 256])
def test_early_aux_distinct_origin_maps_are_not_deduplicated(rows: int) -> None:
    args = (
        torch.empty(rows, 128, dtype=torch.bfloat16),
        torch.empty(128, 64, dtype=torch.bfloat16),
        torch.empty(rows, dtype=torch.float32),
    )
    code = _code(args, kernel=_two_coordinate_maps)
    assert len(_cache_allocations(code)) == (1 if rows == 128 else 2)


def test_early_aux_publication_precedes_consumers() -> None:
    tree = ast.parse(_code(_args()))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    first_mma = next(
        i for i, node in enumerate(kernel.body) if "chain_0_mma =" in ast.unparse(node)
    )
    last_fill = max(
        i
        for i, node in enumerate(kernel.body)
        if isinstance(node, ast.For)
        and ast.unparse(node.target).startswith("chain_early_aux_")
    )
    assert last_fill < first_mma
    assert any(
        ast.unparse(node) == "cute.arch.sync_threads()"
        for node in kernel.body[last_fill + 1 : first_mma]
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_early_aux_runtime_offset_is_nonzero_and_aligned(dtype: torch.dtype) -> None:
    scale = _args("offset", dtype, aligned_offset=True)[2]
    assert scale.storage_offset() > 0
    assert scale.data_ptr() % 16 == 0


def test_early_aux_rejects_external_output_alias() -> None:
    a, b, scale, _ = _args()
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        pytest.raises(helion.exc.InvalidConfig, match="contraction DAG"),
    ):
        _external_output._bind_isolated((a, b, scale, a[:, :64])).to_code(_config())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "kind", ["dense", "offset", "tail", "alias", "stride", "short"]
)
def test_early_aux_runtime(kind: str) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05")
    args = _args(kind, device=DEVICE, aligned_offset=True)
    before = tuple(value.clone() for value in args)
    a, b, scale, weight = args
    scale_values = torch.zeros(a.shape[1], dtype=torch.float32, device=a.device)
    weight_values = torch.zeros_like(scale_values)
    scale_values[: scale.numel()] = scale.float()
    weight_values[: weight.numel()] = weight.float()
    factor = (
        torch.exp((scale_values[-1] - scale_values).clamp(max=0)) * weight_values + 1.0
    )
    left = (a.float() * factor[None, :]).to(a.dtype)
    expected = left.double() @ b.double()
    run = _coefficient_dot._bind_isolated(args).compile_config(_config())
    first = run(*args)
    torch.testing.assert_close(first.double(), expected, rtol=0.015, atol=0.015)
    second = run(*args)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(*args)
    for _ in range(3):
        captured.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(captured, first, rtol=0, atol=0)
    torch.testing.assert_close(args, before, rtol=0, atol=0)
