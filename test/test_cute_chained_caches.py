from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_pointwise import _pointwise_code
from test.test_cute_chained_scan_export import _scan_export
from test.test_cute_chained_scan_export import _scan_export_args
from test.test_cute_chained_scan_export import _scan_export_config
from test.test_cute_chained_scan_export import _scan_export_cpu_codegen
from test.test_cute_chained_tcgen05 import _tcgen_chain
from test.test_cute_chained_tcgen05 import _tcgen_compile
from test.test_cute_chained_tcgen05 import _tcgen_inputs
from test.test_cute_chained_tcgen05 import _without_early_release_seed

import helion
from helion import exc
from helion._compiler.cute.chained_aux_cache import make_late_auxiliary_cache
from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
from helion._compiler.cute.chained_matmul import _Expression
from helion._compiler.cute.chained_matmul import _ReadAccess
from helion._compiler.cute.chained_pointwise_cache import PointwiseReadCache
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_spec import EnumFragment
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


# Aux cache.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _auxiliary_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batch, rows, reduction = a.shape
    q, columns = b.shape[1], v.shape[2]
    out = torch.empty((batch, rows, columns), dtype=a.dtype, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk, qq = hl.arange(reduction), hl.arange(q)
        first = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        if mode == "reverse":
            factor = delta[bi, q - 1 - qq].float()
        else:
            factor = delta[bi, qq].float()
        weights = (
            first * torch.exp(factor)[None, :] * weight[bi, qq][None, :].float()
        ).to(a.dtype)
        result = hl.dot(weights, v[bi, qq, col])
        if mode == "not_last":
            result += hl.dot(a[bi, row, kk], v[bi, kk, col])
        result += delta[bi, delta.shape[1] - 1].float()
        out[bi, row, col] = result.to(a.dtype)
    return out


def _aux_cache_args(
    device: Any, mode: str, dtype: torch.dtype = torch.bfloat16
) -> tuple[Any, ...]:
    q = 100 if mode == "tail" else 128
    generator = torch.Generator(device=device).manual_seed(967)
    return (
        torch.randn((2, 128, 128), device=device, dtype=dtype, generator=generator)
        * 0.03,
        torch.randn((2, q, 128), device=device, dtype=dtype, generator=generator)
        * 0.03,
        torch.randn((2, 128, 64), device=device, dtype=dtype, generator=generator)
        * 0.03,
        torch.randn((2, 128), device=device, dtype=torch.float32, generator=generator)
        * 0.1,
        torch.randn((2, 128), device=device, dtype=dtype, generator=generator) * 0.1,
        mode,
    )


def _aux_cache_config(enabled: bool = True) -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_auxiliary_cache=enabled,
    )


def _aux_cache_code(
    args: tuple[Any, ...], enabled: bool = True, kernel: Any = _auxiliary_chain
) -> str:
    # Reuse the central CPU hardware guard; only replace the selected config.
    with patch(
        "test.test_cute_chained_pointwise._pointwise_config",
        return_value=_aux_cache_config(enabled),
    ):
        return _pointwise_code(args, kernel=kernel)


@pytest.mark.parametrize("mode", ["plain", "reverse", "tail", "not_last"])
def test_auxiliary_cache_codegen(mode: str) -> None:
    source = _aux_cache_code(_aux_cache_args("cpu", mode))
    assert ("chain_late_aux_0 =" in source) == (mode != "not_last")
    if mode != "not_last":
        assert "cute.recast_ptr(chain_a_workspace + 0, dtype=cutlass.Float32)" in source
        assert (
            "cute.recast_ptr(chain_a_workspace + 256, dtype=cutlass.BFloat16)" in source
        )
        fill = source.index("chain_late_aux_0 =")
        bridge = source.index("chain_1_bridge_layout =")
        assert source.rfind("cute.arch.sync_threads()", 0, fill) > 0
        assert source.index("cute.arch.sync_threads()", fill) < bridge
        assert "cutlass.Float32(chain_late_aux_0[" in source[bridge:]
    if mode == "tail":
        # Scalar index127 must not use a vector whose logical extent is100.
        assert "127" in source and "< 100" in source


def test_auxiliary_cache_disabled_is_unchanged() -> None:
    assert "chain_late_aux_0 =" not in _aux_cache_code(
        _aux_cache_args("cpu", "plain"), False
    )


def test_auxiliary_cache_has_no_new_shared_allocation() -> None:
    args = _aux_cache_args("cpu", "plain")
    allocations = [
        [
            line
            for line in _aux_cache_code(args, enabled).splitlines()
            if "alloc_smem(" in line
        ]
        for enabled in (False, True)
    ]
    assert allocations[0] == allocations[1]


def test_auxiliary_cache_respects_borrowed_capacity() -> None:
    def bounded(*args: Any, **kwargs: Any) -> Any:
        kwargs["arena_bytes"] = 128
        return make_late_auxiliary_cache(*args, **kwargs)

    with patch(
        "helion._compiler.cute.chained_tcgen05.make_late_auxiliary_cache",
        side_effect=bounded,
    ):
        assert "chain_late_aux_0 =" not in _aux_cache_code(
            _aux_cache_args("cpu", "plain")
        )


def test_auxiliary_cache_reuses_existing_scan_leaf() -> None:
    args = (*_tcgen_inputs("cpu"), "scan")
    source = _aux_cache_code(args, kernel=_tcgen_chain)
    assert "chain_scan_0_input_0" in source
    assert "chain_late_aux_0 =" not in source


@pytest.mark.parametrize("mode", ["plain", "reverse", "tail"])
def test_auxiliary_cache_real_cpu_compile(mode: str) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUTE_DSL_KEEP": "ptx",
    }
    result = subprocess.run(
        [sys.executable, "-m", __name__, "aux_cache", mode],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["plain", "reverse", "tail", "not_last"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_auxiliary_cache_runtime(mode: str, dtype: torch.dtype) -> None:
    args = _aux_cache_args(DEVICE, mode, dtype)
    a, b, v, delta, weight, _ = args
    frozen = tuple(value.clone() for value in args[:-1])
    q = b.shape[1]
    factor = delta[:, :q].flip(-1) if mode == "reverse" else delta[:, :q]
    weights = (
        (a.float() @ b.float().transpose(-2, -1))
        * factor.exp()[:, None, :]
        * weight[:, None, :q].float()
    ).to(dtype)
    expected = weights.float() @ v[:, :q].float()
    if mode == "not_last":
        expected += a.float() @ v.float()
    expected = (expected + delta[:, -1, None, None]).to(dtype)
    run = _auxiliary_chain._bind_isolated(args).compile_config(_aux_cache_config())
    actual = run(*args)
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.015)
    repeated = run(*args)
    assert actual.data_ptr() != repeated.data_ptr()
    torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
    for before, value in zip(frozen, args[:-1], strict=True):
        torch.testing.assert_close(before, value, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auxiliary_cache_runtime_strides_offsets_and_readonly_aliases() -> None:
    args = _aux_cache_args(DEVICE, "plain")
    run = _auxiliary_chain._bind_isolated(args).compile_config(_aux_cache_config())
    a, b, v, delta, weight, mode = args
    for offset in (0, 1):
        storage = torch.randn((2, 257), device=DEVICE, dtype=delta.dtype) * 0.1
        changed_delta = storage[:, offset : offset + 256 : 2]
        weight_storage = torch.randn((257,), device=DEVICE, dtype=weight.dtype) * 0.1
        changed_weight = weight_storage[offset : offset + 256].view(2, 128)
        changed = (a, b, v, changed_delta, changed_weight, mode)
        weights = (
            (a.float() @ b.float().transpose(-2, -1))
            * changed_delta.exp()[:, None, :]
            * changed_weight[:, None, :].float()
        ).to(a.dtype)
        expected = (weights.float() @ v.float() + changed_delta[:, -1, None, None]).to(
            a.dtype
        )
        torch.testing.assert_close(run(*changed), expected, atol=0.003, rtol=0.015)
    aliased = (*args[:3], weight, weight, mode)
    alias_run = _auxiliary_chain._bind_isolated(aliased).compile_config(
        _aux_cache_config()
    )
    before = weight.clone()
    weights = (
        (a.float() @ b.float().transpose(-2, -1))
        * weight.float().exp()[:, None, :]
        * weight[:, None, :].float()
    ).to(a.dtype)
    expected = (weights.float() @ v.float() + weight[:, -1, None, None].float()).to(
        a.dtype
    )
    torch.testing.assert_close(alias_run(*aliased), expected, atol=0.003, rtol=0.015)
    torch.testing.assert_close(weight, before, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _auxiliary_prior_dot_index(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    batch, rows, reduction = a.shape
    q, columns = b.shape[1], v.shape[2]
    out = torch.empty((batch, rows, columns), dtype=a.dtype, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        bi = bt.begin
        kk, qq = hl.arange(reduction), hl.arange(q)
        prior = hl.dot(a[bi, row, kk], b[bi, qq, kk].T)
        first = hl.dot((a[bi, row, kk] + 0.1).to(a.dtype), b[bi, qq, kk].T)
        result = hl.dot(first.to(a.dtype), v[bi, qq, col])
        index = prior[0, :][col.index].to(torch.int64) % delta.shape[1]
        result += delta[bi, index][None, :].float()
        out[bi, row, col] = result.to(a.dtype)
    return out


def test_auxiliary_cache_prior_dot_index_falls_back() -> None:
    args = _aux_cache_args("cpu", "plain")
    disabled = _aux_cache_code(args, False, _auxiliary_prior_dot_index)
    enabled = _aux_cache_code(args, True, _auxiliary_prior_dot_index)
    assert "OperandSource.TMEM" in disabled
    assert "OperandSource.TMEM" in enabled
    assert "chain_late_aux_0 =" not in enabled


def test_auxiliary_cache_padded_scalar_query_uses_global_fallback() -> None:
    source = _aux_cache_code(_aux_cache_args("cpu", "tail"))
    reads = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.IfExp)
        and "chain_late_aux_0[127]" in ast.unparse(node.body)
    ]
    assert reads
    for read in reads:
        assert ".load()" in ast.unparse(read.orelse)
        assert not eval(
            compile(ast.Expression(read.test), "<aux-cache-domain>", "eval"),
            {},
            {"cutlass": SimpleNamespace(Int32=int, Int64=int)},
        )


@pytest.mark.parametrize("capacity, count", [(511, 1), (512, 1), (767, 1), (768, 2)])
def test_auxiliary_cache_exact_capacity_and_typed_offsets(
    capacity: int, count: int
) -> None:
    def bounded(*args: Any, **kwargs: Any) -> Any:
        kwargs["arena_bytes"] = capacity
        return make_late_auxiliary_cache(*args, **kwargs)

    with patch(
        "helion._compiler.cute.chained_tcgen05.make_late_auxiliary_cache",
        side_effect=bounded,
    ):
        source = _aux_cache_code(_aux_cache_args("cpu", "plain"))
    assert source.count("= cute.make_tensor(cute.recast_ptr(chain_a_workspace") == count
    if capacity == 511:
        # The 512-byte FP32 candidate does not fit; the 256-byte BF16 one does.
        assert "chain_a_workspace + 0, dtype=cutlass.BFloat16" in source
    else:
        assert "chain_a_workspace + 0, dtype=cutlass.Float32" in source
    if count == 2:
        assert "chain_a_workspace + 256, dtype=cutlass.BFloat16" in source


# Early aux cache.


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


def _early_aux_cache_config(
    enabled: bool = True, vectorize: bool = True
) -> helion.Config:
    return helion.Config(
        block_sizes=[128, 64],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_pointwise_vectorize=vectorize,
        cute_chained_auxiliary_cache=enabled,
    )


def _early_aux_cache_args(
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


def _early_aux_cache_code(
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
        return kernel._bind_isolated(args).to_code(
            _early_aux_cache_config(enabled, vectorize)
        )


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
    code = _early_aux_cache_code(_early_aux_cache_args(kind, dtype))
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
        assert not _cache_allocations(
            _early_aux_cache_code(_early_aux_cache_args(), False)
        )


def test_early_aux_budget_falls_back_without_allocation() -> None:
    with patch(
        "helion._compiler.cute.chained_tcgen05._shared_memory_bytes",
        return_value=232448,
    ):
        assert not _cache_allocations(_early_aux_cache_code(_early_aux_cache_args()))


def test_early_aux_scalar_staging_uses_same_cache() -> None:
    code = _early_aux_cache_code(_early_aux_cache_args(), vectorize=False)
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
    code = _early_aux_cache_code(args, kernel=_two_coordinate_maps)
    assert len(_cache_allocations(code)) == (1 if rows == 128 else 2)


def test_early_aux_publication_precedes_consumers() -> None:
    tree = ast.parse(_early_aux_cache_code(_early_aux_cache_args()))
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
    scale = _early_aux_cache_args("offset", dtype, aligned_offset=True)[2]
    assert scale.storage_offset() > 0
    assert scale.data_ptr() % 16 == 0


def test_early_aux_rejects_external_output_alias() -> None:
    a, b, scale, _ = _early_aux_cache_args()
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        pytest.raises(helion.exc.InvalidConfig, match="contraction DAG"),
    ):
        _external_output._bind_isolated((a, b, scale, a[:, :64])).to_code(
            _early_aux_cache_config()
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "kind", ["dense", "offset", "tail", "alias", "stride", "short"]
)
def test_early_aux_runtime(kind: str) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05")
    args = _early_aux_cache_args(kind, device=DEVICE, aligned_offset=True)
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
    run = _coefficient_dot._bind_isolated(args).compile_config(
        _early_aux_cache_config()
    )
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


# Pointwise cache.

POINTWISE_CACHE_KEY = "cute_chained_pointwise_read_cache"


def _pointwise_cache_code(
    value: object = False, *, include: bool = True, kind: str = "dense"
) -> str:
    config = _scan_export_config().config | {"cute_chained_pointwise_vectorize": True}
    if include:
        config[POINTWISE_CACHE_KEY] = value
    with _scan_export_cpu_codegen():
        return _scan_export._bind_isolated(
            (*_scan_export_args(kind=kind), "normal")
        ).to_code(helion.Config.from_dict(config))


def test_default_source_is_exact() -> None:
    assert _pointwise_cache_code() == _pointwise_cache_code(include=False)


@pytest.mark.parametrize("kind", ["dense", "offset", "stride"])
def test_scan_cache_is_typed_guarded_and_after_publication(kind: str) -> None:
    source = _pointwise_cache_code(True, kind=kind)
    assert "read_cache_" in source
    tree = ast.parse(source)
    declarations = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and "read_cache_" in n.targets[0].id
    ]
    assert len(declarations) == 1
    assert ast.unparse(declarations[0].value).endswith("cutlass.Float32)")
    cache_loop = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id.endswith("_cache_element")
    )
    preload = ast.unparse(cache_loop)
    assert "chain_scan_0_values" in preload and "< 128" in preload
    assert "pointwise_row" not in preload
    branch = next(
        n for n in ast.walk(tree) if isinstance(n, ast.If) and cache_loop in n.body
    )
    assert "toint() % 16 == 0" in ast.unparse(branch.test)
    assert "_read_cache_" not in ast.unparse(
        ast.Module(body=branch.orelse, type_ignores=[])
    )
    assert source.index("cute.arch.sync_threads()") < source.index("read_cache_")
    old = ast.parse(_pointwise_cache_code(False, kind=kind))
    assert ast.dump(tree.body[-1]) == ast.dump(old.body[-1])
    for term in ("cute.gemm", "cute.arch.sync_threads", "cute.arch.mbarrier_wait"):

        def calls(module: ast.AST, term: str) -> list[str]:
            return [
                ast.dump(n)
                for n in ast.walk(module)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == term
            ]

        assert calls(tree, term) == calls(old, term)


@pytest.mark.parametrize("value", [0, 1, 2, "true", None])
def test_invalid_cache_values(value: object) -> None:
    with pytest.raises(exc.InvalidConfig, match="must be bool"):
        _pointwise_cache_code(value)


def _expression(rhs: str, dtype: torch.dtype = torch.bfloat16) -> _Expression:
    graph = torch.fx.Graph()
    source = graph.placeholder("src")
    source.meta["val"] = torch.empty(1, dtype=dtype)
    load = graph.call_function(torch.clone, (source,))
    # This unit fixture supplies the real typed expression catalog consumed by
    # prepare; it does not evaluate FX nodes or require a GenerateAST instance.
    expression = _Expression.__new__(_Expression)
    expression.plan = ChainedMatmulPlan(
        root_graph_id=0,
        dots=(),
        store=graph.output(load),
        axes=(),
        shapes=(),
        dtype=dtype,
        threads=128,
    )
    expression.statements = []
    expression.definitions = {}
    expression.definition_inputs = {}
    expression.loaded_inputs = [(load, (), [], "loaded")]
    expression.memo = {}
    binding = expression.record_statement(ast.parse(f"loaded = {rhs}").body[0])
    expression.reads = {
        "loaded": _ReadAccess(ast.parse(rhs, mode="eval").body, dtype, binding)
    }
    expression.record_statement(
        ast.parse("result = cutlass.Float32(loaded) * row").body[0]
    )
    return expression


@pytest.mark.parametrize(
    "rhs",
    [
        "src[col] if row < 8 else cutlass.BFloat16(0)",
        "src[col + row]",
        "src[col] if col < limit else cutlass.BFloat16(0)",
        "src[row]",
        "src[0]",
    ],
)
def test_complete_rhs_and_temporary_dependency_rejected(rhs: str) -> None:
    expression = _expression(rhs)
    expression.definitions["limit"] = "row + 1"
    old = list(expression.lines)
    cache = PointwiseReadCache(True)
    assert cache.prepare(expression, ("row", "col"), "e", "stage", 16, 16) == []
    assert expression.lines == old
    with pytest.raises(exc.BackendUnsupported, match="row-invariant"):
        cache.validate()


@pytest.mark.parametrize(
    "dtype,name",
    [
        (torch.bfloat16, "cutlass.BFloat16"),
        (torch.float16, "cutlass.Float16"),
        (torch.float32, "cutlass.Float32"),
    ],
)
def test_native_load_dtype_and_complete_rhs_preserved(
    dtype: torch.dtype, name: str
) -> None:
    rhs = f"src[col] if 0 <= col < 127 else {name}(0)"
    expression = _expression(rhs, dtype)
    cache = PointwiseReadCache(True)
    with _scan_export_cpu_codegen():
        # Use a real environment's dtype rendering while keeping this test of
        # planner rejection independent of a particular contraction kernel.
        bound = _scan_export._bind_isolated((*_scan_export_args(), "normal"))
        with bound.env:
            lines = cache.prepare(expression, ("row", "col"), "e", "stage", 16, 16)
    assert lines[0].endswith(f"{name})")
    assert (
        lines[-1]
        == f"    stage_read_cache_0[stage_cache_element] = src[stage_cache_col] if 0 <= stage_cache_col < 127 else {name}(0)"
    )
    assert expression.lines == [
        "loaded = stage_read_cache_0[e]",
        "result = cutlass.Float32(loaded) * row",
    ]
    cache.validate()


def test_single_trip_and_disabled_leave_every_byte_alone() -> None:
    for enabled, trips in ((False, 16), (True, 1)):
        expression = _expression("src[col]")
        old = list(expression.lines)
        assert (
            PointwiseReadCache(enabled).prepare(
                expression, ("row", "col"), "e", "stage", 16, trips
            )
            == []
        )
        assert expression.lines == old


def test_activation_is_private() -> None:
    cache = PointwiseReadCache(True)
    PointwiseReadCache(False).validate()
    with pytest.raises(exc.BackendUnsupported):
        cache.validate()


def test_search_knob_default_and_seed_siblings() -> None:
    with _scan_export_cpu_codegen():
        bound = _scan_export._bind_isolated((*_scan_export_args(), "normal"))
    spec = bound.config_spec
    assert spec.cute_chained_pointwise_read_cache_search_enabled
    fragment = spec._flat_fields()[POINTWISE_CACHE_KEY]
    assert isinstance(fragment, EnumFragment)
    assert fragment.search_values() == [False, True]
    seeds = _without_early_release_seed(spec.compiler_seed_configs, spec)
    assert seeds and not seeds[0].config.get(POINTWISE_CACHE_KEY, False)
    cached = [seed for seed in seeds if seed.config.get(POINTWISE_CACHE_KEY)]
    assert cached
    for seed in cached:
        assert seed.config["cute_chained_mma_schedule"] == "tcgen05_tmem"
        assert seed.config["cute_chained_pointwise_vectorize"]
        parent = {k: v for k, v in seed.config.items() if k != POINTWISE_CACHE_KEY}
        assert any(
            parent
            == {k: v for k, v in other.config.items() if k != POINTWISE_CACHE_KEY}
            for other in seeds
            if not other.config.get(POINTWISE_CACHE_KEY, False)
        )


@pytest.mark.parametrize(
    "rhs",
    [
        "src[col + row]",
        "src[col] if row < 127 else cutlass.Float32(0)",
        "src[col] if col < limit else cutlass.Float32(0)",
        "src[row]",
        "src[0]",
    ],
)
def test_vector_complete_rhs_rejects_unproved_dependence(rhs: str) -> None:
    expression = _expression(rhs, torch.float32)
    node, _, indices, value = expression.loaded_inputs[0]
    node.meta["val"] = torch.empty(128)
    expression.loaded_inputs = [(node, ("col",), indices, value)]
    expression.definitions["limit"] = "row + 1"
    old = list(expression.lines)
    assert not PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)
    assert expression.lines == old


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.int32])
def test_vector_transport_does_not_widen_dtype_family(dtype: torch.dtype) -> None:
    expression = _expression("src[col]", dtype)
    node, _, indices, value = expression.loaded_inputs[0]
    node.meta["val"] = torch.empty(128, dtype=dtype)
    expression.loaded_inputs = [(node, ("col",), indices, value)]
    assert not PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)


def test_vector_read_catalog_disabled_trip_rank_and_capacity() -> None:
    expression = _expression("src[col]", torch.float32)
    node, _, indices, value = expression.loaded_inputs[0]
    node.meta["val"] = torch.empty(128)
    expression.loaded_inputs = [(node, ("col",), indices, value)]
    assert PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)
    for cache, trips in ((False, 8), (True, 1)):
        assert not PointwiseReadCache(cache).vector_reads(
            expression, ("row", "col"), trips
        )
    expression.loaded_inputs = [(node, ("row", "col"), indices, value)]
    assert not PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)
    expression.loaded_inputs = [
        (node, (f"col + {i}",), indices, value) for i in range(5)
    ]
    assert not PointwiseReadCache(True).vector_reads(expression, ("row", "col"), 8)


def test_copy_coordinate_and_fp32_bit_identity() -> None:
    bits = torch.tensor(
        [0, -2147483648, 1, 2139095040, -8388608, 2143289345, 2143289346, 1065353216],
        dtype=torch.int32,
    ).repeat(16)
    for half in range(2):
        for thread in range(128):
            cols = half * 64 + thread % 8 * 8 + torch.arange(8)
            retained = bits[cols].view(torch.float32).clone().view(torch.int32)
            for step in range(8):
                row = thread // 8 + step * 16
                assert 0 <= row < 128
                assert torch.equal(retained, bits[cols])


@pytest.mark.parametrize("width", [64, 128])
def test_actual_static_fp32_partition_has_identical_row_addresses(width: int) -> None:
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            columns, rows = width // 8, 128 // (width // 8)
            tiled_copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.Float32,
                    num_bits_per_copy=128,
                ),
                cute.make_layout((rows, columns), stride=(columns, 1)),
                cute.make_layout((1, 8)),
            )
            physical = cute.make_layout((128, 128), stride=(0, 1))
            for half in range(128 // width):
                tile = cute.local_tile(
                    cute.make_identity_tensor((128, 128)), (128, width), (0, half)
                )
                for thread in range(128):
                    coords = tiled_copy.get_slice(thread).partition_S(tile)
                    assert int(cute.size(coords[None, 0, 0])) == 8
                    for element in range(8):
                        first = tuple(map(int, coords[element, 0, 0]))
                        for step in range(128 // rows):
                            current = tuple(map(int, coords[element, step, 0]))
                            assert current == (
                                thread // columns + step * rows,
                                half * width + thread % columns * 8 + element,
                            )
                            assert int(physical(current)) == int(physical(first))
        assert module.operation.verify()
    assert torch.cuda.is_initialized() == before


if __name__ == "__main__":
    command = sys.argv.pop(1)
    if command == "aux_cache":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert not torch.cuda.is_initialized()
        args = _aux_cache_args("cpu", sys.argv[1])
        with patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
        ):
            ptx = _tcgen_compile(
                _aux_cache_code(args), args, None, entry="_auxiliary_chain"
            )
        assert "tcgen05.mma" in ptx
    else:
        raise AssertionError(f"Unknown test driver: {command}")
