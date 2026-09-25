from __future__ import annotations

import ast
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test.test_cute_chained_pointwise import _pointwise_code
from test.test_cute_chained_tcgen05 import _tcgen_chain
from test.test_cute_chained_tcgen05 import _tcgen_compile
from test.test_cute_chained_tcgen05 import _tcgen_inputs

import helion
from helion._compiler.cute.chained_aux_cache import make_late_auxiliary_cache
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
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
