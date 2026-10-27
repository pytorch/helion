"""A host ``sum``/``mean`` of a 1-D FP32 tensor runs as one CuTe CTA with ATen's tree."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute.host_full_sum import prove_host_full_sum
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.runtime.cute.full_sum import full_sum_plan
from helion.runtime.cute.paired_sum import _compatible_aten_sum

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = skipUnlessBackends("cute")

_requires_pinned_aten_build = pytest.mark.skipif(
    not _compatible_aten_sum(), reason="host sum tails require the audited ATen build"
)


def _row_sum_total(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    total = torch.sum(out)
    return total, out


def _row_sum_mean(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    average = out.mean()
    return average, out


def _row_sum_returned_mean(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    return out.mean()


def _row_sum_returned_tuple(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    return torch.sum(out), out


def _row_sum_half(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([x.size(0)], dtype=torch.float16, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].sum(-1).to(torch.float16)
    total = torch.sum(out)
    return total, out


def _row_sum_dim(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    total = torch.sum(out, dim=0)
    return total, out


def _row_sum_input(x: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, ...]:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1) + bias[tile]
    # Any 1-D FP32 tensor is admitted, not only kernel outputs.
    offset = bias.sum()
    return out, offset


def _row_sum_shadowed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    _cute_try_full_sum = torch.sum(out)
    return _cute_try_full_sum, out


def _kernel(fn: Callable[..., Any], **settings: Any) -> Kernel:
    return helion.kernel(
        fn,
        backend="cute",
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
        **settings,
    )


def _bind(fn: Callable[..., object], *args: object) -> BoundKernel:
    return _cpu_bind(_kernel(fn, autotune_effort="full"), args)


def _host_code(bound: BoundKernel, layout: str) -> str:
    config = bound.config_spec.default_config()
    if bound.config_spec.cute_host_paired_sum_available:
        config.config["cute_host_paired_sum"] = layout
    else:
        assert layout == "off"
    module = ast.parse(bound.to_code(config))
    host = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and not node.decorator_list
    )
    return ast.unparse(host)


def _width(numel: int) -> int:
    plan = full_sum_plan(numel)
    assert plan is not None
    return plan.width


def test_plan_mirrors_aten_vectorized_block_width() -> None:
    assert full_sum_plan(127) is None
    assert (_width(128), _width(130)) == (32, 32)
    assert _width(512) == 128
    assert _width(2048) == 512
    assert _width(65536) == 512
    # 512 lanes with 256 values each: ATen accumulates across CTAs instead.
    assert full_sum_plan(512 * 256) is None
    plan = full_sum_plan(1000)
    assert plan is not None
    assert plan.factor == float((torch.tensor(1.0) / torch.tensor(1000.0)).item())


@_requires_pinned_aten_build
@pytest.mark.parametrize("layout", ["mapped", "narrow"])
@skipIfRefEager("generates code")
def test_assigned_sum_and_mean_call_the_helper_with_the_original_fallback(
    layout: str,
) -> None:
    with _mock_cuda_unavailable():
        for fn, mean, original in (
            (_row_sum_total, False, "total = torch.sum(out)"),
            (_row_sum_mean, True, "average = out.mean()"),
        ):
            bound = _bind(fn, torch.empty((256, 32), dtype=torch.bfloat16))
            assert bound.config_spec.cute_host_paired_sum_available
            assert bound.host_function is not None
            proof = prove_host_full_sum(bound.host_function)
            assert [(site.source, site.mean) for site in proof.sites] == [("out", mean)]
            code = _host_code(bound, layout)
            target = original.split(" = ")[0]
            assert (
                f"{target} = _cute_try_full_sum(out, mean={mean!r}, "
                "_launcher=_launcher)" in code
            )
            assert f"if {target} is None:\n        {original}" in code
            assert code.count(original) == 1


@_requires_pinned_aten_build
@skipIfRefEager("generates code")
def test_returned_sum_and_tuple_are_rewritten() -> None:
    with _mock_cuda_unavailable():
        bound = _bind(_row_sum_returned_mean, torch.empty((256, 32)))
        code = _host_code(bound, "mapped")
        assert "_helion_full_sum = _cute_try_full_sum(out, mean=True" in code
        assert "if _helion_full_sum is None:\n        return out.mean()" in code
        assert code.rstrip().endswith("return _helion_full_sum")
        bound = _bind(_row_sum_returned_tuple, torch.empty((256, 32)))
        code = _host_code(bound, "narrow")
        assert "_helion_full_sum = _cute_try_full_sum(out, mean=False" in code
        assert (
            "if _helion_full_sum is None:\n        return (torch.sum(out), out)" in code
        )
        assert code.rstrip().endswith("return (_helion_full_sum, out)")


@_requires_pinned_aten_build
@skipIfRefEager("generates code")
def test_kernel_inputs_are_admitted() -> None:
    with _mock_cuda_unavailable():
        bound = _bind(_row_sum_input, torch.empty((256, 32)), torch.empty((256,)))
        code = _host_code(bound, "mapped")
        assert "offset = _cute_try_full_sum(bias, mean=False" in code
        assert "if offset is None:\n        offset = bias.sum()" in code


@_requires_pinned_aten_build
@skipIfRefEager("generates code")
def test_knob_off_keeps_the_host_call_and_is_seeded() -> None:
    from helion._compiler.autotuner_heuristics import compiler_seed_configs

    with _mock_cuda_unavailable():
        bound = _bind(_row_sum_total, torch.empty((256, 32), dtype=torch.bfloat16))
        code = _host_code(bound, "off")
        assert "_cute_try_full_sum" not in code and "total = torch.sum(out)" in code
        assert bound.host_function is not None
        with bound.env:
            seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        assert {seed.config.get("cute_host_paired_sum") for seed in seeds} >= {
            "mapped",
            "narrow",
        }


@pytest.mark.parametrize(
    "fn",
    [_row_sum_half, _row_sum_dim, _row_sum_shadowed],
    ids=["fp16 source", "sum over dim", "helper name bound"],
)
@skipIfRefEager("generates code")
def test_unproven_sites_keep_the_host_call(fn: Callable[..., object]) -> None:
    with _mock_cuda_unavailable():
        bound = _bind(fn, torch.empty((256, 32)))
        assert bound.host_function is not None
        assert not prove_host_full_sum(bound.host_function).sites
        assert not bound.config_spec.cute_host_paired_sum_available
        assert "_cute_try_full_sum(" not in _host_code(bound, "off")


def test_runtime_helper_declines_unsupported_tensors() -> None:
    from helion.runtime.cute.full_sum import try_full_sum

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("no launch expected")

    for tensor in (
        torch.zeros(128),  # cpu
        torch.zeros((2, 64), device=torch.device("meta")),  # rank
    ):
        assert try_full_sum(tensor, mean=False, _launcher=forbidden) is None
    if torch.cuda.is_available():
        for tensor in (
            torch.zeros(64, device=DEVICE),  # ATen does not vectorize
            torch.zeros(129, device=DEVICE)[1:],  # misaligned head
            torch.zeros(256, device=DEVICE)[::2],  # strided
            torch.zeros(128, device=DEVICE, dtype=torch.float16),
            torch.zeros(128, device=DEVICE, requires_grad=True),
        ):
            assert try_full_sum(tensor, mean=False, _launcher=forbidden) is None


@_requires_pinned_aten_build
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "rows", [128, 129, 130, 131, 132, 255, 256, 1000, 2048, 4096, 16383, 16384, 65536]
)
@pytest.mark.parametrize("mean", [False, True])
def test_total_is_bit_identical_to_torch(rows: int, mean: bool) -> None:
    torch.manual_seed(rows)
    x = torch.randn((rows, 64), device=DEVICE, dtype=torch.bfloat16) * 1000
    fn = _row_sum_mean if mean else _row_sum_total
    probe = _kernel(fn, autotune_effort="none")
    config = probe.bind((x,)).config_spec.default_config()
    config.config["cute_host_paired_sum"] = "mapped"
    kernel = _kernel(fn, config=config)
    for _ in range(3):
        total, out = kernel(x)
        expected = out.mean() if mean else torch.sum(out)
        assert total.shape == () and total.dtype == torch.float32
        assert torch.equal(total, expected), (rows, mean, total.item(), expected.item())


@_requires_pinned_aten_build
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_returned_forms_match_torch_on_gpu() -> None:
    torch.manual_seed(5)
    x = torch.randn((1000, 64), device=DEVICE) * 1000
    for fn in (_row_sum_returned_mean, _row_sum_returned_tuple):
        config = (
            _kernel(fn, autotune_effort="none").bind((x,)).config_spec.default_config()
        )
        config.config["cute_host_paired_sum"] = "narrow"
        kernel = _kernel(fn, config=config)
        result = kernel(x)
        if isinstance(result, tuple):
            total, out = result
            assert torch.equal(total, torch.sum(out))
        else:
            assert torch.equal(result, x.sum(-1).mean())


@_requires_pinned_aten_build
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_short_tensor_runs_the_original_call() -> None:
    x = torch.randn((64, 32), device=DEVICE)
    config = (
        _kernel(_row_sum_total, autotune_effort="none")
        .bind((x,))
        .config_spec.default_config()
    )
    config.config["cute_host_paired_sum"] = "mapped"
    kernel = _kernel(_row_sum_total, config=config)
    with patch(
        "helion._compiler.cute.full_sum_runtime.full_sum_kernel",
        side_effect=AssertionError("no launch expected"),
    ):
        total, out = kernel(x)
    assert torch.equal(total, torch.sum(out))
