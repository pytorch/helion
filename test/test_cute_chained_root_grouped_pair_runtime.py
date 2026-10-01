from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_plain_root_runtime import _bits_equal
from .test_cute_chained_root_weighted_pair_runtime import _reference
from .test_cute_chained_vector_group_integration import _config
from .test_cute_chained_vector_group_integration import _root_shared_gram
from helion._compiler.cute import chained_plain_root as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

_CASES = (
    (16, 0, False, 1),
    (32, 0, False, 1),
    (64, 0, False, 1),
    (128, 0, False, 1),
    (32, 1, False, 1),
    (32, 0, True, 1),
)


def _fixture(device, dtype, reduction, mask, early_release, unroll):
    generator = torch.Generator(device=device).manual_seed(65747)
    values = (
        torch.randn((256, reduction), device=device, dtype=dtype, generator=generator)
        * 0.0625,
        torch.randn((128, 256), device=device, dtype=dtype, generator=generator)
        * 0.0625,
        mask,
    )
    config = _config("root", True)
    config.config.update(
        cute_chained_tmem_early_release=early_release,
        cute_chained_pointwise_unroll=unroll,
    )
    return values, config


@pytest.mark.parametrize("reduction,mask,early_release,unroll", _CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_grouped_root_pair_preserves_original_source_cpu(
    reduction, mask, early_release, unroll, dtype
):
    initialized = torch.cuda.is_initialized()
    values, config = _fixture("cpu", dtype, reduction, mask, early_release, unroll)
    with _cpu_codegen(), patch.object(roots, "codegen_plain_root", return_value=False):
        before = _root_shared_gram._bind_isolated(values).to_code(config)
    with (
        _cpu_codegen(),
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = _root_shared_gram._bind_isolated(values).to_code(config)
    assert before == after
    assert "chain_0_vector_group_output_1_copy" in after
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_grouped_root_pair_unported_unroll_keeps_original_route_cpu(dtype):
    values, config = _fixture("cpu", dtype, 32, 0, False, 2)
    with _cpu_codegen(), patch.object(roots, "codegen_plain_root", return_value=False):
        before = _root_shared_gram._bind_isolated(values).to_code(config)
    with (
        _cpu_codegen(),
        patch.object(
            legacy, "codegen_chained_tcgen05", wraps=legacy.codegen_chained_tcgen05
        ) as old_route,
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        after = _root_shared_gram._bind_isolated(values).to_code(config)
    assert after == before
    assert old_route.call_count == 1 and emitted.call_count == 0
    assert "cutlass.range(4, unroll=2)" in after


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reduction,mask,early_release,unroll", _CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_grouped_root_pair_original_bits_fp64_fallbacks_and_graphs_gpu(
    reduction, mask, early_release, unroll, dtype
):
    canonical, config = _fixture(DEVICE, dtype, reduction, mask, early_release, unroll)
    with patch.object(roots, "codegen_plain_root", return_value=False):
        original = _root_shared_gram._bind_isolated(canonical).compile_config(config)
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old root")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        shared = _root_shared_gram._bind_isolated(canonical).compile_config(config)
    assert [call.args[3] for call in emitted.call_args_list] == [0, 1]
    for layout in ("contiguous", "misaligned", "strided"):
        for generation in range(2):
            values = []
            for value in canonical[:2]:
                if layout == "contiguous":
                    tensor = value.clone()
                elif layout == "misaligned":
                    backing = torch.empty(value.numel() + 1, device=DEVICE, dtype=dtype)
                    tensor = backing[1:].view(value.shape)
                    tensor.copy_(value)
                    assert tensor.data_ptr() % 16 != 0
                else:
                    backing = torch.empty(
                        (value.shape[0], value.shape[1] * 2), device=DEVICE, dtype=dtype
                    )
                    tensor = backing[:, ::2]
                    tensor.copy_(value)
                    assert tensor.stride(1) == 2
                if generation:
                    tensor.mul_(0.5)
                values.append(tensor)
            saved = tuple(value.clone() for value in values)
            arguments = (*values, mask)
            actual = shared(*arguments)
            _bits_equal(actual, original(*arguments))
            torch.testing.assert_close(
                actual, _reference(arguments, "gram"), rtol=2e-3, atol=2e-3
            )
            for _ in range(3):
                _bits_equal(shared(*arguments), actual)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = shared(*arguments)
            for _ in range(3):
                captured.fill_(float("nan"))
                graph.replay()
                _bits_equal(captured, actual)
            for value, before in zip(values, saved, strict=True):
                _bits_equal(value, before)
