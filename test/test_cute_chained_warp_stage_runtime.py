from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_matmul import _offset_scaled_operand
from .test_cute_chained_plain_root_runtime import _bits_equal
import helion
from helion._compiler.cute import chained_root_warp_stage as roots
from helion._compiler.cute import chained_warp_stage as shared
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _fixture(device, dtype, ragged, warps, columns, schedule):
    generator = torch.Generator(device=device).manual_seed(95413)
    length, rows, output_columns = (251, 35, 49) if ragged else (256, 32, 64)
    args = tuple(
        torch.randn(shape, device=device, dtype=kind, generator=generator) * 0.0625
        for shape, kind in (
            ((2, length, rows), dtype),
            ((2, length, output_columns), dtype),
            ((2, 2, 128), torch.float32),
        )
    )
    config = helion.Config(
        block_sizes=[16, columns],
        num_warps=warps,
        cute_chained_mma_schedule=schedule,
    )
    return args, config


def _reference(args):
    a, b, scale = args
    chunks = []
    for chunk in range(2):
        start, stop = chunk * 128, min((chunk + 1) * 128, a.shape[1])
        weighted = (
            b[:, start:stop].float()
            * torch.exp(scale[:, chunk, : stop - start])[:, :, None]
        ).to(a.dtype)
        chunks.append(
            (a[:, start:stop].double().transpose(-1, -2) @ weighted.double()).to(
                a.dtype
            )
        )
    return torch.stack(chunks, dim=1)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("ragged", (False, True))
@pytest.mark.parametrize("warps,columns", ((2, 16), (8, 16), (8, 64)))
@pytest.mark.parametrize("schedule", ("cp_async", "cp_async_register_reuse"))
def test_warp_singleton_original_source_cpu(dtype, ragged, warps, columns, schedule):
    args, config = _fixture("cpu", dtype, ragged, warps, columns, schedule)
    with (
        _cpu_codegen(),
        patch.object(roots, "root_warp_stage_action", return_value=None),
    ):
        original = _offset_scaled_operand._bind_isolated(args).to_code(config)
    with (
        _cpu_codegen(),
        patch.object(
            shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
        ) as called,
    ):
        actual = _offset_scaled_operand._bind_isolated(args).to_code(config)
    assert actual == original
    assert called.call_count == 1
    prepared = called.call_args.args[3]
    assert prepared.completion.execution.threads == warps * 32
    assert prepared.threads == 32 * min(warps, 2 ** (columns.bit_length() - 4))


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("ragged", (False, True))
@pytest.mark.parametrize("warps,columns", ((2, 16), (8, 16), (8, 64)))
@pytest.mark.parametrize("schedule", ("cp_async", "cp_async_register_reuse"))
def test_warp_singleton_original_bits_fallbacks_and_replay_gpu(
    dtype, ragged, warps, columns, schedule
):
    args, config = _fixture(DEVICE, dtype, ragged, warps, columns, schedule)
    with patch.object(roots, "root_warp_stage_action", return_value=None):
        original = _offset_scaled_operand._bind_isolated(args).compile_config(config)
    with patch.object(
        shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
    ) as called:
        compiled = _offset_scaled_operand._bind_isolated(args).compile_config(config)
    assert called.call_count == 1
    for layout in ("contiguous", "misaligned", "strided"):
        for generation in range(2):
            inputs = []
            for value in args:
                if layout == "contiguous":
                    tensor = value.clone()
                elif layout == "misaligned":
                    backing = torch.empty(
                        value.numel() + 1, device=DEVICE, dtype=value.dtype
                    )
                    tensor = backing[1:].view(value.shape)
                    tensor.copy_(value)
                    assert tensor.data_ptr() % 16 != 0
                else:
                    backing = torch.empty(
                        (*value.shape[:-1], value.shape[-1] * 2),
                        device=DEVICE,
                        dtype=value.dtype,
                    )
                    tensor = backing[..., ::2]
                    tensor.copy_(value)
                    assert tensor.stride(-1) == 2
                if generation:
                    tensor.mul_(0.5)
                inputs.append(tensor)
            saved = tuple(tensor.clone() for tensor in inputs)
            actual = compiled(*inputs)
            _bits_equal(actual, original(*inputs))
            # Same tolerance as existing general warp-chain runtime checks.
            torch.testing.assert_close(actual, _reference(inputs), atol=0.01, rtol=0.01)
            for _ in range(3):
                _bits_equal(compiled(*inputs), actual)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = compiled(*inputs)
            for _ in range(3):
                captured.fill_(float("nan"))
                graph.replay()
                _bits_equal(captured, actual)
            for tensor, before in zip(inputs, saved, strict=True):
                _bits_equal(tensor, before)
