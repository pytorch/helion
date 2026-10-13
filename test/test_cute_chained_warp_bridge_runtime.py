from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_matmul import _plain_chain
from .test_cute_chained_matmul import _scan_cache_inputs
from .test_cute_chained_matmul import _scan_cached_chain
from .test_cute_chained_plain_root_runtime import _bits_equal
from .test_cute_chained_warp_bridge import _right_bridge
import helion
from helion._compiler.cute import chained_warp_bridge as bridge
from helion._compiler.cute import chained_warp_stage as shared
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _fixture(device, dtype, ragged, warps, kind):
    generator = torch.Generator(device=device).manual_seed(81207)
    suffix = ()
    if kind.startswith("scan_"):
        length = 121 if ragged else 128
        args = tuple(
            value.to(dtype) if value.is_floating_point() else value
            for value in _scan_cache_inputs(device, length, 1)
        )
        kernel, suffix = _scan_cached_chain, (kind.removeprefix("scan_"),)
        schedule = "cp_async_register_reuse_scan"
    else:
        if kind == "plain":
            shapes = (
                (2, 35 if ragged else 32, 64),
                (2, 128, 64),
                (2, 128, 35 if ragged else 32),
            )
            kernel = _plain_chain
        else:
            assert kind == "right"
            shapes = ((2, 16, 64), (2, 128, 64), (2, 35 if ragged else 32, 128))
            kernel = _right_bridge
        args = tuple(
            torch.randn(shape, device=device, dtype=dtype, generator=generator) * 0.0625
            for shape in shapes
        )
        schedule = "cp_async_register"
    config = helion.Config(num_warps=warps, cute_chained_mma_schedule=schedule)
    return kernel, args, suffix, config


def _reference(args, kind):
    a, b, v = args[:3]
    first = a.double() @ b.double().transpose(-1, -2)
    if kind == "plain":
        return (first.to(v.dtype).double() @ v.double()).to(v.dtype)
    if kind == "right":
        rows = torch.arange(a.shape[1], device=a.device)
        columns = torch.arange(b.shape[1], device=b.device)
        weights = torch.where(rows[:, None] >= columns[None, :], first * 0.5, 0.0)
        return (v.double() @ weights.to(v.dtype).double().transpose(-1, -2)).to(v.dtype)
    delta, other = args[3:5]
    length = a.shape[1]
    scan = delta[:, :length].double()
    if kind == "scan_multiple":
        scan = scan * other[:, :length].double()
        selected = other[:, :length]
        residual = delta[:, :length]
    else:
        assert kind == "scan_shift"
        selected = residual = delta[:, 16 : length + 16]
    decay = scan.cumsum(-1)
    weights = first * (decay[:, :, None] - decay[:, None, :]).exp()
    weights = weights * selected[:, None, :].double()
    result = torch.tril(weights).to(v.dtype).double() @ v.double()
    return (result + residual[:, :, None].double()).to(v.dtype)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("ragged", (False, True))
@pytest.mark.parametrize("warps", (4, 8))
@pytest.mark.parametrize("kind", ("plain", "scan_shift", "scan_multiple", "right"))
def test_warp_bridge_runtime_source_cpu(dtype, ragged, warps, kind):
    kernel, args, suffix, config = _fixture("cpu", dtype, ragged, warps, kind)
    with _cpu_codegen():
        with patch.object(bridge, "root_warp_bridge_sequence", return_value=None):
            old = kernel._bind_isolated((*args, *suffix)).to_code(config)
        with patch.object(
            shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
        ) as called:
            new = kernel._bind_isolated((*args, *suffix)).to_code(config)
    assert old == new and called.call_count == 2
    assert "chain_0_c_ptr =" not in new
    assert isinstance(
        called.call_args_list[0].args[3].result, shared.WarpRegisterResult
    )


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("ragged", (False, True))
@pytest.mark.parametrize("warps", (4, 8))
@pytest.mark.parametrize("kind", ("plain", "scan_shift", "scan_multiple", "right"))
def test_warp_bridge_original_bits_layouts_and_replay_gpu(dtype, ragged, warps, kind):
    kernel, args, suffix, config = _fixture(DEVICE, dtype, ragged, warps, kind)
    with patch.object(bridge, "root_warp_bridge_sequence", return_value=None):
        original = kernel._bind_isolated((*args, *suffix)).compile_config(config)
    with patch.object(
        shared, "emit_prepared_warp_stage", wraps=shared.emit_prepared_warp_stage
    ) as called:
        compiled = kernel._bind_isolated((*args, *suffix)).compile_config(config)
    assert called.call_count == 2
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
                if generation and tensor.is_floating_point():
                    tensor.mul_(0.5)
                inputs.append(tensor)
            saved = tuple(value.clone() for value in inputs)
            actual = compiled(*inputs, *suffix)
            _bits_equal(actual, original(*inputs, *suffix))
            # Preserve the corresponding existing plain/scan runtime tolerance.
            atol, rtol = (0.0001, 0.02) if kind.startswith("scan_") else (0.01, 0.01)
            torch.testing.assert_close(
                actual, _reference(inputs, kind), atol=atol, rtol=rtol
            )
            for _ in range(3):
                _bits_equal(compiled(*inputs, *suffix), actual)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = compiled(*inputs, *suffix)
            for _ in range(3):
                captured.fill_(float("nan"))
                graph.replay()
                _bits_equal(captured, actual)
            for value, before in zip(inputs, saved, strict=True):
                _bits_equal(value, before)
