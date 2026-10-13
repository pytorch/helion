from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_plain_root import _config
from .test_cute_chained_plain_root import _plain
from helion._compiler.cute import chained_plain_root as plain
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("early_release", [False, True])
def test_plain_root_runtime_original_copy_and_scalar_fallbacks_gpu(
    dtype: torch.dtype, early_release: bool
) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(61703)
    shapes = ((1, 2, 128, 3, 128), (1, 2, 32, 3, 128))
    canonical = tuple(
        torch.randn(shape, device=DEVICE, dtype=dtype, generator=generator) * 0.05
        for shape in shapes
    )
    config = _config(
        cute_chained_tmem_early_release=early_release,
        cute_chained_auxiliary_cache=True,
    )
    with patch.object(plain, "codegen_plain_root", return_value=False):
        ordinary = _plain._bind_isolated(canonical).compile_config(config)
    with (
        patch.object(
            legacy,
            "codegen_chained_tcgen05",
            side_effect=AssertionError("old root called"),
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as emitted,
    ):
        shared = _plain._bind_isolated(canonical).compile_config(config)
    assert emitted.call_count == 1
    assert emitted.call_args.kwargs["terminal_fragment"] is True

    for layout in ("contiguous", "misaligned", "strided"):
        values = []
        for original in canonical:
            if layout == "contiguous":
                value = original.clone()
            elif layout == "misaligned":
                storage = torch.empty(original.numel() + 1, device=DEVICE, dtype=dtype)
                value = storage[1:].view(original.shape)
                value.copy_(original)
                assert value.data_ptr() % 16 != 0
            else:
                storage = torch.empty(
                    (*original.shape[:-1], original.shape[-1] * 2),
                    device=DEVICE,
                    dtype=dtype,
                )
                value = storage[..., ::2]
                value.copy_(original)
                assert value.stride(-1) == 2
            assert value.numel() == math.prod(original.shape)
            values.append(value)
        saved = tuple(value.clone() for value in values)
        a, b = (value.double().permute(0, 1, 3, 2, 4) for value in values)
        expected = (a @ b.transpose(-1, -2)).float()
        actual = shared(*values)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        _bits_equal(actual, ordinary(*values))
        for _ in range(3):
            _bits_equal(shared(*values), actual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = shared(*values)
        for _ in range(3):
            graph.replay()
            _bits_equal(captured, actual)
        for value, before in zip(values, saved, strict=True):
            _bits_equal(value, before)
