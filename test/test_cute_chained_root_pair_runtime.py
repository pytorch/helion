from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_plain_root_runtime import _bits_equal
from .test_cute_chained_tcgen05 import _tcgen_chain
from .test_cute_chained_tcgen05 import _tcgen_inputs
from .test_cute_chained_tcgen05 import (
    test_tcgen_chain_correctness as _original_correctness,
)
import helion
from helion._compiler.cute import chained_root_stage as roots
from helion._compiler.cute import chained_tcgen05 as legacy
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["plain", "scan"])
@pytest.mark.parametrize("early_release", [False, True])
def test_shared_root_pair_original_oracle_and_prefetched_fallbacks_gpu(
    dtype: torch.dtype, mode: str, early_release: bool
) -> None:
    # Run the unchanged original numerical/replay/input oracle and tolerances.
    with patch.object(
        legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old entry")
    ):
        _original_correctness(mode, dtype)
    arguments = _tcgen_inputs(DEVICE, dtype)
    config = helion.Config(
        block_sizes=[128, 64],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_tmem_early_release=early_release,
    )
    with patch.object(roots, "supports_root_pair", return_value=False):
        reference = _tcgen_chain._bind_isolated((*arguments, mode)).compile_config(
            config
        )
    with (
        patch.object(
            legacy, "codegen_chained_tcgen05", side_effect=AssertionError("old entry")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as calls,
    ):
        shared = _tcgen_chain._bind_isolated((*arguments, mode)).compile_config(config)
    assert calls.call_count == 2
    for layout in ("contiguous", "misaligned", "strided"):
        values = list(arguments)
        original = arguments[2]
        if layout == "misaligned":
            storage = torch.empty(original.numel() + 1, dtype=dtype, device=DEVICE)
            values[2] = storage[1:].view(original.shape)
            values[2].copy_(original)
            assert values[2].data_ptr() % 16 != 0
        elif layout == "strided":
            storage = torch.empty(
                (*original.shape[:-1], original.shape[-1] * 2),
                dtype=dtype,
                device=DEVICE,
            )
            values[2] = storage[..., ::2]
            values[2].copy_(original)
            assert values[2].stride(-1) == 2
        saved = tuple(value.clone() for value in values)
        actual = shared(*values, mode)
        _bits_equal(actual, reference(*values, mode))
        for _ in range(3):
            _bits_equal(actual, shared(*values, mode))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = shared(*values, mode)
        for _ in range(3):
            graph.replay()
            _bits_equal(actual, captured)
        for value, before in zip(values, saved, strict=True):
            _bits_equal(value, before)
