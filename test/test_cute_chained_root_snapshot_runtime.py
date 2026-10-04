from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_plain_root_runtime import _bits_equal
from .test_cute_chained_tcgen05 import _tcgen_chain
from .test_cute_chained_tcgen05 import _tcgen_inputs
from .test_cute_chained_tcgen05 import (
    test_tcgen_chain_correctness as _original_correctness,
)
import helion
from helion._compiler.cute import chained_tcgen05 as root
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _config(columns: int, early: bool) -> helion.Config:
    return helion.Config(
        block_sizes=[128, columns],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_tmem_early_release=early,
    )


def _compile(arguments, mode, config, columns, *, source_only=False):
    original = root.codegen_shared_root_sequence

    def selected(cg, plan):
        return original(cg, plan, snapshot_tile_columns=columns)

    with (
        patch.object(root, "codegen_shared_root_sequence", selected),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as calls,
    ):
        bound = _tcgen_chain._bind_isolated((*arguments, mode))
        result = bound.to_code(config) if source_only else bound.compile_config(config)
    assert [call.args[3] for call in calls.call_args_list] == [0, 1]
    sequence = calls.call_args_list[0].kwargs["root_actions"].sequence
    if columns:
        assert sequence.snapshot is not None
        assert sequence.snapshot_ready and sequence.snapshot_consumed
    else:
        assert sequence.snapshot is None
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["plain", "scan"])
@pytest.mark.parametrize("columns", [32, 64, 128])
@pytest.mark.parametrize("early", [False, True])
def test_root_snapshot_runtime_source_preflight(dtype, mode, columns, early):
    initialized = torch.cuda.is_initialized()
    arguments = _tcgen_inputs("cpu", dtype, n=columns)
    with _cpu_codegen():
        source = _compile(
            arguments, mode, _config(columns, early), 32, source_only=True
        )
    assert isinstance(source, str)
    assert "for chain_1_bridge_panel in cutlass.range(4, unroll=1)" in source
    assert "chain_0_copy =" not in source
    assert "chain_1_copy =" in source
    assert torch.cuda.is_initialized() == initialized


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["plain", "scan"])
@pytest.mark.parametrize("columns", [32, 64, 128])
@pytest.mark.parametrize("early", [False, True])
def test_root_snapshot_original_oracle_values_generations_and_replay_gpu(
    dtype, mode, columns, early
):
    # Keep the original numerical oracle and its tolerances unchanged. The
    # remaining widths also compare bitwise against the ordinary full snapshot.
    if columns == 64 and not early:
        original_codegen = root.codegen_shared_root_sequence

        def selected(cg, plan):
            return original_codegen(cg, plan, snapshot_tile_columns=32)

        with patch.object(root, "codegen_shared_root_sequence", selected):
            _original_correctness(mode, dtype)

    arguments = _tcgen_inputs(DEVICE, dtype, n=columns)
    config = _config(columns, early)
    ordinary = _compile(arguments, mode, config, 0)
    streamed = _compile(arguments, mode, config, 32)
    assert callable(ordinary) and callable(streamed)
    generations = []
    for generation in range(2):
        values = [value.clone() for value in arguments]
        generations.append(values)
        if generation:
            assert all(
                value.data_ptr() != previous.data_ptr()
                for value, previous in zip(values, generations[0], strict=True)
            )
        for layout in ("contiguous", "misaligned", "strided"):
            current = list(values)
            original = values[2]
            if layout == "misaligned":
                storage = torch.empty(original.numel() + 1, dtype=dtype, device=DEVICE)
                current[2] = storage[1:].view(original.shape)
                current[2].copy_(original)
                assert current[2].data_ptr() % 16 != 0
            elif layout == "strided":
                storage = torch.empty(
                    (*original.shape[:-1], original.shape[-1] * 2),
                    dtype=dtype,
                    device=DEVICE,
                )
                current[2] = storage[..., ::2]
                current[2].copy_(original)
                assert current[2].stride(-1) == 2
            saved = tuple(value.clone() for value in current)
            expected = ordinary(*current, mode)
            actual = streamed(*current, mode)
            _bits_equal(actual, expected)
            for _ in range(3):
                _bits_equal(streamed(*current, mode), expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = streamed(*current, mode)
            for _ in range(3):
                captured.fill_(float("nan"))
                graph.replay()
                _bits_equal(captured, expected)
            for value, before in zip(current, saved, strict=True):
                _bits_equal(value, before)
