"""Small CUDA/ROCm integration checks for pipeline scopes and graph evaluation."""

from __future__ import annotations

import math

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion.autotuner.pipeline_benchmark import PipelineEvaluator
import helion.language as hl
from helion.runtime.kernel import BoundKernel
from helion.runtime.pipeline import PipelineConfig
from helion.runtime.pipeline import _active_pipeline

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA or ROCm graph execution"
)


@helion.kernel(config=helion.Config(block_sizes=[64]), static_shapes=True)
def _add_one(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result[tile] = x[tile] + 1
    return result


@helion.aot_kernel(key=lambda x: (x.numel(),), static_shapes=True)
def _aot_add_one(x: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        result[tile] = x[tile] + 1
    return result


def _forbid_autotune(*args, **kwargs):
    raise AssertionError("Explicit pipeline configs must not enter stage autotuning")


def test_repeated_stage_key_graph_and_prepared_dispatch_preserve_bound_config(
    monkeypatch,
):
    x = torch.arange(257, device=DEVICE, dtype=torch.float32)
    # Populate both native prepared/fast dispatch and a direct bound call first.
    for _ in range(2):
        torch.testing.assert_close(_add_one(x), x + 1)
    bound = _add_one.bind((x,))
    torch.testing.assert_close(bound(x), x + 1)
    original_config, original_run = bound._config, bound._run
    monkeypatch.setattr(BoundKernel, "autotune", _forbid_autotune)
    monkeypatch.setenv("HELION_SKIP_CACHE", "1")

    def pipeline(value):
        return _add_one(_add_one(value))

    measured = PipelineEvaluator(
        pipeline,
        [(x,), (-x,)],
        reference=lambda value: value + 2,
        check=torch.testing.assert_close,
        initial_config=lambda kernel, args: helion.Config(block_sizes=[128]),
    ).evaluate(PipelineConfig())
    assert measured.status == "ok", measured.error
    assert len(measured.stages) == 1
    assert all(len(trace) == 2 and trace[0] == trace[1] for trace in measured.traces)
    assert all(math.isfinite(value) and value > 0 for value in measured.timings_ms)
    assert next(iter(measured.stages.values())).config.block_sizes == [128]
    assert bound._config is original_config and bound._run is original_run
    assert _active_pipeline.get() is None
    with measured.bundle.activate() as scope:
        torch.testing.assert_close(pipeline(x), x + 2)
        torch.testing.assert_close(bound(x), x + 1)
        scope.freeze()
        torch.testing.assert_close(pipeline(x), x + 2)
        assert len(scope.stages) == 1
        assert next(iter(scope.stages.values())).config.block_sizes == [128]
    assert bound._config is original_config and bound._run is original_run
    torch.testing.assert_close(_add_one(x), x + 1)
    torch.testing.assert_close(bound(x), x + 1)


def test_real_aot_saved_config_bypasses_cache_without_installing_state(
    monkeypatch, tmp_path
):
    from helion.autotuner import aot_cache

    x = torch.arange(193, device=DEVICE, dtype=torch.float32)
    _aot_add_one.reset()
    bound = _aot_add_one.bind((x,))
    original = bound._config, bound._run
    assert original == (None, None)
    heuristic = tmp_path / "_helion_aot__aot_add_one.py"
    heuristic.write_text(
        "def autotune__aot_add_one(length):\n"
        "    assert length == 193\n"
        "    return {'block_sizes': [128]}\n"
    )
    monkeypatch.setattr(
        aot_cache, "find_heuristic_file", lambda *args, **kwargs: heuristic
    )
    monkeypatch.setattr(BoundKernel, "autotune", _forbid_autotune)
    monkeypatch.setenv("HELION_SKIP_CACHE", "1")

    def pipeline(value):
        return _aot_add_one(_aot_add_one(value))

    measured = PipelineEvaluator(
        pipeline,
        [(x,)],
        reference=lambda value: value + 2,
        check=torch.testing.assert_close,
    ).evaluate(PipelineConfig())
    assert measured.status == "ok", measured.error
    assert len(measured.stages) == 1 and len(measured.traces[0]) == 2
    stage = next(iter(measured.stages.values()))
    assert stage.config_source == f"saved_aot:{heuristic}"
    assert stage.config.block_sizes == [128]
    assert (bound._config, bound._run) == original
    with measured.bundle.activate():
        torch.testing.assert_close(pipeline(x), x + 2)
    assert (bound._config, bound._run) == original
    assert _active_pipeline.get() is None


def test_normal_torch_compile_still_works_outside_pipeline_scope():
    x = torch.arange(257, device=DEVICE, dtype=torch.float32)

    def operation(value):
        return _add_one(value) * 2

    torch.testing.assert_close(operation(x), (x + 1) * 2)
    compiled = torch.compile(operation, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), (x + 1) * 2)
    torch.testing.assert_close(compiled(-x), (-x + 1) * 2)
    assert _active_pipeline.get() is None


def test_bfloat16_topk_pipeline_handles_ties_zero_rows_and_ragged_lengths():
    from pretuned_kernels.pipeline_topk.pipeline_topk import check
    from pretuned_kernels.pipeline_topk.pipeline_topk import reference
    from pretuned_kernels.pipeline_topk.pipeline_topk import topk_pipeline

    generator = torch.Generator(device=DEVICE).manual_seed(5)
    values = torch.randint(
        -2, 3, (2, 8192), generator=generator, device=DEVICE
    ).bfloat16()
    values[:, ::3] = 0
    lengths = torch.tensor([8192, 73], device=DEVICE, dtype=torch.int32)
    zeros = torch.zeros_like(values)
    short = torch.tensor([48, 0], device=DEVICE, dtype=torch.int32)
    measured = PipelineEvaluator(
        topk_pipeline,
        [(values, lengths, 32), (zeros, short, 32)],
        reference=reference,
        check=check,
        initial_config=lambda kernel, args: helion.Config(
            block_sizes=[4096], num_warps=8, num_stages=1
        ),
    ).evaluate(PipelineConfig())
    assert measured.status == "ok", measured.error
    assert all(len(trace) == 2 for trace in measured.traces)
    assert all(math.isfinite(value) and value > 0 for value in measured.timings_ms)
