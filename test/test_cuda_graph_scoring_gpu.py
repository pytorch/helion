from __future__ import annotations

import math
import os

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion.autotuner import cuda_graph_scoring
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.graph_scoring import run_graph_job
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
import helion.language as hl


def _reference(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def _check_output(arguments, output, baseline):
    torch.testing.assert_close(output, baseline)
    torch.testing.assert_close(output, _reference(*arguments))


@helion.kernel(
    backend="cute",
    autotune_effort="none",
    autotune_precompile=None,
    autotune_handoff=False,
    autotune_baseline_fn=_reference,
)
def _add_one_cute(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        output[tile] = x[tile] + 1
    return output


@helion.kernel(
    backend="triton",
    autotune_effort="none",
    autotune_precompile=None,
    autotune_handoff=False,
    autotune_baseline_fn=_reference,
)
def _add_one_triton(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        output[tile] = x[tile] + 1
    return output


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)
@pytest.mark.parametrize("backend", ["cute", "triton"])
def test_native_graph_worker_smoke(backend):
    storage = torch.randn(4104, device=DEVICE)
    arguments = (storage[8:],)
    pristine = arguments[0].clone()
    kernel = _add_one_cute if backend == "cute" else _add_one_triton
    bound = kernel.bind(arguments)
    with cuda_graph_scoring(check_output=_check_output) as scoring:
        provider = LocalBenchmarkProvider(
            bound,
            bound.settings,
            bound.config_spec,
            arguments,
            AutotuningLogger(bound.settings),
            AutotuneMetrics(),
        )
        try:
            provider.setup()
            configs = [
                helion.Config(block_sizes=[128]),
                helion.Config(block_sizes=[256]),
            ]
            functions = [
                bound.compile_config(config, allow_print=False) for config in configs
            ]
            # Candidate and finalist scoring use the native provider dispatch.
            candidate = provider._benchmark_function(configs[0], functions[0])
            assert math.isfinite(candidate) and candidate > 0
            finalists = provider.benchmark_isolated(functions, warmup=1, rep=3)
            assert finalists is not None and all(
                isinstance(value, float) and value > 0 for value in finalists
            )
            mirrored = run_graph_job(provider, functions, repeat=2)
            assert len(mirrored["trace"].medians_ms) == mirrored["captures"] == 2
            assert mirrored["sample_count"] == 6
            assert len(scoring.evidence) == 4
            assert all(
                row["worker_pid"] != os.getpid()
                and row["graph_output_accuracy"] == row["preserved_inputs"] == "ok"
                and row["eager_fallback"] is False
                for row in scoring.evidence
            )
            torch.testing.assert_close(arguments[0], pristine, atol=0, rtol=0)
        finally:
            provider.cleanup()
