"""Reference retries require resource evidence, independently of candidate skips."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from torch._inductor.runtime.triton_compat import OutOfResources

from helion import exc
from helion.autotuner import benchmark_provider
from helion.autotuner.benchmark_provider import _MAX_REFERENCE_BASELINE_ATTEMPTS
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.profiler_timing import ProfilerTimingCapabilityError
from helion.runtime.config import Config


def _provider(monkeypatch, failures, *, minimum=1):
    provider = object.__new__(LocalBenchmarkProvider)
    original = Config(block_sizes=[32])
    spec = SimpleNamespace(
        autotune_reference_config=lambda: original,
        block_sizes=[SimpleNamespace(min_size=minimum)],
        normalize=Mock(),
        backend=SimpleNamespace(classify_autotune_exception=Mock(return_value="warn")),
    )
    spec.shrink_block_sizes_once = Mock(
        side_effect=lambda config: ConfigSpec.shrink_block_sizes_once(
            cast("ConfigSpec", spec), config
        )
    )
    calls = []

    def compile_config(config, **kwargs):
        def invoke(value):
            calls.append((config["block_sizes"][0], value, value.clone()))
            value.add_(10)
            if len(calls) <= len(failures):
                raise failures[len(calls) - 1]
            return value + 1

        return invoke

    provider.config_spec = cast("ConfigSpec", spec)
    provider.args = (torch.tensor([1.0, 2.0]),)
    spec.kernel = SimpleNamespace(
        compile_config=compile_config,
        env=SimpleNamespace(process_group_name=None),
        format_kernel_decorator=Mock(return_value="@helion.kernel()"),
        maybe_log_repro=Mock(),
    )
    provider.kernel = cast("Any", spec.kernel)
    provider.settings = cast("Any", SimpleNamespace())
    spec.log = Mock()
    provider.log = spec.log
    spec.source_log = Mock()
    monkeypatch.setattr(benchmark_provider, "synchronize_device", Mock())
    monkeypatch.setattr(
        benchmark_provider, "log_generated_triton_code_debug", spec.source_log
    )
    return provider, spec, calls, original


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("unsupported op"),
        AssertionError("out of resource: shared memory"),
        TypeError("CUDA out of memory"),
        RuntimeError("PassManager::run failed"),
        RuntimeError("failed to translate module to LLVM IR"),
        RuntimeError("TServiceRouterException"),
        RuntimeError("CUDA error: an illegal memory access was encountered"),
        exc.InvalidConfig("unimplemented schedule"),
    ],
)
def test_unrepairable_reference_error_is_not_a_candidate_skip(monkeypatch, error):
    provider, spec, calls, original = _provider(monkeypatch, [error] * 4)
    with pytest.raises(exc.InvalidConfig) as caught:
        provider._compute_reference_baseline()
    assert caught.value.__cause__ is error
    assert [call[0] for call in calls] == [32]
    assert original["block_sizes"] == [32]
    spec.shrink_block_sizes_once.assert_not_called()
    spec.backend.classify_autotune_exception.assert_not_called()


@pytest.mark.parametrize("oom", [False, True])
def test_resource_retry_reclones_pristine_inputs(monkeypatch, oom):
    def error():
        return (
            torch.cuda.OutOfMemoryError("CUDA out of memory")
            if oom
            else OutOfResources(4096, 2048, "shared memory")
        )

    provider, spec, calls, original = _provider(monkeypatch, [error(), error()])
    output, post_args = provider._compute_reference_baseline()
    assert [call[0] for call in calls] == [32, 16, 8]
    assert len({call[1].data_ptr() for call in calls}) == 3
    for _, _, pristine in calls:
        torch.testing.assert_close(pristine, provider.args[0])
    torch.testing.assert_close(provider.args[0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(output, torch.tensor([12.0, 13.0]))
    torch.testing.assert_close(post_args[0], torch.tensor([11.0, 12.0]))
    assert original["block_sizes"] == [32]
    spec.backend.classify_autotune_exception.assert_not_called()


@pytest.mark.parametrize("minimum,attempts", [(1, 4), (16, 2), (32, 1)])
def test_reference_retry_bound_and_minimum(monkeypatch, minimum, attempts):
    failures = [OutOfResources(4096, 2048, "shared memory") for _ in range(4)]
    provider, spec, calls, original = _provider(monkeypatch, failures, minimum=minimum)
    with pytest.raises(exc.InvalidConfig) as caught:
        provider._compute_reference_baseline()
    assert caught.value.__cause__ is failures[0]
    assert len(calls) == attempts <= _MAX_REFERENCE_BASELINE_ATTEMPTS
    assert [call[0] for call in calls] == [32 >> index for index in range(attempts)]
    assert original["block_sizes"] == [32]


def test_timing_capability_error_escapes_reference_before_retry_or_wrapping(
    monkeypatch,
):
    error = ProfilerTimingCapabilityError("CUDA out of memory in invalid timing policy")
    provider, spec, calls, original = _provider(monkeypatch, [error] * 4)
    with pytest.raises(ProfilerTimingCapabilityError) as caught:
        provider._compute_reference_baseline()
    assert caught.value is error
    assert len(calls) == 1
    spec.shrink_block_sizes_once.assert_not_called()
    spec.backend.classify_autotune_exception.assert_not_called()
    spec.log.record_timing_failure.assert_called_once_with(error)
    spec.kernel.maybe_log_repro.assert_not_called()
    spec.kernel.format_kernel_decorator.assert_not_called()
    spec.source_log.assert_not_called()


def test_successful_reference_does_not_shrink_or_reclassify(monkeypatch):
    provider, spec, calls, original = _provider(monkeypatch, [])
    output, post_args = provider._compute_reference_baseline()
    assert [call[0] for call in calls] == [32]
    torch.testing.assert_close(output, torch.tensor([12.0, 13.0]))
    assert post_args[0] is calls[0][1]
    spec.shrink_block_sizes_once.assert_not_called()
    spec.backend.classify_autotune_exception.assert_not_called()
