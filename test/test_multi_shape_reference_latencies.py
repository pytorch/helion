from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from .test_multi_shape_autotune import (
    TestMultiShapeBenchmarkProvider as _ProviderFixture,
)
from .test_multi_shape_autotune import _FakeBoundKernel
from .test_multi_shape_autotune import _make_config_spec
from .test_multi_shape_autotune import _runtime_settings
from .test_multi_shape_autotune import _RuntimeBackend
from .test_multi_shape_autotune import _RuntimeBoundKernel
from helion import exc
from helion.autotuner.benchmark_provider import _aggregate_multi_shape_timings
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.runtime.config import Config
from helion.runtime.kernel import Kernel


def _facade(**options):
    backend = _RuntimeBackend()
    settings = _runtime_settings(autotune_baseline_fn=Mock())
    bounds = [_RuntimeBoundKernel(backend, settings) for _ in range(2)]
    kernel = object.__new__(Kernel)
    kernel.settings = settings
    kernel._annotations = [object, object]
    kernel.normalize_args = Mock(side_effect=lambda *args: tuple(args))
    kernel.bind = Mock(side_effect=bounds)
    kernel._base_specialization_key = Mock(side_effect=[("shape", 8), ("shape", 16)])
    kernel._get_bound_kernel_cache_key = Mock(
        side_effect=[
            SimpleNamespace(
                specialization_key=("shape", size),
                extra_results=(),
                compiler_seed_results=(),
            )
            for size in (8, 16)
        ]
    )
    with (
        patch("helion.runtime.kernel.dist.is_initialized", return_value=False),
        patch("helion.runtime.kernel._current_device_index", return_value=0),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
    ):
        winner = kernel.autotune_multi(
            [("case-a", 8), ("case-b", 16)],
            aggregation="max",
            relative_to="baseline",
            cache_tag="same-inputs",
            **options,
        )
    carrier = backend.autotune_calls[0][1]
    assert isinstance(carrier, _MultiShapeAutotuneArgs)
    assert all(bound.set_config.call_args.args[0] == winner for bound in bounds)
    return bounds[0], carrier


def test_reference_tuple_is_detached_and_default_key_is_unchanged() -> None:
    _, default = _facade()
    assert default.reference_latencies is None
    assert default.workload_key == (
        "multi_shape:v1",
        "max",
        "baseline",
        "same-inputs",
        ((("shape", 8), (), ()), (("shape", 16), (), ())),
    )
    supplied = [0.040, 0.052]
    _, carrier = _facade(reference_latencies_ms=supplied)
    supplied[0] = 999.0
    assert carrier.reference_latencies == (0.040, 0.052)
    assert carrier.workload_key[:-2] == default.workload_key
    assert carrier.workload_key[-2:] == (
        "reference_latencies_ms:v1",
        (0.040, 0.052),
    )


def test_exact_reference_tuple_salts_actual_local_cache_key() -> None:
    def key(values):
        bound, carrier = _facade(reference_latencies_ms=values)
        cache = object.__new__(LocalAutotuneCache)
        cache.kernel = bound
        cache.args = carrier
        cache.autotuner = SimpleNamespace(settings=bound.settings)
        with (
            patch("helion.autotuner.local_cache.torch.version.cuda", "13.0"),
            patch(
                "helion.autotuner.local_cache.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "helion.autotuner.local_cache.get_device_name", return_value="Fake GPU"
            ),
        ):
            return cache._generate_key()

    assert key((0.040, 0.052)) != key((0.041, 0.052))
    assert key((0.040, 0.052)) == key([0.040, 0.052])


@pytest.mark.parametrize(
    "references",
    [
        [],
        [1.0],
        [1.0, 2.0, 3.0],
        "1,2",
        (0.0, 1.0),
        (-1.0, 1.0),
        (math.inf, 1.0),
        (math.nan, 1.0),
        (True, 1.0),
        ("1", 2.0),
        (None, 1.0),
        (1 + 0j, 1.0),
        (10**1000, 1.0),
    ],
)
def test_invalid_host_values_fail_before_binding(references) -> None:
    kernel = object.__new__(Kernel)
    kernel.normalize_args = Mock()
    kernel.bind = Mock()
    with pytest.raises(exc.InvalidAPIUsage, match="reference_latencies_ms"):
        kernel.autotune_multi(
            [("a",), ("b",)], relative_to="default", reference_latencies_ms=references
        )
    kernel.normalize_args.assert_not_called()
    kernel.bind.assert_not_called()


def test_reference_requires_relative_mode_before_binding() -> None:
    kernel = object.__new__(Kernel)
    kernel.bind = Mock()
    with pytest.raises(exc.InvalidAPIUsage, match="requires relative_to"):
        kernel.autotune_multi([("a",), ("b",)], reference_latencies_ms=(0.040, 0.052))
    kernel.bind.assert_not_called()


@pytest.mark.parametrize(
    "references", [torch.tensor([0.040, 0.052]), [torch.tensor(0.040), 0.052]]
)
def test_tensor_references_are_rejected_without_scalar_conversion(references) -> None:
    kernel = object.__new__(Kernel)
    kernel.bind = Mock()
    with (
        patch.object(
            torch.Tensor, "__float__", side_effect=AssertionError("tensor conversion")
        ) as converted,
        pytest.raises(exc.InvalidAPIUsage, match="reference_latencies_ms"),
    ):
        kernel.autotune_multi(
            [("a",), ("b",)],
            relative_to="default",
            reference_latencies_ms=references,
        )
    converted.assert_not_called()
    kernel.bind.assert_not_called()


def test_supplied_scales_do_not_admit_custom_autotuner() -> None:
    kernel = object.__new__(Kernel)
    kernel.settings = _runtime_settings(autotuner_fn=Mock())
    kernel.bind = Mock()
    with (
        patch("helion.runtime.kernel.dist.is_initialized", return_value=False),
        pytest.raises(exc.InvalidAPIUsage, match="default registered autotuner"),
    ):
        kernel.autotune_multi(
            [("a",), ("b",)],
            relative_to="default",
            reference_latencies_ms=(0.040, 0.052),
        )
    kernel.bind.assert_not_called()


def test_baseline_correctness_callback_is_still_required() -> None:
    kernel = object.__new__(Kernel)
    kernel.settings = _runtime_settings()
    kernel.bind = Mock()
    with (
        patch("helion.runtime.kernel.dist.is_initialized", return_value=False),
        pytest.raises(exc.InvalidAPIUsage, match="requires autotune_baseline_fn"),
    ):
        kernel.autotune_multi(
            [("a",), ("b",)],
            relative_to="baseline",
            reference_latencies_ms=(0.040, 0.052),
        )
    kernel.bind.assert_not_called()


def test_precomputed_scales_skip_reference_timing_but_run_child_setup() -> None:
    spec = _make_config_spec()
    provider, _ = _ProviderFixture()._make_provider(
        [_FakeBoundKernel(spec, [0.040]), _FakeBoundKernel(spec, [0.052])],
        aggregation="max",
        relative_to="baseline",
        references=(0.040, 0.052),
    )
    with patch.object(
        provider, "_measure_reference", side_effect=AssertionError("reference timed")
    ) as timed:
        provider.setup()
    timed.assert_not_called()
    assert all(child.setup_count == 1 for child in provider.children)


def test_normalized_objective_selects_the_config_covering_both_cases() -> None:
    configs = [Config(block_sizes=[64]), Config(block_sizes=[128])]

    def select(references):
        spec = _make_config_spec()
        provider, _ = _ProviderFixture()._make_provider(
            [
                _FakeBoundKernel(spec, [0.041, 0.040]),
                _FakeBoundKernel(spec, [0.051, 0.052]),
            ],
            aggregation="max",
            references=references,
        )
        search = SimpleNamespace(
            configs=configs,
            benchmark_batch=lambda candidates, *, desc: provider.benchmark(
                candidates, desc=desc
            ),
        )
        return FiniteSearch._autotune(search)

    assert select(None) is configs[0]
    assert select((0.040, 0.052)) is configs[1]
    assert (
        _aggregate_multi_shape_timings(
            (0.040, 0.052), aggregation="max", references=(0.040, 0.052)
        )
        == 1.0
    )
