from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

import helion
from helion import exc
from helion._compiler.backend import TritonBackend
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion.autotuner import benchmark_provider as benchmark_provider_module
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import BenchmarkResult
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeAggregation
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeReference
from helion.autotuner.benchmark_provider import _aggregate_multi_shape_timings
from helion.autotuner.benchmark_provider import _format_selected_multi_shape_measurement
from helion.autotuner.benchmark_provider import _materialize_multi_shape_config
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import LoopOrderSpec
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.llm_seeded_lfbo import LLMSeededSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.metrics import AutotuneMetrics
import helion.language as hl
from helion.runtime.config import Config
from helion.runtime.kernel import Kernel
from helion.runtime.settings import Settings
from helion.runtime.settings import default_autotuner_fn

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence


class TestMultiShapeTimingAggregation(unittest.TestCase):
    def test_raw_geometric_mean_and_max(self) -> None:
        self.assertAlmostEqual(
            _aggregate_multi_shape_timings(
                [1.0, 4.0], aggregation="geomean", references=None
            ),
            2.0,
        )
        self.assertEqual(
            _aggregate_multi_shape_timings(
                [1.0, 4.0], aggregation="max", references=None
            ),
            4.0,
        )

    def test_reference_max_returns_ratio(self) -> None:
        self.assertEqual(
            _aggregate_multi_shape_timings(
                [4.0, 8.0], aggregation="max", references=[2.0, 8.0]
            ),
            2.0,
        )
        self.assertAlmostEqual(
            _aggregate_multi_shape_timings(
                [4.0, 8.0], aggregation="geomean", references=[2.0, 8.0]
            ),
            math.sqrt(2.0),
        )

    def test_invalid_values_and_references_return_infinity(self) -> None:
        for timings in ([], [0.0], [-1.0], [math.inf], [math.nan]):
            with self.subTest(timings=timings):
                self.assertEqual(
                    _aggregate_multi_shape_timings(
                        timings, aggregation="geomean", references=None
                    ),
                    math.inf,
                )
        for references in ([1.0], [1.0, 0.0], [1.0, math.inf]):
            with self.subTest(references=references):
                self.assertEqual(
                    _aggregate_multi_shape_timings(
                        [2.0, 8.0], aggregation="max", references=references
                    ),
                    math.inf,
                )


def _make_config_spec() -> ConfigSpec:
    spec = ConfigSpec(backend=TritonBackend())
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
    spec.loop_orders.append(LoopOrderSpec([0]))
    return spec


def _make_carrier(
    cases: Sequence[tuple[object, tuple[object, ...]]],
    *,
    aggregation: MultiShapeAggregation = "geomean",
    relative_to: MultiShapeReference = None,
    cache_tag: str | None = None,
    workload_key: tuple[object, ...] = ("workload",),
    reference_latencies: tuple[float, ...] | None = None,
) -> _MultiShapeAutotuneArgs:
    return _MultiShapeAutotuneArgs(
        cases=tuple(cases),  # type: ignore[arg-type]
        aggregation=aggregation,
        relative_to=relative_to,
        cache_tag=cache_tag,
        workload_key=workload_key,
        reference_latencies=reference_latencies,
    )


class TestMultiShapeConfigMaterialization(unittest.TestCase):
    def test_alias_overlay_is_detached(self) -> None:
        spec = _make_config_spec()
        default = spec.default_config()
        sparse = Config(block_size=64)

        result = _materialize_multi_shape_config(spec, sparse)

        self.assertEqual(result.block_sizes, [64])
        self.assertEqual(result.num_warps, default.num_warps)
        self.assertNotIn("block_size", result.config)
        self.assertEqual(sparse.config, {"block_size": 64})
        result.block_sizes[0] = 128
        result.loop_orders[0][0] = 1
        self.assertEqual(default.block_sizes, [32])
        self.assertEqual(default.loop_orders, [[0]])

    def test_singular_plural_collision_is_invalid(self) -> None:
        with self.assertRaises(helion.exc.InvalidConfig):
            _materialize_multi_shape_config(
                _make_config_spec(), Config(block_size=32, block_sizes=[64])
            )

    def test_explicit_reset_removes_compiler_default(self) -> None:
        spec = Mock()
        spec.default_config.return_value = Config(block_sizes=[32], num_threads=[8])

        def normalize(config: Config) -> None:
            if "block_size" in config.config:
                config.config["block_sizes"] = [config.config.pop("block_size")]
            if config.config.get("num_threads") == [0]:
                config.config.pop("num_threads")

        spec.normalize.side_effect = normalize

        result = _materialize_multi_shape_config(
            spec, Config(block_size=64, num_threads=[0])
        )

        self.assertEqual(result.block_sizes, [64])
        self.assertNotIn("num_threads", result.config)

    def test_missing_optional_reset_does_not_restore_compiler_default(self) -> None:
        spec = Mock()
        spec.default_config.return_value = Config.from_dict(
            {"block_sizes": [32], "epilogue_subtile": 2}
        )

        def normalize(config: Config) -> None:
            config.config.setdefault("block_sizes", [32])
            config.config.setdefault("num_warps", 4)

        spec.normalize.side_effect = normalize

        result = _materialize_multi_shape_config(
            spec,
            Config.from_dict({"block_sizes": [64]}),
        )

        self.assertEqual(result.block_sizes, [64])
        self.assertEqual(result.num_warps, 4)
        self.assertNotIn("epilogue_subtile", result.config)
        spec.default_config.assert_not_called()


class TestMultiShapeAutotuneArgs(unittest.TestCase):
    def test_sequence_operations_use_anchor_args(self) -> None:
        anchor_args = ("first", 2, [3])
        carrier = _make_carrier(((object(), anchor_args), (object(), ("secondary",))))

        self.assertEqual(len(carrier), 3)
        self.assertEqual(carrier[0], "first")
        self.assertIs(carrier[-1], anchor_args[-1])
        self.assertEqual(list(cast("Iterable[object]", carrier)), list(anchor_args))


def _cache_kernel_source(value: object) -> object:
    return value


class _RuntimeConfigSpec:
    def structural_fingerprint(
        self, *, advanced_controls_files: list[str] | None
    ) -> tuple[tuple[str, int]]:
        return (("block_sizes", 1),)

    def structural_fingerprint_hash(
        self, *, advanced_controls_files: list[str] | None
    ) -> str:
        return "fake-config-spec"

    def normalize(self, config: Config) -> None:
        config.config.setdefault("num_warps", 4)


class _RuntimeBackend:
    name = "triton"

    def __init__(self) -> None:
        self.autotune_calls: list[tuple[object, object]] = []
        self.finalized: list[object] = []

    def make_ephemeral_cache(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def autotune(
        self, bound_kernel: object, args: object, *, force: bool, **options: object
    ) -> Config:
        self.autotune_calls.append((bound_kernel, args))
        return Config(block_sizes=[64])

    def finalize_ephemeral_cache(self, bound_kernel: object, config: Config) -> None:
        self.finalized.append(bound_kernel)


class _RuntimeBoundKernel:
    def __init__(self, backend: _RuntimeBackend, settings: SimpleNamespace) -> None:
        self.kernel = SimpleNamespace(
            fn=_cache_kernel_source,
            settings=settings,
            _base_specialization_key=Mock(
                side_effect=AssertionError("single-shape cache key recomputed")
            ),
            _create_bound_kernel_cache_key=Mock(
                side_effect=AssertionError("single-shape cache key recomputed")
            ),
        )
        self.settings = settings
        self.config_spec = _RuntimeConfigSpec()
        self.env = SimpleNamespace(
            backend=backend,
            device=torch.device("cuda", 0),
            process_group_name=None,
        )
        self.compile_config = Mock()
        self.set_config = Mock()
        self.extra_cache_key = Mock(return_value="")


def _runtime_settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "backend": "triton",
        "autotuner_fn": default_autotuner_fn,
        "autotune_cache": "LocalAutotuneCache",
        "autotune_baseline_fn": None,
        "autotune_baseline_accuracy_check_fn": None,
        "autotune_benchmark_fn": None,
        "autotune_config_filter": None,
        "autotune_search_acf": [],
        "autotune_accuracy_check": True,
        "autotune_baseline_atol": None,
        "autotune_baseline_rtol": None,
        "autotune_best_of_k": 1,
        "static_shapes": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestMultiShapeRuntime(unittest.TestCase):
    @patch("helion.autotuner.local_cache.torch.version.cuda", "13.0")
    @patch("helion.autotuner.local_cache.torch.cuda.is_available", return_value=True)
    @patch("helion.autotuner.local_cache.get_device_name", return_value="Fake GPU")
    def test_local_cache_uses_joint_workload_key(
        self, device_name: object, cuda_available: object
    ) -> None:
        settings = _runtime_settings()
        bound = _RuntimeBoundKernel(_RuntimeBackend(), settings)
        workload_key = (
            "multi_shape:v1",
            "max",
            "default",
            "workload-v2",
            (("shape-a",), ("shape-b",)),
        )
        cache = object.__new__(LocalAutotuneCache)
        cache.kernel = bound  # pyrefly: ignore [bad-assignment]
        cache.args = _make_carrier(  # pyrefly: ignore [bad-assignment]
            ((bound, ("anchor", 8)), (bound, ("secondary", 16))),
            workload_key=workload_key,
        )
        cache.autotuner = SimpleNamespace(  # pyrefly: ignore [bad-assignment]
            settings=settings
        )

        key = cache._generate_key()

        self.assertEqual(key.specialization_key, workload_key)
        self.assertEqual(key.extra_results, ())
        bound.kernel._base_specialization_key.assert_not_called()
        bound.kernel._create_bound_kernel_cache_key.assert_not_called()

    def test_option_validation_precedes_normalization_and_binding(self) -> None:
        invalid_options = (
            ({"aggregation": "median"}, "aggregation"),
            ({"relative_to": "fastest"}, "relative_to"),
            ({"cache_tag": ""}, "non-empty string"),
            ({"cache_tag": 1}, "non-empty string"),
        )
        for kwargs, message in invalid_options:
            kernel = object.__new__(Kernel)
            kernel.normalize_args = Mock()
            kernel.bind = Mock()

            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(exc.InvalidAPIUsage, message),
            ):
                kernel.autotune_multi([("real-argument",)], **kwargs)  # type: ignore[arg-type]

            kernel.normalize_args.assert_not_called()
            kernel.bind.assert_not_called()

    @patch("helion.runtime.kernel.dist.is_initialized", return_value=False)
    def test_unsupported_settings_fail_before_binding(
        self, distributed: object
    ) -> None:
        cases = (
            (
                "custom accuracy callback without cache tag",
                _runtime_settings(autotune_baseline_accuracy_check_fn=Mock()),
                {},
                "requires cache_tag",
            ),
            (
                "custom benchmark callback",
                _runtime_settings(autotune_benchmark_fn=Mock()),
                {"cache_tag": "callback-v1"},
                "does not support",
            ),
            (
                "dynamic shapes without cache tag",
                _runtime_settings(static_shapes=False),
                {},
                "static_shapes=False",
            ),
        )
        for name, settings, kwargs, message in cases:
            kernel = object.__new__(Kernel)
            kernel.settings = settings  # pyrefly: ignore [bad-assignment]
            kernel.normalize_args = Mock()
            kernel.bind = Mock()

            with (
                self.subTest(case=name),
                self.assertRaisesRegex(exc.InvalidAPIUsage, message),
            ):
                kernel.autotune_multi([("real-argument",)], **kwargs)

            kernel.normalize_args.assert_not_called()
            kernel.bind.assert_not_called()

    @patch("helion.runtime.kernel.dist.is_initialized", return_value=False)
    def test_runtime_numeric_values_require_cache_tag(
        self, distributed: object
    ) -> None:
        kernel = object.__new__(Kernel)
        kernel.settings = _runtime_settings()  # pyrefly: ignore [bad-assignment]
        kernel._annotations = [object]
        kernel.normalize_args = Mock(side_effect=lambda *args: tuple(args))
        kernel.bind = Mock()

        with self.assertRaisesRegex(exc.InvalidAPIUsage, "runtime numeric value"):
            kernel.autotune_multi([([1, 2],)])

        kernel.bind.assert_not_called()

    @patch("helion.runtime.kernel.target_device_capability", return_value=(9, 0))
    @patch("helion.runtime.kernel._current_device_index", return_value=0)
    @patch("helion.runtime.kernel.dist.is_initialized", return_value=False)
    def test_builds_workload_key_and_installs_one_winner(
        self,
        distributed: object,
        current: object,
        capability: object,
    ) -> None:
        backend = _RuntimeBackend()
        settings = _runtime_settings()
        bounds = [
            _RuntimeBoundKernel(backend, settings),
            _RuntimeBoundKernel(backend, settings),
        ]
        case_keys = [
            SimpleNamespace(specialization_key=("shape", 8), extra_results=()),
            SimpleNamespace(specialization_key=("shape", 16), extra_results=()),
        ]
        kernel = object.__new__(Kernel)
        kernel.settings = settings  # pyrefly: ignore [bad-assignment]
        kernel._annotations = [object, object]
        kernel.normalize_args = Mock(side_effect=lambda *args: tuple(args))
        kernel.bind = Mock(side_effect=bounds)
        kernel._base_specialization_key = Mock(
            side_effect=[("shape", 8), ("shape", 16)]
        )
        kernel._get_bound_kernel_cache_key = Mock(side_effect=case_keys)

        winner = kernel.autotune_multi(
            [("case-a", 8), ("case-b", 16)], cache_tag="numeric-shapes-v1"
        )

        self.assertEqual(winner.config, {"block_sizes": [64], "num_warps": 4})
        self.assertEqual(len(backend.autotune_calls), 1)
        carrier = backend.autotune_calls[0][1]
        self.assertIsInstance(carrier, _MultiShapeAutotuneArgs)
        self.assertEqual(
            carrier.workload_key,
            (
                "multi_shape:v1",
                "geomean",
                None,
                "numeric-shapes-v1",
                ((("shape", 8), ()), (("shape", 16), ())),
            ),
        )
        self.assertEqual(backend.finalized, bounds)
        for bound in bounds:
            bound.compile_config.assert_called_once_with(winner)
            bound.set_config.assert_called_once_with(winner)


class _Log:
    def __init__(self) -> None:
        self.debug_messages: list[object] = []

    def debug(self, message: object) -> None:
        self.debug_messages.append(message)

    def register_config(self, config: Config) -> None:
        return None

    def record_autotune_entry(self, entry: object) -> None:
        pass


class _FakeBoundKernel:
    def __init__(
        self,
        config_spec: ConfigSpec,
        timings: list[float],
        statuses: list[str] | None = None,
        compile_times: list[float | None] | None = None,
        *,
        error: Exception | None = None,
        accuracy_failure_indices: list[int] | None = None,
        compile_failure_indices: list[int] | None = None,
    ) -> None:
        self.config_spec = config_spec
        self.settings = SimpleNamespace(
            autotune_baseline_fn=None,
            autotune_ignore_errors=False,
        )
        self.env = SimpleNamespace(process_group_name=None)
        self.timings = timings
        self.statuses = statuses or ["ok"] * len(timings)
        self.compile_times = compile_times or [None] * len(timings)
        self.error = error
        self.accuracy_failure_indices = accuracy_failure_indices or []
        self.compile_failure_indices = compile_failure_indices or []
        self.accuracy_failures = len(self.accuracy_failure_indices)
        self.compile_failures = len(self.compile_failure_indices)


class _FakeLocalBenchmarkProvider:
    mutated_arg_indices: tuple[int, ...] = ()

    def __init__(
        self,
        kernel: _FakeBoundKernel,
        settings: object,
        config_spec: ConfigSpec,
        args: tuple[object, ...],
        log: object,
        autotune_metrics: AutotuneMetrics,
    ) -> None:
        self.kernel = kernel
        self.settings = settings
        self.config_spec = config_spec
        self.args = args
        self._autotune_metrics = autotune_metrics
        self._accuracy_failure_config_ids: list[int] = []
        self._compile_failure_config_ids: list[int] = []
        self.benchmark_calls: list[tuple[list[Config], str]] = []
        self.setup_count = 0
        self.cleanup_count = 0
        self.budget_exceeded_fn: Callable[[], bool] = lambda: False
        self.fns = [lambda: None for _ in kernel.timings]

    def set_budget_exceeded_fn(self, fn: Callable[[], bool]) -> None:
        self.budget_exceeded_fn = fn

    def setup(self) -> None:
        self.setup_count += 1

    def cleanup(self) -> None:
        self.cleanup_count += 1

    def benchmark(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        self.benchmark_calls.append((configs, desc))
        if self.kernel.error is not None:
            raise self.kernel.error
        self._autotune_metrics.num_accuracy_failures += self.kernel.accuracy_failures
        self._autotune_metrics.num_compile_failures += self.kernel.compile_failures
        self._accuracy_failure_config_ids.extend(
            id(config)
            for index, config in enumerate(configs)
            if index in self.kernel.accuracy_failure_indices
        )
        self._compile_failure_config_ids.extend(
            id(config)
            for index, config in enumerate(configs)
            if index in self.kernel.compile_failure_indices
        )
        if len(configs) > len(self.kernel.timings):
            raise AssertionError("fake child received too many configs")
        return [
            BenchmarkResult(
                config,
                self.fns[index],
                self.kernel.timings[index],
                self.kernel.statuses[index],  # type: ignore[arg-type]
                self.kernel.compile_times[index],
            )
            for index, config in enumerate(configs)
        ]


class TestMultiShapeBenchmarkProvider(unittest.TestCase):
    def _make_provider(
        self,
        cases: list[_FakeBoundKernel],
        *,
        aggregation: MultiShapeAggregation = "geomean",
        relative_to: MultiShapeReference = None,
        references: tuple[float, ...] | None = None,
        carrier: _MultiShapeAutotuneArgs | None = None,
    ) -> tuple[MultiShapeBenchmarkProvider, AutotuneMetrics]:
        if carrier is None:
            carrier = _make_carrier(
                tuple((case, (index,)) for index, case in enumerate(cases)),
                aggregation=aggregation,
                relative_to=relative_to,
                reference_latencies=references,
            )
        metrics = AutotuneMetrics()
        with patch.object(
            benchmark_provider_module,
            "LocalBenchmarkProvider",
            _FakeLocalBenchmarkProvider,
        ):
            provider = MultiShapeBenchmarkProvider(
                kernel=cases[0],  # type: ignore[arg-type]
                settings=cases[0].settings,  # type: ignore[arg-type]
                config_spec=cases[0].config_spec,
                args=cast("Sequence[object]", carrier),
                log=_Log(),  # type: ignore[arg-type]
                autotune_metrics=metrics,
            )
        return provider, metrics

    def test_rectangular_aggregation_preserves_original_config_identity(self) -> None:
        spec = _make_config_spec()
        provider, metrics = self._make_provider(
            [
                _FakeBoundKernel(spec, [1.0, 8.0], compile_times=[0.1, None]),
                _FakeBoundKernel(spec, [4.0, 2.0], compile_times=[0.2, 0.4]),
            ]
        )
        configs = [Config(block_size=64), Config(block_sizes=[128])]

        results = provider.benchmark(configs, desc="joint")

        self.assertEqual([result.perf for result in results], [2.0, 4.0])
        self.assertEqual([result.compile_time for result in results], [0.2, 0.4])
        self.assertIs(results[0].config, configs[0])
        self.assertIs(results[1].config, configs[1])
        self.assertEqual(metrics.num_configs_tested, 2)
        self.assertTrue(provider.args.found_valid_config)
        children = cast("list[_FakeLocalBenchmarkProvider]", provider.children)
        self.assertTrue(
            all(child._autotune_metrics is not metrics for child in children)
        )
        debug_message = cast("_Log", provider.log).debug_messages[0]
        assert callable(debug_message)
        self.assertIn(
            "objective: geomean(latencies)=2.000000 ms",
            debug_message(),
        )
        for index, child in enumerate(children):
            received, desc = child.benchmark_calls[0]
            self.assertEqual(desc, f"joint shape {index + 1}")
            self.assertEqual(received[0].block_sizes, [64])
            self.assertIsNot(received[0], configs[0])

    def test_raw_and_relative_max_choose_different_configs(self) -> None:
        spec = _make_config_spec()
        configs = [Config(block_sizes=[64]), Config(block_sizes=[128])]

        def select(references: tuple[float, ...] | None) -> Config:
            provider, _ = self._make_provider(
                [
                    _FakeBoundKernel(spec, [1.0, 4.0]),
                    _FakeBoundKernel(spec, [100.0, 6.0]),
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
            return FiniteSearch._autotune(cast("object", search))  # type: ignore[arg-type]

        self.assertIs(select(None), configs[1])
        self.assertIs(select((1.0, 100.0)), configs[0])

    def test_relative_logging_reports_shape_rows_and_keeps_perf_ms_raw(self) -> None:
        spec = _make_config_spec()
        provider, _ = self._make_provider(
            [_FakeBoundKernel(spec, [4.0]), _FakeBoundKernel(spec, [8.0])],
            aggregation="max",
            relative_to="default",
            references=(2.0, 8.0),
        )
        provider.log = Mock()
        provider.log.register_config.return_value = "config-id"
        config = Config(block_sizes=[64])

        result = provider.benchmark([config])[0]

        self.assertEqual(result.perf, 2.0)
        entry = provider.log.record_autotune_entry.call_args.args[0]
        self.assertEqual(entry.perf_ms, 8.0)
        self.assertIsNot(entry.config, config)
        self.assertEqual(entry.config.num_warps, 4)
        debug_message = provider.log.debug.call_args.args[0]()
        self.assertIn("arg_sets[0]: latency=4.000000 ms", debug_message)
        self.assertIn("default reference=2.000000 ms", debug_message)
        self.assertIn("arg_sets[1]: latency=8.000000 ms", debug_message)
        self.assertIn(
            "objective: max(latency ratios vs default)=2.000000x", debug_message
        )

        materialized = _materialize_multi_shape_config(spec, config)
        self.assertEqual(provider.raw_latency(config), 8.0)
        summary = _format_selected_multi_shape_measurement(provider.args, materialized)
        assert summary is not None
        self.assertIn("Selected multi-shape config", summary)
        self.assertIn("ratio=2.000000x", summary)
        self.assertIn("ratio=1.000000x", summary)

    def test_relative_metrics_keep_millisecond_contract(self) -> None:
        spec = _make_config_spec()
        provider, _ = self._make_provider(
            [_FakeBoundKernel(spec, [4.0]), _FakeBoundKernel(spec, [8.0])],
            aggregation="max",
            relative_to="default",
            references=(2.0, 8.0),
        )
        config = Config(block_sizes=[64])
        result = provider.benchmark([config])[0]
        search = object.__new__(FiniteSearch)
        search.args = provider.args
        search.benchmark_provider = provider
        search.best_perf_so_far = result.perf
        search._autotune_metrics = AutotuneMetrics()

        search._finalize_autotune_metrics(config)
        self.assertEqual(search._autotune_metrics.best_perf_ms, 8.0)

        search._autotune_metrics = AutotuneMetrics()
        search._finalize_autotune_metrics(None)
        self.assertEqual(search._autotune_metrics.best_perf_ms, 0.0)

    def test_any_invalid_shape_discards_config_and_timeout_wins_status(self) -> None:
        spec = _make_config_spec()
        provider, _ = self._make_provider(
            [
                _FakeBoundKernel(
                    spec,
                    [1.0, math.inf],
                    statuses=["ok", "timeout"],
                    compile_times=[0.1, 0.3],
                ),
                _FakeBoundKernel(
                    spec,
                    [math.inf, math.inf],
                    statuses=["error", "error"],
                    compile_times=[0.2, 0.4],
                ),
            ]
        )

        results = provider.benchmark(
            [Config(block_sizes=[64]), Config(block_sizes=[128])]
        )

        self.assertEqual([result.perf for result in results], [math.inf, math.inf])
        self.assertEqual([result.status for result in results], ["error", "timeout"])
        self.assertFalse(provider.args.found_valid_config)
        summary = _format_selected_multi_shape_measurement(
            provider.args,
            _materialize_multi_shape_config(spec, Config(block_sizes=[64])),
        )
        assert summary is not None
        self.assertIn("objective: rejected", summary)

    def test_failure_metrics_count_config_union_once(self) -> None:
        spec = _make_config_spec()
        provider, metrics = self._make_provider(
            [
                _FakeBoundKernel(
                    spec,
                    [math.inf, math.inf, 3.0],
                    accuracy_failure_indices=[0, 1],
                    compile_failure_indices=[0],
                ),
                _FakeBoundKernel(
                    spec,
                    [math.inf, 2.0, math.inf],
                    accuracy_failure_indices=[0, 2],
                    compile_failure_indices=[0, 1],
                ),
            ]
        )

        provider.benchmark(
            [
                Config(block_sizes=[64]),
                Config(block_sizes=[128]),
                Config(block_sizes=[256]),
            ]
        )

        self.assertEqual(metrics.num_accuracy_failures, 3)
        self.assertEqual(metrics.num_compile_failures, 2)

    def test_invalid_anchor_materialization_discards_only_one_config(self) -> None:
        spec = _make_config_spec()
        provider, metrics = self._make_provider(
            [_FakeBoundKernel(spec, [2.0]), _FakeBoundKernel(spec, [8.0])]
        )

        results = provider.benchmark(
            [
                Config(block_size=32, block_sizes=[64]),
                Config(block_sizes=[128]),
            ]
        )

        self.assertEqual([result.perf for result in results], [math.inf, 4.0])
        self.assertEqual(metrics.num_configs_tested, 2)
        for child in provider.children:
            self.assertEqual(len(child.benchmark_calls[0][0]), 1)

    def test_skippable_batch_failure_becomes_infinite_rows(self) -> None:
        spec = _make_config_spec()
        provider, _ = self._make_provider(
            [
                _FakeBoundKernel(spec, [1.0, 2.0]),
                _FakeBoundKernel(spec, [1.0, 2.0], error=RuntimeError("compile")),
            ]
        )
        with patch.object(provider, "_is_skippable_child_failure", return_value=True):
            results = provider.benchmark(
                [Config(block_sizes=[64]), Config(block_sizes=[128])]
            )
        self.assertEqual([result.perf for result in results], [math.inf, math.inf])

    def test_fatal_batch_failure_propagates(self) -> None:
        spec = _make_config_spec()
        provider, _ = self._make_provider(
            [
                _FakeBoundKernel(spec, [1.0]),
                _FakeBoundKernel(spec, [1.0], error=RuntimeError("fatal")),
            ]
        )
        with (
            patch.object(provider, "_is_skippable_child_failure", return_value=False),
            self.assertRaisesRegex(RuntimeError, "fatal"),
        ):
            provider.benchmark([Config(block_sizes=[64])])

    def test_warn_classified_batch_failure_is_not_hidden(self) -> None:
        child = SimpleNamespace(
            config_spec=SimpleNamespace(
                backend=SimpleNamespace(
                    classify_autotune_exception=lambda error: "warn"
                )
            ),
            settings=SimpleNamespace(autotune_ignore_errors=False),
        )

        self.assertFalse(
            MultiShapeBenchmarkProvider._is_skippable_child_failure(
                cast("object", child), RuntimeError("compiler bug")
            )
        )

    def test_reference_timings_are_reused_across_search_providers(self) -> None:
        spec = _make_config_spec()
        cases = [_FakeBoundKernel(spec, [2.0]), _FakeBoundKernel(spec, [8.0])]
        provider, _ = self._make_provider(
            cases,
            relative_to="default",
        )

        provider.setup()
        later_provider, _ = self._make_provider(cases, carrier=provider.args)
        later_provider.setup()

        self.assertEqual(provider.args.reference_latencies, (2.0, 8.0))
        first_children = cast("list[_FakeLocalBenchmarkProvider]", provider.children)
        later_children = cast(
            "list[_FakeLocalBenchmarkProvider]", later_provider.children
        )
        self.assertEqual([child.setup_count for child in later_children], [1, 1])
        self.assertEqual(
            [len(child.benchmark_calls) for child in first_children], [1, 1]
        )
        self.assertTrue(all(not child.benchmark_calls for child in later_children))

    def test_reference_failure_names_shape_and_cleans_up(self) -> None:
        spec = _make_config_spec()
        provider, _ = self._make_provider(
            [
                _FakeBoundKernel(spec, [2.0]),
                _FakeBoundKernel(spec, [math.inf], statuses=["error"]),
            ],
            relative_to="default",
        )

        with self.assertRaisesRegex(
            helion.exc.AutotuneError, r"reference benchmark failed for arg_sets\[1\]"
        ):
            provider.setup()

        children = cast("list[_FakeLocalBenchmarkProvider]", provider.children)
        self.assertEqual([child.cleanup_count for child in children], [1, 1])

    def test_rebenchmark_reruns_joint_pipeline_and_records_latest_failure(self) -> None:
        spec = _make_config_spec()
        first = _FakeBoundKernel(spec, [1.0, 4.0])
        second = _FakeBoundKernel(spec, [9.0, 1.0])
        provider, metrics = self._make_provider([first, second])
        configs = [Config(block_sizes=[64]), Config(block_sizes=[128])]
        initial = provider.benchmark(configs)
        tested = metrics.num_configs_tested
        second.timings = [16.0, math.inf]
        second.statuses = ["ok", "error"]

        timings = provider.rebenchmark(
            configs,
            [result.perf for result in initial],
            desc="confirm",
        )

        self.assertEqual(timings, [4.0, math.inf])
        self.assertEqual(metrics.num_configs_tested, tested)
        first_summary = _format_selected_multi_shape_measurement(
            provider.args,
            _materialize_multi_shape_config(spec, configs[0]),
        )
        second_summary = _format_selected_multi_shape_measurement(
            provider.args,
            _materialize_multi_shape_config(spec, configs[1]),
        )
        assert first_summary is not None
        assert second_summary is not None
        self.assertIn("latency=16.000000 ms", first_summary)
        self.assertIn("objective: rejected", second_summary)
        self.assertTrue(provider.has_valid_measurement(configs[0]))
        self.assertFalse(provider.has_valid_measurement(configs[1]))
        for child in provider.children:
            self.assertEqual(len(child.benchmark_calls), 2)

    def test_rebenchmark_preserves_previous_values_after_budget(self) -> None:
        spec = _make_config_spec()
        provider, _ = self._make_provider(
            [_FakeBoundKernel(spec, [1.0, 2.0]), _FakeBoundKernel(spec, [3.0, 4.0])]
        )
        provider.set_budget_exceeded_fn(lambda: True)
        previous = [1.5, 2.5]

        self.assertEqual(
            provider.rebenchmark(
                [Config(block_sizes=[64]), Config(block_sizes=[128])],
                previous,
                desc="confirm",
            ),
            previous,
        )
        self.assertTrue(all(not child.benchmark_calls for child in provider.children))

    def test_completed_rebenchmark_failure_wins_over_crossed_budget(self) -> None:
        spec = _make_config_spec()
        provider, _ = self._make_provider(
            [
                _FakeBoundKernel(spec, [math.inf], statuses=["error"]),
                _FakeBoundKernel(spec, [1.0]),
            ]
        )
        budget_exceeded = Mock(side_effect=(False, True))
        provider.set_budget_exceeded_fn(budget_exceeded)

        timings = provider.rebenchmark(
            [Config(block_sizes=[64])],
            [2.0],
            desc="confirm",
        )

        self.assertEqual(timings, [math.inf])
        budget_exceeded.assert_called_once_with()

    def test_population_search_uses_provider_rebenchmark_hook(self) -> None:
        provider = object.__new__(MultiShapeBenchmarkProvider)
        provider.rebenchmark = Mock()  # type: ignore[method-assign]
        provider.rebenchmark.return_value = [3.0, 1.0]
        configs = [Config(block_sizes=[64]), Config(block_sizes=[128])]
        members = [
            PopulationMember(lambda: None, [2.0], [], config) for config in configs
        ]
        search = SimpleNamespace(
            benchmark_provider=provider,
            best_perf_so_far=2.0,
        )

        PopulationBasedSearch.rebenchmark(cast("object", search), members, desc="again")  # type: ignore[arg-type]

        self.assertEqual([member.perfs for member in members], [[2.0, 3.0], [2.0, 1.0]])
        self.assertEqual(search.best_perf_so_far, 1.0)
        provider.rebenchmark.assert_called_once_with(configs, [2.0, 2.0], desc="again")

    def test_local_provider_rejects_carrier_directly(self) -> None:
        carrier = _make_carrier(((object(), (object(),)),))
        with self.assertRaisesRegex(TypeError, "cannot benchmark multi-shape"):
            LocalBenchmarkProvider(
                kernel=cast("object", object()),
                settings=cast("object", object()),
                config_spec=cast("object", object()),
                args=cast("Sequence[object]", carrier),
                log=cast("object", object()),
                autotune_metrics=AutotuneMetrics(),
            )


class TestMultiShapeSearchOrchestration(unittest.TestCase):
    @staticmethod
    def _make_base_search(args: object) -> BaseSearch:
        search = BaseSearch.__new__(BaseSearch)
        search.args = args  # pyrefly: ignore [bad-assignment]
        search.settings = Settings(autotune_log=False, autotune_log_details=False)
        search.log = Mock()
        search.best_perf_so_far = math.inf
        search._autotune_metrics = AutotuneMetrics()
        search.config_spec = Mock()
        search.config_spec.default_config.return_value = Config(num_warps=4)
        search.kernel = Mock()
        search.kernel.format_kernel_decorator.return_value = "@helion.kernel(...)"
        search.kernel.get_cached_path.return_value = None
        search.benchmark_provider = SimpleNamespace(  # pyrefly: ignore [bad-assignment]
            setup=Mock(), cleanup=Mock()
        )
        search._prepare = Mock()  # type: ignore[method-assign]
        search._autotune = Mock(  # type: ignore[method-assign]
            side_effect=exc.NoConfigFound
        )
        search._finalize_autotune_metrics = Mock()  # type: ignore[method-assign]
        return search

    def test_finite_search_rejects_all_nonfinite_configs(self) -> None:
        configs = [Config(num_warps=2), Config(num_warps=4)]
        results = [
            BenchmarkResult(config, lambda: None, perf, "error", None)
            for config, perf in zip(configs, (math.inf, math.nan), strict=True)
        ]
        search = SimpleNamespace(
            args=_make_carrier(((object(), ()),)),
            configs=configs,
            benchmark_batch=lambda configs, *, desc: results,
        )

        with self.assertRaises(exc.NoConfigFound):
            FiniteSearch._autotune(cast("object", search))  # type: ignore[arg-type]

    def test_base_search_propagates_no_config_and_cleans_up(self) -> None:
        search = self._make_base_search(_make_carrier(((object(), ()),)))

        with self.assertRaises(exc.NoConfigFound):
            BaseSearch.autotune(search)

        search.benchmark_provider.cleanup.assert_called_once_with()

    def test_final_pick_resolution_handles_partial_and_multi_shape_searches(
        self,
    ) -> None:
        partial_search = object.__new__(PopulationBasedSearch)
        self.assertIsNone(partial_search._resolve_device_micros_paired_bench())

        search = object.__new__(PopulationBasedSearch)
        search.benchmark_provider = object.__new__(MultiShapeBenchmarkProvider)
        search.settings = SimpleNamespace(static_shapes=True)
        search.config_spec = SimpleNamespace(backend=Mock())

        self.assertIsNone(search._resolve_device_micros_paired_bench())
        search.config_spec.backend.get_paired_device_micros_bench.assert_not_called()


class TestMultiShapeCacheBoundary(unittest.TestCase):
    @staticmethod
    def _make_cache(carrier: _MultiShapeAutotuneArgs) -> LocalAutotuneCache:
        cache = object.__new__(LocalAutotuneCache)
        cache.args = carrier  # pyrefly: ignore [bad-assignment]
        cache.autotuner = SimpleNamespace(  # pyrefly: ignore [bad-assignment]
            log=Mock(),
            config_spec=_make_config_spec(),
            settings=SimpleNamespace(autotune_log=None),
        )
        cache._run_autotune_trials = Mock(  # type: ignore[method-assign]
            return_value=Config(block_size=64)
        )
        cache.put = Mock()  # type: ignore[method-assign]
        return cache

    def test_all_invalid_run_raises_before_cache_put(self) -> None:
        carrier = _make_carrier(((object(), ()),))
        carrier.search_started = True
        cache = self._make_cache(carrier)

        with self.assertRaises(exc.NoConfigFound):
            cache.autotune(skip_cache=True)

        cache.put.assert_not_called()

    @patch("helion.autotuner.base_cache.should_skip_cache", return_value=False)
    def test_valid_run_materializes_winner_before_cache_put(
        self, skip_cache: object
    ) -> None:
        carrier = _make_carrier(((object(), ()),))
        carrier.search_started = True
        carrier.found_valid_config = True
        cache = self._make_cache(carrier)
        materialized = _materialize_multi_shape_config(
            cache.autotuner.config_spec,
            Config(block_size=64),
        )
        carrier.measurements[repr(materialized)] = ((1.0,), 1.0, ("ok",))

        result = cache.autotune(skip_cache=True)

        self.assertEqual(result.block_sizes, [64])
        self.assertNotIn("block_size", result.config)
        cache.put.assert_called_once_with(result)


class _TrialLog:
    def __call__(self, message: str, *args: object, **kwargs: object) -> None:
        pass


class TestMultiShapeBestOfK(unittest.TestCase):
    @staticmethod
    def _make_cache(
        outcomes: list[tuple[Config, float] | type[exc.NoConfigFound]],
    ) -> LocalAutotuneCache:
        settings = SimpleNamespace(
            autotune_best_of_k=len(outcomes),
            autotune_random_seed=20,
            autotune_compile_timeout=60,
        )
        outcome_iter = iter(outcomes)

        class _Search:
            def __init__(self) -> None:
                self.settings = settings
                self.log = _TrialLog()
                self.best_perf_so_far = math.inf

            def autotune(self) -> Config:
                outcome = next(outcome_iter)
                if outcome is exc.NoConfigFound:
                    raise exc.NoConfigFound
                config, perf = outcome
                self.best_perf_so_far = perf
                return config

        cache = object.__new__(LocalAutotuneCache)
        cache.args = _make_carrier(  # pyrefly: ignore [bad-assignment]
            ((object(), ()),)
        )
        cache.autotuner = _Search()  # pyrefly: ignore [bad-assignment]
        cache._autotuner_factory = _Search
        cache._release_trial_state = Mock()  # type: ignore[method-assign]
        return cache

    def test_failed_trials_are_skipped(self) -> None:
        invalid = Config(num_warps=2)
        winner = Config(num_warps=4)
        cache = self._make_cache(
            [exc.NoConfigFound, (invalid, math.inf), (winner, 1.0)]
        )
        cache._rebench_trial_configs = Mock(  # type: ignore[method-assign]
            return_value=[0.8]
        )

        self.assertEqual(cache._run_autotune_trials(), winner)
        cache._rebench_trial_configs.assert_called_once_with([winner])

    def test_all_failed_trials_raise_before_rebenchmark(self) -> None:
        cache = self._make_cache([exc.NoConfigFound, exc.NoConfigFound])
        cache._rebench_trial_configs = Mock()  # type: ignore[method-assign]

        with self.assertRaises(exc.NoConfigFound):
            cache._run_autotune_trials()
        cache._rebench_trial_configs.assert_not_called()

    def test_all_failed_final_rebenchmarks_raise(self) -> None:
        cache = self._make_cache(
            [(Config(num_warps=2), 1.0), (Config(num_warps=4), 2.0)]
        )
        cache._rebench_trial_configs = Mock(  # type: ignore[method-assign]
            return_value=[math.inf, math.inf]
        )

        with self.assertRaises(exc.NoConfigFound):
            cache._run_autotune_trials()


class _StageSearch:
    def __init__(self, config: Config | None, perf: float, tested: int) -> None:
        self.config = config
        self.best_perf_so_far = perf
        self._autotune_metrics = AutotuneMetrics(num_configs_tested=tested)

    def autotune(self, *, skip_cache: bool = False) -> Config:
        if self.config is None:
            raise exc.NoConfigFound
        return self.config


class TestMultiShapeLLMSeeded(unittest.TestCase):
    @staticmethod
    def _make_search(
        llm_stage: _StageSearch, second_stage: _StageSearch
    ) -> tuple[LLMSeededSearch, list[bool]]:
        kernel = SimpleNamespace(settings=Settings(), config_spec=SimpleNamespace())
        search = LLMSeededSearch(
            kernel,
            _make_carrier(((object(), ()),)),
            llm_max_rounds=1,
            second_stage_algorithm="FiniteSearch",
        )
        search._autotune_metrics = AutotuneMetrics()
        search._make_llm_search = lambda: llm_stage  # type: ignore[method-assign,return-value]
        seeded_values: list[bool] = []

        def make_second_stage(*, seeded: bool) -> _StageSearch:
            seeded_values.append(seeded)
            return second_stage

        search._make_second_stage_search = make_second_stage  # type: ignore[method-assign,return-value]
        return search, seeded_values

    def test_failed_llm_stage_runs_second_stage_unseeded(self) -> None:
        winner = Config(num_warps=4)
        search, seeded = self._make_search(
            _StageSearch(None, math.inf, 2),
            _StageSearch(winner, 0.7, 3),
        )

        self.assertEqual(search._autotune(), winner)
        self.assertEqual(seeded, [False])
        self.assertFalse(search.hybrid_stage_breakdown["used_llm_seed"])

    def test_invalid_second_stage_returns_valid_llm_seed(self) -> None:
        winner = Config(num_warps=4)
        invalid_second_stages = (
            _StageSearch(None, math.inf, 3),
            _StageSearch(Config(num_warps=8), math.inf, 3),
        )

        for second_stage in invalid_second_stages:
            with self.subTest(config=second_stage.config):
                search, seeded = self._make_search(
                    _StageSearch(winner, 0.6, 2), second_stage
                )

                self.assertEqual(search._autotune(), winner)
                self.assertEqual(seeded, [True])
                self.assertEqual(search.best_perf_so_far, 0.6)

    def test_both_failed_stages_raise_after_metrics(self) -> None:
        search, seeded = self._make_search(
            _StageSearch(None, math.inf, 2),
            _StageSearch(None, math.inf, 3),
        )

        with self.assertRaises(exc.NoConfigFound):
            search._autotune()
        self.assertEqual(seeded, [False])
        self.assertEqual(search._autotune_metrics.num_configs_tested, 5)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
class TestMultiShapeAutotuneIntegration(unittest.TestCase):
    @staticmethod
    def _make_add_case(
        *, best_of_k: int | None = None, log_path: str | None = None
    ) -> tuple[Kernel, list[tuple[torch.Tensor, torch.Tensor]]]:
        settings: dict[str, object] = {}
        if best_of_k is not None:
            settings["autotune_best_of_k"] = best_of_k
        if log_path is not None:
            settings["autotune_log"] = log_path

        @helion.kernel(
            configs=[
                helion.Config(block_sizes=[64], num_warps=4),
                helion.Config(block_sizes=[128], num_warps=4),
            ],
            autotune_benchmark_subprocess=False,
            autotune_precompile=None,
            **settings,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(a.size()):
                out[tile] = a[tile] + b[tile]
            return out

        arg_sets = [
            (torch.randn(256, device=DEVICE), torch.randn(256, device=DEVICE)),
            (torch.randn(2048, device=DEVICE), torch.randn(2048, device=DEVICE)),
        ]
        return add, arg_sets

    def test_finite_search_installs_one_config_for_two_shapes(self) -> None:
        add, arg_sets = self._make_add_case()

        winner = add.autotune_multi(
            arg_sets,
            aggregation="max",
            relative_to="default",
            force=False,
        )

        self.assertIn(winner.block_sizes[0], (64, 128))
        for args in arg_sets:
            torch.testing.assert_close(add(*args), args[0] + args[1])

    def test_cache_best_of_k_persists_selected_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "multi_shape"
            add, arg_sets = self._make_add_case(
                best_of_k=2,
                log_path=str(log_path),
            )
            with patch.dict(
                os.environ,
                {
                    "HELION_AUTOTUNER": "FiniteSearch",
                    "HELION_SKIP_CACHE": "1",
                },
            ):
                add.autotune_multi(
                    arg_sets,
                    aggregation="max",
                    relative_to="default",
                )

            log_text = log_path.with_suffix(".log").read_text()
            self.assertIn("Selected multi-shape config", log_text)
            self.assertIn("arg_sets[0]", log_text)
            self.assertIn("arg_sets[1]", log_text)
            self.assertIn("objective: max(latency ratios vs default)=", log_text)


if __name__ == "__main__":
    unittest.main()
