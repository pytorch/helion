"""The search facts an autotune surfaces: config values no candidate could be
measured with, and the role of a config the benchmark timeout dropped."""

from __future__ import annotations

import logging
import math
import operator
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfRefEager
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.benchmark_provider import BenchmarkResult
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmark_provider import UnmeasuredConfigValue
from helion.autotuner.benchmark_provider import _categorical_config_values
from helion.autotuner.benchmark_provider import _unset_fn
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
import helion.language as hl
from helion.runtime.settings import Settings


def _provider() -> LocalBenchmarkProvider:
    provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
    provider.settings = Settings(
        autotune_log_level=logging.CRITICAL, autotune_benchmark_timeout=30
    )
    provider.log = Mock()
    provider._autotune_metrics = AutotuneMetrics()
    provider._compile_failure_config_ids = []
    provider._compiler_seed_configs = set()
    return provider


def _measured(*configs: helion.Config) -> list[BenchmarkResult]:
    return [
        BenchmarkResult(config, _unset_fn, 1.0 + index, "ok", None, 1.0)
        for index, config in enumerate(configs)
    ]


def _make_bound(
    **settings: object,
) -> tuple[object, tuple[torch.Tensor, torch.Tensor]]:
    kwargs: dict[str, object] = {
        "autotune_log_level": logging.CRITICAL,
        "autotune_benchmark_subprocess": False,
        "autotune_baseline_fn": operator.add,
        "autotune_random_seed": 1234,
        "autotune_progress_bar": False,
        **settings,
    }

    @helion.kernel(**kwargs)  # pyrefly: ignore[bad-argument-type]
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(a)
        for tile in hl.tile(out.size()):
            out[tile] = a[tile] + b[tile]
        return out

    args = (
        torch.randn([1024], device=DEVICE),
        torch.randn([1024], device=DEVICE),
    )
    return add.bind(args), args


def _flip_one_categorical_value(
    config: helion.Config,
) -> tuple[helion.Config, tuple[str, object]]:
    """A copy of ``config`` with one categorical knob changed (a boolean
    inverted, else a string renamed), and the ``(key, value)`` pair the copy
    now carries."""
    values = dict(config.config)
    for key, value in values.items():
        if isinstance(value, bool):
            values[key] = not value
            return helion.Config(**values), (key, not value)
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, bool):
                    flipped = list(value)
                    flipped[index] = not item
                    values[key] = flipped
                    return helion.Config(**values), (f"{key}[{index}]", not item)
    for key, value in values.items():
        if isinstance(value, str):
            values[key] = value + "_flipped"
            return helion.Config(**values), (key, values[key])
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, str):
                    flipped = list(value)
                    flipped[index] = item + "_flipped"
                    values[key] = flipped
                    return helion.Config(**values), (f"{key}[{index}]", flipped[index])
    raise AssertionError(f"no categorical knob to flip in {config!r}")


class TestUnmeasuredConfigValues(TestCase):
    def test_categorical_values_expand_lists_and_skip_numbers(self) -> None:
        config = helion.Config(
            block_sizes=[64, 32],
            cute_collective_mma=True,
            cute_collective_compute="warp",
            cute_lane_layouts=["blocked", "strided"],
            range_flattens=[None, True],
            num_warps=4,
        )
        self.assertEqual(
            sorted(_categorical_config_values(config)),
            [
                ("cute_collective_compute", "warp"),
                ("cute_collective_mma", True),
                ("cute_lane_layouts[0]", "blocked"),
                ("cute_lane_layouts[1]", "strided"),
                ("range_flattens[1]", True),
            ],
        )

    def test_values_every_failing_config_carried_are_reported(self) -> None:
        provider = _provider()
        error = helion.exc.BackendUnsupported(
            "cute", "tile shape [64, 64] binds block id 2 to two axes of one tensor"
        )
        reason = (
            "BackendUnsupported: Backend 'cute' does not support: tile shape "
            "[64, 64] binds block id 2 to two axes of one tensor"
        )
        # Every tensor-core config fails; the SIMT ones measure fine.
        for block in (64, 128):
            provider._record_compile_failure(
                helion.Config(
                    block_sizes=[block],
                    cute_collective_mma=True,
                    cute_collective_compute="tcgen05",
                    cute_collective_epilogue="scalar",
                ),
                error,
            )
        provider._record_compile_failure(
            helion.Config(
                block_sizes=[64],
                cute_collective_mma=True,
                cute_collective_compute="warp",
                cute_collective_epilogue="vector",
            ),
            RuntimeError("second failure: a different reason\nwith a traceback"),
        )
        provider._record_measured_configs(
            [
                *_measured(
                    *(
                        helion.Config(
                            block_sizes=[64],
                            cute_collective_mma=False,
                            cute_collective_compute="warp",
                            cute_collective_epilogue=epilogue,
                        )
                        for epilogue in ("scalar", "vector")
                    )
                ),
                # A failed result carries nothing.
                BenchmarkResult(
                    helion.Config(block_sizes=[64], cute_collective_mma=True),
                    _unset_fn,
                    math.inf,
                    "error",
                    None,
                    1.0,
                ),
            ]
        )
        facts = provider.unmeasured_config_values()
        self.assertEqual(
            facts,
            [
                UnmeasuredConfigValue("cute_collective_mma", True, 3, reason),
                UnmeasuredConfigValue("cute_collective_compute", "tcgen05", 2, reason),
            ],
        )
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 3)
        # A failure without an error object only counts.
        provider._record_compile_failure(
            helion.Config(block_sizes=[64], cute_collective_mma=True)
        )
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 4)
        self.assertEqual(provider.unmeasured_config_values()[0].count, 3)

    def test_first_error_is_one_bounded_line(self) -> None:
        provider = _provider()
        provider._record_compile_failure(
            helion.Config(cute_collective_mma=True), RuntimeError("x" * 500)
        )
        (fact,) = provider.unmeasured_config_values()
        self.assertEqual(len(fact.first_error), 240)
        self.assertTrue(fact.first_error.endswith("..."))

    def test_multi_shape_provider_aggregates_its_children(self) -> None:
        # A value measured on any shape was reachable; the shapes' failure
        # counts add up and the first shape's first error is kept.
        first, second = _provider(), _provider()
        first._record_compile_failure(
            helion.Config(cute_collective_mma=True, cute_pdl=True),
            RuntimeError("first shape"),
        )
        second._record_compile_failure(
            helion.Config(cute_collective_mma=True, cute_pdl=False),
            RuntimeError("second shape"),
        )
        second._record_measured_configs(
            _measured(helion.Config(cute_collective_mma=False, cute_pdl=True))
        )
        provider = MultiShapeBenchmarkProvider.__new__(MultiShapeBenchmarkProvider)
        provider.children = [first, second]
        self.assertEqual(
            provider.unmeasured_config_values(),
            [
                UnmeasuredConfigValue(
                    "cute_collective_mma", True, 2, "RuntimeError: first shape"
                ),
                UnmeasuredConfigValue(
                    "cute_pdl", False, 1, "RuntimeError: second shape"
                ),
            ],
        )
        roles = {helion.Config(block_sizes=[64]): "default config"}
        provider.set_config_roles(roles)
        for child in provider.children:
            self.assertEqual(child._config_roles, roles)

    def test_search_logs_the_facts_after_the_compile_failure_count(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = BaseSearch.__new__(BaseSearch)
        search.log = Mock(spec=AutotuningLogger(settings))
        facts = [
            UnmeasuredConfigValue(
                f"knob_{index}", True, 10 - index, f"BackendUnsupported: reason {index}"
            )
            for index in range(10)
        ]
        search.benchmark_provider = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
            unmeasured_config_values=lambda: facts
        )
        search._log_unmeasured_config_values()
        search.log.assert_called_once()
        (message,) = search.log.call_args.args
        self.assertIn("knob_0=True: 10 config(s) failed to compile", message)
        self.assertIn("first: BackendUnsupported: reason 0", message)
        self.assertIn("knob_7=True", message)
        self.assertNotIn("knob_8=True", message)
        self.assertIn("... and 2 more", message)

        search.log.reset_mock()
        search.benchmark_provider = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
            unmeasured_config_values=list
        )
        search._log_unmeasured_config_values()
        search.log.assert_not_called()


class TestTimedOutConfigRole(TestCase):
    def test_default_and_seed_timeouts_are_named(self) -> None:
        default = helion.Config(block_sizes=[64], num_warps=4)
        seed = helion.Config(block_sizes=[128], num_warps=8)
        retry_seed = helion.Config(block_sizes=[256], num_warps=8)
        provider = _provider()
        provider.set_config_roles({default: "default config", seed: "compiler seed"})
        provider._compiler_seed_configs = {retry_seed}
        self.assertEqual(
            provider._timed_out_config_role(
                helion.Config(block_sizes=[64], num_warps=4)
            ),
            "default config",
        )
        self.assertEqual(provider._timed_out_config_role(seed), "compiler seed")
        self.assertEqual(provider._timed_out_config_role(retry_seed), "compiler seed")
        self.assertIsNone(
            provider._timed_out_config_role(helion.Config(block_sizes=[32]))
        )

        error = BenchmarkTimeout("benchmark timeout after 30.0s")
        provider._warn_timed_out_role(helion.Config(block_sizes=[32]), error)
        provider.log.warning.assert_not_called()  # pyrefly: ignore[missing-attribute]
        provider._warn_timed_out_role(default, error)
        provider.log.warning.assert_called_once()  # pyrefly: ignore[missing-attribute]
        (message,) = provider.log.warning.call_args.args  # pyrefly: ignore[missing-attribute]
        self.assertIn(
            "default config timed out in the benchmark worker "
            "(benchmark timeout after 30.0s",
            message,
        )
        self.assertIn("HELION_AUTOTUNE_BENCHMARK_TIMEOUT", message)

    def test_hand_built_provider_has_no_roles(self) -> None:
        # Providers built without __init__ (the worker tests) have no default
        # or seed configs on record: nothing is named, nothing raises.
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.log = Mock()
        self.assertIsNone(provider._timed_out_config_role(helion.Config()))
        self.assertIsNone(provider._timed_out_config_role("not a config"))  # pyrefly: ignore[bad-argument-type]
        provider._warn_timed_out_role(helion.Config(), BenchmarkTimeout("t"))
        provider.log.warning.assert_not_called()  # pyrefly: ignore[missing-attribute]
        self.assertEqual(provider.unmeasured_config_values(), [])


class TestSearchFactsFromARealSearch(TestCase):
    """The bookkeeping fed by a real provider's benchmark batch and a real
    search's preparation, not by hand."""

    @skipIfRefEager("the search bookkeeping is not exercised in interpret mode")
    def test_measured_values_come_from_the_benchmark_batch(self) -> None:
        bound, args = _make_bound()
        search = PatternSearch(bound, args, initial_population=4, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        gen = search.config_gen
        default = gen.unflatten(gen.default_flat())
        if not _categorical_config_values(default):
            self.skipTest("this backend's default config has no categorical knobs")
        # Differs from the default in a numeric knob only, so the batch's
        # measured categorical values are exactly the default's.
        other = helion.Config(
            **{
                **default.config,
                "num_warps": 4 if default.config.get("num_warps") == 8 else 8,
            }
        )
        self.assertIsNone(provider._measured_config_values)
        with patch.object(
            LocalBenchmarkProvider, "_benchmark_function", return_value=1.0
        ):
            results = provider.benchmark([default, other])
        self.assertTrue(all(math.isfinite(result.perf) for result in results))
        measured = provider._measured_config_values
        assert measured is not None
        for config in (default, other):
            for pair in _categorical_config_values(config):
                self.assertIn(pair, measured)
        # A compile failure carrying only measured values is not a fact: the
        # selected config does have those values ...
        provider._record_compile_failure(default, RuntimeError("fake"))
        self.assertEqual(provider.unmeasured_config_values(), [])
        # ... while a value no measured candidate carried is one.
        flipped, pair = _flip_one_categorical_value(default)
        provider._record_compile_failure(flipped, RuntimeError("fake"))
        self.assertEqual(
            provider.unmeasured_config_values(),
            [UnmeasuredConfigValue(pair[0], pair[1], 1, "RuntimeError: fake")],
        )

    @skipIfRefEager("the search bookkeeping is not exercised in interpret mode")
    def test_search_registers_the_default_it_benchmarks(self) -> None:
        # The search benchmarks the fragment default (``default_flat``), which
        # differs from the execution default whenever a knob is overridden or
        # a seed is promoted; the role follows the config the search uses.
        plain, _args = _make_bound()
        plain_default = plain.config_spec.default_config()  # pyrefly: ignore[missing-attribute]
        if "num_warps" not in plain_default.config:
            self.skipTest("this backend's config spec has no num_warps knob")
        num_warps = 4 if plain_default.config["num_warps"] == 8 else 8
        bound, args = _make_bound(autotune_config_overrides={"num_warps": num_warps})
        search = PatternSearch(bound, args, initial_population=4, copies=1)
        search._prepare()
        provider = search.benchmark_provider
        assert isinstance(provider, LocalBenchmarkProvider)
        gen = search.config_gen
        _flat, search_default = gen.canonicalize_flat(gen.default_flat())
        execution_default = bound.config_spec.default_config()  # pyrefly: ignore[missing-attribute]
        self.assertNotEqual(search_default, execution_default)
        self.assertEqual(
            provider._timed_out_config_role(search_default), "default config"
        )
        for _flat, seed in gen.seed_flat_config_pairs():
            self.assertEqual(provider._timed_out_config_role(seed), "compiler seed")
        roles = search._search_config_roles()
        self.assertEqual(roles[search_default], "default config")
        stranger = next(
            config
            for config in (gen.unflatten(gen.random_flat()) for _ in range(64))
            if config not in roles
        )
        self.assertIsNone(provider._timed_out_config_role(stranger))
        provider.log = Mock()
        provider._warn_timed_out_role(
            search_default, BenchmarkTimeout("benchmark timeout after 30.0s")
        )
        provider.log.warning.assert_called_once()  # pyrefly: ignore[missing-attribute]


def test_cute_backend_probes_long_kernels_before_timing() -> None:
    pytest.importorskip("cutlass")
    from helion._compiler.cute.backend import CuteBackend

    backend = CuteBackend.__new__(CuteBackend)
    assert backend.probe_long_autotune_kernels(SimpleNamespace()) is True  # pyrefly: ignore[bad-argument-type]
