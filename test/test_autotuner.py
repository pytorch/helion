from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import copy
import csv
from dataclasses import replace
import functools
import inspect
import json
import logging
import math
import multiprocessing as mp
import operator
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
from types import SimpleNamespace
from typing import Callable
from typing import ClassVar
from typing import Sequence
import unittest
from unittest import skip
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion import _compat
from helion import _hardware
from helion import exc
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion._compiler.cute.cute_flash import _flash_clc_heads_per_batch_candidates
from helion._compiler.cute.cute_flash import (
    _flash_clc_heads_per_batch_coverage_candidates,
)
from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
from helion._compiler.tile_dispatch import BlockIDStrategyMapping
from helion._compiler.tile_dispatch import TileStrategyDispatch
from helion._hardware import HardwareInfo
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import assert_close_with_mismatch_tolerance
from helion._testing import get_test_float32_matmul_precision
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing import skipIfXPU
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner import DESurrogateHybrid
from helion.autotuner import DifferentialEvolutionSearch
from helion.autotuner import LFBOPatternSearch
from helion.autotuner import LFBOTreeSearch
from helion.autotuner import LLMGuidedSearch
from helion.autotuner import LLMSeededLFBOTreeSearch
from helion.autotuner import LLMSeededSearch
from helion.autotuner import PatternSearch
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.benchmark_provider import _compile_config_failure_source_hash
from helion.autotuner.benchmark_provider import _MultiShapeAutotuneArgs
from helion.autotuner.benchmarking import MirroredBenchmarkTrace
from helion.autotuner.benchmarking import _mirrored_bench_call_layout
from helion.autotuner.config_fragment import BlockSizeFragment
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import NumThreadsFragment
from helion.autotuner.config_fragment import NumWarpsFragment
from helion.autotuner.config_fragment import PermutationFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_generation import CoordinateNeighborProjection
from helion.autotuner.config_generation import _flash_log_maximin_refinements
from helion.autotuner.config_spec import SMALL_DIM_BLOCK_SIZE_OVERSHOOT
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.effort_profile import AutotuneEffortProfile
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import StrictLocalAutotuneCache
from helion.autotuner.local_cache import _cute_flash_search_policy_hash
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import KernelMetadata
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.autotuner.random_search import RandomSearch
from helion.autotuner.search_space_logger import canonical_config_id
from helion.autotuner.surrogate_pattern_search import (
    flash_terminal_measurement_is_valid,
)
from helion.autotuner.surrogate_pattern_search import (
    flash_terminal_refinement_result_is_valid,
)
import helion.language as hl
from helion.language import loops
from helion.runtime.settings import Settings
from helion.runtime.settings import _get_backend
from helion.runtime.settings import default_autotuner_fn

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")

_CACHE_POLICY_BASELINE_SCALE = 1
_CACHE_POLICY_DYNAMIC_HELPER: object = None
_TEST_CUTE_FLASH_BACKEND = SimpleNamespace(
    generated_source_hash=lambda _fn: None,
)


def _cute_flash_test_config_spec() -> SimpleNamespace:
    return SimpleNamespace(
        cute_flash_search_enabled=True,
        backend=_TEST_CUTE_FLASH_BACKEND,
    )


# Pin the arch for config-population goldens: the pointwise seed promotes to the
# autotune-off default only on PROMOTE_TARGETS (sm90/sm100), so an unpinned test
# would emit the promoted tile on H100/B200 runners and the base default on
# others. Fixing the arch keeps the expected journal deterministic everywhere.
_HOPPER_HARDWARE = HardwareInfo(
    device_kind="cuda",
    hardware_name="NVIDIA H100",
    runtime_version="12.8",
    compute_capability="sm90",
)
examples_dir = Path(__file__).parent.parent / "examples"


_FINAL_REBENCHMARK_ENV_KEYS = (
    "HELION_CAP_REBENCHMARK_REPEAT",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE",
)


def _phase_referenced_config_ids(value: object, field: str | None = None) -> set[str]:
    if field == "config_manifest":
        return set()
    if field is not None and (
        field == "config_id" or field.endswith(("_config_id", "_config_ids"))
    ):
        if isinstance(value, str):
            return {value}
        if isinstance(value, list):
            return {
                item
                for nested in value
                for item in _phase_referenced_config_ids(nested, field)
            }
        if isinstance(value, dict):
            return {
                item
                for nested in value.values()
                for item in _phase_referenced_config_ids(nested, field)
            }
        return set()
    if isinstance(value, list):
        return {
            item for nested in value for item in _phase_referenced_config_ids(nested)
        }
    if isinstance(value, dict):
        return {
            item
            for key, nested in value.items()
            for item in _phase_referenced_config_ids(nested, key)
        }
    return set()


def _assert_phase_config_manifest(
    testcase: unittest.TestCase,
    phase: dict[str, object],
) -> None:
    manifest = phase["config_manifest"]
    testcase.assertIsInstance(manifest, dict)
    assert isinstance(manifest, dict)
    referenced_ids = _phase_referenced_config_ids(phase)
    testcase.assertLessEqual(referenced_ids, set(manifest))
    testcase.assertEqual(len(manifest), len(set(manifest)))
    for config_id, raw_entry in manifest.items():
        testcase.assertIsInstance(config_id, str)
        testcase.assertIsInstance(raw_entry, dict)
        assert isinstance(config_id, str)
        assert isinstance(raw_entry, dict)
        config = raw_entry["config"]
        testcase.assertIsInstance(config, dict)
        assert isinstance(config, dict)
        testcase.assertEqual(
            canonical_config_id(helion.Config.from_dict(config)),
            config_id,
        )
        testcase.assertEqual(set(raw_entry), {"config"})


@contextmanager
def clean_final_rebenchmark_env(**overrides: str):
    saved = {key: os.environ.get(key) for key in _FINAL_REBENCHMARK_ENV_KEYS}
    try:
        for key in _FINAL_REBENCHMARK_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(overrides)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _get_examples_matmul():
    """Lazy accessor to avoid CUDA init during pytest-xdist collection."""
    return import_path(examples_dir / "matmul.py").matmul


# Pin the compute capability for config-space golden tests. The matmul-seed
# heuristics are hardware-gated (the H100 formula fires on sm90, the B200 table on
# sm100), so which compiler seed configs get injected into ``random_population``
# depends on the CI runner's GPU. Force a fixed capability so the golden is stable
# across every runner (H100/B200/A10G) and does not shift as new sm-gated seed
# heuristics are added. sm90 is used because ``examples/matmul.py`` is a clean 2-D
# static GEMM that the sm90 budget-formula seed fires on.
_SM90_HARDWARE = _hardware.HardwareInfo(
    device_kind="cuda",
    hardware_name="NVIDIA H100",
    runtime_version="12.8",
    compute_capability="sm90",
)


def _pin_sm90(fn):
    """Run ``fn`` with ``get_hardware_info`` pinned to sm90, and evict the shared
    ``examples/matmul`` bound-kernel cache on both entry and exit.

    The pin only patches ``get_hardware_info``; the ``Kernel`` bind cache keys on the
    *real* device capability instead (``_device_specialization_key``). Without the
    eviction, a config computed under this pin (the sm90 budget-formula seed, e.g.
    ``block_sizes=[64, 64, 64], num_stages=6``) is cached against the real-GPU key and
    then reused by a later *un-pinned* test that actually compiles/executes it — which
    OOMs on a smaller GPU (A10G's 99KB shared memory). Clearing on entry gives this test
    a clean compute; clearing on exit guarantees no sm90-simulated config leaks into a
    test that runs on the true hardware.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        matmul = _get_examples_matmul()
        with patch.object(
            _hardware, "get_hardware_info", lambda device=None: _SM90_HARDWARE
        ):
            matmul._bound_kernels.clear()
            try:
                return fn(*args, **kwargs)
            finally:
                matmul._bound_kernels.clear()

    return wrapper


@contextmanager
def without_env_var(name: str):
    sentinel = object()
    previous = os.environ.pop(name, sentinel)
    try:
        yield
    finally:
        if previous is not sentinel:
            os.environ[name] = previous


class RecordingRandomSearch(RandomSearch):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.samples: list[float] = []

    def _autotune(self) -> helion.Config:
        self.samples.append(random.random())
        # The seed tests only assert on the sample recorded above, which is
        # drawn from the RNG seeded in _prepare(); skip the real search to
        # avoid compiling/benchmarking configs the assertions never look at.
        return self.config_spec.default_config()


class TestMismatchTolerance(TestCase):
    def test_get_test_float32_matmul_precision_pallas_interpret(self) -> None:
        with (
            patch("helion._testing._get_backend", return_value="pallas"),
            patch("helion._testing.is_pallas_interpret", return_value=True),
        ):
            self.assertEqual(get_test_float32_matmul_precision(), "high")

    def test_get_test_float32_matmul_precision_real_pallas(self) -> None:
        with (
            patch("helion._testing._get_backend", return_value="pallas"),
            patch("helion._testing.is_pallas_interpret", return_value=False),
        ):
            self.assertEqual(get_test_float32_matmul_precision(), "medium")

    def test_assert_close_with_mismatch_tolerance_bounds_mismatches(self) -> None:
        with self.assertRaisesRegex(
            AssertionError, "Mismatched absolute diff too large"
        ):
            assert_close_with_mismatch_tolerance(
                torch.tensor([10.0, 1.0, 1.0, 1.0], device=DEVICE),
                torch.tensor([1.0, 1.0, 1.0, 1.0], device=DEVICE),
                max_mismatch_pct=0.5,
                max_mismatched_abs_diff=5.0,
            )


@onlyBackends(["triton"])
class TestAutotuneIgnoreErrors(TestCase):
    def _make_search(
        self, settings: Settings, *, args: tuple[object, ...] = ()
    ) -> BaseSearch:
        # NOTE: construct via __init__ (mock kernel) instead of hand-mirroring
        # its attributes, so new __init__ fields don't need to be added here.
        config_spec = SimpleNamespace(
            default_config=lambda: helion.Config(block_sizes=[1]),
            cute_flash_search_enabled=False,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                should_deduplicate_generated_sources=lambda config_spec: False,
                get_do_bench=lambda: None,
                classify_autotune_exception=lambda error: None,
            ),
        )
        kernel = SimpleNamespace(
            settings=settings,
            config_spec=config_spec,
            format_kernel_decorator=lambda config, s: "decorator",
            to_triton_code=lambda config: "code",
            maybe_log_repro=lambda log_func, args, config=None: None,
            supports_subprocess_benchmark=lambda: False,
            env=SimpleNamespace(process_group_name=None),
        )
        search = BaseSearch(kernel, args)
        with patch.object(
            LocalBenchmarkProvider,
            "_compute_baseline",
            return_value=(None, [], None),
        ):
            search._prepare()
        return search

    def _make_compile_failure_search(self) -> BaseSearch:
        search = self._make_search(
            Settings(
                autotune_precompile=None,
                autotune_log_level=logging.CRITICAL,
            )
        )
        search.kernel.env = SimpleNamespace(process_group_name=None)
        search.kernel.compile_config = None
        return search

    def _run_late_compile_failures(
        self,
        late_configs: Sequence[str],
        error_for: Callable[[str], Exception],
    ) -> tuple[list[object], list[object]]:
        search = self._make_compile_failure_search()

        def compile_config(config: str, **_kwargs: object) -> Callable[..., None]:
            if config == "valid":
                return lambda *args, **kwargs: None
            raise error_for(config)

        with (
            patch.object(search.kernel, "compile_config", side_effect=compile_config),
            patch.object(
                search.benchmark_provider,
                "_benchmark_function",
                return_value=1.0,
            ),
        ):
            initial = search.benchmark_batch(["valid"], desc="initial")
            late = search.benchmark_batch(list(late_configs), desc="late")
        return initial, late

    def test_settings_flag_from_env(self):
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_IGNORE_ERRORS": "1"}, clear=False
        ):
            settings = Settings()
        self.assertTrue(settings.autotune_ignore_errors)

    def test_benchmark_raise_includes_hint(self):
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("boom")

        with patch("torch.accelerator.synchronize", autospec=True) as sync:
            sync.return_value = None
            with pytest.raises(exc.TritonError) as err:
                search.benchmark_provider._benchmark_function("cfg", bad_fn)

        assert "HELION_AUTOTUNE_IGNORE_ERRORS" in str(err.value)

    def test_llvm_translation_failure_skips_config(self):
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("failed to translate module to LLVM IR")

        with patch("torch.accelerator.synchronize", autospec=True) as sync:
            sync.return_value = None
            result = search.benchmark_provider._benchmark_function("cfg", bad_fn)

        self.assertEqual(result, float("inf"))
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)

    def test_cuda_oom_skips_config(self):
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        with patch("torch.accelerator.synchronize", autospec=True) as sync:
            sync.return_value = None
            result = search.benchmark_provider._benchmark_function("cfg", bad_fn)

        self.assertEqual(result, float("inf"))
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)

    def test_ignore_errors_skips_logging_and_raise(self):
        settings = Settings(
            autotune_ignore_errors=True,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("boom")

        with patch("torch.accelerator.synchronize", autospec=True) as sync:
            sync.return_value = None
            with patch.object(search.log, "warning") as warn:
                result = search.benchmark_provider._benchmark_function("cfg", bad_fn)

        self.assertEqual(result, float("inf"))
        warn.assert_not_called()

    def test_clear_jit_fast_path_caches(self):
        settings = Settings(
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        calls = []

        class FakeJITFunction:
            def clear_fast_path_caches(self) -> None:
                calls.append("cleared")

        def generated_kernel() -> None:
            return None

        globals_key = f"_helion_{generated_kernel.__name__}"
        generated_kernel.__globals__[globals_key] = FakeJITFunction()
        try:
            search.benchmark_provider._clear_jit_fast_path_caches(generated_kernel)
        finally:
            del generated_kernel.__globals__[globals_key]

        self.assertEqual(calls, ["cleared"])

    def test_clear_jit_fast_path_caches_does_not_clear_device_caches(self):
        settings = Settings(
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        class FakeJITFunction:
            def __init__(self) -> None:
                self.device_caches = {"compiled": object()}

            def clear_fast_path_caches(self) -> None:
                return None

        def generated_kernel() -> None:
            return None

        jit_fn = FakeJITFunction()
        device_caches = jit_fn.device_caches
        globals_key = f"_helion_{generated_kernel.__name__}"
        generated_kernel.__globals__[globals_key] = jit_fn
        try:
            search.benchmark_provider._clear_jit_fast_path_caches(generated_kernel)
        finally:
            del generated_kernel.__globals__[globals_key]

        self.assertIs(jit_fn.device_caches, device_caches)
        self.assertEqual(list(jit_fn.device_caches), ["compiled"])

    def test_clear_jit_fast_path_caches_ignores_cleanup_errors(self):
        settings = Settings(
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        class FakeJITFunction:
            def clear_fast_path_caches(self) -> None:
                raise RuntimeError("cleanup failed")

        def generated_kernel() -> None:
            return None

        globals_key = f"_helion_{generated_kernel.__name__}"
        generated_kernel.__globals__[globals_key] = FakeJITFunction()
        try:
            search.benchmark_provider._clear_jit_fast_path_caches(generated_kernel)
        finally:
            del generated_kernel.__globals__[globals_key]

    def test_benchmark_function_clears_jit_fast_path_caches_on_success(self):
        settings = Settings(
            autotune_accuracy_check=False,
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings, args=("arg0",))

        def compiled_fn(*_args):
            return None

        bench_fn = Mock(return_value=None)
        search.kernel.env = SimpleNamespace(process_group_name=None)
        search.kernel.bench_compile_config = Mock(return_value=bench_fn)

        with (
            patch("torch.accelerator.synchronize", autospec=True) as sync,
            patch(
                "helion.autotuner.benchmark_provider.do_bench",
                return_value=1.25,
            ),
            patch.object(
                search.benchmark_provider, "_clear_jit_fast_path_caches"
            ) as clear,
        ):
            sync.return_value = None
            result = search.benchmark_provider._benchmark_function("cfg", compiled_fn)

        self.assertEqual(result, 1.25)
        clear.assert_called_once_with(compiled_fn)

    def test_benchmark_function_clears_jit_fast_path_caches_on_error(self):
        settings = Settings(
            autotune_ignore_errors=True,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("boom")

        with (
            patch("torch.accelerator.synchronize", autospec=True) as sync,
            patch.object(
                search.benchmark_provider, "_clear_jit_fast_path_caches"
            ) as clear,
        ):
            sync.return_value = None
            result = search.benchmark_provider._benchmark_function("cfg", bad_fn)

        self.assertEqual(result, float("inf"))
        clear.assert_called_once_with(bad_fn)

    def test_traceback_cleared_str(self):
        """Test that str(e) still has meaningful content after e.__traceback__ = None."""
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        def bad_fn(*_args):
            raise RuntimeError("test error with meaningful message")

        with (
            patch("torch.accelerator.synchronize", autospec=True) as sync,
            patch(
                "helion.autotuner.benchmark_provider.classify_triton_exception",
                return_value="raise",
            ),
        ):
            sync.return_value = None
            with pytest.raises(exc.TritonError) as err:
                search.benchmark_provider._benchmark_function("cfg", bad_fn)

        # Verify the traceback was cleared
        assert err.value.__cause__.__traceback__ is None
        # Verify the error message is still accessible and meaningful
        assert "RuntimeError: test error with meaningful message" in str(err.value)

    def test_traceback_cleared_raise_from(self):
        """Test that 'raise ... from e' still has meaningful stack after e.__traceback__ = None."""
        settings = Settings(
            autotune_ignore_errors=False,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        original_exception = RuntimeError("original error in except block")

        def bad_fn(*_args):
            raise original_exception

        with (
            patch("torch.accelerator.synchronize", autospec=True) as sync,
            patch(
                "helion.autotuner.benchmark_provider.classify_triton_exception",
                return_value="raise",
            ),
        ):
            sync.return_value = None
            with pytest.raises(exc.TritonError) as err:
                search.benchmark_provider._benchmark_function("cfg", bad_fn)

        # Verify the traceback was cleared
        assert err.value.__cause__.__traceback__ is None
        # Verify the exception chain is preserved even after __traceback__ = None
        assert err.value.__cause__ is original_exception
        assert str(original_exception) == "original error in except block"
        # Verify we can still get the error type and message
        assert type(err.value.__cause__).__name__ == "RuntimeError"

    def test_benchmark_results_aligned_when_compile_fails(self):
        """benchmark_batch must return one result per input config even when some
        fail to compile."""
        settings = Settings(
            autotune_precompile=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)

        call_count = 0

        def fail_second(config, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated compile failure")
            return lambda *a, **kw: None

        search.kernel.compile_config = None
        search.kernel.env = SimpleNamespace(process_group_name=None)
        configs = ["cfg_a", "cfg_b", "cfg_c"]
        with (
            patch.object(search.kernel, "compile_config", side_effect=fail_second),
            patch.object(
                search.benchmark_provider,
                "_benchmark_function",
                return_value=1.0,
            ),
        ):
            results = search.benchmark_batch(configs, desc="test")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].perf, 1.0)
        self.assertEqual(results[1].perf, float("inf"))
        self.assertEqual(results[1].status, "error")
        self.assertEqual(results[2].perf, 1.0)
        self.assertEqual(search._autotune_metrics.num_compile_failures, 1)

    def test_initial_compile_failures_raise(self) -> None:
        cases: tuple[tuple[str, type[Exception], str, Exception], ...] = (
            (
                "invalid",
                exc.InvalidConfig,
                "invalid initial config",
                exc.InvalidConfig("invalid initial config"),
            ),
            (
                "runtime",
                RuntimeError,
                "initial compiler failure",
                RuntimeError("initial compiler failure"),
            ),
        )
        for config, error_type, message, error in cases:
            with self.subTest(config=config):
                search = self._make_compile_failure_search()
                with (
                    patch.object(
                        search.kernel,
                        "compile_config",
                        side_effect=error,
                    ),
                    self.assertRaisesRegex(error_type, message),
                ):
                    search.benchmark_batch([config], desc="initial")

    def test_late_compile_failures_are_skipped(self) -> None:
        cases = (
            (
                "invalid",
                lambda config: exc.InvalidConfig(f"invalid late neighbor: {config}"),
            ),
            ("runtime", lambda config: RuntimeError("late compiler failure")),
            (
                "unsupported",
                lambda config: exc.BackendUnsupported(
                    "cute", f"unsupported late neighbor {config}"
                ),
            ),
        )
        for prefix, error_for in cases:
            with self.subTest(prefix=prefix):
                initial, late = self._run_late_compile_failures(
                    (f"{prefix}_a", f"{prefix}_b"), error_for
                )
                self.assertEqual(initial[0].perf, 1.0)
                self.assertEqual([result.perf for result in late], [float("inf")] * 2)
                self.assertEqual([result.status for result in late], ["error"] * 2)

    def test_autotune_log_sink_writes_csv_and_log(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune_run"
        settings = Settings(
            autotune_log=str(base_path),
            autotune_log_level=logging.CRITICAL,
        )
        logger = AutotuningLogger(settings)
        config = helion.Config(block_sizes=[32], num_warps=4)
        with logger.autotune_logging():
            config_id = logger.register_config(config)
            entry = AutotuneLogEntry(
                generation=5,
                status="ok",
                perf_ms=1.234,
                compile_time=0.5,
                config_id=config_id,
                config=config,
                source_hash="source-abc",
            )
            logger.record_autotune_entry(entry)
            logger("finalized entry", level=logging.CRITICAL)

        csv_path = base_path.with_suffix(".csv")
        log_path = base_path.with_suffix(".log")
        self.assertTrue(csv_path.exists())
        self.assertTrue(log_path.exists())
        rows = list(csv.reader(csv_path.read_text().splitlines()))
        self.assertEqual(len(rows), 2)  # header + exactly the one recorded entry
        header, row = rows[0], rows[1]
        self.assertEqual(
            header,
            [
                "run_id",
                "timestamp_s",
                "config_id",
                "generation",
                "status",
                "perf_ms",
                "compile_time_s",
                "config",
            ],
        )

        def cell(name: str) -> str:
            return row[header.index(name)]

        # No metadata supplied here, so the run_id join key is empty.
        self.assertEqual(cell("run_id"), "")
        self.assertEqual(cell("config_id"), config_id)
        self.assertEqual(cell("generation"), "5")
        self.assertEqual(cell("status"), "ok")
        self.assertEqual(cell("perf_ms"), "1.234000")
        self.assertEqual(cell("compile_time_s"), "0.50")
        self.assertEqual(cell("config"), str(config))
        source_rows = list(
            csv.reader(base_path.with_suffix(".sources.csv").read_text().splitlines())
        )
        self.assertEqual(source_rows[0][-2:], ["status", "source_hash"])
        self.assertEqual(source_rows[1][-2:], ["ok", "source-abc"])
        log_text = log_path.read_text()
        self.assertIn("finalized entry", log_text)

    def test_differential_evolution_immediate_iter_uses_batch_helper(self):
        search = DifferentialEvolutionSearch.__new__(DifferentialEvolutionSearch)
        search.immediate_update = True
        search.population = [object(), object(), object()]

        calls: list[list[int]] = []

        def batch(indices: Sequence[int]) -> list[PopulationMember]:
            calls.append(list(indices))
            members: list[PopulationMember] = []
            for idx in indices:
                members.append(
                    PopulationMember(
                        lambda *args, **kwargs: None,
                        [float(idx)],
                        [],
                        SimpleNamespace(config={"idx": idx}),
                        status="ok",
                    )
                )
            return members

        search._benchmark_mutation_batch = batch  # type: ignore[assignment]
        candidates = list(search.iter_candidates())
        self.assertEqual(calls, [[0], [1], [2]])
        self.assertEqual([idx for idx, _ in candidates], [0, 1, 2])

    def test_differential_evolution_parallel_iter_uses_batch_helper(self):
        search = DifferentialEvolutionSearch.__new__(DifferentialEvolutionSearch)
        search.immediate_update = False
        search.population = [object(), object()]

        def batch(indices: Sequence[int]) -> list[PopulationMember]:
            members: list[PopulationMember] = []
            for idx in indices:
                members.append(
                    PopulationMember(
                        lambda *args, **kwargs: None,
                        [float(idx)],
                        [],
                        SimpleNamespace(config={"idx": idx}),
                        status="ok",
                    )
                )
            return members

        calls: list[list[int]] = []

        def recording_batch(indices: Sequence[int]) -> list[PopulationMember]:
            calls.append(list(indices))
            return batch(indices)

        search._benchmark_mutation_batch = recording_batch  # type: ignore[assignment]
        candidates = list(search.iter_candidates())
        self.assertEqual(calls, [[0, 1]])
        self.assertEqual([idx for idx, _ in candidates], [0, 1])

    def test_benchmarked_members_survive_config_mutation(self):
        # Regression test: _benchmarked_members / _pinned_finalist_members must
        # not be keyed by Config objects. Config is mutable and hashes on its
        # contents, so mutating a recorded config in place (as normalize /
        # neighbor generation does during the search) corrupted the bookkeeping
        # -> _prune_benchmarked_members raised KeyError, and final verification
        # could select a config that was never accuracy-validated.
        search = DifferentialEvolutionSearch.__new__(DifferentialEvolutionSearch)
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        def make_member(perf: float, num_warps: int) -> PopulationMember:
            return PopulationMember(
                lambda *a, **k: None,
                [perf],
                [],
                helion.Config(block_sizes=[64], num_warps=num_warps),
                status="ok",
            )

        fast = make_member(1.0, num_warps=4)
        slow = make_member(2.0, num_warps=8)
        search._record_best_member_for_config(
            search._benchmarked_members, fast.config, fast
        )
        search._record_best_member_for_config(
            search._benchmarked_members, slow.config, slow
        )

        # Mutate the original configs in place *after* they were recorded.
        fast.config.config["num_warps"] = 999
        slow.config.config["num_warps"] = 999

        # Pruning to the single fastest config must not raise (the old code did
        # `del dict[config]` on a config whose hash had changed -> KeyError) and
        # must keep the fast member with the config it was actually benchmarked
        # with (snapshot), not the later in-place mutation.
        search._prune_benchmarked_members(top_k=1)
        self.assertEqual(len(search._benchmarked_members), 1)
        kept = next(iter(search._benchmarked_members.values()))
        self.assertEqual(kept.perfs[0], 1.0)
        self.assertEqual(kept.config["num_warps"], 4)

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork start method is unavailable on this platform",
    )
    def test_fork_precompile_avoids_cuda_reinit(self):
        settings = Settings(
            autotune_precompile="fork",
            autotune_log_level=logging.CRITICAL,
            autotune_compile_timeout=5,
        )
        search = self._make_search(settings, args=("arg0",))

        parent_pid = os.getpid()
        lazy_calls: list[int] = []

        def fake_lazy_init() -> None:
            lazy_calls.append(os.getpid())

        def fake_make_precompiler(_kernel_obj, _config, _bound_kernel):
            def binder(*_args: object, **_kwargs: object):
                def run() -> None:
                    return None

                return run

            return binder

        def fake_compiled_fn(
            *fn_args: object, _launcher: Callable[..., object]
        ) -> None:
            torch.cuda._lazy_init()
            _launcher("fake_kernel", (1,), *fn_args)

        with (
            patch(
                "helion.autotuner.precompile_future.make_precompiler",
                side_effect=fake_make_precompiler,
            ),
            patch("torch.cuda._lazy_init", side_effect=fake_lazy_init),
        ):
            future = search.benchmark_provider._create_precompile_future(
                "cfg", fake_compiled_fn
            )
            self.assertTrue(future())

        self.assertEqual(set(lazy_calls), {parent_pid})

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork start method is unavailable on this platform",
    )
    def test_fork_precompile_expected_errors_skip_config(self):
        from torch._inductor.runtime.triton_compat import OutOfResources

        expected_errors = [
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            OutOfResources(128, 64, "shared memory"),
            RuntimeError("out of resource: shared memory"),
            RuntimeError("too many resources requested for launch"),
            RuntimeError("CUDA error: out of memory"),
            RuntimeError("[CUDA]: out of memory"),
            RuntimeError("failed to translate module to LLVM IR"),
        ]
        for err in expected_errors:
            with self.subTest(error=type(err).__name__, msg=str(err)):
                settings = Settings(
                    autotune_precompile="fork",
                    autotune_ignore_errors=False,
                    autotune_log_level=logging.CRITICAL,
                )
                search = self._make_search(settings, args=("arg0",))

                def fake_compiled_fn(
                    *fn_args: object, _launcher: Callable[..., object]
                ) -> None:
                    _launcher("fake_kernel", (1,), *fn_args)

                with patch(
                    "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
                    side_effect=err,
                ):
                    future = search.benchmark_provider._create_precompile_future(
                        "cfg", fake_compiled_fn
                    )

                self.assertFalse(future.ok)

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork start method is unavailable on this platform",
    )
    def test_fork_precompile_illegal_memory_access_raises(self):
        settings = Settings(
            autotune_precompile="fork",
            autotune_ignore_errors=True,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings, args=("arg0",))

        def fake_compiled_fn(
            *fn_args: object, _launcher: Callable[..., object]
        ) -> None:
            _launcher("fake_kernel", (1,), *fn_args)

        with (
            patch(
                "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
                side_effect=RuntimeError("an illegal memory access was encountered"),
            ),
            pytest.raises(RuntimeError, match="illegal memory access"),
        ):
            search.benchmark_provider._create_precompile_future("cfg", fake_compiled_fn)

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork start method is unavailable on this platform",
    )
    def test_fork_precompile_unexpected_error_skipped_with_ignore_errors(self):
        settings = Settings(
            autotune_precompile="fork",
            autotune_ignore_errors=True,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings, args=("arg0",))

        def fake_compiled_fn(
            *fn_args: object, _launcher: Callable[..., object]
        ) -> None:
            _launcher("fake_kernel", (1,), *fn_args)

        with patch(
            "helion.autotuner.precompile_future._prepare_precompiler_for_fork",
            side_effect=RuntimeError("something unexpected"),
        ):
            future = search.benchmark_provider._create_precompile_future(
                "cfg", fake_compiled_fn
            )

        self.assertFalse(future.ok)

    def _run_autotuner_and_check_logging(
        self, search_factory: Callable[[object, tuple[object, ...]], BaseSearch]
    ) -> None:
        """Helper to verify started/completion logging for any autotuner."""
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune_run"

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNE_LOG": str(base_path),
                "HELION_AUTOTUNE_LOG_LEVEL": "0",
            },
        ):
            # started/completed entries are recorded in the parent's benchmark
            # loop either way; skip the benchmark worker subprocess (several
            # seconds of interpreter+CUDA startup per search).
            @helion.kernel(autotune_benchmark_subprocess=False)
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            args = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_kernel = add.bind(args)
            random.seed(123)
            search = search_factory(bound_kernel, args)
            search.autotune()

        csv_path = base_path.with_suffix(".csv")
        self.assertTrue(csv_path.exists())
        rows = list(csv.reader(csv_path.read_text().splitlines()))
        status_idx = rows[0].index("status")  # look up by name; column order may change
        statuses = [row[status_idx] for row in rows[1:]]  # skip header
        started_count = sum(1 for s in statuses if s == "started")
        completed_count = sum(1 for s in statuses if s in ("ok", "error", "timeout"))
        self.assertGreater(started_count, 0, "Should log started entries")
        self.assertEqual(
            started_count, completed_count, "Each started should have completion"
        )

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_autotune_log_started_completed(self):
        """Test started/completion logging with all autotuning algorithms."""
        configs = [
            helion.Config(block_sizes=[32], num_warps=4),
            helion.Config(block_sizes=[64], num_warps=8),
        ]
        search_factories = [
            (
                "FiniteSearch",
                lambda kernel, args: FiniteSearch(kernel, args, configs=configs),
            ),
            ("RandomSearch", lambda kernel, args: RandomSearch(kernel, args, count=2)),
            (
                "PatternSearch",
                lambda kernel, args: PatternSearch(
                    kernel,
                    args,
                    initial_population=2,
                    max_generations=1,
                    copies=1,
                    num_neighbors_cap=2,
                ),
            ),
            (
                "DifferentialEvolutionSearch",
                lambda kernel, args: DifferentialEvolutionSearch(
                    kernel, args, population_size=2, max_generations=1
                ),
            ),
        ]
        for name, factory in search_factories:
            with self.subTest(algorithm=name):
                self._run_autotuner_and_check_logging(factory)

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_autotune_skips_restricted_search(self):
        """A run restricted to user-pinned configs (``configs=[...]`` without
        ``force_autotune``) is a biased slice excluded from the dataset: even
        with the dataset flag on, no ``.meta.jsonl`` is written. The debug
        ``.csv`` is still written, since it is gated only by the log path
        (PRD FR8/FR9)."""
        configs = [
            helion.Config(block_sizes=[32], num_warps=4),
            helion.Config(block_sizes=[64], num_warps=8),
        ]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune_run"

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNE_LOG": str(base_path),
                "HELION_AUTOTUNE_LOG_DETAILS": "1",
                "HELION_AUTOTUNE_LOG_LEVEL": "0",
            },
        ):
            # The CSV/sidecar gating under test is parent-side logging; skip
            # the benchmark worker subprocess (seconds of startup overhead).
            @helion.kernel(configs=configs, autotune_benchmark_subprocess=False)
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            args = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_kernel = add.bind(args)
            random.seed(123)
            search = FiniteSearch(bound_kernel, args, configs=configs)
            search.autotune()

        # Restricted search -> debug CSV written, but no dataset sidecar.
        self.assertTrue(base_path.with_suffix(".csv").exists())
        self.assertFalse(base_path.with_suffix(".meta.jsonl").exists())


class TestConfigFragmentCardinality(TestCase):
    """cardinality()/search_values() are pure, GPU-independent fragment
    methods, so they run in both normal and ref-eager modes."""

    def test_fragment_cardinality(self):
        # Number of distinct search values per fragment type. Feeds the
        # search-space size product and coverage denominators.
        self.assertIsNone(ConfigSpecFragment().cardinality())  # unknown by default
        self.assertEqual(PermutationFragment(1).cardinality(), 1)
        self.assertEqual(PermutationFragment(4).cardinality(), 24)  # 4!
        self.assertEqual(IntegerFragment(1, 8).cardinality(), 8)
        self.assertEqual(PowerOfTwoFragment(1, 64).cardinality(), 7)  # 1..64
        self.assertEqual(PowerOfTwoFragment(1, 1).cardinality(), 1)  # boundary
        self.assertEqual(BlockSizeFragment(16, 256).cardinality(), 5)
        self.assertEqual(NumWarpsFragment(1, 32).cardinality(), 6)
        self.assertEqual(EnumFragment(("a", "b", "c")).cardinality(), 3)
        # search_choices restrict the searched cardinality to the subset.
        self.assertEqual(
            EnumFragment(("a", "b", "c", "d"), search_choices=("a", "c")).cardinality(),
            2,
        )
        self.assertEqual(BooleanFragment().cardinality(), 2)
        # NumThreads: "0" (auto) plus powers of two up to high.
        self.assertEqual(NumThreadsFragment(256).cardinality(), 10)
        # ListOf is combinatorial (inner ** length), not linear.
        self.assertEqual(
            ListOf(EnumFragment(("a", "b", "c")), length=4).cardinality(), 81
        )
        # ListOf over an unknown-cardinality inner is itself unknown.
        self.assertIsNone(ListOf(ConfigSpecFragment(), length=3).cardinality())

    def test_fragment_search_values(self):
        # Explicit enumerable values, and the limit guard that avoids
        # materializing very large ranges.
        self.assertIsNone(ConfigSpecFragment().search_values())
        self.assertEqual(
            IntegerFragment(1, 8).search_values(), [1, 2, 3, 4, 5, 6, 7, 8]
        )
        self.assertEqual(
            PowerOfTwoFragment(1, 64).search_values(), [1, 2, 4, 8, 16, 32, 64]
        )
        self.assertEqual(
            BlockSizeFragment(16, 256).search_values(), [16, 32, 64, 128, 256]
        )
        self.assertEqual(EnumFragment(("a", "b", "c")).search_values(), ["a", "b", "c"])
        self.assertEqual(
            EnumFragment(
                ("a", "b", "c", "d"), search_choices=("a", "c")
            ).search_values(),
            ["a", "c"],
        )
        self.assertEqual(BooleanFragment().search_values(), [False, True])
        self.assertEqual(
            NumThreadsFragment(256).search_values(),
            [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        )
        # limit guard: None (not a materialized list) above the limit, full below.
        self.assertIsNone(IntegerFragment(0, 200).search_values())
        self.assertEqual(len(IntegerFragment(0, 200).search_values(limit=1000)), 201)
        # ListOf combinations are not enumerated.
        self.assertIsNone(
            ListOf(EnumFragment(("a", "b", "c")), length=4).search_values()
        )

    def test_enum_fragment_coverage_choices(self):
        fragment = EnumFragment(
            ("a", "b", "c"),
            search_choices=("a", "b", "c"),
            coverage_choices=("a", "c"),
        )

        self.assertEqual(fragment.cardinality(), 3)
        self.assertEqual(fragment.search_values(), ["a", "b", "c"])
        self.assertEqual(fragment.pattern_neighbors("a"), ["b", "c"])
        with patch("helion.autotuner.config_fragment.random.choice", return_value="b"):
            self.assertEqual(fragment.random(), "b")
        self.assertEqual(
            fragment.fingerprint(),
            (
                "enum",
                "'a'",
                "'b'",
                "'c'",
                "search",
                "'a'",
                "'b'",
                "'c'",
                "coverage",
                "'a'",
                "'c'",
            ),
        )

        with self.assertRaisesRegex(ValueError, "coverage_choices must not be empty"):
            EnumFragment(("a", "b"), coverage_choices=())
        with self.assertRaisesRegex(
            ValueError, "coverage_choices must be a subset of active search choices"
        ):
            EnumFragment(("a", "b"), coverage_choices=("c",))
        with self.assertRaisesRegex(
            ValueError, "coverage_choices must be a subset of active search choices"
        ):
            EnumFragment(
                ("a", "b", "c"),
                search_choices=("a", "b"),
                coverage_choices=("c",),
            )

    def test_block_id_sequence_cardinality(self):
        # Product of per-item fragment cardinalities; empty -> 1 (neutral);
        # an unknown-cardinality item makes the whole sequence unknown.
        from helion.autotuner.block_id_sequence import BlockIdSequence
        from helion.autotuner.block_id_sequence import _BlockIdItem

        class _Item(_BlockIdItem):
            def __init__(self, ids, frag):
                super().__init__(ids)
                self._f = frag

            def _fragment(self, base):
                return self._f

        seq = BlockIdSequence()
        seq.append(_Item([0], EnumFragment(("a", "b", "c"))))
        seq.append(_Item([1], EnumFragment(("x", "y"))))
        self.assertEqual(seq.cardinality(None), 6)  # 3 * 2
        self.assertEqual(BlockIdSequence().cardinality(None), 1)

        unknown = BlockIdSequence()
        unknown.append(_Item([0], ConfigSpecFragment()))
        self.assertIsNone(unknown.cardinality(None))


@onlyBackends(["triton"])
class TestAutotuner(RefEagerTestDisabled, TestCase):
    def setUp(self):
        super().setUp()
        random.seed(112)

    @_pin_sm90
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    def test_config_fragment0(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        spec = _get_examples_matmul().bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch(
        "helion.autotuner.config_generation.warps_to_threads",
        lambda num_warps: num_warps * 32,
    )
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @patch("torch.version.hip", None)
    @patch("torch.version.xpu", None)
    @patch("helion._hardware.get_hardware_info", return_value=_HOPPER_HARDWARE)
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    def test_config_fragment1(self, _mock_hardware):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch(
        "helion.autotuner.config_generation.warps_to_threads",
        lambda num_warps: num_warps * 32,
    )
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @patch("torch.version.hip", None)
    @patch("torch.version.xpu", None)
    @patch("helion._hardware.get_hardware_info", return_value=_HOPPER_HARDWARE)
    @skipIfTileIR("tileir backend will ignore `warp specialization` hint")
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    def test_config_warp_specialize_unroll(self, _mock_hardware):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        overrides = {"range_unroll_factors": [4], "range_warp_specializes": ([True])}
        # We expect all the unroll factors to be set to 0
        configs = ConfigGeneration(spec, overrides=overrides).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @_pin_sm90
    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    @patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16))
    @patch.object(_compat, "_supports_maxnreg", lambda: True)
    @patch.object(loops, "_supports_warp_specialize", lambda: True)
    @skipIfRocm("config space differs on ROCm")
    @skipIfXPU("maxnreg uses CUDA-specific register query")
    @skipIfTileIR("block-size overshoot is gated to the triton backend")
    def test_small_dim_block_size_overshoot(self):
        # All dims are 16, smaller than SMALL_DIM_BLOCK_SIZE_OVERSHOOT, so the
        # generated configs may use block sizes larger than the dimensions
        # themselves (e.g. 32 or 64) -- the extra rows/cols are masked off.
        self.assertEqual(SMALL_DIM_BLOCK_SIZE_OVERSHOOT, 64)
        args = (
            torch.randn([16, 16], device=DEVICE),
            torch.randn([16, 16], device=DEVICE),
        )
        spec = _get_examples_matmul().bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        self.assertExpectedJournal("\n".join(map(repr, configs)))

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: True)
    def test_config_generation_overrides(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        overrides = {"indexing": "tensor_descriptor"}
        gen = ConfigGeneration(spec, overrides=overrides)

        flat = gen.default_flat()
        config = gen.unflatten([*flat])
        self.assertEqual(config["indexing"], "tensor_descriptor")
        configs = [gen.unflatten(gen.random_flat()) for _ in range(3)]
        self.assertEqual({cfg["indexing"] for cfg in configs}, {"tensor_descriptor"})
        indexing_choices = spec.valid_indexing_types()
        indexing_index = next(
            i
            for i, fragment in enumerate(gen.flat_spec)
            if isinstance(fragment, ListOf)
            and isinstance(fragment.inner, EnumFragment)
            and fragment.inner.choices == tuple(indexing_choices)
        )
        mutated = gen.random_flat()
        mutated[indexing_index] = "pointer"
        new_config = gen.unflatten(mutated)
        self.assertEqual(new_config["indexing"], "tensor_descriptor")
        self.assertEqual(mutated[indexing_index], "pointer")

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    def test_save_load_config(self):
        config = helion.Config(
            block_sizes=[64, 64, 32],
            loop_orders=[[1, 0]],
            num_warps=2,
            num_stages=1,
            indexing="block_ptr",
            l2_grouping=32,
        )
        with tempfile.NamedTemporaryFile() as f:
            config.save(f.name)
            loaded_config = helion.Config.load(f.name)
            self.assertEqual(config, loaded_config)
        self.assertExpectedJournal(config.to_json())

    def test_config_pickle_roundtrip(self):
        config = helion.Config(
            block_sizes=[64, 64, 32],
            loop_orders=[[1, 0]],
            num_warps=4,
            num_stages=2,
            indexing="tensor_descriptor",
            extra_metadata={"nested": [1, 2, 3]},
        )
        restored = pickle.loads(pickle.dumps(config))
        self.assertIsInstance(restored, helion.Config)
        self.assertEqual(config, restored)
        self.assertIsNot(config, restored)
        self.assertIsNot(config.config, restored.config)

    def test_run_fixed_config(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[1024, 1, 1],
                flatten_loops=[True],
                loop_orders=[[0, 2, 1]],
                num_warps=8,
            )
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))

    def test_finite_search_all_configs_fail_raises(self):
        """Test that when all configs fail, the error is re-raised.

        Without this, compile failures would be silently swallowed and the
        autotuner would return no results. We must surface the error so
        users know their configs are incompatible with the input shape.
        """

        @helion.kernel(
            configs=[
                helion.Config(block_sizes=[64]),
                helion.Config(block_sizes=[128]),
            ],
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        with self.assertRaises(exc.InvalidConfig):
            add(*args)

    def test_run_finite_search(self):
        @helion.kernel(
            configs=[
                helion.Config(
                    block_sizes=[1024, 1, 1],
                    flatten_loops=[True],
                    loop_orders=[[0, 2, 1]],
                    num_warps=8,
                ),
                helion.Config(
                    block_sizes=[1024, 1, 1], flatten_loops=[True], num_warps=8
                ),
                helion.Config(block_sizes=[1, 64, 64], num_warps=8),
                helion.Config(block_sizes=[1, 1, 512], num_warps=8),
            ],
            autotune_log_level=0,
            # Config selection is the behavior under test; skip the benchmark
            # worker subprocess (seconds of startup overhead). The worker path
            # keeps coverage via test_autotune_noncontiguous_arg/broadcast_arg.
            autotune_benchmark_subprocess=False,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))
        torch.testing.assert_close(add(*args), sum(args))

    def test_finite_search_skips_bad_configs(self):
        """Test that configs that fail to compile are skipped.

        Uses a config with wrong number of block_sizes (1 instead of 3)
        placed between two good configs, to verify the skip logic doesn't
        disrupt processing of subsequent valid configs.
        """

        # The compile-failure skip logic under test runs at the compile stage;
        # skip the benchmark worker subprocess (seconds of startup overhead).
        @helion.kernel(
            configs=[
                # Good config
                helion.Config(block_sizes=[1, 64, 64], num_warps=8),
                # Bad config: insufficient block_sizes for a 3D kernel
                helion.Config(block_sizes=[64]),
                # Good config after bad one — must still work
                helion.Config(block_sizes=[1, 1, 512], num_warps=8),
            ],
            autotune_log_level=0,
            autotune_benchmark_subprocess=False,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        # Bad config (block_sizes=[64]) has wrong number of block_sizes for
        # 3D input and should fail to compile. The surrounding good configs
        # should allow autotuning to succeed.
        torch.testing.assert_close(add(*args), sum(args))

    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_random_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = _get_examples_matmul().bind(args)
        bound_kernel.settings.autotune_precompile = None
        # Smoke test of the RandomSearch loop itself; skip the benchmark
        # worker subprocess (seconds of startup overhead).
        bound_kernel.settings.autotune_benchmark_subprocess = False
        random.seed(123)
        best = RandomSearch(bound_kernel, args, 5).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    @skip("too slow")
    def test_differential_evolution_search(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = _get_examples_matmul().bind(args)
        random.seed(123)
        best = DifferentialEvolutionSearch(
            bound_kernel, args, 5, max_generations=3
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    @skip("too slow")
    def test_de_surrogate_hybrid(self):
        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        bound_kernel = _get_examples_matmul().bind(args)
        random.seed(123)
        best = DESurrogateHybrid(
            bound_kernel, args, population_size=5, max_generations=3
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), args[0] @ args[1], rtol=1e-2, atol=1e-1)

    def test_differential_evolution_early_stopping_parameters(self):
        """Test that early stopping is disabled by default and can be enabled."""
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)

        # Test 1: Default parameters (early stopping disabled)
        search = DifferentialEvolutionSearch(
            bound_kernel, args, population_size=5, max_generations=3
        )
        self.assertIsNone(search.min_improvement_delta)
        self.assertIsNone(search.patience)

        # Test 2: Enable early stopping with custom parameters
        search_custom = DifferentialEvolutionSearch(
            bound_kernel,
            args,
            population_size=5,
            max_generations=3,
            min_improvement_delta=0.01,
            patience=5,
        )
        self.assertEqual(search_custom.min_improvement_delta, 0.01)
        self.assertEqual(search_custom.patience, 5)

    def test_de_surrogate_early_stopping_parameters(self):
        """Test that DE-Surrogate early stopping parameters are optional with correct defaults."""
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)

        # Test 1: Default parameters (optional)
        search = DESurrogateHybrid(
            bound_kernel, args, population_size=5, max_generations=3
        )
        self.assertEqual(search.min_improvement_delta, 0.001)
        self.assertEqual(search.patience, 3)

        # Test 2: Custom parameters
        search_custom = DESurrogateHybrid(
            bound_kernel,
            args,
            population_size=5,
            max_generations=3,
            min_improvement_delta=0.01,
            patience=5,
        )
        self.assertEqual(search_custom.min_improvement_delta, 0.01)
        self.assertEqual(search_custom.patience, 5)

    @skip("too slow")
    def test_pattern_search(self):
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)
        random.seed(123)
        best = PatternSearch(
            bound_kernel, args, initial_population=10, max_generations=2, copies=1
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), sum(args), rtol=1e-2, atol=1e-1)

    def test_pattern_search_neighbor_values(self):
        self.assertEqual(
            PowerOfTwoFragment(1, 128, 32).pattern_neighbors(32),
            [16, 64],
        )
        self.assertEqual(
            sorted(IntegerFragment(1, 5, 3).pattern_neighbors(3)),
            [2, 4],
        )
        self.assertEqual(BooleanFragment().pattern_neighbors(True), [False])
        self.assertEqual(
            sorted(EnumFragment(("a", "b", "c")).pattern_neighbors("b")),
            ["a", "c"],
        )

    def test_pattern_search_neighbor_values_radius(self):
        # PowerOfTwoFragment: radius=2 should return 2 steps in exponent space
        self.assertEqual(
            PowerOfTwoFragment(1, 128, 32).pattern_neighbors(32, radius=2),
            [8, 16, 64, 128],
        )
        # PowerOfTwoFragment: radius=2 clamped at lower boundary
        self.assertEqual(
            PowerOfTwoFragment(16, 128, 16).pattern_neighbors(16, radius=2),
            [32, 64],
        )
        # PowerOfTwoFragment: radius=2 clamped at upper boundary
        self.assertEqual(
            PowerOfTwoFragment(1, 64, 64).pattern_neighbors(64, radius=2),
            [16, 32],
        )
        # IntegerFragment: radius=2 returns ±2 neighbors
        self.assertEqual(
            sorted(IntegerFragment(1, 10, 5).pattern_neighbors(5, radius=2)),
            [3, 4, 6, 7],
        )
        # IntegerFragment: radius=2 clamped at boundaries
        self.assertEqual(
            sorted(IntegerFragment(1, 5, 1).pattern_neighbors(1, radius=2)),
            [2, 3],
        )
        # BooleanFragment: radius is ignored, always returns [not current]
        self.assertEqual(BooleanFragment().pattern_neighbors(True, radius=3), [False])
        # EnumFragment: radius is ignored, always returns all other choices
        self.assertEqual(
            sorted(EnumFragment(("a", "b", "c")).pattern_neighbors("b", radius=5)),
            ["a", "c"],
        )
        # ListOf: radius is forwarded to inner fragment
        list_frag = ListOf(inner=IntegerFragment(1, 10, 5), length=2)
        neighbors = list_frag.pattern_neighbors([5, 5], radius=2)
        # Each position yields 4 neighbors (3,4,6,7) = 8 single-position
        # changes, plus uniform lists (all elements set to the same value
        # near the inner default): [3,3], [4,4], [6,6], [7,7].
        single = [n for n in neighbors if n.count(5) == 1]
        uniform = [n for n in neighbors if n.count(5) == 0]
        self.assertEqual(len(single), 8)
        self.assertEqual(sorted(uniform), [[3, 3], [4, 4], [6, 6], [7, 7]])
        self.assertEqual(len(neighbors), 12)

    def test_lfbo_flash_terminal_coordinate_refinement_runs_two_rounds(self):
        def make_fn(source_id: int):
            def fn():
                return None

            fn.source_hash = f"{source_id:064x}"
            return fn

        def make_member(value: int, perf: float | None) -> PopulationMember:
            return PopulationMember(
                fn=make_fn(value),
                perfs=[] if perf is None else [perf],
                flat_values=[value],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=True,
                    cute_flash_wait_hint=value,
                ),
                status="unknown" if perf is None else "ok",
            )

        def projections(value: int) -> list[CoordinateNeighborProjection]:
            targets = (1, 2) if value == 0 else (0, 2)
            result = []
            for target in targets:
                projected = make_member(target, None).config
                projected.config["cute_vector_widths"] = [1, 1, 1]
                result.append(
                    CoordinateNeighborProjection(
                        flat_index=0,
                        key=cute_flash.FLASH_WAIT_HINT_KEY,
                        sequence_index=None,
                        from_value=value,
                        to_value=target,
                        outcome="candidate",
                        flat_values=[target],
                        config=projected,
                    )
                )
            return result

        def canonicalize_projections(projections, *, base_config):
            return [
                replace(
                    projection,
                    config=make_member(int(projection.to_value), None).config,
                )
                for projection in projections
            ]

        best = make_member(0, 10.0)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.best_perf_so_far = best.perf
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
            ),
        )
        search.config_gen = SimpleNamespace(
            flatten=lambda config: [config.config[cute_flash.FLASH_WAIT_HINT_KEY]],
            canonicalize_coordinate_projections=canonicalize_projections,
        )
        leaf_generation = SimpleNamespace(
            flatten=search.config_gen.flatten,
            coordinate_neighbor_projections=lambda flat, *, radius: projections(
                int(flat[0])
            ),
        )
        search._flash_leaf_config_generation = Mock(return_value=leaf_generation)
        search.flash_structural_search = replace(
            get_effort_profile("full").flash_structural_search,
            terminal_coordinate_rounds=2,
            terminal_coordinate_beam_width=2,
        )
        search._cute_flash_lane_policy_enabled = True
        search._terminal_refinement_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search._benchmarked_members = {}
        search._autotune_metrics = SimpleNamespace(
            num_generations=20,
            num_configs_tested=100,
            search_phase_metrics={},
        )
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=None,
        )
        search.radius = 2
        search.min_improvement_delta = 0.001
        search.log = Mock()
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.make_unbenchmarked = Mock(
            side_effect=lambda flat: make_member(int(flat[0]), None)
        )

        initial_perfs = {1: 9.9, 2: 10.2}

        def benchmark_population(members, *, desc):
            self.assertIn("Terminal coordinate refinement", desc)
            self.assertEqual(
                search._autotune_metrics.search_phase_metrics[
                    "terminal_coordinate_refinement"
                ]["preterminal_num_configs_tested"],
                100,
            )
            search._autotune_metrics.num_configs_tested += len(members)
            for member in members:
                value = member.config.config[cute_flash.FLASH_WAIT_HINT_KEY]
                member.perfs.append(initial_perfs[value])
                member.status = "ok"
                search._record_benchmarked_member(member)
            return members

        search.benchmark_population = Mock(side_effect=benchmark_population)
        timing_rounds = (
            {0: 10.0, 1: 9.8, 2: 10.3},
            {0: 10.0, 1: 10.4, 2: 9.5},
            {0: 10.2, 1: 9.9, 2: 9.6},
        )

        def mirrored_rebenchmark(members, *, desc, target_ms):
            values = timing_rounds[mirrored_rebenchmark.calls]
            mirrored_rebenchmark.calls += 1
            timings = [
                values[member.config.config[cute_flash.FLASH_WAIT_HINT_KEY]]
                for member in members
            ]
            for member, timing in zip(members, timings, strict=True):
                member.perfs.append(timing)
            order = list(range(len(members)))
            return MirroredBenchmarkTrace(
                orders=[order, list(reversed(order))],
                elapsed_ms=[timings, list(reversed(timings))],
                medians_ms=timings,
                sweep_count=2,
                calls_per_sample=1,
                total_calls=2,
            )

        mirrored_rebenchmark.calls = 0
        search.mirrored_rebenchmark = Mock(side_effect=mirrored_rebenchmark)

        selected = search.run_terminal_refinement(best)

        self.assertEqual(
            selected.config.config[cute_flash.FLASH_WAIT_HINT_KEY],
            2,
        )
        self.assertEqual(search._autotune_metrics.num_generations, 20)
        self.assertEqual(search.mirrored_rebenchmark.call_count, 3)
        self.assertEqual(
            [
                invocation.kwargs["target_ms"]
                for invocation in search.mirrored_rebenchmark.call_args_list
            ],
            [200.0, 200.0, 5000.0],
        )
        transcript = search._autotune_metrics.search_phase_metrics[
            "terminal_coordinate_refinement"
        ]
        self.assertTrue(
            all(
                "cute_vector_widths" not in entry["config"]
                for entry in transcript["config_manifest"].values()
            )
        )
        self.assertTrue(transcript["completed"])
        self.assertEqual(transcript["termination_reason"], "round_limit")
        self.assertEqual(transcript["rounds_started"], 2)
        self.assertEqual(transcript["rounds_completed"], 2)
        self.assertEqual(
            [round_metric["accepted"] for round_metric in transcript["rounds"]],
            [True, True],
        )
        self.assertEqual(transcript["new_candidate_count"], 2)
        self.assertEqual(transcript["preterminal_num_configs_tested"], 100)
        self.assertEqual(search._autotune_metrics.num_configs_tested, 102)
        self.assertEqual(transcript["preterminal_registry_config_count"], 1)
        self.assertRegex(
            transcript["preterminal_registry_config_ids_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(transcript["beam_width"], 2)
        self.assertEqual(transcript["projection_parent_count"], 3)
        self.assertEqual(transcript["maximum_projection_parent_count"], 3)
        self.assertTrue(
            all(
                len(round_metric["parent_config_ids"]) <= transcript["beam_width"]
                for round_metric in transcript["rounds"]
            )
        )
        self.assertEqual(transcript["reused_candidate_count"], 0)
        self.assertEqual(transcript["intra_terminal_reused_candidate_count"], 1)
        self.assertEqual(
            transcript["accepted_config_ids"],
            [canonical_config_id(make_member(value, 1.0).config) for value in (1, 2)],
        )
        self.assertNotIn(
            canonical_config_id(make_member(1, 1.0).config),
            transcript["rounds"][1]["beam_config_ids"],
        )
        self.assertEqual(
            transcript["final_config_id"], canonical_config_id(selected.config)
        )
        self.assertEqual(
            transcript["confirmation"]["candidate_config_ids"],
            [
                canonical_config_id(make_member(value, 1.0).config)
                for value in (0, 1, 2)
            ],
        )
        self.assertEqual(
            transcript["rounds"][0]["measurement"]["elapsed_ms"][0],
            [10.0, 9.8, 10.3],
        )

    def test_lfbo_flash_terminal_coordinate_refinement_skips_multi_shape(self):
        best = Mock(spec=PopulationMember)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=None,
        )
        search.benchmark_provider = object.__new__(MultiShapeBenchmarkProvider)

        self.assertIs(search.run_terminal_refinement(best), best)
        self.assertEqual(search._autotune_metrics.search_phase_metrics, {})

    def test_lfbo_flash_terminal_coordinate_refinement_finalizes_missing_leaf(self):
        best = PopulationMember(
            fn=lambda: None,
            perfs=[10.0],
            flat_values=[1],
            config=helion.Config(block_sizes=[1, 128, 128]),
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._cute_flash_lane_policy_enabled = True
        search._terminal_refinement_members = {}
        search._autotune_metrics = SimpleNamespace(
            num_generations=20,
            num_configs_tested=100,
            search_phase_metrics={},
        )
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=None,
        )
        search.radius = 2
        search.min_improvement_delta = 0.001
        search.log = Mock()

        self.assertIs(search.run_terminal_refinement(best), best)
        transcript = search._autotune_metrics.search_phase_metrics[
            "terminal_coordinate_refinement"
        ]
        self.assertTrue(transcript["completed"])
        self.assertEqual(transcript["termination_reason"], "no_candidates")
        self.assertEqual(
            transcript["confirmation"]["skipped_reason"],
            "missing_structural_leaf",
        )
        self.assertRegex(transcript["config_manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_terminal_refinement_registry_preserves_successful_duplicate(self):
        config = helion.Config(block_sizes=[1, 128, 128])
        successful = PopulationMember(
            fn=lambda: None,
            perfs=[10.0],
            flat_values=[1],
            config=config,
            status="ok",
        )
        failed = PopulationMember(
            fn=lambda: None,
            perfs=[math.inf],
            flat_values=[1],
            config=copy.deepcopy(config),
            status="error",
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
        search._terminal_refinement_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search._benchmarked_members = {}

        search._record_benchmarked_member(successful)
        search._record_benchmarked_member(failed)

        stored = search._terminal_refinement_members[config]
        self.assertEqual(stored.status, "ok")
        self.assertEqual(stored.perf, 10.0)

    def test_terminal_refinement_registry_drops_quarantined_config(self):
        config = helion.Config(block_sizes=[1, 128, 128])
        successful = PopulationMember(
            fn=lambda: None,
            perfs=[10.0],
            flat_values=[1],
            config=config,
            status="ok",
        )
        quarantined = PopulationMember(
            fn=lambda: None,
            perfs=[math.inf],
            flat_values=[1],
            config=copy.deepcopy(config),
            status="error",
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True, backend_name="cute"
        )
        search._terminal_refinement_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search._benchmarked_members = {}

        search._record_benchmarked_member(successful)
        self.assertIn(config, search._terminal_refinement_members)

        search._refresh_benchmarked_members_after_rebenchmark([quarantined])

        self.assertNotIn(config, search._terminal_refinement_members)

    def test_lfbo_flash_terminal_refinement_skips_budgeted_search(self):
        best = PopulationMember(
            fn=lambda: None,
            perfs=[10.0],
            flat_values=[0],
            config=helion.Config(block_sizes=[1]),
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=1.0,
        )
        search.benchmark_population = Mock()
        search.mirrored_rebenchmark = Mock()

        self.assertIs(search.run_terminal_refinement(best), best)
        self.assertEqual(search._autotune_metrics.search_phase_metrics, {})
        search.benchmark_population.assert_not_called()
        search.mirrored_rebenchmark.assert_not_called()

    def test_lfbo_flash_terminal_coordinate_refinement_crosses_beam_valley(self):
        def member(
            e2e_offset: int,
            rescale_threshold: int,
            perf: float | None,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[e2e_offset, rescale_threshold],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=True,
                    cute_flash_e2e_offset=e2e_offset,
                    cute_flash_rescale_threshold=float(rescale_threshold),
                ),
                status="unknown" if perf is None else "ok",
            )

        best = member(7, 12, 10.0)
        intermediate_config = member(5, 12, None).config
        endpoint_config = member(5, 4, None).config

        def projections(flat) -> list[CoordinateNeighborProjection]:
            if flat == [7, 12.0]:
                return [
                    CoordinateNeighborProjection(
                        0,
                        cute_flash.FLASH_E2E_OFFSET_KEY,
                        None,
                        7,
                        5,
                        "candidate",
                        [5, 12.0],
                        intermediate_config,
                    )
                ]
            if flat == [5, 12.0]:
                return [
                    CoordinateNeighborProjection(
                        1,
                        cute_flash.FLASH_RESCALE_THRESHOLD_KEY,
                        None,
                        12.0,
                        4.0,
                        "candidate",
                        [5, 4.0],
                        endpoint_config,
                    )
                ]
            return []

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.args = ()
        search.best_perf_so_far = best.perf
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            backend=SimpleNamespace(generated_source_hash=lambda _fn: None),
        )
        search.config_gen = SimpleNamespace(
            flatten=lambda config: [
                config.config[cute_flash.FLASH_E2E_OFFSET_KEY],
                config.config[cute_flash.FLASH_RESCALE_THRESHOLD_KEY],
            ],
            canonicalize_coordinate_projections=lambda projections, *, base_config: (
                projections
            ),
        )
        search._flash_leaf_config_generation = Mock(
            return_value=SimpleNamespace(
                flatten=search.config_gen.flatten,
                coordinate_neighbor_projections=lambda flat, *, radius: projections(
                    flat
                ),
            )
        )
        search.flash_structural_search = replace(
            get_effort_profile("full").flash_structural_search,
            terminal_coordinate_rounds=2,
        )
        search._cute_flash_lane_policy_enabled = True
        search._terminal_refinement_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search._benchmarked_members = {}
        search._autotune_metrics = SimpleNamespace(
            num_generations=20,
            search_phase_metrics={},
        )
        search.settings = SimpleNamespace(
            autotune_benchmark_fn=None,
            autotune_budget_seconds=None,
        )
        search.radius = 2
        search.min_improvement_delta = 0.001
        search.log = Mock()
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.make_unbenchmarked = Mock(
            side_effect=lambda flat: member(int(flat[0]), int(flat[1]), None)
        )

        def benchmark_population(members, **_kwargs):
            attempt_perfs = {(5, 12): 10.1, (5, 4): 9.9}
            for candidate in members:
                values = tuple(int(value) for value in candidate.flat_values)
                candidate.perfs.append(attempt_perfs[values])
                candidate.status = "ok"
                search._record_benchmarked_member(candidate)
            return members

        search.benchmark_population = Mock(side_effect=benchmark_population)

        timing_rounds = (
            {(7, 12): 10.0, (5, 12): 10.005},
            {(7, 12): 10.0, (5, 12): 10.02, (5, 4): 9.8},
            {(7, 12): 10.0, (5, 12): 10.02, (5, 4): 9.75},
        )

        def mirrored_rebenchmark(members, **_kwargs):
            values = timing_rounds[mirrored_rebenchmark.calls]
            mirrored_rebenchmark.calls += 1
            timings = [
                values[tuple(int(value) for value in candidate.flat_values)]
                for candidate in members
            ]
            for candidate, timing in zip(members, timings, strict=True):
                candidate.perfs.append(timing)
            order = list(range(len(members)))
            return MirroredBenchmarkTrace(
                orders=[order, list(reversed(order))],
                elapsed_ms=[timings, list(reversed(timings))],
                medians_ms=timings,
                sweep_count=2,
                calls_per_sample=1,
                total_calls=2,
            )

        mirrored_rebenchmark.calls = 0
        search.mirrored_rebenchmark = Mock(side_effect=mirrored_rebenchmark)

        selected = search.run_terminal_refinement(best)

        self.assertEqual(selected.config, endpoint_config)
        self.assertEqual(search.mirrored_rebenchmark.call_count, 3)
        transcript = search._autotune_metrics.search_phase_metrics[
            "terminal_coordinate_refinement"
        ]
        self.assertEqual(transcript["rounds_completed"], 2)
        self.assertEqual(transcript["termination_reason"], "round_limit")
        self.assertEqual(
            [round_metric["accepted"] for round_metric in transcript["rounds"]],
            [False, True],
        )
        self.assertIn(
            canonical_config_id(intermediate_config),
            transcript["rounds"][1]["parent_config_ids"],
        )
        self.assertEqual(
            transcript["rounds"][0]["candidate_results"][0]["selection_perf"],
            10.005,
        )
        self.assertTrue(transcript["confirmation"]["accepted"])
        self.assertEqual(
            transcript["final_config_id"], canonical_config_id(endpoint_config)
        )

    def test_pattern_search_block_size_pair_neighbors(self):
        search = PatternSearch.__new__(PatternSearch)
        search._visited = set()
        search.config_gen = SimpleNamespace(
            flat_spec=[
                PowerOfTwoFragment(16, 128, 32),
                PowerOfTwoFragment(16, 128, 64),
                EnumFragment(("a", "b")),
            ],
            block_size_indices=[0, 1],
            overridden_flat_indices=set(),
            config_spec=SimpleNamespace(tensor_numel_constraints=[]),
        )
        search.num_neighbors_cap = -1

        base = [32, 64, "a"]
        neighbors = search._generate_neighbors(base)

        def diff_count(flat):
            return sum(
                1
                for current, original in zip(flat, base, strict=False)
                if current != original
            )

        pair_neighbors = [
            flat for flat in neighbors if diff_count(flat) == 2 and flat[2] == "a"
        ]
        expected = [
            [16, 32, "a"],
            [16, 128, "a"],
            [64, 32, "a"],
            [64, 128, "a"],
        ]
        self.assertEqual(sorted(pair_neighbors), sorted(expected))

    def test_pattern_search_skips_overridden_indices(self):
        """Neighbors are not generated along overridden (frozen) indices."""
        search = PatternSearch.__new__(PatternSearch)
        search._visited = set()
        search.config_gen = SimpleNamespace(
            flat_spec=[
                PowerOfTwoFragment(16, 128, 32),  # block_size[0] — index 0
                PowerOfTwoFragment(16, 128, 64),  # block_size[1] — index 1
                EnumFragment(("a", "b")),  # some enum — index 2
            ],
            block_size_indices=[0, 1],
            overridden_flat_indices={1},  # freeze block_size[1]
        )
        search.num_neighbors_cap = -1

        base = [32, 64, "a"]
        neighbors = search._generate_neighbors(base)

        # No neighbor should change index 1 (frozen)
        for flat in neighbors:
            self.assertEqual(flat[1], 64)

        # Neighbors should still vary indices 0 and 2
        changed_indices = set()
        for flat in neighbors:
            for i, (v, b) in enumerate(zip(flat, base, strict=False)):
                if v != b:
                    changed_indices.add(i)
        self.assertIn(0, changed_indices)
        self.assertIn(2, changed_indices)
        self.assertNotIn(1, changed_indices)

        # No block-size pair neighbors should be generated (only 1 non-frozen block index)
        pair_neighbors = [
            flat
            for flat in neighbors
            if sum(1 for v, b in zip(flat, base, strict=False) if v != b) == 2
        ]
        self.assertEqual(pair_neighbors, [])

    def test_differential_mutation_skips_overridden_indices(self):
        """Differential mutation does not mutate overridden indices."""
        random.seed(42)
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        overrides = {"num_warps": 8}
        gen = ConfigGeneration(spec, overrides=overrides)

        # Find the num_warps flat index
        warp_idx = gen.num_warps_index
        self.assertIn(warp_idx, gen.overridden_flat_indices)

        base = gen.default_flat()
        a = gen.random_flat()
        b = gen.random_flat()
        c = gen.random_flat()

        # Run many mutations — overridden index should never change
        for _ in range(50):
            result = gen.differential_mutation(base, a, b, c, crossover_rate=0.9)
            self.assertEqual(result[warp_idx], base[warp_idx])

    def test_population_member_canonicalizes_overridden_flat_values(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        gen = ConfigGeneration(spec, overrides={"num_warps": 8})
        raw_flat = gen.default_flat()
        original_flat = copy.deepcopy(raw_flat)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.config_gen = gen

        member = search.make_unbenchmarked(raw_flat)

        self.assertIsNotNone(member)
        assert member is not None
        self.assertEqual(raw_flat, original_flat)
        self.assertIsNot(member.flat_values, raw_flat)
        self.assertEqual(member.config.num_warps, 8)
        self.assertEqual(member.flat_values[gen.num_warps_index], 8)
        self.assertEqual(member.flat_values, gen.flatten(member.config))
        gen.encode_config(member.flat_values)

        [(seed_flat, seed_config)] = gen.user_seed_flat_config_pairs(
            [helion.Config(num_warps=4)]
        )
        self.assertEqual(seed_config.num_warps, 8)
        self.assertEqual(seed_flat, gen.flatten(seed_config))

    def test_scalar_list_override_has_encodable_flat_values(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        spec = basic_kernels.add.bind(args).config_spec
        gen = ConfigGeneration(spec, overrides={"indexing": "pointer"})

        flat, config = gen.canonicalize_flat(gen.default_flat())

        indexing_indices, is_sequence = gen._key_to_flat_indices["indexing"]
        self.assertFalse(is_sequence)
        self.assertEqual(len(indexing_indices), 1)
        self.assertEqual(flat[indexing_indices[0]], ["pointer"] * spec.indexing.length)
        self.assertEqual(flat, gen.flatten(config))
        gen.encode_config(flat)

    def test_population_adopts_config_filter_replacement(self):
        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)
        search = PatternSearch(bound_kernel, args, initial_population=1)
        search.benchmark_provider = SimpleNamespace(take_effective_source_repairs=dict)
        member = search.make_unbenchmarked(search.config_gen.default_flat())
        assert member is not None
        replacement = helion.Config.from_dict({**member.config.config, "num_warps": 8})
        result = SimpleNamespace(
            config=replacement,
            perf=float("inf"),
            fn=lambda: None,
            status="error",
            compile_time=None,
        )

        with patch.object(search, "benchmark_batch", return_value=[result]):
            search.benchmark_population([member])

        self.assertIs(member.config, replacement)
        self.assertEqual(member.flat_values, search.config_gen.flatten(replacement))
        self.assertEqual(
            member.flat_values[search.config_gen.num_warps_index],
            8,
        )
        search.config_gen.encode_config(member.flat_values)

    def test_lfbo_pattern_search_skips_overridden_indices(self):
        """LFBOPatternSearch._generate_neighbors skips overridden indices."""
        random.seed(123)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.num_neighbors = 50
        search.radius = 2
        search.config_gen = SimpleNamespace(
            flat_spec=[
                PowerOfTwoFragment(16, 128, 32),  # block_size[0]
                PowerOfTwoFragment(16, 128, 64),  # block_size[1]
                PowerOfTwoFragment(2, 16, 4),  # num_warps
                EnumFragment(("a", "b", "c")),  # some enum
                BooleanFragment(),  # some boolean
            ],
            block_size_indices=[0, 1],
            num_warps_index=2,
            overridden_flat_indices={1, 2},  # freeze block_size[1] and num_warps
        )
        search.num_neighbors_cap = -1

        base = [32, 64, 4, "b", True]
        neighbors = search._generate_neighbors(base)

        # No neighbor should change indices 1 or 2
        for flat in neighbors:
            self.assertEqual(flat[1], 64)
            self.assertEqual(flat[2], 4)

    def test_lfbo_flash_starting_points_retain_all_live_families(self):
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        degree2 = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        degree1 = member(0.6, "fa4_2cta", "deg1_16x8", 1)
        two_cta_plain = member(0.7, "fa4_2cta", "1x1", 2)
        fa4 = member(0.8, "fa4", "1x1", 3)
        ws = member(0.9, "ws_overlap", "1x1", 4)
        clc = member(1.0, "fa4_clc", "1x1", 5)
        tma_4d = member(5.0, "fa4_tma_4d", "1x1", 6)

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 8
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=None)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [
            degree2,
            degree1,
            two_cta_plain,
            fa4,
            ws,
            clc,
            tma_4d,
        ]
        paths = search._select_starting_paths()
        self.assertEqual(len(paths), search.copies)
        self.assertEqual(len({item.config for item, _constraints in paths}), 7)
        self.assertEqual(
            {
                item.config.config["cute_flash_pipeline_family"]
                for item, _constraints in paths
            },
            {"fa4_2cta", "fa4", "ws_overlap", "fa4_clc", "fa4_tma_4d"},
        )
        self.assertEqual(
            sum(
                item.config.config["cute_flash_pipeline_family"] == "fa4_2cta"
                for item, _constraints in paths
            ),
            4,
        )
        self.assertEqual(
            paths[-1][1],
            (),
        )
        self.assertIs(paths[-1][0], degree2)
        self.assertEqual(
            dict(next(c for item, c in paths if item is two_cta_plain))[
                "cute_flash_pipeline_family"
            ],
            "fa4_2cta",
        )
        self.assertIn(degree1.config, {item.config for item, _ in paths})
        self.assertEqual(
            {
                item["family"]
                for item in search._autotune_metrics.search_phase_metrics[
                    "retained_families"
                ]
            },
            {"fa4_2cta", "fa4", "ws_overlap", "fa4_clc", "fa4_tma_4d"},
        )
        retained_paths = [
            path
            for family in search._autotune_metrics.search_phase_metrics[
                "retained_families"
            ]
            for path in family["starting_paths"]
        ]
        self.assertEqual(
            [path["config_id"] for path in retained_paths if path["unrestricted"]],
            [canonical_config_id(degree2.config)],
        )
        self.assertTrue(
            all(
                family["score_compound_packet"] is None
                for family in search._autotune_metrics.search_phase_metrics[
                    "retained_families"
                ]
            )
        )

    def test_lfbo_flash_family_probe_covers_every_family_and_compound_leaf(self):
        def member(
            perf: float, family: str, packet: str, softmax_disc: bool, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_softmax_disc=softmax_disc,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        fa4 = member(1.0, "fa4", "1x1", True, 0)
        fa4_better_protocol = member(0.9, "fa4", "1x1", False, 1)
        ws = member(1.1, "ws_overlap", "1x1", True, 2)
        degree2 = member(0.7, "fa4_2cta", "deg2_16x6", False, 3)
        degree1 = member(0.8, "fa4_2cta", "deg1_16x8", False, 4)
        unqualified = member(0.1, "fa4_2cta", "deg2_16x6", False, 5)
        population = [
            fa4,
            fa4_better_protocol,
            ws,
            degree2,
            degree1,
            unqualified,
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search._flash_qualified_compound_config_ids = {
            flash_structural_leaf_from_config(degree2.config.config): {
                canonical_config_id(degree2.config)
            },
            flash_structural_leaf_from_config(degree1.config.config): {
                canonical_config_id(degree1.config)
            },
        }

        paths = search._flash_family_probe_paths(population)

        self.assertEqual(len(paths), 5)
        self.assertEqual(paths[-1], (degree2, None, (), True))
        self.assertEqual(
            {path[0].config for path in paths[:-1]},
            {
                fa4_better_protocol.config,
                ws.config,
                degree2.config,
                degree1.config,
            },
        )
        self.assertTrue(all(path[2] for path in paths[:-1]))
        self.assertNotIn(unqualified.config, {path[0].config for path in paths})

    def test_lfbo_flash_post_probe_scores_control_parent_promotion(self):
        def member(perf: float, family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initially_slow = member(1.5, "fa4_2cta", 0)
        probe_winner = member(0.8, "fa4_2cta", 1)
        qualification_leader = member(1.0, "fa4", 2)
        unrestricted_family_base = member(2.0, "ws_overlap", 3)
        unrestricted_winner = member(0.5, "ws_overlap", 4)
        population = [
            initially_slow,
            probe_winner,
            qualification_leader,
            unrestricted_family_base,
            unrestricted_winner,
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=1)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search._flash_parent_score_config_ids = {
            canonical_config_id(member.config)
            for member in (
                initially_slow,
                probe_winner,
                qualification_leader,
                unrestricted_family_base,
            )
        }
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (unrestricted_winner, ()))
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertTrue(
            next(
                family["parent_promoted"]
                for family in retained
                if family["family"] == "fa4_2cta"
            )
        )
        self.assertFalse(
            next(
                family["parent_promoted"]
                for family in retained
                if family["family"] == "ws_overlap"
            )
        )

    def test_lfbo_flash_starting_points_keep_distinct_softmax_protocols(self):
        def member(perf: float, softmax_disc: bool) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=softmax_disc,
                ),
                status="ok",
            )

        whole_row = member(1.0, True)
        chunked = member(1.1, False)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 3
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [whole_row, chunked]

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (whole_row, ()))
        self.assertEqual(paths[0][0], whole_row)
        self.assertEqual(paths[1][0], chunked)
        self.assertEqual(
            dict(paths[1][1]),
            {
                "cute_flash_pipeline_family": "fa4",
                "cute_flash_softmax_disc": False,
            },
        )
        family = search._autotune_metrics.search_phase_metrics["retained_families"][0]
        self.assertTrue(family["score_softmax_disc"])
        self.assertEqual(
            {path["softmax_disc"] for path in family["starting_paths"]},
            {False, True},
        )

    def test_lfbo_flash_constructor_reserves_structural_starting_paths(self):
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None

        def initialize_pattern_search(
            search, *, kernel, args, copies: int, **_kwargs
        ) -> None:
            search.config_spec = kernel.config_spec
            search.settings = kernel.settings
            search.benchmark_provider = SimpleNamespace()
            search._terminal_refinement_members = None
            search.args = args
            search.copies = copies
            search.config_gen = SimpleNamespace(
                flash_structural_starting_path_limit=lambda **_kwargs: 17,
                flash_structural_family_probe_path_limit=lambda _cap, _generations: 18,
            )

        flash_kernel = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            settings=Settings(autotune_budget_seconds=None),
        )
        budgeted_flash_kernel = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            settings=Settings(autotune_budget_seconds=1.0),
        )
        multi_shape_args = _MultiShapeAutotuneArgs(
            cases=((Mock(), (object(),)),),
            aggregation="geomean",
            relative_to=None,
            cache_tag=None,
            workload_key=("test",),
        )
        non_flash_kernel = SimpleNamespace(
            config_spec=SimpleNamespace(cute_flash_search_enabled=False),
            settings=Settings(autotune_budget_seconds=None),
        )
        with (
            patch("helion.autotuner.surrogate_pattern_search.HAS_ML_DEPS", True),
            patch.object(PatternSearch, "__init__", initialize_pattern_search),
        ):
            flash_search = LFBOPatternSearch(
                flash_kernel,
                (),
                copies=2,
                flash_structural_search=policy,
            )
            budgeted_flash_search = LFBOPatternSearch(
                budgeted_flash_kernel,
                (),
                copies=2,
                flash_structural_search=policy,
            )
            multi_shape_search = LFBOPatternSearch(
                flash_kernel,
                multi_shape_args,
                copies=2,
                flash_structural_search=policy,
            )
            non_flash_search = LFBOPatternSearch(
                non_flash_kernel,
                (),
                copies=2,
                flash_structural_search=policy,
            )

        self.assertEqual(flash_search._flash_promoted_path_limit, 17)
        self.assertEqual(flash_search._flash_family_probe_path_limit, 18)
        self.assertEqual(flash_search.copies, 18)
        self.assertEqual(flash_search._terminal_refinement_members, {})
        self.assertIsNone(budgeted_flash_search._terminal_refinement_members)
        self.assertIsNone(multi_shape_search._terminal_refinement_members)
        self.assertEqual(non_flash_search.copies, 2)

    def test_lfbo_flash_starting_points_do_not_prune_close_sibling_leaf(self):
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        best_family = member(1.0, "fa4", "1x1", 0)
        best_family_second = member(1.001, "fa4", "1x1", 1)
        degree2 = member(1.01, "fa4_2cta", "deg2_16x6", 2)
        degree1 = member(1.011, "fa4_2cta", "deg1_16x8", 3)
        two_cta_plain = member(1.015, "fa4_2cta", "1x1", 4)
        clc = member(3.0, "fa4_clc", "1x1", 5)
        tma = member(3.1, "fa4_tma_4d", "1x1", 6)

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 6
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [
            best_family,
            best_family_second,
            degree2,
            degree1,
            two_cta_plain,
            clc,
            tma,
        ]

        paths = search._select_starting_paths()

        selected = {item.config for item, _constraints in paths}
        self.assertEqual(paths[-1], (best_family, ()))
        self.assertEqual(
            selected,
            {
                best_family.config,
                degree2.config,
                degree1.config,
                two_cta_plain.config,
                best_family_second.config,
            },
        )
        self.assertNotIn(clc.config, selected)
        self.assertNotIn(tma.config, selected)
        self.assertEqual(paths[-1], (best_family, ()))
        self.assertEqual(
            dict(next(c for item, c in paths if item is degree2))[
                "cute_flash_exp2_packet"
            ],
            "deg2_16x6",
        )
        self.assertEqual(
            dict(next(c for item, c in paths if item is degree1))[
                "cute_flash_exp2_packet"
            ],
            "deg1_16x8",
        )

    def test_lfbo_flash_family_score_ignores_compound_leaf_count(self) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        many_leaf_family = [
            member(1.5, "fa4_2cta", "1x1", 0),
            member(0.9, "fa4_2cta", "deg2_16x6", 1),
            member(0.95, "fa4_2cta", "deg1_16x8", 2),
        ]
        ordinary_families = [
            member(perf, family, "1x1", index + 3)
            for index, (perf, family) in enumerate(
                (
                    (0.8, "fa4"),
                    (1.0, "ws_overlap"),
                    (1.1, "fa4_clc"),
                    (1.2, "fa4_tma_4d"),
                )
            )
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 5
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [*many_leaf_family, *ordinary_families]

        search._select_starting_paths()

        retained = {
            item["family"]
            for item in search._autotune_metrics.search_phase_metrics[
                "retained_families"
            ]
        }
        self.assertEqual(retained, {"fa4", "ws_overlap", "fa4_clc", "fa4_tma_4d"})
        self.assertNotIn("fa4_2cta", retained)

    def test_lfbo_flash_global_compound_winner_is_unrestricted(self) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        winner = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        population = [
            winner,
            member(1.5, "fa4_2cta", "1x1", 1),
            member(1.0, "fa4", "1x1", 2),
            member(1.1, "ws_overlap", "1x1", 3),
            member(1.2, "fa4_clc", "1x1", 4),
            member(1.3, "fa4_tma_4d", "1x1", 5),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (winner, ()))
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(
            [
                path["config_id"]
                for family in retained
                for path in family["starting_paths"]
                if path["unrestricted"]
            ],
            [canonical_config_id(winner.config)],
        )
        self.assertEqual(
            {family["family"] for family in retained},
            {"fa4_2cta", "fa4", "ws_overlap", "fa4_clc"},
        )
        self.assertFalse(
            next(
                family["parent_promoted"]
                for family in retained
                if family["family"] == "fa4_2cta"
            )
        )

    def test_lfbo_flash_dominated_compound_winner_gets_only_global_path(
        self,
    ) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        winner = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        dominated_parent = member(3.0, "fa4_2cta", "1x1", 1)
        population = [
            winner,
            dominated_parent,
            member(1.0, "fa4", "1x1", 2),
            member(1.1, "ws_overlap", "1x1", 3),
            member(1.2, "fa4_clc", "1x1", 4),
            member(1.3, "fa4_tma_4d", "1x1", 5),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (winner, ()))
        self.assertNotIn(dominated_parent.config, {item.config for item, _ in paths})
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(
            [family["family"] for family in retained],
            ["fa4_2cta", "fa4", "ws_overlap", "fa4_clc"],
        )
        self.assertFalse(retained[0]["parent_promoted"])
        self.assertEqual(len(retained[0]["starting_paths"]), 1)
        self.assertTrue(retained[0]["starting_paths"][0]["unrestricted"])
        self.assertTrue(all(family["parent_promoted"] for family in retained[1:]))

    def test_lfbo_flash_compound_only_family_gets_only_global_path(self) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        winner = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        failed_parent = member(math.inf, "fa4_2cta", "1x1", 1)
        failed_parent.status = "error"
        population = [
            winner,
            failed_parent,
            member(1.0, "fa4", "1x1", 2),
            member(1.1, "ws_overlap", "1x1", 3),
            member(1.2, "fa4_clc", "1x1", 4),
            member(1.3, "fa4_tma_4d", "1x1", 5),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (winner, ()))
        self.assertNotIn(failed_parent.config, {item.config for item, _ in paths})
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(
            [family["family"] for family in retained],
            ["fa4_2cta", "fa4", "ws_overlap", "fa4_clc"],
        )
        self.assertFalse(retained[0]["parent_promoted"])
        self.assertEqual(len(retained[0]["starting_paths"]), 1)
        self.assertEqual(sum(not constraints for _member, constraints in paths), 1)

    def test_lfbo_flash_sparse_successes_retain_each_compound_leaf(self) -> None:
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        winner = member(0.5, "fa4_2cta", "deg2_16x6", 0)
        ordinary = member(1.0, "fa4", "1x1", 1)
        compound_only_filler = member(1.1, "fa4_2cta", "deg1_16x8", 2)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 5
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [winner, ordinary, compound_only_filler]

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (winner, ()))
        self.assertEqual(len(paths), 4)
        self.assertEqual(sum(not constraints for _member, constraints in paths), 1)
        self.assertIn(
            compound_only_filler.config, {member.config for member, _ in paths}
        )

    def test_lfbo_flash_dynamic_capacity_covers_asymmetric_promoted_families(
        self,
    ) -> None:
        def member(
            perf: float,
            family: str,
            packet: str,
            softmax_disc: bool,
            wait_hint: int,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_softmax_disc=softmax_disc,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        compound = [
            member(0.5, "fa4_2cta", "deg2_16x6", False, 0),
            member(0.6, "fa4_2cta", "deg1_16x8", False, 1),
        ]
        wide_family = [
            member(1.0, "fa4", "1x1", True, 2),
            member(1.2, "fa4", "1x1", True, 3),
            member(1.01, "fa4", "1x1", False, 4),
            member(1.21, "fa4", "1x1", False, 5),
        ]
        narrow_family = [
            member(1.1, "ws_overlap", "1x1", True, 6),
            member(1.3, "ws_overlap", "1x1", True, 7),
        ]
        unpromoted = member(5.0, "fa4_clc", "1x1", True, 8)
        population = [*compound, *wide_family, *narrow_family, unpromoted]
        catalog = list(
            dict.fromkeys(
                flash_structural_leaf_from_config(item.config.config)
                for item in population
            )
        )
        self.assertNotIn(None, catalog)

        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        policy = replace(
            policy,
            retained_families=2,
            retained_candidates_per_leaf=2,
        )
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        with patch.object(
            generation, "flash_structural_leaf_catalog", return_value=catalog
        ):
            capacity = generation.flash_structural_starting_path_limit(
                minimum=1,
                retained_families=policy.retained_families,
                retained_candidates_per_leaf=policy.retained_candidates_per_leaf,
            )

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = capacity
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = policy
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        selected = {item.config for item, _constraints in paths}
        self.assertEqual(capacity, 8)
        self.assertEqual(len(paths), capacity)
        self.assertEqual(
            selected,
            {
                item.config
                for item in population
                if item not in (wide_family[3], unpromoted)
            },
        )
        self.assertEqual(paths[-1], (compound[0], ()))
        self.assertTrue({item.config for item in compound} <= selected)
        self.assertEqual(
            {
                (
                    item.config.config["cute_flash_pipeline_family"],
                    item.config.config["cute_flash_softmax_disc"],
                )
                for item, _constraints in paths
                if item.config.config["cute_flash_exp2_packet"] == "1x1"
            },
            {("fa4", True), ("fa4", False), ("ws_overlap", True)},
        )

    def test_lfbo_flash_starting_points_replace_dominated_family(self):
        def member(
            perf: float, family: str, packet: str, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        best = member(1.0, "fa4", "1x1", 0)
        best_sibling = member(1.1, "fa4", "1x1", 1)
        best_third = member(1.15, "fa4", "1x1", 5)
        close_leaf = member(1.2, "fa4", "deg2_16x6", 2)
        competitive = member(1.3, "fa4_2cta", "1x1", 3)
        dominated = member(2.01, "ws_overlap", "1x1", 4)

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 5
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = [
            best,
            best_sibling,
            best_third,
            close_leaf,
            competitive,
            dominated,
        ]

        paths = search._select_starting_paths()

        selected = {item.config for item, _constraints in paths}
        self.assertEqual(
            selected,
            {
                best.config,
                best_sibling.config,
                close_leaf.config,
                competitive.config,
            },
        )
        self.assertEqual(len(paths), 5)
        self.assertEqual(paths[-1], (best, ()))
        self.assertNotIn(best_third.config, selected)
        self.assertNotIn(dominated.config, selected)
        self.assertEqual(
            {
                item["family"]
                for item in search._autotune_metrics.search_phase_metrics[
                    "retained_families"
                ]
            },
            {"fa4", "fa4_2cta"},
        )

    def test_lfbo_flash_starting_points_break_performance_ties_by_config_id(self):
        def member(wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[1.0],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        population = [member(wait_hint) for wait_hint in (3, 1, 2)]
        expected = sorted(
            population, key=lambda item: canonical_config_id(item.config)
        )[:2]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 3
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(
            [member for member, _constraints in paths],
            [*expected, expected[0]],
        )
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(
            [path["config_id"] for path in retained[0]["starting_paths"]],
            [
                canonical_config_id(expected[0].config),
                canonical_config_id(expected[1].config),
                canonical_config_id(expected[0].config),
            ],
        )
        self.assertEqual(
            [path["unrestricted"] for path in retained[0]["starting_paths"]],
            [False, False, True],
        )

    def test_lfbo_flash_starting_points_retain_alternate_pipeline_depth(self):
        stage_key = "cute_flash_kv_stage"

        def member(perf: float, stage: int, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        population = [
            member(1.0, 2, 0),
            member(1.01, 2, 1),
            member(1.1, 3, 2),
            member(1.2, 4, 3),
            member(1.3, 5, 4),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 4
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population
        lanes = tuple((stage_key, stage) for stage in (2, 3, 4, 5))
        leaf = flash_structural_leaf_from_config(population[0].config.config)
        assert leaf is not None
        search._flash_qualified_pipeline_lanes = {leaf: lanes}
        search._flash_pipeline_lanes = Mock(
            side_effect=AssertionError("must reuse qualified lane catalog")
        )

        paths = search._select_starting_paths()

        self.assertEqual(
            [item.config.config[stage_key] for item, _ in paths], [2, 3, 2]
        )
        self.assertEqual(paths[-1], (population[0], ()))
        self.assertEqual(
            [
                dict(constraints)[stage_key]
                for _item, constraints in paths
                if stage_key in dict(constraints)
            ],
            [3],
        )
        retained = search._autotune_metrics.search_phase_metrics["retained_families"][0]
        self.assertEqual(
            [path["pipeline_lane"] for path in retained["starting_paths"]],
            [
                None,
                {"key": stage_key, "value": 3},
                None,
            ],
        )
        search._flash_pipeline_lanes.assert_not_called()

    def test_lfbo_flash_global_lane_alternate_is_not_crowded_out(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            perf: float,
            family: str,
            packet: str,
            softmax_disc: bool,
            stage: int,
            wait_hint: int,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_softmax_disc=softmax_disc,
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        global_winner = member(1.0, "fa4_2cta", "1x1", True, 2, 0)
        lane_alternate = member(1.2, "fa4_2cta", "1x1", True, 3, 1)
        compound_siblings = [
            member(1.001, "fa4_2cta", "deg2_16x6", True, 2, 2),
            member(1.002, "fa4_2cta", "deg1_16x8", True, 2, 3),
        ]
        family_perfs = (
            ("fa4_2cta", 1.01),
            ("fa4", 1.05),
            ("ws_overlap", 1.06),
            ("fa4_clc", 1.07),
            ("fa4_tma_4d", 1.08),
        )
        ordinary_members: dict[tuple[str, bool], PopulationMember] = {
            ("fa4_2cta", True): global_winner
        }
        wait_hint = 4
        for family, perf in family_perfs:
            for softmax_disc in (True, False):
                if (family, softmax_disc) in ordinary_members:
                    continue
                ordinary_members[(family, softmax_disc)] = member(
                    perf + (0.001 if not softmax_disc else 0.0),
                    family,
                    "1x1",
                    softmax_disc,
                    2,
                    wait_hint,
                )
                wait_hint += 1
        population = [
            global_winner,
            lane_alternate,
            *compound_siblings,
            *(
                ordinary
                for key, ordinary in ordinary_members.items()
                if key != ("fa4_2cta", True)
            ),
        ]
        global_leaf = flash_structural_leaf_from_config(global_winner.config.config)
        assert global_leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        assert search.flash_structural_search is not None
        search.copies = search.flash_structural_search.starting_paths
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population
        search._flash_qualified_pipeline_lanes = {
            global_leaf: ((stage_key, 2), (stage_key, 3))
        }

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (global_winner, ()))
        self.assertEqual(len(paths), 12)
        lane_path = next(path for path in paths if path[0] is lane_alternate)
        self.assertEqual(dict(lane_path[1])[stage_key], 3)
        selected = {item.config for item, _constraints in paths}
        self.assertIn(compound_siblings[0].config, selected)
        self.assertIn(compound_siblings[1].config, selected)
        retained_parent_families = {"fa4_2cta", "fa4", "ws_overlap", "fa4_clc"}
        self.assertEqual(
            {
                (
                    item.config.config["cute_flash_pipeline_family"],
                    item.config.config.get("cute_flash_softmax_disc", True),
                )
                for item, _constraints in paths
                if item.config.config["cute_flash_exp2_packet"] == "1x1"
            },
            {
                (family, softmax_disc)
                for family in retained_parent_families
                for softmax_disc in (True, False)
            },
        )
        self.assertFalse(
            {
                ordinary_members[("fa4_tma_4d", True)].config,
                ordinary_members[("fa4_tma_4d", False)].config,
            }
            & selected
        )
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        self.assertEqual(len(retained), 4)
        global_paths = retained[0]["starting_paths"]
        lane_metric = next(
            path
            for path in global_paths
            if path["config_id"] == canonical_config_id(lane_alternate.config)
        )
        self.assertEqual(
            lane_metric["pipeline_lane"],
            {"key": stage_key, "value": 3},
        )

    def test_lfbo_flash_unpromoted_global_family_preserves_compound_slots(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            perf: float,
            family: str,
            packet: str,
            softmax_disc: bool,
            stage: int,
            wait_hint: int,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet=packet,
                    cute_flash_softmax_disc=softmax_disc,
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        compound_global = member(0.5, "fa4_2cta", "deg2_16x6", True, 2, 0)
        ordinary_kv2 = member(3.0, "fa4_2cta", "1x1", True, 2, 1)
        ordinary_kv3 = member(3.1, "fa4_2cta", "1x1", True, 3, 2)
        compound_filler = member(0.6, "fa4", "deg2_16x6", True, 2, 3)
        promoted_family_perfs = (
            ("fa4", 1.0),
            ("ws_overlap", 1.1),
            ("fa4_clc", 1.2),
            ("fa4_tma_4d", 1.3),
        )
        promoted_ordinary: dict[tuple[str, bool], PopulationMember] = {}
        wait_hint = 4
        for family, perf in promoted_family_perfs:
            for softmax_disc in (True, False):
                promoted_ordinary[(family, softmax_disc)] = member(
                    perf + (0.01 if not softmax_disc else 0.0),
                    family,
                    "1x1",
                    softmax_disc,
                    2,
                    wait_hint,
                )
                wait_hint += 1
        promoted_secondaries = {
            family: member(
                perf + 0.2,
                family,
                "1x1",
                True,
                3,
                wait_hint + index,
            )
            for index, (family, perf) in enumerate(promoted_family_perfs)
        }
        population = [
            compound_global,
            ordinary_kv2,
            ordinary_kv3,
            compound_filler,
            *promoted_ordinary.values(),
            *promoted_secondaries.values(),
        ]
        ordinary_leaf = flash_structural_leaf_from_config(ordinary_kv2.config.config)
        assert ordinary_leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(flash_pipeline_lane_catalog=dict)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(policy, retained_families=4)
        assert search.flash_structural_search is not None
        search.copies = search.flash_structural_search.starting_paths + 1
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population
        search._flash_qualified_pipeline_lanes = {
            ordinary_leaf: ((stage_key, 2), (stage_key, 3))
        }

        paths = search._select_starting_paths()

        self.assertEqual(paths[-1], (compound_global, ()))
        self.assertEqual(len(paths), 15)
        selected = {item.config for item, _constraints in paths}
        self.assertNotIn(ordinary_kv3.config, selected)
        self.assertNotIn(ordinary_kv2.config, selected)
        self.assertIn(compound_filler.config, selected)
        self.assertEqual(
            selected,
            {
                compound_global.config,
                compound_filler.config,
                *(item.config for item in promoted_ordinary.values()),
                *(item.config for item in promoted_secondaries.values()),
            },
        )
        self.assertEqual(
            {
                (
                    item.config.config["cute_flash_pipeline_family"],
                    item.config.config["cute_flash_softmax_disc"],
                )
                for item, _constraints in paths
                if item.config.config["cute_flash_pipeline_family"]
                in dict(promoted_family_perfs)
            },
            set(promoted_ordinary),
        )
        self.assertEqual(
            [
                item
                for item, _constraints in paths
                if item.config.config["cute_flash_exp2_packet"] != "1x1"
            ],
            [compound_global, compound_filler, compound_global],
        )
        retained = search._autotune_metrics.search_phase_metrics["retained_families"]
        global_paths = retained[0]["starting_paths"]
        self.assertFalse(retained[0]["parent_promoted"])
        self.assertEqual(
            {family["family"] for family in retained if family["parent_promoted"]},
            set(dict(promoted_family_perfs)),
        )
        self.assertFalse(
            any(
                path["config_id"] == canonical_config_id(ordinary_kv3.config)
                for path in global_paths
            )
        )
        compound_metric = next(
            path
            for family in retained
            for path in family["starting_paths"]
            if path["config_id"] == canonical_config_id(compound_filler.config)
        )
        self.assertIsNone(compound_metric["pipeline_lane"])

    def test_lfbo_flash_pipeline_lane_neighbor_limits_preserve_budget(self):
        quotas = [
            (("cute_flash_kv_stage", 2), 1),
            (("cute_flash_kv_stage", 3), 2),
            (("cute_flash_kv_stage", 4), 1),
        ]
        limits = LFBOPatternSearch._flash_lane_neighbor_limits(quotas, 300)
        self.assertEqual(sum(limits), 300)
        self.assertEqual(limits, [75, 150, 75])

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        self.assertEqual(search._flash_qualification_neighbor_limit(), 200)
        search.num_neighbors_cap = 3
        self.assertEqual(search._flash_qualification_neighbor_limit(), 3)

    def test_lfbo_flash_pipeline_lane_passes_measure_then_condition_every_value(
        self,
    ):
        lanes = tuple(("cute_flash_kv_stage", value) for value in range(2, 11))

        passes = LFBOPatternSearch._flash_lane_qualification_passes(
            lanes,
            candidate_limit=4,
            conditional_candidates_per_lane=1,
            minimum_passes=2,
        )

        self.assertEqual(len(passes), 6)
        self.assertTrue(
            all(kind == "witness" for batch in passes[:3] for kind, _ in batch)
        )
        self.assertTrue(
            all(kind == "conditional" for batch in passes[3:] for kind, _ in batch)
        )
        for kind in ("witness", "conditional"):
            self.assertEqual(
                [lane for batch in passes for job, lane in batch if job == kind],
                list(lanes),
            )

    def test_lfbo_flash_unmeasured_schedule_anchor_records_provenance(self):
        def member(
            perf: float | None, softmax_disc: bool, flat_value: bool
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[flat_value],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=softmax_disc,
                ),
                status="unknown" if perf is None else "ok",
            )

        initial = member(1.0, True, True)
        anchor = member(None, False, False)
        leaves = [
            flash_structural_leaf_from_config(item.config.config)
            for item in (initial, anchor)
        ]
        assert all(leaf is not None for leaf in leaves)
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flatten=lambda config: [config.config["cute_flash_softmax_disc"]],
            flash_structural_leaf_catalog=lambda: leaves,
            flash_low_confound_schedule_anchor_configs=lambda: [
                anchor.config,
                initial.config,
            ],
            flash_pipeline_lane_catalog=lambda: dict.fromkeys(leaves, ()),
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: [
                initial.config,
                anchor.config,
            ],
        )
        search.copies = 2
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = replace(policy, qualification_rounds=0)
        search.population = [initial]
        search.initial_population = 2
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.make_unbenchmarked = Mock(return_value=anchor)
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("anchor-only test must not generate neighbors")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            self.assertEqual(members, [anchor])
            self.assertEqual(desc, "Low-confound schedule anchors:")
            anchor.perfs = [0.75]
            anchor.status = "ok"

        search.benchmark_population = Mock(side_effect=benchmark)
        visited = {initial.config}

        self.assertEqual(search._run_flash_structural_qualification(visited), 0)

        self.assertIn(anchor.config, visited)
        search.make_unbenchmarked.assert_called_once_with([False])
        search.set_generation.assert_called_once_with(0)
        search.benchmark_population.assert_called_once()
        search._pruned_pattern_search_from.assert_not_called()
        self.assertEqual(search.train_x, [[False]])
        self.assertEqual(search.train_y, [0.75])
        self.assertEqual(search.train_configs, [anchor.config])

        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertEqual(
            metrics["schedule_anchor_design_source"],
            "live family x ordinary packet x softmax protocol from fragment defaults",
        )
        self.assertTrue(metrics["schedule_anchor_pass_planned"])
        self.assertTrue(metrics["schedule_anchor_pass_started"])
        self.assertTrue(metrics["schedule_anchor_complete"])
        self.assertEqual(metrics["schedule_anchor_count"], 2)
        self.assertEqual(metrics["qualification_passes_planned"], 1)
        self.assertEqual(metrics["qualification_passes_started"], 1)
        self.assertEqual(metrics["qualification_passes_completed"], 1)
        self.assertTrue(metrics["completed"])
        self.assertEqual(
            [result["config_id"] for result in metrics["schedule_anchor_results"]],
            [
                canonical_config_id(anchor.config),
                canonical_config_id(initial.config),
            ],
        )
        result = metrics["schedule_anchor_results"][0]
        self.assertEqual(result["config_id"], canonical_config_id(anchor.config))
        self.assertFalse(result["softmax_disc"])
        self.assertEqual(result["attempt_perf"], 0.75)
        self.assertEqual(result["selection_perf"], 0.75)
        self.assertEqual(result["measurement_pass_index"], 1)
        self.assertEqual(
            metrics["measurement_timeline"][1],
            {
                "pass_index": 1,
                "updates": [
                    {
                        "config_id": canonical_config_id(anchor.config),
                        "attempt_perf": 0.75,
                        "selection_perf": 0.75,
                        "status": "ok",
                        "source_hash": None,
                    }
                ],
            },
        )

    def test_lfbo_flash_phase_records_initial_generation_order(self):
        def member(wait_hint: int, perf: float) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=True,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        generated = [member(0, 2.0), member(1, 1.0)]
        leaf = flash_structural_leaf_from_config(generated[0].config.config)
        assert leaf is not None
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flatten=lambda config: [config.config["cute_flash_wait_hint"]],
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=lambda: [
                item.config for item in generated
            ],
            flash_pipeline_lane_catalog=lambda: {leaf: ()},
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: [
                item.config for item in generated
            ],
        )
        search.copies = 2
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = replace(policy, qualification_rounds=0)
        search.population = list(reversed(generated))
        search.initial_population = len(generated)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search.train_source_hashes = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.make_unbenchmarked = Mock(
            side_effect=AssertionError("all exact-space configs are already measured")
        )
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("exhausted exact space needs no neighbors")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search.benchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in generated},
                initial_population=generated,
            ),
            0,
        )

        expected_ids = [canonical_config_id(item.config) for item in generated]
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["initial_config_ids"], expected_ids)
        self.assertEqual(
            [result["config_id"] for result in metrics["initial_results"]],
            expected_ids,
        )
        self.assertEqual(metrics["leaf_results"][0]["initial_config_ids"], expected_ids)
        self.assertEqual(search.population, list(reversed(generated)))

    def test_lfbo_flash_pipeline_lane_conditional_count_is_linear(self):
        stage_key = "cute_flash_kv_stage"

        def member(perf: float | None, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(1.0, 0)
        candidate_groups = [
            (member(None, 1), member(None, 2)),
            (member(None, 3), member(None, 4)),
        ]
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            conditional_candidates_per_pipeline_lane=2,
        )
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        selected_limits: list[int] = []

        def qualification_path(
            _index, current, _visited, _constraints, *, selected_limit, **_kwargs
        ):
            selected_limits.append(selected_limit)
            candidates = candidate_groups[len(selected_limits) - 1]
            return iter(((current, *candidates[: selected_limit - 1]),))

        search._pruned_pattern_search_from = qualification_path

        def benchmark(members, *, desc):
            self.assertEqual(desc, "Structural qualification 2:")
            self.assertCountEqual(
                members,
                [candidate_groups[0][0], candidate_groups[1][0]],
            )
            for candidate in members:
                candidate.perfs = [2.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 2
        )
        self.assertEqual(selected_limits, [2, 2])
        lane = search._autotune_metrics.search_phase_metrics["leaf_results"][0][
            "pipeline_lanes"
        ][0]
        self.assertEqual(len(lane["conditional_candidate_ids"]), 2)
        self.assertEqual(
            search._autotune_metrics.search_phase_metrics["candidate_count"], 2
        )
        self.assertTrue(lane["complete"])

    def test_lfbo_flash_exact_lane_exhaustion_skips_conditional_child(self):
        stage_key = "cute_flash_kv_stage"

        def member(wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[1.0 + wait_hint],
                flat_values=[2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = [member(0), member(1)]
        leaf = flash_structural_leaf_from_config(initial[0].config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: [
                item.config for item in initial
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = initial
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("exhausted lane must not generate a child")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in initial}
            ),
            2,
        )
        search._pruned_pattern_search_from.assert_not_called()
        search.benchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertTrue(metrics["exact_space_enumerated"])
        self.assertTrue(metrics["exact_space_exhausted"])
        self.assertEqual(
            metrics["exact_space_config_ids"],
            [canonical_config_id(item.config) for item in initial],
        )
        lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
        self.assertTrue(lane["space_exhausted"])
        self.assertEqual(lane["space_config_count"], 2)
        self.assertFalse(lane["conditional_required"])
        self.assertEqual(lane["conditional_candidate_ids"], [])
        self.assertTrue(lane["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_conditional_ids_never_relabel_initial_configs(self):
        stage_key = "cute_flash_kv_stage"
        initial = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[2, 0],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_wait_hint=0,
            ),
            status="ok",
        )
        duplicate = PopulationMember(
            fn=lambda: None,
            perfs=[],
            flat_values=[2, 0],
            config=copy.deepcopy(initial.config),
            status="ok",
        )
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._pruned_pattern_search_from = lambda *_args, **_kwargs: iter(
            ((initial, duplicate),)
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 2
        )
        search.benchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
        self.assertTrue(lane["conditional_required"])
        self.assertEqual(lane["conditional_candidate_ids"], [])
        self.assertFalse(lane["complete"])
        self.assertFalse(metrics["completed"])
        self.assertEqual(metrics["candidate_count"], 0)

    def test_lfbo_flash_alias_collapsed_exact_space_uses_raw_initial_budget(self):
        initial = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[0],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_wait_hint=0,
            ),
            status="ok",
        )
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        generation.flat_spec = [EnumFragment(tuple(range(7)))]
        generation._override_values = {}
        generation.unflatten = lambda _flat: initial.config
        self.assertIsNone(generation.flash_exact_effective_search_space_configs(1))
        self.assertEqual(
            generation.flash_exact_effective_search_space_configs(7),
            [initial.config],
        )
        exact_space = Mock(wraps=generation.flash_exact_effective_search_space_configs)
        generation.flash_exact_effective_search_space_configs = exact_space
        generation.encode_config = lambda flat: flat
        generation.flash_structural_leaf_catalog = lambda: [leaf]
        generation.flash_low_confound_schedule_anchor_configs = list
        generation.flash_pipeline_lane_catalog = lambda: {leaf: ()}
        generation.flash_clc_lane_catalog = dict
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = generation
        search.copies = 5
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = 7
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("exhausted leaf must not generate a child")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 0
        )
        search._pruned_pattern_search_from.assert_not_called()
        search.benchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertTrue(metrics["exact_space_exhausted"])
        self.assertEqual(metrics["exact_space_raw_budget"], 7)
        exact_space.assert_called_once_with(7)
        self.assertEqual(metrics["qualification_passes_planned"], 0)
        leaf_metric = metrics["leaf_results"][0]
        self.assertTrue(leaf_metric["space_exhausted"])
        self.assertEqual(leaf_metric["space_config_count"], 1)
        self.assertFalse(leaf_metric["ordinary_search_required"])
        self.assertEqual(leaf_metric["rounds"], [])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_no_lane_exhaustion_is_leaf_local(self):
        def member(perf: float | None, family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[family, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        exhausted = member(1.0, "fa4", 0)
        searchable = member(5.0, "ws_overlap", 0)
        children = [member(None, "ws_overlap", wait_hint) for wait_hint in (1, 2)]
        exhausted_leaf = flash_structural_leaf_from_config(exhausted.config.config)
        searchable_leaf = flash_structural_leaf_from_config(searchable.config.config)
        assert exhausted_leaf is not None and searchable_leaf is not None
        leaves = [exhausted_leaf, searchable_leaf]
        exact_space = [
            exhausted.config,
            searchable.config,
            *(child.config for child in children),
        ]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: leaves,
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: dict.fromkeys(leaves, ()),
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: exact_space,
        )
        search.copies = 5
        search.num_neighbors = 200
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [exhausted, searchable]
        search.initial_population = len(exact_space)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        generated: list[PopulationMember] = []

        def qualification_path(_index, current, *_args, **_kwargs):
            self.assertEqual(
                current.config.config["cute_flash_pipeline_family"], "ws_overlap"
            )
            child = children[len(generated)]
            generated.append(child)
            return iter(((current, child),))

        search._pruned_pattern_search_from = qualification_path
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, **_kwargs):
            self.assertEqual(members, [children[len(generated) - 1]])
            members[0].perfs = [2.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {exhausted.config, searchable.config}
            ),
            2,
        )
        self.assertEqual(generated, children)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertFalse(metrics["exact_space_exhausted"])
        leaf_results = {result["family"]: result for result in metrics["leaf_results"]}
        exhausted_metric = leaf_results["fa4"]
        self.assertTrue(exhausted_metric["space_exhausted"])
        self.assertFalse(exhausted_metric["ordinary_search_required"])
        self.assertEqual(
            [item["neighbor_generation_limit"] for item in exhausted_metric["rounds"]],
            [0, 0],
        )
        searchable_metric = leaf_results["ws_overlap"]
        self.assertFalse(searchable_metric["space_exhausted"])
        self.assertTrue(searchable_metric["ordinary_search_required"])
        self.assertEqual(
            [item["neighbor_generation_limit"] for item in searchable_metric["rounds"]],
            [200, 200],
        )
        self.assertTrue(metrics["completed"])

    def test_flash_clc_refinements_cover_every_legal_divisor(self):
        for num_bh in (96, 120, 360, 720):
            with self.subTest(num_bh=num_bh):
                legal = _flash_clc_heads_per_batch_candidates(num_bh)
                anchors = _flash_clc_heads_per_batch_coverage_candidates(num_bh)
                refinements = _flash_log_maximin_refinements(legal, anchors)

                self.assertEqual({*anchors, *refinements}, set(legal))
                self.assertFalse(set(anchors) & set(refinements))
                self.assertEqual(len(refinements), len(legal) - len(anchors))

    def test_flash_clc_refinements_keep_deterministic_log_maximin_order(self):
        legal = _flash_clc_heads_per_batch_candidates(720)
        anchors = _flash_clc_heads_per_batch_coverage_candidates(720)
        self.assertEqual(
            _flash_log_maximin_refinements(legal, anchors),
            (
                2,
                16,
                90,
                240,
                3,
                36,
                8,
                4,
                45,
                180,
                20,
                72,
                120,
                6,
                12,
                30,
                9,
                18,
                40,
                80,
                48,
                15,
            ),
        )
        self.assertEqual(_flash_log_maximin_refinements((1, 2, 4), ()), (1, 2, 4))

    def test_flash_clc_generation_catalog_witnesses_and_fixed_overrides(self):
        for num_bh in (96, 120):
            with self.subTest(num_bh=num_bh):
                with patch("helion.autotuner.config_spec.get_num_xcd", return_value=1):
                    spec = ConfigSpec(
                        backend=CuteBackend(),
                        target_device_capability=(10, 0),
                        num_sm=148,
                    )
                for block_id, target in enumerate((1, 128, 128)):
                    spec.block_sizes.append(
                        BlockSizeSpec(block_id=block_id, size_hint=target)
                    )
                spec.enable_cute_flash_search(
                    head_dim=64,
                    num_kv=512,
                    num_bh=num_bh,
                    dtype=torch.float16,
                    block_size_targets={0: 1, 1: 128, 2: 128},
                    standard_dense_output=True,
                )
                fragments = cute_flash.flash_autotune_fragments(
                    64,
                    512,
                    num_bh=num_bh,
                    dtype=torch.float16,
                    standard_dense_output=True,
                    pipeline_family_override="fa4_clc",
                )
                overrides = {
                    key: fragment.default()
                    for key, fragment in fragments.items()
                    if key != cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY
                }
                overrides[cute_flash.FLASH_PIPELINE_FAMILY_KEY] = "fa4_clc"
                generation = spec.create_config_generation(overrides=overrides)

                catalogs = generation.flash_clc_lane_catalog()
                self.assertEqual(len(catalogs), 1)
                leaf, catalog = next(iter(catalogs.items()))
                legal = _flash_clc_heads_per_batch_candidates(num_bh)
                anchors = _flash_clc_heads_per_batch_coverage_candidates(num_bh)
                self.assertEqual(catalog["legal_values"], legal)
                self.assertEqual(catalog["anchor_values"], anchors)
                self.assertEqual(
                    catalog["refinement_values"],
                    _flash_log_maximin_refinements(legal, anchors),
                )
                self.assertEqual(set(catalog["attempted_values"]), set(legal))

                witnesses = generation.flash_clc_lane_witnesses()
                self.assertEqual(
                    {
                        value
                        for witness_leaf, value in witnesses
                        if witness_leaf == leaf
                    },
                    set(legal),
                )
                for value in legal:
                    witness = witnesses[(leaf, value)]
                    _flat, canonical = generation.canonicalize_flat(
                        generation.flatten(witness)
                    )
                    self.assertEqual(
                        flash_structural_leaf_from_config(canonical.config), leaf
                    )
                    self.assertEqual(
                        canonical.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY],
                        value,
                    )

                refinement = catalog["refinement_values"][0]
                fixed_generation = spec.create_config_generation(
                    overrides={
                        **overrides,
                        cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY: refinement,
                    }
                )
                _flat, fixed = fixed_generation.canonicalize_flat(
                    fixed_generation.default_flat()
                )
                self.assertEqual(
                    fixed.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY],
                    refinement,
                )
                self.assertEqual(fixed_generation.flash_clc_lane_catalog(), {})

    def test_lfbo_flash_structural_transfers_preserve_pipeline_depths(self):
        with patch("helion.autotuner.config_spec.get_num_xcd", return_value=1):
            spec = ConfigSpec(
                backend=CuteBackend(),
                target_device_capability=(10, 0),
                num_sm=148,
            )
        for block_id, target in enumerate((1, 128, 128)):
            spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=512,
            num_bh=96,
            dtype=torch.float16,
            block_size_targets={0: 1, 1: 128, 2: 128},
            standard_dense_output=True,
        )

        def make_search(generation):
            search = LFBOPatternSearch.__new__(LFBOPatternSearch)
            search.config_gen = generation

            def make_unbenchmarked(flat):
                canonical_flat, config = generation.canonicalize_flat(flat)
                return PopulationMember(
                    fn=lambda: None,
                    perfs=[],
                    flat_values=canonical_flat,
                    config=config,
                    status="ok",
                )

            search.make_unbenchmarked = make_unbenchmarked
            return search

        def make_member(generation, config):
            flat, canonical = generation.canonicalize_flat(generation.flatten(config))
            return PopulationMember(
                fn=lambda: None,
                perfs=[1.0],
                flat_values=flat,
                config=canonical,
                status="ok",
            )

        clc_generation = spec.create_config_generation(
            overrides={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_clc"}
        )
        clc_search = make_search(clc_generation)
        divisor_member = make_member(
            clc_generation,
            helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4_clc",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_s_stage=2,
                cute_flash_clc_heads_per_batch=2,
                cute_flash_epi_tma=True,
                cute_flash_epi_stg=False,
            ),
        )
        for depth in (11, 12):
            with self.subTest(depth=depth):
                depth_member = make_member(
                    clc_generation,
                    helion.Config(
                        block_sizes=[1, 128, 128],
                        cute_flash_pipeline_family="fa4_clc",
                        cute_flash_exp2_packet="1x1",
                        cute_flash_kv_stage=depth,
                        cute_flash_s_stage=2,
                        cute_flash_clc_heads_per_batch=1,
                        cute_flash_epi_tma=False,
                        cute_flash_epi_stg=False,
                    ),
                )
                leaf = flash_structural_leaf_from_config(depth_member.config.config)
                assert leaf is not None

                old_orientation = clc_search._flash_config_variant(
                    divisor_member,
                    {
                        cute_flash.FLASH_KV_STAGE_KEY: depth,
                        cute_flash.FLASH_S_STAGE_KEY: 2,
                    },
                    expected_leaf=leaf,
                )
                self.assertIsNotNone(old_orientation)
                assert old_orientation is not None
                self.assertEqual(
                    old_orientation.config.config[cute_flash.FLASH_KV_STAGE_KEY],
                    10,
                )
                with patch.object(
                    clc_search,
                    "_flash_config_variant",
                    return_value=old_orientation,
                ):
                    self.assertIsNone(
                        clc_search._flash_clc_depth_variant(
                            depth_member,
                            2,
                            expected_leaf=leaf,
                        )
                    )

                combined = clc_search._flash_clc_depth_variant(
                    depth_member,
                    2,
                    expected_leaf=leaf,
                )
                self.assertIsNotNone(combined)
                assert combined is not None
                self.assertEqual(
                    combined.config.config[cute_flash.FLASH_KV_STAGE_KEY], depth
                )
                self.assertEqual(
                    combined.config.config[cute_flash.FLASH_S_STAGE_KEY], 2
                )
                self.assertEqual(
                    combined.config.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY],
                    2,
                )
                self.assertFalse(combined.config.config[cute_flash.FLASH_EPI_TMA_KEY])

        with patch.dict(os.environ, {"HELION_CUTE_FLASH_MMA_PTX": "0"}):
            compound_generation = spec.create_config_generation(
                overrides={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"}
            )
            compound_search = make_search(compound_generation)
            source = make_member(
                compound_generation,
                helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_2cta",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=6,
                    cute_flash_s_stage=2,
                    cute_flash_softmax_disc=True,
                    cute_flash_stat_transport="ring2",
                ),
            )
            raw_compound = copy.deepcopy(source.config)
            raw_compound.config[cute_flash.FLASH_EXP2_PACKET_KEY] = "deg2_16x6"
            _flat, compound_config = compound_generation.canonicalize_flat(
                compound_generation.flatten(raw_compound)
            )
            compound_leaf = flash_structural_leaf_from_config(compound_config.config)
            assert compound_leaf is not None
            self.assertEqual(compound_config.config[cute_flash.FLASH_KV_STAGE_KEY], 2)
            self.assertIsNone(
                compound_search._flash_compound_variant(
                    source,
                    "deg2_16x6",
                    expected_leaf=compound_leaf,
                )
            )

        offset_generation = spec.create_config_generation(
            overrides={cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"}
        )
        offset_search = make_search(offset_generation)
        offset_source = make_member(
            offset_generation,
            helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4_2cta",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_s_stage=2,
                cute_flash_softmax_disc=False,
                cute_flash_e2e_schedule="16/6",
                cute_flash_e2e_offset=12,
            ),
        )
        raw_offset_target = copy.deepcopy(offset_source.config)
        raw_offset_target.config[cute_flash.FLASH_EXP2_PACKET_KEY] = "deg1_8x2_corr10"
        _flat, offset_target_config = offset_generation.canonicalize_flat(
            offset_generation.flatten(raw_offset_target)
        )
        offset_target_leaf = flash_structural_leaf_from_config(
            offset_target_config.config
        )
        assert offset_target_leaf is not None
        offset_target = offset_search._flash_compound_variant(
            offset_source,
            "deg1_8x2_corr10",
            expected_leaf=offset_target_leaf,
        )
        self.assertIsNotNone(offset_target)
        assert offset_target is not None
        self.assertEqual(
            offset_source.config.config[cute_flash.FLASH_E2E_OFFSET_KEY], 12
        )
        self.assertEqual(
            offset_target.config.config[cute_flash.FLASH_E2E_OFFSET_KEY], 4
        )

    def test_flash_initial_prefix_second_witnesses_only_ordinary_leaves(self):
        with patch("helion.autotuner.config_spec.get_num_xcd", return_value=1):
            spec = ConfigSpec(
                backend=CuteBackend(),
                target_device_capability=(10, 0),
                num_sm=148,
            )
        for block_id, target in enumerate((1, 128, 128)):
            spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
        spec.enable_cute_flash_search(
            head_dim=64,
            num_kv=512,
            num_bh=64,
            dtype=torch.float16,
            block_size_targets={0: 1, 1: 128, 2: 128},
            standard_dense_output=True,
        )
        fragments = cute_flash.flash_autotune_fragments(
            64,
            512,
            num_bh=64,
            dtype=torch.float16,
            standard_dense_output=True,
        )
        ws_values = cute_flash.flash_effective_config_values(
            cute_flash.resolve_flash_config(
                64,
                512,
                {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap"},
                dtype=torch.float16,
                num_bh=64,
                standard_dense_output=True,
            )
        )
        live_prefix_keys = {
            cute_flash.FLASH_PIPELINE_FAMILY_KEY,
            cute_flash.FLASH_EXP2_PACKET_KEY,
            cute_flash.FLASH_WAIT_HINT_KEY,
            *(
                key
                for key, fragment in fragments.items()
                if ws_values.get(key) != fragment.default()
            ),
        }
        generation = spec.create_config_generation(
            overrides={
                key: fragment.default()
                for key, fragment in fragments.items()
                if key not in live_prefix_keys
            }
        )

        rows = generation.flash_deterministic_population_configs()
        leaves = generation.flash_structural_leaf_catalog()
        parent_count = generation.flash_structural_parent_coverage_prefix_count()
        qualification_count = generation.flash_structural_qualification_prefix_count()
        ordinary_leaves = [leaf for leaf in leaves if leaf.compound_exp2_packet is None]
        compound_leaves = [
            leaf for leaf in leaves if leaf.compound_exp2_packet is not None
        ]

        def prefix_leaf_counts(limit: int) -> dict[object, int]:
            counts = dict.fromkeys(leaves, 0)
            for config in rows[:limit]:
                leaf = flash_structural_leaf_from_config(config.config)
                if leaf in counts:
                    counts[leaf] += 1
            return counts

        parent_counts = prefix_leaf_counts(parent_count)
        qualification_counts = prefix_leaf_counts(qualification_count)
        self.assertTrue(compound_leaves)
        self.assertTrue(all(parent_counts[leaf] == 1 for leaf in leaves))
        self.assertTrue(
            all(qualification_counts[leaf] == 2 for leaf in ordinary_leaves)
        )
        self.assertTrue(
            all(qualification_counts[leaf] == 1 for leaf in compound_leaves)
        )
        self.assertEqual(
            generation.flash_structural_coverage_underqualified_values(), []
        )
        self.assertEqual(
            qualification_count,
            parent_count + len(ordinary_leaves),
        )
        self.assertEqual(
            generation.flash_structural_population_budget(parent_count),
            parent_count,
        )
        self.assertEqual(
            generation.flash_structural_population_budget(2 * qualification_count),
            qualification_count,
        )

    def test_lfbo_flash_pipeline_lane_witness_uses_generation_catalog(self):
        stage_key = "cute_flash_kv_stage"
        config = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_exp2_packet="1x1",
            cute_flash_kv_stage=3,
        )
        leaf = flash_structural_leaf_from_config(config.config)
        assert leaf is not None
        flat = [3]
        expected = PopulationMember(
            fn=lambda: None,
            perfs=[],
            flat_values=flat,
            config=config,
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_gen = SimpleNamespace(
            flash_pipeline_lane_witnesses=lambda: {(leaf, stage_key, 3): config},
            flatten=Mock(return_value=flat),
            canonicalize_flat=Mock(return_value=(flat, config)),
        )
        search.make_unbenchmarked = Mock(return_value=expected)

        self.assertIs(
            search._flash_pipeline_lane_witness(leaf, (stage_key, 3)),
            expected,
        )
        search.config_gen.flatten.assert_called_once_with(config)
        search.config_gen.canonicalize_flat.assert_called_once_with(flat)
        search.make_unbenchmarked.assert_called_once_with(flat)

    def test_lfbo_flash_starting_points_fall_back_to_finite_rebenchmarks(self):
        def member(wait_hint: int, perf: float) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float("inf"), perf],
                flat_values=[],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="timeout",
            )

        population = [member(0, 1.0), member(1, 2.0)]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.copies = 2
        search.config_spec = _cute_flash_test_config_spec()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._autotune_metrics = SimpleNamespace(search_phase_metrics={})
        search.population = population

        paths = search._select_starting_paths()

        self.assertEqual(paths, [(population[0], ())])
        self.assertEqual(
            search._autotune_metrics.search_phase_metrics["retained_families"], []
        )

    def test_lfbo_flash_qualification_runs_equal_rounds_per_leaf(self):
        def member(perf: float | None, family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[family, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(5.0, "fa4", 0)
        first_child = member(None, "fa4", 1)
        winning_child = member(None, "fa4", 2)
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        assert search.flash_structural_search is not None
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: {leaf: ()},
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.set_generation = Mock()
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)

        def rebenchmark(members, *, desc):
            if winning_child in members and winning_child.perfs:
                winning_child.perfs.append(0.5)

        search.rebenchmark_population = Mock(side_effect=rebenchmark)
        search._fit_surrogate = Mock()

        def qualification_path(
            _index,
            current,
            _visited,
            constraints,
            *,
            selected_limit=None,
            neighbor_limit=None,
            required_leaf=None,
            conditional_surface=False,
            disable_early_stopping=False,
        ):
            self.assertEqual(selected_limit, 5)
            self.assertEqual(neighbor_limit, 300)
            self.assertEqual(required_leaf, leaf)
            self.assertTrue(conditional_surface)
            self.assertTrue(disable_early_stopping)
            child = first_child if current is initial else winning_child
            return iter(((current, child),))

        def benchmark(members, *, desc):
            if members == [first_child]:
                self.assertEqual(desc, "Structural qualification 1:")
                first_child.perfs = [4.0]
            else:
                self.assertEqual(members, [winning_child])
                self.assertEqual(desc, "Structural qualification 2:")
                winning_child.perfs = [0.75]

        search._pruned_pattern_search_from = qualification_path
        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 2
        )
        self.assertEqual(search.population[0], winning_child)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["phase"], "cute_flash_structural_qualification_v22")
        self.assertEqual(metrics["cute_flash_lane_policy_version"], 14)
        self.assertTrue(metrics["completed"])
        self.assertEqual(metrics["qualification_rounds"], 2)
        self.assertEqual(metrics["qualification_rounds_started"], 2)
        self.assertEqual(metrics["qualification_rounds_completed"], 2)
        self.assertEqual(metrics["retained_family_cap"], 4)
        self.assertEqual(metrics["retained_family_limit"], 1)
        self.assertEqual(metrics["retained_family_slowdown_limit"], 2.0)
        self.assertFalse(metrics["budget_exhausted"])
        self.assertFalse(metrics["family_probe_required"])
        self.assertTrue(metrics["family_probe_complete"])
        self.assertEqual(metrics["family_probe_paths"], [])
        self.assertEqual(metrics["starting_path_limit"], 5)
        self.assertTrue(metrics["unrestricted_path_exhausts_generation_budget"])
        self.assertEqual(metrics["candidate_count"], 2)
        self.assertEqual(len(metrics["leaf_results"][0]["rounds"]), 2)
        self.assertEqual(
            [
                round_metric["neighbor_generation_limit"]
                for round_metric in metrics["leaf_results"][0]["rounds"]
            ],
            [300, 300],
        )
        self.assertEqual(
            [
                round_metric["ordinary_neighbor_generation_limit"]
                for round_metric in metrics["leaf_results"][0]["rounds"]
            ],
            [300, 300],
        )
        self.assertEqual(
            metrics["leaf_results"][0]["retained_config_ids"][0],
            canonical_config_id(winning_child.config),
        )
        qualified = {
            result["config_id"]: result
            for result in metrics["leaf_results"][0]["qualified_results"]
        }
        self.assertEqual(
            qualified[canonical_config_id(winning_child.config)]["attempt_perf"],
            0.75,
        )
        self.assertEqual(
            qualified[canonical_config_id(winning_child.config)]["selection_perf"],
            0.5,
        )
        self.assertEqual(search.set_generation.call_args_list, [call(1), call(2)])
        self.assertEqual(search._fit_surrogate.call_count, 2)

    def test_lfbo_flash_qualification_probes_every_family_before_promotion(self):
        families = ("fa4", "fa4_2cta", "ws_overlap", "fa4_clc", "fa4_tma_4d")

        def member(perf: float | None, family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[family, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial_perfs = dict(zip(families, (1.0, 1.1, 1.2, 4.0, 5.0), strict=True))
        initial = {
            family: member(initial_perfs[family], family, index)
            for index, family in enumerate(families)
        }
        qualification_children = {
            family: (
                member(None, family, 10 + 2 * index),
                member(None, family, 11 + 2 * index),
            )
            for index, family in enumerate(families)
        }
        constrained_probe_children = {
            family: member(None, family, 30 + index)
            for index, family in enumerate(families)
        }
        global_probe_child = member(None, "fa4_clc", 40)
        leaves = {
            family: flash_structural_leaf_from_config(item.config.config)
            for family, item in initial.items()
        }
        assert all(leaf is not None for leaf in leaves.values())

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.copies = 14
        search._flash_promoted_path_limit = 14
        search._flash_family_probe_path_limit = 6
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        assert search.flash_structural_search is not None
        search.population = list(initial.values())
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        typed_leaves = []
        for family in families:
            leaf = leaves[family]
            assert leaf is not None
            typed_leaves.append(leaf)
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: typed_leaves,
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: dict.fromkeys(typed_leaves, ()),
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.set_generation = Mock()
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def search_path(
            _index,
            current,
            _visited,
            _constraints,
            *,
            selected_limit=None,
            neighbor_limit=None,
            required_leaf=None,
            conditional_surface=False,
            disable_early_stopping=False,
            selected_only_visited=False,
        ):
            self.assertTrue(disable_early_stopping)
            if selected_limit == 5:
                self.assertIsNotNone(required_leaf)
                self.assertTrue(conditional_surface)
                self.assertIsNotNone(neighbor_limit)
                family = required_leaf.pipeline_family
                first, second = qualification_children[family]
                return iter(((current, first), (first, second)))
            self.assertEqual(selected_limit, 20)
            self.assertIsNone(neighbor_limit)
            if selected_only_visited:
                self.assertIsNone(required_leaf)
                self.assertFalse(conditional_surface)
                return iter(((current, global_probe_child),))
            assert required_leaf is not None
            self.assertTrue(conditional_surface)
            return iter(
                ((current, constrained_probe_children[required_leaf.pipeline_family]),)
            )

        def benchmark(members, *, desc):
            if desc == "Structural family probe 1:":
                for item in members:
                    item.perfs = [
                        0.1
                        if item is global_probe_child
                        else 0.8
                        if item is constrained_probe_children["fa4_tma_4d"]
                        else initial_perfs[
                            item.config.config["cute_flash_pipeline_family"]
                        ]
                    ]
                return
            for item in members:
                family = item.config.config["cute_flash_pipeline_family"]
                assert isinstance(family, str)
                item.perfs = [initial_perfs[family]]

        search._pruned_pattern_search_from = search_path
        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in initial.values()}
            ),
            3,
        )
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertTrue(metrics["family_probe_required"])
        self.assertTrue(metrics["family_probe_complete"])
        self.assertEqual(metrics["family_probe_path_limit"], 6)
        self.assertEqual(len(metrics["family_probe_paths"]), 6)
        self.assertEqual(metrics["qualification_passes_completed"], 3)

        search._select_starting_paths()
        retained = {
            family["family"]: family
            for family in search._autotune_metrics.search_phase_metrics[
                "retained_families"
            ]
        }
        self.assertTrue(retained["fa4_tma_4d"]["parent_promoted"])
        self.assertFalse(retained["fa4_clc"]["parent_promoted"])
        unrestricted = [
            path
            for family in retained.values()
            for path in family["starting_paths"]
            if path["unrestricted"]
        ]
        self.assertEqual(len(unrestricted), 1)
        self.assertEqual(unrestricted[0]["family"], "fa4_clc")

    def test_lfbo_flash_qualification_finishes_round_before_budget_stops_next(self):
        def member(perf: float | None, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(2.0, 0)
        child = member(None, 1)
        declined_child = member(None, 2)
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: {leaf: ()},
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search._autotune_budget_exceeded_across_ranks = Mock(side_effect=(False, True))
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        generated: list[PopulationMember] = []

        def qualification_path(_index, current, visited, *_args, **_kwargs):
            candidate = child if not generated else declined_child
            generated.append(candidate)
            visited.add(candidate.config)
            return iter(((current, candidate),))

        search._pruned_pattern_search_from = qualification_path

        def benchmark(members, *, desc):
            self.assertEqual(members, [child])
            self.assertEqual(desc, "Structural qualification 1:")
            child.perfs = [1.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 1
        )
        search.rebenchmark_population.assert_called_once()
        rebenchmark_call = search.rebenchmark_population.call_args
        self.assertCountEqual(rebenchmark_call.args[0], [initial, child])
        self.assertEqual(
            rebenchmark_call.kwargs["desc"],
            "Structural qualification 1: verifying",
        )
        search._fit_surrogate.assert_called_once_with()
        self.assertEqual(
            search._autotune_budget_exceeded_across_ranks.call_args_list,
            [call(), call()],
        )
        search.set_generation.assert_called_once_with(1)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertFalse(metrics["completed"])
        self.assertEqual(metrics["qualification_rounds_started"], 1)
        self.assertEqual(metrics["qualification_rounds_completed"], 1)
        self.assertTrue(metrics["budget_exhausted"])
        self.assertEqual(generated, [child])
        self.assertNotIn(declined_child, search.population)
        self.assertFalse(declined_child.perfs)

    def test_lfbo_flash_qualification_reserves_low_generation_budget(self):
        def member(perf: float | None, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(2.0, 0)
        first_child = member(None, 1)
        declined_child = member(None, 2)
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_pipeline_lane_catalog=lambda: {leaf: ()},
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 2
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        visited = {initial.config}
        generated: list[PopulationMember] = []

        def qualification_path(_index, current, path_visited, *_args, **_kwargs):
            candidate = first_child if not generated else declined_child
            generated.append(candidate)
            path_visited.add(candidate.config)
            return iter(((current, candidate),))

        search._pruned_pattern_search_from = qualification_path

        def benchmark(members, *, desc):
            self.assertEqual(members, [first_child])
            self.assertEqual(desc, "Structural qualification 1:")
            first_child.perfs = [1.0]

        search.benchmark_population = benchmark

        self.assertEqual(search._run_flash_structural_qualification(visited), 1)
        self.assertEqual(generated, [first_child])
        self.assertNotIn(declined_child.config, visited)
        self.assertNotIn(declined_child, search.population)
        self.assertFalse(declined_child.perfs)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["qualification_passes_started"], 1)
        self.assertEqual(metrics["qualification_passes_completed"], 1)
        self.assertTrue(metrics["budget_exhausted"])

    def test_lfbo_flash_qualification_synthesizes_missing_pipeline_lane(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            perf: float | None, stage: int, wait_hint: int = 0
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[stage],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(2.0, 2)
        synthesized = member(None, 3)
        conditional_children = {stage: member(None, stage, 1) for stage in (2, 3)}
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )
        witness = Mock(return_value=synthesized)
        search._flash_pipeline_lane_witness = witness
        search._pruned_pattern_search_from = lambda _index, current, *_args, **_kwargs: (
            iter(((current, conditional_children[current.config.config[stage_key]]),))
        )
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()

        def rebenchmark(_members, *, desc):
            if desc == "Structural qualification 2: verifying":
                initial.perfs.append(2.2)
                synthesized.perfs.append(1.2)

        search.rebenchmark_population = Mock(side_effect=rebenchmark)
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if members == [synthesized]:
                self.assertEqual(desc, "Structural qualification 1:")
                synthesized.perfs = [1.0]
                return
            self.assertCountEqual(members, conditional_children.values())
            self.assertEqual(desc, "Structural qualification 2:")
            for child in members:
                child.perfs = [3.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 2
        )

        witness.assert_called_once()
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["candidate_count"], 3)
        lanes = metrics["leaf_results"][0]["pipeline_lanes"]
        self.assertTrue(all(lane["complete"] for lane in lanes))
        self.assertEqual(
            lanes[1]["rounds"][0]["candidate_config_ids"],
            [canonical_config_id(synthesized.config)],
        )
        self.assertEqual(
            metrics["leaf_results"][0]["retained_config_ids"],
            [
                canonical_config_id(synthesized.config),
                canonical_config_id(initial.config),
            ],
        )
        conditional_decisions = metrics["leaf_results"][0]["rounds"][1][
            "parent_decisions"
        ]
        decision_by_lane = {
            decision["pipeline_lane"]["value"]: decision
            for decision in conditional_decisions
        }
        self.assertEqual(
            decision_by_lane[2]["candidate_results"],
            [
                {
                    "config_id": canonical_config_id(initial.config),
                    "attempt_perf": 2.0,
                    "selection_perf": 2.0,
                    "status": "ok",
                    "source_hash": None,
                    "measurement_pass_index": 1,
                }
            ],
        )
        self.assertEqual(
            decision_by_lane[3]["candidate_results"],
            [
                {
                    "config_id": canonical_config_id(synthesized.config),
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                    "source_hash": None,
                    "measurement_pass_index": 1,
                }
            ],
        )
        qualified = {
            result["config_id"]: result
            for result in metrics["leaf_results"][0]["qualified_results"]
        }
        self.assertEqual(
            qualified[canonical_config_id(initial.config)]["selection_perf"],
            2.2,
        )
        self.assertEqual(
            qualified[canonical_config_id(synthesized.config)]["selection_perf"],
            1.2,
        )

    def test_lfbo_flash_failed_existing_lane_uses_catalog_witness(self):
        stage_key = "cute_flash_kv_stage"
        failed = PopulationMember(
            fn=lambda: None,
            perfs=[float("inf")],
            flat_values=[2, 0],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_wait_hint=0,
            ),
            status="error",
        )
        catalog_witness = PopulationMember(
            fn=lambda: None,
            perfs=[],
            flat_values=[2, 1],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_wait_hint=1,
            ),
            status="ok",
        )
        child = PopulationMember(
            fn=lambda: None,
            perfs=[],
            flat_values=[2, 2],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_kv_stage=2,
                cute_flash_wait_hint=2,
            ),
            status="ok",
        )
        leaf = flash_structural_leaf_from_config(failed.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [failed]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_pipeline_lane_witness = Mock(return_value=catalog_witness)

        def qualification_path(_index, current, *_args, **_kwargs):
            self.assertIs(current, catalog_witness)
            return iter(((current, child),))

        search._pruned_pattern_search_from = qualification_path
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if desc == "Structural qualification 1:":
                self.assertEqual(members, [catalog_witness])
                catalog_witness.perfs = [1.0]
                return
            self.assertEqual(desc, "Structural qualification 2:")
            self.assertEqual(members, [child])
            child.perfs = [2.0]

        search.benchmark_population = benchmark

        self.assertEqual(search._run_flash_structural_qualification({failed.config}), 2)
        search._flash_pipeline_lane_witness.assert_called_once_with(
            leaf, (stage_key, 2)
        )
        lane = search._autotune_metrics.search_phase_metrics["leaf_results"][0][
            "pipeline_lanes"
        ][0]
        self.assertTrue(lane["witness_succeeded"])
        self.assertFalse(lane["terminal_failure_exhausted"])
        self.assertEqual(
            lane["witness_config_id"], canonical_config_id(catalog_witness.config)
        )
        self.assertEqual(
            lane["conditional_candidate_ids"],
            [canonical_config_id(child.config)],
        )
        self.assertEqual(
            lane["successful_conditional_candidate_ids"],
            [canonical_config_id(child.config)],
        )
        rounds = search._autotune_metrics.search_phase_metrics["leaf_results"][0][
            "rounds"
        ]
        witness_decision = rounds[0]["parent_decisions"][0]
        self.assertEqual(witness_decision["selection_kind"], "catalog_witness")
        self.assertEqual(
            witness_decision["candidate_results"],
            [
                {
                    "config_id": canonical_config_id(catalog_witness.config),
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "unknown",
                    "source_hash": None,
                    "measurement_pass_index": None,
                }
            ],
        )
        conditional_decision = rounds[1]["parent_decisions"][0]
        self.assertEqual(
            conditional_decision["candidate_results"],
            [
                {
                    "config_id": canonical_config_id(catalog_witness.config),
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                    "source_hash": None,
                    "measurement_pass_index": 1,
                },
                {
                    "config_id": canonical_config_id(failed.config),
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "error",
                    "source_hash": None,
                    "measurement_pass_index": 1,
                },
            ],
        )
        self.assertEqual(
            conditional_decision["generated_config_ids"],
            [canonical_config_id(child.config)],
        )
        self.assertTrue(lane["complete"])
        self.assertTrue(search._autotune_metrics.search_phase_metrics["completed"])

    def test_lfbo_flash_failed_lane_gets_one_bounded_repair(self):
        stage_key = "cute_flash_kv_stage"

        def member(wait_hint: int, *, failed: bool = False) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float("inf")] if failed else [],
                flat_values=[2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="error" if failed else "ok",
            )

        failed_witness = member(0, failed=True)
        failed_conditional = member(1)
        successful_repair = member(2)
        leaf = flash_structural_leaf_from_config(failed_witness.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [failed_witness]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_pipeline_lane_witness = Mock(return_value=failed_witness)

        generated = iter((failed_conditional, successful_repair))
        search_parents: list[PopulationMember] = []

        def qualification_path(_index, current, *_args, **_kwargs):
            search_parents.append(current)
            child = next(generated)
            return iter(((current, child),))

        search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        benchmark_descs: list[str] = []

        def benchmark(members, *, desc):
            benchmark_descs.append(desc)
            if desc == "Structural qualification 2:":
                self.assertEqual(members, [failed_conditional])
                failed_conditional.perfs = [float("inf")]
                failed_conditional.status = "error"
                return
            self.assertEqual(desc, "Structural qualification failure repairs 1:")
            self.assertEqual(members, [successful_repair])
            successful_repair.perfs = [1.0]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({failed_witness.config}), 3
        )
        search._flash_pipeline_lane_witness.assert_called_once_with(
            leaf, (stage_key, 2)
        )
        self.assertEqual(search._pruned_pattern_search_from.call_count, 2)
        repair_parent = min(
            (failed_witness, failed_conditional),
            key=search._flash_member_rank_key,
        )
        self.assertEqual(search_parents, [failed_witness, repair_parent])
        self.assertEqual(
            benchmark_descs,
            [
                "Structural qualification 2:",
                "Structural qualification failure repairs 1:",
            ],
        )
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["qualification_failure_retries"], 1)
        lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
        self.assertFalse(lane["witness_succeeded"])
        self.assertEqual(
            lane["conditional_candidate_ids"],
            [canonical_config_id(failed_conditional.config)],
        )
        self.assertEqual(lane["successful_conditional_candidate_ids"], [])
        self.assertEqual(
            lane["repair_candidate_ids"],
            [canonical_config_id(successful_repair.config)],
        )
        self.assertEqual(
            lane["successful_repair_candidate_ids"],
            [canonical_config_id(successful_repair.config)],
        )
        self.assertEqual(len(lane["repair_parent_decisions"]), 1)
        self.assertEqual(
            lane["repair_parent_decisions"][0]["selected_config_id"],
            canonical_config_id(repair_parent.config),
        )
        self.assertTrue(lane["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_terminal_failures_exhaust_lane_not_leaf(self):
        stage_key = "cute_flash_kv_stage"

        def run(*, leaf_has_success: bool):
            def member(
                stage: int,
                wait_hint: int,
                perf: float | None,
                *,
                failed: bool = False,
            ) -> PopulationMember:
                return PopulationMember(
                    fn=lambda: None,
                    perfs=[] if perf is None else [perf],
                    flat_values=[stage, wait_hint],
                    config=helion.Config(
                        block_sizes=[1, 128, 128],
                        cute_flash_pipeline_family="fa4",
                        cute_flash_exp2_packet="1x1",
                        cute_flash_kv_stage=stage,
                        cute_flash_wait_hint=wait_hint,
                    ),
                    status="error" if failed else "ok",
                )

            failed_witness = member(2, 0, float("inf"), failed=True)
            catalog_shadow = member(2, 0, None)
            failed_conditional = member(2, 1, None)
            failed_repair = member(2, 2, None)
            successful_other = member(3, 0, 1.0)
            leaf = flash_structural_leaf_from_config(failed_witness.config.config)
            assert leaf is not None

            search = LFBOPatternSearch.__new__(LFBOPatternSearch)
            search.config_spec = _cute_flash_test_config_spec()
            search.config_gen = SimpleNamespace(
                encode_config=lambda flat: flat,
                flash_structural_leaf_catalog=lambda: [leaf],
                flash_low_confound_schedule_anchor_configs=list,
                flash_clc_lane_catalog=dict,
                flash_exact_effective_search_space_configs=lambda _limit: None,
            )
            search.copies = 5
            search.num_neighbors = 300
            search.num_neighbors_cap = -1
            search.max_generations = 20
            search.flash_structural_search = get_effort_profile(
                "full"
            ).flash_structural_search
            search.population = [failed_witness]
            if leaf_has_success:
                search.population.append(successful_other)
            search.initial_population = len(search.population)
            search.train_x = []
            search.train_y = []
            search.train_configs = []
            search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
            search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
            search._flash_pipeline_lane_witness = Mock(return_value=catalog_shadow)
            generated = iter((failed_conditional, failed_repair))

            def qualification_path(_index, current, *_args, **_kwargs):
                self.assertIn(current, (failed_witness, failed_conditional))
                child = next(generated)
                return iter(((current, child),))

            search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
            search._budgeted_range = lambda *args: range(*args)
            search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
            search.set_generation = Mock()
            search.rebenchmark_population = Mock()
            search._fit_surrogate = Mock()

            def benchmark(members, *, desc):
                only = members[0]
                self.assertIn(only, (failed_conditional, failed_repair))
                self.assertIn(
                    desc,
                    (
                        "Structural qualification 2:",
                        "Structural qualification failure repairs 1:",
                    ),
                )
                only.perfs = [float("inf")]
                only.status = "error"

            search.benchmark_population = benchmark

            self.assertEqual(
                search._run_flash_structural_qualification(
                    {member.config for member in search.population}
                ),
                3,
            )
            search._flash_pipeline_lane_witness.assert_called_once_with(
                leaf, (stage_key, 2)
            )
            metrics = search._autotune_metrics.search_phase_metrics
            assert isinstance(metrics, dict)
            lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
            self.assertEqual(
                lane["witness_config_id"], canonical_config_id(failed_witness.config)
            )
            witness_decision = metrics["leaf_results"][0]["rounds"][0][
                "parent_decisions"
            ][0]
            self.assertEqual(witness_decision["selection_kind"], "catalog_witness")
            self.assertEqual(
                witness_decision["candidate_results"][0]["status"], "error"
            )
            self.assertTrue(lane["terminal_failure_exhausted"])
            self.assertTrue(lane["complete"])
            return metrics

        self.assertTrue(run(leaf_has_success=True)["completed"])
        self.assertFalse(run(leaf_has_success=False)["completed"])

    def test_lfbo_flash_later_repair_batch_rechecks_source_repairs(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            stage: int,
            wait_hint: int,
            *,
            failed: bool = False,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float("inf")] if failed else [],
                flat_values=[stage, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="error" if failed else "ok",
            )

        failed_witnesses = {stage: member(stage, 0, failed=True) for stage in (2, 3)}
        failed_conditionals = {stage: member(stage, 1) for stage in (2, 3)}
        successful_repair = member(2, 2)
        source_hashes: dict[int, str] = {}
        shared_source_hash = "a" * 64
        source_hashes[id(failed_conditionals[3].fn)] = shared_source_hash
        source_hashes[id(successful_repair.fn)] = shared_source_hash
        leaf = flash_structural_leaf_from_config(failed_witnesses[2].config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: source_hashes.get(id(fn))
            ),
        )
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            pipeline_candidates_per_leaf_per_round=1,
        )
        search.population = list(failed_witnesses.values())
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )

        def catalog_witness(_leaf, lane):
            stage = lane[1]
            assert isinstance(stage, int)
            return failed_witnesses[stage]

        search._flash_pipeline_lane_witness = Mock(side_effect=catalog_witness)
        generated = {
            (2, 0): failed_conditionals[2],
            (3, 0): failed_conditionals[3],
            (2, 1): successful_repair,
        }
        calls_by_stage = {2: 0, 3: 0}

        def qualification_path(_index, current, *_args, **_kwargs):
            stage = current.config.config[stage_key]
            call_index = calls_by_stage[stage]
            calls_by_stage[stage] += 1
            return iter(((current, generated[(stage, call_index)]),))

        search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        benchmark_descs: list[str] = []

        def benchmark(members, *, desc):
            benchmark_descs.append(desc)
            only = members[0]
            if only in failed_conditionals.values():
                only.perfs = [float("inf")]
                only.status = "error"
                return
            self.assertIs(only, successful_repair)
            successful_repair.perfs = [1.0]
            failed_conditionals[3].perfs = [1.5]
            failed_conditionals[3].status = "deduplicated"

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {member.config for member in failed_witnesses.values()}
            ),
            5,
        )
        self.assertEqual(
            benchmark_descs,
            [
                "Structural qualification 3:",
                "Structural qualification 4:",
                "Structural qualification failure repairs 1:",
            ],
        )
        self.assertEqual(search._pruned_pattern_search_from.call_count, 3)
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["qualification_passes_planned"], 5)
        self.assertEqual(metrics["qualification_passes_completed"], 5)
        self.assertEqual(len(metrics["measurement_timeline"]), 6)
        lanes = {
            lane["value"]: lane for lane in metrics["leaf_results"][0]["pipeline_lanes"]
        }
        self.assertEqual(len(lanes[2]["repair_parent_decisions"]), 1)
        self.assertEqual(lanes[3]["repair_parent_decisions"], [])
        self.assertEqual(lanes[3]["successful_repair_candidate_ids"], [])
        self.assertIn(
            canonical_config_id(failed_conditionals[3].config),
            lanes[3]["successful_conditional_candidate_ids"],
        )
        final_updates = {
            update["config_id"]: update
            for update in metrics["measurement_timeline"][-1]["updates"]
        }
        self.assertEqual(
            final_updates[canonical_config_id(failed_conditionals[3].config)][
                "source_hash"
            ],
            shared_source_hash,
        )
        self.assertEqual(
            final_updates[canonical_config_id(successful_repair.config)]["source_hash"],
            shared_source_hash,
        )
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_witness_and_conditional_failures_are_repaired(self):
        stage_key = "cute_flash_kv_stage"
        clc_key = "cute_flash_clc_heads_per_batch"

        def member(wait_hint: int, perf: float | None) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[2, 2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_clc_heads_per_batch=2,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        base = member(0, 10.0)
        pipeline_child = member(1, None)
        failed_witness = member(2, None)
        witness_repair = member(3, None)
        failed_conditional = member(4, None)
        conditional_repair = member(5, None)
        leaf = flash_structural_leaf_from_config(base.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (2,),
                    "search_values": (2,),
                    "anchor_values": (2,),
                    "refinement_values": (),
                    "attempted_values": (2,),
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            retained_candidates_per_leaf=1,
        )
        search.population = [base]
        search.initial_population = 1
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_pipeline_lane_witness = Mock(
            side_effect=AssertionError("the measured pipeline witness must be reused")
        )
        search._flash_clc_lane_witness = lambda _leaf, _value: failed_witness
        generated = {
            id(base): pipeline_child,
            id(failed_witness): witness_repair,
            id(witness_repair): failed_conditional,
            id(failed_conditional): conditional_repair,
        }
        constraints_by_parent: dict[int, dict[str, object]] = {}

        def qualification_path(_index, current, _visited, constraints, **_kwargs):
            constraints_by_parent[id(current)] = dict(constraints)
            return iter(((current, generated[id(current)]),))

        search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
        search._flash_clc_depth_variant = Mock(return_value=conditional_repair)
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        benchmark_descs: list[str] = []

        def benchmark(members, *, desc):
            benchmark_descs.append(desc)
            only = members[0]
            if only is failed_witness or only is failed_conditional:
                only.perfs = [float("inf")]
                only.status = "error"
            else:
                only.perfs = [
                    {
                        id(pipeline_child): 1.0,
                        id(witness_repair): 2.0,
                        id(conditional_repair): 0.5,
                    }[id(only)]
                ]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({base.config}),
            7,
        )
        self.assertEqual(
            benchmark_descs,
            [
                "Structural qualification 2:",
                "CLC divisor witnesses 1:",
                "CLC divisor witness failure repairs 1:",
                "CLC divisor conditional children:",
                "CLC divisor conditional failure repairs 1:",
            ],
        )
        for repaired_parent in (failed_witness, failed_conditional):
            self.assertEqual(constraints_by_parent[id(repaired_parent)][clc_key], 2)
            self.assertEqual(
                constraints_by_parent[id(repaired_parent)][
                    "cute_flash_pipeline_family"
                ],
                "fa4_clc",
            )
        metrics = search._autotune_metrics.search_phase_metrics
        clc = metrics["clc_families"][0]
        self.assertEqual(
            clc["witness_repair_candidate_ids"],
            {"2": [canonical_config_id(witness_repair.config)]},
        )
        self.assertEqual(
            clc["conditional_repair_candidate_ids"],
            {"2": [canonical_config_id(conditional_repair.config)]},
        )
        self.assertEqual(
            clc["selected_config_ids"],
            [canonical_config_id(witness_repair.config)],
        )
        self.assertEqual(
            clc["retained_config_ids"],
            [canonical_config_id(conditional_repair.config)],
        )
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_empty_conditional_generation_is_retried(self):
        clc_key = "cute_flash_clc_heads_per_batch"

        def run(*, retry_generates_candidate: bool, repair_witness: bool = False):
            def member(wait_hint: int, perf: float | None) -> PopulationMember:
                return PopulationMember(
                    fn=lambda: None,
                    perfs=[] if perf is None else [perf],
                    flat_values=[2, wait_hint],
                    config=helion.Config(
                        block_sizes=[1, 128, 128],
                        cute_flash_pipeline_family="fa4_clc",
                        cute_flash_exp2_packet="1x1",
                        cute_flash_kv_stage=2,
                        cute_flash_clc_heads_per_batch=2,
                        cute_flash_wait_hint=wait_hint,
                    ),
                    status="ok",
                )

            base = member(-1, 1.0) if repair_witness else member(0, 1.0)
            witness = member(0, None) if repair_witness else base
            witness_repair = member(1, None) if repair_witness else witness
            retry = member(2, None)
            leaf = flash_structural_leaf_from_config(witness.config.config)
            assert leaf is not None

            search = LFBOPatternSearch.__new__(LFBOPatternSearch)
            search.config_spec = _cute_flash_test_config_spec()
            search.config_gen = SimpleNamespace(
                encode_config=lambda flat: flat,
                flash_structural_leaf_catalog=lambda: [leaf],
                flash_low_confound_schedule_anchor_configs=list,
                flash_clc_lane_catalog=lambda: {
                    leaf: {
                        "legal_values": (2,),
                        "search_values": (2,),
                        "anchor_values": (2,),
                        "refinement_values": (),
                        "attempted_values": (2,),
                    }
                },
                flash_exact_effective_search_space_configs=lambda _limit: None,
            )
            search.copies = 5
            search.num_neighbors = 300
            search.num_neighbors_cap = -1
            search.max_generations = 20
            policy = get_effort_profile("full").flash_structural_search
            assert policy is not None
            search.flash_structural_search = replace(
                policy,
                qualification_rounds=0,
                retained_candidates_per_leaf=1,
            )
            search.population = [base]
            search.initial_population = 1
            search.train_x = []
            search.train_y = []
            search.train_configs = []
            search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
            search._flash_pipeline_lanes = lambda _leaf: ()
            search._flash_clc_lane_witness = lambda _leaf, _value: witness
            search._budgeted_range = lambda *args: range(*args)
            search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
            search.set_generation = Mock()
            search.rebenchmark_population = Mock()
            search._fit_surrogate = Mock()
            search._flash_clc_depth_variant = Mock(
                return_value=(retry if retry_generates_candidate else witness_repair)
            )

            conditional_generation_calls = 0

            def qualification_path(_index, current, _visited, constraints, **_kwargs):
                self.assertEqual(dict(constraints)[clc_key], 2)
                if repair_witness and current is witness:
                    return iter(((witness, witness_repair),))
                nonlocal conditional_generation_calls
                conditional_generation_calls += 1
                self.assertIs(current, witness_repair)
                if conditional_generation_calls == 1 or not retry_generates_candidate:
                    return iter(((witness_repair,),))
                return iter(((witness_repair, retry),))

            search._pruned_pattern_search_from = Mock(side_effect=qualification_path)

            def benchmark(members, *, desc):
                only = members[0]
                if only is witness:
                    self.assertTrue(repair_witness)
                    self.assertEqual(desc, "CLC divisor witnesses 1:")
                    witness.perfs = [float("inf")]
                    witness.status = "error"
                elif only is witness_repair:
                    self.assertTrue(repair_witness)
                    self.assertEqual(desc, "CLC divisor witness failure repairs 1:")
                    witness_repair.perfs = [0.75]
                else:
                    self.assertTrue(retry_generates_candidate)
                    self.assertIs(only, retry)
                    self.assertEqual(desc, "CLC divisor conditional failure repairs 1:")
                    retry.perfs = [0.5]

            search.benchmark_population = Mock(side_effect=benchmark)

            self.assertEqual(
                search._run_flash_structural_qualification({base.config}),
                5 if repair_witness else 4,
            )
            self.assertEqual(conditional_generation_calls, 2)
            metrics = search._autotune_metrics.search_phase_metrics
            clc = metrics["clc_families"][0]
            self.assertEqual(clc["conditional_candidate_ids"], {"2": []})
            self.assertEqual(len(clc["conditional_repair_parent_decisions"]), 1)
            decision = clc["conditional_repair_parent_decisions"][0]
            self.assertEqual(
                decision["selected_config_id"],
                canonical_config_id(witness_repair.config),
            )
            self.assertEqual(
                [result["config_id"] for result in decision["candidate_results"]],
                [canonical_config_id(witness_repair.config)],
            )
            return search, metrics, clc, retry, witness_repair

        search, metrics, clc, retry, _witness = run(retry_generates_candidate=True)
        retry_id = canonical_config_id(retry.config)
        self.assertEqual(clc["conditional_repair_candidate_ids"], {"2": [retry_id]})
        self.assertEqual(
            clc["conditional_repair_parent_decisions"][0]["generated_config_ids"],
            [retry_id],
        )
        search.benchmark_population.assert_called_once()
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

        search, metrics, clc, _retry, _witness = run(retry_generates_candidate=False)
        self.assertEqual(clc["conditional_repair_candidate_ids"], {"2": []})
        self.assertEqual(
            clc["conditional_repair_parent_decisions"][0]["generated_config_ids"],
            [],
        )
        search.benchmark_population.assert_not_called()
        self.assertFalse(clc["complete"])
        self.assertFalse(metrics["completed"])

        search, metrics, clc, retry, witness_repair = run(
            retry_generates_candidate=True,
            repair_witness=True,
        )
        self.assertEqual(
            clc["selected_config_ids"],
            [canonical_config_id(witness_repair.config)],
        )
        self.assertEqual(
            clc["conditional_repair_candidate_ids"],
            {"2": [canonical_config_id(retry.config)]},
        )
        self.assertEqual(search.benchmark_population.call_count, 3)
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_repair_batches_compact_after_source_repair(self):
        clc_key = "cute_flash_clc_heads_per_batch"
        values = tuple(range(1, 10))

        def member(value: int, wait_hint: int, *, failed: bool) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float("inf")] if failed else [],
                flat_values=[value, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_clc_heads_per_batch=value,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="error" if failed else "ok",
            )

        failed_witnesses = {value: member(value, 0, failed=True) for value in values}
        repairs = {value: member(value, 1, failed=False) for value in values}
        leaf = flash_structural_leaf_from_config(
            failed_witnesses[values[0]].config.config
        )
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": values,
                    "search_values": values,
                    "anchor_values": values,
                    "refinement_values": (),
                    "attempted_values": values,
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: [
                witness.config for witness in failed_witnesses.values()
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            pipeline_candidates_per_leaf_per_round=4,
        )
        search.population = list(failed_witnesses.values())
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ()
        search._flash_clc_lane_witness = lambda _leaf, value: failed_witnesses[value]

        def qualification_path(_index, current, *_args, **_kwargs):
            value = current.config.config[clc_key]
            return iter(((current, repairs[value]),))

        search._pruned_pattern_search_from = Mock(side_effect=qualification_path)
        search._flash_clc_depth_variant = Mock(
            side_effect=AssertionError("an exhausted CLC leaf needs no combinations")
        )
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        repaired_batches: list[list[int]] = []

        def benchmark(members, *, desc):
            self.assertTrue(desc.startswith("CLC divisor witness failure repairs "))
            repaired_values = [member.config.config[clc_key] for member in members]
            repaired_batches.append(repaired_values)
            for repaired in members:
                value = repaired.config.config[clc_key]
                repaired.perfs = [1.0 + value / 100]
            if len(repaired_batches) == 1:
                failed_witnesses[5].perfs = [1.05]
                failed_witnesses[5].status = "deduplicated"

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {witness.config for witness in failed_witnesses.values()}
            ),
            3,
        )
        self.assertEqual(repaired_batches, [[1, 2, 3, 4], [6, 7, 8, 9]])
        self.assertEqual(search._pruned_pattern_search_from.call_count, 8)
        search._flash_clc_depth_variant.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertEqual(metrics["qualification_passes_planned"], 3)
        self.assertEqual(metrics["qualification_passes_completed"], 3)
        self.assertEqual(len(metrics["measurement_timeline"]), 4)
        clc = metrics["clc_families"][0]
        self.assertEqual(
            [decision["value"] for decision in clc["witness_repair_parent_decisions"]],
            [1, 2, 3, 4, 6, 7, 8, 9],
        )
        self.assertEqual(
            [
                decision["candidate_results"][0]["measurement_pass_index"]
                for decision in clc["witness_repair_parent_decisions"]
            ],
            [1, 1, 1, 1, 2, 2, 2, 2],
        )
        self.assertEqual(
            [
                decision["neighbor_generation_limit"]
                for decision in clc["witness_repair_parent_decisions"]
            ],
            [75] * 8,
        )
        self.assertEqual(clc["witness_repair_candidate_ids"].get("5"), None)
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_axis_coverage_rejects_policy_failures(self):
        allowed = ("ok", "deduplicated", "error", "timeout", "peer_compilation_fail")
        for status in allowed:
            with self.subTest(status=status):
                succeeded = status in {"ok", "deduplicated"}
                self.assertTrue(
                    LFBOPatternSearch._flash_clc_combination_statuses_allowed(
                        [
                            {
                                "config_id": "candidate",
                                "projected_config_id": "candidate",
                                "attempt_perf": 1.0 if succeeded else None,
                                "selection_perf": 1.0 if succeeded else None,
                                "status": status,
                            },
                            {
                                "config_id": None,
                                "projected_config_id": None,
                                "attempt_perf": None,
                                "selection_perf": None,
                                "status": "projection_rejected",
                            },
                        ]
                    )
                )
        for status in ("accuracy_error", "source_rejected", "filtered", "unknown"):
            with self.subTest(status=status):
                self.assertFalse(
                    LFBOPatternSearch._flash_clc_combination_statuses_allowed(
                        [
                            {
                                "config_id": "candidate",
                                "projected_config_id": "candidate",
                                "attempt_perf": 1.0,
                                "selection_perf": 1.0,
                                "status": "ok",
                            },
                            {
                                "config_id": "failed",
                                "projected_config_id": "failed",
                                "attempt_perf": None,
                                "selection_perf": None,
                                "status": status,
                            },
                        ]
                    )
                )
        self.assertFalse(
            LFBOPatternSearch._flash_clc_combination_statuses_allowed(
                [
                    {
                        "config_id": "candidate",
                        "projected_config_id": "candidate",
                        "attempt_perf": 1.0,
                        "selection_perf": None,
                        "status": "ok",
                    }
                ]
            )
        )

    def test_lfbo_flash_terminal_refinement_allows_recorded_policy_failures(self):
        for status in (
            "ok",
            "deduplicated",
            "error",
            "timeout",
            "peer_compilation_fail",
            "accuracy_error",
            "source_rejected",
        ):
            with self.subTest(status=status):
                succeeded = status in {"ok", "deduplicated"}
                record = {
                    "config_id": "candidate",
                    "attempt_perf": 1.0 if succeeded else None,
                    "selection_perf": 1.0 if succeeded else None,
                    "status": status,
                    "source_hash": "a" * 64,
                }
                self.assertTrue(flash_terminal_refinement_result_is_valid(record))
                if status in {"accuracy_error", "source_rejected"}:
                    self.assertFalse(flash_terminal_measurement_is_valid(record))

        for status in ("filtered", "unknown", "projection_rejected"):
            with self.subTest(status=status):
                self.assertFalse(
                    flash_terminal_refinement_result_is_valid(
                        {
                            "config_id": "candidate",
                            "attempt_perf": None,
                            "selection_perf": None,
                            "status": status,
                            "source_hash": None,
                        }
                    )
                )

    def test_lfbo_flash_qualification_reserves_lanes_before_surrogate_pruning(
        self,
    ):
        family_key = "cute_flash_pipeline_family"
        softmax_key = "cute_flash_softmax_disc"
        stage_key = "cute_flash_kv_stage"

        def member(perf: float | None, stage: int, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=["fa4", stage, wait_hint, True],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_softmax_disc=True,
                    cute_flash_kv_stage=stage,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = [member(1.0, 2, 0), member(10.0, 3, 0)]
        children = {
            (stage, wait_hint): member(None, stage, wait_hint)
            for stage in (2, 3)
            for wait_hint in range(1, 5)
        }
        leaf = flash_structural_leaf_from_config(initial[0].config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            _key_to_flat_indices={
                family_key: ([0], False),
                stage_key: ([1], False),
                softmax_key: ([3], False),
            },
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 200
        search.num_neighbors_cap = 3
        search.max_generations = 20
        search.patience = 0
        search.frac_selected = 1.0
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.log = Mock()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [*initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()
        neighbor_limits: list[int] = []
        next_wait_hint = {2: 1, 3: 1}

        def generate(current, _leaf, constraints, neighbor_limit):
            stage = current.config.config[stage_key]
            self.assertEqual(dict(constraints)[stage_key], stage)
            neighbor_limits.append(neighbor_limit)
            start = next_wait_hint[stage]
            next_wait_hint[stage] += neighbor_limit
            return [
                children[(stage, wait_hint)].flat_values
                for wait_hint in range(start, start + neighbor_limit)
            ]

        search._generate_flash_leaf_neighbors = generate
        search.make_unbenchmarked = lambda flat: children[(flat[1], flat[2])]
        surrogate_stages: list[int] = []

        def hostile_surrogate(candidates, count):
            stages = {item.config.config[stage_key] for item in candidates}
            self.assertEqual(len(stages), 1)
            surrogate_stages.extend(stages)
            return sorted(
                candidates,
                key=lambda item: (
                    item.config.config[stage_key],
                    item.config.config["cute_flash_wait_hint"],
                ),
            )[:count]

        search._surrogate_select = hostile_surrogate

        def benchmark(members, *, desc):
            self.assertLessEqual(len(members), 4)
            for child in members:
                child.perfs = [20.0 + child.config.config["cute_flash_wait_hint"]]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in initial}
            ),
            2,
        )

        self.assertEqual(surrogate_stages, [2, 3])
        self.assertEqual(neighbor_limits, [1, 2])
        metrics = search._autotune_metrics.search_phase_metrics
        rounds = metrics["leaf_results"][0]["rounds"]
        self.assertEqual([len(item["candidate_config_ids"]) for item in rounds], [0, 2])
        self.assertEqual([item["neighbor_generation_limit"] for item in rounds], [0, 3])
        self.assertEqual(metrics["neighbor_generation_limit_per_leaf_per_round"], 3)
        self.assertTrue(
            all(
                lane["complete"]
                for lane in metrics["leaf_results"][0]["pipeline_lanes"]
            )
        )

    def test_lfbo_flash_orphan_compound_leaf_fails_catalog_completion(self):
        initial = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[True],
            config=helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_pipeline_family="fa4",
                cute_flash_exp2_packet="1x1",
                cute_flash_softmax_disc=True,
            ),
            status="ok",
        )
        orphan_config = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_exp2_packet="deg2_16x6",
            cute_flash_softmax_disc=False,
        )
        ordinary_leaf = flash_structural_leaf_from_config(initial.config.config)
        orphan_leaf = flash_structural_leaf_from_config(orphan_config.config)
        assert ordinary_leaf is not None and orphan_leaf is not None
        self.assertIsNone(ordinary_leaf.compound_exp2_packet)
        self.assertIsNotNone(orphan_leaf.compound_exp2_packet)
        self.assertNotEqual(ordinary_leaf.softmax_disc, orphan_leaf.softmax_disc)

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [ordinary_leaf, orphan_leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: [initial.config],
            flash_pipeline_lane_catalog=lambda: {
                ordinary_leaf: (),
                orphan_leaf: (),
            },
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = 1
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock(
            side_effect=AssertionError("orphan compound leaf has no candidates")
        )
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 1
        )

        search.benchmark_population.assert_not_called()
        search.rebenchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertFalse(metrics["completed"])
        self.assertFalse(metrics["compound_catalog_complete"])
        expected_error = {
            "family": orphan_leaf.pipeline_family,
            "compound_packet": orphan_leaf.compound_exp2_packet,
            "softmax_disc": orphan_leaf.softmax_disc,
            "error": "missing_ordinary_protocol_leaf",
            "required_parent": {
                "family": orphan_leaf.pipeline_family,
                "compound_packet": None,
                "softmax_disc": orphan_leaf.softmax_disc,
            },
        }
        self.assertEqual(metrics["compound_catalog_errors"], [expected_error])
        self.assertEqual(metrics["compound_leaf_count"], 1)
        self.assertEqual(len(metrics["compound_transfers"]), 1)
        transfer = metrics["compound_transfers"][0]
        self.assertEqual(transfer["catalog_error"], expected_error["error"])
        self.assertEqual(transfer["transfer_target_count"], 0)
        self.assertEqual(transfer["transfers"], [])
        self.assertFalse(transfer["failure_statuses_allowed"])
        self.assertFalse(transfer["complete"])
        self.assertEqual(
            search._flash_qualified_compound_config_ids, {orphan_leaf: set()}
        )

    def test_lfbo_flash_compound_failure_backfills_from_next_ordinary_source(
        self,
    ):
        stage_key = "cute_flash_kv_stage"

        def member(
            perf: float | None,
            *,
            packet: str,
            wait_hint: int,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[packet, 2, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_2cta",
                    cute_flash_exp2_packet=packet,
                    cute_flash_kv_stage=2,
                    cute_flash_wait_hint=wait_hint,
                    cute_flash_e2e_schedule=("16/6" if packet == "1x1" else "8/2"),
                    cute_flash_e2e_offset=12 if packet == "1x1" else 4,
                ),
                status="ok",
            )

        initial = member(1.0, packet="1x1", wait_hint=0)
        conditional = member(None, packet="1x1", wait_hint=1)
        collapsed_source = member(2.5, packet="1x1", wait_hint=2)
        backfill_source = member(3.0, packet="1x1", wait_hint=3)
        final_source = member(4.0, packet="1x1", wait_hint=4)
        preexisting_failure = member(
            float("inf"), packet="deg1_8x2_corr10", wait_hint=5
        )
        preexisting_failure.status = "error"
        transfers = {
            wait_hint: member(None, packet="deg1_8x2_corr10", wait_hint=wait_hint)
            for wait_hint in (0, 1, 2)
        }
        ordinary_leaf = flash_structural_leaf_from_config(initial.config.config)
        compound_leaf = flash_structural_leaf_from_config(transfers[0].config.config)
        assert ordinary_leaf is not None and compound_leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [ordinary_leaf, compound_leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=dict,
            flash_exact_effective_search_space_configs=lambda _limit: None,
            flash_pipeline_lane_catalog=lambda: {
                ordinary_leaf: ((stage_key, 2),),
                compound_leaf: (),
            },
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [
            initial,
            collapsed_source,
            backfill_source,
            final_source,
            preexisting_failure,
        ]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = Mock(
            side_effect=lambda leaf: (
                ((stage_key, 2),)
                if leaf == ordinary_leaf
                else self.fail("compound leaf requested pipeline qualification")
            )
        )
        search._pruned_pattern_search_from = lambda _index, current, *_args, **_kwargs: (
            iter(((current, conditional),))
        )

        def transfer_variant(source, _overrides, *, expected_leaf):
            self.assertEqual(expected_leaf, compound_leaf)
            wait_hint = source.config.config["cute_flash_wait_hint"]
            if wait_hint == 0:
                return preexisting_failure
            if wait_hint in (1, 2):
                return transfers[0]
            if wait_hint == 3:
                return transfers[1]
            return transfers[2]

        search._flash_config_variant = Mock(side_effect=transfer_variant)
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()

        def rebenchmark(_members, *, desc):
            if desc == "Compound packet transfers: verifying":
                initial.perfs.append(100.0)
                conditional.perfs.append(200.0)
                collapsed_source.perfs.append(0.1)
                backfill_source.perfs.append(0.2)
                final_source.perfs.append(0.3)

        search.rebenchmark_population = Mock(side_effect=rebenchmark)
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if members == [conditional]:
                self.assertEqual(desc, "Structural qualification 2:")
                conditional.perfs = [2.0]
                return
            if desc == "Compound packet transfers:":
                self.assertCountEqual(members, [transfers[0], transfers[1]])
                transfers[0].perfs = [float("inf")]
                transfers[0].status = "error"
                transfers[1].perfs = [3.0]
                return
            self.assertEqual(desc, "Compound packet failure backfills 1:")
            self.assertEqual(members, [transfers[2]])
            transfers[2].perfs = [3.0]
            # The backfill discovers the failed primary has the same effective
            # source, so source repair promotes it as well as the new candidate.
            transfers[0].perfs = [3.1]
            transfers[0].status = "deduplicated"

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {
                    initial.config,
                    collapsed_source.config,
                    backfill_source.config,
                    final_source.config,
                    preexisting_failure.config,
                }
            ),
            4,
        )
        search._flash_pipeline_lanes.assert_called_once_with(ordinary_leaf)
        self.assertEqual(search._flash_config_variant.call_count, 5)
        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertTrue(metrics["completed"])
        self.assertEqual(metrics["ordinary_leaf_count"], 1)
        self.assertEqual(metrics["compound_leaf_count"], 1)
        self.assertTrue(metrics["compound_catalog_complete"])
        self.assertEqual(metrics["compound_catalog_errors"], [])
        self.assertEqual(metrics["leaves_with_candidates"], 2)
        self.assertEqual(len(metrics["leaf_results"]), 1)
        compound = metrics["compound_transfers"][0]
        self.assertEqual(compound["transfer_target_count"], 2)
        self.assertEqual(compound["transfer_count"], 3)
        self.assertEqual(
            compound["primary_transfer_config_ids"],
            [
                canonical_config_id(transfers[0].config),
                canonical_config_id(transfers[1].config),
            ],
        )
        self.assertEqual(
            compound["successful_transfer_config_ids"],
            [
                canonical_config_id(transfers[0].config),
                canonical_config_id(transfers[1].config),
                canonical_config_id(transfers[2].config),
            ],
        )
        self.assertEqual(
            compound["qualified_transfer_config_ids"],
            [
                canonical_config_id(transfers[0].config),
                canonical_config_id(transfers[1].config),
            ],
        )
        self.assertEqual(
            compound["backfill_rounds"],
            [
                {
                    "repair_index": 0,
                    "required_successes": 1,
                    "failed_transfer_config_ids": [
                        canonical_config_id(transfers[0].config)
                    ],
                    "attempted_source_config_ids": [
                        canonical_config_id(final_source.config)
                    ],
                    "generated_config_ids": [canonical_config_id(transfers[2].config)],
                }
            ],
        )
        self.assertTrue(compound["complete"])
        self.assertEqual(
            [
                transfer["source_config_id"]
                for transfer in metrics["compound_transfers"][0]["transfers"]
            ],
            [
                canonical_config_id(conditional.config),
                canonical_config_id(backfill_source.config),
                canonical_config_id(final_source.config),
            ],
        )
        source_selection = metrics["compound_transfers"][0]["source_selection"]
        self.assertEqual(
            [
                (result["config_id"], result["selection_perf"])
                for result in source_selection["candidate_results"]
            ],
            [
                (canonical_config_id(initial.config), 1.0),
                (canonical_config_id(conditional.config), 2.0),
                (canonical_config_id(collapsed_source.config), 2.5),
                (canonical_config_id(backfill_source.config), 3.0),
                (canonical_config_id(final_source.config), 4.0),
            ],
        )
        self.assertEqual(
            source_selection["attempted_config_ids"],
            [
                canonical_config_id(initial.config),
                canonical_config_id(conditional.config),
                canonical_config_id(collapsed_source.config),
                canonical_config_id(backfill_source.config),
                canonical_config_id(final_source.config),
            ],
        )
        self.assertEqual(
            source_selection["selected_config_ids"],
            [
                canonical_config_id(conditional.config),
                canonical_config_id(backfill_source.config),
                canonical_config_id(final_source.config),
            ],
        )
        self.assertNotIn(
            canonical_config_id(preexisting_failure.config),
            {
                transfer["transferred_config_id"]
                for transfer in metrics["compound_transfers"][0]["transfers"]
            },
        )
        source_configs = {
            canonical_config_id(source.config): source.config.config
            for source in (conditional, backfill_source, final_source)
        }
        for transfer in metrics["compound_transfers"][0]["transfers"]:
            is_repaired = transfer["transferred_config_id"] == canonical_config_id(
                transfers[0].config
            )
            self.assertEqual(transfer["attempt_perf"], 3.1 if is_repaired else 3.0)
            self.assertEqual(transfer["selection_perf"], 3.1 if is_repaired else 3.0)
            self.assertEqual(
                transfer["status"], "deduplicated" if is_repaired else "ok"
            )
            self.assertEqual(
                transfer["projected_config_id"],
                transfer["transferred_config_id"],
            )
            self.assertEqual(
                transfer["source_config"],
                source_configs[transfer["source_config_id"]],
            )
            source_snapshot = transfer["source_config"]
            projected_snapshot = transfer["projected_config"]
            self.assertIsInstance(source_snapshot, dict)
            self.assertIsInstance(projected_snapshot, dict)
            assert isinstance(source_snapshot, dict)
            assert isinstance(projected_snapshot, dict)
            self.assertEqual(
                canonical_config_id(helion.Config.from_dict(source_snapshot)),
                transfer["source_config_id"],
            )
            projected_snapshot_id = canonical_config_id(
                helion.Config.from_dict(projected_snapshot)
            )
            self.assertEqual(
                projected_snapshot_id,
                transfer["projected_config_id"],
            )
            self.assertEqual(
                projected_snapshot_id,
                transfer["transferred_config_id"],
            )
            self.assertEqual(
                transfer["projection_overrides"],
                {cute_flash.FLASH_EXP2_PACKET_KEY: "deg1_8x2_corr10"},
            )
            self.assertEqual(source_snapshot[cute_flash.FLASH_E2E_OFFSET_KEY], 12)
            self.assertEqual(projected_snapshot[cute_flash.FLASH_E2E_OFFSET_KEY], 4)
            self.assertEqual(
                transfer["preserved_pipeline_values"],
                {stage_key: 2},
            )

        untransferred = member(0.1, packet="deg1_8x2_corr10", wait_hint=99)
        search.population.append(untransferred)
        search._flash_pipeline_lanes = lambda leaf: (
            search.config_gen.flash_pipeline_lane_catalog()[leaf]
        )
        selected = {member.config for member, _ in search._select_starting_paths()}
        self.assertNotIn(untransferred.config, selected)
        self.assertNotIn(transfers[2].config, selected)
        self.assertLessEqual({transfers[0].config, transfers[1].config}, selected)

    def test_lfbo_flash_exact_clc_space_skips_conditional_and_combination_work(
        self,
    ):
        stage_key = "cute_flash_kv_stage"

        def member(clc_value: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float(clc_value)],
                flat_values=[2, clc_value],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_clc_heads_per_batch=clc_value,
                ),
                status="ok",
            )

        initial = [member(1), member(2)]
        leaf = flash_structural_leaf_from_config(initial[0].config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (1, 2),
                    "search_values": (1, 2),
                    "anchor_values": (1, 2),
                    "refinement_values": (),
                    "attempted_values": (1, 2),
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: [
                item.config for item in initial
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = initial
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_clc_lane_witness = lambda _leaf, value: initial[value - 1]
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("exhausted CLC space must not search")
        )
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in initial}
            ),
            3,
        )
        search._pruned_pattern_search_from.assert_not_called()
        search.benchmark_population.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertTrue(metrics["exact_space_exhausted"])
        self.assertEqual(metrics["candidate_count"], 0)
        clc = metrics["clc_families"][0]
        self.assertTrue(clc["space_exhausted"])
        self.assertEqual(clc["value_space_exhausted"], {"1": True, "2": True})
        self.assertEqual(clc["selected_values"], [1, 2])
        self.assertEqual(clc["conditional_values"], [])
        self.assertEqual(clc["conditional_candidate_ids"], {})
        self.assertFalse(clc["combination_required"])
        self.assertEqual(clc["combination_candidate_ids"], [])
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_exact_anchors_do_not_suppress_clc_refinement(self):
        stage_key = "cute_flash_kv_stage"
        clc_key = "cute_flash_clc_heads_per_batch"

        def member(
            perf: float | None, *, clc_value: int, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[2, clc_value, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=2,
                    cute_flash_clc_heads_per_batch=clc_value,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        anchors = [
            member(1.0, clc_value=1, wait_hint=0),
            member(2.0, clc_value=2, wait_hint=0),
        ]
        refinement_witnesses = [
            member(None, clc_value=value, wait_hint=0) for value in (4, 8, 16, 32)
        ]
        pipeline_child = member(None, clc_value=1, wait_hint=1)
        refinement_children = {
            value: member(None, clc_value=value, wait_hint=2)
            for value in (4, 8, 16, 32)
        }
        combinations = {
            value: member(None, clc_value=value, wait_hint=3)
            for value in (1, 2, 4, 8, 16, 32)
        }
        leaf = flash_structural_leaf_from_config(anchors[0].config.config)
        assert leaf is not None
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (1, 2, 4, 8, 16, 32),
                    "search_values": (1, 2, 4, 8, 16, 32),
                    "anchor_values": (1, 2),
                    "refinement_values": (4, 8, 16, 32),
                    "attempted_values": (1, 2, 4, 8, 16, 32),
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: [
                item.config for item in anchors
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            retained_candidates_per_leaf=1,
        )
        search.population = anchors
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: ((stage_key, 2),)
        search._flash_clc_lane_witness = lambda _leaf, value: {
            1: anchors[0],
            2: anchors[1],
            **dict(zip((4, 8, 16, 32), refinement_witnesses, strict=True)),
        }[value]

        def qualification_path(_index, current, _visited, constraints, **_kwargs):
            constraint_values = dict(constraints)
            child = (
                refinement_children[constraint_values[clc_key]]
                if clc_key in constraint_values
                else pipeline_child
            )
            return iter(((current, child),))

        search._pruned_pattern_search_from = qualification_path
        composition_sources: list[tuple[PopulationMember, int]] = []

        def clc_depth_variant(source, value, **_kwargs):
            composition_sources.append((source, value))
            return combinations[value]

        search._flash_clc_depth_variant = clc_depth_variant
        search._budgeted_range = lambda *args: range(*args)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if desc == "Structural qualification 2:":
                self.assertEqual(members, [pipeline_child])
                pipeline_child.perfs = [0.8]
            elif desc == "CLC divisor witnesses 1:":
                self.assertEqual(members, refinement_witnesses)
                for witness, perf in zip(
                    refinement_witnesses, (0.5, 3.0, 4.0, 5.0), strict=True
                ):
                    witness.perfs = [perf]
            elif desc == "CLC divisor conditional children:":
                self.assertCountEqual(members, refinement_children.values())
                for offset, child in enumerate(refinement_children.values()):
                    child.perfs = [0.4 + offset * 0.01]
            else:
                self.assertEqual(desc, "CLC depth/divisor combinations:")
                self.assertCountEqual(members, combinations.values())
                for combination in members:
                    combination.perfs = [0.3]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification(
                {item.config for item in anchors}
            ),
            5,
        )
        metrics = search._autotune_metrics.search_phase_metrics
        self.assertFalse(metrics["exact_space_exhausted"])
        lane = metrics["leaf_results"][0]["pipeline_lanes"][0]
        self.assertFalse(lane["space_exhausted"])
        self.assertTrue(lane["conditional_required"])
        clc = metrics["clc_families"][0]
        self.assertEqual(
            clc["value_space_exhausted"],
            {
                "1": True,
                "2": True,
                "4": False,
                "8": False,
                "16": False,
                "32": False,
            },
        )
        self.assertEqual(clc["attempted_values"], [1, 2, 4, 8, 16, 32])
        self.assertEqual(clc["selected_values"], [4, 1, 2, 8, 16, 32])
        self.assertEqual(clc["conditional_values"], [4, 8, 16, 32])
        self.assertTrue(clc["combination_required"])
        self.assertEqual(
            composition_sources,
            [(refinement_children[4], value) for value in (4, 8, 16, 32, 1, 2)],
        )
        self.assertEqual(
            set(clc["combination_candidate_ids"]),
            {
                canonical_config_id(combination.config)
                for combination in combinations.values()
            },
        )
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_flash_clc_refines_ranks_and_combines_bounded_values(self):
        stage_key = "cute_flash_kv_stage"
        clc_key = "cute_flash_clc_heads_per_batch"

        def member(
            perf: float | None, *, stage: int = 2, clc_value: int, wait_hint: int
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[stage, clc_value, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_clc_heads_per_batch=clc_value,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        initial = member(10.0, clc_value=1, wait_hint=0)
        pipeline_witness = member(None, stage=3, clc_value=1, wait_hint=0)
        pipeline_children = {
            stage: member(None, stage=stage, clc_value=1, wait_hint=1)
            for stage in (2, 3)
        }
        clc_witnesses = {
            value: initial if value == 1 else member(None, clc_value=value, wait_hint=0)
            for value in (1, 8, 2, 4)
        }
        clc_children = {
            value: member(None, clc_value=value, wait_hint=2) for value in (1, 8, 2, 4)
        }
        # The stage-2/value-4 projection is already measured by the CLC
        # conditional pass, matching a normal canonical-combination collision.
        clc_children[4] = member(None, clc_value=4, wait_hint=1)
        combined = {
            (stage, value): member(
                None,
                stage=stage,
                clc_value=value,
                wait_hint=3,
            )
            for stage in (2, 3)
            for value in (1, 8, 2, 4)
        }
        combined[(2, 4)] = clc_children[4]
        leaf = flash_structural_leaf_from_config(initial.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (1, 2, 4, 8),
                    "search_values": (1, 2, 4, 8),
                    "anchor_values": (1, 8),
                    "refinement_values": (2, 4),
                    "attempted_values": (1, 8, 2, 4),
                }
            },
            flash_exact_effective_search_space_configs=lambda _limit: None,
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search.population = [initial]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )
        search._flash_pipeline_lane_witness = lambda _leaf, lane: (
            pipeline_witness if lane == (stage_key, 3) else None
        )
        search._flash_clc_lane_witness = lambda _leaf, value: clc_witnesses[value]

        def qualification_path(_index, current, _visited, constraints, **_kwargs):
            constraint_values = dict(constraints)
            if clc_key in constraint_values:
                child = clc_children[constraint_values[clc_key]]
            else:
                child = pipeline_children[current.config.config[stage_key]]
            return iter(((current, child),))

        search._pruned_pattern_search_from = qualification_path
        composition_sources: list[tuple[PopulationMember, int]] = []

        def clc_depth_variant(source, value, **_kwargs):
            composition_sources.append((source, value))
            return combined[(source.config.config[stage_key], value)]

        search._flash_clc_depth_variant = clc_depth_variant
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()

        def rebenchmark(_members, *, desc):
            if desc == "CLC divisor conditional children: verifying":
                clc_witnesses[2].perfs.append(50.0)
                clc_witnesses[4].perfs.append(60.0)
            elif desc == "CLC depth/divisor combinations: verifying":
                pipeline_children[2].perfs.append(9.0)
                pipeline_children[3].perfs.append(8.0)

        search.rebenchmark_population = Mock(side_effect=rebenchmark)
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            if members == [pipeline_witness]:
                self.assertEqual(desc, "Structural qualification 1:")
                pipeline_witness.perfs = [1.1]
                return
            if desc == "Structural qualification 2:":
                self.assertCountEqual(members, pipeline_children.values())
                self.assertEqual(desc, "Structural qualification 2:")
                pipeline_children[2].perfs = [0.1]
                pipeline_children[3].perfs = [0.2]
                return
            if desc == "CLC divisor witnesses 1:":
                self.assertCountEqual(
                    members,
                    [clc_witnesses[8], clc_witnesses[2], clc_witnesses[4]],
                )
                self.assertEqual(desc, "CLC divisor witnesses 1:")
                for witness in members:
                    witness.perfs = [float(witness.config.config[clc_key])]
                return
            if desc == "CLC depth/divisor combinations:":
                self.assertCountEqual(
                    members,
                    [
                        combination
                        for key, combination in combined.items()
                        if key != (2, 4)
                    ],
                )
                for combination in members:
                    combination.perfs = [3.0]
                return
            self.assertCountEqual(members, clc_children.values())
            self.assertEqual(desc, "CLC divisor conditional children:")
            clc_children[2].perfs = [float("inf")]
            clc_children[2].status = "error"
            clc_children[4].perfs = [0.5]
            clc_children[1].perfs = [1.5]
            clc_children[8].perfs = [7.5]

        search.benchmark_population = benchmark

        self.assertEqual(
            search._run_flash_structural_qualification({initial.config}), 6
        )
        metrics = search._autotune_metrics.search_phase_metrics
        _assert_phase_config_manifest(self, metrics)
        self.assertFalse(metrics["completed"])
        self.assertEqual(metrics["candidate_count"], 17)
        clc = metrics["clc_families"][0]
        self.assertEqual(clc["legal_values"], [1, 2, 4, 8])
        self.assertEqual(clc["anchor_values"], [1, 8])
        self.assertEqual(clc["refinement_values"], [2, 4])
        self.assertEqual(clc["attempted_values"], [1, 8, 2, 4])
        self.assertEqual(
            clc["witness_config_ids"],
            {
                str(value): canonical_config_id(witness.config)
                for value, witness in clc_witnesses.items()
            },
        )
        self.assertEqual(clc["conditional_values"], [2, 4, 8, 1])
        self.assertEqual(
            [
                (result["value"], result["selection_perf"])
                for result in clc["witness_candidate_results"]
            ],
            [(1, 10.0), (8, 8.0), (2, 2.0), (4, 4.0)],
        )
        self.assertEqual(
            [
                (result["value"], result["selection_perf"])
                for result in clc["witness_selection_results"]
            ],
            [(2, 2.0), (4, 4.0), (8, 8.0), (1, 10.0)],
        )
        self.assertEqual(
            clc["selected_config_ids"],
            [
                canonical_config_id(clc_witnesses[2].config),
                canonical_config_id(clc_witnesses[4].config),
                canonical_config_id(clc_witnesses[8].config),
                canonical_config_id(clc_witnesses[1].config),
            ],
        )
        self.assertEqual(
            [
                (
                    decision["value"],
                    decision["candidate_results"][0]["selection_perf"],
                )
                for decision in clc["conditional_parent_decisions"]
            ],
            [(2, 2.0), (4, 4.0), (8, 8.0), (1, 10.0)],
        )
        self.assertEqual(
            clc["conditional_candidate_ids"],
            {
                str(value): [canonical_config_id(child.config)]
                for value, child in clc_children.items()
            },
        )
        self.assertEqual(clc["retained_values"], [4, 1, 8, 2])
        self.assertEqual(
            [
                (result["value"], result["selection_perf"])
                for result in clc["retained_ranking_results"]
            ],
            [(4, 0.5), (1, 1.5), (8, 7.5), (2, 50.0)],
        )
        self.assertEqual(
            clc["retained_config_ids"],
            [
                canonical_config_id(clc_children[4].config),
                canonical_config_id(clc_children[1].config),
                canonical_config_id(clc_children[8].config),
                canonical_config_id(clc_witnesses[2].config),
            ],
        )
        retained_value_decisions = {
            decision["value"]: decision for decision in clc["retained_value_decisions"]
        }
        self.assertEqual(
            retained_value_decisions[2]["candidate_results"][-1]["status"],
            "error",
        )
        self.assertIsNone(
            retained_value_decisions[2]["candidate_results"][-1]["selection_perf"]
        )
        self.assertEqual(
            [
                (
                    representative["config_id"],
                    (
                        None
                        if representative["assigned_pipeline_lane"] is None
                        else representative["assigned_pipeline_lane"]["value"]
                    ),
                )
                for representative in clc["depth_selection"]["selected_representatives"]
            ],
            [
                (canonical_config_id(pipeline_children[2].config), None),
                (canonical_config_id(pipeline_children[3].config), 3),
            ],
        )
        depth_results = {
            result["config_id"]: result
            for result in clc["depth_selection"]["candidate_results"]
        }
        reused_combination_id = canonical_config_id(clc_children[4].config)
        self.assertIn(reused_combination_id, depth_results)
        self.assertIn(reused_combination_id, clc["combination_candidate_ids"])
        self.assertEqual(
            depth_results[canonical_config_id(pipeline_children[2].config)][
                "selection_perf"
            ],
            0.1,
        )
        self.assertEqual(
            depth_results[canonical_config_id(pipeline_children[3].config)][
                "selection_perf"
            ],
            0.2,
        )
        self.assertEqual(
            set(clc["combination_candidate_ids"]),
            {
                canonical_config_id(combination.config)
                for combination in combined.values()
            },
        )
        self.assertEqual(len(clc["combination_candidate_ids"]), 8)
        self.assertCountEqual(
            composition_sources,
            [
                (pipeline_children[stage], value)
                for stage in (2, 3)
                for value in (4, 1, 8, 2)
            ],
        )
        self.assertFalse(clc["complete"])
        self.assertEqual(
            clc["conditional_repair_candidate_ids"],
            {"2": []},
        )
        self.assertEqual(
            clc["conditional_repair_parent_decisions"][0]["selected_config_id"],
            canonical_config_id(clc_children[2].config),
        )

    def test_lfbo_flash_clc_combination_failure_uses_axis_coverage(self):
        stage_key = "cute_flash_kv_stage"

        def member(
            stage: int,
            clc_value: int,
            wait_hint: int,
            perf: float | None,
        ) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[] if perf is None else [perf],
                flat_values=[stage, clc_value, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_clc",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_kv_stage=stage,
                    cute_flash_clc_heads_per_batch=clc_value,
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        depth_2 = member(2, 2, 0, 1.0)
        depth_3 = member(3, 4, 0, 2.0)
        failed_cell = member(2, 4, 1, None)
        successful_cell = member(3, 2, 1, None)
        combined = {
            (2, 2): depth_2,
            (2, 4): failed_cell,
            (3, 2): successful_cell,
            (3, 4): depth_3,
        }
        unmeasured = member(2, 8, 0, None).config
        leaf = flash_structural_leaf_from_config(depth_2.config.config)
        assert leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.config_gen = SimpleNamespace(
            encode_config=lambda flat: flat,
            flash_structural_leaf_catalog=lambda: [leaf],
            flash_low_confound_schedule_anchor_configs=list,
            flash_clc_lane_catalog=lambda: {
                leaf: {
                    "legal_values": (2, 4),
                    "search_values": (2, 4),
                    "anchor_values": (2, 4),
                    "refinement_values": (),
                    "attempted_values": (2, 4),
                }
            },
            # Value-specific spaces are measured, avoiding unrelated CLC
            # conditional searches. The extra config keeps the full leaf
            # non-exhaustive so the depth/divisor projection is still required.
            flash_exact_effective_search_space_configs=lambda _limit: [
                depth_2.config,
                depth_3.config,
                unmeasured,
            ],
        )
        search.copies = 5
        search.num_neighbors = 300
        search.num_neighbors_cap = -1
        search.max_generations = 20
        policy = get_effort_profile("full").flash_structural_search
        assert policy is not None
        search.flash_structural_search = replace(
            policy,
            qualification_rounds=1,
            conditional_candidates_per_pipeline_lane=0,
        )
        search.population = [depth_2, depth_3]
        search.initial_population = len(search.population)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._flash_pipeline_lanes = lambda _leaf: (
            (stage_key, 2),
            (stage_key, 3),
        )
        search._flash_pipeline_lane_witness = Mock(
            side_effect=AssertionError("measured lane members must be reused")
        )
        search._flash_clc_lane_witness = lambda _leaf, value: {
            2: depth_2,
            4: depth_3,
        }[value]
        search._pruned_pattern_search_from = Mock(
            side_effect=AssertionError("no conditional search is required")
        )
        search._flash_clc_depth_variant = lambda source, value, **_kwargs: combined[
            (source.config.config[stage_key], value)
        ]
        search._budgeted_range = lambda *args: range(*args)
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=False)
        search.set_generation = Mock()
        search.rebenchmark_population = Mock()
        search._fit_surrogate = Mock()

        def benchmark(members, *, desc):
            self.assertEqual(desc, "CLC depth/divisor combinations:")
            self.assertCountEqual(members, [failed_cell, successful_cell])
            failed_cell.perfs = [float("inf")]
            failed_cell.status = "error"
            successful_cell.perfs = [0.5]

        search.benchmark_population = Mock(side_effect=benchmark)

        self.assertEqual(
            search._run_flash_structural_qualification(
                {depth_2.config, depth_3.config}
            ),
            3,
        )
        search.benchmark_population.assert_called_once()
        search._flash_pipeline_lane_witness.assert_not_called()
        search._pruned_pattern_search_from.assert_not_called()
        metrics = search._autotune_metrics.search_phase_metrics
        clc = metrics["clc_families"][0]
        self.assertEqual(len(clc["combination_cells"]), 4)
        self.assertCountEqual(
            clc["combination_depth_config_ids"],
            [
                canonical_config_id(depth_2.config),
                canonical_config_id(depth_3.config),
            ],
        )
        self.assertEqual(clc["combination_divisor_values"], [2, 4])
        self.assertCountEqual(
            clc["successful_combination_depth_config_ids"],
            [
                canonical_config_id(depth_2.config),
                canonical_config_id(depth_3.config),
            ],
        )
        self.assertEqual(clc["successful_combination_divisor_values"], [2, 4])
        self.assertTrue(clc["combination_row_coverage_complete"])
        self.assertTrue(clc["combination_column_coverage_complete"])
        cell_by_axis = {
            (cell["depth_config_id"], cell["divisor_value"]): cell
            for cell in clc["combination_cells"]
        }
        failed_result = cell_by_axis[(canonical_config_id(depth_2.config), 4)]
        self.assertEqual(
            failed_result["projected_config_id"],
            canonical_config_id(failed_cell.config),
        )
        self.assertEqual(
            failed_result["config_id"], canonical_config_id(failed_cell.config)
        )
        self.assertIsNone(failed_result["attempt_perf"])
        self.assertIsNone(failed_result["selection_perf"])
        self.assertEqual(failed_result["status"], "error")
        self.assertTrue(clc["complete"])
        self.assertTrue(metrics["completed"])

    def test_lfbo_only_full_flash_unrestricted_path_exhausts_budget(self):
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = _cute_flash_test_config_spec()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search

        self.assertTrue(search._path_exhausts_generation_budget(()))
        self.assertFalse(
            search._path_exhausts_generation_budget(
                (("cute_flash_pipeline_family", "fa4"),)
            )
        )

        search.flash_structural_search = get_effort_profile(
            "quick"
        ).flash_structural_search
        self.assertFalse(search._path_exhausts_generation_budget(()))
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=False)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        self.assertFalse(search._path_exhausts_generation_budget(()))

    def test_lfbo_pipeline_qualification_is_flash_policy_gated(self):
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.max_generations = 20
        search.config_gen = Mock()

        search.config_spec = SimpleNamespace(cute_flash_search_enabled=False)
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        self.assertEqual(search._run_flash_structural_qualification(set()), 0)

        search.config_spec = _cute_flash_test_config_spec()
        search.flash_structural_search = None
        self.assertEqual(search._run_flash_structural_qualification(set()), 0)
        self.assertFalse(search.config_gen.flash_structural_leaf_catalog.called)

    def test_lfbo_disable_early_stopping_exhausts_generation_budget(self):
        family_key = "cute_flash_pipeline_family"

        def run_path(
            *, constraints: tuple[tuple[str, object], ...], disable: bool
        ) -> tuple[int, int]:
            def member(index: int) -> PopulationMember:
                return PopulationMember(
                    fn=lambda: None,
                    perfs=[1.0 if index == 0 else 2.0],
                    flat_values=[index, "fa4"],
                    config=helion.Config(
                        block_sizes=[1, 128, 128],
                        cute_flash_pipeline_family="fa4",
                        cute_flash_wait_hint=index,
                    ),
                    status="ok",
                )

            current = member(0)
            neighbors = iter(member(index) for index in range(1, 10))
            search = LFBOPatternSearch.__new__(LFBOPatternSearch)
            search.max_generations = 4
            search.patience = 0
            search.frac_selected = 1.0
            search.kernel = SimpleNamespace(
                env=SimpleNamespace(process_group_name=None)
            )
            search.log = lambda *_args, **_kwargs: None
            search.config_gen = SimpleNamespace(
                _key_to_flat_indices={family_key: ([1], False)}
            )
            search._generate_neighbors = lambda *_args, **_kwargs: [
                next(neighbors).flat_values
            ]
            by_index = {index: member(index) for index in range(1, 10)}
            search.make_unbenchmarked = lambda flat: by_index[flat[0]]
            search._surrogate_select = lambda candidates, _count: candidates
            search._check_early_stopping = Mock(return_value=True)

            generations = list(
                search._pruned_pattern_search_from(
                    0,
                    current,
                    set(),
                    constraints,
                    disable_early_stopping=disable,
                )
            )
            return len(generations), search._check_early_stopping.call_count

        self.assertEqual(run_path(constraints=(), disable=True), (4, 0))
        self.assertEqual(run_path(constraints=(), disable=False), (1, 1))
        self.assertEqual(
            run_path(constraints=((family_key, "fa4"),), disable=False),
            (1, 1),
        )

    def test_lfbo_non_flash_does_not_require_json_serializable_configs(self):
        config = helion.Config(
            block_sizes=[1],
            custom_non_json_value=Path("custom-value"),
        )
        member = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[1],
            config=config,
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.log = Mock()
        search.copies = 1
        search.max_generations = 0
        search.similarity_penalty = 1.0
        search._generate_initial_population_flat = lambda: [[1]]
        search.make_unbenchmarked = lambda _flat: member
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.compile_timeout_lower_bound = 0.0
        search.compile_timeout_quantile = 0.0
        search.set_adaptive_compile_timeout = Mock()
        search.rebenchmark_population = Mock()
        search.kernel = SimpleNamespace(
            env=SimpleNamespace(process_group_name=None),
        )
        search.capture_compiler_seed_members = Mock()
        search.config_gen = SimpleNamespace(encode_config=lambda _flat: [0.0])
        search.train_x = []
        search.train_y = []
        search.train_configs = None
        search.train_source_hashes = None
        search._fit_surrogate = Mock()
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=False)
        search._cute_flash_lane_policy_enabled = False
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._finalize = lambda: config

        with patch(
            "helion.autotuner.surrogate_pattern_search.check_population_consistency"
        ):
            self.assertIs(search._autotune(), config)

    def test_lfbo_flash_qualification_preserves_initial_generation_order(self):
        def member(wait_hint: int, perf: float) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[perf],
                flat_values=[wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        generated = [member(0, 2.0), member(1, 1.0)]
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.log = Mock()
        search.copies = 1
        search.max_generations = 0
        search.similarity_penalty = 1.0
        search._generate_initial_population_flat = lambda: [
            item.flat_values for item in generated
        ]
        search.make_unbenchmarked = lambda flat: generated[flat[0]]
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.compile_timeout_lower_bound = 0.0
        search.compile_timeout_quantile = 0.0
        search.set_adaptive_compile_timeout = Mock()
        search.rebenchmark_population = Mock()
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.capture_compiler_seed_members = Mock()
        search.config_gen = SimpleNamespace(encode_config=lambda flat: flat)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search.train_source_hashes = []
        search._fit_surrogate = Mock()
        search.config_spec = _cute_flash_test_config_spec()
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)

        def qualify(_visited, *, initial_population):
            self.assertEqual(initial_population, generated)
            self.assertEqual(search.population, list(reversed(generated)))
            return 0

        search._run_flash_structural_qualification = Mock(side_effect=qualify)
        search._select_starting_paths = Mock(return_value=[(generated[1], ())])
        search._finalize = lambda: generated[1].config

        with patch(
            "helion.autotuner.surrogate_pattern_search.check_population_consistency"
        ):
            self.assertIs(search._autotune(), generated[1].config)

        search._run_flash_structural_qualification.assert_called_once()

    def test_lfbo_retained_flash_paths_use_conditional_surfaces(self):
        def member(family: str, wait_hint: int) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[1.0],
                flat_values=[family, wait_hint],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family=family,
                    cute_flash_exp2_packet="1x1",
                    cute_flash_wait_hint=wait_hint,
                ),
                status="ok",
            )

        unrestricted = member("fa4", 0)
        constrained = member("fa4_2cta", 1)
        constrained_leaf = flash_structural_leaf_from_config(constrained.config.config)
        assert constrained_leaf is not None
        constraints = LFBOPatternSearch._flash_leaf_constraints(constrained_leaf)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.log = Mock()
        search.copies = 2
        search.max_generations = 0
        search.similarity_penalty = 1.0
        search._generate_initial_population_flat = lambda: [unrestricted.flat_values]
        search.make_unbenchmarked = lambda _flat: unrestricted
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.compile_timeout_lower_bound = 0.0
        search.compile_timeout_quantile = 0.0
        search.set_adaptive_compile_timeout = Mock()
        search.rebenchmark_population = Mock()
        search.kernel = SimpleNamespace(
            env=SimpleNamespace(process_group_name=None),
        )
        search.capture_compiler_seed_members = Mock()
        search.config_gen = SimpleNamespace(encode_config=lambda flat: flat)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._fit_surrogate = Mock()
        search.config_spec = _cute_flash_test_config_spec()
        search.flash_structural_search = get_effort_profile(
            "full"
        ).flash_structural_search
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)
        search._run_flash_structural_qualification = Mock(return_value=0)
        search._select_starting_paths = Mock(
            return_value=[(unrestricted, ()), (constrained, constraints)]
        )
        search._pruned_pattern_search_from = Mock(return_value=iter(()))
        search._finalize = lambda: unrestricted.config

        with patch(
            "helion.autotuner.surrogate_pattern_search.check_population_consistency"
        ):
            self.assertEqual(search._autotune(), unrestricted.config)

        calls = search._pruned_pattern_search_from.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0].kwargs["required_leaf"])
        self.assertFalse(calls[0].kwargs["conditional_surface"])
        self.assertTrue(calls[0].kwargs["selected_only_visited"])
        self.assertEqual(calls[1].kwargs["required_leaf"], constrained_leaf)
        self.assertTrue(calls[1].kwargs["conditional_surface"])
        self.assertFalse(calls[1].kwargs["selected_only_visited"])

    def test_lfbo_flash_wall_budget_returns_best_before_required_probe(self):
        config = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_exp2_packet="1x1",
        )
        member = PopulationMember(
            fn=lambda: None,
            perfs=[1.0],
            flat_values=[1],
            config=config,
            status="ok",
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.log = Mock()
        search.copies = 1
        search.max_generations = 20
        search.similarity_penalty = 1.0
        search._generate_initial_population_flat = lambda: [[1]]
        search.make_unbenchmarked = lambda _flat: member
        search.set_generation = Mock()
        search.benchmark_population = Mock()
        search.compile_timeout_lower_bound = 0.0
        search.compile_timeout_quantile = 0.0
        search.set_adaptive_compile_timeout = Mock()
        search.rebenchmark_population = Mock()
        search.kernel = SimpleNamespace(
            env=SimpleNamespace(process_group_name=None),
        )
        search.capture_compiler_seed_members = Mock()
        search.config_gen = SimpleNamespace(encode_config=lambda flat: flat)
        search.train_x = []
        search.train_y = []
        search.train_configs = []
        search._fit_surrogate = Mock()
        search.config_spec = _cute_flash_test_config_spec()
        search._cute_flash_lane_policy_enabled = True
        search._autotune_metrics = SimpleNamespace(search_phase_metrics=None)

        def incomplete_qualification(_visited, *, initial_population):
            self.assertEqual(initial_population, [member])
            search._autotune_metrics.search_phase_metrics = {
                "family_probe_required": True,
                "family_probe_complete": False,
            }
            return 1

        search._run_flash_structural_qualification = incomplete_qualification
        search._autotune_budget_exceeded_across_ranks = Mock(return_value=True)
        search._select_starting_paths = Mock()
        search._finalize = Mock(return_value=config)

        with patch(
            "helion.autotuner.surrogate_pattern_search.check_population_consistency"
        ):
            self.assertEqual(search._autotune(), config)

        search._select_starting_paths.assert_not_called()
        search._finalize.assert_called_once_with()

    def test_cute_flash_best_available_partial_and_zero_population(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(5)}

        def canonicalize_flat(flat):
            value = flat[0]
            if value < 0:
                raise exc.InvalidConfig("invalid")
            canonical_value = value % len(configs)
            return [canonical_value], configs[canonical_value]

        config_gen = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            flash_deterministic_population_configs=lambda: [configs[1], configs[2]],
            flash_exact_effective_search_space_configs=lambda _limit: None,
            flash_structural_parent_coverage_prefix_count=lambda: 2,
            flash_structural_population_budget=lambda _target: 2,
            flash_structural_qualification_prefix_count=lambda: 2,
            flatten=lambda config: [config.config["block_sizes"][0] - 1],
            canonicalize_flat=canonicalize_flat,
            random_flat=Mock(side_effect=([-1], [4])),
        )
        search = PatternSearch.__new__(PatternSearch)
        search.config_gen = config_gen
        search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_BEST_AVAILABLE
        )
        search.best_available_pad_random = True
        search.log = Mock()
        search._pinned_finalist_configs = set()
        search._generate_best_available_population_flat = lambda: [[5], [0], [3]]

        search.initial_population = 1
        partial = search._generate_initial_population_flat()
        self.assertEqual(partial, [[1]])
        self.assertEqual(len(partial), len({tuple(flat) for flat in partial}))

        search.initial_population = 5
        padded = search._generate_initial_population_flat()
        self.assertEqual(padded, [[1], [2], [0], [3], [4]])
        self.assertEqual(config_gen.random_flat.call_count, 2)

        config_gen.flash_structural_parent_coverage_prefix_count = lambda: 0
        search.initial_population = 0
        zero = search._generate_initial_population_flat()
        self.assertEqual(zero, [[0], [3]])

    def test_cute_flash_quick_population_merges_before_limiting(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(5)}

        def canonicalize_flat(flat):
            value = flat[0]
            return [value], configs[value]

        config_gen = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            flash_deterministic_population_configs=lambda: [configs[1], configs[2]],
            flash_exact_effective_search_space_configs=lambda _limit: None,
            flash_structural_population_budget=lambda _target: 2,
            flash_structural_qualification_prefix_count=lambda: 2,
            flatten=lambda config: [config.config["block_sizes"][0] - 1],
            canonicalize_flat=canonicalize_flat,
            random_flat=Mock(side_effect=AssertionError("quick search must not pad")),
        )
        search = PatternSearch.__new__(PatternSearch)
        search.config_gen = config_gen
        search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_BEST_AVAILABLE
        )
        search.best_available_pad_random = False
        search.initial_population = 3
        # Config 1 is both a structural row and a pinned compiler seed. Config 3
        # is another pinned seed; configs 0 and 4 model optional cache rows.
        search._pinned_finalist_configs = {configs[1], configs[3]}
        search._generate_best_available_population_flat = lambda: [
            [1],
            [3],
            [0],
            [4],
        ]

        population = search._generate_initial_population_flat()

        self.assertEqual(population, [[1], [2], [3]])
        self.assertEqual(len(population), len({tuple(flat) for flat in population}))
        config_gen.random_flat.assert_not_called()

    def test_cute_flash_best_available_matches_random_exact_space(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(5)}
        exact_configs = [configs[0], configs[1], configs[2]]

        def unflatten(flat):
            return configs[flat[0]]

        exact_space = Mock(return_value=exact_configs)
        config_gen = ConfigGeneration.__new__(ConfigGeneration)
        config_gen.config_spec = _cute_flash_test_config_spec()
        config_gen._config_value_priors = {}
        config_gen.default_flat = lambda: [0]
        config_gen.validate_flash_structural_coverage = Mock()
        config_gen._flash_deterministic_coverage_flats = lambda: [[0]]
        config_gen.flash_deterministic_population_configs = lambda: [configs[0]]
        config_gen.flash_structural_qualification_prefix_count = lambda: 1
        config_gen.flash_structural_population_budget = lambda _target: 1
        config_gen.flash_exact_effective_search_space_configs = exact_space
        config_gen.unflatten = unflatten
        config_gen.flatten = lambda config: [config.config["block_sizes"][0] - 1]
        config_gen.canonicalize_flat = lambda flat: (
            config_gen.flatten(unflatten(flat)),
            unflatten(flat),
        )
        config_gen.user_seed_flat_config_pairs = lambda _configs, _log: []
        config_gen.seed_flat_config_pairs = lambda _log: []
        config_gen.random_flat = Mock(
            side_effect=AssertionError("an enumerated space must not draw randomly")
        )

        random_search = PatternSearch.__new__(PatternSearch)
        random_search.config_gen = config_gen
        random_search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_RANDOM
        )
        random_search.initial_population = 5
        random_search._autotune_seed_configs = list
        random_search.log = Mock()
        random_population = random_search._generate_initial_population_flat()
        random_unique = [
            list(flat)
            for flat in dict.fromkeys(tuple(flat) for flat in random_population)
        ]

        best_search = PatternSearch.__new__(PatternSearch)
        best_search.config_gen = config_gen
        best_search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_BEST_AVAILABLE
        )
        best_search.best_available_pad_random = True
        best_search.initial_population = 5
        best_search.log = Mock()
        best_search._pinned_finalist_configs = {configs[0]}
        # Config 2 aliases an exact row; config 3 models a legal cache row that
        # is outside the active search values used by the exact enumeration.
        best_search._generate_best_available_population_flat = lambda: [
            [0],
            [2],
            [3],
        ]

        best_population = best_search._generate_initial_population_flat()

        self.assertEqual(random_population, [[0], [1], [2], [0], [1]])
        self.assertEqual(random_unique, [[0], [1], [2]])
        self.assertEqual(best_population, [[0], [1], [2], [3]])
        self.assertEqual(best_population[: len(random_unique)], random_unique)
        self.assertLessEqual(len(best_population), best_search.initial_population)
        self.assertEqual(len(best_population), len({tuple(x) for x in best_population}))
        self.assertEqual(exact_space.call_args_list, [call(5), call(5)])
        config_gen.random_flat.assert_not_called()

    def test_cute_flash_best_available_exact_space_is_mandatory(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(5)}

        def canonicalize_flat(flat):
            value = flat[0] % len(configs)
            return [value], configs[value]

        config_gen = SimpleNamespace(
            config_spec=_cute_flash_test_config_spec(),
            flash_deterministic_population_configs=lambda: [configs[1]],
            flash_exact_effective_search_space_configs=Mock(
                return_value=[configs[1], configs[2]]
            ),
            flash_structural_population_budget=lambda _target: 1,
            flash_structural_qualification_prefix_count=lambda: 1,
            flatten=lambda config: [config.config["block_sizes"][0] - 1],
            canonicalize_flat=canonicalize_flat,
            random_flat=Mock(side_effect=AssertionError("population is complete")),
        )
        search = PatternSearch.__new__(PatternSearch)
        search.config_gen = config_gen
        search.initial_population_strategy = (
            InitialPopulationStrategy.FROM_BEST_AVAILABLE
        )
        search.best_available_pad_random = False
        search.initial_population = 2
        search.log = Mock()
        search._pinned_finalist_configs = {configs[3]}
        # Configs 1 and 2 are optional cache aliases of required rows. Config 4
        # must not displace the exact space after structural and pinned rows.
        search._generate_best_available_population_flat = lambda: [
            [3],
            [1],
            [2],
            [4],
        ]

        population = search._generate_initial_population_flat()

        self.assertEqual(population, [[1], [3], [2]])
        self.assertEqual(len(population), len({tuple(flat) for flat in population}))
        config_gen.flash_exact_effective_search_space_configs.assert_called_once_with(2)
        config_gen.random_flat.assert_not_called()

    def test_cute_flash_population_bounds_compiler_seeds(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(6)}
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        generation._config_value_priors = {}
        generation.default_flat = lambda: [0]
        generation.validate_flash_structural_coverage = Mock()
        generation._flash_deterministic_coverage_flats = lambda: [[0], [1], [2]]
        generation.flash_structural_qualification_prefix_count = lambda: 2
        generation.flash_structural_population_budget = lambda _target: 3
        generation.flash_exact_effective_search_space_configs = lambda _limit: None
        generation.unflatten = lambda flat: configs[flat[0]]
        generation.flatten = lambda config: [config.config["block_sizes"][0] - 1]
        generation.user_seed_flat_config_pairs = lambda _configs, _log: [
            ([3], configs[3]),
            ([1], configs[1]),
        ]
        generation.seed_flat_config_pairs = lambda _log: [([4], configs[4])]
        generation.biased_random_flat = Mock(
            side_effect=AssertionError("no priors are active")
        )
        generation.random_flat = Mock(return_value=[5])

        population = generation.random_population_flat(
            6, user_seed_configs=[configs[3], configs[1]]
        )
        normalized = [generation.unflatten(flat) for flat in population]
        self.assertEqual(population, [[0], [1], [3], [4], [2], [5]])
        self.assertEqual(len(normalized), len(set(normalized)))
        self.assertEqual(len(normalized), 6)
        self.assertLessEqual({configs[0], configs[1], configs[2]}, set(normalized))
        self.assertLessEqual({configs[3], configs[4]}, set(normalized))
        generation.validate_flash_structural_coverage.assert_called_once_with()

        # A target smaller than the mandatory set may grow for explicit user
        # seeds, but compiler hints stay within the requested population size.
        generation.random_flat.reset_mock()
        small = generation.random_population_flat(
            2, user_seed_configs=[configs[3], configs[1]]
        )
        self.assertEqual(small, [[0], [1], [3]])
        generation.random_flat.assert_not_called()
        self.assertEqual(generation.validate_flash_structural_coverage.call_count, 2)

    def test_cute_flash_exact_space_survives_oversized_mandatory_population(self):
        configs = {value: helion.Config(block_sizes=[value + 1]) for value in range(4)}
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        generation._config_value_priors = {}
        generation.default_flat = lambda: [0]
        generation.validate_flash_structural_coverage = Mock()
        generation._flash_deterministic_coverage_flats = lambda: [[0]]
        generation.flash_structural_qualification_prefix_count = lambda: 1
        generation.flash_structural_population_budget = lambda _target: 1
        generation.flash_exact_effective_search_space_configs = lambda _limit: [
            configs[0],
            configs[1],
        ]
        generation.unflatten = lambda flat: configs[flat[0]]
        generation.flatten = lambda config: [config.config["block_sizes"][0] - 1]
        generation.user_seed_flat_config_pairs = lambda _configs, _log: [
            ([3], configs[3])
        ]
        generation.seed_flat_config_pairs = lambda _log: []

        population = generation.random_population_flat(
            2, user_seed_configs=[configs[3]]
        )

        self.assertEqual(population, [[0], [3], [1]])
        self.assertEqual(
            {generation.unflatten(flat) for flat in population},
            {configs[0], configs[1], configs[3]},
        )

    def test_incomplete_cute_flash_structural_design_is_rejected(self):
        generation = ConfigGeneration.__new__(ConfigGeneration)
        generation.config_spec = _cute_flash_test_config_spec()
        generation._override_values = {}
        generation._flash_deterministic_coverage_flats = Mock(return_value=[])
        generation._flash_coverage_uncovered_cache = [("family", "missing")]
        generation._flash_coverage_underqualified_cache = [("packet", "thin", 1)]
        generation._flash_coverage_uncovered_interactions_cache = [
            (("family", "packet"), ("missing", "thin"))
        ]

        with self.assertRaisesRegex(
            exc.AutotuneError,
            "incomplete CuTe flash structural coverage design.*uncovered values.*"
            "uncovered interactions",
        ):
            generation.flash_deterministic_population_configs()

        # An unrelated override must not hide a broken flash covering design.
        generation._override_values = {"num_warps": 4}
        with self.assertRaisesRegex(
            exc.AutotuneError,
            "incomplete CuTe flash structural coverage design",
        ):
            generation.flash_deterministic_population_configs()
        generation._override_values = {"cute_flash_wait_hint": 0}
        with self.assertRaisesRegex(
            exc.AutotuneError,
            "incomplete CuTe flash structural coverage design",
        ):
            generation.flash_deterministic_population_configs()
        generation._override_values = {"cute_flash_exp2_packet": "1x1"}
        with self.assertRaisesRegex(
            exc.AutotuneError,
            "incomplete CuTe flash structural coverage design",
        ):
            generation.flash_deterministic_population_configs()

        # A family may legitimately normalize to one effective config. Keep the
        # two-witness shortfall as telemetry for the strict harness, not a fatal
        # invariant for ordinary tuning.
        generation._flash_coverage_uncovered_cache = []
        generation._flash_coverage_uncovered_interactions_cache = []
        self.assertEqual(generation.flash_deterministic_population_configs(), [])

        # Explicit overrides intentionally narrow the advertised search surface;
        # its unpruned diagnostics must not reject that constrained search.
        generation._override_values = {"cute_flash_pipeline_family": "fa4"}
        generation._flash_coverage_uncovered_cache = [("family", "missing")]
        self.assertEqual(generation.flash_deterministic_population_configs(), [])

    def test_lfbo_flash_path_constraint_filters_before_visited(self):
        packet_key = "cute_flash_exp2_packet"

        def member(flat: int, packet: str) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float(flat)],
                flat_values=[flat, packet],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=flat,
                ),
                status="ok",
            )

        current = member(1, "deg2_16x6")
        rejected = member(2, "1x1")
        retained = member(3, "deg2_16x6")
        unselected = member(4, "deg2_16x6")
        by_flat = {2: rejected, 3: retained, 4: unselected}
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.max_generations = 1
        search.patience = 1
        search.frac_selected = 1.0
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.log = lambda *_args, **_kwargs: None
        search.config_gen = SimpleNamespace(
            _key_to_flat_indices={packet_key: ([1], False)}
        )
        search._generate_neighbors = lambda *_args, **_kwargs: [
            rejected.flat_values,
            retained.flat_values,
            unselected.flat_values,
        ]
        search.make_unbenchmarked = lambda flat: by_flat[flat[0]]
        search._surrogate_select = lambda candidates, _count: candidates[:2]

        visited: set[helion.Config] = set()
        path = search._pruned_pattern_search_from(
            0,
            current,
            visited,
            ((packet_key, "deg2_16x6"),),
        )

        self.assertEqual(next(path), [current, retained])
        self.assertNotIn(rejected.config, visited)
        self.assertIn(retained.config, visited)
        self.assertNotIn(unselected.config, visited)

        selected_counts = []
        search.frac_selected = 0.1
        search._surrogate_select = lambda candidates, count: (
            selected_counts.append(count) or candidates[:count]
        )
        small_path = search._pruned_pattern_search_from(
            0,
            current,
            set(),
            ((packet_key, "deg2_16x6"),),
        )
        self.assertEqual(next(small_path), [current, retained])
        self.assertEqual(selected_counts, [2])

        search._surrogate_select = lambda candidates, count: list(
            reversed(candidates[-count:])
        )
        reordered_visited: set[helion.Config] = set()
        reordered_path = search._pruned_pattern_search_from(
            0,
            current,
            reordered_visited,
            ((packet_key, "deg2_16x6"),),
            selected_limit=2,
        )
        self.assertEqual(next(reordered_path), [current, unselected])
        self.assertIn(current.config, reordered_visited)
        self.assertIn(unselected.config, reordered_visited)
        self.assertNotIn(retained.config, reordered_visited)

        # Unconstrained/non-flash paths preserve historical eager visited
        # bookkeeping, including candidates discarded by the surrogate.
        search.frac_selected = 0.5
        search._surrogate_select = lambda candidates, count: candidates[:count]
        search._generate_neighbors = lambda _base: [
            rejected.flat_values,
            retained.flat_values,
            unselected.flat_values,
        ]
        unconstrained_visited: set[helion.Config] = set()
        unconstrained = search._pruned_pattern_search_from(
            0, current, unconstrained_visited
        )
        self.assertEqual(next(unconstrained), [current, rejected])
        self.assertEqual(
            unconstrained_visited,
            {rejected.config, retained.config, unselected.config},
        )

        # CuTe's unrestricted continuation runs after constrained paths in each
        # generation. Only selected candidates become visited so an unmeasured
        # neighbor remains available to constrained paths and later generations.
        search.max_generations = 2
        search.frac_selected = 0.67
        search._surrogate_select = lambda candidates, count: candidates[:count]
        selected_only_visited: set[helion.Config] = set()
        selected_only = search._pruned_pattern_search_from(
            0,
            current,
            selected_only_visited,
            selected_only_visited=True,
            disable_early_stopping=True,
        )
        self.assertEqual(next(selected_only), [current, rejected])
        self.assertEqual(selected_only_visited, {current.config, rejected.config})
        self.assertEqual(next(selected_only), [current, retained])
        self.assertEqual(
            selected_only_visited,
            {current.config, rejected.config, retained.config},
        )
        self.assertNotIn(unselected.config, selected_only_visited)

    def test_lfbo_ordinary_flash_leaf_rejects_compound_packet(self):
        family_key = "cute_flash_pipeline_family"

        def member(flat: int, packet: str) -> PopulationMember:
            return PopulationMember(
                fn=lambda: None,
                perfs=[float(flat)],
                flat_values=[flat, "fa4", packet],
                config=helion.Config(
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_exp2_packet=packet,
                    cute_flash_wait_hint=flat,
                ),
                status="ok",
            )

        current = member(1, "1x1")
        compound = member(2, "hybrid_deg1_16x8")
        ordinary = member(3, "4x1")
        required_leaf = LFBOPatternSearch._flash_structural_leaf(current)
        assert required_leaf is not None

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.max_generations = 1
        search.patience = 1
        search.frac_selected = 1.0
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.log = lambda *_args, **_kwargs: None
        search.config_gen = SimpleNamespace(
            _key_to_flat_indices={family_key: ([1], False)}
        )
        search._generate_neighbors = lambda *_args, **_kwargs: [
            compound.flat_values,
            ordinary.flat_values,
        ]
        by_flat = {2: compound, 3: ordinary}
        search.make_unbenchmarked = lambda flat: by_flat[flat[0]]
        search._surrogate_select = lambda candidates, count: candidates[:count]

        visited: set[helion.Config] = set()
        path = search._pruned_pattern_search_from(
            0,
            current,
            visited,
            ((family_key, "fa4"),),
            required_leaf=required_leaf,
        )

        self.assertEqual(next(path), [current, ordinary])
        self.assertNotIn(compound.config, visited)
        self.assertIn(ordinary.config, visited)

    def test_lfbo_pattern_search_generate_neighbors(self):
        """Test LFBOPatternSearch._generate_neighbors method."""
        random.seed(123)
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.num_neighbors = 50
        search.radius = 2
        search.config_gen = SimpleNamespace(
            flat_spec=[
                PowerOfTwoFragment(16, 128, 32),  # block_size[0]
                PowerOfTwoFragment(16, 128, 64),  # block_size[1]
                PowerOfTwoFragment(2, 16, 4),  # num_warps
                EnumFragment(("a", "b", "c")),  # some enum
                BooleanFragment(),  # some boolean
            ],
            block_size_indices=[0, 1],
            num_warps_index=2,
            overridden_flat_indices=set(),
            config_spec=SimpleNamespace(tensor_numel_constraints=[]),
        )
        search.num_neighbors_cap = -1

        base = [32, 64, 4, "b", True]
        neighbors = search._generate_neighbors(base)

        # Check we generate the correct number of neighbors
        self.assertEqual(len(neighbors), search.num_neighbors)

        # Check all neighbors are different from base
        for neighbor in neighbors:
            self.assertNotEqual(neighbor, base)

        # Verify all block sizes are valid powers of two in range
        for neighbor in neighbors:
            # Check block_size[0]
            self.assertIn(neighbor[0], [16, 32, 64, 128])
            # Check block_size[1]
            self.assertIn(neighbor[1], [16, 32, 64, 128])
            # Check num_warps
            self.assertIn(neighbor[2], [2, 4, 8, 16])
            # Check enum
            self.assertIn(neighbor[3], ["a", "b", "c"])
            # Check boolean
            self.assertIn(neighbor[4], [True, False])

        random.seed(123)
        unchanged = search._generate_neighbors(base, fixed_flat_values={})
        random.seed(123)
        self.assertEqual(unchanged, search._generate_neighbors(base))

        random.seed(123)
        fixed = search._generate_neighbors(base, fixed_flat_values={3: "b"})
        self.assertTrue(fixed)
        self.assertTrue(all(neighbor[3] == "b" for neighbor in fixed))

    def test_lfbo_tree_search_generate_neighbors_cap(self):
        """LFBOTreeSearch applies num_neighbors_cap in the tree-guided path."""

        class MockEstimator:
            tree_ = SimpleNamespace(feature=np.array([0], dtype=int))

            def decision_path(self, _x):
                return SimpleNamespace(indices=np.array([0], dtype=int))

            def predict_proba(self, x):
                scores = np.asarray(x)[:, 0]
                probas = scores / np.max(scores)
                return np.column_stack((1.0 - probas, probas))

        spec = PowerOfTwoFragment(16, 128, 32)
        search = LFBOTreeSearch.__new__(LFBOTreeSearch)
        search.num_neighbors = 10
        search.num_neighbors_cap = 3
        search.radius = 2
        search.surrogate = SimpleNamespace(estimators_=[MockEstimator()])
        search._autotune_metrics = SimpleNamespace(num_generations=2)
        search._encoded_to_flat_mapping = None
        search.config_gen = SimpleNamespace(
            flat_spec=[spec],
            block_size_indices=[0],
            num_warps_index=-1,
            overridden_flat_indices=set(),
            encode_config=lambda flat: spec.encode(flat[0]),
        )

        capped_neighbors = search._generate_neighbors([32])

        search.num_neighbors_cap = -1
        uncapped_neighbors = search._generate_neighbors([32])

        self.assertEqual(len(capped_neighbors), 3)
        self.assertEqual(len(uncapped_neighbors), 10)
        self.assertTrue(all(neighbor != [32] for neighbor in capped_neighbors))

    def test_lfbo_pattern_search_surrogate_select_matches_legacy_prefix(self):
        """Top-k LFBO selection should match the legacy full-ranking implementation."""

        class MockSurrogate:
            def __init__(
                self, proba_by_id: dict[int, float], leaf_by_id: dict[int, list[int]]
            ) -> None:
                self.proba_by_id = proba_by_id
                self.leaf_by_id = leaf_by_id

            def predict_proba(self, X):
                ids = np.asarray(X)[:, 0].astype(int)
                return np.array(
                    [[1.0 - self.proba_by_id[i], self.proba_by_id[i]] for i in ids]
                )

            def apply(self, X):
                ids = np.asarray(X)[:, 0].astype(int)
                return np.array([self.leaf_by_id[i] for i in ids], dtype=int)

        def legacy_select(
            search: LFBOPatternSearch,
            candidates: list[SimpleNamespace],
            n_sorted: int,
        ) -> list[SimpleNamespace]:
            candidate_X = np.array(
                [
                    search.config_gen.encode_config(member.flat_values)
                    for member in candidates
                ]
            )
            proba = np.asarray(search.surrogate.predict_proba(candidate_X))[:, 1]
            similarity_matrix = search.compute_leaf_similarity(
                search.surrogate, candidate_X
            )
            selected_indices = []
            remaining_indices = list(range(len(candidate_X)))
            scores = np.zeros(len(candidate_X))

            for rank in range(len(candidate_X)):
                if selected_indices:
                    mean_similarities = np.zeros(len(remaining_indices))
                    for i, idx in enumerate(remaining_indices):
                        similarities_to_selected = similarity_matrix[
                            idx, selected_indices
                        ]
                        mean_similarities[i] = np.mean(similarities_to_selected)
                    ranked_scores = (
                        proba[remaining_indices]
                        - search.similarity_penalty * mean_similarities
                    )
                else:
                    ranked_scores = proba[remaining_indices]

                best_local_idx = int(np.argmax(ranked_scores))
                best_global_idx = remaining_indices[best_local_idx]
                scores[best_global_idx] = rank
                selected_indices.append(best_global_idx)
                remaining_indices.remove(best_global_idx)

            ranked = sorted(
                zip(candidates, scores, strict=True),
                key=operator.itemgetter(1),
            )[:n_sorted]
            return [member for member, _ in ranked]

        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_gen = SimpleNamespace(encode_config=lambda flat: [flat[0]])
        search.similarity_penalty = 0.35
        search.log = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
        search.surrogate = MockSurrogate(
            proba_by_id={
                0: 0.95,
                1: 0.92,
                2: 0.90,
                3: 0.86,
                4: 0.84,
                5: 0.83,
            },
            leaf_by_id={
                0: [10, 20, 30, 40],
                1: [10, 20, 31, 41],
                2: [11, 21, 32, 42],
                3: [50, 60, 70, 80],
                4: [50, 61, 71, 81],
                5: [12, 22, 33, 43],
            },
        )
        candidates = [SimpleNamespace(name=f"c{i}", flat_values=[i]) for i in range(6)]

        expected = legacy_select(search, candidates, 3)

        with patch.object(
            search,
            "compute_leaf_similarity",
            side_effect=AssertionError("dense similarity matrix should not be built"),
        ):
            actual = search._surrogate_select(candidates, 3)

        self.assertEqual([c.name for c in actual], [c.name for c in expected])

    def test_tile_strategy_dispatch_compact_shape_uses_cached_block_lookup(self):
        """Fallback block-id lookups should reuse the precomputed strategy cache."""

        class DummyStrategy:
            block_ids: ClassVar[list[int]] = [3, 4]

            def block_size_var(self, block_idx: int) -> str:
                return f"_BLOCK_{block_idx}"

            def compact_shape(self, shapes):
                return shapes

        dispatch = TileStrategyDispatch.__new__(TileStrategyDispatch)
        dispatch.strategies = [DummyStrategy()]
        dispatch.block_id_to_strategy = BlockIDStrategyMapping()
        dispatch.block_id_to_strategy[(3, 4)] = dispatch.strategies[0]

        with patch(
            "helion._compiler.tile_dispatch.CompileEnvironment.current",
            return_value=SimpleNamespace(
                get_block_id=lambda _shape: 3,
                resolve_block_id=lambda _shape: 3,
            ),
        ):
            compacted = dispatch._compact_shape([object()])

        self.assertEqual(len(compacted), 1)
        self.assertEqual(compacted[0].size_str, "_BLOCK_3")
        self.assertEqual(compacted[0].block_ids, [3])

    @skip("too slow")
    def test_lfbo_pattern_search(self):
        args = (
            torch.randn([64, 64], device=DEVICE),
            torch.randn([64, 64], device=DEVICE),
        )
        bound_kernel = basic_kernels.add.bind(args)
        random.seed(123)
        best = LFBOPatternSearch(
            bound_kernel,
            args,
            initial_population=10,
            max_generations=2,
            copies=1,
            num_neighbors=10,
        ).autotune()
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(*args), sum(args), rtol=1e-2, atol=1e-1)

    def _check_accuracy_filters_bad_config(
        self,
        kernel: helion.Kernel,
        bad_config: helion.Config,
        good_config: helion.Config,
        wrap_bad_fn: Callable[[Callable[..., object]], Callable[..., object]],
        *,
        exact_final_failure_count: bool,
        check_spawn: bool = True,
    ) -> None:
        """Shared body for the accuracy-check filtering tests: benchmark a
        known-bad and a known-good config (the bad one's compiled fn is
        wrapped by ``wrap_bad_fn`` to corrupt its behavior) and assert the
        accuracy check rejects only the bad one in fork mode. In spawn mode
        the patched compile result cannot be serialized, so autotuning must
        fail with an error pointing at fork mode."""

        def run_mode(mode: str, *, expect_error: bool) -> None:
            a = torch.randn([32], device=DEVICE)
            b = torch.randn([32], device=DEVICE)
            bound_kernel = kernel.bind((a, b))
            original_compile = bound_kernel.compile_config
            bound_kernel.settings.autotune_precompile = mode

            def wrapping_compile(config: helion.Config, *, allow_print: bool = True):
                fn = original_compile(config, allow_print=allow_print)
                if config == bad_config:
                    return wrap_bad_fn(fn)
                return fn

            import helion.autotuner.base_search as base_search_module

            with patch.object(
                bound_kernel,
                "compile_config",
                side_effect=wrapping_compile,
            ):
                search = FiniteSearch(
                    bound_kernel, (a, b), configs=[bad_config, good_config]
                )
                search._prepare()
                if mode == "fork":
                    start_cm = patch.object(
                        search.benchmark_provider,
                        "_create_precompile_future",
                        side_effect=lambda config, fn: (
                            base_search_module.PrecompileFuture.skip(
                                search.benchmark_provider._precompile_context(),
                                config,
                                True,
                            )
                        ),
                    )
                else:
                    start_cm = nullcontext()

                with start_cm:
                    if expect_error:
                        with self.assertRaisesRegex(
                            helion.exc.AutotuneError,
                            'Set HELION_AUTOTUNE_PRECOMPILE="fork"',
                        ):
                            search.autotune()
                        return

                    bad_time = search.benchmark(bad_config).perf
                    assert math.isinf(bad_time)
                    self.assertEqual(search._autotune_metrics.num_accuracy_failures, 1)
                    search._autotune_metrics.num_accuracy_failures = 0

                    good_time = search.benchmark(good_config).perf
                    assert not math.isinf(good_time)
                    self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)
                    search._autotune_metrics.num_accuracy_failures = 0

                    best = search.autotune()
                    self.assertEqual(best, good_config)
                    if exact_final_failure_count:
                        self.assertEqual(
                            search._autotune_metrics.num_accuracy_failures, 1
                        )
                    else:
                        self.assertGreaterEqual(
                            search._autotune_metrics.num_accuracy_failures, 1
                        )

        run_mode("fork", expect_error=False)
        if check_spawn:
            run_mode("spawn", expect_error=True)

    def test_accuracy_check_filters_bad_config_wrong_output(self) -> None:
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(configs=[bad_config, good_config], autotune_log_level=0)
        def add_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(b.size()):
                b[tile] = a[tile] + b[tile]
            return b

        self._check_accuracy_filters_bad_config(
            add_inplace,
            bad_config,
            good_config,
            lambda fn: lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs) + 1,
            exact_final_failure_count=True,
        )

    def test_autotune_noncontiguous_arg(self) -> None:
        """Autotuning a kernel bound on a non-contiguous arg must succeed.

        A kernel bound on a non-contiguous arg hardcodes that arg's load strides
        into the compiled kernel as compile-time constants. The autotuner reruns
        that compiled kernel on a clone of the args to compute its accuracy
        baseline, so the clone must keep the same layout. If the clone is made
        contiguous, those hardcoded strides address the wrong (or out-of-bounds)
        memory and read garbage, so every config mismatches the baseline and
        autotuning fails to select one.

        The arg is a strided slice (``buf[::2]``): its contiguous clone compacts
        the storage, so the hardcoded stride-2 loads read the wrong elements.
        """
        config_a = helion.Config(block_sizes=[32], num_warps=4)
        config_b = helion.Config(block_sizes=[64], num_warps=4)

        @helion.kernel(configs=[config_a, config_b], autotune_log_level=0)
        def double(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2.0
            return out

        # Strided 1-D view: size 1024, stride 2 over a 2048-element buffer.
        x = torch.arange(2048, device=DEVICE, dtype=torch.float32)[::2]
        self.assertFalse(x.is_contiguous())
        self.assertEqual(x.stride(), (2,))

        bound_kernel = double.bind((x,))
        search = FiniteSearch(bound_kernel, (x,), configs=[config_a, config_b])
        best = search.autotune()
        self.assertIn(best, (config_a, config_b))

        # Compile the winning config directly; calling double(x) would rerun
        # the whole FiniteSearch since the manual search above is not attached
        # to the bound kernel.
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(x), x * 2.0)

    def test_autotune_broadcast_arg(self) -> None:
        """Autotuning a kernel bound on a broadcast/expanded arg must succeed.

        Same accuracy-baseline issue as the strided-slice case, but for a
        ``stride-0`` view from ``expand``. Here ``b`` is ``[M, 1]`` expanded to
        ``[M, N]`` (stride ``(1, 0)``); the kernel indexes ``b[ti, tj]`` so the
        stride-0 column load is hardcoded into the compiled kernel. A clone that
        materialized ``b`` to a contiguous ``[M, N]`` (stride ``(N, 1)``) would
        make that hardcoded load read a different element per column, so the
        baseline diverges and every config is rejected.
        """
        config_a = helion.Config(block_sizes=[32, 32], num_warps=4)
        config_b = helion.Config(block_sizes=[64, 64], num_warps=4)

        @helion.kernel(configs=[config_a, config_b], autotune_log_level=0)
        def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_i, tile_j in hl.tile(x.size()):
                out[tile_i, tile_j] = x[tile_i, tile_j] + b[tile_i, tile_j]
            return out

        m, n = 128, 128
        x = torch.randn(m, n, device=DEVICE)
        # Broadcast/expanded view: [m, 1] -> [m, n], stride (1, 0).
        b = torch.randn(m, 1, device=DEVICE).expand(m, n)
        self.assertFalse(b.is_contiguous())
        self.assertEqual(b.stride(), (1, 0))

        bound_kernel = add_bias.bind((x, b))
        search = FiniteSearch(bound_kernel, (x, b), configs=[config_a, config_b])
        best = search.autotune()
        self.assertIn(best, (config_a, config_b))

        # Compile the winning config directly; calling add_bias(x, b) would
        # rerun the whole FiniteSearch since the manual search above is not
        # attached to the bound kernel.
        fn = bound_kernel.compile_config(best)
        torch.testing.assert_close(fn(x, b), x + b)

    def test_accuracy_check_filters_bad_config_wrong_arg_mutation(self) -> None:
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(configs=[bad_config, good_config], autotune_log_level=0)
        def add_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(b.size()):
                b[tile] = a[tile] + b[tile]
            return b

        def wrap_bad_fn(fn):
            def wrong_fn(*fn_args, **fn_kwargs):
                result = fn(*fn_args, **fn_kwargs)
                # Introduce an extra mutation so inputs differ from baseline
                fn_args[1].add_(1)
                return result

            return wrong_fn

        self._check_accuracy_filters_bad_config(
            add_inplace,
            bad_config,
            good_config,
            wrap_bad_fn,
            exact_final_failure_count=False,
        )

    def test_autotune_baseline_fn(self) -> None:
        """Test that custom baseline function is used for accuracy checking."""
        config1 = helion.Config(block_sizes=[32], num_warps=4)
        config2 = helion.Config(block_sizes=[64], num_warps=8)

        # Track whether the baseline function was called
        baseline_calls = []

        def custom_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            baseline_calls.append(True)
            # Return the expected result using PyTorch operations
            return a + b

        # The custom baseline fn is invoked on the parent-side baseline path;
        # skip the benchmark worker subprocess (seconds of startup overhead).
        @helion.kernel(
            configs=[config1, config2],
            autotune_baseline_fn=custom_baseline,
            autotune_log_level=0,
            autotune_benchmark_subprocess=False,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )

        # Run autotuning
        result = add(*args)

        # Verify the custom baseline function was called during autotuning
        self.assertGreater(
            len(baseline_calls), 0, "Custom baseline function should be called"
        )

        # Verify the result is correct
        torch.testing.assert_close(result, args[0] + args[1])

    def test_autotune_baseline_fn_filters_bad_config(self) -> None:
        """Test that custom baseline function correctly filters incorrect configs."""
        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        def custom_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
            # Return the correct expected result
            return a + b

        # The custom baseline fn is invoked on the parent-side baseline path;
        # skip the benchmark worker subprocess (seconds of startup overhead).
        @helion.kernel(
            configs=[bad_config, good_config],
            autotune_baseline_fn=custom_baseline,
            autotune_log_level=0,
            autotune_benchmark_subprocess=False,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        self._check_accuracy_filters_bad_config(
            add,
            bad_config,
            good_config,
            lambda fn: lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs) + 1,
            exact_final_failure_count=True,
            check_spawn=False,
        )

    def test_autotune_baseline_fn_raises_on_failure(self) -> None:
        """Test that AutotuneError is raised when custom baseline function fails."""
        config1 = helion.Config(block_sizes=[32], num_warps=4)
        config2 = helion.Config(block_sizes=[64], num_warps=8)

        def failing_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("Baseline computation failed!")

        @helion.kernel(
            configs=[config1, config2],
            autotune_baseline_fn=failing_baseline,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )

        # Attempting to run should raise AutotuneError
        with self.assertRaisesRegex(
            helion.exc.AutotuneError,
            "Custom baseline function failed while computing baseline",
        ):
            add(*args)

    def test_autotune_baseline_tolerance(self) -> None:
        cfg1 = helion.Config(block_sizes=[1], num_warps=4)
        cfg2 = helion.Config(block_sizes=[1], num_warps=8)
        a, b = torch.randn([32], device=DEVICE), torch.randn([32], device=DEVICE)

        # Baseline that returns slightly incorrect result (1e-4 error)
        def incorrect_baseline(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b + 1e-4

        # Test both strict (1e-5) and lenient (1e-3) tolerances
        for tol, expect_reject in [(1e-5, True), (1e-3, False)]:
            # The tolerance plumbing under test is shared with the in-process
            # accuracy path; skip the benchmark worker subprocess (seconds of
            # startup overhead per kernel).
            @helion.kernel(
                configs=[cfg1, cfg2],
                autotune_baseline_fn=incorrect_baseline,
                autotune_baseline_atol=tol,
                autotune_baseline_rtol=tol,
                autotune_benchmark_subprocess=False,
            )
            def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                o = torch.empty_like(a)
                for t in hl.tile(o.size()):
                    o[t] = a[t] + b[t]
                return o

            bound = add.bind((a, b))
            search = FiniteSearch(bound, (a, b), configs=[cfg1, cfg2])

            if expect_reject:
                # FiniteSearch currently raises AssertionError if every config fails validation
                with self.assertRaises(AssertionError):
                    search.autotune()
                # All configs should have tripped the accuracy mismatch counter
                self.assertEqual(
                    search._autotune_metrics.num_accuracy_failures, len(search.configs)
                )
            else:
                winner = search.autotune()
                self.assertIn(winner, (cfg1, cfg2))
                self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_autotune_fp8_automatic_tolerance(self) -> None:
        """Test that fp8 dtypes automatically get 0.0 tolerances."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=8)

        # Test with float8_e4m3fn as a representative fp8 dtype. The automatic
        # tolerance selection under test is parent-side; skip the benchmark
        # worker subprocess (seconds of startup overhead).
        @helion.kernel(configs=[cfg1, cfg2], autotune_benchmark_subprocess=False)
        def cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.size(), dtype=torch.float8_e4m3fn, device=x.device)
            for t in hl.tile(x.size()):
                out[t] = x[t].to(torch.float8_e4m3fn)
            return out

        x = torch.randn([64], device=DEVICE)
        bound = cast_to_fp8.bind((x,))
        search = FiniteSearch(bound, (x,), configs=[cfg1, cfg2])
        search._prepare()

        # Verify that effective tolerances were set to 0.0 automatically
        self.assertEqual(
            search.benchmark_provider._effective_atol,
            0.0,
            f"Expected automatic atol=0.0 for fp8, got {search.benchmark_provider._effective_atol}",
        )
        self.assertEqual(
            search.benchmark_provider._effective_rtol,
            0.0,
            f"Expected automatic rtol=0.0 for fp8, got {search.benchmark_provider._effective_rtol}",
        )

        # Should successfully autotune without error
        winner = search.autotune()
        self.assertIn(winner, (cfg1, cfg2))
        self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_autotune_fp8_explicit_tolerance_override(self) -> None:
        """Test that explicit tolerances override automatic fp8 detection."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=8)

        # User explicitly sets non-zero tolerances despite fp8 output
        @helion.kernel(
            configs=[cfg1, cfg2],
            autotune_baseline_atol=1e-5,
            autotune_baseline_rtol=1e-5,
        )
        def cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty(x.size(), dtype=torch.float8_e4m3fn, device=x.device)
            for t in hl.tile(x.size()):
                out[t] = x[t].to(torch.float8_e4m3fn)
            return out

        x = torch.randn([64], device=DEVICE)
        bound = cast_to_fp8.bind((x,))
        search = FiniteSearch(bound, (x,), configs=[cfg1, cfg2])
        search._prepare()

        # Should respect user's explicit tolerances, not override to 0.0
        self.assertEqual(search.benchmark_provider._effective_atol, 1e-5)
        self.assertEqual(search.benchmark_provider._effective_rtol, 1e-5)

    @skipIfCudaCapabilityLessThan((9, 0), reason="FP8 requires CUDA capability >= 9.0")
    def test_autotune_mixed_fp8_and_fp32_output(self) -> None:
        """Test that the accuracy check works with mixed fp8+fp32 outputs."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=8)

        # The fp8 handling under test lives in the shared accuracy.assert_close;
        # skip the benchmark worker subprocess (seconds of startup overhead).
        @helion.kernel(configs=[cfg1, cfg2], autotune_benchmark_subprocess=False)
        def mixed_output(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            fp8_out = torch.empty(x.size(), dtype=torch.float8_e4m3fn, device=x.device)
            fp32_out = torch.empty(x.size(), dtype=torch.float32, device=x.device)
            for t in hl.tile(x.size()):
                fp8_out[t] = x[t].to(torch.float8_e4m3fn)
                fp32_out[t] = x[t] * 2.0
            return fp8_out, fp32_out

        x = torch.randn([64], device=DEVICE)
        bound = mixed_output.bind((x,))
        search = FiniteSearch(bound, (x,), configs=[cfg1, cfg2])

        # Should successfully autotune without error
        winner = search.autotune()
        self.assertIn(winner, (cfg1, cfg2))
        self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

    def test_max_generations(self):
        """Autotuner max generation respects explicit kwargs then setting override."""

        with patch.dict(os.environ, {"HELION_AUTOTUNER": "PatternSearch"}):

            @helion.kernel(autotune_max_generations=1)
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            args = (
                torch.randn([8], device=DEVICE),
                torch.randn([8], device=DEVICE),
            )

            bound = add.bind(args)
            autotuner_factory = bound.settings.autotuner_fn

            # Settings override defaults
            autotuner = autotuner_factory(bound, args)
            self.assertEqual(autotuner.autotuner.max_generations, 1)

            # Explicit constructor value wins
            autotuner_override = autotuner_factory(bound, args, max_generations=2)
            self.assertEqual(autotuner_override.autotuner.max_generations, 2)

    def test_autotune_effort_none(self):
        @helion.kernel(autotune_effort="none")
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        result = add(*args)
        torch.testing.assert_close(result, sum(args))

    def test_env_autotune_effort_none_ignores_force_autotune(self):
        autotuner_fn = Mock(
            side_effect=AssertionError("autotuner should not run with effort=none")
        )

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNE_EFFORT": "none",
                "HELION_FORCE_AUTOTUNE": "1",
            },
        ):

            @helion.kernel(autotuner_fn=autotuner_fn)
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

        args = (
            torch.randn([8], device=DEVICE),
            torch.randn([8], device=DEVICE),
        )
        result = add(*args)
        torch.testing.assert_close(result, sum(args))
        autotuner_fn.assert_not_called()

    def test_autotune_effort_quick(self):
        """Test that quick effort profile uses correct default values."""
        # Get the quick profile defaults
        quick_profile = get_effort_profile("quick")
        assert quick_profile.lfbo_pattern_search is not None
        self.assertIsNone(quick_profile.flash_structural_search)
        expected_initial_pop = quick_profile.lfbo_pattern_search.initial_population
        expected_copies = quick_profile.lfbo_pattern_search.copies
        expected_max_gen = quick_profile.lfbo_pattern_search.max_generations

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )

        # Test 1: Default quick mode values from effort profile (LFBOTreeSearch is default)
        with patch.dict(os.environ, {"HELION_AUTOTUNER": "LFBOTreeSearch"}):

            @helion.kernel(autotune_effort="quick")
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            bound = add.bind(args)
            autotuner = bound.settings.autotuner_fn(bound, args)
            lfbo_tree = autotuner.autotuner
            self.assertIsInstance(lfbo_tree, LFBOTreeSearch)
            # Use exact values from quick profile
            self.assertEqual(lfbo_tree.initial_population, expected_initial_pop)
            self.assertEqual(lfbo_tree.copies, expected_copies)
            self.assertEqual(lfbo_tree.max_generations, expected_max_gen)
            self.assertIsNone(lfbo_tree.flash_structural_search)

        # Test 2: HELION_AUTOTUNE_MAX_GENERATIONS overrides effort profile
        override_max_gen = 100
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "LFBOTreeSearch",
                "HELION_AUTOTUNE_MAX_GENERATIONS": str(override_max_gen),
            },
        ):

            @helion.kernel(autotune_effort="quick")
            def add_with_override(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            bound = add_with_override.bind(args)
            autotuner = bound.settings.autotuner_fn(bound, args)
            lfbo_tree = autotuner.autotuner
            self.assertIsInstance(lfbo_tree, LFBOTreeSearch)
            # initial_population and copies from profile, but max_generations from env var
            self.assertEqual(lfbo_tree.initial_population, expected_initial_pop)
            self.assertEqual(lfbo_tree.copies, expected_copies)
            self.assertEqual(lfbo_tree.max_generations, override_max_gen)

        # Test 3: Explicit constructor values take highest priority
        explicit_initial_pop = 500
        explicit_copies = 300
        explicit_max_gen = 150

        bound = add.bind(args)
        lfbo_tree = LFBOTreeSearch(
            bound,
            args,
            initial_population=explicit_initial_pop,
            copies=explicit_copies,
            max_generations=explicit_max_gen,
        )
        # All values from explicit constructor args
        self.assertEqual(lfbo_tree.initial_population, explicit_initial_pop)
        self.assertEqual(lfbo_tree.copies, explicit_copies)
        self.assertEqual(lfbo_tree.max_generations, explicit_max_gen)
        self.assertIsNone(lfbo_tree.flash_structural_search)

    def test_finishing_rounds(self):
        """finishing_rounds comes from profile, env var overrides, explicit ctor arg wins."""
        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )

        @helion.kernel(autotune_effort="quick")
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        bound = add.bind(args)
        quick_profile = get_effort_profile("quick")

        # Default: comes from effort profile
        with patch.dict(os.environ, {"HELION_AUTOTUNER": "PatternSearch"}):
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertEqual(
                autotuner.autotuner.finishing_rounds, quick_profile.finishing_rounds
            )

        # Env var overrides effort profile
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "PatternSearch",
                "HELION_AUTOTUNE_FINISHING_ROUNDS": "7",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertEqual(autotuner.autotuner.finishing_rounds, 7)

        # Explicit constructor arg wins over env var
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "PatternSearch",
                "HELION_AUTOTUNE_FINISHING_ROUNDS": "7",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args, finishing_rounds=3)
            self.assertEqual(autotuner.autotuner.finishing_rounds, 3)

    def test_num_neighbors_cap(self):
        """num_neighbors_cap defaults to -1, env var overrides, explicit ctor arg wins."""
        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )

        @helion.kernel()
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        bound = add.bind(args)

        # Default: -1 (no cap)
        with (
            without_env_var("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS"),
            patch.dict(os.environ, {"HELION_AUTOTUNER": "PatternSearch"}),
        ):
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertEqual(autotuner.autotuner.num_neighbors_cap, -1)

        # Env var overrides default
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "PatternSearch",
                "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "50",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertEqual(autotuner.autotuner.num_neighbors_cap, 50)

        # Env var also applies to LFBO-based pattern search.
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "LFBOTreeSearch",
                "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "50",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertEqual(autotuner.autotuner.num_neighbors_cap, 50)

        # Explicit constructor arg wins over env var
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "PatternSearch",
                "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "50",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args, num_neighbors_cap=10)
            self.assertEqual(autotuner.autotuner.num_neighbors_cap, 10)

        # Explicit constructor arg wins over env var for LFBO-based search too.
        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "LFBOTreeSearch",
                "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "50",
            },
        ):
            autotuner = bound.settings.autotuner_fn(bound, args, num_neighbors_cap=10)
            self.assertEqual(autotuner.autotuner.num_neighbors_cap, 10)

    def test_autotuner_disabled(self):
        @helion.kernel()
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 512, 512], device=DEVICE),
            torch.randn([8, 512, 512], device=DEVICE),
        )
        with (
            patch.dict(os.environ, {"HELION_DISALLOW_AUTOTUNING": "1"}),
            pytest.raises(
                expected_exception=helion.exc.AutotuningDisallowedInEnvironment,
                match="Autotuning is disabled by HELION_DISALLOW_AUTOTUNING=1, please provide a config to @helion.kernel via the config= argument.",
            ),
        ):
            add(*args)

    def test_fragment_encoding(self):
        """Test encoding functionality for all ConfigSpecFragment types."""
        # Test BooleanFragment
        bool_frag = BooleanFragment()
        self.assertEqual(bool_frag.dim(), 1)
        self.assertEqual(bool_frag.encode(True), [1.0])
        self.assertEqual(bool_frag.encode(False), [0.0])

        # Test IntegerFragment
        int_frag = IntegerFragment(low=1, high=10, default_val=5)
        self.assertEqual(int_frag.dim(), 1)
        self.assertEqual(int_frag.encode(5), [5.0])

        # Test PowerOfTwoFragment (log2 transformation)
        pow2_frag = PowerOfTwoFragment(low=2, high=128, default_val=8)
        self.assertEqual(pow2_frag.dim(), 1)
        self.assertEqual(pow2_frag.encode(8), [3.0])  # log2(8) = 3
        self.assertEqual(pow2_frag.encode(16), [4.0])  # log2(16) = 4

        # Test NumThreadsFragment (0 is the CuTe auto-thread sentinel)
        num_threads_frag = NumThreadsFragment(high=128)
        self.assertEqual(num_threads_frag.dim(), 1)
        self.assertEqual(num_threads_frag.encode(0), [0.0])
        self.assertEqual(num_threads_frag.encode(8), [4.0])

        # Test EnumFragment (one-hot encoding)
        enum_frag = EnumFragment(choices=("a", "b", "c"))
        self.assertEqual(enum_frag.dim(), 3)
        self.assertEqual(enum_frag.encode("a"), [1.0, 0.0, 0.0])
        self.assertEqual(enum_frag.encode("b"), [0.0, 1.0, 0.0])

        # Test PermutationFragment
        perm_frag = PermutationFragment(length=3)
        self.assertEqual(perm_frag.dim(), 3)
        encoded = perm_frag.encode([0, 1, 2])
        self.assertEqual(encoded, [0.0, 1.0, 2.0])

        # Test ListOf with BooleanFragment
        list_frag = ListOf(inner=BooleanFragment(), length=3)
        self.assertEqual(list_frag.dim(), 3)
        self.assertEqual(list_frag.encode([True, False, True]), [1.0, 0.0, 1.0])

        # Test encode_dim consistency
        for fragment, value in [
            (BooleanFragment(), True),
            (IntegerFragment(1, 10, 5), 5),
            (PowerOfTwoFragment(2, 128, 8), 16),
            (NumThreadsFragment(128), 0),
            (EnumFragment(choices=("a", "b")), "b"),
        ]:
            dim = fragment.dim()
            encoded = fragment.encode(value)
            self.assertEqual(len(encoded), dim)

    def test_block_size_fragment_autotuner_min_clamp(self):
        """random_config() must not crash when autotuner_min > max_size."""
        from examples.attention import attention

        q, k, v = [
            torch.randn(4, 48, 128, 128, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(3)
        ]
        bound = attention.bind((q, k, v))
        config_spec = bound.config_spec
        config_spec.raise_grid_block_minimums()
        gen = ConfigGeneration(config_spec)
        config = gen.random_config()
        self.assertEqual(config["block_sizes"][0], 1)

    def test_autotune_benchmark_fn(self) -> None:
        """Test that custom benchmark function is used during rebenchmarking."""
        # Track benchmark function calls
        benchmark_calls: list[tuple[int, int]] = []  # (num_fns, repeat)

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            benchmark_calls.append((len(fns), repeat))
            # Return fake timings
            return [1.0] * len(fns)

        @helion.kernel(
            autotune_benchmark_fn=custom_benchmark_fn,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )

        bound_kernel = add.bind(args)
        # Use PatternSearch which has rebenchmark method
        search = PatternSearch(bound_kernel, args)

        # Compile two configs
        config1 = search.config_gen.random_config()
        config2 = search.config_gen.random_config()
        fn1 = bound_kernel.compile_config(config1)
        fn2 = bound_kernel.compile_config(config2)

        # Create population members (flat_values not used in rebenchmark)
        member1 = PopulationMember(fn1, [1.0], (), config1)
        member2 = PopulationMember(fn2, [1.1], (), config2)

        search._prepare()
        search.best_perf_so_far = 1.0

        # Call rebenchmark directly
        search.rebenchmark([member1, member2])

        # Verify custom benchmark function was called
        self.assertGreater(
            len(benchmark_calls), 0, "Custom benchmark function should be called"
        )
        # Should have been called with 2 functions
        self.assertEqual(benchmark_calls[0][0], 2)

    def test_rebenchmark_clears_jit_fast_path_caches(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.config_spec = SimpleNamespace(backend=None)
        events: list[str] = []

        class FakeJITFunction:
            def __init__(self, name: str) -> None:
                self.name = name

            def clear_fast_path_caches(self) -> None:
                events.append(f"clear {self.name}")

        def generated_kernel_a() -> None:
            events.append("run a")

        def generated_kernel_b() -> None:
            events.append("run b")

        globals_key_a = f"_helion_{generated_kernel_a.__name__}"
        globals_key_b = f"_helion_{generated_kernel_b.__name__}"
        generated_kernel_a.__globals__[globals_key_a] = FakeJITFunction("a")
        generated_kernel_b.__globals__[globals_key_b] = FakeJITFunction("b")

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            for _ in range(2):
                for fn in fns:
                    fn()
            return [100.0, 101.0]

        search.settings.autotune_benchmark_fn = custom_benchmark_fn
        member_a = PopulationMember(generated_kernel_a, [100.0], [], helion.Config())
        member_b = PopulationMember(generated_kernel_b, [101.0], [], helion.Config())
        try:
            search.rebenchmark([member_a, member_b])
        finally:
            del generated_kernel_a.__globals__[globals_key_a]
            del generated_kernel_b.__globals__[globals_key_b]

        self.assertEqual(
            events,
            [
                "run a",
                "clear a",
                "run b",
                "clear b",
                "run a",
                "clear a",
                "run b",
                "clear b",
                "clear a",
                "clear b",
            ],
        )

    def test_final_rebenchmark_can_restore_earlier_stable_best(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0

        stable_config = helion.Config(num_warps=4)
        noisy_config = helion.Config(num_warps=8)
        stable = PopulationMember(lambda: None, [10.0], (), stable_config, status="ok")
        noisy = PopulationMember(lambda: None, [9.0], (), noisy_config, status="ok")
        search._benchmarked_members = {
            stable_config: stable,
            noisy_config: noisy,
        }
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [noisy]

        def fake_rebenchmark(
            members: list[PopulationMember],
            *,
            desc: str = "Rebenchmarking",
            target_ms: float = 200.0,
            use_isolated: bool = True,
            confirm_suspicious: bool = True,
            use_interleaved: bool = True,
        ) -> None:
            self.assertIn("Final verification", desc)
            self.assertEqual(target_ms, 5000.0)
            self.assertFalse(use_isolated)
            self.assertFalse(confirm_suspicious)
            self.assertFalse(use_interleaved)
            for member in members:
                member.perfs.append(10.0 if member is stable else 12.0)

        with (
            clean_final_rebenchmark_env(),
            patch.object(search, "rebenchmark", side_effect=fake_rebenchmark),
        ):
            self.assertIs(search.final_rebenchmark_best(noisy), stable)

    def test_final_rebenchmark_rejects_all_failed_finalists(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0
        search.config_spec = SimpleNamespace(backend_name="cute")

        config_a = helion.Config(num_warps=4)
        config_b = helion.Config(num_warps=8)
        member_a = PopulationMember(lambda: None, [10.0], (), config_a, status="ok")
        member_b = PopulationMember(lambda: None, [9.0], (), config_b, status="ok")
        search._benchmarked_members = {
            config_a: member_a,
            config_b: member_b,
        }
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [member_b]

        def fail_rebenchmark(members, **_kwargs):
            for member in members:
                member.perfs[:] = [float("inf")]
                member.status = "timeout"

        with (
            clean_final_rebenchmark_env(),
            patch.object(search, "rebenchmark", side_effect=fail_rebenchmark),
            self.assertRaises(exc.NoConfigFound),
        ):
            search.final_rebenchmark_best(member_b)

    def test_final_rebenchmark_replaces_invalid_best_with_only_live_candidate(
        self,
    ) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 10.0
        search.config_spec = SimpleNamespace(backend_name="cute")

        invalid_config = helion.Config(num_warps=8)
        live_config = helion.Config(num_warps=4)
        invalid = PopulationMember(
            lambda: None,
            [float("inf")],
            (),
            invalid_config,
            status="timeout",
        )
        live = PopulationMember(lambda: None, [10.0], (), live_config, status="ok")
        search._benchmarked_members = {live_config: live}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [invalid]

        with (
            clean_final_rebenchmark_env(),
            patch.object(
                search,
                "_autotune_budget_exceeded_across_ranks",
                return_value=False,
            ) as budget_exceeded,
            patch.object(search, "rebenchmark") as rebenchmark,
        ):
            self.assertIs(search.final_rebenchmark_best(invalid), live)

        budget_exceeded.assert_called_once_with()
        rebenchmark.assert_not_called()

    def test_final_rebenchmark_target_ms_env_is_bounded(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)

        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="50"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 200.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="1500"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 1500.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="1e308"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 60000.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="inf"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 5000.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS="nan"
        ):
            self.assertEqual(search._final_rebenchmark_target_ms(), 5000.0)

    def test_final_rebenchmark_pinned_tolerance_env(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)

        with clean_final_rebenchmark_env():
            self.assertEqual(search._final_rebenchmark_pinned_tolerance(), 0.0)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE="0.02"
        ):
            self.assertEqual(search._final_rebenchmark_pinned_tolerance(), 0.02)
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE="-1"
        ):
            self.assertEqual(search._final_rebenchmark_pinned_tolerance(), 0.0)

    def test_final_rebenchmark_top_k_default_scoped_to_cute(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)

        with clean_final_rebenchmark_env():
            # The larger finalist set is scoped to cute; other backends use the
            # cheaper default so autotune wall-time is not inflated.
            search.config_spec = SimpleNamespace(backend_name="cute")
            self.assertEqual(search._final_rebenchmark_top_k(), 32)
            search.config_spec = SimpleNamespace(backend_name="triton")
            self.assertEqual(search._final_rebenchmark_top_k(), 8)
            search.config_spec = SimpleNamespace(backend_name="pallas")
            self.assertEqual(search._final_rebenchmark_top_k(), 8)

        # An explicit env override wins regardless of backend.
        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search.config_spec = SimpleNamespace(backend_name="cute")
            self.assertEqual(search._final_rebenchmark_top_k(), 2)

    def test_final_rebenchmark_prefers_faster_generated_config_by_default(
        self,
    ) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0

        pinned_config = helion.Config(num_warps=4)
        noisy_config = helion.Config(num_warps=8)
        pinned = PopulationMember(lambda: None, [10.0], (), pinned_config, status="ok")
        noisy = PopulationMember(lambda: None, [9.0], (), noisy_config, status="ok")
        search._benchmarked_members = {
            pinned_config: pinned,
            noisy_config: noisy,
        }
        search._pinned_finalist_configs = {pinned_config}
        search._pinned_finalist_members = {pinned_config: pinned}
        search.population = [noisy]

        def fake_rebenchmark(
            members: list[PopulationMember],
            *,
            desc: str = "Rebenchmarking",
            target_ms: float = 200.0,
            use_isolated: bool = True,
            confirm_suspicious: bool = True,
            use_interleaved: bool = True,
        ) -> None:
            for member in members:
                member.perfs.append(10.04 if member is pinned else 10.0)

        with (
            clean_final_rebenchmark_env(),
            patch.object(search, "rebenchmark", side_effect=fake_rebenchmark),
        ):
            self.assertIs(search.final_rebenchmark_best(noisy), noisy)

    def test_final_rebenchmark_can_prefer_near_tied_pinned_config(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0

        pinned_config = helion.Config(num_warps=4)
        noisy_config = helion.Config(num_warps=8)
        pinned = PopulationMember(lambda: None, [10.0], (), pinned_config, status="ok")
        noisy = PopulationMember(lambda: None, [9.0], (), noisy_config, status="ok")
        search._benchmarked_members = {
            pinned_config: pinned,
            noisy_config: noisy,
        }
        search._pinned_finalist_configs = {pinned_config}
        search._pinned_finalist_members = {pinned_config: pinned}
        search.population = [noisy]

        def fake_rebenchmark(
            members: list[PopulationMember],
            *,
            desc: str = "Rebenchmarking",
            target_ms: float = 200.0,
            use_isolated: bool = True,
            confirm_suspicious: bool = True,
            use_interleaved: bool = True,
        ) -> None:
            for member in members:
                member.perfs.append(10.04 if member is pinned else 10.0)

        with (
            clean_final_rebenchmark_env(
                HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE="0.005"
            ),
            patch.object(search, "rebenchmark", side_effect=fake_rebenchmark),
        ):
            self.assertIs(search.final_rebenchmark_best(noisy), pinned)

    def test_final_rebenchmark_isolated_env(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)

        with clean_final_rebenchmark_env():
            self.assertFalse(search._final_rebenchmark_use_isolated())
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED="1"
        ):
            self.assertTrue(search._final_rebenchmark_use_isolated())
        with clean_final_rebenchmark_env(
            HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED="maybe"
        ):
            self.assertFalse(search._final_rebenchmark_use_isolated())

    def test_final_rebenchmark_isolated_env_keeps_suspicious_confirmation(
        self,
    ) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0

        config_a = helion.Config(num_warps=4)
        config_b = helion.Config(num_warps=8)
        member_a = PopulationMember(lambda: None, [10.0], (), config_a, status="ok")
        member_b = PopulationMember(lambda: None, [9.0], (), config_b, status="ok")
        search._benchmarked_members = {
            config_a: member_a,
            config_b: member_b,
        }
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [member_b]

        def fake_rebenchmark(
            members: list[PopulationMember],
            *,
            desc: str = "Rebenchmarking",
            target_ms: float = 200.0,
            use_isolated: bool = True,
            confirm_suspicious: bool = True,
            use_interleaved: bool = True,
        ) -> None:
            self.assertIn("Final verification", desc)
            self.assertEqual(target_ms, 5000.0)
            self.assertTrue(use_isolated)
            self.assertTrue(confirm_suspicious)
            self.assertFalse(use_interleaved)
            for member in members:
                member.perfs.append(9.5 if member is member_a else 10.0)

        with (
            clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED="1"),
            patch.object(search, "rebenchmark", side_effect=fake_rebenchmark),
        ):
            self.assertIs(search.final_rebenchmark_best(member_b), member_a)

    def test_rebenchmark_repeat_for_target_ms(self) -> None:
        self.assertEqual(PopulationBasedSearch._repeat_for_target_ms(200.0, 100.0), 3)
        self.assertEqual(
            PopulationBasedSearch._repeat_for_target_ms(1000.0, 0.05), 20000
        )
        self.assertEqual(
            PopulationBasedSearch._repeat_for_target_ms(1e308, 1e-308), 20_000
        )

    def test_mirrored_rebenchmark_attests_batched_call_sizing(self) -> None:
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_progress_bar=False,
        )
        search.args = ()
        search.log = AutotuningLogger(search.settings)
        search.best_perf_so_far = 20.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search._apply_rebenchmark_timings = Mock()
        members = [
            PopulationMember(lambda: None, [40.0], (), helion.Config(block_sizes=[1])),
            PopulationMember(lambda: None, [20.0], (), helion.Config(block_sizes=[2])),
        ]
        repeats: list[int] = []

        def mirrored_bench(_fns, *, repeat, desc=None, after_call=None):
            repeats.append(repeat)
            self.assertIsNone(desc)
            self.assertIsNotNone(after_call)
            sweep_count, calls_per_sample, total_calls = _mirrored_bench_call_layout(
                repeat
            )
            return MirroredBenchmarkTrace(
                orders=[[0, 1], [1, 0]] * (sweep_count // 2),
                elapsed_ms=[[40.0, 20.0], [20.0, 40.0]] * (sweep_count // 2),
                medians_ms=[40.0, 20.0],
                sweep_count=sweep_count,
                calls_per_sample=calls_per_sample,
                total_calls=total_calls,
            )

        with (
            clean_final_rebenchmark_env(),
            patch(
                "helion.autotuner.base_search.mirrored_bench_generic",
                side_effect=mirrored_bench,
            ),
            patch(
                "helion.autotuner.base_search.sync_object",
                side_effect=lambda value, **_kwargs: value,
            ),
            patch("helion.autotuner.base_search.clear_jit_fast_path_caches"),
        ):
            trace = search.mirrored_rebenchmark(
                members,
                desc="Terminal coordinate refinement",
                target_ms=200.0,
            )
            confirmation_trace = search.mirrored_rebenchmark(
                members,
                desc="Terminal coordinate refinement: confirming",
                target_ms=5000.0,
            )

        self.assertEqual(repeats, [6, 126])
        self.assertEqual(trace.target_ms, 200.0)
        self.assertEqual(trace.repeat_reference_perf_ms, 40.0)
        self.assertEqual(trace.sweep_count, 6)
        self.assertEqual(trace.calls_per_sample, 1)
        self.assertEqual(trace.total_calls, 6)
        self.assertEqual(confirmation_trace.target_ms, 5000.0)
        self.assertEqual(confirmation_trace.repeat_reference_perf_ms, 40.0)
        self.assertEqual(confirmation_trace.sweep_count, 64)
        self.assertEqual(confirmation_trace.calls_per_sample, 2)
        self.assertEqual(confirmation_trace.total_calls, 128)
        self.assertEqual(
            search._apply_rebenchmark_timings.call_args_list,
            [call(members, [40.0, 20.0]), call(members, [40.0, 20.0])],
        )
        metric = LFBOPatternSearch._flash_terminal_trace_metric(
            ["config-a", "config-b"],
            trace,
        )
        self.assertEqual(metric["target_ms"], 200.0)
        self.assertEqual(metric["repeat_reference_perf_ms"], 40.0)
        self.assertEqual(metric["sweep_count"], 6)
        self.assertEqual(metric["calls_per_sample"], 1)
        self.assertEqual(metric["total_calls"], 6)
        self.assertEqual(len(metric["elapsed_ms"]), 6)
        self.assertNotIn("sweeps", metric)

    def test_isolated_rebenchmark_rep_ms_is_timeout_bounded(self) -> None:
        self.assertEqual(PopulationBasedSearch._isolated_rep_ms(1000.0, 30), 1000)
        self.assertEqual(PopulationBasedSearch._isolated_rep_ms(60000.0, 30), 15000)
        with patch.dict(os.environ, {"HELION_CAP_REBENCHMARK_REPEAT": "50"}):
            self.assertEqual(PopulationBasedSearch._isolated_rep_ms(1000.0, 30), 50)

    def test_steady_rebenchmark_rep_ms_honors_cap(self) -> None:
        self.assertEqual(PopulationBasedSearch._steady_rebenchmark_rep_ms(1000.0), 1000)
        with patch.dict(os.environ, {"HELION_CAP_REBENCHMARK_REPEAT": "50"}):
            self.assertEqual(
                PopulationBasedSearch._steady_rebenchmark_rep_ms(1000.0), 50
            )

    def test_rebenchmark_repeat_ignores_global_optimistic_outlier(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        repeats: list[int] = []

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            repeats.append(repeat)
            return [50.0 for _ in fns]

        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.settings.autotune_benchmark_fn = custom_benchmark_fn
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 0.001
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.config_spec = SimpleNamespace(backend=None)
        member_a = PopulationMember(lambda: None, [50.0], (), helion.Config())
        member_b = PopulationMember(lambda: None, [51.0], (), helion.Config())

        search.rebenchmark([member_a, member_b], target_ms=1000.0)

        self.assertEqual(repeats, [19])

    def test_final_rebenchmark_can_skip_suspicious_confirmation(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0.9,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        search.benchmark_provider = SimpleNamespace(
            mutated_arg_indices=[],
            benchmark_isolated=Mock(side_effect=AssertionError("unexpected isolated")),
        )
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        search.config_spec = SimpleNamespace(backend=None)

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            return [50.0 for _ in fns]

        search.settings.autotune_benchmark_fn = custom_benchmark_fn
        member_a = PopulationMember(lambda: None, [100.0], (), helion.Config())
        member_b = PopulationMember(lambda: None, [101.0], (), helion.Config())

        search.rebenchmark(
            [member_a, member_b],
            target_ms=1000.0,
            use_isolated=False,
            confirm_suspicious=False,
        )

        self.assertEqual(member_a.perfs, [100.0, 50.0])
        self.assertEqual(member_b.perfs, [101.0, 50.0])

    def test_rebenchmark_can_use_steady_backend_timer(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 100.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        calls: list[tuple[int, int, str]] = []

        def steady_bench(
            fn: Callable[[], object],
            *,
            warmup: int,
            rep: int,
            return_mode: str,
            process_group_name: str | None = None,
        ) -> float:
            fn()
            calls.append((warmup, rep, return_mode))
            return 50.0 + len(calls)

        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(get_do_bench=lambda: steady_bench)
        )
        member_a = PopulationMember(lambda: None, [100.0], (), helion.Config())
        member_b = PopulationMember(lambda: None, [101.0], (), helion.Config())

        with patch("helion.autotuner.base_search.clear_jit_fast_path_caches") as clear:
            search.rebenchmark(
                [member_a, member_b],
                target_ms=1500.0,
                use_isolated=False,
                confirm_suspicious=False,
                use_interleaved=False,
            )

        self.assertEqual(calls, [(1000, 1500, "median"), (1000, 1500, "median")])
        self.assertEqual(member_a.perfs, [100.0, 51.0])
        self.assertEqual(member_b.perfs, [101.0, 52.0])
        self.assertEqual(clear.call_count, 2)

    def test_final_rebenchmark_respects_custom_benchmark_fn(
        self,
    ) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.args = ()
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 9.0
        search.benchmark_provider = SimpleNamespace(mutated_arg_indices=[])
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        calls: list[str] = []

        def fn_a() -> None:
            calls.append("a")

        def fn_b() -> None:
            calls.append("b")

        def steady_bench(
            fn: Callable[[], object],
            *,
            warmup: int,
            rep: int,
            return_mode: str,
            process_group_name: str | None = None,
        ) -> float:
            raise AssertionError("unexpected steady benchmark")

        def custom_benchmark_fn(
            fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
        ) -> list[float]:
            self.assertIn("Final verification", desc or "")
            self.assertEqual(repeat, 500)
            timings = []
            for fn in fns:
                fn()
                timings.append(9.5 if calls[-1] == "a" else 10.0)
            return timings

        settings.autotune_benchmark_fn = custom_benchmark_fn

        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(get_do_bench=lambda: steady_bench)
        )
        config_a = helion.Config(num_warps=4)
        config_b = helion.Config(num_warps=8)
        member_a = PopulationMember(fn_a, [10.0], (), config_a, status="ok")
        member_b = PopulationMember(fn_b, [9.0], (), config_b, status="ok")
        search._benchmarked_members = {
            config_a: member_a,
            config_b: member_b,
        }
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = [member_b]

        with (
            clean_final_rebenchmark_env(),
            without_env_var("HELION_CAP_REBENCHMARK_REPEAT"),
        ):
            self.assertIs(search.final_rebenchmark_best(member_b), member_a)

    def test_rebenchmark_uses_target_ms_for_isolated_benchmark(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.best_perf_so_far = 0.05
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        reps: list[int] = []

        def benchmark_isolated(
            fns: list[Callable[[], object]], *, warmup: int, rep: int, desc: str
        ) -> list[float]:
            reps.append(rep)
            return [0.05 for _ in fns]

        search.benchmark_provider = SimpleNamespace(
            benchmark_isolated=benchmark_isolated,
            mutated_arg_indices=[],
        )
        member_a = PopulationMember(lambda: None, [0.05], (), helion.Config())
        member_b = PopulationMember(lambda: None, [0.06], (), helion.Config())

        search.rebenchmark([member_a, member_b], target_ms=1000.0)

        self.assertEqual(reps, [1000])

    def test_benchmarked_member_history_is_bounded_and_deduped(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        config_a = helion.Config(num_warps=4)
        config_b = helion.Config(num_warps=8)
        config_c = helion.Config(num_warps=16)
        slow_a = PopulationMember(lambda: None, [10.0], (), config_a, status="ok")
        fast_a = PopulationMember(lambda: None, [8.0], (), config_a, status="ok")
        member_b = PopulationMember(lambda: None, [9.0], (), config_b, status="ok")
        member_c = PopulationMember(lambda: None, [7.0], (), config_c, status="ok")

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(slow_a)
            search._record_benchmarked_member(fast_a)
            search._record_benchmarked_member(member_b)
            search._record_benchmarked_member(member_c)

        # A snapshot of the faster member for config_a is retained (deduped).
        self.assertEqual(search._benchmarked_members[config_a].perfs, fast_a.perfs)
        self.assertIsNot(search._benchmarked_members[config_a].perfs, fast_a.perfs)
        self.assertNotIn(config_b, search._benchmarked_members)
        self.assertEqual(set(search._benchmarked_members), {config_a, config_c})

    def test_cute_finalist_history_uses_latest_verified_perf(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="cute", cute_flash_search_enabled=True
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        initial_fast_final_slow = PopulationMember(
            lambda: None,
            [1.0, 10.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        middle = PopulationMember(
            lambda: None,
            [2.0, 5.0],
            [],
            helion.Config(num_warps=8),
            status="ok",
        )
        initial_slow_final_fast = PopulationMember(
            lambda: None,
            [3.0, 0.5],
            [],
            helion.Config(num_warps=16),
            status="ok",
        )

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(initial_fast_final_slow)
            search._record_benchmarked_member(middle)
            search._record_benchmarked_member(initial_slow_final_fast)

        self.assertEqual(
            set(search._benchmarked_members),
            {middle.config, initial_slow_final_fast.config},
        )

    def test_non_cute_finalist_history_keeps_low_water_perf(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="triton", cute_flash_search_enabled=False
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        initial_fast_final_slow = PopulationMember(
            lambda: None,
            [1.0, 10.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        middle = PopulationMember(
            lambda: None,
            [2.0, 5.0],
            [],
            helion.Config(num_warps=8),
            status="ok",
        )
        initial_slow_final_fast = PopulationMember(
            lambda: None,
            [3.0, 0.5],
            [],
            helion.Config(num_warps=16),
            status="ok",
        )

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(initial_fast_final_slow)
            search._record_benchmarked_member(middle)
            search._record_benchmarked_member(initial_slow_final_fast)

        self.assertEqual(
            set(search._benchmarked_members),
            {initial_fast_final_slow.config, initial_slow_final_fast.config},
        )

    def test_non_cute_rebenchmark_refreshes_retained_snapshot(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="triton", cute_flash_search_enabled=False
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.best_perf_so_far = 5.5
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))

        improves = PopulationMember(
            lambda: None,
            [6.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        unchanged = PopulationMember(
            lambda: None,
            [5.5],
            [],
            helion.Config(num_warps=8),
            status="ok",
        )

        search.benchmark_provider = SimpleNamespace(
            benchmark_isolated=lambda fns, **kwargs: [5.0, 5.5],
            mutated_arg_indices=[],
        )

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(improves)
            search._record_benchmarked_member(unchanged)
            search.rebenchmark([improves, unchanged], target_ms=200.0)

        snapshot = search._benchmarked_members[improves.config]
        self.assertEqual(snapshot.perfs, [6.0, 5.0])
        self.assertIsNot(snapshot.perfs, improves.perfs)

    def test_non_cute_history_keeps_finite_low_water_after_failed_rebenchmark(
        self,
    ) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="triton", cute_flash_search_enabled=False
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        member = PopulationMember(
            lambda: None,
            [5.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(member)
            member.perfs.append(float("inf"))
            search._refresh_benchmarked_members_after_rebenchmark([member])

        self.assertEqual(
            search._benchmarked_members[member.config].perfs,
            [5.0, float("inf")],
        )

    def test_cute_rebenchmark_refresh_reinserts_pruned_finalist(self) -> None:
        settings = Settings(
            autotune_log_level=logging.CRITICAL,
            autotune_suspicious_rebenchmark_ratio=0,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search.config_spec = SimpleNamespace(
            backend_name="cute", cute_flash_search_enabled=True
        )
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.best_perf_so_far = 1.0
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))

        initial_fast_final_slow = PopulationMember(
            lambda: None,
            [1.0],
            [],
            helion.Config(num_warps=4),
            status="ok",
        )
        middle = PopulationMember(
            lambda: None,
            [2.0],
            [],
            helion.Config(num_warps=8),
            status="ok",
        )
        initial_slow_final_fast = PopulationMember(
            lambda: None,
            [3.0],
            [],
            helion.Config(num_warps=16),
            status="ok",
        )
        members = [initial_fast_final_slow, middle, initial_slow_final_fast]

        def benchmark_isolated(
            fns: list[Callable[[], object]], *, warmup: int, rep: int, desc: str
        ) -> list[float]:
            self.assertEqual(len(fns), 3)
            return [10.0, 5.0, 0.5]

        search.benchmark_provider = SimpleNamespace(
            benchmark_isolated=benchmark_isolated,
            mutated_arg_indices=[],
        )

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            for member in members:
                search._record_benchmarked_member(member)
            self.assertNotIn(
                initial_slow_final_fast.config, search._benchmarked_members
            )
            initial_snapshot = search._benchmarked_members[
                initial_fast_final_slow.config
            ]
            self.assertIsNot(initial_snapshot.perfs, initial_fast_final_slow.perfs)

            search.rebenchmark(members, target_ms=200.0)

        self.assertEqual(
            set(search._benchmarked_members),
            {middle.config, initial_slow_final_fast.config},
        )
        refreshed = search._benchmarked_members[initial_slow_final_fast.config]
        self.assertEqual(refreshed.perfs, [3.0, 0.5])
        self.assertIsNot(refreshed.perfs, initial_slow_final_fast.perfs)

    def test_pinned_finalist_survives_benchmarked_member_pruning(self) -> None:
        settings = Settings(autotune_log_level=logging.CRITICAL)
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}

        pinned_config = helion.Config(num_warps=4)
        fast_config = helion.Config(num_warps=8)
        faster_config = helion.Config(num_warps=16)
        slow_config = helion.Config(num_warps=32)
        pinned = PopulationMember(lambda: None, [10.0], (), pinned_config, status="ok")
        fast = PopulationMember(lambda: None, [8.0], (), fast_config, status="ok")
        faster = PopulationMember(lambda: None, [7.0], (), faster_config, status="ok")
        slow = PopulationMember(lambda: None, [9.0], (), slow_config, status="ok")
        search.pin_finalist_config(pinned_config)

        with clean_final_rebenchmark_env(HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"):
            search._record_benchmarked_member(pinned)
            search._record_benchmarked_member(fast)
            search._record_benchmarked_member(faster)
            search._record_benchmarked_member(slow)

        # The pinned config survives pruning (kept as a snapshot in the map).
        self.assertIn(pinned_config, search._pinned_finalist_members)
        self.assertEqual(
            search._pinned_finalist_members[pinned_config].perfs, pinned.perfs
        )
        self.assertEqual(set(search._benchmarked_members), {fast_config, faster_config})

    def test_pinned_finalist_refresh_groups_equal_config_duplicates(self) -> None:
        for backend_name, cute_flash_search_enabled in (
            ("triton", False),
            ("cute", True),
        ):
            with self.subTest(backend=backend_name):
                settings = Settings(autotune_log_level=logging.CRITICAL)
                search = PopulationBasedSearch.__new__(PopulationBasedSearch)
                search.settings = settings
                search.log = AutotuningLogger(settings)
                search.config_spec = SimpleNamespace(
                    backend_name=backend_name,
                    cute_flash_search_enabled=cute_flash_search_enabled,
                )
                search._benchmarked_members = {}
                search._pinned_finalist_configs = set()
                search._pinned_finalist_members = {}

                config = helion.Config(num_warps=4)
                original = PopulationMember(
                    lambda: None, [10.0], (), config, status="ok"
                )
                better_duplicate = PopulationMember(
                    lambda: None,
                    [6.0],
                    (),
                    helion.Config(num_warps=4),
                    status="ok",
                )
                failed_duplicate = PopulationMember(
                    lambda: None,
                    [float("inf")],
                    (),
                    helion.Config(num_warps=4),
                    status="timeout",
                )
                worse_duplicate = PopulationMember(
                    lambda: None,
                    [9.0],
                    (),
                    helion.Config(num_warps=4),
                    status="ok",
                )
                search.pin_finalist_config(config)

                with clean_final_rebenchmark_env(
                    HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"
                ):
                    search._record_benchmarked_member(original)
                    search._refresh_benchmarked_members_after_rebenchmark(
                        [better_duplicate, failed_duplicate, worse_duplicate]
                    )

                self.assertIn(config, search._pinned_finalist_members)
                self.assertEqual(
                    search._pinned_finalist_members[config].perfs,
                    better_duplicate.perfs,
                )

    def test_failed_duplicate_does_not_evict_pinned_finalist(self) -> None:
        for backend_name, cute_flash_search_enabled in (
            ("triton", False),
            ("cute", True),
        ):
            with self.subTest(backend=backend_name):
                settings = Settings(autotune_log_level=logging.CRITICAL)
                search = PopulationBasedSearch.__new__(PopulationBasedSearch)
                search.settings = settings
                search.log = AutotuningLogger(settings)
                search.config_spec = SimpleNamespace(
                    backend_name=backend_name,
                    cute_flash_search_enabled=cute_flash_search_enabled,
                )
                search._benchmarked_members = {}
                search._pinned_finalist_configs = set()
                search._pinned_finalist_members = {}

                config = helion.Config(num_warps=4)
                original = PopulationMember(
                    lambda: None, [10.0], (), config, status="ok"
                )
                failed_duplicate = PopulationMember(
                    lambda: None,
                    [float("inf")],
                    (),
                    helion.Config(num_warps=4),
                    status="timeout",
                )
                search.pin_finalist_config(config)

                with clean_final_rebenchmark_env(
                    HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K="2"
                ):
                    search._record_benchmarked_member(original)
                    search._refresh_benchmarked_members_after_rebenchmark(
                        [failed_duplicate]
                    )

                self.assertIn(config, search._pinned_finalist_members)
                self.assertEqual(
                    search._pinned_finalist_members[config].perfs, original.perfs
                )

    def test_autotune_configuration_cloning(self) -> None:
        """Tests base_search._clone_args function."""

        config1 = helion.Config(block_sizes=[32, 32], num_warps=4)
        config2 = helion.Config(block_sizes=[64, 64], num_warps=8)

        @helion.kernel(
            configs=[config1, config2],
            autotune_log_level=0,
        )
        def nested_in_place_add(
            a: Sequence[torch.Tensor],
            b: Sequence[torch.Tensor],
            epsilon: float,
            out: Sequence[torch.Tensor],
        ):
            for tile in hl.tile(out[0].size()):
                out[0][tile] += a[0][tile] + b[0][tile] + epsilon
            for tile in hl.tile(out[1].size()):
                out[1][tile] += a[1][tile] + b[1][tile] + epsilon

        epsilon = 1e-6
        args = (
            [torch.ones([128], device=DEVICE), torch.ones([128], device=DEVICE)],
            [torch.ones([128], device=DEVICE), torch.ones([128], device=DEVICE)],
            epsilon,
            [torch.zeros([128], device=DEVICE), torch.zeros([128], device=DEVICE)],
        )

        # Run autotuning
        nested_in_place_add(*args)

        # test that we overwrite c only once and the arguments are correctly
        #  cloned for each autotune run
        ref_out = [
            torch.full([128], 2.0, device=DEVICE) + epsilon,
            torch.full([128], 2.0, device=DEVICE) + epsilon,
        ]
        torch.testing.assert_close(args[3], ref_out)

    def test_only_mutated_tensors_cloned_during_benchmark(self) -> None:
        """
        During benchmarking, only mutated tensors should be cloned.
        Non-mutated tensors should only be cloned during initialization.
        """
        config1 = helion.Config(block_sizes=[32], num_warps=4)
        config2 = helion.Config(block_sizes=[64], num_warps=4)

        @helion.kernel(configs=[config1, config2], autotune_log_level=0)
        def inplace_add(
            a: torch.Tensor,
            b: torch.Tensor,
            epsilon: float,
            out: torch.Tensor,
        ):
            for tile in hl.tile(out.size()):
                out[tile] += a[tile] + b[tile] + epsilon

        a = torch.full([128], 1.0, device=DEVICE)
        b = torch.full([128], 2.0, device=DEVICE)
        epsilon = 1e-6
        out = torch.zeros([128], device=DEVICE)

        # Track clones separately for mutated vs non-mutated tensors
        mutated_ptrs = {out.data_ptr()}
        non_mutated_ptrs = {a.data_ptr(), b.data_ptr()}
        mutated_clones = [0]
        non_mutated_clones = [0]

        original_clone = torch.Tensor.clone

        def tracking_clone(self, *args, **kwargs):
            result = original_clone(self, *args, **kwargs)
            if self.data_ptr() in mutated_ptrs:
                mutated_ptrs.add(result.data_ptr())
                mutated_clones[0] += 1
            if self.data_ptr() in non_mutated_ptrs:
                non_mutated_ptrs.add(result.data_ptr())
                non_mutated_clones[0] += 1
            return result

        with patch.object(torch.Tensor, "clone", tracking_clone):
            inplace_add(a, b, epsilon, out)

        # Mutated tensor (out) should be cloned during baseline AND benchmarking:
        #   _compute_baseline: 1 + baseline_post_args: 1
        #   + 2 benchmark runs = 4 total
        self.assertEqual(
            mutated_clones[0],
            4,
            f"Mutated tensor cloned {mutated_clones[0]} times, expected 4.",
        )

        # Non-mutated tensors (a, b) should only be cloned during baseline:
        #   _compute_baseline: 2 = 2 total
        self.assertEqual(
            non_mutated_clones[0],
            2,
            f"Non-mutated tensors cloned {non_mutated_clones[0]} times, expected 2. "
            f"Only mutated tensors should be cloned during benchmarking.",
        )

        expected = torch.full([128], 3.0, device=DEVICE) + epsilon
        torch.testing.assert_close(out, expected)

    @skipIfXPU("CUDA specific API used to check memory usage")
    def test_chunked_allclose_memory(self):
        """Test that autotuning accuracy checks use chunked comparison for large tensors."""
        import helion.autotuner.accuracy as _accuracy

        numel = 2**22  # 4M float32 elements (~16 MB each)
        # The default chunk_size (2**22) would not chunk a tensor this small,
        # so pass a smaller one through the patched helper below; 4 chunks is
        # enough to exercise the chunked path and its memory bound.
        chunk_size = 2**20

        config1 = helion.Config(block_sizes=[128], num_warps=4)
        config2 = helion.Config(block_sizes=[256], num_warps=4)

        # Pin the accuracy check to the parent so the patched chunked helper
        # below is observed; the default subprocess path runs it in the worker.
        @helion.kernel(
            configs=[config1, config2],
            autotune_log_level=0,
            autotune_benchmark_subprocess=False,
        )
        def vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(a.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn(numel, device=DEVICE)
        b = torch.randn(numel, device=DEVICE)

        # Measure naive baseline: peak memory of torch.testing.assert_close
        # on tensors of the same size
        ref_a = torch.randn(numel, device=DEVICE)
        ref_b = ref_a.clone()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        torch.testing.assert_close(ref_a, ref_b, atol=1e-2, rtol=1e-2)
        naive_peak = torch.cuda.max_memory_allocated() - base_mem
        del ref_a, ref_b

        # Patch the moved chunked helper to record peak memory delta per call.
        real_chunked_assert_close = _accuracy._chunked_assert_close
        peaks: list[int] = []

        def measuring_chunked_assert_close(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            real_chunked_assert_close(*args, chunk_size=chunk_size, **kwargs)
            peak = torch.cuda.max_memory_allocated() - before
            peaks.append(peak)

        with patch.object(
            _accuracy, "_chunked_assert_close", measuring_chunked_assert_close
        ):
            out = vec_add(a, b)

        # Accuracy check was called at least once
        self.assertGreater(len(peaks), 0, "Expected _chunked_assert_close to be called")

        # Every call's peak memory should be less than naive peak
        for i, p in enumerate(peaks):
            self.assertLess(
                p,
                naive_peak * 0.5,
                f"Call {i}: peak {p} should be < 50% of naive {naive_peak}",
            )

        torch.testing.assert_close(out, a + b)

    def test_autotune_baseline_accuracy_check_fn(self) -> None:
        """Test the built-in assert_close_with_mismatch_tolerance utility.

        Simulates a scenario where most elements match exactly, but a
        tiny fraction (1/N) have large diffs.  The default
        torch.testing.assert_close would reject this, but the utility
        falls back to checking mismatch_pct, max_abs_diff, and
        max_rel_diff thresholds and accepts it.
        """
        import functools

        import helion.autotuner.base_search as base_search_module

        bad_config = helion.Config(block_sizes=[1], num_warps=8)
        good_config = helion.Config(block_sizes=[1], num_warps=4)

        @helion.kernel(
            configs=[bad_config, good_config],
            autotune_log_level=0,
            autotune_baseline_accuracy_check_fn=functools.partial(
                assert_close_with_mismatch_tolerance,
                max_mismatch_pct=0.01,
                max_rel_diff=15.0,
            ),
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            o = torch.empty_like(a)
            for t in hl.tile(o.size()):
                o[t] = a[t] + b[t]
            return o

        # Use a large tensor so mismatch fraction is tiny (1/N)
        N = 4096
        a = torch.randn([N], device=DEVICE)
        b = torch.randn([N], device=DEVICE)
        bound = add.bind((a, b))
        original_compile = bound.compile_config

        def inject_large_diffs_to_some_elements(
            config: helion.Config, *, allow_print: bool = True
        ):
            fn = original_compile(config, allow_print=allow_print)
            if config == bad_config:
                # Simulate mismatch: 1 element out of N with rel diff ~12
                def patched(*fn_args, **fn_kwargs):
                    result = fn(*fn_args, **fn_kwargs)
                    result[0] = result[0] + 12.0 * result[0].abs().clamp(min=1e-6)
                    return result

                return patched
            return fn

        with patch.object(
            bound,
            "compile_config",
            side_effect=inject_large_diffs_to_some_elements,
        ):
            search = FiniteSearch(bound, (a, b), configs=[bad_config, good_config])
            search._prepare()

            with patch.object(
                search.benchmark_provider,
                "_create_precompile_future",
                side_effect=lambda config, fn: base_search_module.PrecompileFuture.skip(
                    search.benchmark_provider._precompile_context(), config, True
                ),
            ):
                # bad_config has a few large diffs — custom check should accept it
                bad_time = search.benchmark(bad_config).perf
                assert not math.isinf(bad_time), (
                    "custom check should allow config with 1/N large diffs"
                )
                self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

                # good_config produces exact match — should also pass
                good_time = search.benchmark(good_config).perf
                assert not math.isinf(good_time)
                self.assertEqual(search._autotune_metrics.num_accuracy_failures, 0)

        # Direct checks: element 0 has abs_diff=9.0, rel_diff=9.0
        actual = torch.tensor([10.0, 1.0, 1.0, 1.0], device=DEVICE)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0], device=DEVICE)

        # Only max_rel_diff exceeded (abs_diff=9 < 20, rel_diff=9 > 5)
        with self.assertRaisesRegex(AssertionError, "Relative diff too large"):
            assert_close_with_mismatch_tolerance(
                actual,
                expected,
                max_mismatch_pct=0.5,
                max_abs_diff=20.0,
                max_rel_diff=5.0,
            )

        # Only max_abs_diff exceeded (abs_diff=9 > 5, rel_diff=9 < 20)
        with self.assertRaisesRegex(AssertionError, "Absolute diff too large"):
            assert_close_with_mismatch_tolerance(
                actual,
                expected,
                max_mismatch_pct=0.5,
                max_abs_diff=5.0,
                max_rel_diff=20.0,
            )

    def test_autotune_baseline_accuracy_check_fn_rejects(self) -> None:
        """Test that a strict custom check function properly rejects configs."""
        cfg1 = helion.Config(block_sizes=[1], num_warps=4)
        cfg2 = helion.Config(block_sizes=[1], num_warps=8)

        def strict_check(actual: object, expected: object) -> None:
            # Always reject
            raise AssertionError("strict check: always fails")

        # Custom accuracy-check fns always run in-process; skip the benchmark
        # worker subprocess (seconds of startup overhead).
        @helion.kernel(
            configs=[cfg1, cfg2],
            autotune_log_level=0,
            autotune_baseline_accuracy_check_fn=strict_check,
            autotune_benchmark_subprocess=False,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            o = torch.empty_like(a)
            for t in hl.tile(o.size()):
                o[t] = a[t] + b[t]
            return o

        a = torch.randn([32], device=DEVICE)
        b = torch.randn([32], device=DEVICE)
        bound = add.bind((a, b))
        search = FiniteSearch(bound, (a, b), configs=[cfg1, cfg2])

        with self.assertRaises(AssertionError):
            search.autotune()
        self.assertEqual(
            search._autotune_metrics.num_accuracy_failures, len(search.configs)
        )


@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
@skipUnlessCuteAvailable("CUTLASS CuTe Python DSL is not available")
@onlyBackends(["cute"])
class TestCuteAutotuner(TestCase):
    def test_implicit_call_uses_autotuner_fn(self) -> None:
        calls: list[bool] = []

        def autotuner_fn(bound_kernel, args, **kwargs):
            class RecordingAutotuner:
                def autotune(self, *, skip_cache: bool = False):
                    calls.append(skip_cache)
                    return bound_kernel.config_spec.default_config()

            return RecordingAutotuner()

        @helion.kernel(
            backend="cute",
            autotuner_fn=autotuner_fn,
            autotune_log_level=0,
        )
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8], device=DEVICE),
            torch.randn([8], device=DEVICE),
        )
        torch.testing.assert_close(add(*args), sum(args))
        self.assertEqual(calls, [False])

    def test_cute_config_generation_repairs_num_threads(self) -> None:
        @helion.kernel(backend="cute", autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([16, 64], device=DEVICE),
            torch.randn([16, 64], device=DEVICE),
        )
        bound = add.bind(args)
        gen = ConfigGeneration(bound.config_spec)
        flat_keys = {
            key for key, _count, _is_sequence in gen.config_spec.flat_key_layout()
        }
        # ``loop_orders`` is exposed for the CuTe non-tcgen05 search
        # surface. ``cute_vector_widths`` is the per-axis vec width slot
        # registered for non-reduction tile blocks.
        # ``load_eviction_policies`` carries per-load-site L1 eviction
        # hints (lowered on the vectorized load forms). The set still
        # excludes Triton-style knobs that the CuTe path does not consume.
        self.assertEqual(
            flat_keys,
            {
                "block_sizes",
                "num_threads",
                "loop_orders",
                "cute_vector_widths",
                "cute_lane_layouts",
                "cute_cluster_n",
                "cute_min_blocks_per_mp",
                "load_eviction_policies",
            },
        )

        repaired = gen.unflatten(
            gen.flatten(helion.Config(block_sizes=[16, 64], num_threads=[128, 128]))
        )
        self.assertEqual(repaired.block_sizes, [16, 64])
        self.assertEqual(repaired.num_threads, [16, 64])

        configs = [gen.random_config() for _ in range(20)]
        self.assertTrue(any(config.num_threads for config in configs))
        for config in configs:
            self.assertLessEqual(
                set(config.config),
                {
                    "block_sizes",
                    "num_threads",
                    "loop_orders",
                    "cute_vector_widths",
                    "cute_lane_layouts",
                    "cute_cluster_n",
                    "cute_min_blocks_per_mp",
                    "load_eviction_policies",
                },
            )
            self.assertNotIn("persistent", config.pid_type)
            explicit_threads = [nt for nt in config.num_threads if nt > 0]
            if explicit_threads:
                self.assertLessEqual(math.prod(explicit_threads), 1024)
            for block_size, num_threads in zip(
                config.block_sizes,
                config.num_threads,
                strict=False,
            ):
                if num_threads > 0:
                    self.assertLessEqual(num_threads, block_size)
                    self.assertEqual(block_size % num_threads, 0)
        # Deterministic round-trip pins the widened surface: a config
        # explicitly built with ``loop_orders=[[1, 0]]`` must survive
        # flatten/unflatten unchanged (otherwise the autotuner cannot
        # actually explore the alternate order).
        round_tripped = gen.unflatten(
            gen.flatten(
                helion.Config(
                    block_sizes=[16, 64],
                    num_threads=[16, 64],
                    loop_orders=[[1, 0]],
                )
            )
        )
        self.assertEqual(round_tripped.loop_orders, [[1, 0]])

    @skipIfCudaCapabilityLessThan(
        (10, 0), reason="tcgen05 requires CUDA capability >= 10.0"
    )
    def test_cute_tcgen05_search_surface_includes_loop_orders(self) -> None:
        """tcgen05 autotuning can explore alternate output-tile orders."""
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
        )
        bound = _get_examples_matmul().bind(args)
        # Confirm we are on the tcgen05 search branch.
        self.assertTrue(bound.config_spec.cute_tcgen05_search_enabled)
        flat_keys = {
            key for key, _count, _is_sequence in bound.config_spec.flat_key_layout()
        }
        self.assertIn("loop_orders", flat_keys)

        gen = ConfigGeneration(bound.config_spec)
        config = helion.Config(
            block_sizes=[128, 128, 64],
            loop_orders=[[1, 0]],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=1,
            tcgen05_cluster_n=1,
        )
        round_tripped = gen.unflatten(gen.flatten(config))
        self.assertEqual(round_tripped.loop_orders, [[1, 0]])
        code = bound.to_triton_code(round_tripped)
        self.assertIn("cute.gemm(", code)
        self.assertIn("StaticPersistentTileScheduler", code)

        actual = bound.compile_config(round_tripped)(*args)
        torch.testing.assert_close(actual, args[0] @ args[1], atol=0.125, rtol=0.02)

    def test_cute_flash_search_surface(self) -> None:
        """Flash attention exposes a general, normalized CuTe search surface.

        The standard surface must be broad enough to tune rather than replay a
        sequence-length winner. Structural legality can still constrain the
        available families. Legacy and manual fixed configs remain accepted,
        while unrelated CuTe kernels never receive flash-attention knobs.
        """
        from helion._compiler.cute import cute_flash

        @helion.kernel(backend="cute", static_shapes=True)
        def flash_attention(q_in, k_in, v_in):
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim])
            out = torch.empty_like(q_view)
            qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                qt = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    kt = k_view[tile_b, tile_n, :]
                    qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    qk = qk - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    vt = v_view[tile_b, tile_n, :]
                    acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
                    m_i = m_ij
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        @helion.kernel(backend="cute", static_shapes=True)
        def flash_attention_with_aux(q_in, k_in, v_in):
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim])
            out = torch.empty_like(q_view)
            aux = torch.empty(
                [q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32
            )
            qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                qt = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    kt = k_view[tile_b, tile_n, :]
                    qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
                    m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                    qk = qk - m_ij[:, :, None]
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    vt = v_view[tile_b, tile_n, :]
                    acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
                    m_i = m_ij
                acc = acc / l_i[:, :, None]
                aux[tile_b, tile_m] = torch.zeros_like(l_i)
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size()), aux.view(q_in.size()[:-1])

        @helion.kernel(backend="cute", static_shapes=True)
        def sparse_flash_attention(q_in, k_in, v_in):
            m_dim = q_in.size(-2)
            n_dim = k_in.size(-2)
            head_dim = hl.specialize(q_in.size(-1))
            q_view = q_in.reshape([-1, m_dim, head_dim])
            v_view = v_in.reshape([-1, n_dim, head_dim])
            k_view = k_in.reshape([-1, n_dim, head_dim])
            out = torch.empty_like(q_view)
            qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
            for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
                m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
                l_i = torch.full_like(m_i, 1.0)
                acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
                qt = q_view[tile_b, tile_m, :]
                for tile_n in hl.tile(v_view.size(1)):
                    kt = k_view[tile_b, tile_n, :]
                    qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
                    delta = tile_m.index[None, :, None] - tile_n.index[None, None, :]
                    qk = torch.where((delta >= 0) & (delta <= 64), qk, float("-inf"))
                    m_ij_keepdim = torch.maximum(
                        m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
                    )
                    qk = qk - m_ij_keepdim
                    m_ij = m_ij_keepdim.squeeze(-1)
                    p = torch.exp2(qk)
                    l_ij = torch.sum(p, -1)
                    alpha = torch.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, :, None]
                    vt = v_view[tile_b, tile_n, :]
                    acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
                    m_i = m_ij
                acc = acc / l_i[:, :, None]
                out[tile_b, tile_m, :] = acc.to(out.dtype)
            return out.view(q_in.size())

        def active_choices(fragment):
            return (
                fragment.choices
                if fragment.search_choices is None
                else fragment.search_choices
            )

        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        attn_bound = flash_attention.bind((q, k, v))
        self.assertTrue(attn_bound.config_spec.cute_flash_search_enabled)
        cache = default_autotuner_fn(attn_bound, (q, k, v))
        self.assertIsInstance(cache, LocalAutotuneCache)
        self.assertTrue(cache.key.search_policy_hash)  # type: ignore[attr-defined]
        self.assertIs(cache.autotuner._search_policy_cacheable, True)
        strict_cache = StrictLocalAutotuneCache(cache.autotuner)
        self.assertEqual(
            strict_cache.key.search_policy_hash,
            cache.key.search_policy_hash,  # type: ignore[attr-defined]
        )
        attn_keys = {
            key
            for key, _count, _is_sequence in attn_bound.config_spec.flat_key_layout()
        }
        self.assertLessEqual(
            set(cute_flash.FLASH_AUTOTUNE_CONFIG_KEYS),
            attn_keys,
        )
        self.assertTrue(set(cute_flash.FLASH_LEGACY_CONFIG_KEYS).isdisjoint(attn_keys))
        self.assertNotIn(cute_flash.FLASH_Q_TILE_COUNT_KEY, attn_keys)
        for generic_key in ("num_threads", "loop_orders", "cute_vector_widths"):
            self.assertNotIn(generic_key, attn_keys)

        bound_fragments = attn_bound.config_spec._flat_fields()
        self.assertLessEqual(
            {"fa4", "ws_overlap"},
            set(active_choices(bound_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
        )
        for key in (
            cute_flash.FLASH_KV_STAGE_KEY,
            cute_flash.FLASH_E2E_SCHEDULE_KEY,
            cute_flash.FLASH_EXP2_PACKET_KEY,
            cute_flash.FLASH_STAT_TRANSPORT_KEY,
            cute_flash.FLASH_PIPELINE_FAMILY_KEY,
            cute_flash.FLASH_SOFTMAX_DISC_KEY,
            cute_flash.FLASH_EPI_TMA_KEY,
            cute_flash.FLASH_PACKED_REDUCE_KEY,
            cute_flash.FLASH_PERSISTENT_LOOP_KEY,
            cute_flash.FLASH_SP_ROW_SUM_KEY,
            cute_flash.FLASH_SOFTMAX_SETUP_KEY,
            cute_flash.FLASH_EPI_TMA_SETUP_KEY,
        ):
            self.assertGreater(
                len(active_choices(bound_fragments[key])),
                1,
                key,
            )

        sparse_bound = sparse_flash_attention.bind((q, k, v))
        sparse_spec = sparse_bound.config_spec
        self.assertTrue(sparse_spec._cute_flash_has_kv_tile_pruning)
        self.assertTrue(sparse_spec._cute_flash_requires_ws_overlap)
        sparse_bound_fragments = sparse_spec._flat_fields()
        self.assertEqual(
            set(
                active_choices(
                    sparse_bound_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
                )
            ),
            {"ws_overlap"},
        )
        sparse_default = sparse_spec.default_config().config
        self.assertEqual(
            sparse_default[cute_flash.FLASH_PIPELINE_FAMILY_KEY], "ws_overlap"
        )
        self.assertTrue(sparse_default[cute_flash.FLASH_PACKED_REDUCE_KEY])
        stale_sparse = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
        )
        sparse_spec.normalize(stale_sparse)
        self.assertEqual(
            stale_sparse.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
            "ws_overlap",
        )
        self.assertNotIn(cute_flash.FLASH_TOPOLOGY_KEY, stale_sparse.config)

        # Aligned lengths share the same general search space. Only structural
        # legality, such as the parity needed by two-CTA families, may narrow it.
        def direct_fragments(num_kv, *, is_causal=False):
            return cute_flash.flash_autotune_fragments(
                64,
                num_kv,
                num_bh=16,
                dtype=torch.float16,
                is_causal=is_causal,
                standard_dense_output=not is_causal,
                standard_causal_output=is_causal,
            )

        dense_short = direct_fragments(64)
        dense_long = direct_fragments(512)
        self.assertEqual(
            {
                key: set(active_choices(fragment))
                for key, fragment in dense_short.items()
            },
            {
                key: set(active_choices(fragment))
                for key, fragment in dense_long.items()
            },
        )
        dense_families = set(
            active_choices(dense_short[cute_flash.FLASH_PIPELINE_FAMILY_KEY])
        )
        self.assertLessEqual(
            {"fa4", "ws_overlap", "fa4_2cta", "fa4_clc"},
            dense_families,
        )

        paired_only = direct_fragments(66)
        self.assertNotIn(
            "fa4_2cta",
            active_choices(paired_only[cute_flash.FLASH_PIPELINE_FAMILY_KEY]),
        )
        odd = direct_fragments(65)
        self.assertEqual(
            set(active_choices(odd[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
            {"ws_overlap"},
        )

        causal_short = direct_fragments(64, is_causal=True)
        causal_long = direct_fragments(512, is_causal=True)
        self.assertEqual(
            {
                key: set(active_choices(fragment))
                for key, fragment in causal_short.items()
            },
            {
                key: set(active_choices(fragment))
                for key, fragment in causal_long.items()
            },
        )
        self.assertEqual(
            set(active_choices(causal_short[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
            {"fa4", "fa4_2cta_causal", "ws_overlap"},
        )
        for key in (
            cute_flash.FLASH_MASKED_E2E_SCHEDULE_KEY,
            cute_flash.FLASH_CAUSAL_KV_ORDER_KEY,
            cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY,
        ):
            self.assertGreater(len(active_choices(causal_short[key])), 1, key)
        self.assertEqual(
            set(active_choices(causal_short[cute_flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY])),
            {1},
        )

        sparse = cute_flash.flash_autotune_fragments(
            64,
            64,
            num_bh=16,
            dtype=torch.float16,
            has_kv_tile_pruning=True,
            requires_ws_overlap=True,
        )
        self.assertEqual(
            set(active_choices(sparse[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
            {"ws_overlap"},
        )

        regular_small = cute_flash.flash_autotune_fragments(
            64,
            1,
            small_biased_candidate=False,
        )
        biased_small = cute_flash.flash_autotune_fragments(
            64,
            1,
            small_biased_candidate=True,
        )
        self.assertEqual(
            set(active_choices(regular_small[cute_flash.FLASH_SMALL_BIASED_KEY])),
            {True},
        )
        self.assertEqual(
            set(active_choices(biased_small[cute_flash.FLASH_SMALL_BIASED_KEY])),
            {False, True},
        )

        # Environment defaults may bias ordering but cannot collapse a standard
        # tuning dimension into a fixed value.
        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_KV_STAGE": "6",
                "HELION_CUTE_FLASH_E2E_SCHEDULE": "xu",
            },
        ):
            env_fragments = direct_fragments(64)
        self.assertIn(
            6,
            active_choices(env_fragments[cute_flash.FLASH_KV_STAGE_KEY]),
        )
        self.assertGreater(
            len(active_choices(env_fragments[cute_flash.FLASH_KV_STAGE_KEY])),
            1,
        )
        self.assertIn(
            "xu",
            active_choices(env_fragments[cute_flash.FLASH_E2E_SCHEDULE_KEY]),
        )
        self.assertGreater(
            len(active_choices(env_fragments[cute_flash.FLASH_E2E_SCHEDULE_KEY])),
            1,
        )

        with patch.dict(
            os.environ,
            {"HELION_CUTE_FLASH_TOPOLOGY": "invalid-topology"},
        ):
            invalid_topology_fragments = direct_fragments(64)
        invalid_family_fragment = invalid_topology_fragments[
            cute_flash.FLASH_PIPELINE_FAMILY_KEY
        ]
        self.assertEqual(invalid_family_fragment.default(), "ws_overlap")
        self.assertGreater(len(active_choices(invalid_family_fragment)), 1)

        for manual_family, causal in (("fa4_deep_1cta", True),):
            with (
                self.subTest(manual_family=manual_family),
                patch.dict(
                    os.environ,
                    {"HELION_CUTE_FLASH_PIPELINE_FAMILY": manual_family},
                ),
            ):
                manual_fragments = direct_fragments(512, is_causal=causal)
            manual_fragment = manual_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
            self.assertIn(manual_family, manual_fragment.choices)
            self.assertNotEqual(manual_fragment.default(), manual_family)
            self.assertNotIn(manual_family, active_choices(manual_fragment))

        long_q, long_k, long_v = (
            torch.empty(1, 1, 8192, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        long_bound = flash_attention.bind((long_q, long_k, long_v))
        long_gen = ConfigGeneration(long_bound.config_spec)

        legacy_compound = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
            cute_flash_clc=True,
            cute_flash_local_tma_partition=True,
            cute_flash_tensor_4d_tma=True,
            cute_flash_clc_heads_per_batch=1,
            cute_flash_clc_pdl=True,
            cute_flash_clc_stages=3,
            cute_flash_q_tile_count=-1,
            cute_flash_mma_interleave=True,
        )
        family_compound = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_clc_local_tma_4d",
            cute_flash_clc_heads_per_batch=1,
            cute_flash_clc_pdl=True,
            cute_flash_clc_stages=3,
        )
        self.assertEqual(
            long_gen.flatten(legacy_compound),
            long_gen.flatten(family_compound),
        )
        long_bound.config_spec.normalize(legacy_compound)
        long_bound.config_spec.normalize(family_compound)
        self.assertEqual(legacy_compound, family_compound)
        self.assertEqual(
            legacy_compound.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
            "fa4_clc_local_tma_4d",
        )
        self.assertEqual(
            legacy_compound.config[cute_flash.FLASH_Q_TILE_COUNT_KEY],
            2,
        )
        for legacy_key in cute_flash.FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
            self.assertNotIn(legacy_key, legacy_compound)

        legacy_two_cta = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
            cute_flash_use_2cta=True,
        )
        family_two_cta = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_2cta",
        )
        long_bound.config_spec.normalize(legacy_two_cta)
        long_bound.config_spec.normalize(family_two_cta)
        self.assertEqual(legacy_two_cta, family_two_cta)
        self.assertEqual(
            family_two_cta.config[cute_flash.FLASH_Q_TILE_COUNT_KEY],
            2,
        )

        family_authoritative = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_topology="ws_overlap",
            cute_flash_use_2cta=True,
            cute_flash_cga2_local=True,
            cute_flash_clc=True,
            cute_flash_local_tma_partition=True,
            cute_flash_tensor_4d_tma=True,
        )
        long_bound.config_spec.normalize(family_authoritative, _fix_invalid=True)
        self.assertEqual(
            family_authoritative.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
            "fa4",
        )
        for legacy_key in cute_flash.FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
            self.assertNotIn(legacy_key, family_authoritative)

        inactive_children = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_persistent=False,
            cute_flash_persistent_ctas_per_sm=4,
            cute_flash_recompute_tile_coords=True,
            cute_flash_persistent_loop="counted",
            cute_flash_softmax_disc=True,
            cute_flash_sp_row_sum="whole",
            cute_flash_epi_tma=False,
            cute_flash_epi_tma_setup="role_local",
            cute_flash_clc_heads_per_batch=64,
            cute_flash_clc_pdl=True,
            cute_flash_clc_stages=3,
        )
        long_bound.config_spec.normalize(inactive_children, _fix_invalid=True)
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_PERSISTENT_CTAS_PER_SM_KEY],
            1,
        )
        self.assertFalse(
            inactive_children.config[cute_flash.FLASH_RECOMPUTE_TILE_COORDS_KEY]
        )
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_PERSISTENT_LOOP_KEY],
            "while",
        )
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_SP_ROW_SUM_KEY],
            "fragment",
        )
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_EPI_TMA_SETUP_KEY],
            "shared",
        )
        self.assertEqual(
            inactive_children.config[cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY],
            0,
        )

        source_schedule = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_persistent=True,
            cute_flash_persistent_loop="counted",
            cute_flash_softmax_disc=False,
            cute_flash_sp_row_sum="whole",
            cute_flash_softmax_setup="stage_local",
            cute_flash_epi_tma=True,
            cute_flash_epi_tma_setup="role_local",
        )
        long_bound.config_spec.normalize(source_schedule)
        self.assertEqual(
            source_schedule.config[cute_flash.FLASH_PERSISTENT_LOOP_KEY],
            "counted",
        )
        self.assertEqual(
            source_schedule.config[cute_flash.FLASH_SP_ROW_SUM_KEY],
            "whole",
        )
        self.assertEqual(
            source_schedule.config[cute_flash.FLASH_SOFTMAX_SETUP_KEY],
            "stage_local",
        )
        self.assertEqual(
            source_schedule.config[cute_flash.FLASH_EPI_TMA_SETUP_KEY],
            "role_local",
        )

        manual_values = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_rescale_threshold=12.0,
            cute_flash_rescale_chunk_cols=64,
            cute_flash_corr_regs=72,
            cute_flash_other_regs=40,
        )
        long_bound.config_spec.normalize(manual_values)
        self.assertEqual(
            manual_values.config[cute_flash.FLASH_RESCALE_THRESHOLD_KEY],
            12.0,
        )
        self.assertEqual(
            manual_values.config[cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY],
            64,
        )
        self.assertEqual(
            manual_values.config[cute_flash.FLASH_CORR_REGS_KEY],
            72,
        )
        self.assertEqual(
            manual_values.config[cute_flash.FLASH_OTHER_REGS_KEY],
            40,
        )

        cga2_paired_epilogue = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_cga2_local_tma_4d",
            cute_flash_persistent=True,
            cute_flash_epi_stg=True,
            cute_flash_epi_stg_gmem="pair",
        )
        long_bound.config_spec.normalize(cga2_paired_epilogue)
        self.assertEqual(
            cga2_paired_epilogue.config[cute_flash.FLASH_EPI_STG_GMEM_KEY],
            "stage",
        )

        clc_wide_rescale = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_clc_local_tma_4d",
            cute_flash_rescale_chunk_cols=64,
            cute_flash_clc_heads_per_batch=1,
        )
        long_bound.config_spec.normalize(clc_wide_rescale)
        self.assertEqual(
            clc_wide_rescale.config[cute_flash.FLASH_RESCALE_CHUNK_COLS_KEY],
            32,
        )

        manual_packet = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_exp2_packet="deg2_16x6",
        )
        long_bound.config_spec.normalize(manual_packet)
        self.assertEqual(
            manual_packet.config[cute_flash.FLASH_EXP2_PACKET_KEY],
            "deg2_16x6",
        )
        self.assertEqual(
            manual_packet.config[cute_flash.FLASH_E2E_SCHEDULE_KEY],
            "16/6",
        )

        legacy_split = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_e2e_schedule="xu",
            cute_flash_exp2_impl="split",
            cute_flash_e2e_freq=16,
            cute_flash_e2e_res=4,
            cute_flash_e2e_offset=2,
        )
        long_bound.config_spec.normalize(legacy_split)
        self.assertEqual(
            legacy_split.config[cute_flash.FLASH_E2E_SCHEDULE_KEY],
            "16/4",
        )
        self.assertEqual(
            legacy_split.config[cute_flash.FLASH_E2E_OFFSET_KEY],
            2,
        )

        invalid_manual = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_corr_regs=96,
        )
        with self.assertRaises(exc.InvalidConfig):
            long_bound.config_spec.normalize(invalid_manual)

        for config in (
            family_compound,
            family_two_cta,
            source_schedule,
            manual_values,
            manual_packet,
        ):
            flat = long_gen.flatten(config)
            long_gen.encode_config(flat)
            expected = dict(config.config)
            expected.pop("cute_vector_widths", None)
            expected.pop("cute_lane_layouts", None)
            self.assertEqual(long_gen.unflatten(flat).config, expected)

        random_configs = [long_gen.random_config() for _ in range(24)]
        self.assertGreater(
            len({tuple(long_gen.flatten(config)) for config in random_configs}),
            1,
        )
        for random_config in random_configs:
            values = random_config.config
            effective = cute_flash.flash_effective_config_values(
                cute_flash.flash_config_from_config(
                    values,
                    64,
                    64,
                    num_bh=1,
                    standard_dense_output=True,
                )
            )
            for key, value in effective.items():
                self.assertEqual(values[key], value, key)
            for legacy_key in cute_flash.FLASH_LEGACY_CONFIG_KEYS:
                self.assertNotIn(legacy_key, values)

        q_odd, k_odd, v_odd = (
            torch.randn(2, 8, 384, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        odd_bound = flash_attention.bind((q_odd, k_odd, v_odd))
        odd_fragments = odd_bound.config_spec._flat_fields()
        self.assertEqual(
            set(active_choices(odd_fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY])),
            {"ws_overlap"},
        )
        stale_fa4 = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
        )
        odd_bound.config_spec.normalize(stale_fa4)
        self.assertEqual(
            stale_fa4.config[cute_flash.FLASH_PIPELINE_FAMILY_KEY],
            "ws_overlap",
        )
        self.assertNotIn(cute_flash.FLASH_TOPOLOGY_KEY, stale_fa4)

        aux_bound = flash_attention_with_aux.bind((q, k, v))
        self.assertFalse(aux_bound.config_spec.cute_flash_search_enabled)
        aux_keys = {
            key for key, _count, _is_sequence in aux_bound.config_spec.flat_key_layout()
        }
        self.assertIn("num_threads", aux_keys)
        self.assertTrue(set(cute_flash.FLASH_CONFIG_KEYS).isdisjoint(aux_keys))

        mm_bound = _get_examples_matmul().bind(
            (
                torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
                torch.randn([1024, 1024], device=DEVICE, dtype=torch.bfloat16),
            )
        )
        self.assertFalse(mm_bound.config_spec.cute_flash_search_enabled)
        mm_keys = {
            key for key, _count, _is_sequence in mm_bound.config_spec.flat_key_layout()
        }
        self.assertTrue(set(cute_flash.FLASH_CONFIG_KEYS).isdisjoint(mm_keys))

    def test_cute_flash_two_cta_uses_general_softmax_register_search(self) -> None:
        from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_REGS_KEY
        from helion._compiler.cute.cute_flash import flash_autotune_fragments

        for num_kv in (512, 1536, 2048):
            with self.subTest(num_kv=num_kv):
                fragment = flash_autotune_fragments(
                    64,
                    num_kv,
                    dtype=torch.float16,
                    is_causal=False,
                    standard_dense_output=True,
                    pipeline_family_override="fa4_2cta",
                )[FLASH_SOFTMAX_REGS_KEY]
                self.assertEqual(
                    set(fragment.search_choices or ()), {176, 184, 192, 200}
                )
                self.assertLessEqual({176, 184, 192, 200}, set(fragment.choices))


@onlyBackends(["triton"])
class TestAutotuneRandomSeed(RefEagerTestDisabled, TestCase):
    def _autotune_and_record(self, **settings: object) -> float:
        search_capture: dict[str, RecordingRandomSearch] = {}

        def autotuner_factory(bound_kernel, args, **kwargs):
            search = RecordingRandomSearch(bound_kernel, args, count=2, **kwargs)
            search_capture["search"] = search
            return search

        kernel_settings = {
            "autotuner_fn": autotuner_factory,
        }
        kernel_settings.update(settings)

        @helion.kernel(**kernel_settings)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound_kernel = add.bind(args)
        bound_kernel.autotune(args)
        torch.testing.assert_close(bound_kernel(*args), sum(args), rtol=1e-2, atol=1e-1)

        search = search_capture["search"]
        assert search.samples, (
            "expected RecordingRandomSearch to record a random sample"
        )
        return search.samples[0]

    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_autotune_random_seed_from_env_var(self) -> None:
        # same env var value -> same random sample
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "4242"}, clear=False
        ):
            first = self._autotune_and_record()
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "4242"}, clear=False
        ):
            second = self._autotune_and_record()
        self.assertEqual(first, second)

        # different env var values -> different random samples
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "101"}, clear=False
        ):
            first = self._autotune_and_record()
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_RANDOM_SEED": "102"}, clear=False
        ):
            second = self._autotune_and_record()
        self.assertNotEqual(first, second)

    @skipIfXPU("maxnreg parameter not supported on XPU backend")
    def test_autotune_random_seed_from_settings(self) -> None:
        # same autotune_random_seed setting -> same random sample
        first = self._autotune_and_record(autotune_random_seed=4242)
        second = self._autotune_and_record(autotune_random_seed=4242)
        self.assertEqual(first, second)

        # different autotune_random_seed settings -> different random samples
        first = self._autotune_and_record(autotune_random_seed=101)
        second = self._autotune_and_record(autotune_random_seed=102)
        self.assertNotEqual(first, second)


class TestAutotuneBestOfKSettings(TestCase):
    """Settings-only coverage for ``autotune_best_of_k`` (no GPU/backend
    dependency)."""

    def test_default_is_one(self) -> None:
        with without_env_var("HELION_AUTOTUNE_BEST_OF_K"):
            self.assertEqual(helion.Settings().autotune_best_of_k, 1)

    def test_env_var_override(self) -> None:
        with patch.dict(os.environ, {"HELION_AUTOTUNE_BEST_OF_K": "5"}, clear=False):
            self.assertEqual(helion.Settings().autotune_best_of_k, 5)

    def test_setting_override(self) -> None:
        self.assertEqual(helion.Settings(autotune_best_of_k=3).autotune_best_of_k, 3)

    def test_k_zero_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"autotune_best_of_k must be >= 1"):
            helion.Settings(autotune_best_of_k=0)

    def test_k_negative_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"autotune_best_of_k must be >= 1"):
            helion.Settings(autotune_best_of_k=-3)

    def test_effort_profile_preserves_positional_fields(self) -> None:
        profile = AutotuneEffortProfile(
            None,
            None,
            None,
            None,
            None,
            7,
            1.25,
        )

        self.assertEqual(profile.finishing_rounds, 7)
        self.assertEqual(profile.rebenchmark_threshold, 1.25)
        self.assertIsNone(profile.flash_structural_search)


class TestForcedAutotuneSeedCachePolicy(unittest.TestCase):
    @staticmethod
    def _search(*, cute_flash: bool) -> BaseSearch:
        search = BaseSearch.__new__(BaseSearch)
        search._skip_cache = True
        search.config_spec = SimpleNamespace(
            cute_flash_search_enabled=cute_flash,
            cache_fingerprint_hash=Mock(return_value="spec"),
        )
        search.settings = SimpleNamespace(
            autotune_search_acf=None,
            autotune_best_available_max_cache_scan=16,
        )
        search.log = Mock()
        search._get_current_hardware_and_specialization = Mock(
            return_value=("gpu", "specialization")
        )
        return search

    def test_force_bypasses_warm_start_only_for_cute_flash(self) -> None:
        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=False),
            patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path("/cache"),
            ),
            patch(
                "helion.autotuner.local_cache.iter_cache_entries",
                return_value=iter(()),
            ) as scan,
            patch(
                "helion.autotuner.remote_cache._load_remote_backend_if_configured",
                return_value=None,
            ),
        ):
            self.assertEqual(
                self._search(cute_flash=False)._find_similar_cached_configs(4), []
            )
            scan.assert_called_once_with(Path("/cache"), max_scan=16)

            scan.reset_mock()
            self.assertEqual(
                self._search(cute_flash=True)._find_similar_cached_configs(4), []
            )
            scan.assert_not_called()


class TestCuteFlashSearchPolicyCacheKey(unittest.TestCase):
    @staticmethod
    def _search(
        *,
        effort: str = "full",
        budget_seconds: int | None = None,
        max_generations: int = 20,
        initial_population_strategy: InitialPopulationStrategy = (
            InitialPopulationStrategy.FROM_RANDOM
        ),
        random_seed: int = 0,
        seed_configs: object = None,
        best_available_max_configs: int = 20,
        config_filter: object = None,
        baseline_fn: object = None,
        baseline_atol: float | None = None,
        baseline_rtol: float | None = None,
        best_of_k: int = 1,
        precompile_jobs: int | None = None,
        with_torch_compile_fusion: bool = False,
        search_acf: list[str] | None = None,
        compiler_seed_timeout_retry_repetitions: int | None = None,
    ) -> LFBOTreeSearch:
        settings = helion.Settings(
            backend="cute",
            autotune_effort=effort,
            autotune_budget_seconds=budget_seconds,
            autotune_random_seed=random_seed,
            autotune_seed_configs=seed_configs,
            autotune_best_available_max_configs=best_available_max_configs,
            autotune_config_filter=config_filter,  # type: ignore[arg-type]
            autotune_baseline_fn=baseline_fn,  # type: ignore[arg-type]
            autotune_baseline_atol=baseline_atol,
            autotune_baseline_rtol=baseline_rtol,
            autotune_best_of_k=best_of_k,
            autotune_precompile_jobs=precompile_jobs,
            autotune_with_torch_compile_fusion=with_torch_compile_fusion,
            autotune_search_acf=search_acf or [],
        )
        profile = get_effort_profile(effort)  # type: ignore[arg-type]
        pattern = profile.lfbo_pattern_search
        assert pattern is not None
        search = object.__new__(LFBOTreeSearch)
        search.settings = settings
        search.config_spec = SimpleNamespace(  # type: ignore[assignment]
            compiler_seed_configs=[],
            compiler_seed_timeout_retry_repetitions=(
                compiler_seed_timeout_retry_repetitions
            ),
            backend=SimpleNamespace(config_value_priors_version=1),
            backend_name="cute",
            cute_flash_search_enabled=True,
        )
        search.initial_population = pattern.initial_population
        search.copies = pattern.copies
        search.max_generations = max_generations
        search.initial_population_strategy = initial_population_strategy
        search.best_available_pad_random = pattern.best_available_pad_random
        search.num_neighbors_cap = -1
        search.finishing_rounds = profile.finishing_rounds
        search.min_improvement_delta = 0.001
        search.num_neighbors = 300
        search.radius = 2
        search.frac_selected = 0.1
        search.quantile = 0.1
        search.patience = 1
        search.similarity_penalty = 1.0
        search.compile_timeout_lower_bound = pattern.compile_timeout_lower_bound
        search.compile_timeout_quantile = pattern.compile_timeout_quantile
        search.flash_structural_search = profile.flash_structural_search
        search._cute_flash_lane_policy_enabled = (
            search.config_spec.cute_flash_search_enabled
            and search.flash_structural_search is not None
        )
        search._flash_promoted_path_limit = search.copies
        search._flash_family_probe_path_limit = 0
        if search._cute_flash_lane_policy_enabled:
            search._flash_promoted_path_limit = 17
            search._flash_family_probe_path_limit = 18
            search.copies = 18
        return search

    def _hash(self, search: BaseSearch, *, enabled: bool = True) -> str:
        return _cute_flash_search_policy_hash(
            search,
            cute_flash_search_enabled=enabled,
        )

    def test_non_flash_preserves_legacy_empty_policy(self) -> None:
        full = self._search()
        quick = self._search(effort="quick", max_generations=5)
        self.assertEqual(self._hash(full, enabled=False), "")
        self.assertEqual(self._hash(quick, enabled=False), "")

    def test_config_generation_policy_version_covers_flash_searches(self) -> None:
        def random_search(*, cute_flash: bool) -> RandomSearch:
            template = self._search(effort="quick", max_generations=5)
            search = object.__new__(RandomSearch)
            search.settings = template.settings
            search.config_spec = copy.deepcopy(template.config_spec)
            search.config_spec.cute_flash_search_enabled = cute_flash
            search.count = 100
            search._benchmark_provider_cls = LocalBenchmarkProvider
            return search

        full = self._search()
        quick = self._search(effort="quick", max_generations=5)
        random_flash = random_search(cute_flash=True)
        for search in (full, quick, random_flash):
            with self.subTest(search=type(search).__name__):
                policy = search.cache_policy()
                assert policy is not None
                algorithm = policy["algorithm"]
                assert isinstance(algorithm, dict)
                self.assertEqual(
                    algorithm["cute_flash_config_generation_policy_version"], 4
                )

        full_policy = full.cache_policy()
        quick_policy = quick.cache_policy()
        random_policy = random_flash.cache_policy()
        assert full_policy is not None
        assert quick_policy is not None
        assert random_policy is not None
        full_algorithm = full_policy["algorithm"]
        quick_algorithm = quick_policy["algorithm"]
        random_algorithm = random_policy["algorithm"]
        assert isinstance(full_algorithm, dict)
        assert isinstance(quick_algorithm, dict)
        assert isinstance(random_algorithm, dict)
        self.assertEqual(
            full_algorithm["cute_flash_lane_policy_version"],
            14,
        )
        self.assertEqual(
            full_algorithm["cute_flash_terminal_coordinate_refinement"],
            {
                "schema_version": 2,
                "policy_version": 2,
                "coordinate_policy": (
                    "same_leaf_full_surface_normalized_coordinate_v2"
                ),
                "rounds": 2,
                "beam_width": 4,
                "radius": 2,
                "minimum_improvement_fraction": 0.001,
                "measurement_policy": "mirrored_rotating_batched_wall_v2",
                "round_target_ms": 200.0,
                "confirmation_target_ms": 5000.0,
            },
        )
        self.assertNotIn("cute_flash_lane_policy_version", quick_algorithm)
        self.assertNotIn("cute_flash_lane_policy_version", random_algorithm)

        non_flash = self._search(effort="quick", max_generations=5)
        non_flash.config_spec.cute_flash_search_enabled = False
        non_flash._cute_flash_lane_policy_enabled = False
        random_non_flash = random_search(cute_flash=False)
        for search in (non_flash, random_non_flash):
            with self.subTest(non_flash_search=type(search).__name__):
                policy = search.cache_policy()
                assert policy is not None
                algorithm = policy["algorithm"]
                assert isinstance(algorithm, dict)
                self.assertNotIn(
                    "cute_flash_config_generation_policy_version", algorithm
                )
                self.assertEqual(algorithm, search._algorithm_cache_policy())

    def test_lane_policy_version_is_cute_flash_only(self) -> None:
        search = self._search()
        search.config_spec.cute_flash_search_enabled = False
        search._cute_flash_lane_policy_enabled = False
        non_flash = search._algorithm_cache_policy()
        self.assertEqual(non_flash["lfbo_version"], 1)
        self.assertNotIn("cute_flash_lane_policy_version", non_flash)
        self.assertIsNone(non_flash["flash_structural_search"])

        search.config_spec.cute_flash_search_enabled = True
        search._cute_flash_lane_policy_enabled = True
        cute_flash_policy = search._algorithm_cache_policy()
        self.assertEqual(cute_flash_policy["lfbo_version"], 1)
        self.assertEqual(cute_flash_policy["cute_flash_lane_policy_version"], 14)
        self.assertEqual(cute_flash_policy["cute_flash_starting_path_limit"], 17)
        self.assertEqual(cute_flash_policy["cute_flash_family_probe_path_limit"], 18)
        self.assertEqual(cute_flash_policy["cute_flash_maximum_path_capacity"], 18)
        self.assertEqual(
            cute_flash_policy["flash_structural_search"],
            search.flash_structural_search,
        )
        self.assertEqual(
            {
                k: v
                for k, v in cute_flash_policy.items()
                if k
                not in {
                    "cute_flash_lane_policy_version",
                    "cute_flash_starting_path_limit",
                    "cute_flash_family_probe_path_limit",
                    "cute_flash_maximum_path_capacity",
                    "cute_flash_terminal_coordinate_refinement",
                    "flash_structural_search",
                }
            },
            {k: v for k, v in non_flash.items() if k != "flash_structural_search"},
        )

        search.flash_structural_search = None
        search._cute_flash_lane_policy_enabled = False
        quick_policy = search._algorithm_cache_policy()
        self.assertNotIn("cute_flash_lane_policy_version", quick_policy)
        self.assertIsNone(quick_policy["flash_structural_search"])
        self.assertEqual(quick_policy["lfbo_version"], 1)

    def test_unlimited_and_finite_family_retention_have_distinct_cache_keys(
        self,
    ) -> None:
        finite = self._search()
        unlimited = self._search()
        assert unlimited.flash_structural_search is not None
        unlimited.flash_structural_search = replace(
            unlimited.flash_structural_search, retained_families=None
        )

        self.assertNotEqual(self._hash(unlimited), self._hash(finite))

    def test_effective_search_breadth_changes_policy(self) -> None:
        baseline = self._hash(self._search())
        variants = (
            self._search(effort="quick", max_generations=5),
            self._search(budget_seconds=60),
            self._search(max_generations=10),
            self._search(
                initial_population_strategy=(
                    InitialPopulationStrategy.FROM_BEST_AVAILABLE
                )
            ),
            self._search(best_of_k=2),
            self._search(precompile_jobs=2),
            self._search(with_torch_compile_fusion=True),
        )
        self.assertTrue(all(self._hash(variant) != baseline for variant in variants))

    def test_random_seed_does_not_change_policy(self) -> None:
        self.assertEqual(
            self._hash(self._search(random_seed=1)),
            self._hash(self._search(random_seed=2)),
        )

    def test_seed_configs_and_best_available_limits_change_policy(self) -> None:
        baseline = self._hash(self._search())
        self.assertNotEqual(
            baseline,
            self._hash(
                self._search(seed_configs=helion.Config(block_sizes=[128, 128]))
            ),
        )
        self.assertNotEqual(
            baseline,
            self._hash(self._search(best_available_max_configs=7)),
        )

    def test_compiler_seed_timeout_retry_changes_policy(self) -> None:
        self.assertNotEqual(
            self._hash(self._search()),
            self._hash(self._search(compiler_seed_timeout_retry_repetitions=3)),
        )

    def test_custom_filter_and_unknown_search_disable_cache_reuse(self) -> None:
        filtered = self._search(config_filter=lambda config: config)
        other_filtered = self._search(config_filter=lambda config: config)
        self.assertEqual(self._hash(filtered), self._hash(filtered))
        self.assertNotEqual(self._hash(filtered), self._hash(other_filtered))
        self.assertIs(filtered._search_policy_cacheable, False)

        class CustomLFBOTreeSearch(LFBOTreeSearch):
            pass

        custom = object.__new__(CustomLFBOTreeSearch)
        custom.__dict__.update(self._search().__dict__)
        other_custom = object.__new__(CustomLFBOTreeSearch)
        other_custom.__dict__.update(self._search().__dict__)
        self.assertEqual(self._hash(custom), self._hash(custom))
        self.assertNotEqual(self._hash(custom), self._hash(other_custom))

    def test_baseline_function_and_tolerances_change_policy(self) -> None:
        def make_baseline(offset):
            def baseline(value):
                return value + offset

            return baseline

        def baseline_increment(value):
            return value + 1

        baseline_identity = make_baseline(0)
        identity = self._search(baseline_fn=baseline_identity)
        other_closure = self._search(baseline_fn=make_baseline(1))
        increment = self._search(baseline_fn=baseline_increment)
        atol = self._search(
            baseline_fn=baseline_identity,
            baseline_atol=1e-2,
        )
        rtol = self._search(
            baseline_fn=baseline_identity,
            baseline_rtol=1e-2,
        )

        self.assertNotEqual(self._hash(identity), self._hash(other_closure))
        self.assertNotEqual(self._hash(identity), self._hash(increment))
        self.assertNotEqual(self._hash(identity), self._hash(atol))
        self.assertNotEqual(self._hash(identity), self._hash(rtol))

    def test_baseline_function_runtime_globals_change_policy(self) -> None:
        global _CACHE_POLICY_BASELINE_SCALE

        def baseline(value):
            return value * _CACHE_POLICY_BASELINE_SCALE

        original = _CACHE_POLICY_BASELINE_SCALE
        try:
            _CACHE_POLICY_BASELINE_SCALE = 1
            first = self._hash(self._search(baseline_fn=baseline))
            _CACHE_POLICY_BASELINE_SCALE = 2
            second = self._hash(self._search(baseline_fn=baseline))
        finally:
            _CACHE_POLICY_BASELINE_SCALE = original

        self.assertNotEqual(first, second)

    def test_unfingerprintable_baseline_helper_disables_cache(self) -> None:
        global _CACHE_POLICY_DYNAMIC_HELPER

        def baseline(value):
            return _CACHE_POLICY_DYNAMIC_HELPER(value)  # type: ignore[operator]

        namespace: dict[str, object] = {}
        exec(
            compile(
                "def dynamic_helper(value):\n    return value\n",
                "<dynamic-helper>",
                "exec",
            ),
            namespace,
        )
        original = _CACHE_POLICY_DYNAMIC_HELPER
        try:
            _CACHE_POLICY_DYNAMIC_HELPER = namespace["dynamic_helper"]
            search = self._search(baseline_fn=baseline)
            self._hash(search)
        finally:
            _CACHE_POLICY_DYNAMIC_HELPER = original

        self.assertIs(search._search_policy_cacheable, False)

    def test_recursive_baseline_closure_disables_cache(self) -> None:
        recursive_value: list[object] = []
        recursive_value.append(recursive_value)

        def baseline(value):
            if recursive_value:
                return value
            raise AssertionError("unreachable")

        search = self._search(baseline_fn=baseline)
        first = self._hash(search)
        second = self._hash(search)

        self.assertEqual(first, second)
        self.assertIs(search._search_policy_cacheable, False)

    def test_advanced_controls_file_contents_change_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            acf = Path(tmp) / "controls.acf"
            acf.write_text("first")
            first = self._hash(self._search(search_acf=[str(acf)]))
            acf.write_text("second")
            second = self._hash(self._search(search_acf=[str(acf)]))

        self.assertNotEqual(first, second)

    def test_unsupported_baseline_disables_cache_reads_and_writes(self) -> None:
        class CallableBaseline:
            def __call__(self, value):
                return value

        search = self._search(baseline_fn=CallableBaseline())
        self._hash(search)
        self.assertIs(search._search_policy_cacheable, False)
        search.log = Mock()

        cache = object.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache.get = Mock(return_value=helion.Config(block_sizes=[1]))  # type: ignore[method-assign]
        cache.put = Mock()  # type: ignore[method-assign]
        cache._run_autotune_trials = Mock(  # type: ignore[method-assign]
            return_value=helion.Config(block_sizes=[2])
        )

        with patch("helion.autotuner.base_cache.should_skip_cache", return_value=False):
            result = cache.autotune()

        self.assertEqual(result, helion.Config(block_sizes=[2]))
        cache.get.assert_not_called()
        cache.put.assert_not_called()
        cache._run_autotune_trials.assert_called_once_with(skip_cache=True)

    def test_llm_seeded_policy_does_not_construct_child_searches(self) -> None:
        search = object.__new__(LLMSeededLFBOTreeSearch)
        search._make_llm_search = Mock(
            side_effect=AssertionError("cache policy constructed the LLM stage")
        )
        search._make_second_stage_search = Mock(
            side_effect=AssertionError("cache policy constructed the second stage")
        )

        self.assertIsNone(search._algorithm_cache_policy())
        search._make_llm_search.assert_not_called()
        search._make_second_stage_search.assert_not_called()

    def test_random_search_policy_rejects_nonlocal_benchmark_provider(self) -> None:
        search = object.__new__(RandomSearch)
        search.count = 100
        search._benchmark_provider_cls = object
        self.assertIsNone(search._algorithm_cache_policy())

        search._benchmark_provider_cls = LocalBenchmarkProvider
        self.assertEqual(
            search._algorithm_cache_policy(), {"random_version": 1, "count": 100}
        )

    def test_registered_de_policy_covers_direct_search_knobs(self) -> None:
        search = object.__new__(DifferentialEvolutionSearch)
        search.population_size = 40
        search.max_generations = 20
        search.crossover_rate = 0.8
        search.immediate_update = True
        search.min_improvement_delta = None
        search.patience = None
        search.initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        search.best_available_pad_random = True
        search.finishing_rounds = 1
        search.compile_timeout_lower_bound = 30.0
        search.compile_timeout_quantile = 0.9
        policy = search._algorithm_cache_policy()
        self.assertEqual(policy["population_size"], 40)
        self.assertEqual(policy["crossover_rate"], 0.8)
        self.assertIs(policy["immediate_update"], True)

    def test_registered_search_constructor_knobs_are_classified(self) -> None:
        classes = (
            DESurrogateHybrid,
            LFBOPatternSearch,
            LFBOTreeSearch,
            LLMGuidedSearch,
            LLMSeededSearch,
            LLMSeededLFBOTreeSearch,
            DifferentialEvolutionSearch,
            FiniteSearch,
            PatternSearch,
            RandomSearch,
        )
        aliases = {"benchmark_provider_cls": "benchmark_provider"}
        secrets = {
            LLMGuidedSearch: {"api_key"},
        }
        inherited_fields = {
            "_cute_flash_lane_policy_enabled",
            "_flash_family_probe_path_limit",
            "_flash_promoted_path_limit",
            "immediate_update",
            "population_size",
            "provider",
            "second_stage_algorithm",
        }

        for search_cls in classes:
            with self.subTest(search_cls=search_cls.__name__):
                parameters = {
                    name
                    for name in inspect.signature(search_cls.__init__).parameters
                    if name not in {"self", "kernel", "args"}
                }
                search = object.__new__(search_cls)
                for name in parameters | inherited_fields:
                    setattr(search, name, 1)
                if hasattr(search, "flash_structural_search"):
                    search.flash_structural_search = get_effort_profile(
                        "full"
                    ).flash_structural_search
                search._benchmark_provider_cls = LocalBenchmarkProvider
                search.configs = []
                search.provider = "provider"
                search.model = "model"
                policy = search._algorithm_cache_policy()
                if isinstance(search, LLMSeededSearch):
                    self.assertIsNone(policy)
                    continue
                assert policy is not None
                policy_keys = set(policy)
                classified = {
                    aliases.get(name, name)
                    for name in parameters - secrets.get(search_cls, set())
                }
                self.assertLessEqual(classified, policy_keys)

    def test_strict_key_includes_policy_and_preserves_legacy_default(self) -> None:
        from helion.autotuner.base_cache import StrictAutotuneCacheKey

        common = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "abc",
            "hardware": "B200",
            "runtime_name": "13.0",
            "backend": "cute",
            "config_spec_hash": "h1",
            "extra_cache_key": "",
            "helion_key": "H",
            "torch_key": "T",
            "triton_key": "R",
        }
        legacy = StrictAutotuneCacheKey(**common)
        best_of_five = StrictAutotuneCacheKey(**common, best_of_k=5)
        first = StrictAutotuneCacheKey(**common, search_policy_hash="first")
        second = StrictAutotuneCacheKey(**common, search_policy_hash="second")
        self.assertEqual(
            repr(legacy),
            "StrictAutotuneCacheKey(specialization_key=(), extra_results=(), "
            "kernel_source_hash='abc', hardware='B200', runtime_name='13.0', "
            "backend='cute', config_spec_hash='h1', extra_cache_key='', "
            "helion_key='H', torch_key='T', triton_key='R')",
        )
        self.assertEqual(
            legacy.stable_hash(),
            "b92201d3cff5d3b9c92a40d17ac8e98022d86261bfdd0b638d3296a0c0908a4d",
        )
        self.assertIn("best_of_k=5", repr(best_of_five))
        self.assertNotEqual(legacy.stable_hash(), best_of_five.stable_hash())
        self.assertNotEqual(first.stable_hash(), second.stable_hash())

    def test_policy_hash_is_structural_and_legacy_default_is_unchanged(self) -> None:
        from helion.autotuner.base_cache import LooseAutotuneCacheKey

        common = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "abc",
            "hardware": "B200",
            "runtime_name": "13.0",
            "backend": "cute",
            "config_spec_hash": "h1",
        }
        legacy = LooseAutotuneCacheKey(**common, extra_cache_key="")
        policy = LooseAutotuneCacheKey(
            **common,
            extra_cache_key="",
            search_policy_hash="policy-v1",
        )
        suffix_lookalike = LooseAutotuneCacheKey(
            **common,
            extra_cache_key="search_policy_hash='policy-v1'",
        )

        self.assertNotIn("search_policy_hash", repr(legacy))
        self.assertIn("search_policy_hash='policy-v1'", repr(policy))
        self.assertNotEqual(legacy.stable_hash(), policy.stable_hash())
        self.assertNotEqual(suffix_lookalike.stable_hash(), policy.stable_hash())


@onlyBackends(["triton"])
class TestAutotuneBestOfK(RefEagerTestDisabled, TestCase):
    """Best-of-K multi-seed autotune selection — cache key + K-loop coverage.

    Covers:
      - K = 1 leaves the cache hash byte-identical with the pre-feature
        repr (no field appended to the dataclass repr).
      - K > 1 differentiates the cache hash structurally; the K value
        appears as a field on the key dataclass, not concatenated into
        ``extra_cache_key``.
      - K > 1 with no ``_autotuner_factory`` wired raises a clear error.
      - The K-loop runs K trials with deterministic per-trial seeds,
        and the winner is picked by the **final rebench** (not the
        per-trial ``best_perf_so_far`` low-water mark).
      - The autotuner reference on the cache is restored to the
        original after the loop.
    """

    def test_cache_key_byte_identical_when_k_is_one(self) -> None:
        """K=1 cache hash must match the bytes produced by the original
        ``LooseAutotuneCacheKey`` repr (before ``best_of_k`` was added)."""
        from helion.autotuner.base_cache import LooseAutotuneCacheKey

        # Build a key with K=1 and one with K=5; the K=1 repr must equal
        # the repr that omits ``best_of_k`` entirely.
        common_kwargs = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "abc",
            "hardware": "B200",
            "runtime_name": "12.6",
            "backend": "triton",
            "config_spec_hash": "h1",
            "extra_cache_key": "",
        }
        k1 = LooseAutotuneCacheKey(**common_kwargs, best_of_k=1)
        k5 = LooseAutotuneCacheKey(**common_kwargs, best_of_k=5)
        # K=1 repr matches a manually-constructed repr without best_of_k.
        expected_k1_repr = (
            "LooseAutotuneCacheKey("
            "specialization_key=(), extra_results=(), "
            "kernel_source_hash='abc', hardware='B200', "
            "runtime_name='12.6', backend='triton', "
            "config_spec_hash='h1', extra_cache_key='')"
        )
        self.assertEqual(repr(k1), expected_k1_repr)
        # K=5 includes the field in the repr.
        self.assertIn("best_of_k=5", repr(k5))
        # Hashes differ structurally.
        self.assertNotEqual(k1.stable_hash(), k5.stable_hash())

    def test_cache_key_does_not_alias_extra_cache_key(self) -> None:
        """A K=1 key with ``extra_cache_key`` carrying a literal
        ``;best_of_k=5`` suffix must NOT collide with a K=5 key whose
        ``extra_cache_key`` is empty — i.e. the K must be a structural
        field, not folded into the string."""
        from helion.autotuner.base_cache import LooseAutotuneCacheKey

        k1_aliased = LooseAutotuneCacheKey(
            specialization_key=(),
            extra_results=(),
            kernel_source_hash="abc",
            hardware="B200",
            runtime_name="12.6",
            backend="triton",
            config_spec_hash="h1",
            extra_cache_key="foo;best_of_k=5",
            best_of_k=1,
        )
        k5 = LooseAutotuneCacheKey(
            specialization_key=(),
            extra_results=(),
            kernel_source_hash="abc",
            hardware="B200",
            runtime_name="12.6",
            backend="triton",
            config_spec_hash="h1",
            extra_cache_key="foo",
            best_of_k=5,
        )
        self.assertNotEqual(k1_aliased.stable_hash(), k5.stable_hash())

    def test_strict_cache_key_tracks_nondefault_best_of_k(self) -> None:
        from helion.autotuner.base_cache import StrictAutotuneCacheKey

        common_kwargs = {
            "specialization_key": (),
            "extra_results": (),
            "kernel_source_hash": "abc",
            "hardware": "B200",
            "runtime_name": "12.6",
            "backend": "triton",
            "config_spec_hash": "h1",
            "extra_cache_key": "",
            "helion_key": "helion",
            "torch_key": "torch",
            "triton_key": "triton",
        }
        k1 = StrictAutotuneCacheKey(**common_kwargs, best_of_k=1)
        k5 = StrictAutotuneCacheKey(**common_kwargs, best_of_k=5)
        expected_k1_repr = (
            "StrictAutotuneCacheKey("
            "specialization_key=(), extra_results=(), "
            "kernel_source_hash='abc', hardware='B200', "
            "runtime_name='12.6', backend='triton', "
            "config_spec_hash='h1', extra_cache_key='', "
            "helion_key='helion', torch_key='torch', triton_key='triton')"
        )

        self.assertEqual(repr(k1), expected_k1_repr)
        self.assertIn("best_of_k=5", repr(k5))
        self.assertNotEqual(k1.stable_hash(), k5.stable_hash())

    def test_best_of_k_gt_1_without_factory_raises(self) -> None:
        """The bare ``Cache(autotuner)`` constructor must reject K>1 at
        run time rather than silently fall back to single-trial."""
        from helion.autotuner.local_cache import LocalAutotuneCache

        @helion.kernel(autotune_best_of_k=3, autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)

        # Build a cache with NO autotuner_factory wired; this models the
        # external ``Cache(autotuner)`` constructor path.
        class _MinimalSearch:
            def __init__(self):
                self.kernel = bound
                self.settings = bound.settings
                self.args = args
                self.best_perf_so_far = math.inf
                self._skip_cache = False

                class _Log:
                    def __call__(self, *a, **kw):
                        pass

                    def reset(self):
                        pass

                    def warning(self, *a, **kw):
                        pass

                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                return None  # never reached: K>1 must raise before this

        cache = LocalAutotuneCache(_MinimalSearch())  # no autotuner_factory
        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=True),
            self.assertRaisesRegex(
                RuntimeError,
                r"autotune_best_of_k > 1 requires a registered _autotuner_factory",
            ),
        ):
            cache.autotune()

    def test_single_trial_receives_effective_cache_bypass(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache

        config = helion.Config(block_sizes=[16])
        search = SimpleNamespace(
            settings=helion.Settings(autotune_best_of_k=1),
            autotune=Mock(return_value=config),
        )
        cache = object.__new__(LocalAutotuneCache)
        cache.autotuner = search

        self.assertEqual(cache._run_autotune_trials(skip_cache=True), config)
        search.autotune.assert_called_once_with(skip_cache=True)

    def test_k_loop_runs_k_trials_with_deterministic_seeds(self) -> None:
        """The K-loop runs K trials with seeds ``base + i``."""
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        seeds_seen: list[int] = []
        skip_cache_seen: list[bool] = []
        # ``block_sizes`` must be powers of two; pick four distinct values.
        trial_configs = [Config(block_sizes=[16, 1 << (3 + i)]) for i in range(4)]
        # Low-water perfs (per-trial best_perf_so_far) and rebench perfs
        # disagree on the winner: low-water best is index 2, rebench
        # best is index 3.
        low_water_perfs = [3.0, 5.0, 1.0, 2.0]
        rebench_perfs = [3.0, 5.0, 8.0, 2.0]
        trial_idx = {"n": 0}

        class MockTrialSearch:
            def __init__(self, bound_kernel, args, **kwargs):
                self.kernel = bound_kernel
                self.settings = bound_kernel.settings
                self.args = args
                self.best_perf_so_far = math.inf
                self._skip_cache = False

                class _Log:
                    def __call__(self, *a, **kw):
                        pass

                    def reset(self):
                        pass

                    def warning(self, *a, **kw):
                        pass

                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                i = trial_idx["n"]
                seeds_seen.append(self.settings.autotune_random_seed)
                skip_cache_seen.append(skip_cache)
                self.best_perf_so_far = low_water_perfs[i]
                cfg = trial_configs[i]
                trial_idx["n"] += 1
                return cfg

        def mock_autotuner_fn(bound_kernel, args, **kwargs):
            def factory():
                return MockTrialSearch(bound_kernel, args, **kwargs)

            return LocalAutotuneCache(factory(), autotuner_factory=factory)

        @helion.kernel(
            autotuner_fn=mock_autotuner_fn,
            autotune_best_of_k=4,
            autotune_random_seed=100,
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)

        # Patch the final-rebench step so we can return the desired perfs
        # without needing a real benchmark_provider.
        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=True),
            patch.object(
                LocalAutotuneCache,
                "_rebench_trial_configs",
                lambda self, configs: rebench_perfs,
            ),
        ):
            picked = bound.autotune(args)

        # K trials ran.
        self.assertEqual(trial_idx["n"], 4)
        # Deterministic seeds: base + i.
        self.assertEqual(seeds_seen, [100, 101, 102, 103])
        self.assertEqual(skip_cache_seen, [True, True, True, True])
        # Winner is picked by REBENCH (index 3), not by low-water (index 2).
        rebench_winner = rebench_perfs.index(min(rebench_perfs))
        low_water_winner = low_water_perfs.index(min(low_water_perfs))
        self.assertNotEqual(rebench_winner, low_water_winner)
        self.assertEqual(picked, trial_configs[rebench_winner])

    def test_k_loop_rejects_all_failed_final_rebenchmarks(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        settings = helion.Settings(
            autotune_best_of_k=2,
            autotune_random_seed=10,
            autotune_log_level=0,
        )
        search = SimpleNamespace(
            settings=settings,
            log=Mock(),
            config_spec=_cute_flash_test_config_spec(),
        )
        cache = LocalAutotuneCache.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache._autotuner_factory = Mock()
        configs = [Config(block_sizes=[16]), Config(block_sizes=[32])]

        with (
            patch.object(
                cache,
                "_run_one_trial",
                side_effect=[(configs[0], 1.0), (configs[1], 2.0)],
            ),
            patch.object(cache, "_release_trial_state"),
            patch.object(
                cache,
                "_rebench_trial_configs",
                return_value=[math.inf, math.inf],
            ) as rebench,
            self.assertRaises(exc.NoConfigFound),
        ):
            cache._run_autotune_trials()

        self.assertEqual(rebench.call_count, 2)

    def test_cute_k_loop_retries_partial_finalist_failure(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        settings = helion.Settings(
            autotune_best_of_k=2,
            autotune_random_seed=10,
            autotune_log_level=0,
        )
        search = SimpleNamespace(
            settings=settings,
            log=Mock(),
            config_spec=_cute_flash_test_config_spec(),
        )
        cache = LocalAutotuneCache.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache._autotuner_factory = Mock()
        configs = [Config(block_sizes=[16]), Config(block_sizes=[32])]

        with (
            patch.object(
                cache,
                "_run_one_trial",
                side_effect=[(configs[0], 1.0), (configs[1], 2.0)],
            ),
            patch.object(cache, "_release_trial_state"),
            patch.object(
                cache,
                "_rebench_trial_configs",
                side_effect=([math.inf, 2.0], [1.0, math.inf]),
            ) as rebench,
        ):
            selected = cache._run_autotune_trials()

        self.assertEqual(selected, configs[0])
        self.assertEqual(rebench.call_count, 2)

    def test_cute_k_loop_skips_one_failed_trial(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        settings = helion.Settings(
            autotune_best_of_k=2,
            autotune_random_seed=10,
            autotune_log_level=0,
        )
        search = SimpleNamespace(
            settings=settings,
            log=Mock(),
            config_spec=_cute_flash_test_config_spec(),
        )
        cache = LocalAutotuneCache.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache._autotuner_factory = Mock()
        winner = Config(block_sizes=[32])

        with (
            patch.object(
                cache,
                "_run_one_trial",
                side_effect=[exc.NoConfigFound(), (winner, 2.0)],
            ),
            patch.object(cache, "_release_trial_state"),
            patch.object(
                cache,
                "_rebench_trial_configs",
                return_value=[1.5],
            ) as rebench,
        ):
            selected = cache._run_autotune_trials()

        self.assertEqual(selected, winner)
        rebench.assert_called_once_with([winner])

    def test_non_cute_k_loop_preserves_transient_finalist_fallback(self) -> None:
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        settings = helion.Settings(
            autotune_best_of_k=2,
            autotune_random_seed=10,
            autotune_log_level=0,
        )
        search = SimpleNamespace(
            settings=settings,
            log=Mock(),
            config_spec=SimpleNamespace(cute_flash_search_enabled=False),
        )
        cache = LocalAutotuneCache.__new__(LocalAutotuneCache)
        cache.autotuner = search
        cache.args = ()
        cache._autotuner_factory = Mock()
        configs = [Config(block_sizes=[16]), Config(block_sizes=[32])]

        with (
            patch.object(
                cache,
                "_run_one_trial",
                side_effect=[(configs[0], 1.0), (configs[1], 2.0)],
            ),
            patch.object(cache, "_release_trial_state"),
            patch.object(
                cache,
                "_rebench_trial_configs",
                return_value=[math.inf, math.inf],
            ) as rebench,
        ):
            selected = cache._run_autotune_trials()

        self.assertEqual(selected, configs[0])
        rebench.assert_called_once()

    def test_k_loop_restores_autotuner_and_settings(self) -> None:
        """After the K-loop, ``cache.autotuner`` must equal the original
        instance, and the mutated settings must be restored to base."""
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        cfg = Config(block_sizes=[16, 32])

        class MockSearch:
            def __init__(self, bound_kernel, args, **kwargs):
                self.kernel = bound_kernel
                self.settings = bound_kernel.settings
                self.args = args
                self.best_perf_so_far = math.inf
                self._skip_cache = False

                class _Log:
                    def __call__(self, *a, **kw):
                        pass

                    def reset(self):
                        pass

                    def warning(self, *a, **kw):
                        pass

                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                # Simulate the adaptive-timeout mutation that the real
                # BaseSearch does inside _prepare()/set_adaptive_compile_timeout.
                self.settings.autotune_compile_timeout = 5
                self.best_perf_so_far = 1.0
                return cfg

        original_autotuner_ref = {"r": None}

        def mock_autotuner_fn(bound_kernel, args, **kwargs):
            def factory():
                return MockSearch(bound_kernel, args, **kwargs)

            inner = factory()
            original_autotuner_ref["r"] = inner
            return LocalAutotuneCache(inner, autotuner_factory=factory)

        @helion.kernel(
            autotuner_fn=mock_autotuner_fn,
            autotune_best_of_k=3,
            autotune_random_seed=100,
            autotune_compile_timeout=60,
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)

        with (
            patch("helion.autotuner.base_cache.should_skip_cache", return_value=True),
            patch.object(
                LocalAutotuneCache,
                "_rebench_trial_configs",
                lambda self, configs: [1.0] * len(configs),
            ),
        ):
            captured_cache = bound.settings.autotuner_fn(bound, args)
            self.assertIsNotNone(original_autotuner_ref["r"])
            self.assertIs(captured_cache.autotuner, original_autotuner_ref["r"])
            captured_cache.autotune()

        # After the K-loop, the cache's ``autotuner`` reference must be
        # restored to the original instance (no leaked trial swap).
        self.assertIs(captured_cache.autotuner, original_autotuner_ref["r"])
        # And the settings the autotuner mutated must be restored to base.
        self.assertEqual(bound.settings.autotune_compile_timeout, 60)
        self.assertEqual(bound.settings.autotune_random_seed, 100)

    def test_k_one_falls_through_to_single_trial(self) -> None:
        """With ``autotune_best_of_k=1`` the K-loop must not run; the
        cache calls the autotuner exactly once and returns its config.
        """
        from helion.autotuner.local_cache import LocalAutotuneCache
        from helion.runtime.config import Config

        call_count = {"n": 0}
        only_config = Config(block_sizes=[16, 32])

        class _Log:
            def __call__(self, *a, **kw):
                pass

            def reset(self):
                pass

            def warning(self, *a, **kw):
                pass

        class SingleTrialSearch:
            def __init__(self, bound_kernel, args, **kwargs):
                self.kernel = bound_kernel
                self.settings = bound_kernel.settings
                self.args = args
                self.best_perf_so_far = 1.0
                self._skip_cache = False
                self.log = _Log()

            def autotune(self, *, skip_cache: bool = False):
                call_count["n"] += 1
                return only_config

        def mock_autotuner_fn(bound_kernel, args, **kwargs):
            def factory():
                return SingleTrialSearch(bound_kernel, args, **kwargs)

            return LocalAutotuneCache(factory(), autotuner_factory=factory)

        @helion.kernel(
            autotuner_fn=mock_autotuner_fn,
            autotune_best_of_k=1,
            autotune_log_level=0,
        )
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8, 32], device=DEVICE),
            torch.randn([8, 32], device=DEVICE),
        )
        bound = add.bind(args)
        with patch("helion.autotuner.base_cache.should_skip_cache", return_value=True):
            picked = bound.autotune(args)
        self.assertEqual(call_count["n"], 1)
        self.assertEqual(picked, only_config)


@onlyBackends(["triton", "cute"])
class TestAutotuneCacheSelection(TestCase):
    """Selection of the autotune cache via HELION_AUTOTUNE_CACHE."""

    def _make_bound(self):
        @helion.kernel(autotune_baseline_fn=operator.add, autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([8], device=DEVICE),
            torch.randn([8], device=DEVICE),
        )
        return add.bind(args), args

    def test_autotune_cache_default_is_local(self):
        """Default (no env var set) -> LocalAutotuneCache."""
        with without_env_var("HELION_AUTOTUNE_CACHE"):
            bound, args = self._make_bound()
            with patch("torch.accelerator.synchronize", autospec=True) as sync:
                sync.return_value = None
                autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertIsInstance(autotuner, LocalAutotuneCache)
            self.assertNotIsInstance(autotuner, StrictLocalAutotuneCache)

    def test_autotune_cache_strict_selected_by_env(self):
        """HELION_AUTOTUNE_CACHE=StrictLocalAutotuneCache -> StrictLocalAutotuneCache."""
        with patch.dict(
            os.environ,
            {"HELION_AUTOTUNE_CACHE": "StrictLocalAutotuneCache"},
            clear=False,
        ):
            bound, args = self._make_bound()
            with patch("torch.accelerator.synchronize", autospec=True) as sync:
                sync.return_value = None
                autotuner = bound.settings.autotuner_fn(bound, args)
            self.assertIsInstance(autotuner, StrictLocalAutotuneCache)

    def test_autotune_cache_invalid_raises(self):
        """Invalid HELION_AUTOTUNE_CACHE value should raise a ValueError."""
        with patch.dict(
            os.environ, {"HELION_AUTOTUNE_CACHE": "InvalidCacheName"}, clear=False
        ):
            bound, args = self._make_bound()
            with patch("torch.accelerator.synchronize", autospec=True) as sync:
                sync.return_value = None
                with self.assertRaisesRegex(
                    ValueError, "Unknown HELION_AUTOTUNE_CACHE"
                ):
                    bound.settings.autotuner_fn(bound, args)


@onlyBackends(["triton", "cute"])
class TestAutotuneSeedConfigs(TestCase):
    """Tests for seeding initial autotune populations with user configs."""

    def _seed_config(self) -> helion.Config:
        if _get_backend() == "cute":
            return helion.Config(num_threads=[32])
        return helion.Config(num_warps=8)

    def _has_seed_config(self, configs: list[helion.Config]) -> bool:
        if _get_backend() == "cute":
            return any(config.num_threads == [32] for config in configs)
        return any(config.num_warps == 8 for config in configs)

    def _make_kernel_and_args(self, **kernel_kwargs):
        @helion.kernel(autotune_log_level=0, **kernel_kwargs)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )
        return add, args

    def _population_configs(self, search: PatternSearch) -> list[helion.Config]:
        return [
            search.config_gen.unflatten(flat)
            for flat in search._generate_initial_population_flat()
        ]

    def test_decorator_accepts_single_seed_config(self) -> None:
        seed_config = self._seed_config()
        add, _args = self._make_kernel_and_args(autotune_seed_configs=seed_config)

        self.assertEqual(add.settings.autotune_seed_configs, seed_config)
        self.assertEqual(add.configs, [])

    def test_random_initial_population_includes_seed_configs(self) -> None:
        seed_config = self._seed_config()
        add, args = self._make_kernel_and_args(autotune_seed_configs=[seed_config])
        bound = add.bind(args)
        search = PatternSearch(bound, args, initial_population=3)

        configs = self._population_configs(search)

        self.assertGreaterEqual(len(configs), 3)
        self.assertTrue(self._has_seed_config(configs))

    def test_best_available_initial_population_includes_seed_configs(self) -> None:
        seed_config = self._seed_config()
        add, args = self._make_kernel_and_args(autotune_seed_configs=[seed_config])
        bound = add.bind(args)
        search = PatternSearch(
            bound,
            args,
            initial_population_strategy=InitialPopulationStrategy.FROM_BEST_AVAILABLE,
        )

        with patch.object(BaseSearch, "_find_similar_cached_configs", return_value=[]):
            configs = self._population_configs(search)

        self.assertGreaterEqual(len(configs), 2)
        self.assertTrue(self._has_seed_config(configs))

    def test_random_initial_population_logs_invalid_seed_configs(self) -> None:
        seed_config = helion.Config.from_dict({"block_sizes": ["bad"]})
        add, args = self._make_kernel_and_args(autotune_seed_configs=[seed_config])
        bound = add.bind(args)
        search = PatternSearch(bound, args, initial_population=3)
        search.log = Mock()

        configs = self._population_configs(search)

        self.assertGreaterEqual(len(configs), 3)
        search.log.assert_called_once()
        self.assertIn(
            "Failed to transfer autotune seed config 1", search.log.call_args[0][0]
        )


@skipIfRefEager("Autotuning requires compilation, not supported in ref eager mode")
@onlyBackends(["triton"])
class TestConfigFilter(TestCase):
    """Tests for the autotune_config_filter setting."""

    def _make_kernel_and_args(self, **kernel_kwargs):
        @helion.kernel(autotune_log_level=0, **kernel_kwargs)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )
        return add, args

    def test_finite_search_accepts_benchmark_provider_factory(self) -> None:
        class CustomBenchmarkProvider(LocalBenchmarkProvider):
            def __init__(
                self, *args: object, context: object, **kwargs: object
            ) -> None:
                self.context = context
                super().__init__(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]

        configs = [
            helion.Config(block_sizes=[16], num_warps=4),
            helion.Config(block_sizes=[32], num_warps=4),
        ]
        add, args = self._make_kernel_and_args()
        context = object()
        provider_factory = functools.partial(
            CustomBenchmarkProvider,
            context=context,
        )
        search = FiniteSearch(
            add.bind(args),
            args,
            configs=configs,
            benchmark_provider_cls=provider_factory,
        )
        search._prepare()

        self.assertIsInstance(search.benchmark_provider, CustomBenchmarkProvider)
        self.assertIs(search.benchmark_provider.context, context)

    def test_autotune_config_filter_skips_filtered_configs(self) -> None:
        """Filtered configs produce status='filtered' and perf=inf."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=4)
        cfg3 = helion.Config(block_sizes=[64], num_warps=4)

        filtered_out: list[helion.Config] = []

        def my_filter(config: helion.Config) -> helion.Config | None:
            if config.get("block_sizes") == [32]:
                filtered_out.append(config)
                return None
            return config

        add, args = self._make_kernel_and_args(
            autotune_config_filter=my_filter, autotune_precompile=None
        )
        bound = add.bind(args)
        search = FiniteSearch(bound, args, configs=[cfg1, cfg2, cfg3])
        search._prepare()
        results = search.benchmark_batch([cfg1, cfg2, cfg3])

        # cfg2 should be filtered
        self.assertEqual(len(filtered_out), 1)
        self.assertEqual(filtered_out[0].get("block_sizes"), [32])

        statuses = {tuple(r.config.get("block_sizes", [])): r.status for r in results}
        self.assertEqual(statuses[(16,)], "ok")
        self.assertEqual(statuses[(32,)], "filtered")
        self.assertEqual(statuses[(64,)], "ok")

        perfs = {tuple(r.config.get("block_sizes", [])): r.perf for r in results}
        self.assertEqual(perfs[(32,)], float("inf"))

    def test_autotune_config_filter_affects_autotune_winner(self) -> None:
        """The autotuner never picks a filtered config as the winner."""
        # cfg_fast would normally win (smallest block = least work per kernel launch
        # in this trivial test), but we filter it out.
        cfg_fast = helion.Config(block_sizes=[16], num_warps=4)
        cfg_slow = helion.Config(block_sizes=[128], num_warps=4)

        def reject_small_blocks(config: helion.Config) -> helion.Config | None:
            return config if (config.get("block_sizes") or [0])[0] >= 64 else None

        add, args = self._make_kernel_and_args(
            autotune_config_filter=reject_small_blocks,
            # Filtering happens before compile/benchmark; skip the benchmark
            # worker subprocess (seconds of startup overhead).
            autotune_benchmark_subprocess=False,
        )
        bound = add.bind(args)
        search = FiniteSearch(bound, args, configs=[cfg_fast, cfg_slow])
        winner = search.autotune()
        # cfg_fast is filtered out, so cfg_slow must win
        self.assertEqual(winner.get("block_sizes"), [128])

    def test_autotune_config_filter_none_is_noop(self) -> None:
        """When autotune_config_filter=None (default), all configs are benchmarked normally."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=4)

        add, args = self._make_kernel_and_args(
            autotune_precompile=None
        )  # no autotune_config_filter
        bound = add.bind(args)
        search = FiniteSearch(bound, args, configs=[cfg1, cfg2])
        search._prepare()
        results = search.benchmark_batch([cfg1, cfg2])

        for result in results:
            self.assertNotEqual(result.status, "filtered")
            self.assertFalse(math.isinf(result.perf))

    def test_autotune_config_filter_can_override_config(self) -> None:
        """autotune_config_filter can return a modified Config to override values before benchmarking."""
        cfg1 = helion.Config(block_sizes=[16], num_warps=4)
        cfg2 = helion.Config(block_sizes=[32], num_warps=4)

        def override_num_warps(config: helion.Config) -> helion.Config | None:
            # Override num_warps to 2 for all configs
            return helion.Config.from_dict({**config.config, "num_warps": 2})

        add, args = self._make_kernel_and_args(
            autotune_config_filter=override_num_warps, autotune_precompile=None
        )
        bound = add.bind(args)
        search = FiniteSearch(bound, args, configs=[cfg1, cfg2])
        search._prepare()
        results = search.benchmark_batch([cfg1, cfg2])

        # All configs should run successfully (none filtered)
        for result in results:
            self.assertNotEqual(result.status, "filtered")
            self.assertFalse(math.isinf(result.perf))
        # The result configs should reflect the overridden values
        self.assertEqual(results[0].config.get("num_warps"), 2)
        self.assertEqual(results[1].config.get("num_warps"), 2)


class TestFiniteSearchWarmStart(TestCase):
    """Tests for helion.from_cache() — a CachedFiniteSearch autotuner_fn."""

    def _make_kernel_and_args(self):
        @helion.kernel(autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )
        return add, args

    @staticmethod
    def _fake_cache_random(count):
        """Patch _find_similar_cached_configs to return `count` distinct backend-agnostic flats."""
        from helion.autotuner.local_cache import SavedBestConfig

        def fake_find(search_self, max_configs):
            distinct: list[tuple] = []
            for _ in range(20):
                if len(distinct) >= count:
                    break
                batch = search_self.config_gen.random_population_flat(
                    max(count * 3, 10)
                )
                for flat in batch:
                    t = tuple(flat)
                    if t not in distinct:
                        distinct.append(t)
                        if len(distinct) >= count:
                            break
            out = []
            for flat in distinct[:max_configs]:
                out.append(
                    SavedBestConfig(
                        hardware="x",
                        specialization_key="x",
                        config=search_self.config_gen.unflatten(list(flat)),
                        config_spec_hash="x",
                        flat_config=flat,
                    )
                )
            return out

        return fake_find

    def test_from_cache_factory(self):
        """helion.from_cache() returns a callable that creates a CachedFiniteSearch."""

        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        fn = helion.from_cache()
        self.assertTrue(callable(fn))
        with (
            patch.object(BaseSearch, "_find_similar_cached_configs", return_value=[]),
            self.assertRaises(exc.NotEnoughConfigs),
        ):
            fn(bound, args)

    def test_from_cache_empty_cache_raises(self):
        """CachedFiniteSearch with empty cache and no explicit configs raises NotEnoughConfigs."""
        from helion.autotuner.finite_search import CachedFiniteSearch

        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        with (
            patch.object(BaseSearch, "_find_similar_cached_configs", return_value=[]),
            self.assertRaises(exc.NotEnoughConfigs),
        ):
            CachedFiniteSearch(bound, args)

    def test_from_cache_prepends_cached(self):
        """Cached configs appear before explicit configs in CachedFiniteSearch.configs."""
        from helion.autotuner.finite_search import CachedFiniteSearch

        cfg1 = helion.Config(block_sizes=[16])
        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        fake_fn = self._fake_cache_random(2)
        fake_sizes: list[int] = []

        def spy(search_self, max_configs):
            result = fake_fn(search_self, max_configs)
            fake_sizes.append(len(result))
            return result

        with patch.object(BaseSearch, "_find_similar_cached_configs", spy):
            search = CachedFiniteSearch(bound, args, configs=[cfg1])
        self.assertEqual(len(search.configs), fake_sizes[0] + 1)
        self.assertEqual(search.configs[-1], cfg1)

    def test_from_cache_respects_max_parameter(self):
        """from_cache(max_configs=N) caps the number of cached configs."""
        from helion.autotuner.finite_search import CachedFiniteSearch

        cfg1 = helion.Config(block_sizes=[16])
        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        observed_caps: list[int] = []
        fake_sizes: list[int] = []
        fake_fn = self._fake_cache_random(5)

        def spy(search_self, max_configs):
            observed_caps.append(max_configs)
            result = fake_fn(search_self, max_configs)
            fake_sizes.append(len(result))
            return result

        with patch.object(BaseSearch, "_find_similar_cached_configs", spy):
            search = CachedFiniteSearch(bound, args, configs=[cfg1], max_configs=2)
        self.assertEqual(observed_caps, [2])
        self.assertLessEqual(fake_sizes[0], 2)
        self.assertEqual(len(search.configs), fake_sizes[0] + 1)

    def test_from_cache_uses_default_cap_from_settings(self):
        """Without max_configs, the cap falls back to autotune_best_available_max_configs."""
        from helion.autotuner.finite_search import CachedFiniteSearch

        cfg1 = helion.Config(block_sizes=[16])
        add, args = self._make_kernel_and_args()
        bound = add.bind(args)
        observed_caps: list[int] = []
        fake_sizes: list[int] = []
        fake_fn = self._fake_cache_random(5)

        def spy(search_self, max_configs):
            observed_caps.append(max_configs)
            result = fake_fn(search_self, max_configs)
            fake_sizes.append(len(result))
            return result

        with patch.object(BaseSearch, "_find_similar_cached_configs", spy):
            search = CachedFiniteSearch(bound, args, configs=[cfg1])
        cap_in_effect = search.settings.autotune_best_available_max_configs
        self.assertEqual(observed_caps, [cap_in_effect])
        self.assertLessEqual(fake_sizes[0], cap_in_effect)
        self.assertEqual(len(search.configs), fake_sizes[0] + 1)

    def test_kernel_autotuner_fn_accepts_from_cache(self):
        """@helion.kernel(autotuner_fn=helion.from_cache()) stores the callable in settings."""
        fn = helion.from_cache()

        @helion.kernel(autotuner_fn=fn, autotune_log_level=0)
        def add(a, b):
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([128], device=DEVICE),
            torch.randn([128], device=DEVICE),
        )
        bound = add.bind(args)
        self.assertIs(bound.settings.autotuner_fn, fn)


@onlyBackends(["triton", "cute"])
class TestAutotuneBudget(TestCase):
    def _make_search(self, settings: Settings) -> BaseSearch:
        # NOTE: construct via __init__ (mock kernel) instead of hand-mirroring
        # its attributes, so new __init__ fields don't need to be added here.
        config_spec = SimpleNamespace(
            default_config=lambda: helion.Config(block_sizes=[1]),
            compiler_seed_timeout_retry_repetitions=None,
            cute_flash_search_enabled=False,
        )
        kernel = SimpleNamespace(
            settings=settings,
            config_spec=config_spec,
            format_kernel_decorator=lambda config, s: "decorator",
            to_triton_code=lambda config: "code",
            maybe_log_repro=lambda log_func, args, config=None: None,
            supports_subprocess_benchmark=lambda: False,
            env=SimpleNamespace(process_group_name=None),
        )
        search = BaseSearch(kernel, ())
        with patch.object(
            LocalBenchmarkProvider,
            "_compute_baseline",
            return_value=(None, [], None),
        ):
            search._prepare()
        return search

    def test_setting_default_is_none(self) -> None:
        with without_env_var("HELION_AUTOTUNE_BUDGET_SECONDS"):
            settings = Settings()
        self.assertIsNone(settings.autotune_budget_seconds)

    def test_cute_backend_uses_default_autotune_budget_without_mutating(self) -> None:
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.backend import _CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS

        settings = Settings(
            autotune_budget_seconds=None,
            autotune_effort="quick",
            autotune_log_level=logging.CRITICAL,
        )
        bound_kernel = SimpleNamespace(settings=settings)
        observed_budgets: list[int | None] = []

        def fake_autotune(self_, bound_kernel_, args, **kwargs):
            observed_budgets.append(bound_kernel_.settings.autotune_budget_seconds)
            return helion.Config()

        with patch.object(
            Backend, "autotune", autospec=True, side_effect=fake_autotune
        ):
            CuteBackend().autotune(bound_kernel, ())

        self.assertEqual(observed_budgets, [_CUTE_DEFAULT_AUTOTUNE_BUDGET_SECONDS])
        self.assertIsNone(bound_kernel.settings.autotune_budget_seconds)

    def test_cute_backend_leaves_full_autotune_unbudgeted_by_default(self) -> None:
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend

        settings = Settings(
            autotune_budget_seconds=None,
            autotune_effort="full",
            autotune_log_level=logging.CRITICAL,
        )
        bound_kernel = SimpleNamespace(settings=settings)
        observed_budgets: list[int | None] = []

        def fake_autotune(self_, bound_kernel_, args, **kwargs):
            observed_budgets.append(bound_kernel_.settings.autotune_budget_seconds)
            return helion.Config()

        with patch.object(
            Backend, "autotune", autospec=True, side_effect=fake_autotune
        ):
            CuteBackend().autotune(bound_kernel, ())

        self.assertEqual(observed_budgets, [None])
        self.assertIsNone(bound_kernel.settings.autotune_budget_seconds)

    def test_cute_backend_restores_default_autotune_budget_on_error(self) -> None:
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend

        settings = Settings(
            autotune_budget_seconds=None,
            autotune_effort="quick",
            autotune_log_level=logging.CRITICAL,
        )
        bound_kernel = SimpleNamespace(settings=settings)

        with (
            patch.object(
                Backend,
                "autotune",
                autospec=True,
                side_effect=RuntimeError("boom"),
            ),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            CuteBackend().autotune(bound_kernel, ())

        self.assertIsNone(bound_kernel.settings.autotune_budget_seconds)

    def test_cute_backend_preserves_explicit_autotune_budget(self) -> None:
        from helion._compiler.backend import Backend
        from helion._compiler.backend import CuteBackend

        settings = Settings(
            autotune_budget_seconds=42,
            autotune_log_level=logging.CRITICAL,
        )
        bound_kernel = SimpleNamespace(settings=settings)
        observed_budgets: list[int | None] = []

        def fake_autotune(self_, bound_kernel_, args, **kwargs):
            observed_budgets.append(bound_kernel_.settings.autotune_budget_seconds)
            return helion.Config()

        with patch.object(
            Backend, "autotune", autospec=True, side_effect=fake_autotune
        ):
            CuteBackend().autotune(bound_kernel, ())

        self.assertEqual(observed_budgets, [42])
        self.assertEqual(bound_kernel.settings.autotune_budget_seconds, 42)

    def test_setting_from_env_var(self) -> None:
        with patch.dict(
            os.environ,
            {"HELION_AUTOTUNE_BUDGET_SECONDS": "300"},
            clear=False,
        ):
            settings = Settings()
        self.assertEqual(settings.autotune_budget_seconds, 300)

    def test_setting_from_kwarg(self) -> None:
        settings = Settings(autotune_budget_seconds=42)
        self.assertEqual(settings.autotune_budget_seconds, 42)

    def test_no_budget_yields_full_range(self) -> None:
        settings = Settings(
            autotune_budget_seconds=None,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search._autotune_budget_start = time.perf_counter() - 1e9
        self.assertEqual(list(search._budgeted_range(1, 4)), [1, 2, 3])

    def test_budget_yields_while_time_remains(self) -> None:
        settings = Settings(
            autotune_budget_seconds=600,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        self.assertEqual(list(search._budgeted_range(1, 4)), [1, 2, 3])

    def test_budget_stops_range_when_elapsed_exceeds(self) -> None:
        settings = Settings(
            autotune_budget_seconds=1,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search._autotune_budget_start = time.perf_counter() - 2.0
        search.log = Mock()
        self.assertEqual(list(search._budgeted_range(10)), [])
        search.log.assert_called_once()

    def test_budget_exhaustion_is_agreed_across_distributed_ranks(self) -> None:
        settings = Settings(
            autotune_budget_seconds=600,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search.config_spec.cute_flash_search_enabled = True
        search.kernel.env.process_group_name = "autotune"

        with patch(
            "helion.autotuner.base_search.all_gather_object",
            return_value=[False, True],
        ) as gather:
            self.assertTrue(search._autotune_budget_exceeded_across_ranks())

        gather.assert_called_once_with(False, process_group_name="autotune")

    def test_non_flash_budget_exhaustion_does_not_add_collective(self) -> None:
        settings = Settings(
            autotune_budget_seconds=600,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search.kernel.env.process_group_name = "autotune"

        with patch(
            "helion.autotuner.base_search.all_gather_object",
            side_effect=AssertionError("non-flash search must not add a collective"),
        ):
            self.assertFalse(search._autotune_budget_exceeded_across_ranks())

    def test_budget_unset_when_prepare_not_called(self) -> None:
        settings = Settings(
            autotune_budget_seconds=1,
            autotune_log_level=logging.CRITICAL,
        )
        search = BaseSearch.__new__(BaseSearch)
        search.settings = settings
        search._autotune_budget_start = None
        search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
        self.assertEqual(list(search._budgeted_range(3)), [0, 1, 2])

    def test_budget_resets_when_prepare_called_again(self) -> None:
        settings = Settings(
            autotune_budget_seconds=60,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        search._autotune_budget_start = time.perf_counter() - 100.0
        self.assertEqual(list(search._budgeted_range(1)), [])

        search._prepared = False
        with patch.object(
            LocalBenchmarkProvider,
            "_compute_baseline",
            return_value=(None, [], None),
        ):
            search._prepare()
        self.assertEqual(list(search._budgeted_range(1)), [0])

    def test_budget_zero_immediately_exhausts(self) -> None:
        settings = Settings(
            autotune_budget_seconds=0,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        time.sleep(0.001)
        self.assertEqual(list(search._budgeted_range(10)), [])

    def test_setting_has_user_facing_description(self) -> None:
        self.assertIn("autotune_budget_seconds", Settings.__slots__)
        description = Settings.__slots__["autotune_budget_seconds"]
        self.assertIn("HELION_AUTOTUNE_BUDGET_SECONDS", description)
        self.assertIn("best", description.lower())

    def test_generation_loops_use_budgeted_range(self) -> None:
        import inspect

        from helion.autotuner.de_surrogate_hybrid import DESurrogateHybrid
        from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
        from helion.autotuner.llm_search import LLMGuidedSearch
        from helion.autotuner.pattern_search import PatternSearch
        from helion.autotuner.surrogate_pattern_search import LFBOPatternSearch

        for cls in (
            DESurrogateHybrid,
            DifferentialEvolutionSearch,
            LLMGuidedSearch,
            PatternSearch,
            LFBOPatternSearch,
        ):
            source = inspect.getsource(cls)
            self.assertIn(
                "_budgeted_range",
                source,
                f"{cls.__name__} should use _budgeted_range for generation loops",
            )

    def test_finishing_phase_respects_budget(self) -> None:
        import inspect

        from helion.autotuner.base_search import PopulationBasedSearch

        source = inspect.getsource(PopulationBasedSearch.run_finishing_phase)
        self.assertIn(
            "_budgeted_range",
            source,
            "run_finishing_phase should stop when the autotune budget is exhausted",
        )

    def test_final_rebenchmark_respects_budget(self) -> None:
        settings = Settings(
            autotune_budget_seconds=1,
            autotune_log_level=logging.CRITICAL,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search._autotune_budget_start = time.perf_counter() - 2.0
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = []
        member = PopulationMember(lambda: None, [1.0], (), helion.Config())

        with patch.object(search, "rebenchmark") as rebenchmark:
            self.assertIs(search.final_rebenchmark_best(member), member)

        rebenchmark.assert_not_called()

    def test_final_rebenchmark_uses_distributed_flash_budget(self) -> None:
        settings = Settings(
            autotune_budget_seconds=600,
            autotune_log_level=logging.CRITICAL,
        )
        search = PopulationBasedSearch.__new__(PopulationBasedSearch)
        search.settings = settings
        search.log = AutotuningLogger(settings)
        search._autotune_budget_start = time.perf_counter()
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.population = []
        search.config_spec = SimpleNamespace(cute_flash_search_enabled=True)
        search.kernel = SimpleNamespace(
            env=SimpleNamespace(process_group_name="autotune")
        )
        member = PopulationMember(lambda: None, [1.0], (), helion.Config())

        with (
            patch(
                "helion.autotuner.base_search.all_gather_object",
                return_value=[False, True],
            ) as gather,
            patch.object(search, "rebenchmark") as rebenchmark,
        ):
            self.assertIs(search.final_rebenchmark_best(member), member)

        gather.assert_called_once_with(False, process_group_name="autotune")
        rebenchmark.assert_not_called()

    def test_prepare_wires_budget_hook_into_provider(self) -> None:
        """``BaseSearch._prepare`` should install the budget-check hook on
        the benchmark provider so the initial-population
        compile/benchmark phase can short-circuit once the wall-clock
        budget is exhausted.
        """
        settings = Settings(
            autotune_budget_seconds=1,
            autotune_log_level=logging.CRITICAL,
        )
        search = self._make_search(settings)
        self.assertEqual(
            search.benchmark_provider.budget_exceeded_fn,
            search._autotune_budget_exceeded,
        )

    def test_prepare_registers_normalized_compiler_seeds(self) -> None:
        normalized = helion.Config(block_sizes=[64])

        class RecordingProvider(LocalBenchmarkProvider):
            def __init__(self, *, config_spec, **_kwargs) -> None:
                self.config_spec = config_spec

        settings = Settings(autotune_log_level=logging.CRITICAL)
        config_gen = Mock()
        config_gen.seed_flat_config_pairs.return_value = [([64], normalized)]
        config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=3,
            create_config_generation=Mock(return_value=config_gen),
        )
        kernel = SimpleNamespace(
            settings=settings,
            config_spec=config_spec,
            env=SimpleNamespace(process_group_name=None),
        )
        search = BaseSearch(kernel, (), RecordingProvider)

        search._prepare()
        normalized.config["block_sizes"][0] = 128

        self.assertEqual(
            search.benchmark_provider._compiler_seed_configs,
            {helion.Config(block_sizes=[64])},
        )
        search.config_spec.create_config_generation.assert_called_once_with(
            overrides=None,
            advanced_controls_files=None,
            process_group_name=None,
        )

    def _make_stub_provider(self):
        """Construct a minimal ``LocalBenchmarkProvider`` for budget-loop
        tests without standing up a real kernel/config_spec.
        """
        from helion.autotuner.benchmark_provider import LocalBenchmarkProvider

        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        provider.kernel = SimpleNamespace(
            compile_config=lambda config, allow_print: lambda *a, **kw: None,
            format_kernel_decorator=lambda config, s: "decorator",
            env=SimpleNamespace(process_group_name=None),
        )
        provider.settings = Settings(autotune_log_level=logging.CRITICAL)
        # Match the cute backend's path: CuteBackend.supports_precompile()
        # is False, which clears autotune_precompile in autotune setup.
        provider.settings.autotune_precompile = None
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=False,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                should_deduplicate_generated_sources=lambda config_spec: False
            ),
        )
        provider.args = ()
        provider.log = AutotuningLogger(provider.settings)
        provider._autotune_metrics = SimpleNamespace(
            num_configs_tested=0,
            num_compile_failures=0,
            num_worker_failures=0,
            num_accuracy_failures=0,
            num_successful_candidate_measurements=0,
            num_unique_sources=0,
            num_source_deduplications=0,
            num_generations=0,
            kernel_source="",
        )
        provider.mutated_arg_indices = ()
        provider._benchmark_worker = None
        provider._precompile_args_path = None
        provider._precompile_tmpdir = None
        provider._effective_source_hashes = set()
        provider._effective_source_results = {}
        provider._invalid_effective_source_hashes = set()
        provider._pending_effective_source_failures = {}
        provider._effective_source_repairs = {}
        provider._accuracy_failure_config_ids = []
        provider._compile_failure_config_ids = []
        provider._worker_failure_config_ids = []
        provider._compiler_seed_configs = set()
        provider._compiler_seed_source_hashes = set()
        return provider

    def test_cute_flash_benchmark_deduplicates_effective_sources(self) -> None:
        provider = self._make_stub_provider()

        def generated_source_hash(fn):
            return fn.source_hash

        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=generated_source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            def fn():
                return None

            block_size = config.get("block_sizes")[0]
            fn.source_hash = "shared" if block_size in (1, 2, 4) else "unique"
            return fn

        provider.kernel.compile_config = compile_config
        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
            helion.Config(block_sizes=[3]),
        ]
        repeated_config = helion.Config(block_sizes=[4])
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=(1.0, 2.0),
        ) as benchmark:
            results = provider.benchmark(configs)
            repeated_results = provider.benchmark([repeated_config])

        self.assertEqual(benchmark.call_count, 2)
        self.assertEqual([result.perf for result in results], [1.0, 1.0, 2.0])
        self.assertEqual(results[1].status, "deduplicated")
        self.assertIs(results[0].fn, results[1].fn)
        self.assertEqual(repeated_results[0].perf, 1.0)
        self.assertEqual(repeated_results[0].status, "deduplicated")
        self.assertIs(repeated_results[0].config, repeated_config)
        self.assertIs(repeated_results[0].fn, results[0].fn)
        self.assertIsNot(repeated_results[0], results[0])
        self.assertEqual(
            provider._autotune_metrics.num_successful_candidate_measurements, 2
        )
        self.assertEqual(provider._autotune_metrics.num_unique_sources, 2)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 2)

    def test_compile_config_failure_has_unstarted_source_evidence(self) -> None:
        provider = self._make_stub_provider()
        provider.settings.autotune_progress_bar = False
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            block_size = config.get("block_sizes")[0]
            if block_size == 2:
                raise RuntimeError("simulated compile_config failure")

            def fn():
                return None

            fn.source_hash = str(block_size) * 64
            return fn

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            return 1.0

        provider.kernel.compile_config = compile_config
        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
            helion.Config(block_sizes=[3]),
        ]
        failed_config = configs[1]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune"
        metadata = KernelMetadata(kernel_source="source")

        with (
            provider.log.autotune_logging(
                str(base_path), metadata, collect_dataset=True
            ),
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
        ):
            results = provider.benchmark(configs)

        with base_path.with_suffix(".csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        with base_path.with_suffix(".sources.csv").open(newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        metadata_record = json.loads(base_path.with_suffix(".meta.jsonl").read_text())
        failed_id = canonical_config_id(failed_config)

        self.assertEqual([result.status for result in results], ["ok", "error", "ok"])
        self.assertEqual(
            set(metadata_record["configs"]),
            set(map(canonical_config_id, configs)),
        )
        self.assertEqual(len(rows), len(source_rows))
        self.assertEqual(
            [row["status"] for row in rows if row["config_id"] == failed_id],
            ["error"],
        )
        failed_source_hash = next(
            row["source_hash"] for row in source_rows if row["config_id"] == failed_id
        )
        self.assertEqual(
            failed_source_hash, _compile_config_failure_source_hash(failed_config)
        )
        failed_member = PopulationMember(
            results[1].fn,
            [results[1].perf],
            [],
            failed_config,
            status=results[1].status,
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.config_spec = provider.config_spec
        self.assertEqual(
            search._flash_terminal_member_result(failed_member)["source_hash"],
            failed_source_hash,
        )
        self.assertEqual(
            sum(row["status"] == "started" for row in rows),
            provider._autotune_metrics.num_configs_tested,
        )
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 1)
        self.assertEqual(
            len({row["source_hash"] for row in source_rows}),
            provider._autotune_metrics.num_unique_sources,
        )

    def test_all_compile_failures_after_success_keep_prior_result(self) -> None:
        provider = self._make_stub_provider()
        provider.settings.autotune_progress_bar = False
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        fail_compilation = False

        def compile_config(config, allow_print):
            if fail_compilation:
                raise RuntimeError("simulated terminal compile failure")

            def fn():
                return None

            fn.source_hash = "a" * 64
            return fn

        provider.kernel.compile_config = compile_config
        initial_config = helion.Config(block_sizes=[1])
        failed_configs = [
            helion.Config(block_sizes=[2]),
            helion.Config(block_sizes=[3]),
        ]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune"
        metadata = KernelMetadata(kernel_source="source")

        with (
            provider.log.autotune_logging(
                str(base_path), metadata, collect_dataset=True
            ),
            patch.object(
                LocalBenchmarkProvider, "_benchmark_function", return_value=1.0
            ),
        ):
            initial_result = provider.benchmark([initial_config])
            fail_compilation = True
            failed_results = provider.benchmark(failed_configs)

        with base_path.with_suffix(".csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        with base_path.with_suffix(".sources.csv").open(newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        failed_ids = set(map(canonical_config_id, failed_configs))

        self.assertEqual([result.status for result in initial_result], ["ok"])
        self.assertEqual(
            [result.status for result in failed_results], ["error", "error"]
        )
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 2)
        self.assertEqual(
            {
                row["config_id"]
                for row in rows
                if row["status"] == "error" and row["config_id"] in failed_ids
            },
            failed_ids,
        )
        self.assertFalse(
            any(
                row["status"] == "started" and row["config_id"] in failed_ids
                for row in rows
            )
        )
        self.assertEqual(
            {
                row["source_hash"]
                for row in source_rows
                if row["config_id"] in failed_ids
            },
            {_compile_config_failure_source_hash(config) for config in failed_configs},
        )

    def test_initial_all_compile_failures_still_raise(self) -> None:
        provider = self._make_stub_provider()
        provider.settings.autotune_progress_bar = False
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            raise RuntimeError("simulated initial compile failure")

        provider.kernel.compile_config = compile_config
        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
        ]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune"
        metadata = KernelMetadata(kernel_source="source")

        with (
            provider.log.autotune_logging(
                str(base_path), metadata, collect_dataset=True
            ),
            self.assertRaisesRegex(RuntimeError, "simulated initial compile failure"),
        ):
            provider.benchmark(configs)

        with base_path.with_suffix(".csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        failed_ids = set(map(canonical_config_id, configs))
        self.assertEqual(provider._autotune_metrics.num_compile_failures, 2)
        self.assertEqual(
            {row["config_id"] for row in rows if row["status"] == "error"},
            failed_ids,
        )

    def test_invalidated_effective_source_rejects_later_alias(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            def fn():
                return None

            fn.source_hash = "shared"
            return fn

        provider.kernel.compile_config = compile_config
        representative = helion.Config(block_sizes=[1])
        later_alias = helion.Config(block_sizes=[2])
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            return_value=1.0,
        ) as benchmark:
            first = provider.benchmark([representative])
            provider.invalidate_effective_source_hash("shared")
            alias = provider.benchmark([later_alias])

        benchmark.assert_called_once()
        self.assertEqual(first[0].status, "ok")
        self.assertEqual(alias[0].status, "source_rejected")
        self.assertEqual(alias[0].perf, math.inf)
        self.assertFalse(provider.has_measured_source_hash("shared"))

    def test_same_batch_source_alias_is_not_logged_as_benchmarked(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        representative = helion.Config(block_sizes=[1])
        alias = helion.Config(block_sizes=[2])

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            return 1.0

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=lambda config: (
                    "representative" if config is representative else "alias"
                ),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            results = provider.benchmark([representative, alias])

        entries = [call.args[0] for call in record_entry.call_args_list]
        self.assertEqual([result.status for result in results], ["ok", "deduplicated"])
        self.assertEqual(
            [entry.status for entry in entries if entry.config is representative],
            ["started", "ok"],
        )
        self.assertEqual(
            [entry.status for entry in entries if entry.config is alias],
            ["deduplicated"],
        )
        self.assertEqual(provider._autotune_metrics.num_configs_tested, 1)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 1)

    def test_compiler_seed_alias_marks_source_for_timeout_retry(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=3,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        representative = helion.Config(block_sizes=[1])
        compiler_seed_alias = helion.Config(block_sizes=[2])
        provider.set_compiler_seed_configs([compiler_seed_alias])

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            return_value=1.0,
        ) as benchmark:
            results = provider.benchmark([representative, compiler_seed_alias])

        self.assertEqual([result.perf for result in results], [1.0, 1.0])
        benchmark.assert_called_once_with(
            representative,
            results[0].fn,
            effective_source_hash="shared",
        )

    def test_compiler_seed_source_provenance_survives_batches(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=3,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        compiler_seed = helion.Config(block_sizes=[1])
        later_alias = helion.Config(block_sizes=[2])
        provider.set_compiler_seed_configs([compiler_seed])

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=(math.inf, 1.0),
        ) as benchmark:
            first = provider.benchmark([compiler_seed])
            second = provider.benchmark([later_alias])

        self.assertEqual(first[0].perf, math.inf)
        self.assertEqual(second[0].perf, 1.0)
        self.assertEqual(
            benchmark.call_args_list[1],
            unittest.mock.call(
                later_alias,
                second[0].fn,
                effective_source_hash="shared",
            ),
        )

    def test_disabled_compiler_seed_retry_does_not_register_seeds(self) -> None:
        provider = self._make_stub_provider()
        seed = helion.Config(block_sizes=[1])

        provider.set_compiler_seed_configs([seed])

        self.assertFalse(provider._is_compiler_seed_config(seed))
        self.assertEqual(provider._compiler_seed_configs, set())

    def test_effective_source_aliases_remain_positional_after_budget_expiry(
        self,
    ) -> None:
        provider = self._make_stub_provider()

        def generated_source_hash(fn):
            return fn.source_hash

        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=generated_source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )

        def compile_config(config, allow_print):
            def fn():
                return None

            block_size = config.get("block_sizes")[0]
            fn.source_hash = "first" if block_size == 1 else "shared-tail"
            return fn

        provider.kernel.compile_config = compile_config
        benchmark_count = [0]
        provider.set_budget_exceeded_fn(lambda: benchmark_count[0] >= 1)

        def counting_benchmark(self_, config, fn):
            benchmark_count[0] += 1
            return 1.0

        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
            helion.Config(block_sizes=[3]),
        ]
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            new=counting_benchmark,
        ):
            results = provider.benchmark(configs)

        self.assertEqual([result.perf for result in results], [1.0, math.inf, math.inf])
        self.assertEqual(
            [result.status for result in results], ["ok", "error", "error"]
        )
        self.assertEqual([result.config for result in results], configs)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 0)

    def test_backend_policy_can_disable_effective_source_dedup(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=False,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: False,
            ),
        )
        configs = [
            helion.Config(block_sizes=[1]),
            helion.Config(block_sizes=[2]),
        ]
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=(1.0, 2.0),
        ) as benchmark:
            results = provider.benchmark(configs)

        self.assertEqual(benchmark.call_count, 2)
        self.assertEqual([result.perf for result in results], [1.0, 2.0])
        self.assertEqual(provider._autotune_metrics.num_unique_sources, 0)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 0)

    def test_failed_effective_source_is_retried(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            cute_flash_search_enabled=True,
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        configs = [helion.Config(block_sizes=[1])]
        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=(math.inf, 1.0),
        ) as benchmark:
            first = provider.benchmark(configs)
            second = provider.benchmark(configs)

        self.assertEqual(benchmark.call_count, 2)
        self.assertEqual(first[0].status, "error")
        self.assertEqual(second[0].status, "ok")

    def test_failed_effective_source_retries_alias_independently(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        representative = helion.Config(block_sizes=[1])
        alias = helion.Config(block_sizes=[2])

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            if config is representative:
                provider._record_worker_failure(config, "timeout")
                return math.inf
            return 1.0

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ) as benchmark_mock,
            patch.object(
                provider.log,
                "register_config",
                side_effect=lambda config: (
                    "representative" if config is representative else "alias"
                ),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            results = provider.benchmark([representative, alias])

        entries = [call.args[0] for call in record_entry.call_args_list]
        self.assertEqual(benchmark_mock.call_count, 2)
        self.assertEqual([result.status for result in results], ["deduplicated", "ok"])
        self.assertEqual(
            [entry.status for entry in entries if entry.config == representative],
            ["started", "timeout", "deduplicated"],
        )
        self.assertEqual(
            [entry.status for entry in entries if entry.config is alias],
            ["started", "ok"],
        )
        self.assertEqual(
            [entry.perf_ms for entry in entries if entry.status == "deduplicated"],
            [1.0],
        )
        self.assertEqual(
            {entry.source_hash for entry in entries},
            {"shared"},
        )
        self.assertEqual(provider._autotune_metrics.num_configs_tested, 2)
        self.assertEqual(provider._autotune_metrics.num_worker_failures, 1)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 1)

    def test_equal_failed_configs_each_receive_source_repair(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        configs = [helion.Config(block_sizes=[1]) for _ in range(3)]
        calls = 0

        def benchmark(config, fn):
            nonlocal calls
            calls += 1
            provider._autotune_metrics.num_configs_tested += 1
            if calls < 3:
                provider._record_worker_failure(config, "timeout")
                return math.inf
            return 1.0

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=("first", "second", "third"),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            results = provider.benchmark(configs)

        self.assertEqual(
            [result.status for result in results],
            ["deduplicated", "deduplicated", "ok"],
        )
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 2)
        self.assertEqual(
            sum(
                call.args[0].status == "deduplicated"
                for call in record_entry.call_args_list
            ),
            2,
        )

    def test_later_batch_repairs_failed_effective_source(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        representative = helion.Config(block_sizes=[1])
        alias = helion.Config(block_sizes=[2])

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            if config is representative:
                provider._record_worker_failure(config, "timeout")
                return math.inf
            return 1.0

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=("representative", "alias"),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            provider._autotune_metrics.num_generations = 3
            failed = provider.benchmark([representative])
            provider._autotune_metrics.num_generations = 4
            succeeded = provider.benchmark([alias])

        repairs = provider.take_effective_source_repairs()
        representative_entries = [
            call.args[0]
            for call in record_entry.call_args_list
            if call.args[0].config == representative
        ]
        self.assertEqual(failed[0].status, "timeout")
        self.assertEqual(succeeded[0].status, "ok")
        self.assertEqual(repairs[representative].status, "deduplicated")
        self.assertEqual(repairs[representative].perf, 1.0)
        self.assertEqual(
            [entry.status for entry in representative_entries],
            ["started", "timeout", "deduplicated"],
        )
        self.assertEqual(
            [entry.generation for entry in representative_entries],
            [3, 3, 4],
        )
        alias_entries = [
            call.args[0]
            for call in record_entry.call_args_list
            if call.args[0].config == alias
        ]
        self.assertEqual(
            [(entry.status, entry.generation) for entry in alias_entries],
            [("started", 4), ("ok", 4)],
        )

    def test_lfbo_source_repair_updates_prior_surrogate_target(self) -> None:
        config = helion.Config(block_sizes=[1])
        member = PopulationMember(
            fn=lambda: None,
            perfs=[math.inf],
            flat_values=[],
            config=config,
            status="timeout",
        )
        replacement_fn = Mock()
        replacement_fn.source_hash = "shared"
        repair = SimpleNamespace(
            config=config,
            fn=replacement_fn,
            perf=1.0,
            status="deduplicated",
            compile_time=None,
        )
        search = LFBOPatternSearch.__new__(LFBOPatternSearch)
        search.population = [member]
        search._compiler_seed_members = []
        search._benchmarked_members = {}
        search._pinned_finalist_configs = set()
        search._pinned_finalist_members = {}
        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: getattr(fn, "source_hash", None)
            )
        )
        search.train_configs = [copy.deepcopy(config)]
        search.train_source_hashes = [None]
        search.train_y = [math.inf]

        search._apply_effective_source_repairs({config: repair}, [])

        self.assertEqual(member.perfs, [1.0])
        self.assertIs(member.fn, replacement_fn)
        self.assertEqual(member.status, "deduplicated")
        self.assertEqual(search.train_y, [1.0])
        self.assertEqual(search.train_source_hashes, ["shared"])

    def test_completed_benchmark_is_logged_before_later_exception(self) -> None:
        provider = self._make_stub_provider()
        first = helion.Config(block_sizes=[1])
        second = helion.Config(block_sizes=[2])

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=(1.0, RuntimeError("boom")),
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=("first", "second"),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            provider.benchmark([first, second])

        first_entries = [
            call.args[0]
            for call in record_entry.call_args_list
            if call.args[0].config is first
        ]
        self.assertEqual([entry.status for entry in first_entries], ["started", "ok"])
        self.assertEqual(first_entries[-1].perf_ms, 1.0)

    def test_failed_benchmark_is_logged_before_later_exception(self) -> None:
        provider = self._make_stub_provider()
        first = helion.Config(block_sizes=[1])
        second = helion.Config(block_sizes=[2])

        def benchmark(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            if config is first:
                provider._record_worker_failure(config, "timeout")
                return math.inf
            raise RuntimeError("boom")

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=benchmark,
            ),
            patch.object(
                provider.log,
                "register_config",
                side_effect=("first", "second"),
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            provider.benchmark([first, second])

        first_entries = [
            call.args[0]
            for call in record_entry.call_args_list
            if call.args[0].config is first
        ]
        self.assertEqual(
            [entry.status for entry in first_entries], ["started", "timeout"]
        )

    def test_accuracy_failure_invalidates_effective_source_aliases(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: "shared",
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        invalid = helion.Config(block_sizes=[1])
        alias = helion.Config(block_sizes=[2])

        def fail_accuracy(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            provider._record_accuracy_failure(config)
            return math.inf

        with (
            patch.object(
                LocalBenchmarkProvider,
                "_benchmark_function",
                side_effect=fail_accuracy,
            ) as benchmark,
            patch.object(
                provider.log,
                "register_config",
                side_effect=lambda config: "invalid" if config is invalid else "alias",
            ),
            patch.object(provider.log, "record_autotune_entry") as record_entry,
        ):
            results = provider.benchmark([invalid, alias])

        entries = [call.args[0] for call in record_entry.call_args_list]
        benchmark.assert_called_once_with(invalid, results[0].fn)
        self.assertEqual(
            [result.status for result in results],
            ["accuracy_error", "source_rejected"],
        )
        self.assertEqual([result.perf for result in results], [math.inf, math.inf])
        self.assertEqual(
            [entry.status for entry in entries if entry.config is invalid],
            ["started", "accuracy_error"],
        )
        self.assertEqual(
            [entry.status for entry in entries if entry.config is alias],
            ["source_rejected"],
        )
        self.assertEqual(provider._autotune_metrics.num_configs_tested, 1)
        self.assertEqual(provider._autotune_metrics.num_accuracy_failures, 1)
        self.assertEqual(provider._autotune_metrics.num_source_deduplications, 1)
        self.assertEqual(provider._invalid_effective_source_hashes, {"shared"})

    def test_accuracy_status_remains_error_when_source_dedup_is_disabled(self) -> None:
        provider = self._make_stub_provider()
        config = helion.Config(block_sizes=[1])

        def fail_accuracy(config, fn):
            provider._autotune_metrics.num_configs_tested += 1
            provider._record_accuracy_failure(config)
            return math.inf

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=fail_accuracy,
        ):
            result = provider.benchmark([config])[0]

        self.assertEqual(result.status, "error")
        self.assertEqual(provider._autotune_metrics.num_accuracy_failures, 1)
        self.assertEqual(provider._invalid_effective_source_hashes, set())

    def test_prior_accuracy_failure_does_not_poison_later_source(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            compiler_seed_timeout_retry_repetitions=None,
            backend=SimpleNamespace(
                generated_source_hash=lambda fn: fn.source_hash,
                should_deduplicate_generated_sources=lambda config_spec: True,
            ),
        )
        source_hashes = iter(("inaccurate", "timed-out"))

        def compile_config(config, allow_print):
            fn = Mock()
            fn.source_hash = next(source_hashes)
            return fn

        provider.kernel.compile_config = compile_config
        config = helion.Config(block_sizes=[1])
        calls = 0

        def fail_differently(config, fn):
            nonlocal calls
            calls += 1
            provider._autotune_metrics.num_configs_tested += 1
            if calls == 1:
                provider._record_accuracy_failure(config)
            else:
                provider._record_worker_failure(config, "timeout")
            return math.inf

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            side_effect=fail_differently,
        ):
            first = provider.benchmark([config])[0]
            second = provider.benchmark([config])[0]

        self.assertEqual(first.status, "accuracy_error")
        self.assertEqual(second.status, "timeout")
        self.assertEqual(provider._invalid_effective_source_hashes, {"inaccurate"})

    def test_effective_source_dedup_is_disabled_for_distributed(self) -> None:
        provider = self._make_stub_provider()
        provider.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                should_deduplicate_generated_sources=lambda config_spec: True
            )
        )
        with patch(
            "helion.autotuner.benchmark_provider.dist.is_initialized",
            return_value=True,
        ):
            self.assertFalse(provider._effective_source_dedup_enabled())

    def test_cute_backend_reports_attached_generated_source_hash(self) -> None:
        from helion._compiler.backend import CuteBackend

        backend = CuteBackend()
        compiled = SimpleNamespace(
            __name__="kernel",
            __globals__={
                "_helion_kernel": SimpleNamespace(_helion_cute_source_hash="source-abc")
            },
        )
        self.assertEqual(backend.generated_source_hash(compiled), "source-abc")
        self.assertIsNone(backend.generated_source_hash(SimpleNamespace()))
        self.assertTrue(
            backend.should_deduplicate_generated_sources(_cute_flash_test_config_spec())
        )
        self.assertFalse(
            backend.should_deduplicate_generated_sources(
                SimpleNamespace(cute_flash_search_enabled=False)
            )
        )

    def test_benchmark_provider_short_circuits_compile_loop(self) -> None:
        """``LocalBenchmarkProvider.benchmark`` must stop compiling
        configs once ``budget_exceeded_fn`` returns ``True`` from inside
        the compile loop, and still return one ``BenchmarkResult`` per
        input config so callers receive a positionally-aligned list.
        """
        from helion.autotuner.benchmark_provider import BenchmarkResult
        from helion.autotuner.benchmark_provider import LocalBenchmarkProvider

        provider = self._make_stub_provider()

        compiled_count = [0]
        budget_calls = [0]

        original_compile = provider.kernel.compile_config

        def counting_compile(config, allow_print):
            compiled_count[0] += 1
            return original_compile(config, allow_print)

        provider.kernel.compile_config = counting_compile

        def budget_check():
            budget_calls[0] += 1
            return compiled_count[0] >= 2

        provider.set_budget_exceeded_fn(budget_check)

        from helion.runtime.config import Config

        configs = [Config() for _ in range(5)]

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            return_value=0.001,
        ):
            results = provider.benchmark(configs)

        self.assertEqual(len(results), len(configs))
        compiled_finite = sum(1 for r in results if math.isfinite(r.perf))
        self.assertLessEqual(compiled_count[0], 2)
        self.assertLessEqual(compiled_finite, compiled_count[0])
        for r in results[compiled_count[0] :]:
            self.assertEqual(r.status, "error")
            self.assertEqual(r.perf, float("inf"))
        for r in results:
            self.assertIsInstance(r, BenchmarkResult)
        # The hook must actually fire; without it the loop wouldn't break.
        self.assertGreater(budget_calls[0], 0)

    def test_benchmark_provider_short_circuits_benchmark_loop(self) -> None:
        """If compilation finishes cleanly and the budget only fires
        partway through the benchmark loop, remaining configs must be
        left at the default ``perf=inf, status="error"`` slots while the
        earlier benchmarked configs keep their measured perf.
        """
        from helion.autotuner.benchmark_provider import BenchmarkResult
        from helion.autotuner.benchmark_provider import LocalBenchmarkProvider

        provider = self._make_stub_provider()

        # All compiles succeed; budget stays clear during compilation.
        benchmark_count = [0]

        def budget_check():
            # Trip the budget after 2 benchmark calls.
            return benchmark_count[0] >= 2

        provider.set_budget_exceeded_fn(budget_check)

        def counting_benchmark(self_, config, fn):
            benchmark_count[0] += 1
            return 0.001

        from helion.runtime.config import Config

        configs = [Config() for _ in range(5)]

        with patch.object(
            LocalBenchmarkProvider,
            "_benchmark_function",
            new=counting_benchmark,
        ):
            results = provider.benchmark(configs)

        self.assertEqual(len(results), len(configs))
        # At most 2 benchmarks ran (the loop checks before each call).
        self.assertLessEqual(benchmark_count[0], 2)
        # First few entries have finite measured perf.
        finite_count = sum(1 for r in results if math.isfinite(r.perf))
        self.assertEqual(finite_count, benchmark_count[0])
        # The tail entries must be the inf/error defaults.
        for r in results[benchmark_count[0] :]:
            self.assertEqual(r.status, "error")
            self.assertEqual(r.perf, float("inf"))
        for r in results:
            self.assertIsInstance(r, BenchmarkResult)

    def test_benchmark_provider_default_hook_is_no_op(self) -> None:
        """A provider that never had its hook installed must read the
        class-level no-op default rather than raising ``AttributeError``
        on the first ``budget_exceeded_fn()`` call.
        """
        from helion.autotuner.benchmark_provider import BenchmarkProvider
        from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
        from helion.autotuner.benchmark_provider import _never_exceeded

        # Class-level default lives on the abstract base so any subclass
        # picks it up even if it forgets to call super().__init__.
        self.assertIs(BenchmarkProvider.budget_exceeded_fn, _never_exceeded)
        # Subclass instance with no __init__ work still resolves the
        # default via class-level descriptor.
        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        self.assertFalse(provider.budget_exceeded_fn())

    def test_cute_benchmark_uses_event_timed_subprocess_worker(self) -> None:
        # CuTe inherits the default (None) do_bench hooks: compiled launchers
        # run on the Torch current stream, so CUDA-event timing is accurate
        # and subprocess benchmark jobs must NOT switch to the wall-clock
        # path that charges per-launch host overhead to the kernel.
        from helion._compiler.backend import CuteBackend

        backend = CuteBackend()
        self.assertIsNone(backend.get_do_bench())
        self.assertIsNone(backend.get_interleaved_bench())

        provider = self._make_stub_provider()
        provider.kernel.supports_subprocess_benchmark = lambda: True
        provider.config_spec = SimpleNamespace(backend=backend)

        self.assertTrue(provider._subprocess_benchmark_enabled())
        self.assertFalse(provider._subprocess_benchmark_uses_wall_clock())

    def test_non_cute_custom_benchmark_stays_in_process(self) -> None:
        from helion.autotuner.benchmarking import do_bench_generic

        class OtherBackend:
            @property
            def name(self) -> str:
                return "other"

            def get_do_bench(self):
                return do_bench_generic

        provider = self._make_stub_provider()
        provider.kernel.supports_subprocess_benchmark = lambda: True
        provider.config_spec = SimpleNamespace(backend=OtherBackend())

        self.assertFalse(provider._subprocess_benchmark_enabled())
        self.assertFalse(provider._subprocess_benchmark_uses_wall_clock())


class TestConfigValuePriors(TestCase):
    """Backend-supplied per-key priors bias the random half of the initial
    population (config_generation), while the other half stays uniform."""

    def _add_config_gen(self) -> tuple[ConfigGeneration, object]:
        @helion.kernel(autotune_log_level=0)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        a = torch.randn(256, 256, device=DEVICE)
        bound = add.bind((a, a))
        return ConfigGeneration(bound.config_spec), bound

    @staticmethod
    def _forceable_slot(gen: ConfigGeneration) -> tuple[str, int, object]:
        """Pick a config key the live kernel exposes and a value to force it to.

        Backends expose different keys (e.g. the default backend has a scalar
        ``num_warps`` slot; the cute backend exposes only sequence keys such as
        ``block_sizes``), so the priors tests select a target dynamically rather
        than hard-coding ``num_warps``. Returns ``(key, flat_index, value)``
        where ``value`` is representable by the slot's fragment.
        """
        # Prefer a scalar ``num_warps`` slot when present: a power-of-two value
        # in its range that won't be altered by repair logic.
        for key, (indices, is_sequence) in gen._key_to_flat_indices.items():
            if key == "num_warps" and not is_sequence and indices:
                return key, indices[0], 4
        # Otherwise use the first block-size slot, which every backend exposes.
        block_key, (block_indices, _is_seq) = next(
            (k, v) for k, v in gen._key_to_flat_indices.items() if k == "block_sizes"
        )
        flat_idx = block_indices[0]
        fragment = gen.flat_spec[flat_idx]
        # Use the fragment's own default: guaranteed representable and stable.
        return block_key, flat_idx, fragment.default()

    def test_no_priors_falls_through_to_uniform(self) -> None:
        gen, _ = self._add_config_gen()
        # With no priors, biased sampling is exactly uniform sampling and the
        # population fill is unchanged. Control the priors explicitly rather than
        # relying on the ambient backend default (the cute backend, for example,
        # supplies non-empty priors).
        gen._config_value_priors = {}
        self.assertEqual(gen._config_value_priors, {})
        flat = gen.biased_random_flat()
        self.assertEqual(len(flat), len(gen.flat_spec))

    def test_prior_forces_value(self) -> None:
        from helion.autotuner.config_priors import weighted_choice

        gen, _ = self._add_config_gen()
        key, flat_idx, value = self._forceable_slot(gen)
        # Control the priors directly on the instance so the test is independent
        # of whatever priors the active backend supplies.
        gen._config_value_priors = {key: weighted_choice({value: 1.0})}
        # Disable size repair/shrink so the forced value is observed verbatim.
        with (
            patch.object(gen, "shrink_config", lambda *a, **k: None),
            patch.object(gen, "_repair_cute_num_threads", lambda *a, **k: None),
        ):
            for _ in range(25):
                self.assertEqual(gen.biased_random_flat()[flat_idx], value)

    def test_population_fill_is_half_biased(self) -> None:
        from helion.autotuner.config_priors import weighted_choice

        gen, _ = self._add_config_gen()
        key, _flat_idx, value = self._forceable_slot(gen)
        gen._config_value_priors = {key: weighted_choice({value: 1.0})}
        with (
            # Suppress backend-supplied compiler seeds so the random-fill count
            # is deterministic across backends (cute seeds a few configs).
            patch.object(gen, "seed_flat_config_pairs", return_value=[]),
            patch.object(
                gen, "biased_random_flat", wraps=gen.biased_random_flat
            ) as biased,
            patch.object(gen, "random_flat", wraps=gen.random_flat) as uniform,
        ):
            # 1 default + 10 random fill slots (seeds suppressed above).
            gen.random_population_flat(11)
        # Duplicate or invalid draws are retried, but attempts continue to
        # alternate between biased and uniform sampling.
        self.assertGreaterEqual(biased.call_count + uniform.call_count, 10)
        self.assertLessEqual(abs(biased.call_count - uniform.call_count), 1)

    def test_population_fill_pads_after_unique_configs_exhausted(self) -> None:
        gen, _ = self._add_config_gen()
        default_flat = gen.default_flat()
        normalized = gen.unflatten(default_flat)
        gen.config_spec.cute_flash_search_enabled = True
        with (
            patch.object(gen, "default_flat", return_value=default_flat),
            patch.object(gen, "user_seed_flat_config_pairs", return_value=[]),
            patch.object(gen, "seed_flat_config_pairs", return_value=[]),
            patch.object(gen, "biased_random_flat", return_value=default_flat),
            patch.object(gen, "random_flat", return_value=default_flat),
            patch.object(gen, "unflatten", return_value=normalized),
            patch.object(gen, "flatten", return_value=default_flat),
        ):
            population = gen.random_population_flat(6)
        self.assertEqual(population, [default_flat] * 6)

    def test_flash_population_stores_normalized_flat_configs(self) -> None:
        gen, _ = self._add_config_gen()
        default_flat = gen.default_flat()
        normalized = gen.unflatten(default_flat)
        gen.config_spec.cute_flash_search_enabled = True
        canonical_flat = [*default_flat]
        with (
            patch.object(gen, "default_flat", return_value=default_flat),
            patch.object(gen, "user_seed_flat_config_pairs", return_value=[]),
            patch.object(gen, "seed_flat_config_pairs", return_value=[]),
            patch.object(gen, "unflatten", return_value=normalized),
            patch.object(gen, "flatten", return_value=canonical_flat) as flatten,
        ):
            population = gen.random_population_flat(1)
        self.assertEqual(population, [canonical_flat])
        flatten.assert_called_once_with(normalized)

    def test_non_flash_population_propagates_generation_failure(self) -> None:
        gen, _ = self._add_config_gen()
        gen.config_spec.cute_flash_search_enabled = False
        with (
            patch.object(
                gen,
                "random_population_flat",
                side_effect=exc.InvalidConfig("invalid default"),
            ),
            self.assertRaisesRegex(exc.InvalidConfig, "invalid default"),
        ):
            gen.random_population(1)


class TestSelectedSourceMetrics(TestCase):
    @staticmethod
    def _make_search(
        *,
        source_was_measured: bool,
        population_search: bool = True,
        source_tracking_enabled: bool = True,
    ) -> tuple[BaseSearch, helion.Config]:
        from helion.autotuner.metrics import AutotuneMetrics

        config = helion.Config(block_sizes=[64])
        fn = SimpleNamespace(source_hash="selected-source")
        search_type = PopulationBasedSearch if population_search else BaseSearch
        search = search_type.__new__(search_type)
        search.args = ()
        search.best_perf_so_far = 1.25
        search._autotune_metrics = AutotuneMetrics()
        search.config_spec = SimpleNamespace(
            backend=SimpleNamespace(
                generated_source_hash=lambda compiled: compiled.source_hash,
                should_deduplicate_generated_sources=(
                    lambda config_spec: source_tracking_enabled
                ),
            )
        )
        search.benchmark_provider = Mock()
        search.benchmark_provider.has_measured_source_hash.return_value = (
            source_was_measured
        )
        best = PopulationMember(
            fn=fn,
            perfs=[1.25],
            flat_values=[64],
            config=config,
            status="ok",
        )
        if population_search:
            search.population = [best]
        else:
            search.best = best
        return search, config

    def test_selected_source_records_verified_measurement(self) -> None:
        search, config = self._make_search(source_was_measured=True)

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        self.assertEqual(metrics.selected_source_hash, "selected-source")
        self.assertTrue(metrics.selected_source_was_measured)
        search.benchmark_provider.has_measured_source_hash.assert_called_once_with(
            "selected-source"
        )
        config.config["block_sizes"][0] = 128
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        serialized = metrics.to_dict()
        self.assertEqual(serialized["selected_config"], {"block_sizes": [64]})
        self.assertEqual(serialized["selected_source_hash"], "selected-source")
        self.assertTrue(serialized["selected_source_was_measured"])

    def test_selected_source_records_unmeasured_hash(self) -> None:
        search, config = self._make_search(source_was_measured=False)

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_source_hash, "selected-source")
        self.assertFalse(metrics.selected_source_was_measured)

    def test_selected_source_is_scoped_to_source_deduplication(self) -> None:
        search, config = self._make_search(
            source_was_measured=True,
            source_tracking_enabled=False,
        )

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        self.assertIsNone(metrics.selected_source_hash)
        self.assertFalse(metrics.selected_source_was_measured)
        search.benchmark_provider.has_measured_source_hash.assert_not_called()

    def test_selected_source_uses_finalist_after_noisy_population_best(self) -> None:
        search, config = self._make_search(source_was_measured=True)
        selected = search.population[0]
        stale_config = helion.Config(block_sizes=[128])
        stale = PopulationMember(
            fn=SimpleNamespace(source_hash="stale-source"),
            perfs=[0.5],
            flat_values=[128],
            config=stale_config,
            status="ok",
        )
        search.population = [stale]
        search._benchmarked_members = {config: selected}
        search._pinned_finalist_members = {}
        search.benchmark_provider.has_measured_source_hash.side_effect = (
            lambda source_hash: source_hash == "selected-source"
        )

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        self.assertEqual(metrics.selected_source_hash, "selected-source")
        self.assertTrue(metrics.selected_source_was_measured)

    def test_non_population_winner_leaves_source_unset(self) -> None:
        search, config = self._make_search(
            source_was_measured=True,
            population_search=False,
        )

        search._finalize_autotune_metrics(config)

        metrics = search._autotune_metrics
        self.assertEqual(metrics.selected_config, {"block_sizes": [64]})
        self.assertIsNone(metrics.selected_source_hash)
        self.assertFalse(metrics.selected_source_was_measured)
        search.benchmark_provider.has_measured_source_hash.assert_not_called()

    def test_local_provider_requires_successful_finite_measurement(self) -> None:
        from helion.autotuner.benchmark_provider import BenchmarkResult

        provider = LocalBenchmarkProvider.__new__(LocalBenchmarkProvider)
        config = helion.Config(block_sizes=[64])

        def fn() -> None:
            return None

        provider._effective_source_results = {
            "ok": BenchmarkResult(config, fn, 1.0, "ok", None),
            "deduplicated": BenchmarkResult(config, fn, 1.0, "deduplicated", None),
            "failed": BenchmarkResult(config, fn, math.inf, "error", None),
            "nonfinite": BenchmarkResult(config, fn, math.inf, "ok", None),
        }

        self.assertTrue(provider.has_measured_source_hash("ok"))
        self.assertTrue(provider.has_measured_source_hash("deduplicated"))
        self.assertFalse(provider.has_measured_source_hash("failed"))
        self.assertFalse(provider.has_measured_source_hash("nonfinite"))
        self.assertFalse(provider.has_measured_source_hash("missing"))


if __name__ == "__main__":
    unittest.main()
