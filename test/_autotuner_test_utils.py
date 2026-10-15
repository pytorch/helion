from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
import functools
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Callable
from unittest.mock import patch

import torch
from torch._inductor.runtime.triton_compat import OutOfResources

import helion
from helion import _hardware
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import import_path
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.random_search import RandomSearch
from helion.autotuner.search_space_logger import canonical_config_id
import helion.language as hl
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    import unittest

    from helion.autotuner import LFBOPatternSearch

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")
_TEST_CUTE_FLASH_BACKEND = SimpleNamespace(
    generated_source_hash=lambda _fn: None,
    autotune_config_is_viable=lambda _config_spec, _config: True,
)
examples_dir = Path(__file__).parent.parent / "examples"
_FINAL_REBENCHMARK_ENV_KEYS = (
    "HELION_CAP_REBENCHMARK_REPEAT",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED",
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE",
)
# Pin the architecture and SM count for config-space golden tests. The matmul-seed
# heuristics are hardware-gated and use the SM count for wave sizing, so which
# compiler seed configs get injected into ``random_population`` depends on the CI
# runner's GPU. Force a fixed H100 identity so the golden is stable across every
# runner (H100/B200/A10G) and does not shift as new hardware-sensitive seed
# heuristics are added. sm90 is used because ``examples/matmul.py`` is a clean 2-D
# static GEMM that the sm90 budget-formula seed fires on.
# Pin the arch for config-population goldens: the pointwise seed promotes to the
# autotune-off default only on PROMOTE_TARGETS (sm90/sm100), so an unpinned test
# would emit the promoted tile on H100/B200 runners and the base default on
# others. Fixing the arch keeps the expected journal deterministic everywhere.
_SM90_HARDWARE = _hardware.HardwareInfo(
    device_kind="cuda",
    hardware_name="NVIDIA H100",
    runtime_version="12.8",
    compute_capability="sm90",
)
_SM90_NUM_SM = 132


def _cute_flash_test_config_spec() -> SimpleNamespace:
    return SimpleNamespace(
        cute_flash_search_enabled=True,
        compiler_coverage_groups=(),
        backend=_TEST_CUTE_FLASH_BACKEND,
    )


def _stub_flash_conditional_proposals(search: LFBOPatternSearch) -> None:
    """Use an orchestration fixture's scripted proposals at the current seam.

    These fixtures inspect qualification, retry, and composition bookkeeping.
    Real parent discovery and its replayable ledger are tested separately in
    test_conditional_parent_search.py.
    """

    def candidates(members, leaf, constraints, known, neighbor_limit):
        parent = min(members, key=search._flash_member_rank_key)
        proposed = next(
            search._pruned_pattern_search_from(
                0,
                parent,
                known,
                constraints,
                selected_limit=2,
                neighbor_limit=neighbor_limit,
                required_leaf=leaf,
                conditional_surface=True,
                disable_early_stopping=True,
            ),
            (),
        )
        return parent, [item for item in proposed if item.config != parent.config], {}

    search._flash_conditional_parent_candidates = candidates


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


def _pin_sm90(fn):
    """Run ``fn`` with the hardware identity and SM count pinned to H100, and
    evict the shared ``examples/matmul`` bound-kernel cache on entry and exit.

    The ``Kernel`` bind cache keys on the real device capability instead
    (``_device_specialization_key``). Without the eviction, a config computed under
    this pin (the sm90 budget-formula seed, e.g. ``block_sizes=[64, 64, 64],
    num_stages=6``) is cached against the real-GPU key and then reused by a later
    *un-pinned* test that actually compiles/executes it, which OOMs on a smaller GPU
    (A10G's 99KB shared memory). Clearing on entry gives this test a clean compute;
    clearing on exit guarantees no sm90-simulated config leaks into a test that runs
    on the true hardware.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        matmul = _get_examples_matmul()
        with (
            patch.object(
                _hardware, "get_hardware_info", lambda device=None: _SM90_HARDWARE
            ),
            patch("helion.runtime.get_num_sm", return_value=_SM90_NUM_SM),
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


class AutotunerTestCase(RefEagerTestDisabled, TestCase):
    """Shared setup and helpers of the autotuner test classes (test_autotuner_*.py)."""

    def setUp(self):
        super().setUp()
        random.seed(112)

    @staticmethod
    def _scalar_override() -> tuple[str, int, int]:
        """A scalar tunable of ``basic_kernels.add`` on the active backend.

        Returns ``(key, override_value, other_value)``; the override value is
        not the key's default.
        """
        if _get_backend() == "cute":
            return "cute_min_blocks_per_mp", 2, 1
        return "num_warps", 8, 4

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
        # CuTe has no precompile step (the search clears autotune_precompile),
        # so nothing is serialized for a spawn worker.
        if check_spawn and _get_backend() != "cute":
            run_mode("spawn", expect_error=True)

    @staticmethod
    def _rejecting_baseline_search(
        reject: Callable[[int, int], bool],
        attempted: list[int],
        error: Exception | None = None,
    ) -> tuple[FiniteSearch, tuple[torch.Tensor, torch.Tensor], int]:
        """A search whose baseline compiles are rejected by *reject*.

        ``reject`` is called with the attempted block size and the reference
        config's block size.  A rejected compile raises *error*, by default the
        ``OutOfResources`` Intel's backend raises when a tile overflows its
        per-thread scratch space.  ``_prepare()`` builds the benchmark provider
        (and with it the accuracy baseline) but does not compile the candidate
        configs, so replacing ``compile_config`` here only intercepts the
        baseline.
        """

        @helion.kernel(autotune_log_level=0, autotune_benchmark_subprocess=False)
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(a)
            for tile in hl.tile(out.size()):
                out[tile] = a[tile] + b[tile]
            return out

        args = (
            torch.randn([1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        bound = add.bind(args)
        reference_block = bound.config_spec.autotune_reference_config()["block_sizes"][
            0
        ]
        original = bound.compile_config

        def compile_config(config: helion.Config, **kwargs: object) -> object:
            block = config["block_sizes"][0]
            attempted.append(block)
            if reject(block, reference_block):
                if error is not None:
                    raise error
                raise OutOfResources(406912, 262144, "per-thread scratch space")
            return original(config, **kwargs)  # pyrefly: ignore[bad-argument-type]

        search = FiniteSearch(
            bound,
            args,
            configs=[
                helion.Config(block_sizes=[16], num_warps=4),
                helion.Config(block_sizes=[32], num_warps=4),
            ],
        )
        search.kernel.compile_config = compile_config  # pyrefly: ignore[bad-assignment]
        return search, args, reference_block
