from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import functools
import logging
import math
from math import inf
import os
import random
import re
import sys
import time
import types
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal
from typing import Protocol
from typing import cast
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map_only

from .. import exc
from .._compat import extract_device
from .._compat import get_device_name
from ..runtime.settings import _env_get_int
from .benchmark_provider import BenchmarkProvider
from .benchmark_provider import BenchmarkResult
from .benchmark_provider import LocalBenchmarkProvider
from .benchmark_provider import _clone_args
from .benchmark_provider import _unset_fn
from .benchmarking import clear_jit_fast_path_caches
from .benchmarking import interleaved_bench
from .ir_features import IrGraphRecord
from .ir_features import extract_ir_graph
from .logger import AutotuningLogger
from .metrics import AutotuneMetrics
from .metrics import KernelMetadata
from .metrics import _run_post_autotune_hooks
from .precompile_future import PrecompileFuture as PrecompileFuture
from helion._dist_utils import all_gather_object
from helion._dist_utils import is_master_rank
from helion._dist_utils import sync_object

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from . import ConfigSpec
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig
    from .local_cache import SavedBestConfig
    from helion.autotuner.effort_profile import AutotuneEffortProfile


_DATASET_NO_LOG_WARNED = False


def _warn_dataset_without_log(log: AutotuningLogger) -> None:
    """Warn once: ``autotune_dataset`` needs ``autotune_log`` (the base path the
    ``.meta.jsonl`` sits next to) or nothing is collected."""
    global _DATASET_NO_LOG_WARNED
    if _DATASET_NO_LOG_WARNED:
        return
    _DATASET_NO_LOG_WARNED = True
    log.warning(
        "HELION_AUTOTUNE_DATASET is set but HELION_AUTOTUNE_LOG is not; no "
        "autotune dataset will be collected. Set HELION_AUTOTUNE_LOG to a base "
        "path to enable collection."
    )


# Use the standard do_bench effort for confirmation instead of the adaptive
# rebenchmark repeat, which can amplify a single optimistic subprocess timing.
_SUSPICIOUS_REBENCHMARK_WARMUP = 25
_SUSPICIOUS_REBENCHMARK_REP = 100


class _HasDeviceAndProcessGroupName(Protocol):
    device: torch.device
    process_group_name: str | None


class _AutotunableKernel(Protocol):
    @property
    def config_spec(self) -> ConfigSpec: ...

    @property
    def settings(self) -> Settings: ...

    @property  # pyrefly: ignore[bad-return]
    def env(self) -> _HasDeviceAndProcessGroupName: ...

    @property
    def configs(self) -> Sequence[Config]: ...

    def compile_config(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        allow_print: bool = True,
    ) -> Callable[..., object]:
        """Compile a kernel for the given config, used for accuracy checking."""
        ...

    def bench_compile_config(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        allow_print: bool = True,
    ) -> Callable[..., object]:
        """Compile a kernel for the given config, used for benchmarking.

        By default this is the same as compile_config. Override to return
        a different callable for benchmarking, e.g. a fused kernel that
        includes prologue/epilogue code from Inductor.
        """
        ...

    def format_kernel_decorator(self, config: Config, settings: Settings) -> str: ...

    def get_cached_path(self, config: Config | None = None) -> str | None: ...

    def to_triton_code(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str | None: ...

    def maybe_log_repro(
        self,
        log_func: Callable[[str], None],
        args: Sequence[object],
        config: Config | None = None,
    ) -> None: ...

    def extra_cache_key(self) -> str:
        """Return extra data folded into the disk-cache key.

        Implementations should return ``""`` to leave the cache key
        unchanged, or a non-empty string to differentiate cache entries
        for the same kernel source and args.
        """
        ...

    def supports_subprocess_benchmark(self) -> bool:
        """Whether autotuning may benchmark compiled configs in a subprocess."""
        ...

    def is_cacheable(self) -> bool:
        """Whether this kernel supports the autotuning disk cache."""
        ...


_CODE_OBJECT_RE = re.compile(r"<code object .+?, line \d+>")


class _CodeSentinel:
    """Stable stand-in for types.CodeType so spec key comparison is repr-independent."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<code>"


_CODE_SENTINEL = _CodeSentinel()


def normalize_autotune_seed_configs(settings: Settings) -> tuple[Config, ...]:
    """Return user-provided autotune seed configs from settings as concrete Configs."""
    from ..runtime.config import Config

    seed_configs = settings.autotune_seed_configs
    if seed_configs is None:
        return ()
    if isinstance(seed_configs, Config):
        return (seed_configs,)
    if isinstance(seed_configs, dict):
        return (Config.from_dict(seed_configs),)
    return tuple(
        Config.from_dict(seed_config) if isinstance(seed_config, dict) else seed_config
        for seed_config in seed_configs
    )


def _normalize_spec_key(key: object) -> object:
    """Replace types.CodeType with a stable sentinel in a spec key tree."""
    return tree_map_only(types.CodeType, lambda _: _CODE_SENTINEL, key)


def _normalize_spec_key_str(s: str) -> str:
    """Normalize a specialization_key string for cache comparison.

    Replaces code object repr strings with a stable '<code>' sentinel,
    allowing FROM_BEST_AVAILABLE to match function arguments based
    on their closure values only, ignoring code object identity.
    """
    return _CODE_OBJECT_RE.sub("<code>", s)


class BaseAutotuner(abc.ABC):
    """
    Abstract base class for all autotuners and classes that wrap autotuners, like caching.
    """

    @abc.abstractmethod
    def autotune(self, *, skip_cache: bool = False) -> Config:
        raise NotImplementedError


class BaseSearch(BaseAutotuner):
    """
    Base class for search algorithms. This class defines the interface and utilities for all
    search algorithms.

    Attributes:
        kernel: The kernel to be tuned (any ``_AutotunableKernel``).
        settings: The settings associated with the kernel.
        config_spec: The configuration specification for the kernel.
        args: The arguments to be passed to the kernel.
        counters: A counter to track various metrics during the search.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        benchmark_provider_cls: type[BenchmarkProvider] = LocalBenchmarkProvider,
    ) -> None:
        """
        Initialize the BaseSearch object.

        Args:
            kernel: The kernel to be tuned.
            args: The arguments to be passed to the kernel.
        """
        super().__init__()
        self.kernel = kernel
        self.settings: Settings = kernel.settings
        self.config_spec: ConfigSpec = kernel.config_spec
        self.args: Sequence[object] = args
        self.log = AutotuningLogger(self.settings)
        self.best_perf_so_far = inf
        self._benchmark_provider_cls = benchmark_provider_cls
        self._prepared = False
        self._skip_cache = False
        self._autotune_budget_start: float | None = None

    def _prepare(self) -> None:
        """Some initialization deferred until autotuning actually runs.

        This is called at the start of autotune() so that cache hits skip it.
        """
        if self._prepared:
            return
        self._prepared = True
        self._autotune_budget_start = time.perf_counter()
        seed = self.settings.autotune_random_seed
        random.seed(seed)
        self.log(f"Autotune random seed: {seed}")
        budget = self.settings.autotune_budget_seconds
        if budget is not None:
            self.log(f"Autotune budget: {budget}s")
        kernel_obj = getattr(self.kernel, "kernel", None)
        kernel_source = ""
        if kernel_obj is not None:
            try:
                kernel_source = kernel_obj.kernel_source()
            except OSError:
                self.log.debug("Failed to read Helion kernel source", exc_info=True)
        kernel_name = getattr(kernel_obj, "name", "")
        tensors = [arg for arg in self.args if isinstance(arg, torch.Tensor)]
        input_shapes = str([tuple(t.shape) for t in tensors])
        dtypes = str([str(t.dtype) for t in tensors])
        hardware = get_device_name(extract_device(self.args)) or ""
        self._autotune_metrics: AutotuneMetrics = AutotuneMetrics(
            kernel_name=kernel_name,
            kernel_source=kernel_source,
            input_shapes=input_shapes,
            hardware=hardware,
            random_seed=self.settings.autotune_random_seed,
            search_algorithm=type(self).__name__,
        )
        # Per-run identity for the <autotune_log>.meta.jsonl record (written when
        # dataset collection is on); CSV rows join to it on run_id. The sink adds
        # the configs tested.
        self._kernel_metadata: KernelMetadata = KernelMetadata(
            kernel_name=kernel_name,
            kernel_source=kernel_source,
            input_shapes=input_shapes,
            dtypes=dtypes,
            hardware=hardware,
            settings=self.settings.to_dict(),
        )
        # Device IR is config-independent and captured once here
        self._ir_graph: IrGraphRecord | None = self._extract_ir_graph()
        self.benchmark_provider = self._benchmark_provider_cls(
            kernel=self.kernel,
            settings=self.settings,
            config_spec=self.config_spec,
            args=self.args,
            log=self.log,
            autotune_metrics=self._autotune_metrics,
        )
        self.benchmark_provider.set_budget_exceeded_fn(self._autotune_budget_exceeded)

    def _is_restricted_search(self) -> bool:
        """Whether the search space is the user's pinned configs.

        A kernel decorated with ``configs=[...]`` (and not ``force_autotune``)
        tunes only between those user-chosen configs, so its telemetry is a
        biased, non-representative slice; such runs are excluded from data
        collection. ``force_autotune`` searches the full space and is collected.
        Best-effort: a missing kernel object / ``configs`` attribute reads as
        "not restricted" so a genuine search is never dropped by accident.
        """
        kernel_obj = getattr(self.kernel, "kernel", None)
        configs = getattr(kernel_obj, "configs", None)
        return bool(configs) and not self.settings.force_autotune

    def _extract_ir_graph(self) -> IrGraphRecord | None:
        """Best-effort, config-independent device-IR dump for the .meta.jsonl record.
        Returns ``None`` unless dataset collection is on.
        """
        if not (
            self.settings.autotune_log
            and self.settings.autotune_dataset
            and not self._is_restricted_search()
        ):
            return None
        host_function = getattr(self.kernel, "host_function", None)
        # Read the backing field, not the device_ir property
        device_ir = getattr(host_function, "_device_ir", None)
        if device_ir is None:
            return None
        try:
            return extract_ir_graph(device_ir)
        except Exception:
            self.log.debug("Failed to extract device IR features", exc_info=True)
            return None

    def _autotune_budget_exceeded(self) -> bool:
        budget = self.settings.autotune_budget_seconds
        if budget is None or self._autotune_budget_start is None:
            return False
        elapsed = time.perf_counter() - self._autotune_budget_start
        if elapsed < budget:
            return False
        self.log(
            f"Autotune budget {budget}s exceeded "
            f"(elapsed {elapsed:.1f}s); returning best-so-far."
        )
        return True

    def _budgeted_range(self, *args: int) -> Iterator[int]:
        """Yield ``range(*args)`` until the autotune budget is exhausted."""
        for value in range(*args):
            if self._autotune_budget_exceeded():
                return
            yield value

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        """
        Retrieve extra kwargs from the effort profile for the autotuner.
        """
        kwargs: dict[str, object] = {}

        if settings.autotune_max_generations is not None:
            kwargs.setdefault("max_generations", settings.autotune_max_generations)

        return kwargs

    def set_adaptive_compile_timeout(
        self,
        members: list[PopulationMember],
        min_seconds: float,
        quantile: float,
    ) -> None:
        """
        Compute and set an adaptive compile timeout based on observed compile times.

        Uses the specified quantile of compile times from the population:
            adaptive_timeout = min(max(quantile_value, min_seconds), original_timeout)

        This feature must be enabled via the setting autotune_adaptive_timeout=True
        or the environment variable HELION_AUTOTUNE_ADAPTIVE_TIMEOUT=1.

        Args:
            members: List of population members with compile_time information.
            min_seconds: Lower bound for the adaptive timeout in seconds.
            quantile: The quantile of compile times to use (e.g., 0.9 for 90th percentile).
        """
        if not self.settings.autotune_adaptive_timeout:
            return

        # Collect valid compile times (non-None and positive)
        compile_times = [
            m.compile_time
            for m in members
            if m.compile_time is not None and m.compile_time > 0
        ]

        if not compile_times:
            self.log("No valid compile times found, keeping default timeout")
            return

        original_timeout = self.settings.autotune_compile_timeout

        # Compute the quantile
        compile_times_sorted = sorted(compile_times)
        quantile_index = min(
            int(len(compile_times_sorted) * quantile),
            len(compile_times_sorted) - 1,
        )
        quantile_value = compile_times_sorted[quantile_index]

        # adaptive_timeout = min(max(quantile_value, min_seconds), original_timeout)
        adaptive_timeout = int(min(max(quantile_value, min_seconds), original_timeout))

        self.settings.autotune_compile_timeout = adaptive_timeout

        self.log(
            f"Adaptive compile timeout: {adaptive_timeout}s "
            f"({quantile:.0%} percentile={quantile_value:.1f}s, "
            f"bounds=[{min_seconds}s, {original_timeout}s])"
        )

    def _apply_config_filter(
        self, configs: list[Config]
    ) -> tuple[list[Config], list[int]]:
        """Apply the user-provided config filter, returning passing configs and their indices."""
        config_filter = self.settings.autotune_config_filter
        if config_filter is None:
            return configs, list(range(len(configs)))
        filtered: list[Config | None] = [config_filter(c) for c in configs]
        passing_indices = [i for i, fc in enumerate(filtered) if fc is not None]
        passing_configs = cast(
            "list[Config]",
            [filtered[i] for i in passing_indices],
        )
        return passing_configs, passing_indices

    def benchmark_batch(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        """Compile and benchmark a batch of configurations.

        Applies the config filter, delegates to the provider, and tracks
        best performance.

        Args:
            configs: A list of configurations to benchmark.
            desc: Description for the progress bar.

        Returns:
            A list of BenchmarkResult entries, one per input config.
        """
        passing_configs, passing_indices = self._apply_config_filter(configs)
        inner_results = self.benchmark_provider.benchmark(passing_configs, desc=desc)

        if len(passing_indices) == len(configs):
            results = inner_results
        else:
            inner_iter = iter(inner_results)
            passing_set = set(passing_indices)
            results = []
            for i, config in enumerate(configs):
                if i in passing_set:
                    results.append(next(inner_iter))
                else:
                    self.log.debug(
                        f"Config filtered out by autotune_config_filter: {config!r}"
                    )
                    results.append(
                        BenchmarkResult(
                            config=config,
                            fn=lambda *a, **kw: None,
                            perf=inf,
                            status="filtered",
                            compile_time=None,
                        )
                    )

        for r in results:
            if r.perf < self.best_perf_so_far:
                self.best_perf_so_far = r.perf

        return results

    def benchmark(self, config: Config) -> BenchmarkResult:
        """Compile and benchmark a single configuration.

        Convenience wrapper around ``benchmark_batch`` for the
        single-config case.

        Args:
            config: The configuration to benchmark.

        Returns:
            A BenchmarkResult with the compiled function and performance.
        """
        return self.benchmark_batch([config])[0]

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """
        Perform autotuning to find the best configuration.

        This method searches for the optimal configuration by benchmarking multiple configurations.

        Returns:
            The best configuration found during autotuning.
        """
        self._skip_cache = skip_cache
        self._prepare()
        start = time.perf_counter()
        exit_stack = contextlib.ExitStack()
        with exit_stack:
            if self.settings.autotune_log:
                # .csv/.log follow the log path; the dataset (.meta.jsonl) also
                # needs opt-in and a representative (non-restricted) search.
                collect_dataset = (
                    self.settings.autotune_dataset and not self._is_restricted_search()
                )
                exit_stack.enter_context(
                    self.log.autotune_logging(
                        metadata=self._kernel_metadata,
                        collect_dataset=collect_dataset,
                        ir_graph=self._ir_graph,
                    )
                )
            elif self.settings.autotune_dataset:
                _warn_dataset_without_log(self.log)
            self.log.reset()
            # Autotuner triggers bugs in remote triton compile service.
            # Skip storing Triton intermediate IRs (.ttir, .ttgir, .llir, etc.)
            # during autotuning to reduce cache size by ~40%. Only binaries and
            # metadata are needed for execution.
            env_overrides = {"TRITON_LOCAL_BUILD": "1"}
            if "TRITON_STORE_BINARY_ONLY" not in os.environ:
                env_overrides["TRITON_STORE_BINARY_ONLY"] = "1"
            exit_stack.enter_context(patch.dict(os.environ, env_overrides, clear=False))
            self.benchmark_provider.setup()
            exit_stack.callback(self.benchmark_provider.cleanup)
            try:
                best = self._autotune()
            finally:
                self._finalize_autotune_metrics()
        end = time.perf_counter()
        kernel_decorator = self.kernel.format_kernel_decorator(best, self.settings)

        self.log(
            f"Autotuning complete in {end - start:.1f}s after searching {self._autotune_metrics.num_configs_tested} configs.\n"
            "One can hardcode the best config and skip autotuning with:\n"
            f"    {kernel_decorator}\n",
            level=logging.INFO + 5,
        )
        if self._autotune_metrics.num_accuracy_failures:
            self.log.warning(
                f"{self._autotune_metrics.num_accuracy_failures} of "
                f"{self._autotune_metrics.num_configs_tested} configs failed due "
                "to accuracy checks."
            )
        if self._autotune_metrics.num_compile_failures:
            self.log.warning(
                f"{self._autotune_metrics.num_compile_failures} of "
                f"{self._autotune_metrics.num_configs_tested} configs failed due "
                "to compile failures."
            )
        cached_path = self.kernel.get_cached_path(best)
        if cached_path is not None and is_master_rank():
            self.log(f"Code of selected kernel: {cached_path}")
        self.kernel.maybe_log_repro(self.log.warning, self.args, best)
        if self.settings.print_output_code:
            triton_code = self.kernel.to_triton_code(best)
            if triton_code is not None:
                print(triton_code, file=sys.stderr)
        return best

    def _get_current_hardware_and_specialization(
        self,
    ) -> tuple[str | None, str | None]:
        """Return (hardware, specialization_key) for matching cached configs."""
        hardware = get_device_name(extract_device(self.args))

        inner_kernel = getattr(self.kernel, "kernel", None)
        if inner_kernel is None or not hasattr(
            inner_kernel, "_base_specialization_key"
        ):
            return hardware, None
        spec_key = inner_kernel._base_specialization_key(self.args)
        specialization_key = str(_normalize_spec_key(spec_key))

        return hardware, specialization_key

    def _find_similar_cached_configs(self, max_configs: int) -> list[SavedBestConfig]:
        """Return cached configs matching hardware, specialization_key, and config_spec_hash; empty if cache is skipped.

        Scans the local cache first; if more configs are needed and a remote
        backend is configured, queries it via ``RemoteCacheBackend.list()``
        and merges the results, deduplicating against local entries.
        """
        from .base_cache import should_skip_cache

        if self._skip_cache or should_skip_cache():
            return []

        from .local_cache import get_helion_cache_dir
        from .local_cache import iter_cache_entries
        from .local_cache import parse_cache_entry
        from .remote_cache import _load_remote_backend_if_configured

        current_hardware, current_spec_key = (
            self._get_current_hardware_and_specialization()
        )
        if current_hardware is None or current_spec_key is None:
            return []

        current_fingerprint_hash = self.config_spec.structural_fingerprint_hash(
            advanced_controls_files=self.settings.autotune_search_acf or None
        )

        def is_compatible(entry: SavedBestConfig) -> bool:
            return (
                entry.flat_config is not None
                and entry.hardware == current_hardware
                and _normalize_spec_key_str(entry.specialization_key)
                == current_spec_key
                and entry.config_spec_hash == current_fingerprint_hash
            )

        matching: list[SavedBestConfig] = []
        seen: set[str] = set()

        def consider(entry: SavedBestConfig) -> bool:
            """Return True once we have enough matches to stop scanning."""
            if not is_compatible(entry):
                return False
            assert entry.flat_config is not None
            dedup_key = repr(entry.flat_config)
            if dedup_key in seen:
                return False
            matching.append(entry)
            seen.add(dedup_key)
            return len(matching) >= max_configs

        for entry in iter_cache_entries(
            get_helion_cache_dir(),
            max_scan=self.settings.autotune_best_available_max_cache_scan,
        ):
            if consider(entry):
                return matching

        backend = _load_remote_backend_if_configured()
        if backend is None:
            return matching

        # Over-fetch 2x to absorb filtered / duplicate entries.
        remaining = max_configs - len(matching)
        try:
            # Materialize eagerly so any backend failure surfaces here, not
            # mid-iteration when partial results are already in `matching`.
            raw_entries = list(backend.list(max_results=remaining * 2))
        except Exception:
            self.log.warning(
                "Remote cache list failed, using local matches only", exc_info=True
            )
            return matching

        for raw in raw_entries:
            try:
                entry = parse_cache_entry(raw)
            except ValueError:
                continue
            if consider(entry):
                break

        return matching

    def _autotune(self) -> Config:
        """
        Abstract method to perform the actual autotuning.

        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    def _autotune_seed_configs(self) -> Sequence[Config]:
        """Return user-provided autotune seed configs normalized from settings."""
        return normalize_autotune_seed_configs(self.settings)

    def set_generation(self, generation: int) -> None:
        self._autotune_metrics.num_generations = generation

    def _finalize_autotune_metrics(self) -> None:
        self._autotune_metrics.best_perf_ms = (
            self.best_perf_so_far if math.isfinite(self.best_perf_so_far) else 0.0
        )
        self._autotune_metrics.finalize()
        _run_post_autotune_hooks(self._autotune_metrics)


def check_population_consistency(
    population: Sequence[PopulationMember],
    process_group_name: str | None = None,
) -> None:
    if os.getenv("HELION_DEBUG_DISTRIBUTED") != "1" or not dist.is_initialized():
        return

    # remove unpickled fields
    sanitized_population = tuple((p.config, p.perfs) for p in population)
    all_sanitized_population = all_gather_object(
        sanitized_population, process_group_name=process_group_name
    )
    if all_sanitized_population != all_sanitized_population[:1] * len(
        all_sanitized_population
    ):
        raise exc.InconsistantConfigsAcrossRanks


@dataclasses.dataclass
class PopulationMember:
    """
    Represents a member of the population in population-based search algorithms.

    Attributes:
        perfs (list[float]): The performance of the configuration, accumulated over multiple benchmarks.
        flat_values (FlatConfig): The flat representation of the configuration values.
        config (Config): The full configuration object.
        compile_time (float | None): The compilation time for this configuration.
    """

    fn: Callable[..., object]
    perfs: list[float]
    flat_values: FlatConfig
    config: Config
    status: Literal[
        "ok", "error", "timeout", "peer_compilation_fail", "filtered", "unknown"
    ] = "unknown"
    compile_time: float | None = None

    @property
    def perf(self) -> float:
        return self.perfs[-1]


def performance(member: PopulationMember) -> float:
    """
    Retrieve the performance of a population member.  Used as a sort key.

    Args:
        member: The population member.

    Returns:
        The performance of the member.
    """
    return member.perf


class PopulationBasedSearch(BaseSearch):
    """
    Base class for search algorithms that use a population of configurations.

    Attributes:
        population (list[PopulationMember]): The current population of configurations.
        flat_spec (list[ConfigSpecFragment]): The flattened configuration specification.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        finishing_rounds: int = 0,
    ) -> None:
        """
        Initialize the PopulationBasedSearch object.

        Args:
            kernel: The kernel to be tuned.
            args: The arguments to be passed to the kernel.
            finishing_rounds: Number of finishing rounds to run after the main search.
        """
        super().__init__(kernel, args)
        self.finishing_rounds = finishing_rounds
        self.population: list[PopulationMember] = []
        # Population members corresponding to compiler-seeded configs from
        # ``ConfigSpec.compiler_seed_configs``.  Tracked separately so the
        # final-pick verification phase can re-include them as candidates
        # even when the surrogate-driven search has pruned them away from
        # ``self.population``.
        self._compiler_seed_members: list[PopulationMember] = []
        self._best_available_seed_configs: list[Config] = []
        self.config_gen: ConfigGeneration = self.config_spec.create_config_generation(
            overrides=self.settings.autotune_config_overrides or None,
            advanced_controls_files=self.settings.autotune_search_acf or None,
            process_group_name=kernel.env.process_group_name,
        )

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        """
        Retrieve extra kwargs from the effort profile for the autotuner.
        """
        from ..runtime.settings import _env_get_optional_int

        finishing_rounds = _env_get_optional_int("HELION_AUTOTUNE_FINISHING_ROUNDS")
        if finishing_rounds is None:
            finishing_rounds = profile.finishing_rounds

        return {
            "finishing_rounds": finishing_rounds,
            **super().get_kwargs_from_profile(profile, settings),
        }

    @property
    def best(self) -> PopulationMember:
        """
        Retrieve the best configuration in the population.

        Returns:
            The best population member.
        """
        return min(self.population, key=performance)

    @best.setter
    def best(self, value: PopulationMember) -> None:
        """Replace the current best member in the population."""
        idx = min(range(len(self.population)), key=lambda i: self.population[i].perf)
        self.population[idx] = value

    def capture_compiler_seed_members(
        self, members: Sequence[PopulationMember]
    ) -> None:
        """Record which ``members`` came from compiler-seeded configs (TPU only).

        Only the TPU/Pallas final-pick consumes this, so it's a no-op elsewhere.
        Matches members against ``config_gen.seed_flat_config_pairs()`` so the
        final-pick phase can re-include a seed even if the search later prunes it.
        """
        self._compiler_seed_members = []
        if not self._final_pick_supported():
            return
        try:
            seed_pairs = self.config_gen.seed_flat_config_pairs()
        except Exception:
            return
        if not seed_pairs:
            return
        seed_flats: list[FlatConfig] = [flat for flat, _config in seed_pairs]
        for member in members:
            if member.flat_values in seed_flats:
                self._compiler_seed_members.append(member)

    def benchmark_flat(self, flat_values: FlatConfig) -> PopulationMember:
        """
        Benchmark a flat configuration.

        Args:
            flat_values: The flat configuration values.

        Returns:
            A population member with the benchmark results.
        """
        config = self.config_gen.unflatten(flat_values)
        member = PopulationMember(_unset_fn, [], flat_values, config)
        self.benchmark_population([member], desc="Benchmarking")
        return member

    def benchmark_flat_batch(
        self, to_check: list[FlatConfig]
    ) -> list[PopulationMember]:
        """
        Benchmark multiple flat configurations in parallel.

        The returned list has the same length as ``to_check`` and preserves
        positional correspondence.  Invalid configurations that cannot be
        unflattened are represented as ``PopulationMember`` objects with
        ``perf == inf`` and ``status == "error"`` (they are not benchmarked).

        Args:
            to_check: A list of flat configurations to benchmark.

        Returns:
            A list of population members with the benchmark results, one per
            entry in *to_check*.
        """
        from ..runtime.config import Config

        valid: list[PopulationMember] = []
        result: list[PopulationMember] = []
        for flat in to_check:
            m = self.make_unbenchmarked(flat)
            if m is not None:
                valid.append(m)
                result.append(m)
            else:
                result.append(
                    PopulationMember(
                        _unset_fn, [float("inf")], flat, Config(), status="error"
                    )
                )

        self.benchmark_population(valid)
        return result

    def make_unbenchmarked(self, flat_values: FlatConfig) -> PopulationMember | None:
        """
        Create a population member with unbenchmarked configuration.  You
        should pass the result of this to benchmark_population.

        Args:
            flat_values: The flat configuration values.

        Returns:
            A population member with undefined performance, or None if the
            configuration is invalid.
        """
        try:
            config = self.config_gen.unflatten(flat_values)
        except exc.InvalidConfig:
            return None
        return PopulationMember(_unset_fn, [], flat_values, config)

    def _generate_best_available_population_flat(self) -> list[FlatConfig]:
        """
        Generate initial population using default config, explicit seed configs,
        and cached configs.

        Starts with the default configuration, then adds up to
        MAX_BEST_AVAILABLE_CONFIGS matching cached configs from previous runs.
        Explicit seed configs provided by the caller are added ahead of cached
        configs and are not suppressed by cache-skip settings. No random configs
        are added. Duplicate configs are discarded.

        Returns:
            A list of unique FlatConfig values for the initial population.
            Minimum size is 1 (default), plus any valid unique explicit seed configs
            and up to autotune_best_available_max_configs cached configs.
        """
        max_configs = self.settings.autotune_best_available_max_configs

        seen: set[Config] = set()
        default_flat = self.config_gen.default_flat()
        default_config = self.config_gen.unflatten(default_flat)
        seen.add(default_config)
        result: list[FlatConfig] = [default_flat]
        self.log("Starting with default config")

        # User seed configs are explicit requests, so try them before compiler-owned
        # seeds and cached configs while still deduplicating normalized configs.
        for flat, transferred_config in self.config_gen.user_seed_flat_config_pairs(
            self._autotune_seed_configs(), self.log
        ):
            if transferred_config not in seen:
                seen.add(transferred_config)
                result.append(flat)

        # Compiler-owned seeds come from ConfigSpec.compiler_seed_configs;
        # they encode backend/compiler heuristics and complement user seed configs.
        for flat, transferred_config in self.config_gen.seed_flat_config_pairs(
            self.log
        ):
            if transferred_config not in seen:
                seen.add(transferred_config)
                result.append(flat)

        for config in self._best_available_seed_configs:
            try:
                flat = self.config_gen.flatten(config)
                transferred_config = self.config_gen.unflatten(flat)
                if transferred_config not in seen:
                    seen.add(transferred_config)
                    result.append(flat)
            except (ValueError, TypeError, KeyError, AssertionError) as e:
                self.log(f"Failed to transfer explicit seed config: {e}")

        cached_entries = self._find_similar_cached_configs(max_configs)

        if cached_entries:
            self.log.debug(
                f"Found {len(cached_entries)} cached config(s) from previous runs"
            )

        duplicates = 0
        for i, entry in enumerate(cached_entries):
            try:
                self.log.debug(f"Cached config {i + 1}: {entry.config}")
                flat = entry.to_mutable_flat_config()
                transferred_config = self.config_gen.unflatten(flat)
                if transferred_config in seen:
                    duplicates += 1
                    self.log.debug(
                        f"Cached config {i + 1} is a duplicate, skipping: {transferred_config}"
                    )
                    continue
                seen.add(transferred_config)
                result.append(flat)
                self.log.debug(
                    f"Cached config {i + 1} (transferred): {transferred_config}"
                )
            except (
                ValueError,
                TypeError,
                KeyError,
                AssertionError,
                exc.InvalidConfig,
            ) as e:
                self.log(f"Failed to transfer cached config {i + 1}: {e}")
                continue

        if duplicates > 0:
            self.log.debug(f"Discarded {duplicates} duplicate config(s)")

        self.log(f"Initial population: {len(result)} total")

        return result

    def set_best_available_seed_configs(
        self,
        configs: Sequence[Config],
    ) -> None:
        self._best_available_seed_configs = list(configs)

    def benchmark_population(
        self, members: list[PopulationMember], *, desc: str = "Benchmarking"
    ) -> list[PopulationMember]:
        """
        Benchmark multiple population members in parallel.  Members should be created with make_unbenchmarked.

        Args:
            members: The list of population members to benchmark.
            desc: Description for the progress bar.
        """
        results = self.benchmark_batch([m.config for m in members], desc=desc)
        for member, result in zip(members, results, strict=True):
            assert result.config is member.config
            member.perfs.append(result.perf)
            member.fn = result.fn
            member.status = result.status
            member.compile_time = result.compile_time
        return members

    def compare(self, a: PopulationMember, b: PopulationMember) -> int:
        """
        Compare two population members based on their performance, possibly with re-benchmarking.

        Args:
            a: The first population member.
            b: The second population member.

        Returns:
            -1 if a is better than b, 1 if b is better than a, 0 if they are equal.
        """
        if self.should_rebenchmark(a) and self.should_rebenchmark(b):
            self.rebenchmark([a, b])
        return (a.perf > b.perf) - (a.perf < b.perf)

    def should_rebenchmark(self, member: PopulationMember) -> bool:
        """
        Determine if a population member should be re-benchmarked to avoid outliers.

        Args:
            member: The population member to check.

        Returns:
            True if the member should be re-benchmarked, False otherwise.
        """
        threshold = self.settings.get_rebenchmark_threshold()
        return member.perf < threshold * self.best_perf_so_far and math.isfinite(
            member.perf
        )

    def rebenchmark(
        self, members: list[PopulationMember], *, desc: str = "Rebenchmarking"
    ) -> None:
        """
        Re-benchmark a list of population members to avoid outliers.

        Args:
            members: The list of population members to rebenchmark.
            desc: Description for the progress bar.
        """
        if len(members) < 2:
            return

        # Calculate repeat count based on best performance
        base_repeat = (
            int(200 / self.best_perf_so_far)
            if math.isfinite(self.best_perf_so_far) and self.best_perf_so_far > 0
            else 1000
        )
        repeat = min(1000, max(3, base_repeat))
        if (capstr := os.getenv("HELION_CAP_REBENCHMARK_REPEAT")) is not None:
            repeat = min(repeat, int(capstr))
        if len(self.benchmark_provider.mutated_arg_indices) > 0:
            benchmark_args = _clone_args(
                self.args,
                self.kernel.env.process_group_name,
                idx_to_clone=self.benchmark_provider.mutated_arg_indices,
            )
        else:
            benchmark_args = self.args

        def make_rebenchmark_callable(member: PopulationMember) -> Callable[[], object]:
            run_member = functools.partial(member.fn, *benchmark_args)

            def wrapped() -> object:
                try:
                    return run_member()
                finally:
                    clear_jit_fast_path_caches(member.fn, self.log)

            return wrapped

        iterator = [make_rebenchmark_callable(m) for m in members]
        _backend = getattr(getattr(self, "config_spec", None), "backend", None)
        interleaved_benchmark = (
            _backend.get_interleaved_bench() if _backend is not None else None
        ) or interleaved_bench
        benchmark_function: Callable[..., list[float]] = (
            self.settings.autotune_benchmark_fn or interleaved_benchmark
        )
        try:
            if self.settings.autotune_progress_bar:
                new_timings = benchmark_function(iterator, repeat=repeat, desc=desc)
            else:
                new_timings = benchmark_function(iterator, repeat=repeat)
        finally:
            for m in members:
                clear_jit_fast_path_caches(m.fn, self.log)
        new_timings = self._confirm_suspicious_rebenchmark_timings(
            members,
            new_timings,
            desc=desc,
        )
        new_timings = sync_object(
            new_timings, process_group_name=self.kernel.env.process_group_name
        )
        for m, t in zip(members, new_timings, strict=True):
            m.perfs.append(t)
            if t < self.best_perf_so_far:
                self.best_perf_so_far = t

    def _confirm_suspicious_rebenchmark_timings(
        self,
        members: list[PopulationMember],
        timings: list[float],
        *,
        desc: str,
    ) -> list[float]:
        ratio = self.settings.get_suspicious_rebenchmark_ratio()
        if ratio is None or ratio <= 0:
            return timings

        suspicious = [
            i
            for i, (member, timing) in enumerate(zip(members, timings, strict=True))
            if math.isfinite(timing)
            and math.isfinite(member.perf)
            and timing < ratio * member.perf
        ]
        if not suspicious:
            return timings

        confirmed = self.benchmark_provider.benchmark_isolated(
            [members[i].fn for i in suspicious],
            warmup=_SUSPICIOUS_REBENCHMARK_WARMUP,
            rep=_SUSPICIOUS_REBENCHMARK_REP,
            desc=f"{desc}: confirming suspicious timings",
        )
        if confirmed is None:
            return timings

        updated = list(timings)
        for i, timing in zip(suspicious, confirmed, strict=True):
            if timing is not None:
                updated[i] = timing
        return updated

    def rebenchmark_population(
        self,
        members: list[PopulationMember] | None = None,
        *,
        desc: str = "Rebenchmarking",
    ) -> None:
        """
        Re-benchmark the entire population to avoid outliers.

        Args:
            members: The list of population members to rebenchmark.
            desc: Description for the progress bar.
        """
        if members is None:
            members = self.population
        self.rebenchmark([p for p in members if self.should_rebenchmark(p)], desc=desc)

    def statistics(self) -> str:
        """
        Generate statistics for the current population.

        Returns:
            A string summarizing the population performance.
        """
        return population_statistics(self.population)

    def run_finishing_phase(
        self, best: PopulationMember, rounds: int
    ) -> PopulationMember:
        """
        Run finishing rounds to minimize the configuration by resetting attributes to defaults.

        This phase attempts to simplify the found configuration by resetting as many
        attributes as possible to their default values, while ensuring performance
        does not get worse. It's similar to pattern search but mutations only move
        towards the default configuration.

        Args:
            best: The best configuration found during the main search.
            rounds: Number of finishing rounds to run. If 0, returns best unchanged.

        Returns:
            The minimized configuration (may be the same as input if no simplifications helped).
        """
        if rounds <= 0:
            return best

        self.log(f"Starting finishing phase with {rounds} rounds")
        default_flat = self.config_gen.default_flat()
        current = best

        for round_num in self._budgeted_range(1, rounds + 1):
            simplified = False
            candidates: list[PopulationMember] = [current]

            # Generate candidates by resetting each parameter to its default
            for i in range(len(current.flat_values)):
                if current.flat_values[i] != default_flat[i]:
                    # Create a new config with this parameter reset to default
                    new_flat = [*current.flat_values]
                    new_flat[i] = default_flat[i]
                    candidate = self.make_unbenchmarked(new_flat)
                    # Only add if valid and produces a different config
                    if candidate is not None and candidate.config != current.config:
                        candidates.append(candidate)

            if len(candidates) <= 1:
                self.log(f"Finishing round {round_num}: no more parameters to simplify")
                break

            # Benchmark the candidates
            unbenchmarked = [m for m in candidates if len(m.perfs) == 0]
            if unbenchmarked:
                self.set_generation(self._autotune_metrics.num_generations + 1)
                self.benchmark_population(
                    unbenchmarked, desc=f"Finishing round {round_num}"
                )

            # Rebenchmark all candidates (including current) for fair comparison
            self.rebenchmark(candidates, desc=f"Finishing round {round_num}: verifying")

            # Log performance of each candidate at debug level
            current_perf = current.perf
            for candidate in candidates[1:]:
                delta = candidate.perf - current_perf
                delta_pct = (delta / current_perf * 100) if current_perf != 0 else 0
                status = "ok" if candidate.perf <= current_perf else "worse"
                self.log.debug(
                    f"  reset to {candidate.config}: {candidate.perf:.4f}ms "
                    f"(delta={delta:+.4f}ms, {delta_pct:+.1f}%) [{status}]"
                )

            # Collect all single-attribute resets that maintained performance
            good_candidates = [
                c
                for c in candidates[1:]
                if math.isfinite(c.perf) and c.perf <= current.perf
            ]

            if len(good_candidates) > 1:
                # Try combining all good single-attribute resets at once
                combined_flat = [*current.flat_values]
                for c in good_candidates:
                    for i in range(len(combined_flat)):
                        if c.flat_values[i] != current.flat_values[i]:
                            combined_flat[i] = c.flat_values[i]
                combined = self.make_unbenchmarked(combined_flat)
                if combined is not None and combined.config != current.config:
                    self.benchmark_population(
                        [combined],
                        desc=f"Finishing round {round_num}: combined",
                    )
                    self.rebenchmark(
                        [current, combined],
                        desc=f"Finishing round {round_num}: verifying combined",
                    )
                    if math.isfinite(combined.perf) and combined.perf <= current.perf:
                        current = combined
                        simplified = True

            if not simplified and good_candidates:
                current = good_candidates[0]
                simplified = True

            if simplified:
                self.log(
                    f"Finishing round {round_num}: simplified to {current.config}, perf={current.perf:.4f}ms"
                )
            else:
                self.log(
                    f"Finishing round {round_num}: no simplification maintained performance, stopping early"
                )
                break

        # Minimize the final config by removing values that match defaults
        minimal_config = current.config.minimize(self.config_spec)
        current = PopulationMember(
            fn=current.fn,
            perfs=current.perfs,
            flat_values=current.flat_values,
            config=minimal_config,
            status=current.status,
            compile_time=current.compile_time,
        )
        self.log(f"Finishing phase complete: final config={current.config}")
        return current

    def _final_pick_supported(self) -> bool:
        """True only on Pallas/TPU; final-pick is untested on other backends."""
        return self.config_spec.backend_name == "pallas"

    def run_final_pick_verification(
        self,
        best: PopulationMember,
        top_k: int | None = None,
    ) -> PopulationMember:
        """Re-benchmark the top-``top_k`` candidates once and re-pick the fastest.

        Beats the noisy single search-time median.  Compiler seeds are merged in.
        Knob: ``HELION_AUTOTUNE_FINAL_PICK_TOP_K`` (10).
        """
        if top_k is None:
            top_k = max(0, _env_get_int("HELION_AUTOTUNE_FINAL_PICK_TOP_K", 10))
        if top_k <= 1:
            return best

        # Candidates: ``best`` first, then the finite population + compiler seeds
        # (so a search-pruned seed still competes), ranked by perf and deduped by
        # identity, capped at top_k.  getattr covers ``__new__`` test scaffolds.
        seed_members = getattr(self, "_compiler_seed_members", []) or []
        ranked = sorted(
            (m for m in (*self.population, *seed_members) if math.isfinite(m.perf)),
            key=performance,
        )
        candidates: list[PopulationMember] = []
        seen: set[int] = set()
        for member in (best, *ranked):
            if id(member) in seen:
                continue
            seen.add(id(member))
            candidates.append(member)
            if len(candidates) >= top_k:
                break
        if len(candidates) < 2:
            return best

        # TPU/Pallas: re-rank by per-call on-device µs when available; else fall
        # back to the absolute-median rebench.
        device_micros_bench = self._resolve_device_micros_paired_bench()
        if device_micros_bench is not None:
            return self._run_final_pick_verification_device_micros(
                best, candidates, device_micros_bench=device_micros_bench
            )
        return self._rebench_and_pick(best, candidates)

    def _rebench_and_pick(
        self, best: PopulationMember, candidates: list[PopulationMember]
    ) -> PopulationMember:
        """Re-benchmark ``candidates`` by absolute median and return the fastest."""
        self.rebenchmark(candidates, desc="Final-pick verification")
        best_member = min(candidates, key=performance)
        if not math.isfinite(best_member.perf):
            return best
        if best_member is not best:
            self.log(
                f"Final-pick re-picked {best_member.config} "
                f"({best_member.perf:.4f}ms) over {best.config}"
            )
        self.best_perf_so_far = min(self.best_perf_so_far, best_member.perf)
        return best_member

    def _finalize(self) -> Config:
        """Finishing phase + final-pick re-rank; return the chosen config.

        Shared tail of the search ``_autotune`` methods; the final-pick re-rank
        runs only on TPU/Pallas (see ``_final_pick_supported``).
        """
        best = self.run_finishing_phase(self.best, self.finishing_rounds)
        if self._final_pick_supported():
            best = self.run_final_pick_verification(best)
        return best.config

    def _resolve_device_micros_paired_bench(
        self,
    ) -> Callable[..., list[tuple[float, float]]] | None:
        """Paired device-µs bench from the backend, or None.

        Pallas/TPU returns a ``jax.profiler`` closure (per-call on-chip µs) when
        ``static_shapes`` is on and ``HELION_AUTOTUNE_PALLAS_RANK_BY=device_time`` (the
        default); other backends and the ``wall_time`` opt-out return None.
        """
        # settings/config_spec are unset on the minimal final-pick test scaffolds;
        # real searches always have them.
        settings = getattr(self, "settings", None)
        config_spec = getattr(self, "config_spec", None)
        if settings is None or config_spec is None or not settings.static_shapes:
            return None
        return config_spec.backend.get_paired_device_micros_bench()

    def _run_final_pick_verification_device_micros(
        self,
        best: PopulationMember,
        candidates: list[PopulationMember],
        *,
        device_micros_bench: Callable[..., list[tuple[float, float]]],
    ) -> PopulationMember:
        """Re-rank the cohort by per-call on-chip µs instead of wall-clock.

        ``device_micros_bench`` returns ``(device_micros, delta-vs-best)`` per candidate;
        per-call device µs isn't masked by the ~125µs dispatch overhead. Picks the
        smallest delta (tie-broken by absolute device µs). Device µs is not folded
        into ``perfs`` (those are wall-clock ms). Falls back to the absolute-median
        rebench on any error.
        """
        if len(self.benchmark_provider.mutated_arg_indices) > 0:
            benchmark_args = _clone_args(
                self.args,
                self.kernel.env.process_group_name,
                idx_to_clone=self.benchmark_provider.mutated_arg_indices,
            )
        else:
            benchmark_args = self.args
        candidate_fns: list[Callable[..., object]] = [
            functools.partial(member.fn, *benchmark_args) for member in candidates
        ]
        reference_fn: Callable[..., object] = functools.partial(
            best.fn, *benchmark_args
        )
        desc = (
            "Final-pick verification device_micros"
            if self.settings.autotune_progress_bar
            else None
        )
        try:
            results = device_micros_bench(candidate_fns, reference_fn, desc=desc)
        except Exception as err:
            self.log(f"Device-µs re-rank failed ({err!r}); falling back to rebench.")
            return self._rebench_and_pick(best, candidates)

        # results[i] == (absolute device µs, paired delta vs best).
        device_micros_by_slot = [device_micros for device_micros, _ in results]
        delta_by_slot = [delta for _, delta in results]
        if not any(math.isfinite(u) and math.isfinite(d) for u, d in results):
            self.log(
                "Device-µs re-rank got no finite readings; falling back to rebench."
            )
            return self._rebench_and_pick(best, candidates)

        def _device_key(slot: int) -> tuple[float, float]:
            delta, device_micros = delta_by_slot[slot], device_micros_by_slot[slot]
            if not math.isfinite(delta) or not math.isfinite(device_micros):
                return (inf, inf)
            return (delta, device_micros)

        best_slot = min(range(len(candidates)), key=_device_key)
        if not math.isfinite(delta_by_slot[best_slot]):
            return best
        best_member = candidates[best_slot]

        if best_member is not best:
            self.log(
                f"Final-pick re-picked {best_member.config} (delta "
                f"{delta_by_slot[best_slot]:+.3f}µs, absolute "
                f"{device_micros_by_slot[best_slot]:.3f}µs) over {best.config}"
            )
        return best_member


def population_statistics(population: list[PopulationMember]) -> str:
    """
    Create a summary of the population performance.

    Args:
        population: The population of configurations.

    Returns:
        A string summarizing the performance of the population.
    """
    population = sorted(population, key=performance)
    status_counts: collections.Counter[str] = collections.Counter()
    working: list[PopulationMember] = []
    for member in population:
        status = member.status
        if math.isfinite(member.perf):
            working.append(member)
            if status not in {"ok", "error", "timeout"}:
                status = "ok"
        else:
            if status not in {"error", "timeout"}:
                status = "error"
        if status == "timeout":
            status_counts["timeout"] += 1
        elif status == "error":
            status_counts["error"] += 1
        else:
            status_counts["ok"] += 1
    if len(working) == 0:
        raise exc.NoConfigFound
    count_parts = [
        f"{label}={status_counts[label]}"
        for label in ("error", "timeout", "ok")
        if status_counts[label]
    ]
    perf_parts = (
        f"min={working[0].perf:.4f}",
        f"mid={working[len(working) // 2].perf:.4f}",
        f"max={working[-1].perf:.4f}",
    )
    lines = [f"{' '.join(count_parts)} | {' '.join(perf_parts)}", "best:"]
    best_config = dict(population[0].config)
    key_width = max((len(k) for k in best_config), default=0)
    lines.extend(f"  {k:<{key_width}} = {v!r}" for k, v in best_config.items())
    return "\n" + "\n".join(lines)
