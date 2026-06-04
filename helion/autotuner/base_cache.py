from __future__ import annotations

import abc
import dataclasses
import functools
import gc
import hashlib
import logging
import math
import os
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Hashable

import torch
from torch._inductor.codecache import build_code_hash
from torch._inductor.codecache import torch_key

from .. import exc
from .._utils import counters
from .base_search import BaseAutotuner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from .base_search import BaseSearch

log: logging.Logger = logging.getLogger(__name__)


def should_skip_cache() -> bool:
    """Return True when the user has requested that cache reads be skipped."""
    return os.environ.get("HELION_SKIP_CACHE", "").strip().lower() not in {
        "",
        "0",
        "false",
    }


class AutotuneCacheMeta(abc.ABCMeta):
    """Metaclass that enables the Cache[Search] syntax for autotuner cache classes."""

    def __getitem__(
        cls, search_cls: type[BaseSearch]
    ) -> Callable[[BoundKernel, Sequence[Any]], BaseAutotuner]:
        """Enable Cache[Search] syntax to create a factory function.

        Args:
            search_cls: The search class to use with this cache

        Returns:
            A factory function that creates cache instances with the specified search
        """

        def factory(kernel: BoundKernel, args: Sequence[Any]) -> BaseAutotuner:
            if not kernel.is_cacheable():
                raise TypeError(
                    f"Autotune caches require a cacheable kernel "
                    f"(e.g. BoundKernel); got {type(kernel).__name__}. "
                    f"External kernels are not cacheable."
                )

            def autotuner_factory() -> BaseSearch:
                return search_cls(kernel, args)

            return cls(autotuner_factory(), autotuner_factory=autotuner_factory)  # type: ignore[misc]

        return factory


@functools.cache
def helion_key() -> str:
    here = os.path.abspath(__file__)
    helion_path = os.path.dirname(os.path.dirname(here))

    combined_hash = hashlib.sha256()
    build_code_hash([helion_path], "", combined_hash)
    return combined_hash.hexdigest()


@functools.cache
def torch_key_wrapper() -> str:
    return torch_key().hex()


@functools.cache
def triton_key_wrapper() -> str:
    from torch._inductor.runtime.triton_compat import triton_key

    full_key = triton_key()
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


class CacheKeyBase:
    """
    Base class to provide utility functions to all cache key dataclasses
    """

    def stable_hash(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class BoundKernelInMemoryCacheKey(CacheKeyBase):
    """
    Default in memory cache key.

    This key includes:

    specialization_key: Information about all kernel inputs.
                        For tensors this means their device, shape, size etc.
    extra_results: Information regarding `hl.specialize` decisions
    """

    specialization_key: tuple[Hashable, ...]
    extra_results: tuple[Hashable, ...]


@dataclasses.dataclass(frozen=True)
class LooseAutotuneCacheKey(BoundKernelInMemoryCacheKey):
    """
    Autotune Cache key to use for most use cases.

    This key includes (in addition to BoundKernelInMemoryCacheKey):

    kernel_source_hash: Hash of source code of input Helion kernel
    hardware: Hardware of the input device
    runtime_name: Version of the cuda/rocm arch
    backend: Kernel backend (e.g. triton, pallas)
    config_spec_hash: Hash of the config spec (available knobs and their ranges)
    extra_cache_key: Optional extra cache key from the kernel (e.g. fusion context hash)
    best_of_k: Number of autotune trials run with different random seeds; the
        winning config across trials is the one cached. Default ``1`` (the
        historical single-trial path) is hidden from ``repr`` so K=1 cache
        hashes stay byte-identical to entries written before this field existed.
    """

    kernel_source_hash: str
    hardware: str
    runtime_name: str
    backend: str
    config_spec_hash: str = ""
    extra_cache_key: str = ""
    # ``repr=False`` keeps the field out of the dataclass-generated ``repr``;
    # ``__repr__`` below adds it only when it deviates from the default so the
    # K=1 hash matches historical bytes exactly.
    best_of_k: int = dataclasses.field(default=1, repr=False)

    def __repr__(self) -> str:
        # Reproduce the dataclass-generated repr for all auto-repr fields, then
        # append ``best_of_k`` only when it is non-default. This preserves the
        # byte-identical hash for K=1 entries written before the field existed.
        auto_fields = [f for f in dataclasses.fields(self) if f.repr]
        body = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in auto_fields)
        if self.best_of_k != 1:
            body = f"{body}, best_of_k={self.best_of_k!r}"
        return f"{type(self).__name__}({body})"

    def stable_hash(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class StrictAutotuneCacheKey(LooseAutotuneCacheKey):
    """
    Autotune Cache key to use for utmost strictness in terms of re-autotuning
    when library source code changes.

    This key includes (in addition to StrictAutotuneCacheKey):

    helion_key: Hash of source code of Helion
    torch_key: Hash of source code of PyTorch
    triton_key: Hash of source code of Triton
    """

    helion_key: str = dataclasses.field(default_factory=helion_key)
    torch_key: str = dataclasses.field(default_factory=torch_key_wrapper)
    triton_key: str = dataclasses.field(default_factory=triton_key_wrapper)


class AutotuneCacheBase(BaseAutotuner, abc.ABC, metaclass=AutotuneCacheMeta):
    """
    Abstract base class that all autotune caches need to implement.
    Any user defined cache will need to extend this class, and
    provide implementations for get and put methods.
    """

    def __init__(
        self,
        autotuner: BaseSearch,
        *,
        autotuner_factory: Callable[[], BaseSearch] | None = None,
    ) -> None:
        """Initialize the cache.

        ``autotuner_factory`` is an optional zero-argument callable that
        returns a fresh inner ``BaseSearch`` instance. It is required for
        ``autotune_best_of_k > 1`` (used by :meth:`_run_autotune_trials`
        to construct one ``BaseSearch`` per trial). When ``None``,
        ``autotune_best_of_k`` must equal 1; otherwise
        :meth:`_run_autotune_trials` raises ``RuntimeError``.
        """
        self.autotuner = autotuner
        self._autotuner_factory = autotuner_factory
        kernel = self.autotuner.kernel
        if not kernel.is_cacheable():
            raise TypeError(
                f"Autotune caches require a cacheable kernel "
                f"(e.g. BoundKernel); got {type(kernel).__name__}. "
                f"External kernels are not cacheable."
            )
        self.kernel: BoundKernel = kernel  # type: ignore[assignment]
        self.args = self.autotuner.args

    @abc.abstractmethod
    def get(self) -> Config | None:
        raise NotImplementedError

    @abc.abstractmethod
    def put(self, config: Config) -> None:
        raise NotImplementedError

    def _get_cache_info_message(self) -> str:
        """Return a message describing where the cache is and how to clear it."""
        return ""

    def _should_report_cache_hit(self) -> bool:
        """Whether cache hits should be printed to stderr/autotune logs."""
        return True

    @abc.abstractmethod
    def _get_cache_key(self) -> CacheKeyBase:
        """Return the cache key for this cache instance."""
        raise NotImplementedError

    @abc.abstractmethod
    def _list_cache_entries(self) -> Sequence[tuple[str, CacheKeyBase]]:
        """Return a sequence of (description, key) tuples for all cache entries."""
        raise NotImplementedError

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """Run autotuning, consulting and updating the on-disk cache.

        ``skip_cache`` (set by HELION_FORCE_AUTOTUNE) skips reading but
        still writes back.  HELION_SKIP_CACHE skips both reading and writing.
        """
        skip_cache_env = should_skip_cache()
        skip_read = skip_cache or skip_cache_env

        if not skip_read:
            if (config := self.get()) is not None:
                counters["autotune"]["cache_hit"] += 1
                log.debug("cache hit: %s", str(config))
                if self._should_report_cache_hit():
                    kernel_decorator = self.kernel.format_kernel_decorator(
                        config, self.autotuner.settings
                    )
                    print(
                        f"Using cached config:\n\t{kernel_decorator}", file=sys.stderr
                    )
                    cache_info = self._get_cache_info_message()
                    self.autotuner.log(
                        f"Found cached config for {self.kernel.kernel.name}, skipping autotuning.\n{cache_info}"
                    )
                return config

        counters["autotune"]["cache_miss"] += 1
        log.debug("cache miss")

        if not skip_read and os.environ.get("HELION_ASSERT_CACHE_HIT") == "1":
            current_key = self._get_cache_key()
            print("\n" + "=" * 80, file=sys.stderr)
            print("HELION_ASSERT_CACHE_HIT: Cache miss detected!", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"\nKernel: {self.kernel.kernel.name}", file=sys.stderr)
            print(f"\nCurrent cache key:\n{current_key}", file=sys.stderr)

            cache_entries = self._list_cache_entries()
            if cache_entries:
                print(
                    f"\n{len(cache_entries)} other cache entries exist (but don't match):",
                    file=sys.stderr,
                )
                for i, (desc, cached_key) in enumerate(cache_entries, 1):
                    print(f"\n[Entry {i}] {desc}", file=sys.stderr)
                    print("  Key differences:", file=sys.stderr)
                    has_diff = False
                    for field_name in vars(current_key):
                        current_val = str(getattr(current_key, field_name))
                        cached_val = str(getattr(cached_key, field_name, "<missing>"))
                        if current_val != cached_val:
                            has_diff = True
                            print(f"    {field_name}:", file=sys.stderr)
                            print(f"      Current:  {current_val}", file=sys.stderr)
                            print(f"      Cached:   {cached_val}", file=sys.stderr)
                    if not has_diff:
                        print(
                            "    (no differences found, likely a hash collision)",
                            file=sys.stderr,
                        )
            else:
                print("\nNo existing cache entries found.", file=sys.stderr)

            print("=" * 80 + "\n", file=sys.stderr)
            raise exc.CacheAssertionError(self.kernel.kernel.name)

        self.autotuner.log("Starting autotuning process, this may take a while...")

        config = self._run_autotune_trials()

        if not skip_cache_env:
            self.put(config)
            counters["autotune"]["cache_put"] += 1
            log.debug("cache put: %s", str(config))

        return config

    def _release_trial_state(self) -> None:
        """Reclaim CUDA allocator state between best-of-K trials.

        ``LocalBenchmarkProvider.cleanup`` (called from
        ``BaseSearch.autotune``'s exit stack and from
        ``_run_rebench_round``) breaks the search-provider-budget-hook
        reference cycle and drops the baseline tensors, so refcounting
        can free the provider's GPU memory deterministically. This
        helper then runs a Python ``gc.collect`` pass to catch any other
        cycles, plus ``torch.cuda.empty_cache`` so the caching allocator
        returns the freed blocks to the driver pool before the next
        trial / the post-autotune steady-state measurement allocates.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _run_one_trial(self, i: int, k: int, trial_seed: int) -> tuple[Config, float]:
        """Construct a fresh trial autotuner, run it, return its results.

        The trial autotuner is scoped to this helper's local frame so it
        becomes unreachable as soon as we return. The caller then runs
        ``_release_trial_state`` to release the provider's CUDA memory
        before the next trial / rebench / ``_bench_steady`` step.
        Returns ``(trial_config, trial_low_water_perf)``; any exception
        from the inner ``autotune()`` propagates to the caller (whose
        ``try``/``finally`` still runs ``_release_trial_state``).
        """
        assert self._autotuner_factory is not None
        trial_autotuner = self._autotuner_factory()
        # Swap so logging/error-reporting reflects the active trial.
        self.autotuner = trial_autotuner
        self.autotuner.log(f"Best-of-K trial {i + 1}/{k} starting (seed={trial_seed})")
        trial_config = trial_autotuner.autotune()
        trial_low_water_perf = trial_autotuner.best_perf_so_far
        self.autotuner.log(
            f"Best-of-K trial {i + 1}/{k} complete: "
            f"low-water perf={trial_low_water_perf:.4f}ms "
            f"config={trial_config}"
        )
        return trial_config, trial_low_water_perf

    def _run_autotune_trials(self) -> Config:
        """Run the configured number of autotune trials and return the best config.

        When ``settings.autotune_best_of_k == 1`` (the default) this falls
        through to a single ``self.autotuner.autotune()`` call, leaving
        behavior byte-identical to the historical single-trial path.

        When ``settings.autotune_best_of_k > 1`` this runs K independent
        autotune trials with deterministic per-trial seeds
        (``seed = base + i for i in range(K)``). Each trial uses a fresh
        ``BaseSearch`` built via ``self._autotuner_factory`` so per-trial
        state (population, surrogate, training data, benchmark provider,
        budget timer) is isolated by construction. After all K trials
        complete, each trial's *returned* config is re-benchmarked once
        in a single fresh-provider round to pick the final winner fairly
        — using each trial's own ``best_perf_so_far`` would bias toward
        trials that happened to hit an optimistic outlier timing during
        their search.

        Cost: K full autotune wall-times + one extra K-config benchmark
        round at the end.

        After each trial and after the final rebench round, the trial's
        local autotuner is dropped and ``_release_trial_state`` runs a
        Python GC pass + ``torch.cuda.empty_cache`` to return the
        provider's freed CUDA blocks to the driver pool before the next
        allocation wave (the next trial or the post-autotune
        ``_bench_steady`` measurement). Without this, the caching
        allocator stays fragmented across the autotune and the
        post-autotune steady-state measurement sees a per-launch
        overhead that is proportionally large for fast-launch kernels.
        """
        settings = self.autotuner.settings
        k = settings.autotune_best_of_k
        if k <= 1:
            return self.autotuner.autotune()
        if self._autotuner_factory is None:
            raise RuntimeError(
                "autotune_best_of_k > 1 requires a registered _autotuner_factory; "
                "the public Cache(autotuner) constructor does not support best-of-K. "
                "Use default_autotuner_fn or wire your custom autotuner_fn to attach "
                "the factory via Cache(autotuner, autotuner_factory=...)."
            )

        base_seed = settings.autotune_random_seed
        # Snapshot any settings the autotuner mutates inside ``_prepare`` /
        # ``set_adaptive_compile_timeout`` so trial N+1 starts with the same
        # values trial 1 saw. Without this, trial 1's adaptive-timeout
        # narrowing leaks into later trials and they are not independent.
        base_compile_timeout = settings.autotune_compile_timeout
        original_autotuner = self.autotuner

        self.autotuner.log(
            f"Best-of-K autotune: running {k} trials with seeds "
            f"[{base_seed}, {base_seed + k - 1}]"
        )

        trial_results: list[tuple[int, Config, float]] = []
        try:
            for i in range(k):
                trial_seed = base_seed + i
                try:
                    settings.autotune_random_seed = trial_seed
                    settings.autotune_compile_timeout = base_compile_timeout
                    trial_config, trial_low_water_perf = self._run_one_trial(
                        i, k, trial_seed
                    )
                    trial_results.append((i, trial_config, trial_low_water_perf))
                finally:
                    settings.autotune_random_seed = base_seed
                    settings.autotune_compile_timeout = base_compile_timeout
                    # Restore the cache wrapper's original autotuner so the
                    # finished trial's autotuner has no remaining strong
                    # refs once ``_run_one_trial``'s local scope unwinds.
                    # The provider's ``cleanup`` (run inside ``autotune``'s
                    # exit stack) breaks the search-provider-budget-hook
                    # cycle and drops the baseline tensors;
                    # ``_release_trial_state`` then returns the freed
                    # CUDA blocks to the driver pool before the next
                    # trial / the post-autotune ``_bench_steady`` runs.
                    self.autotuner = original_autotuner
                    self._release_trial_state()
        finally:
            self.autotuner = original_autotuner

        # Final rebench: time each trial's returned config in a single fresh
        # benchmark round so we pick by an apples-to-apples measurement
        # rather than each trial's optimistic low-water mark.
        rebench_perfs = self._rebench_trial_configs(
            [config for (_, config, _) in trial_results]
        )

        best_idx = min(
            range(len(trial_results)),
            key=lambda j: (
                rebench_perfs[j] if math.isfinite(rebench_perfs[j]) else math.inf
            ),
        )
        best_trial_idx, best_config, _ = trial_results[best_idx]
        best_perf = rebench_perfs[best_idx]

        low_water_summary = ", ".join(f"{p:.4f}" for (_, _, p) in trial_results)
        rebench_summary = ", ".join(f"{p:.4f}" for p in rebench_perfs)
        self.autotuner.log(
            f"Best-of-K complete: picked trial {best_trial_idx + 1}/{k} "
            f"(rebench perf={best_perf:.4f}ms); "
            f"per-trial low-water perfs (ms): [{low_water_summary}]; "
            f"per-trial rebench perfs (ms): [{rebench_summary}]"
        )
        return best_config

    def _rebench_trial_configs(self, configs: list[Config]) -> list[float]:
        """Benchmark K candidate configs in a single fresh autotuner round.

        Returns one perf (ms) per input config, in input order. Used to pick
        the best-of-K winner without trusting each trial's own optimistic
        ``best_perf_so_far`` low-water mark, which can be inflated by a
        single fast outlier timing during the search.

        The rebench autotuner is scoped to a private helper so it becomes
        unreachable as soon as we return; ``_release_trial_state`` then
        releases CUDA memory before the caller proceeds to the
        post-autotune ``_bench_steady`` measurement.
        """
        perfs = self._run_rebench_round(configs)
        self._release_trial_state()
        return perfs

    def _run_rebench_round(self, configs: list[Config]) -> list[float]:
        """Build the rebench autotuner, time the configs, return their perfs.

        Scoped to its own frame so the rebench autotuner becomes
        unreachable when this returns and the caller's
        ``_release_trial_state`` pass can return its CUDA blocks to the
        driver pool. Note that ``benchmark_provider.cleanup`` (called in
        the ``finally`` below) is what actually drops the provider's
        baseline tensors and breaks the search-provider-budget-hook
        reference cycle; this helper just controls scope so the released
        memory is reclaimed before the caller's ``empty_cache`` runs.
        """
        assert self._autotuner_factory is not None
        rebench_autotuner = self._autotuner_factory()
        # ``_prepare`` constructs the ``benchmark_provider``; ``setup`` /
        # ``cleanup`` bracket the provider's lifetime. Mirror the lifecycle
        # ``BaseSearch.autotune`` runs internally.
        rebench_autotuner._prepare()
        rebench_autotuner.benchmark_provider.setup()
        try:
            results = rebench_autotuner.benchmark_batch(
                configs, desc="Best-of-K rebench"
            )
        finally:
            rebench_autotuner.benchmark_provider.cleanup()
        return [r.perf for r in results]
