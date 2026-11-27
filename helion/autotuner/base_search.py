from __future__ import annotations

import abc
import collections
import contextlib
import copy
import dataclasses
import functools
import hashlib
import inspect
import logging
import marshal
import math
from math import inf
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal
from typing import NamedTuple
from typing import ParamSpec
from typing import Protocol
from typing import TypeVar
from typing import cast
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten

from .. import exc
from .._compat import extract_device
from .._compat import get_device_name
from ..runtime.config import _config_key
from ..runtime.config import _config_value_key
from ..runtime.config import _copy_config_structure
from ..runtime.config_cache import ConfigCache
from ..runtime.config_cache import ConfigSet
from ..runtime.config_cache import _stored_config
from ..runtime.settings import _env_get_bool
from ..runtime.settings import _env_get_int
from ..runtime.settings import _env_get_optional_int
from ..runtime.structural_policy import StructuralConfig
from ..runtime.structural_policy import StructuralPolicyError
from ..runtime.structural_policy import bound_structural_policy
from ..runtime.structural_policy import require_same_structural_policy
from .accuracy import effective_fp8_accuracy
from .accuracy import keyable_accuracy_check
from .benchmark_prelude import BenchmarkPrelude
from .benchmark_prelude import PreludeSearchFailed
from .benchmark_prelude import active_prelude
from .benchmark_provider import _COMPILER_SEED_TIMEOUT_RETRY_LIMIT
from .benchmark_provider import BenchmarkProvider
from .benchmark_provider import BenchmarkResult
from .benchmark_provider import IsolatedBenchmarkFailure
from .benchmark_provider import IsolatedBenchmarkTiming
from .benchmark_provider import LocalBenchmarkProvider
from .benchmark_provider import MultiShapeBenchmarkProvider
from .benchmark_provider import _clone_args
from .benchmark_provider import _MultiShapeAutotuneArgs
from .benchmark_provider import _searching_candidates
from .benchmark_provider import _unset_fn
from .benchmark_worker import BenchmarkWorkerUnkillable
from .benchmarking import _BENCHMARK_CUDAGRAPH_ENV
from .benchmarking import _FINAL_L2_FLUSH_ENV
from .benchmarking import _L2_FLUSH_ENV
from .benchmarking import _TARGET_REGIME_ENV
from .benchmarking import MirroredBenchmarkTrace
from .benchmarking import _mirrored_bench_call_layout
from .benchmarking import _resolve_l2_flush_mode
from .benchmarking import clear_jit_fast_path_caches
from .benchmarking import do_bench
from .benchmarking import do_bench_generic
from .benchmarking import final_selection_l2_flush_mode
from .benchmarking import final_selection_second_regime
from .benchmarking import final_selection_target_regime
from .benchmarking import interleaved_bench
from .benchmarking import kernel_timer_available
from .benchmarking import l2_flush_sweep
from .benchmarking import mirrored_bench_generic
from .benchmarking import untraced_regime
from .compiler_refinement import MAX_REFINEMENT_TRIALS
from .compiler_refinement import compiler_refinement_candidates
from .compiler_refinement import compiler_refinement_policy
from .compiler_refinement import same_refinement_value
from .effort_profile import get_effort_profile
from .handoff import _active_handoff
from .handoff import _HandoffStop
from .handoff import _observe_handoff
from .handoff import _refresh_handoff
from .input_banks import InputBankError
from .input_banks import InputBankMemoryError
from .logger import AutotuneLogEntry
from .logger import AutotuningLogger
from .logger import capture_output
from .logger import log_candidate_compile_failure
from .metrics import AutotuneMetrics
from .metrics import KernelMetadata
from .metrics import _run_post_autotune_hooks
from .precompile_future import PrecompileFuture as PrecompileFuture
from .search_space_logger import canonical_config_id
from helion._dist_utils import all_gather_object
from helion._dist_utils import is_master_rank
from helion._dist_utils import sync_object

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from collections.abc import MutableSet
    from collections.abc import Sequence
    from typing_extensions import Self

    from .._compiler.backend import Backend
    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from ..runtime.structural_policy import StructuralPolicy
    from . import ConfigSpec
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig
    from .local_cache import SavedBestConfig
    from .search_space_logger import SearchSpaceTracker
    from helion.autotuner.effort_profile import AutotuneEffortProfile
    from helion.autotuner.effort_profile import FinalSelectionConfig

_T = TypeVar("_T")

_P = ParamSpec("_P")


class _RetryWithoutPrivateCopies(Exception):
    """Discard a finalist phase's readings after its optional copies ran out."""


def _retry_without_private_copies(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(fn)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return fn(*args, **kwargs)
        except _RetryWithoutPrivateCopies:
            pass
        # The worker was reaped and copies disabled. Starting the phase over
        # discards every earlier reading, including mirrored sweep positions.
        return fn(*args, **kwargs)

    return wrapped


@functools.cache
def _warn_dataset_without_log(log: AutotuningLogger) -> None:
    """Warn (once) that ``autotune_log_details`` needs ``autotune_log`` (the base path
    the ``.meta.jsonl`` sits next to) or nothing is collected. Cached so the
    warning fires once per logger instead of once per config."""
    log.warning(
        "HELION_AUTOTUNE_LOG_DETAILS is set but HELION_AUTOTUNE_LOG is not; no "
        "autotune dataset will be collected. Set HELION_AUTOTUNE_LOG to a base "
        "path to enable collection."
    )


# Use the standard do_bench effort for confirmation instead of the adaptive
# rebenchmark repeat, which can amplify a single optimistic subprocess timing.
_SUSPICIOUS_REBENCHMARK_WARMUP = 25
_SUSPICIOUS_REBENCHMARK_REP = 100
# A finalist timing is only "suspiciously fast" when it undercuts the search
# time by more than the batched flushed-graph timer resolves (~0.3 us on B200):
# confirming costs a fresh worker process per finalist.
_SUSPICIOUS_REBENCHMARK_MIN_DELTA_MS = 0.001
_FINAL_REBENCHMARK_TOP_K_ENV = "HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K"
# A backend may verify more finalists (``Backend.final_rebenchmark_top_k``).
_FINAL_REBENCHMARK_TOP_K_DEFAULT = 8
_FINAL_REBENCHMARK_TARGET_MS_ENV = "HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS"
_FINAL_REBENCHMARK_TARGET_MS_DEFAULT = 5000.0
_FINAL_REBENCHMARK_TARGET_MS_MAX = 60000.0
_FINAL_REBENCHMARK_ISOLATED_ENV = "HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED"
_FINAL_REBENCHMARK_PINNED_TOLERANCE_ENV = (
    "HELION_AUTOTUNE_FINAL_REBENCHMARK_PINNED_TOLERANCE"
)
_FINAL_REBENCHMARK_PINNED_TOLERANCE_DEFAULT = 0.0
# The single-pass final verification re-times only the shortlist entries the
# search measured within this factor of its best reading: a finalist 1.5x
# slower by the search's own (median, L2-flushed) timing cannot overturn the
# ranking, and every isolated steady window costs seconds (the target window
# plus a worker round trip). The clock regimes a search spans (boost vs. the
# sustained power-limited clock) move a reading by 10-15%, well inside the
# margin; ``inf`` keeps the whole ``top_k``.
_FINAL_REBENCHMARK_MAX_SPREAD_ENV = "HELION_AUTOTUNE_FINAL_REBENCHMARK_MAX_SPREAD"
_FINAL_REBENCHMARK_MAX_SPREAD_DEFAULT = 1.5
# Among the finalists the final verification cannot tell apart (within this
# fraction of the fastest one's steady reading) the backend's preference
# decides (``Backend.finalist_preference``): what a config costs or saves
# outside the kernel's own isolated timing, such as the dependent launch a
# CuTe kernel's successor overlaps with its tail.  The final selection's
# alternating steady windows read identical configs ~0.3-0.5% apart.
_FINAL_SELECTION_PREFERENCE_TOLERANCE_ENV = (
    "HELION_AUTOTUNE_FINAL_SELECTION_PREFERENCE_TOLERANCE"
)
_FINAL_SELECTION_PREFERENCE_TOLERANCE_DEFAULT = 0.005
_REBENCHMARK_TARGET_MS_DEFAULT = 200.0
_REBENCHMARK_INTERLEAVED_REPEAT_MAX = 20_000
# The only env override of the harness-grade final selection (full effort):
# the number of steady-window rounds, for A/B runs against the profile.
# HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS=0 restores the single-pass final
# verification (and its HELION_AUTOTUNE_FINAL_REBENCHMARK_* knobs).
_FINAL_SELECTION_ROUNDS_ENV = "HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS"
# The search phase never gives up more than this share of a wall-clock budget
# to the final selection: the selection only re-times configs the search
# found, so a small budget must stay mostly search; the phase is then sized
# to whatever is left when it starts (``FinalSelectionConfig.fit_to_seconds``).
_FINAL_SELECTION_RESERVE_FRACTION = 0.25
# Under a wall-clock budget the initial population spends at most this share
# of the search phase (the budget minus the final-selection reserve) before it
# is cut to the configs attempted so far, so the search generations that
# refine the population's best keep at least as much time as it took (see
# ``PopulationBasedSearch._initial_population_budget``).
_INITIAL_POPULATION_BUDGET_FRACTION = 0.5
# Single-pass verification knobs the final selection does not read.
_FINAL_SELECTION_IGNORED_ENVS = (
    _FINAL_REBENCHMARK_TARGET_MS_ENV,
    _FINAL_REBENCHMARK_ISOLATED_ENV,
    _FINAL_REBENCHMARK_MAX_SPREAD_ENV,
)
_final_selection_ignored_envs_warned = False
# The final selection's kernel-timer fallback is said once per process.
_kernel_timer_fallback_warned = False
_CUTE_FLASH_CONFIG_GENERATION_POLICY_VERSION = 8


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

    def to_code(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str | None:
        """Return source for the selected backend, or None if unavailable."""
        ...

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


def normalize_autotune_seed_configs(
    settings: Settings,
    *,
    structural_policy: StructuralPolicy
    | Callable[[], StructuralPolicy | None]
    | None = None,
) -> tuple[Config, ...]:
    """Return user-provided autotune seed configs from settings as concrete Configs."""
    from .._compiler.backend_registry import get_backend
    from ..runtime.config import Config

    seed_configs = settings.autotune_seed_configs
    if seed_configs is None:
        return ()
    if isinstance(seed_configs, StructuralConfig):
        seed_configs = (seed_configs,)
    if isinstance(seed_configs, Config):
        return (seed_configs,)
    if isinstance(seed_configs, dict):
        return (Config.from_dict(seed_configs),)
    result = []
    policy_type = get_backend(settings.backend).structural_policy_type()
    for seed_config in seed_configs:
        if isinstance(seed_config, StructuralConfig):
            # Legacy search-policy inspection can run before binding. Only a
            # new envelope needs the captured bound's policy, so resolve lazily.
            policy = (
                structural_policy()
                if callable(structural_policy)
                else structural_policy
            )
            require_same_structural_policy(
                seed_config.policy,
                (policy or policy_type.from_settings(settings))
                if policy_type is not None
                else None,
                context="Late autotune seed envelope",
            )
            result.append(seed_config.config)
        else:
            result.append(
                Config.from_dict(seed_config)
                if isinstance(seed_config, dict)
                else seed_config
            )
    return tuple(result)


def _file_sha256(filename: str | None) -> str | None:
    if filename is None:
        return None
    try:
        return hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    except OSError:
        return None


class _UncacheablePythonFunctionPolicy(TypeError):
    pass


def _python_global_cache_policy(value: object, seen: set[int]) -> object:
    if inspect.isfunction(value):
        policy = _python_function_cache_policy_impl(value, seen)
        if policy is None:
            raise _UncacheablePythonFunctionPolicy(
                f"cannot fingerprint Python helper {value.__qualname__}"
            )
        return policy
    if inspect.ismodule(value):
        return {
            "kind": "module",
            "name": value.__name__,
            "version": getattr(value, "__version__", None),
            "file_sha256": _file_sha256(getattr(value, "__file__", None)),
        }
    if inspect.isbuiltin(value):
        return {
            "kind": "builtin",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    return value


def _python_function_cache_policy_impl(
    callback: object, seen: set[int]
) -> dict[str, object] | None:
    if not inspect.isfunction(callback):
        return None
    identity = {
        "module": callback.__module__,
        "qualname": callback.__qualname__,
    }
    if id(callback) in seen:
        return {**identity, "recursive_reference": True}
    source_file = inspect.getsourcefile(callback)
    if source_file is None:
        return None
    seen.add(id(callback))
    try:
        module_sha256 = _file_sha256(source_file)
        code_sha256 = hashlib.sha256(marshal.dumps(callback.__code__)).hexdigest()
        closure = tuple(cell.cell_contents for cell in (callback.__closure__ or ()))
        closure_vars = inspect.getclosurevars(callback)
    except (OSError, TypeError, ValueError):
        return None
    return {
        **identity,
        "module_sha256": module_sha256,
        "code_sha256": code_sha256,
        "defaults": callback.__defaults__,
        "kwdefaults": callback.__kwdefaults__,
        "closure": closure,
        "globals": {
            name: _python_global_cache_policy(value, seen)
            for name, value in sorted(closure_vars.globals.items())
        },
        "builtins": {
            name: _python_global_cache_policy(value, seen)
            for name, value in sorted(closure_vars.builtins.items())
        },
        "unbound": tuple(sorted(closure_vars.unbound)),
    }


def _python_function_cache_policy(callback: object) -> dict[str, object] | None:
    """Fingerprint a Python function and the runtime globals it reads."""
    try:
        return _python_function_cache_policy_impl(callback, set())
    except _UncacheablePythonFunctionPolicy:
        return None


def _autotune_search_acf_cache_policy(paths: Sequence[str]) -> tuple[object, ...]:
    return tuple({"path": path, "sha256": _file_sha256(path)} for path in paths)


def _normalize_spec_key_str(s: str) -> str:
    """Normalize a specialization_key string for cache comparison.

    Applied to both the ``str()`` that put() stores and the ``str()`` of the
    live key, so the two sides can only differ where this function differs.
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

    # Snapshot the timing policy for all queued measurements in this search.
    _input_bank_count: int = 1
    # The prelude this search's candidates are timed behind, read when the
    # search is made (``benchmark_prelude.timed_behind``); None: alone. Such
    # a search takes every reading in its benchmark worker, as one that
    # reads its candidates from the trace does.
    _benchmark_prelude: BenchmarkPrelude | None = None
    # The config the search timed the returned kernel under, where it returns
    # a simpler config of that kernel (``_simplified_selected_config``): the
    # trace's trial rows are keyed by this one.
    _selected_config_measured_as: Config | None = None
    # Whether the search returns a simpler config of its selection where it
    # knows one. Off for a search whose config only seeds another search
    # (the LLM stage of the LLM-seeded hybrid): the renders that find the
    # simpler config would be spent on a config the next stage times again.
    _simplifies_selected_config: bool = True

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        benchmark_provider_cls: Callable[
            ..., BenchmarkProvider
        ] = LocalBenchmarkProvider,
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
        self._input_bank_count = self.settings.autotune_benchmark_input_banks
        self._benchmark_prelude = active_prelude()
        if isinstance(self.settings.autotune_config_overrides, StructuralConfig):
            raise StructuralPolicyError(
                "Policy-bearing autotune overrides must be supplied before "
                "creating a new Kernel; late overrides must be an ordinary "
                "dictionary for the already selected config schema."
            )
        self.config_spec: ConfigSpec = kernel.config_spec
        if not self.config_spec.backend.supports_precompile():
            # Backend.autotune() does this too, but a search can also be
            # constructed directly on a bound kernel.
            from .._compiler.backend import autotune_precompile_mode

            self.settings.autotune_precompile = autotune_precompile_mode(
                self.config_spec.backend, self.settings.autotune_precompile
            )
        from ..runtime.kernel import BoundKernel

        # Direct search construction bypasses BoundKernel.autotune(), which
        # normally fills evaluated defaults. Workers reload generated source
        # without nonportable defaults, so ship the complete argument list.
        # Multi-shape cases are already normalized; retain their carrier so
        # provider selection and aggregate measurements still see every case.
        if isinstance(kernel, BoundKernel) and not isinstance(
            args, _MultiShapeAutotuneArgs
        ):
            args = kernel.kernel.normalize_args(*args)
        self.args: Sequence[object] = args
        self.log = AutotuningLogger(self.settings)
        self.best_perf_so_far = inf
        self._benchmark_provider_cls = benchmark_provider_cls
        self._prepared = False
        self._skip_cache = False
        self._autotune_budget_start: float | None = None
        self._benchmarked_members: MutableMapping[Config, PopulationMember] = (
            ConfigCache()
        )
        self._pinned_finalist_configs = ConfigSet()
        self._pinned_finalist_members: MutableMapping[Config, PopulationMember] = (
            ConfigCache()
        )
        # Configs excluded from fallback after an unkillable worker: the
        # kernel that may have hung and earlier conclusive selection failures.
        self._fallback_excluded_configs = ConfigSet()
        self._search_space_tracker: SearchSpaceTracker | None = None
        self._uncacheable_search_policy_nonce: str | None = None
        self._search_policy_cacheable = benchmark_provider_cls is LocalBenchmarkProvider

    def _algorithm_cache_policy(self) -> dict[str, object] | None:
        """Return explicit behavior-affecting state for a built-in search."""
        return None

    def _final_rebenchmark_cache_policy(self) -> dict[str, object]:
        return {"enabled": False}

    def cache_policy(self) -> dict[str, object] | None:
        """Return the effective search policy used by CuTe-flash cache keys.

        Unknown search subclasses, benchmark providers and arbitrary callbacks
        fail closed: callers give them a one-shot cache key instead of risking
        reuse across different candidate sets, validation or benchmark objectives.
        """
        from . import search_algorithms

        if (
            type(self) not in search_algorithms.values()
            or self._benchmark_provider_cls is not LocalBenchmarkProvider
        ):
            return None
        algorithm = self._algorithm_cache_policy()
        settings = self.settings
        baseline_fn_policy = (
            None
            if settings.autotune_baseline_fn is None
            else _python_function_cache_policy(settings.autotune_baseline_fn)
        )
        accuracy_check = keyable_accuracy_check(settings)
        if (
            algorithm is None
            or any(
                callback is not None
                for callback in (
                    settings.autotune_config_filter,
                    settings.autotune_benchmark_fn,
                )
            )
            # Of accuracy checks only Helion's own can be keyed: an arbitrary
            # callback cannot be fingerprinted safely.
            or (
                settings.autotune_baseline_accuracy_check_fn is not None
                and accuracy_check is None
            )
            or (
                settings.autotune_baseline_fn is not None and baseline_fn_policy is None
            )
        ):
            return None

        if self.config_spec.cute_flash_search_enabled:
            algorithm = {
                **algorithm,
                "cute_flash_config_generation_policy_version": (
                    _CUTE_FLASH_CONFIG_GENERATION_POLICY_VERSION
                ),
            }

        return {
            "schema": "cute_flash_search_policy_v3",
            "search_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "algorithm": algorithm,
            "settings": {
                **(
                    {"autotune_benchmark_input_banks": self._input_bank_count}
                    if self._input_bank_count > 1
                    else {}
                ),
                "autotune_effort": settings.autotune_effort,
                "autotune_budget_seconds": settings.autotune_budget_seconds,
                "autotune_config_overrides": settings.autotune_config_overrides,
                "autotune_seed_configs": normalize_autotune_seed_configs(
                    settings,
                    structural_policy=lambda: bound_structural_policy(
                        self.kernel, captured=True
                    ),
                ),
                "compiler_seed_configs": tuple(self.config_spec.compiler_seed_configs),
                "compiler_seed_timeout_retry_repetitions": (
                    self.config_spec.compiler_seed_timeout_retry_repetitions
                ),
                "compiler_seed_timeout_retry_limit_per_source": (
                    _COMPILER_SEED_TIMEOUT_RETRY_LIMIT
                ),
                "config_value_priors_version": getattr(
                    self.config_spec.backend, "config_value_priors_version", None
                ),
                "autotune_force_persistent": settings.autotune_force_persistent,
                "disable_autotuner_heuristics": settings.disable_autotuner_heuristics,
                "autotune_accuracy_check": settings.autotune_accuracy_check,
                "autotune_baseline_fn": baseline_fn_policy,
                # What the default check accepts, or the AccuracyCheck that
                # replaces it.
                **(
                    {
                        "autotune_baseline_atol": settings.autotune_baseline_atol,
                        "autotune_baseline_rtol": settings.autotune_baseline_rtol,
                        "autotune_fp8_accuracy": effective_fp8_accuracy(settings),
                    }
                    if accuracy_check is None
                    else {
                        "autotune_baseline_accuracy_check_fn": (
                            accuracy_check.cache_policy()
                        )
                    }
                ),
                "autotune_adaptive_timeout": settings.autotune_adaptive_timeout,
                "autotune_compile_timeout": settings.autotune_compile_timeout,
                "autotune_benchmark_subprocess": (
                    settings.autotune_benchmark_subprocess
                ),
                "autotune_benchmark_timeout": settings.autotune_benchmark_timeout,
                # Rendering candidates in forked workers (a backend without a
                # precompile pool keeping the fork mode) changes no result.
                "autotune_precompile": (
                    settings.autotune_precompile
                    if self.config_spec.backend.supports_precompile()
                    else None
                ),
                # The number of precompile processes or render workers changes
                # no result; the entry stays (as None) so existing keys hold.
                "autotune_precompile_jobs": None,
                "autotune_ignore_errors": settings.autotune_ignore_errors,
                "autotune_best_of_k": settings.autotune_best_of_k,
                "autotune_with_torch_compile_fusion": (
                    settings.autotune_with_torch_compile_fusion
                ),
                "autotune_rebenchmark_threshold": (
                    settings.get_rebenchmark_threshold()
                ),
                "autotune_suspicious_rebenchmark_ratio": (
                    settings.get_suspicious_rebenchmark_ratio()
                ),
                "autotune_best_available_max_configs": (
                    settings.autotune_best_available_max_configs
                ),
                "autotune_best_available_max_cache_scan": (
                    settings.autotune_best_available_max_cache_scan
                ),
                "autotune_search_acf": _autotune_search_acf_cache_policy(
                    settings.autotune_search_acf
                ),
            },
            "final_rebenchmark": self._final_rebenchmark_cache_policy(),
        }

    @property
    def performance_unit(self) -> Literal["ms", "ratio"]:
        args = getattr(self, "args", ())
        if isinstance(args, _MultiShapeAutotuneArgs) and args.relative_to is not None:
            return "ratio"
        return "ms"

    @property
    def performance_suffix(self) -> str:
        return "x" if self.performance_unit == "ratio" else "ms"

    def format_performance(self, value: float) -> str:
        return f"{value:.4f}{self.performance_suffix}"

    def _prepare(self) -> None:
        """Some initialization deferred until autotuning actually runs.

        This is called at the start of autotune() so that cache hits skip it.
        """
        if self._prepared:
            return
        if active_prelude() is not self._benchmark_prelude or (
            self._benchmark_prelude is not None
            and self._benchmark_provider_cls is not LocalBenchmarkProvider
        ):
            # The provider made below reads the block it is made in, and
            # only the built-in one runs a prelude: a search that took the
            # prelude for its cache key and its re-timings must not time
            # alone, nor the other way round.
            raise exc.InvalidAPIUsage(
                "a search runs inside the benchmark_prelude.timed_behind block "
                "it was made in (or outside any, as it was made), on the "
                "built-in benchmark provider"
            )
        if (
            self.settings.megakernel_timeline is not None
            and self.config_spec.offers_megakernel
        ):
            # Stamps change a megakernel's timings. (A cache hit returns before
            # the search, so a tuned config is still used.)
            raise exc.AutotuningDisallowedInEnvironment(
                "while megakernel_timeline is set"
            )
        self._prepared = True
        self._autotune_budget_start = time.perf_counter()
        self._budget_exceeded_logged = ()
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
        if isinstance(self.args, _MultiShapeAutotuneArgs):
            case_tensors = []
            for _, case_args in self.args.cases:
                leaves, _ = tree_flatten(case_args)
                case_tensors.append(
                    [value for value in leaves if isinstance(value, torch.Tensor)]
                )
            input_shapes = str(
                [
                    [tuple(tensor.shape) for tensor in tensors]
                    for tensors in case_tensors
                ]
            )
            dtypes = str(
                [[str(tensor.dtype) for tensor in tensors] for tensors in case_tensors]
            )
        else:
            tensors = [arg for arg in self.args if isinstance(arg, torch.Tensor)]
            input_shapes = str([tuple(t.shape) for t in tensors])
            dtypes = str([str(t.dtype) for t in tensors])
        hardware = get_device_name(extract_device(self.args)) or ""
        self._autotune_metrics: AutotuneMetrics = AutotuneMetrics(
            kernel_name=kernel_name,
            kernel_source=kernel_source,
            input_shapes=input_shapes,
            dtypes=dtypes,
            hardware=hardware,
            random_seed=self.settings.autotune_random_seed,
            search_algorithm=type(self).__name__,
        )
        # Analyze search space if logging is enabled. Diagnostic only; a failure
        # must not prevent autotuning from running. Verbose logging implies the
        # summary, so either setting enables the analysis.
        if (
            self.settings.autotune_log_search_space
            or self.settings.autotune_log_search_space_verbose
        ):
            try:
                from .search_space_logger import SearchSpaceTracker
                from .search_space_logger import analyze_search_space

                hardware_spec, specialization_key = (
                    self._get_current_hardware_and_specialization()
                )
                report = analyze_search_space(
                    self.config_spec,
                    kernel_name=kernel_name,
                    specialization_key=specialization_key,
                    hardware=hardware_spec,
                    config_overrides=self.settings.autotune_config_overrides or None,
                    advanced_controls_files=self.settings.autotune_search_acf or None,
                )
                self._search_space_tracker = SearchSpaceTracker(report)
            except Exception:
                self.log.warning(
                    "Search space analysis setup failed; the end-of-run summary "
                    "will be skipped (autotuning continues normally)",
                    exc_info=True,
                )
                self._search_space_tracker = None
        host_function = getattr(self.kernel, "host_function", None)
        from .base_cache import should_skip_cache

        metadata_settings = self.settings.to_dict()
        metadata_settings["effective_cache_read_bypass"] = bool(
            self._skip_cache or should_skip_cache()
        )
        self._kernel_metadata: KernelMetadata = KernelMetadata(
            kernel_name=kernel_name,
            kernel_source=kernel_source,
            input_shapes=input_shapes,
            dtypes=dtypes,
            hardware=hardware,
            settings=metadata_settings,
            _device_ir=getattr(host_function, "_device_ir", None),
            cute_structural_policy=bound_structural_policy(self.kernel),
        )
        provider_cls = (
            MultiShapeBenchmarkProvider
            if isinstance(self.args, _MultiShapeAutotuneArgs)
            else self._benchmark_provider_cls
        )
        self.benchmark_provider = provider_cls(
            kernel=self.kernel,
            settings=self.settings,
            config_spec=self.config_spec,
            args=self.args,
            log=self.log,
            autotune_metrics=self._autotune_metrics,
        )
        if self.config_spec.compiler_seed_timeout_retry_repetitions is not None:
            seed_config_gen = self.config_spec.create_config_generation(
                overrides=self.settings.autotune_config_overrides or None,
                advanced_controls_files=self.settings.autotune_search_acf or None,
                process_group_name=self.kernel.env.process_group_name,
            )
            self.benchmark_provider.set_compiler_seed_configs(
                [config for _flat, config in seed_config_gen.seed_flat_config_pairs()]
            )
        self.benchmark_provider.set_config_roles(self._search_config_roles())
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

    # Set once the search phase hands over to the final selection, which then
    # spends the budget share ``_budget_reserve_seconds`` kept back for it.
    _budget_reserve_released: bool = False
    # Phases ("search", "final") whose budget cutoff has been logged: the
    # check runs before every compile once the budget is out, the line once.
    _budget_exceeded_logged: tuple[str, ...] = ()
    # Longest wall time between two generation-boundary budget checks of the
    # search phase (``_autotune_budget_exceeded_across_ranks``: the generation
    # loops and the flash qualification passes call it once per generation).
    # A budget check only stops new compiles; the generation in flight still
    # finishes its benchmark pass, which took 60-120 s past the check in
    # flash searches, so the search phase also holds back one generation of
    # overrun for the final selection (``_in_flight_hold_back_seconds``).
    _longest_generation_seconds: float = 0.0
    _last_generation_boundary: float | None = None

    def _budget_reserve_seconds(self) -> float:
        """Wall-clock seconds the search phase leaves for the final selection."""
        return 0.0

    def _minimum_final_selection_seconds(self) -> float:
        """Estimated seconds of the final selection's smallest plan."""
        return 0.0

    def _in_flight_hold_back_seconds(self, budget: float, reserve: float) -> float:
        """Seconds the search phase stops early on top of ``reserve`` so the
        generation in flight can finish without spending the reserve.

        The reserve and the hold-back together take at most the reserve's
        share of the budget (``_FINAL_SELECTION_RESERVE_FRACTION``): the
        selection sizes itself to what is left when it starts, so a long
        generation shortens the selection rather than the search. The
        selection's smallest plan is the exception, it stays reserved
        whenever a quarter of the budget covers it: when the generation
        would eat the reserve down past it, the hold-back grows by the
        difference (never beyond one generation) and the search stops one
        generation plus that plan before the budget ends.
        """
        longest = self._longest_generation_seconds
        cap = max(reserve, _FINAL_SELECTION_RESERVE_FRACTION * budget)
        keeps_minimum = longest + self._minimum_final_selection_seconds() - reserve
        return min(longest, max(cap - reserve, keeps_minimum))

    def _autotune_budget_exceeded(self) -> bool:
        budget = self.settings.autotune_budget_seconds
        if budget is None or self._autotune_budget_start is None:
            return False
        elapsed = time.perf_counter() - self._autotune_budget_start
        reserve = (
            0.0 if self._budget_reserve_released else self._budget_reserve_seconds()
        )
        in_flight = (
            self._in_flight_hold_back_seconds(budget, reserve) if reserve > 0 else 0.0
        )
        if elapsed < budget - reserve - in_flight:
            return False
        # "best-so-far" holds even when nothing has been measured yet: the
        # provider still attempts a few candidates past the budget (see
        # ``LocalBenchmarkProvider.benchmark``) before the search returns.
        phase = "search" if reserve > 0 else "final"
        if phase in self._budget_exceeded_logged:
            return True
        self._budget_exceeded_logged = (*self._budget_exceeded_logged, phase)
        if reserve > 0:
            self.log(
                f"Autotune budget {budget}s exceeded for the search phase "
                f"(elapsed {elapsed:.1f}s, {reserve:.0f}s reserved for the final "
                f"selection plus {in_flight:.0f}s for the generation in flight); "
                "returning best-so-far."
            )
        else:
            self.log(
                f"Autotune budget {budget}s exceeded "
                f"(elapsed {elapsed:.1f}s); returning best-so-far."
            )
        return True

    def _budgeted_range(self, *args: int) -> Iterator[int]:
        """Yield ``range(*args)`` until the autotune budget is exhausted."""
        for value in range(*args):
            if self._autotune_budget_exceeded_across_ranks():
                return
            yield value

    def _autotune_budget_exceeded_across_ranks(self) -> bool:
        """Synchronize CuTe-flash budget exhaustion before the next round."""
        now = time.perf_counter()
        if not self._budget_reserve_released:
            if self._last_generation_boundary is not None:
                self._longest_generation_seconds = max(
                    self._longest_generation_seconds,
                    now - self._last_generation_boundary,
                )
            self._last_generation_boundary = now
        exceeded = self._autotune_budget_exceeded()
        if not getattr(
            getattr(self, "config_spec", None), "cute_flash_search_enabled", False
        ):
            return exceeded
        return any(
            all_gather_object(
                exceeded,
                process_group_name=self.kernel.env.process_group_name,
            )
        )

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
        if not (
            self.settings.autotune_precompile
            and self.config_spec.backend.supports_precompile()
        ):
            # The adapted timeout kills precompile subprocesses. Render
            # workers (CuTe) keep the configured timeout: past a narrower one
            # fitted on the population's render times they would only fall
            # back to rendering the slower configs in this process.
            self.log.debug("No precompile pool; the compile timeout is not adapted")
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
        self, configs: list[Config], *, record_trace: bool = True
    ) -> tuple[list[Config], list[int]]:
        """Apply backend and user filters, returning configs and source indices.

        A config the filters drop is a search candidate that was not tried,
        and the trace gets a ``"filtered"`` trial row for it (not without
        ``record_trace``)."""
        config_filter = self.settings.autotune_config_filter
        filtered: list[Config | None] = []
        for config in configs:
            candidate = config_filter(config) if config_filter is not None else config
            if candidate is not None and not self._backend_config_is_viable(candidate):
                candidate = None
            filtered.append(candidate)
        for config, result in zip(configs, filtered, strict=True):
            if (
                result is None
                and record_trace
                and self.log.trace_enabled
                and (config_id := self.log.register_config(config)) is not None
            ):
                self.log.record_trace_entry(
                    AutotuneLogEntry(
                        generation=self._autotune_metrics.num_generations,
                        status="filtered",
                        perf_ms=None,
                        compile_time=None,
                        config_id=config_id,
                        config=config,
                        objective_unit=self.performance_unit,
                    )
                )
        passing_indices = [i for i, fc in enumerate(filtered) if fc is not None]
        passing_configs = cast(
            "list[Config]",
            [filtered[i] for i in passing_indices],
        )
        return passing_configs, passing_indices

    def _backend_config_is_viable(self, config: Config) -> bool:
        """Return whether the backend can cheaply rule out an autotune config."""
        return self.config_spec.backend.autotune_config_is_viable(
            self.config_spec, config
        )

    def benchmark_batch(
        self,
        configs: list[Config],
        *,
        desc: str = "Benchmarking",
        raise_if_no_viable_config: bool = True,
    ) -> list[BenchmarkResult]:
        """Compile and benchmark a batch of configurations.

        Applies the config filter, delegates to the provider, and tracks
        best performance.

        Args:
            configs: A list of configurations to benchmark.
            desc: Description for the progress bar.
            raise_if_no_viable_config: Re-raise the compile error when nothing
                has been measured yet and the whole batch fails to compile.
                Pass ``False`` for a batch that has a fallback population.

        Returns:
            A list of BenchmarkResult entries, one per input config.
        """
        passing_configs, passing_indices = self._apply_config_filter(configs)
        filtered_at = time.perf_counter()
        # Record configs for exploration tracking. Diagnostic only; must never
        # interfere with benchmarking.
        if self._search_space_tracker is not None:
            try:
                for config in passing_configs:
                    self._search_space_tracker.record_config(config)
            except Exception:
                self.log.debug(
                    "Exploration tracking failed; continuing autotuning",
                    exc_info=True,
                )
        try:
            inner_results = self.benchmark_provider.benchmark(
                passing_configs,
                desc=desc,
                raise_if_no_viable_config=raise_if_no_viable_config,
            )
        except BenchmarkWorkerUnkillable as error:
            if len(passing_indices) != len(configs):
                # The provider numbers the configs that passed the filters;
                # the caller's batch is ``configs``. A rejected config is a
                # failure in it: a batch that completes returns it without
                # a reading.
                if error.fn_index is not None:
                    error.fn_index = passing_indices[error.fn_index]
                passing = set(passing_indices)
                readings = iter(error.batch_results)
                error.batch_results = tuple(
                    next(readings, None)
                    if index in passing
                    else IsolatedBenchmarkFailure("error")
                    for index in range(len(configs))
                )
            raise

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
                        f"Config filtered out before benchmarking: {config!r}"
                    )
                    results.append(
                        BenchmarkResult(
                            config=config,
                            fn=lambda *a, **kw: None,
                            perf=inf,
                            status="filtered",
                            compile_time=None,
                            completed_at=filtered_at,
                        )
                    )

        for r in results:
            if r.perf < self.best_perf_so_far:
                self.best_perf_so_far = r.perf

        # Check for handoff when enabled. If triggered, raise an internal stop
        # signal that find_handoff() catches after nested searches clean up
        # their resources. Configs the budget skipped were never trials.
        _observe_handoff(self, [r for r in results if r.status != "skipped"])
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

    def _generation_invalid_config_count(self) -> int:
        """Number of candidate configs rejected as InvalidConfig during search.

        These are candidates that a config fragment / normalize step ruled out
        before they could be benchmarked (so they never reach the exploration
        tracker's valid-config path). The base implementation reports zero;
        subclasses that own a :class:`ConfigGeneration` expose its running
        count so the search-space logger can report explored-invalid alongside
        explored-valid.
        """
        return 0

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """
        Perform autotuning to find the best configuration.

        This method searches for the optimal configuration by benchmarking multiple configurations.

        Returns:
            The best configuration found during autotuning.
        """
        with self.log.autotune_tracing(type(self).__name__):
            config = self._autotune_with_logging(skip_cache=skip_cache)
            self.log.record_selected_config(
                config, measured_as=self._selected_config_measured_as
            )
            return config

    def _autotune_with_logging(self, *, skip_cache: bool) -> Config:
        # Every timer of the search, from the reference baseline's to the
        # final selection's, follows its L2 flushes with the sweep where the
        # setting asks (in this process; the benchmark worker's jobs carry
        # the block, ``LocalBenchmarkProvider``).
        with l2_flush_sweep(self.settings.autotune_l2_flush_sweep):
            return self._autotune_and_log(skip_cache=skip_cache)

    def _autotune_and_log(self, *, skip_cache: bool) -> Config:
        self._skip_cache = skip_cache
        # Constructing the benchmark provider compiles the reference baseline.
        with _searching_candidates():
            self._prepare()
        start = time.perf_counter()
        exit_stack = contextlib.ExitStack()
        best: Config | None = None
        outcome = "failed"
        error_type: str | None = None
        try:
            with exit_stack:
                # set_adaptive_compile_timeout narrows the compile deadline
                # for this search. The kernel's settings are shared by every
                # shape the kernel is bound to, whose searches (and accuracy
                # baselines) would otherwise inherit the narrowed deadline.
                exit_stack.callback(
                    setattr,
                    self.settings,
                    "autotune_compile_timeout",
                    self.settings.autotune_compile_timeout,
                )
                exit_stack.enter_context(_searching_candidates())
                if self.settings.autotune_log:
                    # .csv/.log follow the log path; the dataset (.meta.jsonl) also
                    # needs opt-in and a representative (non-restricted) search.
                    collect_dataset = (
                        self.settings.autotune_log_details
                        and not self._is_restricted_search()
                    )
                    exit_stack.enter_context(
                        self.log.autotune_logging(
                            metadata=self._kernel_metadata,
                            collect_dataset=collect_dataset,
                        )
                    )
                elif self.settings.autotune_log_details:
                    _warn_dataset_without_log(self.log)
                self.log.reset()
                # Autotuner triggers bugs in remote triton compile service.
                # Skip storing Triton intermediate IRs (.ttir, .ttgir, .llir, etc.)
                # during autotuning to reduce cache size by ~40%. Only binaries and
                # metadata are needed for execution.
                env_overrides = {"TRITON_LOCAL_BUILD": "1"}
                if "TRITON_STORE_BINARY_ONLY" not in os.environ:
                    env_overrides["TRITON_STORE_BINARY_ONLY"] = "1"
                exit_stack.enter_context(
                    patch.dict(os.environ, env_overrides, clear=False)
                )
                self.benchmark_provider.setup()
                exit_stack.callback(self.benchmark_provider.cleanup)
                try:
                    best = self._autotune()
                    if isinstance(
                        self.benchmark_provider, MultiShapeBenchmarkProvider
                    ) and isinstance(self.args, _MultiShapeAutotuneArgs):
                        if not self.benchmark_provider.has_valid_measurement(best):
                            raise exc.NoConfigFound
                        if not self.args.defer_selected_log:
                            self.benchmark_provider.log_selected(best)
                    best = self._simplified_selected_config(best)
                except exc.NoConfigFound as error:
                    # Every candidate dropped by the worker's compile deadline
                    # is a deadline (or host) problem, not a kernel without a
                    # working config: say so where the search's caller looks.
                    cascade = self.benchmark_provider.compile_deadline_cascade()
                    if cascade is not None:
                        raise exc.NoConfigFound(f": {cascade}") from error
                    # Likewise readings the benchmark prelude took: alone, the
                    # kernel may find a config (its caller can tune it alone
                    # then).
                    behind = self.benchmark_provider.prelude_failure_cascade()
                    if behind is not None:
                        raise PreludeSearchFailed(f": {behind}") from error
                    raise
                finally:
                    self._finalize_autotune_metrics(best)
            outcome = "ok"
        except BenchmarkWorkerUnkillable as error:
            if best is None or best in self._fallback_excluded_configs:
                raise
            # The search already selected a config; only the post-search
            # teardown of a benchmark worker failed (a hung kernel keeps the
            # worker alive past SIGKILL). Keep the result instead of
            # discarding the whole search.
            self.log.warning(
                "Keeping the selected config although a benchmark worker could "
                f"not be reaped after the search finished: {error}"
            )
            outcome = "ok"
        except _HandoffStop:
            # A find_handoff early stop is a controlled exit, not a failure.
            outcome = "handoff"
            raise
        except Exception as error:
            # Tag the failure for the report; the exception propagates unchanged.
            error_type = type(error).__name__
            raise
        finally:
            # Failed searches report their coverage too.
            self._log_search_space_report(
                time.perf_counter() - start, outcome=outcome, error_type=error_type
            )
        assert best is not None
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
        if not self.settings.autotune_accuracy_check:
            # Say so where the selection is reported: a log that names a
            # selected config without this line reads as if the config had
            # passed a check it never saw (a kernel that disables the check
            # relies on its caller's own comparison).
            self.log.warning(
                "Accuracy checks were disabled (autotune_accuracy_check=False): "
                f"the selected config was ranked among "
                f"{self._autotune_metrics.num_configs_tested} configs by timing "
                "alone and was never compared with the baseline output."
            )
        if self._autotune_metrics.num_compile_failures:
            # Not "N of num_configs_tested": a config that fails to compile is
            # counted here but never benchmarked, so N can exceed that count.
            self.log.warning(
                f"{self._autotune_metrics.num_compile_failures} configs failed to "
                "compile or run."
            )
            self._log_unmeasured_config_values()
        if self._autotune_metrics.num_worker_failures:
            self.log.warning(
                f"{self._autotune_metrics.num_worker_failures} of "
                f"{self._autotune_metrics.num_configs_tested} configs failed in "
                "isolated benchmark workers."
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

    def _log_search_space_report(
        self,
        elapsed_seconds: float,
        *,
        outcome: str,
        error_type: str | None = None,
    ) -> None:
        """Log and save the end-of-run search space report (best-effort).

        Called from a ``finally`` so failed searches report too; nothing
        here, including the settings check, may mask an in-flight exception.
        """
        try:
            if not (
                self.settings.autotune_log_search_space
                or self.settings.autotune_log_search_space_verbose
            ):
                return
            from .local_cache import stable_autotune_hash
            from .search_space_logger import log_search_space_comparison

            tracker = self._search_space_tracker
            if tracker is None:
                from .search_space_logger import SearchSpaceTracker
                from .search_space_logger import analyze_search_space

                hardware_spec, specialization_key = (
                    self._get_current_hardware_and_specialization()
                )
                tracker = SearchSpaceTracker(
                    analyze_search_space(
                        self.config_spec,
                        kernel_name=self._autotune_metrics.kernel_name,
                        specialization_key=specialization_key,
                        hardware=hardware_spec,
                        config_overrides=(
                            self.settings.autotune_config_overrides or None
                        ),
                        advanced_controls_files=(
                            self.settings.autotune_search_acf or None
                        ),
                    )
                )
            tracker.record_invalid(self._generation_invalid_config_count())
            report = tracker.finish(
                search_algorithm=type(self).__name__,
                elapsed_seconds=elapsed_seconds,
                outcome=outcome,
                error_type=error_type,
            )
            log_search_space_comparison(self.log._logger, report)
            if self.settings.autotune_log_search_space_path:
                saved_path = report.save(
                    self.settings.autotune_log_search_space_path,
                    stable_autotune_hash(self),
                )
                if saved_path:
                    self.log(f"Search space analysis saved to: {saved_path}")
        except Exception:
            self.log.warning(
                "Search space logging failed; the end-of-run summary was not "
                "emitted (autotuning result is unaffected)",
                exc_info=True,
            )

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
        # Compare in the exact form put() stores: str() of the raw key.
        # Normalizing the key as an object tree is not equivalent: pytree
        # rebuilds torch.Size as a plain tuple, which reprs differently.
        specialization_key = _normalize_spec_key_str(str(spec_key))

        return hardware, specialization_key

    def _find_similar_cached_configs(
        self, max_configs: int, *, config_spec_hash: str | None = None
    ) -> list[SavedBestConfig]:
        """Return cached configs matching hardware and specialization.

        Scans the local cache first; if more configs are needed and a remote
        backend is configured, queries it via ``RemoteCacheBackend.list()``
        and merges the results, deduplicating against local entries. Explicit
        cache disabling always suppresses these seeds. Forced autotuning
        suppresses them only for CuTe flash, whose structural search must stay
        independent of previous winners; other backends keep their warm-start
        seeds under forced autotuning, which forces a fresh search rather than
        discarding prior measurements.
        """
        from .base_cache import _requires_uncached_tuning
        from .base_cache import should_skip_cache

        if (
            _requires_uncached_tuning(self.kernel, self.args)
            or (self._skip_cache and self.config_spec.cute_flash_search_enabled)
            or should_skip_cache()
        ):
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

        current_fingerprint_hash = (
            config_spec_hash
            if config_spec_hash is not None
            else self.config_spec.cache_fingerprint_hash(
                advanced_controls_files=self.settings.autotune_search_acf or None
            )
        )

        def is_compatible(entry: SavedBestConfig) -> bool:
            return (
                entry.flat_config is not None
                and entry.hardware == current_hardware
                and _normalize_spec_key_str(entry.specialization_key)
                == current_spec_key
                and entry.config_spec_hash == current_fingerprint_hash
                and entry.cute_structural_policy == bound_structural_policy(self.kernel)
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

    def _simplified_selected_config(self, best: Config) -> Config:
        """The config the search returns for its selection ``best``: the same
        one, unless the search knows a simpler config of the same kernel."""
        return best

    def _autotune_seed_configs(self) -> Sequence[Config]:
        """Return user-provided autotune seed configs normalized from settings."""
        return normalize_autotune_seed_configs(
            self.settings,
            structural_policy=lambda: bound_structural_policy(
                self.kernel, captured=True
            ),
        )

    def _search_config_roles(self) -> MutableMapping[Config, str]:
        """The configs this search benchmarks for a reason of their own, by
        role (see :meth:`BenchmarkProvider.set_config_roles`); a search over
        user-given configs has none."""
        return ConfigCache()

    def set_generation(self, generation: int) -> None:
        self._autotune_metrics.num_generations = generation

    # Most lines the unmeasured-values summary prints.
    _UNMEASURED_CONFIG_VALUES_SHOWN = 8

    def _log_unmeasured_config_values(self) -> None:
        """Surface the lowerings the search could not reach.

        A config value every attempted candidate failed to compile with (a
        tensor-core family on a kernel its matcher declines, a lane layout the
        lowering rejects) never appears in the result: the winner simply has
        the knob off, and only ``HELION_AUTOTUNE_LOG_LEVEL=DEBUG`` or a
        hand-compiled config shows why. One INFO line per such value names the
        first failure's reason.
        """
        facts = self.benchmark_provider.unmeasured_config_values()
        if not facts:
            return
        shown = facts[: self._UNMEASURED_CONFIG_VALUES_SHOWN]
        lines = [
            f"  {fact.key}={fact.value!r}: {fact.count} config(s) failed to "
            f"compile; first: {fact.first_error}"
            for fact in shown
        ]
        if len(facts) > len(shown):
            lines.append(f"  ... and {len(facts) - len(shown)} more")
        self.log(
            "Config values no measured candidate carried (every config using "
            "them failed to compile, so the selected config has none):\n"
            + "\n".join(lines)
        )

    def _finalize_autotune_metrics(self, best_config: Config | None = None) -> None:
        """Finalize run metrics and link population winners to measured source.

        Population searches retain compiled callables for benchmarked members,
        which lets us record the selected generated-source hash without
        recompiling. Final verification can re-pick a member while older noisy
        timings still make ``self.best`` point at a different config, so inspect
        all retained members for the returned config. Search implementations that
        do not retain a match leave the source fields unset rather than claiming
        an unverifiable linkage.
        """
        if best_config is not None:
            self._autotune_metrics.selected_config = copy.deepcopy(best_config.config)
            if isinstance(
                self, PopulationBasedSearch
            ) and self.config_spec.backend.should_deduplicate_generated_sources(
                self.config_spec
            ):
                selected_member = getattr(self, "_selected_member", None)
                retained_members = [
                    *(() if selected_member is None else (selected_member,)),
                    *getattr(self, "population", ()),
                    *getattr(self, "_benchmarked_members", {}).values(),
                    *getattr(self, "_pinned_finalist_members", {}).values(),
                ]
                for member in retained_members:
                    if _config_key(member.config) != _config_key(best_config):
                        continue
                    source_hash = self.config_spec.backend.generated_source_hash(
                        member.fn
                    )
                    if source_hash is None:
                        continue
                    if self._autotune_metrics.selected_source_hash is None:
                        self._autotune_metrics.selected_source_hash = source_hash
                    if self.benchmark_provider.has_measured_source_hash(source_hash):
                        self._autotune_metrics.selected_source_hash = source_hash
                        self._autotune_metrics.selected_source_was_measured = True
                        break
        best_perf = self.best_perf_so_far
        if self.performance_unit == "ratio":
            best_perf = (
                self.benchmark_provider.raw_latency(best_config)
                if best_config is not None
                and isinstance(self.benchmark_provider, MultiShapeBenchmarkProvider)
                else inf
            )
        self._autotune_metrics.best_perf_ms = (
            best_perf if math.isfinite(best_perf) else 0.0
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
        timer_readings_from (int): The index in ``perfs`` of the first reading
            the member took itself. Above 0 for a config settled with a copy
            of an earlier candidate's reading in a search that reads its
            candidates from the trace: the copy and a later reading are not
            two readings of one config (``pattern_search.rereading_spread``).
    """

    fn: Callable[..., object]
    perfs: list[float]
    flat_values: FlatConfig
    config: Config
    status: Literal[
        "ok",
        "error",
        "timeout",
        "peer_compilation_fail",
        "filtered",
        "deduplicated",
        "accuracy_error",
        "source_rejected",
        "skipped",
        "unknown",
    ] = "unknown"
    compile_time: float | None = None
    timer_readings_from: int = 0

    @property
    def perf(self) -> float:
        return self.perfs[-1]


def _config_family(config: Config, keys: Sequence[str]) -> tuple[object, ...]:
    """The kernel body a config selects: its values of the family knobs."""
    return tuple(config.config.get(key) for key in keys)


def _family_label(family: Sequence[object]) -> str:
    return "/".join(str(value) for value in family)


def performance(member: PopulationMember) -> float:
    """
    Retrieve the performance of a population member.  Used as a sort key.

    Args:
        member: The population member.

    Returns:
        The performance of the member.
    """
    return member.perf


class _PrivateBenchmarkArgs(NamedTuple):
    """Candidate-private kernel arguments for an in-process re-timing.

    ``sacrificial`` absorbs the allocator's recycled addresses and must stay
    alive while ``by_member`` is in use: a single recycled clone would let a
    later candidate inherit L2 eviction priority from an earlier candidate
    that touched the same addresses (for example an ``l2_last`` store).
    """

    sacrificial: Sequence[object]
    by_member: list[Sequence[object]]


def _classify_final_selection_result(
    result: IsolatedBenchmarkTiming,
) -> tuple[float | None, Literal["ok", "incomplete", "error", "timeout"]]:
    if isinstance(result, IsolatedBenchmarkFailure):
        return None, result.status
    if result is None or not math.isfinite(result) or result <= 0:
        return None, "incomplete"
    return float(result), "ok"


class _FinalSelectionMeasurer(Protocol):
    """Steady-window timer the final selection drives one finalist at a time.

    A ``BenchmarkWorkerUnkillable`` raised by either method names the finalist
    it was timing in ``fn_index``.
    """

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, *exc_info: object) -> None: ...

    def warm(self, index: int, duration_ms: int) -> IsolatedBenchmarkTiming:
        """Settle clocks, returning the warmup outcome without a rank sample."""
        ...

    def measure(
        self,
        index: int,
        warmup_ms: int,
        rep_ms: int,
        l2_flush_mode: str | None = None,
    ) -> IsolatedBenchmarkTiming:
        """Median of one ``warmup_ms``+``rep_ms`` steady window of finalist
        ``index``; ``None`` when no reading could be taken. ``l2_flush_mode``
        names the window's L2 regime, else the measurer's own."""
        ...


class _WorkerFinalSelectionMeasurer:
    """Times finalists in the killable benchmark worker (one long-lived
    process, so a hung finalist is reaped instead of hanging the search).
    With ``private_copies`` every finalist reads its own argument copies,
    which are released when the phase ends."""

    def __init__(
        self,
        provider: LocalBenchmarkProvider,
        fns: Sequence[Callable[..., object]],
        *,
        private_copies: bool,
        timeout_s: float,
        l2_flush_mode: str | None = None,
        desc: str = "Final selection",
    ) -> None:
        self.provider = provider
        self.fns = list(fns)
        self.private_copies = private_copies
        self.slots: list[int | None] = [
            index if private_copies else None for index in range(len(self.fns))
        ]
        self.timeout_s = timeout_s
        self.l2_flush_mode = l2_flush_mode
        # what the worker's timer really does with L2: do_bench honours the
        # regime, the wall-clock bench (do_bench_generic) always flushes
        self.regime_label = (
            "call (wall clock, always flushed)"
            if provider._subprocess_benchmark_uses_wall_clock()
            else l2_flush_mode or "call"
        )
        self.desc = desc

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, *exc_info: object) -> None:
        # The worker stays up for later phases; its private argument copies
        # (up to half of the free device memory) must not. A worker that
        # raised on the way out (hung, died) is not asked to release them.
        if self.private_copies and exc_type is None:
            self.provider.release_private_argument_copies()

    def warm(self, index: int, duration_ms: int) -> IsolatedBenchmarkTiming:
        return self._window(
            index,
            warmup=duration_ms,
            rep=1,
            desc=f"{self.desc} warmup",
            l2_flush_mode=untraced_regime(self.l2_flush_mode),
        )

    def measure(
        self,
        index: int,
        warmup_ms: int,
        rep_ms: int,
        l2_flush_mode: str | None = None,
        *,
        fixed_repetitions: int | None = None,
    ) -> IsolatedBenchmarkTiming:
        return self._window(
            index,
            warmup=warmup_ms,
            rep=rep_ms,
            desc=self.desc,
            l2_flush_mode=l2_flush_mode or self.l2_flush_mode,
            fixed_repetitions=fixed_repetitions,
        )

    def _window(
        self,
        index: int,
        *,
        warmup: int,
        rep: int,
        desc: str,
        l2_flush_mode: str | None = None,
        fixed_repetitions: int | None = None,
    ) -> IsolatedBenchmarkTiming:
        benchmark = self.provider.benchmark_isolated
        if fixed_repetitions is not None:
            benchmark = functools.partial(
                benchmark, fixed_repetitions=fixed_repetitions
            )
        try:
            results = benchmark(
                [self.fns[index]],
                warmup=warmup,
                rep=rep,
                desc=desc,
                args_slots=[self.slots[index]],
                timeout=self.timeout_s,
                l2_flush_mode=l2_flush_mode,
            )
        except InputBankMemoryError as error:
            if not self.private_copies:
                raise
            self.provider.disable_private_argument_copies(index, str(error))
            raise _RetryWithoutPrivateCopies from error
        except BenchmarkWorkerUnkillable as error:
            # the batch held this one finalist
            error.fn_index = index
            # the calling loop sets batch_results (empty here) where it is read
            raise
        if results is None:
            return None
        return results[0]


class _InProcessFinalSelectionMeasurer:
    """Times finalists in the autotuning process with the backend's steady
    bench; ``args_by_fn`` holds each finalist's (possibly private) arguments."""

    def __init__(
        self,
        fns: Sequence[Callable[..., object]],
        args: _PrivateBenchmarkArgs | Sequence[object],
        steady_bench: Callable[..., float | tuple[float, ...]],
        *,
        process_group_name: str | None,
        log: AutotuningLogger,
        l2_flush_mode: str | None = None,
    ) -> None:
        self.fns = list(fns)
        # the private clones stay alive (with their sacrificial clone) until exit
        self.args: _PrivateBenchmarkArgs | Sequence[object] | None = args
        self.args_by_fn: list[Sequence[object]] = (
            list(args.by_member)
            if isinstance(args, _PrivateBenchmarkArgs)
            else [args] * len(self.fns)
        )
        self.steady_bench = steady_bench
        self.process_group_name = process_group_name
        self.log = log
        # Only the shared timers know the L2 regimes; a backend's own steady
        # bench (Metal) times in its one regime, and the wall-clock bench
        # (Pallas) accepts the regime but always flushes.
        self.l2_flush_mode = (
            l2_flush_mode if steady_bench in (do_bench, do_bench_generic) else None
        )
        if steady_bench is do_bench:
            self.regime_label = l2_flush_mode or "call"
        elif steady_bench is do_bench_generic:
            self.regime_label = "call (wall clock, always flushed)"
        else:
            self.regime_label = "backend bench"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, *exc_info: object) -> None:
        for fn in self.fns:
            clear_jit_fast_path_caches(fn, self.log)
        self.args_by_fn = []
        self.args = None

    def warm(self, index: int, duration_ms: int) -> IsolatedBenchmarkTiming:
        return self._bench(index, duration_ms, 1)

    def measure(
        self,
        index: int,
        warmup_ms: int,
        rep_ms: int,
        l2_flush_mode: str | None = None,
    ) -> IsolatedBenchmarkTiming:
        return self._bench(index, warmup_ms, rep_ms, l2_flush_mode=l2_flush_mode)

    def _bench(
        self,
        index: int,
        warmup_ms: int,
        rep_ms: int,
        l2_flush_mode: str | None = None,
    ) -> float:
        run = functools.partial(self.fns[index], *self.args_by_fn[index])
        kwargs: dict[str, object] = {}
        # a timer that knows the regimes takes the window's, else its own
        if self.l2_flush_mode is not None:
            kwargs["l2_flush_mode"] = l2_flush_mode or self.l2_flush_mode
        try:
            timing = self.steady_bench(
                run,
                warmup=warmup_ms,
                rep=rep_ms,
                return_mode="median",
                process_group_name=self.process_group_name,
                **kwargs,
            )
        finally:
            clear_jit_fast_path_caches(self.fns[index], self.log)
        if isinstance(timing, tuple):
            timing = timing[0]
        return float(timing)


class PopulationBasedSearch(BaseSearch):
    """
    Base class for search algorithms that use a population of configurations.

    Attributes:
        population (list[PopulationMember]): The current population of configurations.
        flat_spec (list[ConfigSpecFragment]): The flattened configuration specification.
    """

    # Harness-grade final selection policy, resolved from the effort profile
    # of ``settings.autotune_effort`` (full effort only; the lower efforts
    # keep the single-pass final verification). It is a benchmarking policy
    # rather than a search-algorithm knob, so it is not a constructor
    # argument and ``_final_rebenchmark_cache_policy`` carries it.
    final_selection: FinalSelectionConfig | None = None
    _compiler_refinement_keys: tuple[str, ...] = ()

    # The first compile error of the last benchmarked batch (None when every
    # config compiled), for the initial population's summary line.
    _last_batch_first_error: Exception | None = None
    # Set while the initial population is benchmarked (see
    # ``_initial_population_batch``): the batch methods keep their signatures
    # for callers that stub them and read the phase from here.
    _benchmarking_initial_population: bool = False

    # ``Backend.autotune_short_kernel_below_us`` of the kernel's backend, read
    # at construction: a search whose initial population's best config reads
    # below this many microseconds times every later candidate from the CUDA
    # activity trace (``_settle_candidate_timer``). ``None``: the backend
    # names no bound, and the event timer reads every kernel. An algorithm
    # that reads no initial population (``benchmark_initial_population``; the
    # LLM-guided search times every batch as it comes) never reaches that
    # choice and keeps the event timer whatever the bound.
    short_kernel_below_us: float | None = None
    # Whether this search settled on the trace timer: its candidates and
    # re-timings are then kernel durations from the trace, a pattern search
    # counts steps within their repeatability as ties
    # (``PatternSearch._reading_tolerance``), and the early stop of a
    # differential-evolution search counts a move of the best within it as
    # no improvement (``DifferentialEvolutionSearch._stop_tolerance``).
    traces_candidates: bool = False
    _candidate_timer_settled: bool = False

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
        self.final_selection = self._final_selection_from_settings(self.settings)
        self.population: list[PopulationMember] = []
        # Population members corresponding to compiler-seeded configs from
        # ``ConfigSpec.compiler_seed_configs``.  Tracked separately so the
        # final-pick verification phase can re-include them as candidates
        # even when the surrogate-driven search has pruned them away from
        # ``self.population``.
        self._compiler_seed_members: list[PopulationMember] = []
        self._best_available_seed_configs: list[Config] = []
        self._selected_member: PopulationMember | None = None
        # The member that holds each kernel this search has timed, by the
        # ``id`` of its compiled function (the member keeps it alive), where
        # the search drops source aliases. Filled where members are settled
        # (``_settle_benchmarked_members``) and read through
        # ``_kernel_holder``. A holder stays here after it has left the
        # population and the final selection's history, so the search can
        # always name the config that stands for a kernel it timed and the
        # reading that kernel has now.
        self._kernels: dict[int, PopulationMember] = {}
        self._terminal_refinement_members: (
            MutableMapping[Config, PopulationMember] | None
        ) = None
        self.config_gen: ConfigGeneration = self.config_spec.create_config_generation(
            overrides=self.settings.autotune_config_overrides or None,
            advanced_controls_files=self.settings.autotune_search_acf or None,
            process_group_name=kernel.env.process_group_name,
            compiler_coverage_enabled=not self.settings.disable_autotuner_heuristics,
        )
        self._compiler_refinement_keys = (
            self.config_spec.search_surface.compiler_refinement_keys
        )
        self.short_kernel_below_us = (
            self.config_spec.backend.autotune_short_kernel_below_us(self.config_spec)
        )

    def _prepare(self) -> None:
        super()._prepare()
        if (
            self._terminal_refinement_members is None
            and self._compiler_refinement_enabled()
        ):
            # Reuse measured successes and remember failed trials, including
            # configurations pruned from the final-selection shortlist.
            # The provider and effective verification policy are available now.
            # Leave any independent terminal-refinement registry intact.
            self._terminal_refinement_members = ConfigCache()

    def _generation_invalid_config_count(self) -> int:
        return self.config_gen.invalid_config_count

    def _search_config_roles(self) -> MutableMapping[Config, str]:
        # The normalized forms the population builders benchmark: the user's
        # seeds, the compiler's seeds and the fragment default (which differs
        # from the execution default whenever a seed is promoted or a knob is
        # overridden). The default's role wins when a seed coincides with it.
        roles: MutableMapping[Config, str] = ConfigCache()
        for _flat, config in self.config_gen.user_seed_flat_config_pairs(
            self._autotune_seed_configs()
        ):
            roles[config] = "user seed config"
        for _flat, config in self.config_gen.seed_flat_config_pairs():
            roles[config] = "compiler seed"
        try:
            default = self.config_gen.unflatten(self.config_gen.default_flat())
        except exc.InvalidConfig:
            # A fragment default the spec refuses to normalize (seen under
            # HELION_INTERPRET=1) is never benchmarked, so it names no role.
            return roles
        roles[default] = "default config"
        return roles

    def _final_rebenchmark_cache_policy(self) -> dict[str, object]:
        policy: dict[str, object] = {
            "enabled": True,
            # 2: the non-isolated finalist shootout uses paired interleaved
            # timing instead of sequential steady windows.
            # 3: an in-process fallback for a mutated-argument kernel gives
            # every finalist private storage, preventing cross-config L2
            # eviction-priority leakage.
            # 4: fresh finalist processes cannot inherit cached argument
            # storage or live allocations from candidates measured earlier
            # in search.
            # 5: isolated finalists read private argument copies in one
            # long-lived worker (fresh processes only when the copies do not
            # fit), and the shortlist is sized to the search's spread.
            "timing_version": 5,
            "top_k": self._final_rebenchmark_top_k(),
            "target_ms": self._final_rebenchmark_target_ms(),
            "isolated": self._final_rebenchmark_use_isolated(),
            "pinned_tolerance": self._final_rebenchmark_pinned_tolerance(),
            "max_spread": self._final_rebenchmark_max_spread(),
            "repeat_cap": os.getenv("HELION_CAP_REBENCHMARK_REPEAT"),
            # The harness-grade final selection replaces the pass above at
            # full effort; like the knobs above it decides which finalist is
            # returned, not which candidates the search visits.
            "final_selection": (
                None
                if self.final_selection is None
                else dataclasses.asdict(self.final_selection)
            ),
        }
        if self._compiler_refinement_keys:
            policy["compiler_refinement"] = compiler_refinement_policy(
                self._compiler_refinement_keys
            )
        if self.final_selection is not None:
            backend_regime = self._backend_final_regime()
            if backend_regime is not None:
                # A backend's own final-selection timer (CuTe: the trace
                # kernel timer) picks other finalists than the search's
                # regime would.
                policy["backend_final_regime"] = backend_regime
        return policy

    @staticmethod
    def _final_selection_from_profile(
        profile: AutotuneEffortProfile,
    ) -> FinalSelectionConfig | None:
        """The profile's final selection policy with the rounds override."""
        policy = profile.final_selection
        if policy is None:
            return None
        rounds = _env_get_optional_int(_FINAL_SELECTION_ROUNDS_ENV)
        if rounds is not None:
            policy = dataclasses.replace(policy, rounds=max(0, rounds))
        if policy.rounds <= 0:
            return None
        return policy

    @classmethod
    def _final_selection_from_settings(
        cls, settings: Settings
    ) -> FinalSelectionConfig | None:
        return cls._final_selection_from_profile(
            get_effort_profile(settings.autotune_effort)
        )

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        """
        Retrieve extra kwargs from the effort profile for the autotuner.
        """
        finishing_rounds = _env_get_optional_int("HELION_AUTOTUNE_FINISHING_ROUNDS")
        if finishing_rounds is None:
            finishing_rounds = profile.finishing_rounds

        return {
            "finishing_rounds": finishing_rounds,
            **super().get_kwargs_from_profile(profile, settings),
        }

    def _final_selection_enabled(self) -> bool:
        """Whether the final pick runs the interleaved steady-window shootout."""
        return (
            self.final_selection is not None
            and self.settings.autotune_benchmark_fn is None
            and not isinstance(self.benchmark_provider, MultiShapeBenchmarkProvider)
        )

    def _final_selection_shortlist_estimate(self) -> int:
        """Longest shortlist ``final_rebenchmark_best`` can hand the selection:
        the retained ``top_k``, the pinned seeds, the incumbent (appended
        when a later phase re-timed it above the shortlist's numbers) and
        the family quota's representatives where the spec names family
        knobs."""
        return (
            self._final_rebenchmark_top_k()
            + len(self._pinned_finalist_configs)
            + 1
            + self._final_selection_sizing_policy().family_finalists
        )

    def _final_selection_family_keys(self) -> tuple[str, ...]:
        """The config knobs whose values pick a distinct kernel body
        (``ConfigSpec.final_selection_family_keys``); empty when the surface
        names none, which disables the final selection's family quota."""
        return tuple(self.config_spec.final_selection_family_keys)

    def _final_selection_sizing_policy(self) -> FinalSelectionConfig:
        """The policy the phase is sized by under a budget: the profile's,
        less the second regime's column when the regimes call for none
        (``final_selection_second_regime``), so a per-call flushed selection
        (the default) neither reserves search time for a column it never
        takes nor sheds rounds to fit one, and less the family quota where
        the spec names no family knob. The cache policy keeps the
        profile's fields (``_final_rebenchmark_cache_policy``), so the key
        ignores the regime environment variables. The regimes are read from
        the environment and the backend here, a superset of what the
        measurer will honour (an upper bound): a timer without regimes (a
        backend's own bench) still reserves the column under ``replay``, and
        ``kernel`` and ``call`` reserve a second column for each other
        without asking the worker whether it can trace (``_traceable``: the
        probe is a profiling session in the worker that times the search, so
        only the selection itself and the timer choice of a search that
        would move, ``_settle_candidate_timer``, ask).
        """
        policy = self.final_selection
        assert policy is not None
        backend_regime = self._backend_final_regime()
        regime = final_selection_l2_flush_mode(backend_regime)
        second = final_selection_second_regime(regime, backend_regime)
        if policy.tiebreak_rounds > 0 and second is None:
            policy = dataclasses.replace(policy, tiebreak_rounds=0)
        # the family quota admits nothing where the spec names no family knob
        if policy.family_finalists > 0 and not self._final_selection_family_keys():
            policy = dataclasses.replace(policy, family_finalists=0)
        return policy

    def _budget_reserve_seconds(self) -> float:
        budget = self.settings.autotune_budget_seconds
        if budget is None or not self._final_selection_enabled():
            return 0.0
        policy = self._final_selection_sizing_policy()
        return min(
            policy.estimated_seconds(self._final_selection_shortlist_estimate()),
            _FINAL_SELECTION_RESERVE_FRACTION * budget,
        )

    def _minimum_final_selection_seconds(self) -> float:
        if not self._final_selection_enabled():
            return 0.0
        smallest = self._final_selection_sizing_policy().budget_ladder()[-1]
        return smallest.estimated_seconds(self._final_selection_shortlist_estimate())

    def _final_selection_plan(self, shortlist: int) -> FinalSelectionConfig | None:
        """The policy sized to the budget left when the phase starts.

        Without a budget the profile's policy runs as is. Rank 0's plan is
        used on every rank so the ranks take the same windows together.
        """
        policy = self.final_selection
        assert policy is not None
        budget = self.settings.autotune_budget_seconds
        if budget is not None and self._autotune_budget_start is not None:
            remaining = budget - (time.perf_counter() - self._autotune_budget_start)
            sizing = self._final_selection_sizing_policy()
            plan = sizing.fit_to_seconds(remaining, shortlist)
            if plan is None:
                smallest = sizing.budget_ladder()[-1].estimated_seconds(shortlist)
                self.log(
                    f"Final selection skipped: {remaining:.0f}s of the "
                    f"{budget}s budget remain, less than the {smallest:.0f}s of "
                    "its smallest plan; returning the search's best."
                )
                policy = None
            elif plan != sizing:
                self.log(
                    f"Final selection sized to the {remaining:.0f}s left of the "
                    f"{budget}s budget: up to {plan.finalists(shortlist)} finalists x "
                    f"{plan.rounds} rounds"
                    + (", no screen" if sizing.screens(shortlist) else "")
                    + (
                        ", one regime"
                        if plan.tiebreak_rounds == 0 < sizing.tiebreak_rounds
                        else ""
                    )
                    + (
                        ", no thermal warmup"
                        if plan.thermal_warmup_ms == 0 < sizing.thermal_warmup_ms
                        else ""
                    )
                )
                policy = plan
            # else the whole plan fits: the profile's policy runs as is (the
            # column the sizing left out cannot run in this regime anyway)
        return sync_object(
            policy, process_group_name=self.kernel.env.process_group_name
        )

    def _warn_ignored_final_rebenchmark_envs(self) -> None:
        """Say once that the single-pass knobs do not steer the selection."""
        global _final_selection_ignored_envs_warned
        if _final_selection_ignored_envs_warned:
            return
        ignored = [name for name in _FINAL_SELECTION_IGNORED_ENVS if name in os.environ]
        if not ignored:
            return
        _final_selection_ignored_envs_warned = True
        self.log.warning(
            f"{', '.join(ignored)} only steer the single-pass final verification; "
            "the full-effort final selection ignores them (set "
            f"{_FINAL_SELECTION_ROUNDS_ENV}=0 to run that verification instead)."
        )

    def _collapses_source_aliases(self) -> bool:
        """Whether this search drops source aliases: the backend asks for it
        (``Backend.collapses_generated_source_aliases``) and the search's
        provider reports aliases (``BenchmarkProvider.settles_source_aliases``).

        Such a search counts an alias as the kernel it repeats.  The alias
        has that kernel's reading (``_settle_benchmarked_members``), a
        re-timing times the config that holds the kernel once and never the
        alias (``_rebenchmark_kernels``), two configs of one kernel compare
        equal (``compare``), and the alias takes no entry in the history the
        final selection draws its shortlist from (a pinned one passes its
        pin to the kernel's config), so the final selection times a kernel
        under one config.  The pattern and LFBO searches also drop a
        generation's new aliases from the population, so no search path and
        no polish round moves to one, and start one path per kernel; with
        the copies gone the LFBO polish descent crosses to new code that
        reads level with its incumbent, within
        ``LFBOPatternSearch.polish_patience``.  Aliases that stay members
        until the final selection: those of an initial population, and of
        the differential-evolution and LLM-guided searches.

        Two providers of such a backend report no aliases: the one of a
        multi-shape search, whose ``benchmark`` results are joint readings
        of every shape, and a local one once torch.distributed is
        initialized, where every rank keeps the same benchmark loop.  There
        every config stays a candidate of its own and is timed as one, and
        none of the above is switched on.
        """
        return bool(
            self.config_spec.backend.collapses_generated_source_aliases(
                self.config_spec
            )
            and self.benchmark_provider.settles_source_aliases()
        )

    def _is_source_alias(self, member: PopulationMember) -> bool:
        """Whether ``member`` repeats the generated source of an earlier
        candidate of this run, in a search that drops such aliases."""
        return member.status == "deduplicated" and self._collapses_source_aliases()

    def _without_source_aliases(
        self,
        members: list[PopulationMember],
        benchmarked: Sequence[PopulationMember],
    ) -> list[PopulationMember]:
        """``members`` without the source aliases among the just benchmarked
        ``benchmarked``.

        An alias is an earlier candidate's code under another config: its
        reading is that candidate's (``_settle_benchmarked_members``), so
        keeping it would let a search path move to code it already has and
        fill the final selection's shortlist with copies of one kernel.
        Members benchmarked earlier stay, aliases included (an initial
        population keeps its own; a verification still times their kernel
        once: ``_rebenchmark_kernels``).
        """
        aliases = {
            id(member) for member in benchmarked if self._is_source_alias(member)
        }
        if not aliases:
            return members
        return [member for member in members if id(member) not in aliases]

    def _kernel_holder(
        self, member: PopulationMember, among: Sequence[PopulationMember] = ()
    ) -> PopulationMember:
        """The member that holds ``member``'s kernel: the one place that
        says which config stands for a kernel and has its reading.

        Where the search drops source aliases, an alias is another config of
        a kernel an earlier candidate produced and holds that candidate's
        compiled function (the ``fn`` of a ``"deduplicated"`` result), so
        several members can be configs of one kernel (an initial population
        keeps its aliases).  One of them holds it: it is the member a
        re-timing times, its reading is the kernel's, and its config is the
        kernel's entry in the final selection.

        It is the member with the function that is no alias: one among
        ``among`` where the members at hand hold the kernel themselves (a
        batch with the config that was timed, a list with the final
        selection's entry of the kernel), else the member the search first
        timed it under (``_kernels``), wherever that member is now.  Where
        the search knows the kernel under no such config, the first alias of
        it stands in.  A member that is no alias holds its own kernel, as
        does every member of a search that keeps its aliases.
        """
        if not self._is_source_alias(member):
            return member
        stand_in = None
        for other in among:
            if other.fn is member.fn:
                if not self._is_source_alias(other):
                    return other
                if stand_in is None:
                    stand_in = other
        return self._kernels.get(id(member.fn)) or stand_in or member

    def _kernels_of(
        self, members: Sequence[PopulationMember]
    ) -> list[PopulationMember]:
        """The members that hold the kernels of ``members``, each once, in
        the order of ``members``."""
        kernels = {
            id(kernel): kernel
            for kernel in (self._kernel_holder(member, members) for member in members)
        }
        return [*kernels.values()]

    def _one_member_per_kernel(
        self, members: Sequence[PopulationMember]
    ) -> list[PopulationMember]:
        """``members`` with one member for each kernel, in their order: the
        one that holds the kernel where ``members`` has it, else the first
        config of it.  (A search path that started from a second config of a
        kernel would generate that kernel's neighborhood again, and a final
        selection that held one would time the kernel twice.)"""
        listed = {id(member) for member in members}
        kept: dict[int, PopulationMember] = {}
        for member in members:
            holder = self._kernel_holder(member, members)
            kept.setdefault(id(holder), holder if id(holder) in listed else member)
        return [*kept.values()]

    def _settle_benchmarked_members(
        self,
        members: Sequence[PopulationMember],
        batch: Sequence[PopulationMember],
    ) -> None:
        """What every member takes once it has a result: ``members`` of the
        just benchmarked ``batch``, or earlier members that a result of it
        repaired.

        A member with a kernel of its own holds that kernel from here on
        (``_kernels``) and is recorded for the final selection
        (``_record_benchmarked_member``).  A source alias is not recorded:
        the history already holds its code under the config that holds the
        kernel.  It takes the reading that kernel has now in place of the
        provider's record of the kernel's first reading (every verification
        has re-timed the kernels near the best since).

        A pinned alias passes its pin to the kernel's config.  A pinned
        config (a compiler seed, the default) reaches the final selection
        whatever the search read for it; one that repeats an earlier
        candidate's code takes no entry there, since the kernel would be
        screened twice (``_refresh_benchmarked_members_after_rebenchmark``),
        so the kernel keeps its place in the final selection whichever of its
        configs was benchmarked first.
        """
        collapse = self._collapses_source_aliases()
        aliases = []
        for member in members:
            if self._is_source_alias(member):
                aliases.append(member)
                continue
            if collapse and member.perfs and math.isfinite(member.perf):
                holder = self._kernels.get(id(member.fn))
                # (an alias stands in until the search times the kernel
                # under a config of its own)
                if holder is None or self._is_source_alias(holder):
                    self._kernels[id(member.fn)] = member
            self._record_benchmarked_member(member)
        for alias in aliases:
            kernel = self._kernel_holder(alias, batch)
            if kernel is alias:
                self._kernels.setdefault(id(alias.fn), alias)
                continue
            if kernel.perfs:
                alias.perfs[-1] = kernel.perf
            if (
                alias.config in self._pinned_finalist_configs
                and not self._is_source_alias(kernel)
            ):
                self.pin_finalist_config(kernel.config)
                self._record_best_member_for_config(
                    self._pinned_finalist_members, kernel.config, kernel
                )

    def _simplified_selected_config(self, best: Config) -> Config:
        """``best``, or a simpler config of the same kernel that the backend
        proposes (``Backend.simplified_selected_config``; CuTe: without the
        load-eviction policies the generated code does not depend on), where
        the search drops source aliases.

        Such a search times a kernel under the first config that produced
        its source, so the selection can carry settings that leave no trace
        in the code.  The backend proposes simpler configs and each is
        rendered here in process; one is taken only when its generated
        source hash is that of the kernel the search measured for its
        selection (the compiled function of the member it holds for
        ``best``), so the returned config is the measured kernel.

        A proposal is a candidate config like any other, and the rules for
        one are applied in one place each, to every proposal:

        - It passes the user's ``autotune_config_filter`` and the backend's
          viability check before it is compiled and stands for the selection
          only as it was proposed.  The filters see a copy, and the config
          they return has to be the proposal: a filter that rewrites its
          argument, in place or into another config, has asked for other
          code than the one that was proposed.
        - Every render is compile work, which ``autotune_budget_seconds``
          covers: none starts once the budget is used up, and the proposals
          accepted until then stand.
        - A render that fails is no kernel: that proposal is not taken.  The
          failure is never raised and is handled as the search handles a
          candidate that fails to compile: the compiler's output is
          captured, and the failure is logged by its classification, not at
          all under ``autotune_ignore_errors``.

        The search returns only a config this step accepted: a backend that
        answers with one it did not ask about gets the selection back.  Not
        for a multi-shape search or in a process with torch.distributed
        initialized (one render does not speak for every shape or rank;
        neither drops aliases, ``_collapses_source_aliases``), not under
        user overrides (their values stay as given), not inside
        ``find_handoff``, whose session confirms the configs that were
        measured, and not in a search whose config only seeds another
        search (``_simplifies_selected_config``).
        """
        if (
            not self._simplifies_selected_config
            or not self._collapses_source_aliases()
            or self.settings.autotune_config_overrides
            or _active_handoff.get() is not None
        ):
            return best
        backend = self.config_spec.backend
        # The kernel that was measured: the member the search holds for its
        # selection.
        searched = self._selected_member
        if searched is None or _config_key(searched.config) != _config_key(best):
            searched = self._benchmarked_members.get(best)
        if searched is None:
            return best
        selected = backend.generated_source_hash(searched.fn)
        if selected is None:
            return best
        accepted: set[Hashable] = set()

        def renders_selected_source(candidate: Config) -> bool:
            if self._autotune_budget_exceeded():
                return False
            try:
                with capture_output():
                    compiled = self.kernel.compile_config(candidate, allow_print=False)
            except Exception as error:
                # As for a search candidate that fails to compile: no kernel.
                log_candidate_compile_failure(
                    self.log,
                    self.settings,
                    backend,
                    error,
                    f"Not taking {candidate} in place of the selected config: "
                    "rendering it to compare the generated sources failed",
                )
                return False
            return backend.generated_source_hash(compiled) == selected

        def same_source(candidate: Config) -> bool:
            # The filters get a copy, so what they return is compared with
            # the proposal as the backend made it.
            key = _config_key(candidate)
            # (a proposal is no candidate of the search: one the filters
            # drop takes no "filtered" trial row)
            passing, _indices = self._apply_config_filter(
                [_stored_config(candidate.config)], record_trace=False
            )
            if (
                not passing
                or _config_key(passing[0]) != key
                or not renders_selected_source(candidate)
            ):
                return False
            accepted.add(key)
            return True

        simplified = backend.simplified_selected_config(
            self.config_spec, best, same_source
        )
        if _config_key(simplified) == _config_key(best):
            return best
        if _config_key(simplified) not in accepted:
            self.log.warning(
                "Keeping the selected config as it is: the backend proposed "
                f"{simplified} in its place without asking whether it "
                "renders to the selected code"
            )
            return best
        self.log(
            "Returning a simpler config with the selected config's generated "
            f"code: {simplified} (selected as {best})"
        )
        # The run's records name the returned config and keep the ones the
        # kernel was measured under. The trace's trial rows are keyed by the
        # configs the search timed: the kernel's is the config that holds it
        # (the selection itself can be a config that was never timed: the
        # finishing phase ends on a minimized one). The final selection's
        # rows are keyed by the configs it timed, the selection among them.
        holder = self._kernels.get(id(searched.fn))
        self._selected_config_measured_as = best if holder is None else holder.config
        selection = self._autotune_metrics.final_selection
        if selection is not None:
            # (only a run with the record computes the ids)
            measured_id = canonical_config_id(best)
            if selection.get("selected_config_id") == measured_id:
                selection["measured_config_id"] = measured_id
                selection["selected_config_id"] = canonical_config_id(simplified)
        # The returned config stays linked to the measured kernel.
        self._selected_member = dataclasses.replace(
            searched,
            config=simplified,
            flat_values=self.config_gen.flatten(simplified),
            perfs=[*searched.perfs],
        )
        return simplified

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
        canonical_flat, config = self.config_gen.canonicalize_flat(flat_values)
        member = PopulationMember(_unset_fn, [], canonical_flat, config)
        self.benchmark_population([member], desc="Benchmarking")
        return member

    def benchmark_flat_batch(
        self,
        to_check: list[FlatConfig],
        *,
        random_fallback_target: int | None = None,
    ) -> list[PopulationMember]:
        """
        Benchmark multiple flat configurations in parallel.

        For an ordinary batch the returned list preserves positional
        correspondence with ``to_check``.  Invalid configurations that cannot
        be unflattened are represented as ``PopulationMember`` objects with
        ``perf == inf`` and ``status == "error"`` (they are not benchmarked).

        Inside :meth:`_initial_population_batch` (or with a
        ``random_fallback_target``) ``to_check`` is the initial population and
        is benchmarked through :meth:`benchmark_initial_population` (budget
        share, the drop of never-attempted configs, the summary line, the
        random fallback when a target is given, and the search's choice of
        its candidates' timer, which can read the members again, drop the
        ones the trace timer cannot read and leave one whose re-timing
        failed with an infinite reading).

        Args:
            to_check: A list of flat configurations to benchmark.
            random_fallback_target: When given, ``to_check`` is a seed-only
                initial population; see :meth:`benchmark_initial_population`.

        Returns:
            For an ordinary batch, one population member per entry in
            ``to_check`` in order.  For the initial population, the members
            that were attempted (never-attempted configs are dropped, random
            fallback members appended, and configs that a search settling on
            the trace timer could not read there are dropped), followed by
            the invalid entries; the positions no longer correspond to
            ``to_check``.
        """
        initial_population = (
            self._benchmarking_initial_population or random_fallback_target is not None
        )
        from ..runtime.config import Config

        valid: list[PopulationMember] = []
        invalid: list[PopulationMember] = []
        result: list[PopulationMember] = []
        for flat in to_check:
            m = self.make_unbenchmarked(flat)
            if m is not None:
                valid.append(m)
                result.append(m)
            else:
                invalid.append(
                    PopulationMember(
                        _unset_fn, [float("inf")], flat, Config(), status="error"
                    )
                )
                result.append(invalid[-1])

        if not initial_population:
            self.benchmark_population(valid)
            return result
        self.benchmark_initial_population(
            valid, random_fallback_target=random_fallback_target
        )
        return [*valid, *invalid]

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
            canonical_flat, config = self.config_gen.canonicalize_flat(flat_values)
        except exc.InvalidConfig:
            self.config_gen.invalid_config_count += 1
            return None
        if not self._backend_config_is_viable(config):
            self.config_gen.invalid_config_count += 1
            return None
        return PopulationMember(_unset_fn, [], canonical_flat, config)

    def _pad_initial_population_with_unique_random(
        self,
        population: Sequence[FlatConfig],
        target: int,
        *,
        generation: ConfigGeneration | None = None,
    ) -> list[FlatConfig]:
        """Pad an initial population with normalized, unique random configs."""
        generation = self.config_gen if generation is None else generation
        result: list[FlatConfig] = []
        seen: MutableSet[Config] = ConfigSet()
        invalid = 0
        duplicate = 0

        def append_if_valid(flat: FlatConfig) -> None:
            nonlocal invalid, duplicate
            try:
                canonical_flat, config = generation.canonicalize_flat(flat)
            except exc.InvalidConfig:
                invalid += 1
                return
            if not self._backend_config_is_viable(config):
                invalid += 1
                return
            if config in seen:
                duplicate += 1
                return
            seen.add(config)
            result.append(canonical_flat)

        for flat in population:
            append_if_valid(flat)

        attempts = 0
        max_attempts = max(64, max(0, target - len(result)) * 64)
        while len(result) < target and attempts < max_attempts:
            attempts += 1
            append_if_valid(generation.random_flat())

        if len(result) < target:
            self.log(
                "Generated only "
                f"{len(result)}/{target} unique valid initial population configs "
                f"after {attempts} random attempts "
                f"({invalid} invalid, {duplicate} duplicate)."
            )
        if attempts or invalid or duplicate:
            self.log(
                f"Initial population after unique random padding: {len(result)} total"
            )
        return result

    def _replace_backend_rejected_initial_configs(
        self,
        population: Sequence[FlatConfig],
        target: int,
    ) -> list[FlatConfig]:
        """Refill only when backend screening rejects an initial candidate."""
        for flat in population:
            try:
                _canonical_flat, config = self.config_gen.canonicalize_flat(flat)
            except exc.InvalidConfig:
                continue
            if not self._backend_config_is_viable(config):
                return self._pad_initial_population_with_unique_random(
                    population, target
                )
        return list(population)

    def _generate_best_available_population_flat(
        self, *, generation: ConfigGeneration | None = None
    ) -> list[FlatConfig]:
        """
        Generate initial population using default config, explicit seed configs,
        and cached configs.

        Starts with explicit seed configs, compiler-owned seed configs,
        caller-provided best-available seeds, and the default configuration,
        then adds up to MAX_BEST_AVAILABLE_CONFIGS matching cached configs from
        previous runs. Seeds are tried before the raw default so backend-promoted
        fast paths (for example CuTe flash attention) are benchmarked first even
        when the seed is not also promoted to
        ``ConfigSpec.compiler_default_config``. Explicit seed configs are not
        suppressed by cache-skip settings. No random configs are added.

        Returns:
            A list of unique FlatConfig values for the initial population.
            Minimum size is 1 (default), plus any valid unique explicit/compiler
            seed configs and up to autotune_best_available_max_configs cached
            configs.
        """
        projected = generation is not None and generation._initial_sampling
        generation = self.config_gen if generation is None else generation
        max_configs = self.settings.autotune_best_available_max_configs

        seen: MutableSet[Config] = ConfigSet()
        result: list[FlatConfig] = []
        pinned_configs: MutableSet[Config] = ConfigSet()

        # User seed configs are explicit requests, so try them before compiler-owned
        # seeds, the raw default, and cached configs while still deduplicating
        # normalized configs.
        for flat, transferred_config in generation.user_seed_flat_config_pairs(
            self._autotune_seed_configs(), self.log
        ):
            if transferred_config not in seen:
                seen.add(transferred_config)
                pinned_configs.add(transferred_config)
                result.append(flat)

        # Compiler-owned seeds come from ConfigSpec.compiler_seed_configs;
        # they encode backend/compiler heuristics and complement user seed configs.
        # Keep them before the raw fragment default so expensive fallback defaults
        # cannot starve a known fast compiler seed in FROM_BEST_AVAILABLE mode.
        for flat, transferred_config in generation.seed_flat_config_pairs(self.log):
            if transferred_config not in seen:
                seen.add(transferred_config)
                pinned_configs.add(transferred_config)
                result.append(flat)

        for config in self._best_available_seed_configs:
            try:
                flat = generation.flatten(config)
                transferred_config = generation.unflatten(flat)
                if transferred_config not in seen:
                    seen.add(transferred_config)
                    pinned_configs.add(transferred_config)
                    result.append(flat)
            except (ValueError, TypeError, KeyError, AssertionError) as e:
                self.log(f"Failed to transfer explicit seed config: {e}")

        default_flat = generation.default_flat()
        default_config = generation.unflatten(default_flat)
        if default_config not in seen:
            seen.add(default_config)
            pinned_configs.add(default_config)
            result.append(default_flat)
        self._pinned_finalist_configs.update(pinned_configs)
        self.log("Starting with seed/default configs")

        cached_entries = (
            self._find_similar_cached_configs(
                max_configs,
                config_spec_hash=self.config_spec.projected_cache_fingerprint_hash(
                    advanced_controls_files=self.settings.autotune_search_acf or None
                ),
            )
            if projected
            else self._find_similar_cached_configs(max_configs)
        )

        if cached_entries:
            self.log.debug(
                f"Found {len(cached_entries)} cached config(s) from previous runs"
            )

        duplicates = 0
        for i, entry in enumerate(cached_entries):
            try:
                self.log.debug(f"Cached config {i + 1}: {entry.config}")
                flat = entry.to_mutable_flat_config()
                if projected:
                    flat, transferred_config = generation.projected_cache_flat_pair(
                        flat
                    )
                else:
                    transferred_config = generation.unflatten(flat)
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

        self.log(f"Seed/default/cache population: {len(result)} total")

        return result

    def _append_compiler_coverage(
        self, population: list[FlatConfig], *, use_cache: bool
    ) -> list[FlatConfig]:
        from .compiler_coverage import append_compiler_coverage

        def cached_configs() -> list[Config]:
            if not use_cache:
                return []
            result: list[Config] = []
            for entry in self._find_similar_cached_configs(
                self.settings.autotune_best_available_max_configs
            ):
                try:
                    result.append(
                        self.config_gen.strict_unflatten(entry.to_mutable_flat_config())
                    )
                except (
                    exc.InvalidConfig,
                    ValueError,
                    TypeError,
                    KeyError,
                    AssertionError,
                ) as error:
                    self.log(
                        f"Failed to transfer expanded coverage cache config: {error}"
                    )
            return result

        result, outcomes = append_compiler_coverage(
            population,
            self.config_gen,
            cached_configs=cached_configs,
            pin=self.pin_finalist_config,
        )
        self.compiler_coverage_outcomes = outcomes
        for outcome in outcomes:
            self.log.debug(
                f"Compiler coverage {outcome.mechanism} ({outcome.origin}): "
                f"{outcome.outcome}; requested={outcome.requested}, effective={outcome.effective}"
            )
        return result

    def set_best_available_seed_configs(
        self,
        configs: Sequence[Config],
    ) -> None:
        self._best_available_seed_configs = list(configs)

    def benchmark_population(
        self,
        members: list[PopulationMember],
        *,
        desc: str = "Benchmarking",
        raise_if_no_viable_config: bool = True,
    ) -> list[PopulationMember]:
        """
        Benchmark multiple population members in parallel.  Members should be created with make_unbenchmarked.

        Inside :meth:`_initial_population_batch` the batch runs under the
        initial population's share of the autotune budget
        (``_initial_population_budget``).

        Args:
            members: The list of population members to benchmark.
            desc: Description for the progress bar.
            raise_if_no_viable_config: See :meth:`BaseSearch.benchmark_batch`.
        """
        initial_population = self._benchmarking_initial_population
        budget = (
            self._initial_population_budget(members)
            if initial_population
            else contextlib.nullcontext()
        )
        with budget:
            results = self.benchmark_batch(
                [m.config for m in members],
                desc=desc,
                raise_if_no_viable_config=raise_if_no_viable_config,
            )
        if initial_population:
            # For the batch summary (see ``_benchmark_initial_batch``).
            self._last_batch_first_error = (
                self.benchmark_provider.batch_first_compile_error
            )
        for member, result in zip(members, results, strict=True):
            member.config = result.config
            member.flat_values = self.config_gen.flatten(result.config)
            member.perfs.append(result.perf)
            member.fn = result.fn
            member.status = result.status
            member.compile_time = result.compile_time
            if self.traces_candidates and member.status == "deduplicated":
                # Settled with a copy of its kernel's reading: no reading of
                # this member's own, and no pair with the first one it takes
                # (``rereading_spread``).
                member.timer_readings_from = len(member.perfs)
        self._settle_benchmarked_members(members, members)
        repairs = self.benchmark_provider.take_effective_source_repairs()
        if repairs:
            self._apply_effective_source_repairs(repairs, members)
        return members

    @contextlib.contextmanager
    def _initial_population_batch(self) -> Iterator[None]:
        """Mark the batches run inside as the initial population.

        :meth:`benchmark_population` then runs under the population's budget
        share and :meth:`benchmark_flat_batch` routes through
        :meth:`benchmark_initial_population`; both keep their signatures.
        """
        previous = self._benchmarking_initial_population
        self._benchmarking_initial_population = True
        try:
            yield
        finally:
            self._benchmarking_initial_population = previous

    def _initial_population_floor(self, total: int) -> int:
        """Leading configs of an initial batch the budget share never cuts.

        A CuTe flash population opens with the structural coverage design,
        one witness per ordinary leaf (and a second for the qualification);
        cutting into those rows would leave the structural qualification to
        re-probe the missing leaves outside the share, or to fail when the
        budget is not actually spent. Other surfaces have no floor.
        """
        if total <= 0 or not self.config_spec.cute_flash_search_enabled:
            return 0
        return min(
            total, self.config_gen.extension.structural_qualification_prefix_count()
        )

    @contextlib.contextmanager
    def _initial_population_budget(
        self, members: Sequence[PopulationMember]
    ) -> Iterator[None]:
        """Cut the initial population to what its share of the budget affords.

        Only under ``autotune_budget_seconds``, and only without a precompile
        pool: there configs compile and run one at a time, so the cost per
        config is known after the first few and a cut wastes nothing compiled
        ahead (with a pool the whole batch is precompiled first and keeps the
        global budget check with its first-measurement exemption). The
        provider asks the installed budget hook before each config; this hook
        adds the population's own limit to the global one. Once two configs
        (and the surface's floor, ``_initial_population_floor``) have been
        attempted and one config has been measured successfully, the average
        wall time per config so far predicts whether the next one would push
        the batch past ``_INITIAL_POPULATION_BUDGET_FRACTION`` of the search
        phase (the budget minus the final-selection reserve, counted from the
        batch's start, so a random fallback batch gets its own share); if so
        the population stops there, logged once with the measured cost. The
        configs never attempted come back ``skipped`` and the caller drops
        them. A population that fits its share is untouched, so cheap kernels
        keep their full population and search trajectory.
        """
        budget = self.settings.autotune_budget_seconds
        start = self._autotune_budget_start
        provider = self.benchmark_provider
        if (
            budget is None
            or start is None
            or not isinstance(provider, LocalBenchmarkProvider)
            or provider.compiles_ahead
        ):
            yield
            return
        share = _INITIAL_POPULATION_BUDGET_FRACTION * (
            budget - self._budget_reserve_seconds()
        )
        total = len(members)
        floor = self._initial_population_floor(total)
        # Each batch of the phase (the seeds, then a random fallback) gets
        # the share from its own start.
        phase_start = time.perf_counter()
        deadline = phase_start + share
        metrics = self._autotune_metrics
        cut = False

        def exceeded() -> bool:
            nonlocal cut
            if self._autotune_budget_exceeded() or cut:
                return True
            attempted = provider.batch_configs_started
            # The cut only ever follows a measured config: with nothing
            # measured the provider would attempt a few more anyway, and
            # the message below is only logged when it stops the batch.
            if (
                attempted < max(2, floor)
                or metrics.num_successful_candidate_measurements == 0
            ):
                return False
            now = time.perf_counter()
            per_config = (now - phase_start) / attempted
            if now + per_config <= deadline:
                return False
            cut = True
            self.log(
                f"Initial population cut to {attempted} of {total} configs: "
                f"{per_config:.1f}s per config (compile and benchmark) against "
                f"the {share:.0f}s the initial population may take of the "
                f"{budget}s budget; the search goes on from the configs measured "
                "so far."
            )
            return True

        previous = provider.budget_exceeded_fn
        provider.set_budget_exceeded_fn(exceeded)
        try:
            yield
        finally:
            provider.set_budget_exceeded_fn(previous)

    def _benchmark_initial_batch(
        self,
        members: list[PopulationMember],
        *,
        desc: str,
        raise_if_no_viable_config: bool,
        visited: MutableSet[Config] | None,
    ) -> None:
        """Benchmark one batch of the initial population under its budget share.

        Members the batch never attempted (``skipped``) are removed from
        ``members`` and ``visited`` rather than kept as failures; the batch's
        outcome is summarized in one log line whatever
        ``autotune_ignore_errors`` says.
        """
        started = time.perf_counter()
        with self._initial_population_batch():
            self.benchmark_population(
                members,
                desc=desc,
                raise_if_no_viable_config=raise_if_no_viable_config,
            )
        skipped = [member for member in members if member.status == "skipped"]
        if skipped:
            members[:] = [member for member in members if member.status != "skipped"]
            if visited is not None:
                for member in skipped:
                    visited.discard(member.config)
        attempted = len(members)
        measured = sum(
            1 for member in members if member.perfs and math.isfinite(member.perf)
        )
        elapsed = time.perf_counter() - started
        per_config = f" ({elapsed / attempted:.1f}s per config)" if attempted else ""
        first_error = self._last_batch_first_error
        failed = f"{attempted - measured} failed to compile or run"
        if attempted > measured and first_error is not None:
            failed += f" (first: {type(first_error).__name__}: {first_error})"
        self.log(
            f"{desc}: {measured} of {attempted} configs measured in "
            f"{elapsed:.0f}s{per_config}; {failed}; {len(skipped)} not attempted "
            "within the autotune budget."
        )

    def benchmark_initial_population(
        self,
        members: list[PopulationMember],
        *,
        random_fallback_target: int | None = None,
        visited: MutableSet[Config] | None = None,
        desc: str = "Initial population",
    ) -> list[PopulationMember]:
        """Benchmark the initial population, recovering from an all-failing seed set.

        ``random_fallback_target`` is the number of unique random configs to
        add when ``members`` holds only seed/default/cache configs and none of
        them compiles or runs (for example a backend default config that
        raises ``BackendUnsupported`` for this kernel). It counts configs
        beyond the seeds, so the fallback is the same size however many
        cached configs were seeded. ``None`` keeps the immediate re-raise of
        the compile error, which is right for random populations: their total
        failure already shows the search space is broken, so padding would
        only double the time to the same error. Fallback members are appended
        to ``members`` (and ``visited``) in place; when no fallback config can
        be generated, or the fallback also fails entirely, the compile error
        is raised exactly as for a random population.

        Under ``autotune_budget_seconds`` the population is also cut to what
        its share of the budget affords (``_initial_population_budget``);
        configs the cut (or the budget itself) never attempted are dropped
        from ``members`` and ``visited`` rather than kept as failures, and
        the outcome is summarized in one log line whatever
        ``autotune_ignore_errors`` says.

        Once the population has a reading, the search chooses the timer of
        its candidates (``_settle_candidate_timer``). A search that settles
        on the CUDA activity trace reads every member with a reading once
        more there, which replaces the member's CUDA-event readings, and
        drops from ``members`` and ``visited`` the configs whose kernel the
        trace read in neither of two passes: a caller that kept its own
        list of the population holds configs that are no longer in it. A
        member whose re-timing there leaves a sticky device error reads
        infinite afterwards, whichever timer the search goes on with.
        """
        self._benchmark_initial_batch(
            members,
            desc=desc,
            raise_if_no_viable_config=random_fallback_target is None,
            visited=visited,
        )
        if random_fallback_target is None or any(
            math.isfinite(member.perf) for member in members
        ):
            self._settle_candidate_timer(members, visited)
            return members
        compile_error = self.benchmark_provider.take_no_viable_config_error()
        seen = (
            ConfigSet(member.config for member in members)
            if visited is None
            else visited
        )
        fallback: list[PopulationMember] = []
        # The padding helper re-adds the seeds first, so pad past them.
        for flat in self._pad_initial_population_with_unique_random(
            [member.flat_values for member in members],
            len(members) + random_fallback_target,
        ):
            member = self.make_unbenchmarked(flat)
            if member is not None and member.config not in seen:
                seen.add(member.config)
                fallback.append(member)
        if not fallback:
            # Nothing to fall back to: surface the real compile error rather
            # than letting the search end in a generic NoConfigFound.
            if compile_error is not None:
                raise compile_error
            return members
        self.log.warning(
            f"None of the {len(members)} seed/default/cache configs compiled or "
            "ran (the backend may not support the default config for this "
            f"kernel); continuing with {len(fallback)} random configs."
        )
        self._benchmark_initial_batch(
            fallback,
            desc="Random fallback population",
            raise_if_no_viable_config=True,
            visited=visited,
        )
        members.extend(fallback)
        self._settle_candidate_timer(members, visited)
        return members

    def _settle_candidate_timer(
        self,
        members: list[PopulationMember],
        visited: MutableSet[Config] | None = None,
    ) -> None:
        """Choose, once per search, the timer that reads its candidates,
        after the initial population ``members`` was benchmarked.

        The initial population is timed with CUDA events, as every search's
        was: one event pair around each flushed launch, which reads the
        kernel plus the launch latency in the event clock's steps. When the
        backend names a bound (``Backend.autotune_short_kernel_below_us``)
        and the best of those readings is below it, the kernel's configs
        differ by less than that timer resolves, and every reading from here
        on (each candidate, each re-timing) is a kernel duration from the
        CUDA activity trace in the benchmark worker instead
        (``LocalBenchmarkProvider.trace_search_candidates``).

        The choice is the search's, made once, rather than each reading's:
        the trace leaves the launch latency out (4-5 us on a B200), so a
        launch near the bound that picked its timer by its own reading read
        on two scales from call to call (59 or 64 us for one config). A
        search stays on one timer for all its candidates, short or long.
        Where the backend names a bound, a search behind a benchmark prelude
        (``benchmark_prelude``) moves to the trace whatever its best reading:
        behind work that the device finishes before the host has queued the
        candidate, an event-timed sample also holds the candidate's launch
        on the host (``do_bench``), which the trace leaves out.
        The initial population moves with it: every member with a reading
        is read once more, on the trace (``_read_on_settled_timer``), before
        anything compares it with a traced candidate, and a member the trace
        cannot read leaves ``members`` and ``visited``. It keeps the events
        when, without a prelude, the best reading is at or above the bound,
        for a ratio objective, in another L2 regime than the per-call flush,
        under ``HELION_BENCHMARK_CUDAGRAPH=1`` (``do_bench`` then times a
        candidate's graph replay and traces nothing), when the population
        was not timed in the benchmark worker, when that worker cannot time
        the re-timings from the trace (``_kernel_timer_usable``: a
        wall-clock worker, a process group, a build without CUDA activity,
        another tool holding the activity API, and ``autotune_benchmark_fn``,
        under which the re-timings run the user's function in this
        process), inside ``find_handoff`` (its session ranks its shortlist
        by, and returns, the readings of every batch it observed, the
        initial population's among them), once ``autotune_budget_seconds``
        is used up (the search then times no further candidate) or when it
        runs out before the population has moved, and when the trace reads
        none of the population's kernels (``_read_on_settled_timer``).

        Whether the worker can trace is asked last, of a search that would
        move otherwise. The first answer runs a profiling session in the
        worker, as every traced window does, and a session leaves the
        activity API attached to that process, which slows its later
        launches (``kernel_timer_available``). A search that keeps the
        events for another reason asks nothing of the worker and is timed
        as before. One that keeps them after a session (a worker whose
        trace does not work, a trace that reads none of the population, a
        budget that runs out during the move) gets a new worker process
        first (``LocalBenchmarkProvider.restart_profiled_benchmark_worker``):
        its later candidates are then read by CUDA events as its initial
        population was, not above it (by about 3 us on a B200).

        Source aliases do not move the choice. An alias carries its
        kernel's reading and none of its own, so the best reading is a timed
        kernel's; and it runs no job, so whether the worker's process is up
        after the last candidate is not asked
        (``LocalBenchmarkProvider.benchmark_worker_started``): a population
        whose last timed kernel ran into the timeout and whose remaining
        candidates repeat earlier code settles as any other.
        """
        if (
            self.short_kernel_below_us is None
            or self._candidate_timer_settled
            or not math.isfinite(self.best_perf_so_far)
        ):
            return
        self._candidate_timer_settled = True
        provider = self.benchmark_provider
        if (
            self.performance_unit != "ms"
            # (behind a prelude a kernel of any length moves to the trace:
            # an event-timed sample there can hold the launch's host time,
            # ``do_bench``)
            or not (
                self._benchmark_prelude is not None
                or self.best_perf_so_far * 1000.0 < self.short_kernel_below_us
            )
            or _resolve_l2_flush_mode(None) != "call"
            # every candidate would be a graph replay, which is not traced
            or _env_get_bool(_BENCHMARK_CUDAGRAPH_ENV, default=False)
            or not isinstance(provider, LocalBenchmarkProvider)
            or not provider.benchmark_worker_started()
            # a handoff session ranks and returns every reading it observed
            # (``HandoffPoint.measurements``), which none can read again
            or _active_handoff.get() is not None
            # reading the population again is timing work, which the budget
            # covers; a search that is over it times no further candidate
            or self._autotune_budget_exceeded()
        ):
            return
        # Asked last: the first answer runs a profiling session in the
        # worker (``worker_kernel_timer_available``), which leaves the
        # activity API attached there, and a search that keeps the events
        # for a reason above goes on timing in that worker. (Also False
        # under ``autotune_benchmark_fn``: the re-timings would run the
        # user's function in this process, on its own scale.)
        if not self._kernel_timer_usable():
            # the initial population was read before the probe's session,
            # where it ran one
            provider.restart_profiled_benchmark_worker()
            return
        best_us = self.best_perf_so_far * 1000.0
        why = (
            f"under {self.short_kernel_below_us:g} us"
            if best_us < self.short_kernel_below_us
            else "behind a benchmark prelude"
        )
        provider.trace_search_candidates = True
        self.traces_candidates = True
        # reading the population again stops when the budget runs out
        with provider.isolated_timings_within_budget():
            moved = self._read_on_settled_timer(members, visited)
        if not moved:
            provider.trace_search_candidates = False
            self.traces_candidates = False
            # the event readings the search keeps were taken before the
            # sessions of the probe and of the windows just tried
            provider.restart_profiled_benchmark_worker()
            reason = (
                "the autotune budget ran out before the CUDA activity trace "
                "had read every initial config"
                if self._autotune_budget_exceeded()
                else "the CUDA activity trace read none of the initial configs"
            )
            self.log(
                f"The best initial config reads {best_us:.1f} us ({why}), but "
                f"{reason}: timing this search's candidates with CUDA events"
            )
            return
        self.log(
            f"The best initial config reads {best_us:.1f} us ({why}): timing "
            "this search's candidates by their kernels' durations in the CUDA "
            "activity trace"
        )

    def _read_on_settled_timer(
        self,
        members: list[PopulationMember],
        visited: MutableSet[Config] | None = None,
    ) -> bool:
        """Move the initial population ``members`` to the trace timer the
        search is settling on; ``False`` when the trace read none of them
        (or none of them has a reading to move) and when the autotune budget
        ran out before the move was through, so the search stays on the
        events.

        Their readings so far are CUDA-event readings, 4-5 us above what the
        trace reads for the same kernel on a B200. A search re-times only
        the members within its re-timing threshold of the best
        (``should_rebenchmark``), and none at an effort whose threshold is
        below 1, so without this the other members kept their event
        readings: a path started from one compared it with traced
        neighbours, a search without a verification pass (differential
        evolution) compared event-read parents with traced children, and a
        surrogate was fit on labels of both scales. Every member with a
        finite reading is read here, whatever the threshold, one timing per
        kernel (``_rebenchmark_kernels``), a population with a single such
        kernel included (``allow_single``), and a member whose reading fails
        (a timeout, a window whose trace cannot be read:
        ``benchmarking.KernelTraceUnreadable``) is read a second time.

        Under ``autotune_budget_seconds`` the two passes are timing work
        the budget covers, as it covers the candidates'. The caller runs
        them where the provider asks the budget hook before each kernel
        (``LocalBenchmarkProvider.isolated_timings_within_budget``), and
        the second pass is not started once the budget is used up: a
        failed reading can cost a whole benchmark timeout, and so would
        its repeat. When the budget is used up and a member is still
        without a trace reading, the move is undone (``_hold_readings``):
        every member holds its event readings again, the best reading and
        the final selection's history are as they were, and the search
        stays on the events like one whose budget ran out before the move.
        Moving with the members read so far would keep whichever configs
        came first in the population and drop the others, the fastest by
        events possibly among them, and a search over its budget times no
        candidate to make up for them. A search without a budget reads
        every kernel as described above.

        A re-timing that leaves a sticky device error in the worker is a
        failure of the kernel, not a reading of the trace. Its member reads
        infinite from there on (``_resolve_isolated_rebenchmark_results``),
        as after a verification that failed, on whichever timer the search
        goes on with: it is not read a second time, it is no reading that
        a population moves with, and it stays where the search keeps the
        events, also in a move that is undone (``_hold_readings``).

        Where no member has a trace reading after both passes, the trace
        does not read this kernel's windows (calls that launch no kernel,
        say), every later candidate would fail the same way, and the search
        keeps the events and every reading it has. Otherwise this is the one
        place where the search changes timers, and nothing keeps an event
        reading past it (``_move_search_state_to_trace``): a member the
        trace read holds trace readings only, one whose re-timing failed
        holds that failure alone, and one it did not read has no reading in
        this search and leaves the population.
        """
        read = [
            member for member in members if member.perfs and math.isfinite(member.perf)
        ]
        kernels = {id(member): self._kernel_holder(member, read) for member in read}
        events = {
            id(member): len(member.perfs) for member in (*read, *kernels.values())
        }
        # (a re-timing also gives the population's aliases their kernels'
        # readings: ``_rebenchmark_kernels``)
        undo = self._hold_readings((*read, *kernels.values(), *self.population))

        def unread() -> list[PopulationMember]:
            # Asked of the kernel: a source alias is read when the member
            # that holds its kernel has a new reading, which the alias then
            # takes (its own list does not say: it gains nothing where that
            # reading equals the one it holds).
            return [
                member
                for member in read
                if len(kernels[id(member)].perfs) == events[id(kernels[id(member)])]
            ]

        def traced() -> bool:
            # Whether the trace read a kernel: it has a new reading, and
            # that is a duration (a re-timing that failed leaves an infinite
            # one).
            return any(
                len(kernel.perfs) > events[id(kernel)] and math.isfinite(kernel.perf)
                for kernel in kernels.values()
            )

        self._rebenchmark_kernels(
            read, desc="Reading the initial population on the trace", allow_single=True
        )
        if unread() and not self._autotune_budget_exceeded():
            self._rebenchmark_kernels(
                unread(),
                desc="Reading the initial population on the trace again",
                allow_single=True,
            )
        if not traced():
            return False
        if unread() and self._autotune_budget_exceeded():
            undo()
            return False
        self._move_search_state_to_trace(
            read, unread(), kernels, events, members, visited
        )
        return True

    def _reading_registries(self) -> list[MutableMapping[Config, PopulationMember]]:
        """The mappings that hold snapshots of members with their readings:
        the final selection's history, its pins and, where the search has
        one, the refinement registry."""
        registries = [self._benchmarked_members, self._pinned_finalist_members]
        terminal_members = getattr(self, "_terminal_refinement_members", None)
        if terminal_members is not None:
            registries.append(terminal_members)
        return registries

    def _hold_readings(self, members: Iterable[PopulationMember]) -> Callable[[], None]:
        """A function that puts the search's readings back as they are now,
        after re-timings of ``members``: each member's readings, the best
        reading, and the entries of the final selection's history, its pins
        and the refinement registry (``_reading_registries``; an entry is a
        snapshot that a re-timing replaces and never edits, so the items of
        the mappings are all there is to keep).

        A failure is not taken back. A member that one of those re-timings
        left at an infinite reading (its kernel left a sticky device error
        in the worker: ``_resolve_isolated_rebenchmark_results``) holds that
        reading again after the ones it had, and the mappings are refreshed
        for it as the re-timing refreshed them: what a failed re-timing
        leaves in a search that never changed timers. With its earlier
        readings alone the config would rank, and could be returned, as if
        it had not failed."""
        counts = [
            (member, len(member.perfs))
            for member in {id(member): member for member in members}.values()
        ]
        best = self.best_perf_so_far
        entries = [
            (registry, [*registry.items()]) for registry in self._reading_registries()
        ]

        def undo() -> None:
            failed = [
                member
                for member, count in counts
                if len(member.perfs) > count and not math.isfinite(member.perf)
            ]
            for member, count in counts:
                del member.perfs[count:]
            self.best_perf_so_far = best
            for registry, items in entries:
                registry.clear()
                registry.update(items)
            for member in failed:
                member.perfs.append(inf)
            self._refresh_benchmarked_members_after_rebenchmark(failed)

        return undo

    def _move_search_state_to_trace(
        self,
        read: Sequence[PopulationMember],
        unread: Sequence[PopulationMember],
        kernels: Mapping[int, PopulationMember],
        events: Mapping[int, int],
        members: list[PopulationMember],
        visited: MutableSet[Config] | None,
    ) -> None:
        """Leave no CUDA-event reading in the search's state, after the
        members ``read`` of the initial population ``members`` were read on
        the trace (``_read_on_settled_timer``; ``kernels[id(member)]`` is
        the member that holds a member's kernel, ``events[id(member)]`` the
        number of event readings a member holds, ``unread`` the members
        whose kernel the trace read in neither pass).

        - A member the trace read keeps its trace readings only, so
          whatever ranks it from here on (a comparison, the final
          selection's low-water reading, the spread of re-readings) sees
          one timer.  A source alias of a kernel the trace read holds that
          kernel's trace reading, whatever it held before.  A member whose
          re-timing failed is among them with that failure, an infinite
          reading, alone: it stays a member, ranks last and takes no entry
          in the history.
        - An unread member has no reading in this search. With its event
          reading, 4-5 us above the readings it would be compared with, a
          kernel that is in fact the fastest would lose to slower traced
          ones, and nothing would read it again (it is outside every
          re-timing threshold). It leaves ``members`` and ``visited``, as a
          config the initial population never attempted does
          (``_benchmark_initial_batch``): it starts no path, is no parent,
          no surrogate row and no finalist, and the search may come back to
          its config, which is then timed on the trace like any candidate.
        - The final selection's history, its pins and the refinement
          registry were filled, and the history was trimmed to its top k,
          by event readings: the entries of every member that had one go,
          and the members the trace read are recorded again, so the trim
          ranks them by their trace readings.
        - ``best_perf_so_far``, which the re-timing threshold is taken
          from, is the best trace reading.
        - The provider's record of each kernel's reading, which settles
          later configs with the same generated source, takes the trace
          reading (``LocalBenchmarkProvider.refresh_source_reading``) or,
          for an unread kernel, is dropped (``forget_source_reading``), so
          the next config with that code is timed.
        - What the algorithm keeps besides its members goes
          (``_discard_event_readings``).
        """
        unread_ids = {id(member) for member in unread}
        traced = [member for member in read if id(member) not in unread_ids]
        for member in traced:
            kernel = kernels[id(member)]
            if kernel is member:
                del member.perfs[: events[id(member)]]
                member.timer_readings_from = 0
            else:
                # a copy of its kernel's trace reading, none of its own
                member.perfs[:] = [kernel.perf]
                member.timer_readings_from = 1
        members[:] = [member for member in members if id(member) not in unread_ids]
        registries = self._reading_registries()
        for member in read:
            for registry in registries:
                registry.pop(member.config, None)
        for member in unread:
            if visited is not None:
                visited.discard(member.config)
            if self._kernels.get(id(member.fn)) is member:
                del self._kernels[id(member.fn)]
        for member in traced:
            if not self._is_source_alias(member):
                self._record_benchmarked_member(member)
        self.best_perf_so_far = min((member.perf for member in traced), default=inf)
        provider = self.benchmark_provider
        if isinstance(provider, LocalBenchmarkProvider):
            for member in traced:
                provider.refresh_source_reading(member.fn, member.perf)
            for member in unread:
                provider.forget_source_reading(member.fn)
        if unread:
            self.log.warning(
                f"The CUDA activity trace read {len(unread)} of the initial "
                "configs in neither of two passes: this search reads its "
                "candidates from the trace, so they leave its initial "
                f"population (the first: {unread[0].config})"
            )
        self._discard_event_readings()

    def _discard_event_readings(self) -> None:
        """Called once, when the search has moved its members' readings to
        the trace timer (``_read_on_settled_timer``): an algorithm drops the
        readings it holds outside its members, which were taken with CUDA
        events and cannot be read again."""

    def _apply_effective_source_repairs(
        self,
        repairs: MutableMapping[Config, BenchmarkResult],
        current_members: Sequence[PopulationMember],
    ) -> None:
        """Repair failed members after a later config proves the same source.

        A repaired member is that config's kernel under its own config: a
        source alias (``"deduplicated"``).  Where the search drops aliases
        it takes no entry in the final selection's history, which holds the
        kernel under the config that was timed, and a pinned one passes its
        pin to that config, as an alias of the batch does
        (``_settle_benchmarked_members``).
        """
        candidates = [
            *current_members,
            *self.population,
            *self._compiler_seed_members,
            *self._benchmarked_members.values(),
            *self._pinned_finalist_members.values(),
            *(getattr(self, "_terminal_refinement_members", None) or {}).values(),
            *self._members_kept_by_algorithm(),
        ]
        seen: set[int] = set()
        repaired: list[PopulationMember] = []
        for member in candidates:
            if id(member) in seen:
                continue
            seen.add(id(member))
            repair = repairs.get(member.config)
            if repair is None or (member.perfs and math.isfinite(member.perf)):
                continue
            if member.perfs:
                member.perfs = [
                    repair.perf if not math.isfinite(perf) else perf
                    for perf in member.perfs
                ]
            else:
                member.perfs.append(repair.perf)
            member.fn = repair.fn
            member.status = repair.status
            member.compile_time = None
            repaired.append(member)
        self._settle_benchmarked_members(repaired, current_members)

    def _members_kept_by_algorithm(self) -> Iterable[PopulationMember]:
        """Members an algorithm still reads after they left the population,
        which a source repair therefore reaches as well
        (``_apply_effective_source_repairs``).  None by default."""
        return ()

    def _record_benchmarked_member(self, member: PopulationMember) -> None:
        """Keep successful benchmarked members available for final verification."""
        terminal_members = getattr(self, "_terminal_refinement_members", None)
        if terminal_members is not None:
            self._record_terminal_refinement_member(terminal_members, member)
        if not member.perfs or not math.isfinite(member.perf):
            return
        if member.config in self._pinned_finalist_configs:
            self._record_best_member_for_config(
                self._pinned_finalist_members, member.config, member
            )
        top_k = self._final_rebenchmark_top_k()
        if top_k <= 1:
            return
        self._record_best_member_for_config(
            self._benchmarked_members, member.config, member
        )
        self._prune_benchmarked_members(top_k)

    def _record_terminal_refinement_member(
        self,
        target: MutableMapping[Config, PopulationMember],
        member: PopulationMember,
    ) -> None:
        """Refresh terminal history without replacing a reusable result by failure."""
        existing = target.get(member.config)

        def reusable(candidate: PopulationMember | None) -> bool:
            return bool(
                candidate is not None
                and candidate.status in {"ok", "deduplicated"}
                and candidate.perfs
                and math.isfinite(candidate.perf)
            )

        if reusable(existing) and not reusable(member):
            return
        self._record_best_member_for_config(
            target,
            member.config,
            member,
            replace=True,
        )

    def _record_best_member_for_config(
        self,
        target: MutableMapping[Config, PopulationMember],
        config: Config,
        member: PopulationMember,
        *,
        replace: bool = False,
        replace_ties: bool = False,
    ) -> None:
        existing = target.get(config)
        member_perf = self._finalist_history_perf(member)
        existing_perf = (
            inf if existing is None else self._finalist_history_perf(existing)
        )
        if (
            replace
            or existing is None
            or member_perf < existing_perf
            or (replace_ties and member_perf == existing_perf)
        ):
            # Own the built-in containers the search mutates, while preserving
            # opaque selections' identity. Match ConfigCache snapshots without
            # invoking arbitrary copy hooks or changing the compiled selection.
            # Performance history remains independent of later rebenchmarks.
            snapshot = _stored_config(config.config)
            target[snapshot] = dataclasses.replace(
                member,
                perfs=copy.deepcopy(member.perfs),
                flat_values=cast(
                    "FlatConfig", _copy_config_structure(member.flat_values)
                ),
                config=snapshot,
            )

    def _uses_verified_finalist_history(self) -> bool:
        """Whether finalist retention should follow the latest verified timing."""
        config_spec = getattr(self, "config_spec", None)
        return bool(
            config_spec is not None
            and getattr(config_spec, "cute_flash_search_enabled", False)
        )

    def _finalist_history_perf(self, member: PopulationMember) -> float:
        if self._uses_verified_finalist_history():
            if member.perfs and math.isfinite(member.perf):
                return member.perf
            return inf
        return self._member_low_water_perf(member)

    def _refresh_benchmarked_members_after_rebenchmark(
        self, members: Sequence[PopulationMember]
    ) -> None:
        """Refresh finalist history after its source members are rebenchmarked.

        Long CuTe kernels can have optimistic first-call timings. Candidates are
        initially recorded before the normal higher-effort rebenchmark, so the
        first sample must not permanently decide which configs survive in the
        bounded final-verification history. Other backends preserve their prior
        low-water ranking and do not reinsert configs that were already pruned;
        they still need an explicit refresh now that snapshots own their perf lists.
        A failed duplicate retains the last successful snapshot for final checking,
        unless source quarantine has invalidated that snapshot too.
        """
        terminal_members = getattr(self, "_terminal_refinement_members", None)
        if terminal_members is not None:
            for member in members:
                self._record_terminal_refinement_member(terminal_members, member)

        benchmarked_members = getattr(self, "_benchmarked_members", None)
        pinned_configs = getattr(self, "_pinned_finalist_configs", None)
        pinned_members = getattr(self, "_pinned_finalist_members", None)
        if (
            benchmarked_members is None
            or pinned_configs is None
            or pinned_members is None
        ):
            return

        use_verified_history = self._uses_verified_finalist_history()
        top_k = self._final_rebenchmark_top_k()
        refreshed_configs: MutableMapping[Config, None] = ConfigCache()
        finite_members: MutableMapping[Config, PopulationMember] = ConfigCache()
        for member in members:
            if not member.perfs:
                continue
            refreshed_configs.setdefault(member.config, None)
            history_perf = self._finalist_history_perf(member)
            if not math.isfinite(history_perf):
                continue
            existing = finite_members.get(member.config)
            if existing is None or history_perf < self._finalist_history_perf(existing):
                finite_members[member.config] = member

        for config in refreshed_configs:
            member = finite_members.get(config)
            if member is None and use_verified_history and terminal_members is not None:
                # The higher-effort rebenchmark invalidated every fresh sample
                # for this config; drop any older ok snapshot so terminal
                # refinement re-measures the config instead of resurrecting a
                # quarantined result.
                terminal_members.pop(config, None)
            if (
                config in pinned_configs
                and member is not None
                # A pinned config that is a source alias is its kernel under
                # another config, and the pin is with the config that holds
                # the kernel (``_settle_benchmarked_members``): an alias
                # that is timed itself (it stands in for a kernel the search
                # timed under no config of its own) takes no entry here.
                and not self._is_source_alias(member)
            ):
                self._record_best_member_for_config(
                    pinned_members,
                    config,
                    member,
                    replace=use_verified_history,
                    replace_ties=not use_verified_history,
                )
            elif config in pinned_configs and use_verified_history:
                existing = pinned_members.get(config)
                if existing is not None and not math.isfinite(
                    self._finalist_history_perf(existing)
                ):
                    pinned_members.pop(config, None)
            if top_k <= 1 or (
                not use_verified_history and config not in benchmarked_members
            ):
                continue
            if member is not None:
                # CuTe flash reinserts configs pruned by their initial sample.
                # Other backends only refresh existing snapshots above.
                self._record_best_member_for_config(
                    benchmarked_members,
                    config,
                    member,
                    replace=use_verified_history,
                    replace_ties=not use_verified_history,
                )
            elif use_verified_history:
                existing = benchmarked_members.get(config)
                if existing is not None and not math.isfinite(
                    self._finalist_history_perf(existing)
                ):
                    benchmarked_members.pop(config, None)
        if use_verified_history and top_k > 1:
            self._prune_benchmarked_members(top_k)

    def _prune_benchmarked_members(self, top_k: int) -> None:
        if len(self._benchmarked_members) <= top_k:
            return
        for config, _member in sorted(
            self._benchmarked_members.items(),
            key=lambda item: self._finalist_history_perf(item[1]),
        )[top_k:]:
            del self._benchmarked_members[config]

    def pin_finalist_config(self, config: Config) -> None:
        """Always include a seed/default config in final verification if benchmarked."""
        # Snapshot: configs are mutated in place after being added (see
        # _record_best_member_for_config), which would corrupt this set.
        self._pinned_finalist_configs.add(config)

    def pin_finalist_configs(self, configs: Sequence[Config]) -> None:
        for config in configs:
            self.pin_finalist_config(config)

    def _final_rebenchmark_backend(self) -> Backend | None:
        """The backend whose finalist policy applies, or ``None`` for a search
        built without a config spec (the unit tests' bare searches)."""
        from .._compiler.backend_registry import get_backend

        backend_name = getattr(getattr(self, "config_spec", None), "backend_name", None)
        if backend_name is None:
            return None
        return get_backend(backend_name)

    def _final_rebenchmark_top_k_default(self) -> int:
        backend = self._final_rebenchmark_backend()
        top_k = None if backend is None else backend.final_rebenchmark_top_k()
        return _FINAL_REBENCHMARK_TOP_K_DEFAULT if top_k is None else top_k

    def _final_rebenchmark_top_k(self) -> int:
        default = self._final_rebenchmark_top_k_default()
        raw = os.getenv(_FINAL_REBENCHMARK_TOP_K_ENV)
        if raw is None:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            self.log.warning(
                f"Ignoring invalid {_FINAL_REBENCHMARK_TOP_K_ENV}={raw!r}; "
                f"using {default}."
            )
            return default

    def _final_rebenchmark_target_ms(self) -> float:
        raw = os.getenv(_FINAL_REBENCHMARK_TARGET_MS_ENV)
        if raw is None:
            return _FINAL_REBENCHMARK_TARGET_MS_DEFAULT
        try:
            target_ms = float(raw)
        except ValueError:
            self.log.warning(
                f"Ignoring invalid {_FINAL_REBENCHMARK_TARGET_MS_ENV}={raw!r}; "
                f"using {_FINAL_REBENCHMARK_TARGET_MS_DEFAULT}."
            )
            return _FINAL_REBENCHMARK_TARGET_MS_DEFAULT
        if not math.isfinite(target_ms):
            self.log.warning(
                f"Ignoring non-finite {_FINAL_REBENCHMARK_TARGET_MS_ENV}={raw!r}; "
                f"using {_FINAL_REBENCHMARK_TARGET_MS_DEFAULT}."
            )
            return _FINAL_REBENCHMARK_TARGET_MS_DEFAULT
        return min(
            _FINAL_REBENCHMARK_TARGET_MS_MAX,
            max(_REBENCHMARK_TARGET_MS_DEFAULT, target_ms),
        )

    def _final_rebenchmark_use_isolated(self) -> bool:
        if self._input_bank_count > 1 or self._benchmark_prelude is not None:
            return True

        # The backend decides whether finalists are timed ISOLATED (each in
        # its own steady window) or interleaved by default
        # (``Backend.final_rebenchmark_isolated``).
        backend = self._final_rebenchmark_backend()
        default = backend is not None and backend.final_rebenchmark_isolated()
        try:
            return _env_get_bool(_FINAL_REBENCHMARK_ISOLATED_ENV, default)
        except ValueError:
            self.log.warning(
                f"Ignoring invalid {_FINAL_REBENCHMARK_ISOLATED_ENV}="
                f"{os.getenv(_FINAL_REBENCHMARK_ISOLATED_ENV)!r}; using {default}."
            )
            return default

    def _final_rebenchmark_pinned_tolerance(self) -> float:
        return self._tolerance_from_env(
            _FINAL_REBENCHMARK_PINNED_TOLERANCE_ENV,
            _FINAL_REBENCHMARK_PINNED_TOLERANCE_DEFAULT,
        )

    def _final_selection_preference_tolerance(self) -> float:
        return self._tolerance_from_env(
            _FINAL_SELECTION_PREFERENCE_TOLERANCE_ENV,
            _FINAL_SELECTION_PREFERENCE_TOLERANCE_DEFAULT,
        )

    def _tolerance_from_env(self, name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            tolerance = float(raw)
        except ValueError:
            self.log.warning(f"Ignoring invalid {name}={raw!r}; using {default}.")
            return default
        if not math.isfinite(tolerance) or tolerance < 0:
            self.log.warning(f"Ignoring invalid {name}={raw!r}; using {default}.")
            return default
        return tolerance

    def _final_rebenchmark_max_spread(self) -> float:
        raw = os.getenv(_FINAL_REBENCHMARK_MAX_SPREAD_ENV)
        if raw is None:
            return _FINAL_REBENCHMARK_MAX_SPREAD_DEFAULT
        try:
            spread = float(raw)
        except ValueError:
            spread = math.nan
        if math.isnan(spread) or spread < 1.0:
            self.log.warning(
                f"Ignoring invalid {_FINAL_REBENCHMARK_MAX_SPREAD_ENV}={raw!r} "
                f"(a factor >= 1, or inf); using "
                f"{_FINAL_REBENCHMARK_MAX_SPREAD_DEFAULT}."
            )
            return _FINAL_REBENCHMARK_MAX_SPREAD_DEFAULT
        return spread

    def _within_final_rebenchmark_spread(
        self,
        finalists: list[PopulationMember],
        keep: Sequence[PopulationMember],
    ) -> list[PopulationMember]:
        """The finalists the search measured within ``max_spread`` of its
        best shortlist reading, plus ``keep`` (pinned seeds, the incumbent).

        The single-pass verification exists to correct a noisy reading of a
        near-tied candidate, not to re-time configs the search already found
        far slower; each dropped finalist saves one isolated steady window.
        """
        spread = self._final_rebenchmark_max_spread()
        if not math.isfinite(spread):
            return finalists
        best = min(self._finalist_history_perf(member) for member in finalists)
        kept_ids = {id(member) for member in keep}
        kept = [
            member
            for member in finalists
            if id(member) in kept_ids
            or self._finalist_history_perf(member) <= best * spread
        ]
        if len(kept) < len(finalists):
            self.log(
                f"Final verification shortlist: {len(kept)} of {len(finalists)} "
                f"configs measured within {spread:g}x of the best search reading "
                f"({self.format_performance(best)}); the rest are not re-timed."
            )
        return kept

    def _prefer_backend_preference_within_tolerance(
        self, after: PopulationMember, live: Sequence[PopulationMember]
    ) -> PopulationMember:
        """Among the live finalists within the preference tolerance of the
        fastest one, the config the backend ranks first wins
        (``Backend.finalist_preference``: lower is better, the fastest of
        equal rank keeps its place).

        The isolated timing of a kernel cannot see everything a config
        costs or saves in the sequence it will run in: a CuTe kernel whose
        back-to-back calls are separated by a plain launch (a host zero fill)
        reads the same with and without dependent launch, while its
        successor in the real call sequence launches 0.3-1 us earlier when
        the kernel releases its dependents.  Finalists that tie within the
        steady windows' resolution are therefore ranked by the backend's
        preference instead of by noise.

        The backend is reached through its registered name in the kernel's
        settings (``Settings.backend`` is validated against the registry):
        the hook is a class method, so no backend instance is needed, and a
        search built around a stand-in spec ranks by the real backend's rule.
        The hook reads a finalist's lowering off its rendered module, which
        it asks for on demand (``BoundKernel.to_code``, once per finalist).
        """
        from .._compiler.backend_registry import get_backend_class

        hook = get_backend_class(self.settings.backend).finalist_preference
        sources: MutableMapping[Config, str] = ConfigCache()
        ranks: MutableMapping[Config, int] = ConfigCache()

        def preference(member: PopulationMember) -> int:
            config = member.config
            if config not in ranks:

                def source() -> str:
                    if config not in sources:
                        sources[config] = self.kernel.to_code(config) or ""
                    return sources[config]

                ranks[config] = hook(config, source)
            return ranks[config]

        best_rank = preference(after)
        if best_rank == 0:
            return after
        tolerance = self._final_selection_preference_tolerance()
        band = [
            member
            for member in live
            if member is not after and member.perf <= after.perf * (1.0 + tolerance)
        ]
        preferred = [member for member in band if preference(member) < best_rank]
        if not preferred:
            return after
        chosen = min(preferred, key=lambda m: (preference(m), m.perf))
        self.log(
            "Final selection preferred a config the backend ranks ahead within "
            f"{tolerance:.2%} of the fastest finalist: "
            f"{self.format_performance(chosen.perf)} vs "
            f"{self.format_performance(after.perf)}"
        )
        return chosen

    @staticmethod
    def _repeat_for_target_ms(target_ms: float, best_perf_so_far: float) -> int:
        if math.isfinite(best_perf_so_far) and best_perf_so_far > 0:
            base_repeat_float = target_ms / best_perf_so_far
            base_repeat = (
                _REBENCHMARK_INTERLEAVED_REPEAT_MAX
                if not math.isfinite(base_repeat_float)
                else int(base_repeat_float)
            )
        else:
            base_repeat = 1000
        return min(_REBENCHMARK_INTERLEAVED_REPEAT_MAX, max(3, base_repeat))

    @staticmethod
    def _repeat_reference_perf(members: list[PopulationMember]) -> float:
        finite = [
            member.perf
            for member in members
            if member.perfs and math.isfinite(member.perf) and member.perf > 0
        ]
        return max(finite, default=inf)

    @staticmethod
    def _isolated_rep_ms(target_ms: float, benchmark_timeout_s: int) -> int:
        timeout_budget_ms = max(1, int(benchmark_timeout_s * 500))
        rep_ms = min(int(target_ms), timeout_budget_ms)
        if (capstr := os.getenv("HELION_CAP_REBENCHMARK_REPEAT")) is not None:
            rep_ms = min(rep_ms, int(capstr))
        return max(1, rep_ms)

    @staticmethod
    def _steady_rebenchmark_rep_ms(target_ms: float) -> int:
        rep_ms = int(target_ms)
        if (capstr := os.getenv("HELION_CAP_REBENCHMARK_REPEAT")) is not None:
            rep_ms = min(rep_ms, int(capstr))
        return max(1, rep_ms)

    @staticmethod
    def _member_low_water_perf(member: PopulationMember) -> float:
        finite = [perf for perf in member.perfs if math.isfinite(perf)]
        return min(finite, default=inf)

    def final_rebenchmark_best(self, best: PopulationMember) -> PopulationMember:
        """Recheck the best seen configs before returning an autotuned result.

        Generation searches can end on a noisy late measurement, especially when a
        budget expires mid-generation. Keep a small history of every compiled
        member and perform one apples-to-apples final verification over the best
        unique configs observed during the run.
        """
        by_config: MutableMapping[Config, PopulationMember] = ConfigCache()
        candidates = [
            *self._pinned_finalist_members.values(),
            *self._benchmarked_members.values(),
            *getattr(self, "population", ()),
            best,
        ]
        # One entry per kernel. A source alias that is still a member (an
        # initial population keeps its own, as do the populations of the
        # differential-evolution and LLM-guided searches) is a kernel that
        # the config holding it stands for. The search's best stands for its
        # kernel under its own config: every other config of that kernel (a
        # member with the same compiled function: the config that holds the
        # kernel where the best is an alias, or the config a finishing phase
        # started from when it reset settings the code does not depend on)
        # stands behind it, so a win returns the config the search ended on.
        # A pin among them passes to the best's config: the kernel keeps the
        # rules of a pinned finalist under whichever config stands for it.
        behind = [
            member
            for member in candidates
            if member is not best and member.fn is best.fn
        ]
        if behind and self._collapses_source_aliases():
            if any(member.config in self._pinned_finalist_configs for member in behind):
                self.pin_finalist_config(best.config)
            candidates = [member for member in candidates if member.fn is not best.fn]
            candidates.append(best)
        candidates = self._one_member_per_kernel(candidates)
        for member in candidates:
            if not member.perfs or not math.isfinite(
                self._finalist_history_perf(member)
            ):
                continue
            existing = by_config.get(member.config)
            if existing is None or self._finalist_history_perf(
                member
            ) < self._finalist_history_perf(existing):
                by_config[member.config] = member

        live_candidates = [
            member for member in candidates if math.isfinite(member.perf)
        ]
        if math.isfinite(best.perf):
            fallback = best
        elif live_candidates:
            fallback = min(live_candidates, key=performance)
        else:
            raise exc.NoConfigFound

        use_final_selection = self._final_selection_enabled()
        # The single-pass verification is skipped once a wall-clock budget is
        # spent; the final selection instead reserved its share of the budget
        # during the search (``_budget_reserve_seconds``) and sizes itself to
        # what is left (``_final_selection_plan``).
        if not use_final_selection and self._autotune_budget_exceeded_across_ranks():
            return fallback
        top_k = self._final_rebenchmark_top_k()
        if top_k <= 1:
            return fallback

        pinned = [
            member
            for config, member in by_config.items()
            if config in self._pinned_finalist_configs
        ]
        pinned_configs = ConfigSet(member.config for member in pinned)
        remaining = [
            member
            for member in by_config.values()
            if member.config not in pinned_configs
        ]
        ranked = sorted(remaining, key=self._finalist_history_perf)
        finalists = [*pinned, *ranked[:top_k]]
        # The incumbent defends its title even when later phases re-timed it
        # above the shortlist's search numbers.
        incumbent = by_config.get(best.config)
        if incumbent is not None and incumbent not in finalists:
            finalists.append(incumbent)
        if use_final_selection:
            # The family quota: the best config of a body the search's
            # readings rank below the cut still gets its screen window (and
            # its rounds when it survives the screen), see
            # ``FinalSelectionConfig.family_finalists``. The incumbent's
            # family is represented already.
            assert self.final_selection is not None
            finalists.extend(
                self._family_representatives(
                    "shortlist",
                    [member.config for member in finalists],
                    [
                        (member, member.config, self._finalist_history_perf(member))
                        for member in ranked[top_k:]
                        if member is not incumbent
                    ],
                    best=min(
                        (
                            self._finalist_history_perf(member)
                            for member in by_config.values()
                        ),
                        default=inf,
                    ),
                    policy=self.final_selection,
                )
            )
        if len(finalists) < 2:
            live_finalists = [
                member for member in finalists if math.isfinite(member.perf)
            ]
            return min(live_finalists, key=performance, default=fallback)

        before = min(finalists, key=performance)
        if use_final_selection:
            self._warn_ignored_final_rebenchmark_envs()
            plan = self._final_selection_plan(len(finalists))
            if plan is None:
                return fallback
            after = self._run_final_selection(
                finalists,
                pinned_configs,
                before if incumbent is None else incumbent,
                plan,
            )
        else:
            # Finalists have already survived the normal benchmark path. Time
            # the last shortlist with the paired event-based interleaved bench
            # so clock/thermal drift between per-candidate timing windows
            # cannot mis-rank near-tied finalists (unpaired sequential steady
            # windows measured 1-3% apart on identical configs). If a user
            # explicitly opts into isolated finalist timing, keep suspicious
            # confirmation enabled for the in-process fallback.
            use_isolated = self._final_rebenchmark_use_isolated()
            finalists = self._within_final_rebenchmark_spread(
                finalists,
                [*pinned, *([incumbent] if incumbent is not None else [])],
            )
            if len(finalists) < 2:
                live_finalists = [
                    member for member in finalists if math.isfinite(member.perf)
                ]
                return min(live_finalists, key=performance, default=fallback)
            self.rebenchmark(
                finalists,
                desc=f"Final verification top {len(finalists)} configs",
                target_ms=self._final_rebenchmark_target_ms(),
                use_isolated=use_isolated,
                confirm_suspicious=use_isolated,
                use_interleaved=not use_isolated,
                candidate_private_args=use_isolated,
            )
            live_finalists = [
                member for member in finalists if math.isfinite(member.perf)
            ]
            if not live_finalists:
                raise exc.NoConfigFound
            after = self._prefer_pinned_within_tolerance(
                self._prefer_backend_preference_within_tolerance(
                    min(live_finalists, key=performance), live_finalists
                ),
                live_finalists,
                pinned_configs,
            )
        if _config_key(after.config) != _config_key(before.config):
            self.log(
                "Final verification selected a different config: "
                f"{self.format_performance(before.perf)} -> "
                f"{self.format_performance(after.perf)}"
            )
        return after

    def _family_representatives(
        self,
        stage: str,
        represented: Iterable[Config],
        candidates: Sequence[tuple[_T, Config, float]],
        *,
        best: float,
        policy: FinalSelectionConfig,
    ) -> list[_T]:
        """The family quota of the final selection at one of its two cuts.

        ``candidates`` are the configs the cut dropped, best first, each with
        the reading the cut ranked by; ``represented`` the configs it kept.
        Returns the first candidate of every family (the values of the
        spec's ``final_selection_family_keys``) no kept config belongs to,
        closest to ``best`` first, at most ``policy.family_finalists`` of
        them, each reading within ``policy.family_spread`` of ``best``.
        """
        keys = self._final_selection_family_keys()
        quota = policy.family_finalists
        if not keys or quota <= 0 or not candidates or not math.isfinite(best):
            return []
        families = {
            _config_value_key(_config_family(config, keys)) for config in represented
        }
        admitted: list[_T] = []
        notes: list[str] = []
        for item, config, reading in candidates:
            family = _config_family(config, keys)
            family_key = _config_value_key(family)
            if family_key in families:
                continue
            if not math.isfinite(reading) or reading > best * policy.family_spread:
                # the candidates are ranked, so no later one is closer
                break
            families.add(family_key)
            admitted.append(item)
            notes.append(
                f"{_family_label(family)} at {self.format_performance(reading)} "
                f"({reading / best - 1.0:+.1%})"
            )
            if len(admitted) >= quota:
                break
        if admitted:
            self.log(
                f"Final selection family quota ({stage}): kept the best config of "
                f"{len(admitted)} more "
                f"{'family' if len(admitted) == 1 else 'families'} "
                f"({'/'.join(keys)}) within {policy.family_spread:g}x of the best "
                f"{self.format_performance(best)}: " + "; ".join(notes)
            )
        return admitted

    def _prefer_pinned_within_tolerance(
        self,
        after: PopulationMember,
        live: Sequence[PopulationMember],
        pinned_configs: MutableSet[Config],
        key: Callable[[PopulationMember], float] = performance,
    ) -> PopulationMember:
        """A pinned seed within the tolerance of the best finalist wins the
        tie; ``key`` reads the timing compared (the member's perf)."""
        live_pinned = [member for member in live if member.config in pinned_configs]
        if not live_pinned:
            return after
        pinned_after = min(live_pinned, key=key)
        tolerance = self._final_rebenchmark_pinned_tolerance()
        if key(pinned_after) <= key(after) * (1.0 + tolerance):
            return pinned_after
        return after

    def _steady_bench(self) -> Callable[..., float | tuple[float, ...]]:
        """The backend's steady-window timer, or the generic ``do_bench``."""
        return self.config_spec.backend.get_do_bench() or do_bench

    def _private_benchmark_args(self, count: int) -> _PrivateBenchmarkArgs:
        """Private clones of the kernel arguments for ``count`` candidates.

        Raises ``torch.OutOfMemoryError`` when they do not fit on the device.
        """
        process_group_name = self.kernel.env.process_group_name
        clones = [
            _clone_args(self.args, process_group_name, idx_to_clone=None)
            for _ in range(count + 1)
        ]
        return _PrivateBenchmarkArgs(clones[0], clones[1:])

    def _shared_benchmark_args(self) -> Sequence[object]:
        """The kernel arguments every candidate of a re-timing shares: a clone
        of the mutated arguments when the kernel has any, else the originals."""
        mutated = self.benchmark_provider.mutated_arg_indices
        if len(mutated) == 0:
            return self.args
        return _clone_args(
            self.args, self.kernel.env.process_group_name, idx_to_clone=mutated
        )

    def _final_selection_in_worker(self) -> bool:
        """Whether the final selection times its finalists in the benchmark
        worker (``_final_selection_measurer``) rather than in this process."""
        provider = self.benchmark_provider
        return (
            isinstance(provider, LocalBenchmarkProvider)
            and provider.steady_window_benchmark_available()
        )

    def _kernel_timer_usable(self) -> bool:
        """Whether the final selection can time ``kernel`` windows: the
        benchmark worker times the finalists, the build and the process group
        admit the trace (``kernel_timer_available``), and the worker's trace
        works (``LocalBenchmarkProvider.worker_kernel_timer_available``).
        Profiling sessions stay out of this process: one leaves the activity
        API attached, slowing every later launch."""
        provider = self.benchmark_provider
        return (
            self._final_selection_in_worker()
            and isinstance(provider, LocalBenchmarkProvider)
            # a wall-clock worker always flushes and ignores the regime
            and not provider._subprocess_benchmark_uses_wall_clock()
            and kernel_timer_available(self.kernel.env.process_group_name)
            and provider.worker_kernel_timer_available()
        )

    def _traceable(self, regime: str | None) -> str | None:
        """``regime``, with ``kernel`` (from the backend or the environment)
        replaced by ``call`` where the selection cannot time it
        (``_kernel_timer_usable``): a selection labelled ``kernel`` must not
        time with CUDA events unseen. Said once per process."""
        global _kernel_timer_fallback_warned
        if regime != "kernel" or self._kernel_timer_usable():
            return regime
        if not _kernel_timer_fallback_warned:
            _kernel_timer_fallback_warned = True
            message = (
                "The final selection times per-call flushed with CUDA events: "
                "the kernel trace timer runs in the benchmark worker only (not "
                "a wall-clock one), on an NVIDIA build that collects CUDA "
                "activity, outside a process group, where the worker's trace "
                "works"
            )
            # loud only where ``kernel`` was asked for, not the backend default
            # (a plain compare: ``_l2_flush_mode_from_env`` raises on the
            # target regime's ``auto``)
            if any(
                os.environ.get(name, "").strip().lower() == "kernel"
                for name in (_L2_FLUSH_ENV, _FINAL_L2_FLUSH_ENV, _TARGET_REGIME_ENV)
            ):
                self.log.warning(message)
            else:
                self.log.debug(message)
        return "call"

    def _backend_final_regime(self) -> str | None:
        """The backend's final-selection regime
        (``Backend.default_final_selection_regime``)."""
        return self.config_spec.backend.default_final_selection_regime()

    def _final_selection_second_regime(self, regime: str) -> str | None:
        """``final_selection_second_regime`` where this process can time it,
        ``None`` when that falls back to ``regime`` itself (a ``kernel``
        target timed as ``call`` beside a ``call`` selection)."""
        second = self._traceable(
            final_selection_second_regime(regime, self._backend_final_regime())
        )
        return None if second == regime else second

    def _final_selection_regime(self) -> str:
        """The regime the final selection times its finalists in, with the
        backend's default (``final_selection_l2_flush_mode``)."""
        regime = self._traceable(
            final_selection_l2_flush_mode(self._backend_final_regime())
        )
        assert regime is not None
        return regime

    def _final_selection_measurer(
        self, finalists: Sequence[PopulationMember], policy: FinalSelectionConfig
    ) -> _FinalSelectionMeasurer:
        """Steady-window timer for the finalists: the killable benchmark worker
        when the search benchmarks in one, otherwise in-process. Both time in
        the final selection's L2 regime (``_final_selection_regime``)."""
        provider = self.benchmark_provider
        l2_flush_mode = self._final_selection_regime()
        window_s = (policy.warmup_ms + policy.rep_ms) / 1000.0
        thermal_s = (policy.thermal_warmup_ms + policy.rep_ms) / 1000.0
        # The timer sizes its repetitions from an estimate pass, so a kernel
        # that runs slower in the window than in the estimate stretches the
        # window; three times the longest window still finishes, and 15 s
        # covers a worker spawn with its wrapper and argument loads.
        timeout_s = max(
            float(self.settings.autotune_benchmark_timeout),
            3.0 * max(window_s, thermal_s) + 15.0,
        )
        if isinstance(provider, LocalBenchmarkProvider):
            private_copies = provider.private_argument_copies_affordable(len(finalists))
            if self._final_selection_in_worker():
                provider.register_isolated_candidates(
                    [member.fn for member in finalists],
                    [member.config for member in finalists],
                )
                return _WorkerFinalSelectionMeasurer(
                    provider,
                    [member.fn for member in finalists],
                    private_copies=private_copies,
                    timeout_s=timeout_s,
                    l2_flush_mode=l2_flush_mode,
                )
        else:
            private_copies = False
        if self._input_bank_count > 1:
            raise exc.InvalidAPIUsage(
                "rotating input banks require isolated final selection"
            )
        if self._benchmark_prelude is not None:
            raise exc.InvalidAPIUsage(
                "a benchmark prelude requires the final selection's windows in "
                "the benchmark worker"
            )
        args: _PrivateBenchmarkArgs | Sequence[object] | None = None
        if private_copies and not dist.is_initialized():
            try:
                args = self._private_benchmark_args(len(finalists))
            except torch.OutOfMemoryError:
                self.log.warning(
                    "Final selection could not allocate private argument copies; "
                    "finalists share one argument set."
                )
        if args is None:
            args = self._shared_benchmark_args()
        return _InProcessFinalSelectionMeasurer(
            [member.fn for member in finalists],
            args,
            self._steady_bench(),
            process_group_name=self.kernel.env.process_group_name,
            log=self.log,
            l2_flush_mode=l2_flush_mode,
        )

    @_retry_without_private_copies
    def _run_final_selection(
        self,
        finalists: list[PopulationMember],
        pinned_configs: MutableSet[Config],
        incumbent: PopulationMember,
        policy: FinalSelectionConfig,
    ) -> PopulationMember:
        """Pick the returned config with interleaved harness-grade windows.

        The shortlist is screened once in the sustained clock regime (when it
        is long enough to be worth it), the best ``top_k`` survivors plus the
        best pinned seed and the ``incumbent`` then run ``rounds`` rounds of
        steady windows whose order alternates every round, and the median
        across rounds decides (see ``FinalSelectionConfig``). When the
        selection's L2 regime calls for a second one
        (``final_selection_second_regime``), the finalists tied within
        ``policy.tie_tolerance`` of the best median (every complete finalist
        under an explicit target regime) run ``policy.tiebreak_rounds`` more
        rounds in it, and that column breaks the tie (selects outright under
        the explicit target, whose medians then become the finalists' perf).
        Every decision that depends on a reading is taken by rank 0 and
        broadcast, so distributed ranks time the same finalist in every
        window.
        """
        start = time.perf_counter()
        count = len(finalists)
        # Rebenchmark failures can invalidate these lists in place, including
        # source aliases that the screen drops. Keep the original readings.
        search_perfs = [
            member.perfs[0] if member.perfs else None for member in finalists
        ]
        screen: list[float | None] = [None] * count
        screen_attempted = [False] * count
        readings: list[list[float]] = [[] for _ in range(count)]
        failures: list[Literal["error", "timeout"] | None] = [None] * count
        screen_failures: list[Literal["error", "timeout"] | None] = [None] * count
        warmup_failures: list[Literal["error", "timeout"] | None] = [None] * count
        broadcast_failures: list[Literal["error", "timeout"] | None] = [None] * count
        warmup_status: Literal["not_run", "ok", "incomplete", "error", "timeout"] = (
            "not_run"
        )
        # The second regime's column keeps its own readings and failures: a
        # finalist that fails only there keeps its ranking-regime median.
        second: list[list[float]] = [[] for _ in range(count)]
        second_failures: list[Literal["error", "timeout"] | None] = [None] * count
        second_regime: str | None = None
        select_by_second = False
        group: list[int] = []
        family_survivors: list[int] = []
        process_group_name = self.kernel.env.process_group_name

        def policy_statuses(
            failed: list[Literal["error", "timeout"] | None],
        ) -> list[Literal["error", "timeout"] | None]:
            _, statuses = self._resolve_isolated_rebenchmark_results(
                finalists,
                [
                    IsolatedBenchmarkFailure(status) if status else None
                    for status in failed
                ],
            )
            return statuses

        def commit_results(
            members: Sequence[PopulationMember],
            results: Sequence[IsolatedBenchmarkTiming],
            failed: list[Literal["error", "timeout"] | None],
        ) -> list[PopulationMember]:
            # A completed selection: the survivors' results and the failures of
            # every stage in one application. A terminal local-worker abort
            # applies its failures through ``_exclude_failed_members``. Retry
            # attempts never reach this commit.
            failed_members = [
                (member, status)
                for member, status in zip(
                    finalists, policy_statuses(failed), strict=True
                )
                if status is not None
            ]
            observed = [*members, *(member for member, _ in failed_members)]
            if not observed:
                return []
            timings, statuses = self._resolve_isolated_rebenchmark_results(
                members, results
            )
            return self._apply_rebenchmark_timings(
                observed,
                [*timings, *([inf] * len(failed_members))],
                failure_statuses=[*statuses, *(status for _, status in failed_members)],
                trace_results=[
                    *results,
                    *(IsolatedBenchmarkFailure(status) for _, status in failed_members),
                ],
            )

        def quarantine_targets(
            failed: list[Literal["error", "timeout"] | None],
        ) -> set[int]:
            # Decide eligibility without changing search state: a private-copy
            # retry must discard this attempt's observations in every phase.
            targets, _ = self._rebenchmark_failure_targets(
                finalists, policy_statuses(failed)
            )
            return {id(member) for member, _ in targets}

        def take(
            index: int,
            warmup_ms: int,
            rep_ms: int,
            regime: str | None = None,
            failed: list[Literal["error", "timeout"] | None] = failures,
        ) -> float | None:
            result = measurer.measure(index, warmup_ms, rep_ms, l2_flush_mode=regime)
            reading, status = _classify_final_selection_result(result)
            if status == "error" or status == "timeout":
                failed[index] = status
            return reading

        def complete_medians(
            samples: list[list[float]],
            failed: list[Literal["error", "timeout"] | None],
            rounds: int,
        ) -> dict[int, float]:
            # A finalist competes on the median of a reading from every
            # round: a round it failed, timed out or produced no reading in
            # disqualifies it (one lucky reading must not win over
            # consistently timed rivals).
            return {
                index: statistics.median(samples[index])
                for index in survivors
                if failed[index] is None and len(samples[index]) == rounds > 0
            }

        def run_rounds(
            order_base: list[int],
            rounds: int,
            samples: list[list[float]],
            failed: list[Literal["error", "timeout"] | None],
            regime: str | None,
        ) -> None:
            for round_index in range(rounds):
                # A finalist that failed, timed out or missed a reading in
                # an earlier round cannot compete (it needs a reading from
                # every round), so it is not timed again: a hung finalist
                # would otherwise cost every remaining round a timeout.
                disqualified = sync_object(
                    [
                        index
                        for index in order_base
                        if failed[index] is not None
                        or len(samples[index]) < round_index
                    ],
                    process_group_name=process_group_name,
                )
                order = order_base if round_index % 2 == 0 else order_base[::-1]
                for index in order:
                    if index in disqualified:
                        continue
                    reading = take(
                        index, policy.warmup_ms, policy.rep_ms, regime, failed
                    )
                    if reading is not None:
                        samples[index].append(reading)

        regime_label = self._final_selection_regime()
        try:
            with self._final_selection_measurer(finalists, policy) as measurer:
                regime_label = getattr(measurer, "regime_label", regime_label)
                warmup_skipped: set[int] = set()
                if policy.thermal_warmup_ms > 0:
                    warmup_index = finalists.index(incumbent)
                    warmup_result = measurer.warm(
                        warmup_index, policy.thermal_warmup_ms
                    )
                    _, warmup_status = _classify_final_selection_result(warmup_result)
                    if warmup_status == "error" or warmup_status == "timeout":
                        warmup_failures[warmup_index] = warmup_status
                    # Rank 0 decides which warmup failures must not run again.
                    # Synchronize before choosing screen windows so every rank
                    # takes the same windows even when local warmups disagree.
                    warmup_quarantined = quarantine_targets(warmup_failures)
                    warmup_skipped = {
                        index
                        for index in range(count)
                        if warmup_failures[index] is not None
                        or id(finalists[index]) in warmup_quarantined
                    }
                    # Rank-specialized source equivalence can differ. These
                    # indices protect the screen-window schedule only; they
                    # do not globalize later source-alias invalidation. The
                    # distributed in-process measurer returns no typed failures.
                    warmup_failures, warmup_status, warmup_skipped = sync_object(
                        (warmup_failures, warmup_status, warmup_skipped),
                        process_group_name=process_group_name,
                    )
                    broadcast_failures = warmup_failures
                if warmup_status == "error" or warmup_status == "timeout":
                    self.log.warning(
                        f"Final selection warmup failed for {incumbent.config} "
                        f"in this attempt: {warmup_status}."
                    )
                screened = policy.screens(count)
                if screened:
                    for index in range(count):
                        if index in warmup_skipped:
                            continue
                        screen_attempted[index] = True
                        screen[index] = take(
                            index,
                            policy.screen_warmup_ms,
                            policy.screen_rep_ms,
                            failed=screen_failures,
                        )

                rank_perfs = (
                    [inf if reading is None else reading for reading in screen]
                    if screened
                    else [self._finalist_history_perf(member) for member in finalists]
                )
                # All ranks report the evidence that rank 0 used for the cut,
                # even if another rank saw a different reading or failure.
                search_perfs, screen, screen_failures, rank_perfs = sync_object(
                    (search_perfs, screen, screen_failures, rank_perfs),
                    process_group_name=process_group_name,
                )
                # Failed warmups were not screened, so these observations are
                # disjoint. Keep one failure per finalist for the later commit.
                preselection_failures: list[Literal["error", "timeout"] | None] = [
                    warmup_failure or screen_failure
                    for warmup_failure, screen_failure in zip(
                        warmup_failures, screen_failures, strict=True
                    )
                ]
                broadcast_failures = preselection_failures
                # Direct failures cannot compete on any backend. Only the
                # existing flash policy also excludes config/source aliases.
                preselection_quarantined = quarantine_targets(preselection_failures)
                alive = [
                    index
                    for index in range(count)
                    if preselection_failures[index] is None
                    and id(finalists[index]) not in preselection_quarantined
                ]
                ranked = sorted(alive, key=rank_perfs.__getitem__)
                survivors = ranked[: policy.top_k]
                pinned_alive = [
                    index
                    for index in ranked
                    if finalists[index].config in pinned_configs
                ]
                if pinned_alive and pinned_alive[0] not in survivors:
                    survivors.append(pinned_alive[0])
                # The incumbent defends its title like the pinned seed: it
                # was shortlisted even where later phases re-timed it above
                # the shortlist's search numbers, which rank it last here
                # whenever the shortlist is not screened.
                incumbent_index = finalists.index(incumbent)
                if incumbent_index in alive and incumbent_index not in survivors:
                    survivors.append(incumbent_index)
                # The family quota: the best-ranked config of a body the cut
                # dropped runs the rounds too (the screen is one window, the
                # rounds a median of four at the sustained clock). The seed
                # and the incumbent count as represented, so the quota goes
                # to families neither of them belongs to.
                family_survivors = self._family_representatives(
                    "screen" if screened else "rounds",
                    [finalists[index].config for index in survivors],
                    [
                        (index, finalists[index].config, rank_perfs[index])
                        for index in ranked[policy.top_k :]
                        if index not in survivors
                    ],
                    best=rank_perfs[ranked[0]] if ranked else inf,
                    policy=policy,
                )
                survivors.extend(family_survivors)
                # rank 0 decides who goes on so every rank runs the same windows
                survivors = sync_object(
                    survivors, process_group_name=process_group_name
                )
                run_rounds(survivors, policy.rounds, readings, failures, None)

                if policy.tiebreak_rounds > 0:
                    second_regime = self._final_selection_second_regime(regime_label)
                if second_regime is not None:
                    select_by_second = final_selection_target_regime() != "auto"
                    round_quarantined = quarantine_targets(failures)
                    group = sync_object(
                        self._final_selection_second_group(
                            {
                                index: median
                                for index, median in complete_medians(
                                    readings, failures, policy.rounds
                                ).items()
                                if id(finalists[index]) not in round_quarantined
                            },
                            policy,
                            select_by_second=select_by_second,
                        ),
                        process_group_name=process_group_name,
                    )
                    run_rounds(
                        group,
                        policy.tiebreak_rounds,
                        second,
                        second_failures,
                        second_regime,
                    )
        except BenchmarkWorkerUnkillable as error:
            self._record_unkillable(finalists, error)
            # Worker benchmarks are local: the provider disables them when
            # distributed is initialized. Preserve this attempt's known
            # failures on a terminal abort, without committing retry attempts.
            # Failed warmups skip screening; failed screens skip the rounds,
            # so these three failure lists are disjoint at every index.
            # Keep second-column failures under their existing separate policy.
            # For a synthetic distributed abort, only already-broadcast
            # evidence is safe here; never synchronize from this handler or
            # promote partial rank-local screen/round observations.
            distributed = dist.is_initialized()
            abort_failures = (
                list(broadcast_failures)
                if distributed
                else [
                    warmup or screen or primary
                    for warmup, screen, primary in zip(
                        warmup_failures, screen_failures, failures, strict=True
                    )
                ]
            )
            if (
                not distributed
                and error.fn_index is not None
                and 0 <= error.fn_index < count
            ):
                # The fatal local candidate needs the same flash alias policy
                # as ordinary timeouts. Synthetic distributed aborts add no
                # unbroadcast evidence beyond the existing direct exclusion.
                abort_failures[error.fn_index] = "timeout"
            # Non-flash history stays intact; direct failures cannot win this
            # fallback. Flash also invalidates their config/source aliases.
            self._exclude_failed_members(finalists, abort_failures)
            raise

        members = [finalists[index] for index in survivors]
        results: list[IsolatedBenchmarkTiming] = []
        medians = complete_medians(readings, failures, policy.rounds)
        for index in survivors:
            status = failures[index]
            if status is not None:
                results.append(IsolatedBenchmarkFailure(status))
            else:
                results.append(medians.get(index))
        # The second column, per survivor: None where it was not timed or
        # missed a round, the failure where it failed there.
        second_results: list[IsolatedBenchmarkTiming] = []
        second_medians = complete_medians(
            second, second_failures, policy.tiebreak_rounds
        )
        for index in survivors:
            status = second_failures[index]
            if index not in group:
                second_results.append(None)
            elif status is not None:
                second_results.append(IsolatedBenchmarkFailure(status))
            else:
                second_results.append(second_medians.get(index))
        results, second_results, reported_readings, reported_second = sync_object(
            (results, second_results, readings, second),
            process_group_name=process_group_name,
        )
        # Warmup and screen failures are absent from the survivors, but must
        # quarantine their source just like failures in the rounds. Commit
        # all stages only after completing without a private-copy retry.
        invalidated = commit_results(members, results, preselection_failures)
        quarantined = {id(member): member.status for member in invalidated}
        selected_by = regime_label

        def log_selection(selected: PopulationMember | None) -> None:
            self._log_final_selection(
                finalists,
                survivors,
                screen,
                reported_readings,
                results,
                selected,
                pinned_configs,
                policy,
                wall_s=time.perf_counter() - start,
                regime_label=regime_label,
                second=reported_second,
                second_results=second_results,
                second_group=group,
                second_regime=second_regime,
                select_by_second=select_by_second,
                selected_by=selected_by if selected is not None else None,
                family_indices=family_survivors,
                incumbent=incumbent,
                quarantined=quarantined,
                screen_attempted=screen_attempted,
                search_perfs=search_perfs,
                rank_perfs=rank_perfs,
                screen_failures=screen_failures,
                warmup_status=warmup_status,
            )

        live = [
            member
            for member, result in zip(members, results, strict=True)
            if isinstance(result, float) and math.isfinite(member.perf)
        ]
        if not live:
            live = [
                member
                for member, result in zip(members, results, strict=True)
                if not isinstance(result, IsolatedBenchmarkFailure)
                and math.isfinite(member.perf)
            ]
            if not live:
                # A screen-passed finalist cut from the rounds still has a
                # valid search measurement. A failure of every round survivor
                # must not discard it, nor revive direct or source-alias failures.
                live = [
                    member
                    for index, member in enumerate(finalists)
                    if index not in survivors
                    and screen[index] is not None
                    and preselection_failures[index] is None
                    and id(member) not in quarantined
                    and math.isfinite(member.perf)
                ]
                if live:
                    selected_by = "search fallback"
            fallback = (
                "keeping the best eligible search config."
                if live
                else "no eligible config remains."
            )
            self.log.warning(
                "Final selection took no eligible complete steady-window readings; "
                f"{fallback}"
            )
        elif len(live) < len(members):
            # finalists rarely fail here: they all ran in the search
            self.log.warning(
                f"Final selection took no eligible complete steady-window readings for "
                f"{len(members) - len(live)} of {len(members)} finalists; "
                "selecting among the rest."
            )
        if not live:
            log_selection(None)
            raise exc.NoConfigFound
        after = self._prefer_pinned_within_tolerance(
            self._prefer_backend_preference_within_tolerance(
                min(live, key=performance), live
            ),
            live,
            pinned_configs,
        )

        # Group eligibility was proved before timing. Keep that invariant
        # explicit when consuming its authoritative second-regime results.
        second_by_position = {
            position: result
            for position, result in enumerate(second_results)
            if isinstance(result, float)
        }
        assert all(
            id(members[position]) not in quarantined for position in second_by_position
        )
        # the regime whose column selected ``after``
        if select_by_second:
            if second_by_position:
                # The target regime is the one the kernel is scored in, so
                # its medians are the finalists' perf from here on. A
                # finalist that failed or missed a round in that column
                # keeps the ranking regime's median as its perf (the table
                # names the regime of each column and its
                # ``second_disqualified`` verdict) and cannot be selected,
                # as a finalist without a complete ranking median keeps the
                # search's perf and cannot be.
                timed = [members[position] for position in second_by_position]
                self._apply_rebenchmark_timings(
                    timed, list(second_by_position.values())
                )
                live_second = [member for member in timed if math.isfinite(member.perf)]
                if live_second:
                    after = self._prefer_pinned_within_tolerance(
                        self._prefer_backend_preference_within_tolerance(
                            min(live_second, key=performance), live_second
                        ),
                        live_second,
                        pinned_configs,
                    )
                    selected_by = second_regime
            if selected_by != second_regime:
                self.log.warning(
                    f"Final selection took no eligible complete readings in the L2 "
                    f"{second_regime} regime; selecting by the L2 {regime_label} "
                    "column."
                )
        elif second_by_position:
            after = self._break_tie_by_second_column(
                after, members, second_by_position, pinned_configs, policy
            )
        log_selection(after)
        return after

    @staticmethod
    def _final_selection_second_group(
        medians: dict[int, float],
        policy: FinalSelectionConfig,
        *,
        select_by_second: bool,
    ) -> list[int]:
        """The finalists the second regime times, in survivor order: every
        finalist with a complete median under an explicit target regime,
        else those tied within ``policy.tie_tolerance`` of the best median
        (none when the best stands alone: nothing to break)."""
        if not medians:
            return []
        if select_by_second:
            return list(medians)
        best = min(medians.values())
        tied = [
            index
            for index, median in medians.items()
            if median <= best * (1.0 + policy.tie_tolerance)
        ]
        return tied if len(tied) > 1 else []

    def _break_tie_by_second_column(
        self,
        after: PopulationMember,
        members: Sequence[PopulationMember],
        second_by_position: dict[int, float],
        pinned_configs: MutableSet[Config],
        policy: FinalSelectionConfig,
    ) -> PopulationMember:
        """Among the finalists tied in the ranking regime, the fastest in the
        second regime wins unless ``after`` (the ranking regime's pick) is
        within ``policy.tie_tolerance`` of it there too; a pinned seed within
        its tolerance of that fastest keeps its preference."""
        positions = [
            position for position in second_by_position if members[position] is after
        ]
        if not positions:
            # ``after`` has no complete median in the second column: either
            # the pinned preference reached outside the tied group, or the
            # ranking regime's pick itself failed or missed a round there.
            # Without cold evidence against it the ranking pick stands (a
            # rival's cold reading alone is not evidence that ``after`` is
            # slower cold).
            return after
        after_position = positions[0]
        by_member = {
            id(members[position]): median
            for position, median in second_by_position.items()
        }

        def key(member: PopulationMember) -> float:
            return by_member[id(member)]

        candidates = [members[position] for position in second_by_position]
        best = self._prefer_pinned_within_tolerance(
            min(candidates, key=key), candidates, pinned_configs, key=key
        )
        if second_by_position[after_position] <= key(best) * (
            1.0 + policy.tie_tolerance
        ):
            return after
        return best

    def _log_final_selection(
        self,
        finalists: Sequence[PopulationMember],
        survivors: Sequence[int],
        screen: Sequence[float | None],
        readings: Sequence[Sequence[float]],
        results: Sequence[IsolatedBenchmarkTiming],
        selected: PopulationMember | None,
        pinned_configs: MutableSet[Config],
        policy: FinalSelectionConfig,
        *,
        wall_s: float,
        regime_label: str,
        incumbent: PopulationMember,
        quarantined: Mapping[int, str],
        screen_attempted: Sequence[bool],
        search_perfs: Sequence[float | None],
        rank_perfs: Sequence[float],
        screen_failures: Sequence[Literal["error", "timeout"] | None],
        warmup_status: Literal["not_run", "ok", "incomplete", "error", "timeout"],
        second: Sequence[Sequence[float]] | None = None,
        second_results: Sequence[IsolatedBenchmarkTiming] | None = None,
        second_group: Sequence[int] = (),
        second_regime: str | None = None,
        select_by_second: bool = False,
        selected_by: str | None = None,
        family_indices: Sequence[int] = (),
    ) -> None:
        if selected_by is None and selected is not None:
            selected_by = regime_label
        family_keys = self._final_selection_family_keys()
        result_by_index = dict(zip(survivors, results, strict=True))
        second_by_index: dict[int, IsolatedBenchmarkTiming] = (
            {}
            if second_results is None
            else dict(zip(survivors, second_results, strict=True))
        )
        # the finalists whose windows the second column spent, whether or
        # not they produced a reading
        second_timed = {index for index in survivors if index in second_group}
        second_eligible = {
            index
            for index in second_timed
            if isinstance(second_by_index.get(index), float)
        }
        # the table is ordered by the column that selected
        sort_by_second = second_regime is not None and selected_by == second_regime

        def verdict(result: IsolatedBenchmarkTiming) -> str | None:
            if isinstance(result, IsolatedBenchmarkFailure):
                return result.status
            if result is None:
                return "incomplete"
            return None

        def sort_key(index: int) -> float:
            result = (
                second_by_index.get(index) if sort_by_second else result_by_index[index]
            )
            return result if isinstance(result, float) else inf

        rows: list[dict[str, object]] = []
        lines: list[str] = []
        survivor_indices = set(survivors)
        shortlist_rows: list[dict[str, object]] = []
        member_rows: list[dict[str, object]] = [
            {
                "config_id": canonical_config_id(member.config),
                "pinned": member.config in pinned_configs,
                "incumbent": member is incumbent,
                "screen_family_quota": index in family_indices,
                "quarantined": quarantined.get(id(member)),
                "search_ms": search_perfs[index],
                "screen_ms": screen[index],
                "warmup_status": warmup_status if member is incumbent else "not_run",
            }
            for index, member in enumerate(finalists)
        ]
        if family_keys:
            for member, row in zip(finalists, member_rows, strict=True):
                row["family"] = _family_label(
                    _config_family(member.config, family_keys)
                )
        for index in range(len(finalists)):
            screen_status = "not_run"
            if screen_attempted[index]:
                screen_status = screen_failures[index] or (
                    "ok" if screen[index] is not None else "incomplete"
                )
            shortlist_rows.append(
                {
                    **member_rows[index],
                    "index": index,
                    "rank_ms": rank_perfs[index]
                    if math.isfinite(rank_perfs[index])
                    else None,
                    "screen_status": screen_status,
                    "survived": index in survivor_indices,
                }
            )
        for index in sorted(survivors, key=sort_key):
            member = finalists[index]
            samples = list(readings[index])
            result = result_by_index[index]
            median = result if isinstance(result, float) else None
            config_id = member_rows[index]["config_id"]
            row: dict[str, object] = {
                **member_rows[index],
                # Preserve the existing survivor-table field for consumers.
                "family_quota": index in family_indices,
                "rounds_ms": samples,
                "median_ms": median,
                "disqualified": verdict(result),
            }
            second_text = ""
            if index in second_timed:
                second_result = second_by_index.get(index)
                second_median = (
                    second_result if isinstance(second_result, float) else None
                )
                second_samples = [] if second is None else list(second[index])
                row["second_rounds_ms"] = second_samples
                row["second_median_ms"] = second_median
                row["second_disqualified"] = verdict(second_result)
                second_median_text = (
                    f"{second_median:.4g}"
                    if second_median is not None
                    else verdict(second_result) or "n/a"
                )
                second_text = (
                    f" {second_regime}={second_median_text}"
                    f"[{' '.join(f'{sample:.4g}' for sample in second_samples)}]"
                )
            rows.append(row)
            marks = (
                ("*" if member is selected else " ")
                + ("P" if member.config in pinned_configs else " ")
                + ("F" if index in family_indices else " ")
            )
            family_text = f" family={row['family']}" if family_keys else ""
            quarantine_text = (
                f" quarantined={row['quarantined']}" if row["quarantined"] else ""
            )
            # four significant digits: microsecond kernels differ in the
            # fourth decimal of a millisecond reading
            median_text = (
                f"{median:.4g}{self.performance_suffix}"
                if median is not None
                else verdict(result) or "n/a"
            )
            rounds_text = " ".join(f"{sample:.4g}" for sample in samples)
            screen_text = "" if screen[index] is None else f"{screen[index]:.4g}"
            search_perf = search_perfs[index]
            search_text = (
                self.format_performance(search_perf)
                if search_perf is not None
                else "n/a"
            )
            lines.append(
                f"  {marks} {median_text:>10} [{rounds_text}]{second_text} "
                f"screen={screen_text} search={search_text}{family_text}"
                f"{quarantine_text} {config_id}"
            )
        if second_regime is None:
            second_clause = ""
        elif select_by_second and selected_by == second_regime:
            second_clause = (
                f" by {policy.tiebreak_rounds} rounds of L2 {second_regime} windows"
            )
        elif select_by_second:
            second_clause = (
                f" by the L2 {regime_label} column (no eligible complete reading in "
                f"{policy.tiebreak_rounds} rounds of L2 {second_regime} windows)"
            )
        elif len(second_eligible) > 1 and any(
            finalists[index] is selected for index in second_eligible
        ):
            second_clause = (
                f", the tie within {policy.tie_tolerance:.1%} broken by "
                f"{policy.tiebreak_rounds} rounds of L2 {second_regime} windows"
            )
        elif second_eligible:
            second_clause = (
                f", the tie within {policy.tie_tolerance:.1%} unresolved: no "
                f"eligible complete comparison in {policy.tiebreak_rounds} rounds "
                f"of L2 {second_regime} windows"
            )
        elif second_timed:
            second_clause = (
                f", the tie within {policy.tie_tolerance:.1%} unresolved: no "
                f"complete reading in {policy.tiebreak_rounds} rounds of L2 "
                f"{second_regime} windows"
            )
        else:
            second_clause = (
                f", no tie within {policy.tie_tolerance:.1%} for L2 "
                f"{second_regime} windows to break"
            )
        selected_text = (
            self.format_performance(selected.perf)
            if selected is not None
            else "none (no eligible finalist)"
        )
        self.log(
            f"Final selection: {len(finalists)} shortlisted, {len(survivors)} "
            f"finalists x {policy.rounds} rounds of {policy.warmup_ms}+{policy.rep_ms} ms "
            f"windows (L2 {regime_label}) in {wall_s:.1f}s; selected "
            f"{selected_text}{second_clause} "
            "(* selected, P pinned seed, F family quota):\n" + "\n".join(lines)
        )
        self._autotune_metrics.final_selection = {
            "policy": dataclasses.asdict(policy),
            "family_keys": list(family_keys),
            "input_banks": self._input_bank_count,
            # (only behind a prelude: the earlier kernel calls every window
            # ran between its flush and its timed call)
            **(
                {}
                if self._benchmark_prelude is None
                else {"prelude_steps": len(self._benchmark_prelude.steps)}
            ),
            "l2_flush_mode": regime_label,
            "second_l2_flush_mode": second_regime,
            "target_regime": final_selection_target_regime(),
            "selected_by": selected_by,
            "shortlisted": len(finalists),
            "finalists": len(survivors),
            "wall_s": wall_s,
            "selected_config_id": canonical_config_id(selected.config)
            if selected is not None
            else None,
            "shortlist_rows": shortlist_rows,
            "rows": rows,
        }

    def compare(self, a: PopulationMember, b: PopulationMember) -> int:
        """
        Compare two population members based on their performance, possibly with re-benchmarking.

        Args:
            a: The first population member.
            b: The second population member.

        Returns:
            -1 if a is better than b, 1 if b is better than a, 0 if they are equal.
        """
        if self._kernel_holder(a, (a, b)) is self._kernel_holder(b, (a, b)):
            # Two configs of one kernel are equal, whatever each has read:
            # there is nothing to time, and a child that repeats its
            # parent's code does not replace it.
            return 0
        # A config reads what its kernel reads now. A candidate that was
        # settled before an earlier comparison re-timed its kernel (a batch
        # that is benchmarked first and compared afterwards) is not a member
        # of the population yet, so that re-timing did not reach it.
        self._follow_kernel_readings((a, b), (a, b))
        if self.should_rebenchmark(a) and self.should_rebenchmark(b):
            self._rebenchmark_kernels([a, b])
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
        self,
        members: list[PopulationMember],
        *,
        desc: str = "Rebenchmarking",
        target_ms: float = _REBENCHMARK_TARGET_MS_DEFAULT,
        use_isolated: bool = True,
        confirm_suspicious: bool = True,
        use_interleaved: bool = True,
        candidate_private_args: bool = False,
        allow_single: bool = False,
    ) -> None:
        """
        Re-benchmark a list of population members to avoid outliers.

        Args:
            members: The list of population members to rebenchmark.
            desc: Description for the progress bar.
            candidate_private_args: Use a fresh worker process per candidate;
                when unavailable, keep argument storage private in-process.
            allow_single: Re-time a single member too. A re-timing compares
                members, so one member alone needs none (the default); a
                pass that moves the readings to another timer
                (``_read_on_settled_timer``) reads every member.
        """
        if len(members) < (1 if allow_single else 2):
            return
        # A search that reads its candidates from the trace re-times in the
        # benchmark worker whatever the caller asks: the in-process timers
        # read with CUDA events, on another scale than the members' readings
        # (the single-pass final verification asks for them under
        # ``HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED=0``).
        # So does a search behind a prelude: the in-process timers would
        # time its members alone.
        use_isolated = (
            use_isolated
            or self.traces_candidates
            or self._benchmark_prelude is not None
        )
        if isinstance(self.benchmark_provider, MultiShapeBenchmarkProvider):
            try:
                provider_timings = self.benchmark_provider.rebenchmark(
                    [member.config for member in members],
                    [member.perf for member in members],
                    desc=desc,
                )
            except BenchmarkWorkerUnkillable as error:
                # A joint re-timing that completes applies its readings
                # without the alias policy of a flash search, so an abandoned
                # one quarantines nothing either.
                self._record_abandoned_batch(members, error, quarantine=False)
                raise
            # The provider records only fresh joint measurements; skipped work
            # can return retained timings that must not become new trace rows.
            self._apply_rebenchmark_timings(
                members, provider_timings, record_trace=False
            )
            return

        # Size the in-process repeat from the candidates being rechecked. A
        # global optimistic outlier can be the reason we are rebenchmarking.
        repeat = PopulationBasedSearch._repeat_for_target_ms(
            target_ms, PopulationBasedSearch._repeat_reference_perf(members)
        )
        if (capstr := os.getenv("HELION_CAP_REBENCHMARK_REPEAT")) is not None:
            repeat = min(repeat, int(capstr))
        repeat = max(1, repeat)

        in_process_isolation = False
        if use_isolated and self.settings.autotune_benchmark_fn is None:
            try:
                isolated_results = self._isolated_rebenchmark_results(
                    members,
                    rep=PopulationBasedSearch._isolated_rep_ms(
                        target_ms, self.settings.autotune_benchmark_timeout
                    ),
                    desc=desc,
                    candidate_private_args=candidate_private_args,
                )
            except BenchmarkWorkerUnkillable as error:
                self._record_abandoned_batch(members, error)
                raise
            if isolated_results is not None:
                new_timings, failure_statuses = (
                    self._resolve_isolated_rebenchmark_results(
                        members, isolated_results
                    )
                )
                new_timings, failure_statuses = sync_object(
                    (new_timings, failure_statuses),
                    process_group_name=self.kernel.env.process_group_name,
                )
                self._apply_rebenchmark_timings(
                    members,
                    new_timings,
                    failure_statuses=failure_statuses,
                    trace_results=isolated_results,
                )
                return
            if self.traces_candidates:
                # The search reads from the trace in the benchmark worker,
                # which cannot time these members now: the in-process timers
                # read with CUDA events, on another scale than the readings
                # the members have, so they keep those.
                self.log.warning(
                    f"{desc}: the benchmark worker cannot time the "
                    f"{len(members)} configs; this search reads its candidates "
                    "from the CUDA activity trace, so they keep their readings"
                )
                return
            if self._benchmark_prelude is not None:
                # Likewise behind a prelude, which only the worker runs.
                self.log.warning(
                    f"{desc}: the benchmark worker cannot time the "
                    f"{len(members)} configs; this search times its candidates "
                    "behind a prelude, so they keep their readings"
                )
                return
            if candidate_private_args and dist.is_initialized():
                self.log.warning(
                    "Candidate-private mutated-argument isolation is unavailable "
                    "for distributed in-process rebenchmarking."
                )
            in_process_isolation = candidate_private_args and not dist.is_initialized()

        if self._input_bank_count > 1:
            raise exc.InvalidAPIUsage(
                "rotating input banks require isolated rebenchmarking"
            )

        private_args: _PrivateBenchmarkArgs | None = None
        if (
            in_process_isolation
            and len(self.benchmark_provider.mutated_arg_indices) > 0
        ):
            # Every finalist gets private live tensor storage, including
            # inputs that the kernel only reads (see ``_PrivateBenchmarkArgs``
            # for the sacrificial clone it keeps alive).
            try:
                private_args = self._private_benchmark_args(len(members))
            except torch.OutOfMemoryError as error:
                raise exc.AutotuneError(
                    "Unable to allocate candidate-private mutated arguments "
                    "for in-process finalist isolation. Reduce "
                    f"{_FINAL_REBENCHMARK_TOP_K_ENV} and retry."
                ) from error
            benchmark_args_by_member = private_args.by_member
        else:
            benchmark_args_by_member = [self._shared_benchmark_args()] * len(members)

        def make_rebenchmark_callable(
            member: PopulationMember,
            benchmark_args: Sequence[object],
            *,
            clear_each_call: bool,
        ) -> Callable[[], object]:
            run_member = functools.partial(member.fn, *benchmark_args)

            def wrapped() -> object:
                if clear_each_call:
                    try:
                        return run_member()
                    finally:
                        clear_jit_fast_path_caches(member.fn, self.log)
                else:
                    return run_member()

            # The timers read the host function's launch count
            # (``_helion_cute_launches``: a call that launches several kernels
            # is timed as a graph replay where a window reads it like its
            # rivals) off the callable they are given. Only that attribute is
            # carried, not the host's name or signature: a member callable
            # from user code (the external autotuner's ``compile_fn``) may
            # have no ``__dict__`` (a builtin such as ``operator.add``) or one
            # holding the ``__wrapped__`` of ``functools.wraps``, which
            # ``inspect.signature`` would follow. One without the attribute
            # counts one launch.
            launches = getattr(member.fn, "_helion_cute_launches", None)
            if launches is not None:
                wrapped._helion_cute_launches = launches  # pyrefly: ignore[missing-attribute]
            return wrapped

        try:
            if use_interleaved or self.settings.autotune_benchmark_fn is not None:
                iterator = [
                    make_rebenchmark_callable(
                        member,
                        benchmark_args,
                        clear_each_call=True,
                    )
                    for member, benchmark_args in zip(
                        members, benchmark_args_by_member, strict=True
                    )
                ]
                benchmark_function: Callable[..., list[float]]
                if self.settings.autotune_benchmark_fn is not None:
                    benchmark_function = self.settings.autotune_benchmark_fn
                else:
                    interleaved_benchmark = (
                        self.config_spec.backend.get_interleaved_bench()
                        or interleaved_bench
                    )
                    # ``repeat`` is sized from candidate kernel time alone; for
                    # microsecond kernels the fixed per-call overhead dominates
                    # and the capped repeat count can take minutes of wall
                    # clock. Keep the whole pass near the sequential-window
                    # budget it replaced (target_ms per candidate).
                    interleaved_kwargs: dict[str, object] = {
                        "max_total_ms": target_ms * len(members)
                    }
                    if interleaved_benchmark is interleaved_bench:
                        # the batched graph timer agrees its batch over the
                        # kernel's process group so every rank launches alike
                        interleaved_kwargs["process_group_name"] = (
                            self.kernel.env.process_group_name
                        )
                    benchmark_function = functools.partial(
                        interleaved_benchmark, **interleaved_kwargs
                    )
                if self.settings.autotune_progress_bar:
                    new_timings = benchmark_function(iterator, repeat=repeat, desc=desc)
                else:
                    new_timings = benchmark_function(iterator, repeat=repeat)
            else:
                iterator = [
                    make_rebenchmark_callable(
                        member,
                        benchmark_args,
                        clear_each_call=False,
                    )
                    for member, benchmark_args in zip(
                        members, benchmark_args_by_member, strict=True
                    )
                ]
                steady_bench = self._steady_bench()
                rep_ms = self._steady_rebenchmark_rep_ms(target_ms)
                warmup_ms = min(1000, rep_ms)
                new_timings = []
                for fn in iterator:
                    timing = steady_bench(
                        fn,
                        warmup=warmup_ms,
                        rep=rep_ms,
                        return_mode="median",
                        process_group_name=self.kernel.env.process_group_name,
                    )
                    if isinstance(timing, tuple):
                        timing = timing[0]
                    new_timings.append(float(timing))
        finally:
            for m in members:
                clear_jit_fast_path_caches(m.fn, self.log)
        if confirm_suspicious:
            new_timings = self._confirm_suspicious_rebenchmark_timings(
                members,
                new_timings,
                desc=desc,
            )
        resolved_timings, failure_statuses = self._resolve_isolated_rebenchmark_results(
            members, new_timings
        )
        resolved_timings, failure_statuses = sync_object(
            (resolved_timings, failure_statuses),
            process_group_name=self.kernel.env.process_group_name,
        )
        self._apply_rebenchmark_timings(
            members,
            resolved_timings,
            failure_statuses=failure_statuses,
            trace_results=new_timings,
        )

    @_retry_without_private_copies
    def _isolated_rebenchmark_results(
        self,
        members: Sequence[PopulationMember],
        *,
        rep: int,
        desc: str,
        candidate_private_args: bool,
    ) -> list[IsolatedBenchmarkTiming] | None:
        """One isolated steady window per member in the killable worker;
        ``None`` when the worker cannot time them (the caller falls back to
        the in-process timers).

        With ``candidate_private_args`` every member reads its own copy of the
        kernel arguments in the one long-lived worker -- the regime of the
        full-effort final selection: a candidate's L2-resident (or
        ``evict_last``-pinned) input lines cannot become hits for the next
        candidate because the next one reads different addresses. That keeps
        the isolation a fresh process per finalist provided without its cost
        (a spawn re-imports torch, the backend and the caller's ``__main__``,
        then loads the arguments: 7 s per finalist here, 45-60 s under a heavy
        ``__main__``). The copies are released when the batch ends. A fresh
        process per finalist remains the fallback when the copies do not fit
        on the device next to the originals.
        """
        provider = self.benchmark_provider
        fns = [member.fn for member in members]
        if isinstance(provider, LocalBenchmarkProvider):
            # A failed re-timing names its candidate in the log.
            provider.register_isolated_candidates(
                fns, [member.config for member in members]
            )
        if (
            candidate_private_args
            and isinstance(provider, LocalBenchmarkProvider)
            and provider.steady_window_benchmark_available()
            and provider.private_argument_copies_affordable(len(fns))
        ):
            # The timer sizes its repetitions from an estimate pass, so a
            # window can stretch; three windows plus the wrapper and argument
            # loads still finish (see ``_final_selection_measurer``).
            window_s = (1 + rep) / 1000.0
            timeout_s = max(
                float(self.settings.autotune_benchmark_timeout),
                3.0 * window_s + 15.0,
            )
            results: list[IsolatedBenchmarkTiming] = []
            try:
                with _WorkerFinalSelectionMeasurer(
                    provider,
                    fns,
                    private_copies=True,
                    timeout_s=timeout_s,
                    desc=desc,
                ) as measurer:
                    for index in range(len(fns)):
                        results.append(measurer.measure(index, 1, rep))
                        if not provider.steady_window_benchmark_available():
                            # A wrapper the worker cannot load disables it for
                            # the rest of the batch; time every finalist
                            # in-process rather than mixing regimes.
                            return None
            except BenchmarkWorkerUnkillable as error:
                # The measurer's batches hold one member each. The batch this
                # ends is the members so far, also when the worker survives
                # its kill at the release of their copies.
                error.batch_results = tuple(results)
                raise
            return results
        return provider.benchmark_isolated(
            fns,
            warmup=1,
            rep=rep,
            desc=desc,
            # Behind a prelude neither is needed: every timed call follows
            # a flush and the prelude's work, which hands each member its
            # arguments anew, and every window first displaces the L2 lines
            # earlier ones pinned (``do_bench``), so no member reads what an
            # earlier one left.
            fresh_process=candidate_private_args and self._benchmark_prelude is None,
        )

    def mirrored_rebenchmark(
        self,
        members: list[PopulationMember],
        *,
        desc: str,
        target_ms: float = _REBENCHMARK_TARGET_MS_DEFAULT,
    ) -> MirroredBenchmarkTrace:
        """Rebenchmark candidates with deterministic mirrored wall-time sweeps."""
        if len(members) < 2:
            return MirroredBenchmarkTrace([], [], [member.perf for member in members])
        if isinstance(self.benchmark_provider, MultiShapeBenchmarkProvider):
            raise exc.AutotuneError(
                "mirrored terminal refinement does not support multi-shape benchmarking"
            )
        if self.settings.autotune_benchmark_fn is not None:
            raise exc.AutotuneError(
                "mirrored terminal refinement requires the default benchmark function"
            )
        if self._benchmark_prelude is not None:
            # (these sweeps run in this process, which has no prelude)
            raise exc.AutotuneError(
                "mirrored terminal refinement cannot time behind a benchmark prelude"
            )

        repeat_reference_perf_ms = self._repeat_reference_perf(members)
        repeat = self._repeat_for_target_ms(target_ms, repeat_reference_perf_ms)
        if (capstr := os.getenv("HELION_CAP_REBENCHMARK_REPEAT")) is not None:
            repeat = min(repeat, int(capstr))
            repeat = max(2, repeat - repeat % 2)
        else:
            repeat = max(2, repeat + repeat % 2)

        if self._input_bank_count > 1:
            return self._mirrored_bank_rebenchmark(
                members, repeat, desc, target_ms, repeat_reference_perf_ms
            )

        if self.benchmark_provider.mutated_arg_indices:
            benchmark_args = _clone_args(
                self.args,
                self.kernel.env.process_group_name,
                idx_to_clone=self.benchmark_provider.mutated_arg_indices,
            )
        else:
            benchmark_args = self.args

        def after_call(index: int) -> None:
            clear_jit_fast_path_caches(members[index].fn, self.log)

        try:
            trace = mirrored_bench_generic(
                [functools.partial(member.fn, *benchmark_args) for member in members],
                repeat=repeat,
                desc=desc if self.settings.autotune_progress_bar else None,
                after_call=after_call,
            )
            trace = dataclasses.replace(
                trace,
                target_ms=target_ms,
                repeat_reference_perf_ms=repeat_reference_perf_ms,
            )
            trace = sync_object(
                trace,
                process_group_name=self.kernel.env.process_group_name,
            )
            self._apply_rebenchmark_timings(members, trace.medians_ms)
            return trace
        finally:
            for member in members:
                clear_jit_fast_path_caches(member.fn, self.log)

    @_retry_without_private_copies
    def _mirrored_bank_rebenchmark(
        self,
        members: list[PopulationMember],
        repeat: int,
        desc: str,
        target_ms: float,
        reference_ms: float,
    ) -> MirroredBenchmarkTrace:
        """Keep mirrored ordering while each worker sample times complete cycles."""
        provider = self.benchmark_provider
        if not isinstance(provider, LocalBenchmarkProvider):
            raise InputBankError("mirrored input banks require the isolated provider")
        fns = [member.fn for member in members]
        provider.register_isolated_candidates(
            fns, [member.config for member in members]
        )
        sweeps, cycles, total_cycles = _mirrored_bench_call_layout(
            math.ceil(repeat / self._input_bank_count)
        )
        samples: list[list[float]] = [[] for _ in members]
        orders: list[list[int]] = []
        elapsed: list[list[float]] = []
        failures: list[Literal["error", "timeout"] | None] = [None] * len(members)
        incomplete = [False] * len(members)
        try:
            with _WorkerFinalSelectionMeasurer(
                provider,
                fns,
                private_copies=provider.private_argument_copies_affordable(len(fns)),
                timeout_s=float(self.settings.autotune_benchmark_timeout),
                l2_flush_mode=final_selection_l2_flush_mode(),
                desc=desc,
            ) as measurer:
                indices = list(range(len(members)))
                for sweep in range(sweeps):
                    offset = (sweep // 2) % len(indices)
                    rotated = indices[offset:] + indices[:offset]
                    order = rotated if sweep % 2 == 0 else list(reversed(rotated))
                    values: list[float] = []
                    for index in order:
                        if failures[index] is not None or incomplete[index]:
                            samples[index].append(inf)
                            values.append(inf)
                            continue
                        result = measurer.measure(
                            index,
                            0,
                            1,
                            fixed_repetitions=cycles,
                        )
                        if result is None:
                            # An unavailable worker reading is not a config failure.
                            # Keep its prior observation, but exclude this candidate
                            # from the comparison rather than retrying part of it.
                            incomplete[index] = True
                            value = inf
                        elif isinstance(result, IsolatedBenchmarkFailure):
                            failures[index] = result.status
                            value = inf
                        else:
                            value = result
                        samples[index].append(value)
                        values.append(value)
                    orders.append(order)
                    elapsed.append(values)
        except BenchmarkWorkerUnkillable as error:
            # The measurer's batches hold one member each. The batch this ends
            # is the sweeps so far, also when the worker survives its kill at
            # the release of the members' copies.
            error.batch_results = tuple(
                IsolatedBenchmarkFailure(failed) if failed else None
                for failed in failures
            )
            self._record_abandoned_batch(members, error)
            raise
        medians = [
            inf if failed or missing else statistics.median(values)
            for failed, missing, values in zip(
                failures, incomplete, samples, strict=True
            )
        ]
        raw_results: list[IsolatedBenchmarkTiming] = [
            None if missing else IsolatedBenchmarkFailure(failed) if failed else median
            for failed, missing, median in zip(
                failures, incomplete, medians, strict=True
            )
        ]
        timings, statuses = self._resolve_isolated_rebenchmark_results(
            members, raw_results
        )
        self._apply_rebenchmark_timings(
            members, timings, failure_statuses=statuses, trace_results=raw_results
        )
        return MirroredBenchmarkTrace(
            orders,
            elapsed,
            medians,
            target_ms=target_ms,
            repeat_reference_perf_ms=reference_ms,
            sweep_count=sweeps,
            calls_per_sample=cycles * self._input_bank_count,
            total_calls=total_cycles * self._input_bank_count,
            statuses=[
                failed or ("incomplete" if missing else "ok")
                for failed, missing in zip(failures, incomplete, strict=True)
            ],
        )

    def _resolve_isolated_rebenchmark_results(
        self,
        members: Sequence[PopulationMember],
        results: Sequence[IsolatedBenchmarkTiming],
    ) -> tuple[list[float], list[Literal["error", "timeout"] | None]]:
        invalidate_failures = bool(
            getattr(
                getattr(self, "config_spec", None),
                "cute_flash_search_enabled",
                False,
            )
        )
        timings: list[float] = []
        failure_statuses: list[Literal["error", "timeout"] | None] = []
        for member, result in zip(members, results, strict=True):
            if isinstance(result, IsolatedBenchmarkFailure):
                if invalidate_failures:
                    timings.append(inf)
                    failure_statuses.append(result.status)
                elif result.status == "error":
                    # Preserve the pre-existing behavior for a sticky runtime
                    # error outside CuTe flash: remove it from the current
                    # ranking without changing its benchmark status.
                    timings.append(inf)
                    failure_statuses.append(None)
                else:
                    timings.append(member.perf)
                    failure_statuses.append(None)
            else:
                timings.append(member.perf if result is None else result)
                failure_statuses.append(None)
        return timings, failure_statuses

    def _apply_rebenchmark_timings(
        self,
        members: Sequence[PopulationMember],
        timings: Sequence[float],
        *,
        failure_statuses: Sequence[Literal["error", "timeout"] | None] | None = None,
        trace_results: Sequence[IsolatedBenchmarkTiming] | None = None,
        record_trace: bool = True,
    ) -> list[PopulationMember]:
        """Apply observations and return every member quarantined by a failure."""
        if failure_statuses is None:
            failure_statuses = [None] * len(members)
        if trace_results is None:
            trace_results = timings
        for member, timing, failure_status, trace_result in zip(
            members, timings, failure_statuses, trace_results, strict=True
        ):
            config_id = (
                self.log.register_config(member.config)
                if record_trace and trace_result is not None and self.log.trace_enabled
                else None
            )
            if config_id is not None and trace_result is not None:
                trace_status = (
                    trace_result.status
                    if isinstance(trace_result, IsolatedBenchmarkFailure)
                    else failure_status
                )
                perf = timing if trace_status is None else inf
                self.log.record_trace_entry(
                    AutotuneLogEntry(
                        generation=self._autotune_metrics.num_generations,
                        status=trace_status
                        or ("ok" if math.isfinite(perf) else "error"),
                        perf_ms=(
                            perf
                            if self.performance_unit == "ms"
                            else self.benchmark_provider.raw_latency(member.config)
                            if isinstance(
                                self.benchmark_provider, MultiShapeBenchmarkProvider
                            )
                            else None
                        ),
                        compile_time=None,
                        config_id=config_id,
                        config=member.config,
                        objective=perf,
                        objective_unit=self.performance_unit,
                    ),
                    event="rebenchmark",
                )
            if (
                failure_status is not None
                or trace_result is None
                or (
                    isinstance(trace_result, IsolatedBenchmarkFailure)
                    and trace_result.status == "timeout"
                )
            ):
                continue
            member.perfs.append(timing)
            if timing < self.best_perf_so_far:
                self.best_perf_so_far = timing
        invalidated = self._invalidate_cute_flash_rebenchmark_failures(
            members, failure_statuses
        )
        refreshed = list(members)
        refreshed.extend(invalidated)
        self._refresh_benchmarked_members_after_rebenchmark(refreshed)
        if invalidated:
            self._recompute_cute_flash_best_perf(refreshed)
        _refresh_handoff(refreshed)
        return invalidated

    def _rebenchmark_failure_targets(
        self,
        members: Sequence[PopulationMember],
        failure_statuses: Sequence[Literal["error", "timeout"] | None],
    ) -> tuple[list[tuple[PopulationMember, Literal["error", "timeout"]]], set[str]]:
        """Find failed members and their config/source aliases without mutation."""
        failed_member_statuses = {
            id(member): status
            for member, status in zip(members, failure_statuses, strict=True)
            if status is not None
        }
        if not failed_member_statuses:
            return [], set()

        backend = self.config_spec.backend
        failed_config_statuses = ConfigCache(
            (member.config, status)
            for member, status in zip(members, failure_statuses, strict=True)
            if status is not None
        )
        failed_source_statuses: dict[str, Literal["error", "timeout"]] = {}
        for member, status in zip(members, failure_statuses, strict=True):
            if status is None:
                continue
            source_hash = backend.generated_source_hash(member.fn)
            if source_hash is not None:
                existing = failed_source_statuses.get(source_hash)
                if existing != "timeout":
                    failed_source_statuses[source_hash] = status

        candidates = [
            *members,
            *getattr(self, "population", ()),
            *getattr(self, "_compiler_seed_members", ()),
            *getattr(self, "_benchmarked_members", {}).values(),
            *getattr(self, "_pinned_finalist_members", {}).values(),
            *(getattr(self, "_terminal_refinement_members", None) or {}).values(),
        ]
        targets: list[tuple[PopulationMember, Literal["error", "timeout"]]] = []
        seen: set[int] = set()
        for member in candidates:
            if id(member) in seen:
                continue
            seen.add(id(member))
            status = failed_member_statuses.get(id(member))
            if status is None:
                status = failed_config_statuses.get(member.config)
            if status is None and failed_source_statuses:
                source_hash = backend.generated_source_hash(member.fn)
                if source_hash is not None:
                    status = failed_source_statuses.get(source_hash)
            if status is None:
                continue
            targets.append((member, status))
        return targets, set(failed_source_statuses)

    def _invalidate_cute_flash_rebenchmark_failures(
        self,
        members: Sequence[PopulationMember],
        failure_statuses: Sequence[Literal["error", "timeout"] | None],
    ) -> list[PopulationMember]:
        targets, source_hashes = self._rebenchmark_failure_targets(
            members, failure_statuses
        )
        if not targets:
            return []
        for source_hash in source_hashes:
            self.benchmark_provider.invalidate_effective_source_hash(source_hash)
        invalidated = []
        invalidated_configs = ConfigSet()
        for member, status in targets:
            member.perfs[:] = [inf]
            member.status = status
            invalidated.append(member)
            invalidated_configs.add(member.config)
        self._invalidate_rebenchmark_training_targets(
            invalidated_configs,
            source_hashes,
        )
        return invalidated

    def _invalidate_rebenchmark_training_targets(
        self,
        failed_configs: MutableSet[Config],
        failed_source_hashes: set[str],
    ) -> None:
        """Let surrogate searches discard targets for quarantined CuTe sources."""
        return None

    def _recompute_cute_flash_best_perf(
        self, members: Sequence[PopulationMember]
    ) -> None:
        candidates = [
            *members,
            *getattr(self, "population", ()),
            *getattr(self, "_compiler_seed_members", ()),
            *getattr(self, "_benchmarked_members", {}).values(),
            *getattr(self, "_pinned_finalist_members", {}).values(),
            *(getattr(self, "_terminal_refinement_members", None) or {}).values(),
        ]
        self.best_perf_so_far = min(
            (
                member.perf
                for member in candidates
                if member.perfs and math.isfinite(member.perf)
            ),
            default=inf,
        )

    def _confirm_suspicious_rebenchmark_timings(
        self,
        members: list[PopulationMember],
        timings: list[float],
        *,
        desc: str,
    ) -> list[IsolatedBenchmarkTiming]:
        updated: list[IsolatedBenchmarkTiming] = list(timings)
        ratio = self.settings.get_suspicious_rebenchmark_ratio()
        if ratio is None or ratio <= 0:
            return updated

        suspicious = [
            i
            for i, (member, timing) in enumerate(zip(members, timings, strict=True))
            if math.isfinite(timing)
            and math.isfinite(member.perf)
            and timing < ratio * member.perf
            and member.perf - timing > _SUSPICIOUS_REBENCHMARK_MIN_DELTA_MS
        ]
        if not suspicious:
            return updated

        try:
            confirmed = self.benchmark_provider.benchmark_isolated(
                [members[i].fn for i in suspicious],
                warmup=_SUSPICIOUS_REBENCHMARK_WARMUP,
                rep=_SUSPICIOUS_REBENCHMARK_REP,
                desc=f"{desc}: confirming suspicious timings",
            )
        except BenchmarkWorkerUnkillable as error:
            self._record_abandoned_batch([members[i] for i in suspicious], error)
            raise
        if confirmed is None:
            return updated

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

        Where the search drops source aliases, each kernel is re-timed once
        however many of ``members`` are configs of it
        (``_rebenchmark_kernels``).

        Args:
            members: The list of population members to rebenchmark.
            desc: Description for the progress bar.
        """
        if members is None:
            members = self.population
        self._rebenchmark_kernels(
            [p for p in members if self.should_rebenchmark(p)], desc=desc
        )

    def _rebenchmark_kernels(
        self,
        members: Sequence[PopulationMember],
        *,
        desc: str = "Rebenchmarking",
        allow_single: bool = False,
    ) -> None:
        """Re-benchmark ``members`` with one timing per kernel, and keep one
        reading per kernel (``rebenchmark`` says what ``allow_single`` is
        for).

        Every re-timing of a search phase comes through here.  A source
        alias is its kernel under another config: the member that holds the
        kernel is timed for it (``_kernel_holder``; its config is the
        kernel's entry in the final selection's history, and only a member
        that is timed refreshes its entry there), once however many of
        ``members`` are configs of it.  A list that is one kernel is not
        timed at all: a re-timing compares kernels, and ``rebenchmark``
        leaves a single member as it is.  Afterwards every other config of a
        kernel, in ``members`` and in the population, has the reading its
        holder has: an alias that a re-timing leaves out (a polish round
        verifies its candidates, ``compare`` a pair) does not keep a reading
        its kernel no longer has.  In a search that keeps its aliases every
        member is timed, as before.
        """
        if not self._collapses_source_aliases():
            self.rebenchmark([*members], desc=desc, allow_single=allow_single)
            return
        self.rebenchmark(
            self._kernels_of(members), desc=desc, allow_single=allow_single
        )
        self._follow_kernel_readings((*members, *self.population), members)

    def _follow_kernel_readings(
        self,
        members: Iterable[PopulationMember],
        among: Sequence[PopulationMember],
    ) -> None:
        """Give each of ``members`` that is a config of another member's
        kernel the reading that kernel has now (``_kernel_holder``, with
        ``among`` at hand), as a new reading of its own."""
        for member in {id(m): m for m in members}.values():
            kernel = self._kernel_holder(member, among)
            if kernel is not member and kernel.perfs and member.perf != kernel.perf:
                member.perfs.append(kernel.perf)

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
            self._selected_member = best
            return best

        self.log(f"Starting finishing phase with {rounds} rounds")
        default_flat = self.config_gen.default_flat()
        current = best

        for round_num in self._budgeted_range(1, rounds + 1):
            simplified = False
            candidates: list[PopulationMember] = [current]

            # Generate candidates by resetting each parameter to its default
            for i in range(len(current.flat_values)):
                if _config_value_key(current.flat_values[i]) != _config_value_key(
                    default_flat[i]
                ):
                    # Create a new config with this parameter reset to default
                    new_flat = [*current.flat_values]
                    new_flat[i] = default_flat[i]
                    candidate = self.make_unbenchmarked(new_flat)
                    # Only add if valid and produces a different config
                    if candidate is not None and _config_key(
                        candidate.config
                    ) != _config_key(current.config):
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

            # Rebenchmark all candidates (including current) for fair
            # comparison. A reset the generated code does not depend on is
            # the incumbent's kernel again: it takes the incumbent's reading
            # instead of a timing of its own, so it counts as a reset that
            # kept the performance.
            self._rebenchmark_kernels(
                candidates, desc=f"Finishing round {round_num}: verifying"
            )

            # Log performance of each candidate at debug level
            current_perf = current.perf
            for candidate in candidates[1:]:
                delta = candidate.perf - current_perf
                delta_pct = (delta / current_perf * 100) if current_perf != 0 else 0
                status = "ok" if candidate.perf <= current_perf else "worse"
                self.log.debug(
                    f"  reset to {candidate.config}: "
                    f"{self.format_performance(candidate.perf)} "
                    f"(delta={delta:+.4f}{self.performance_suffix}, "
                    f"{delta_pct:+.1f}%) [{status}]"
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
                        if _config_value_key(c.flat_values[i]) != _config_value_key(
                            current.flat_values[i]
                        ):
                            combined_flat[i] = c.flat_values[i]
                combined = self.make_unbenchmarked(combined_flat)
                if combined is not None and _config_key(combined.config) != _config_key(
                    current.config
                ):
                    self.benchmark_population(
                        [combined],
                        desc=f"Finishing round {round_num}: combined",
                    )
                    self._rebenchmark_kernels(
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
                    f"Finishing round {round_num}: simplified to {current.config}, "
                    f"perf={self.format_performance(current.perf)}"
                )
            else:
                self.log(
                    f"Finishing round {round_num}: no simplification maintained performance, stopping early"
                )
                break

        # Multi-shape winners must stay explicit: minimizing can collapse an
        # optional tuned reset and a compiler-promoted default to the same key.
        minimal_config = (
            current.config
            if isinstance(self.args, _MultiShapeAutotuneArgs)
            else current.config.minimize(self.config_spec)
        )
        current = PopulationMember(
            fn=current.fn,
            perfs=current.perfs,
            flat_values=self.config_gen.flatten(minimal_config),
            config=minimal_config,
            status=current.status,
            compile_time=current.compile_time,
            timer_readings_from=current.timer_readings_from,
        )
        self._selected_member = current
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
                f"({self.format_performance(best_member.perf)}) over {best.config}"
            )
        self.best_perf_so_far = min(self.best_perf_so_far, best_member.perf)
        return best_member

    def _record_unkillable(
        self, members: list[PopulationMember], error: BenchmarkWorkerUnkillable
    ) -> None:
        """Exclude the config under test when an isolated worker cannot be reaped."""
        if error.fn_index is not None and 0 <= error.fn_index < len(members):
            self._fallback_excluded_configs.add(members[error.fn_index].config)

    def _record_abandoned_batch(
        self,
        members: list[PopulationMember],
        error: BenchmarkWorkerUnkillable,
        *,
        quarantine: bool = True,
    ) -> None:
        """Apply what a re-timing batch over ``members`` found before a
        worker of it could not be reaped. The batch returns nothing, so
        nothing else applies its failures.

        The member under test may have hung that worker and counts as a
        timeout, beside every member that ``error.batch_results`` holds a
        failure for. None of them can win the fallback of ``_finalize``, an
        error and a timeout alike, nor can another config of their kernels
        where the search drops source aliases, and a CuTe flash search
        quarantines them with their config and source aliases, as it does
        when the batch completes (``_exclude_failed_members``). An abandoned
        final selection treats its failed finalists the same way. The joint
        re-timing of a multi-shape search passes ``quarantine=False``: it
        applies no alias policy when it completes, so its failed members
        leave the fallback and keep their readings. Where
        ``torch.distributed`` is initialized only the config under test is
        excluded, and as on that abort no evidence of one rank is promoted.
        That branch is a guard, in step with the final selection's handler:
        no provider starts a benchmark worker in such a process
        (``LocalBenchmarkProvider._subprocess_benchmark_enabled``), so no
        batch ends this way there today.
        """
        if dist.is_initialized():
            self._record_unkillable(members, error)
            return
        failed: list[Literal["error", "timeout"] | None] = [None] * len(members)
        for index, result in enumerate(error.batch_results):
            if isinstance(result, IsolatedBenchmarkFailure):
                failed[index] = result.status
        if error.fn_index is not None and 0 <= error.fn_index < len(members):
            failed[error.fn_index] = "timeout"
        self._exclude_failed_members(members, failed, quarantine=quarantine)

    def _exclude_failed_members(
        self,
        members: Sequence[PopulationMember],
        failed: Sequence[Literal["error", "timeout"] | None],
        *,
        quarantine: bool = True,
    ) -> None:
        """Apply the failures of a batch over ``members`` that ended in a
        worker which could not be reaped: ``failed`` holds each member's
        failure by position, ``None`` where it has none.

        A failed member cannot win the fallback of ``_finalize``. Where the
        search drops source aliases (``_collapses_source_aliases``) neither
        can another config of its kernel: a member of the population with
        the same compiled function is the kernel that failed, and has only
        that kernel's reading. Non-flash readings stay as they are; with
        ``quarantine`` a CuTe flash search sets every config and source
        alias of a failed member to inf, with the failure for its status.
        """
        excluded = [
            member
            for member, status in zip(members, failed, strict=True)
            if status is not None
        ]
        self._fallback_excluded_configs.update(member.config for member in excluded)
        kernels = {id(member.fn) for member in excluded}
        other_configs = [
            member.config
            for member in self.population
            if id(member.fn) in kernels
            and member.config not in self._fallback_excluded_configs
        ]
        if other_configs and self._collapses_source_aliases():
            self._fallback_excluded_configs.update(other_configs)
        if not quarantine:
            return
        _, statuses = self._resolve_isolated_rebenchmark_results(
            members,
            [IsolatedBenchmarkFailure(status) if status else None for status in failed],
        )
        quarantined = [
            (member, status)
            for member, status in zip(members, statuses, strict=True)
            if status is not None
        ]
        if quarantined:
            self._apply_rebenchmark_timings(
                [member for member, _ in quarantined],
                [inf] * len(quarantined),
                failure_statuses=[status for _, status in quarantined],
                trace_results=[
                    IsolatedBenchmarkFailure(status) for _, status in quarantined
                ],
            )

    def _finalize(self) -> Config:
        """Finishing phase, terminal refinement, final selection (or the
        single-pass final verification) and the final-pick re-rank.

        Shared tail of every search's ``_autotune``; the final-pick re-rank
        runs only on TPU/Pallas (see ``_final_pick_supported``).
        """
        best = self.best
        try:
            if self._final_selection_enabled():
                # The finishing phase and the terminal refinement generate
                # candidates and compare them with the search's short windows;
                # their picks join the shortlist of the final selection, which
                # runs last so the returned config is the one that won the
                # harness-grade shootout rather than a later short-window pick.
                # They spend the search phase's share of the budget: the
                # reserve kept for the final selection opens only once they
                # are done.
                best = self.run_finishing_phase(best, self.finishing_rounds)
                best = self.run_terminal_refinement(best)
                best = self.run_compiler_refinement(best)
                self._budget_reserve_released = True
                best = self.final_rebenchmark_best(best)
                self._selected_member = best
            else:
                self._budget_reserve_released = True
                best = self.final_rebenchmark_best(best)
                best = self.run_finishing_phase(best, self.finishing_rounds)
                best = self.run_terminal_refinement(best)
            if self._final_pick_supported():
                best = self.run_final_pick_verification(best)
        except BenchmarkWorkerUnkillable as error:
            # A benchmark worker survived SIGKILL (a hung kernel), so no further
            # worker may be launched and the verification phases cannot run.
            # Keep the best config the search measured instead of discarding
            # the search, except configs already known to have failed.
            live = [
                member
                for member in (best, self.best, *self.population)
                if math.isfinite(member.perf)
                and member.config not in self._fallback_excluded_configs
            ]
            if not live:
                raise
            best = min(live, key=performance)
            self.log.warning(
                f"Final verification abandoned; keeping {best.config} "
                f"({self.format_performance(best.perf)}): {error}"
            )
        self.best = best
        return best.config

    def run_terminal_refinement(self, best: PopulationMember) -> PopulationMember:
        """Run an optional backend/search-specific post-search refinement."""
        return best

    def _compiler_refinement_enabled(self) -> bool:
        return bool(
            self._compiler_refinement_keys
            and self._final_selection_enabled()
            and self._final_rebenchmark_top_k() > 1
            and not self.settings.disable_autotuner_heuristics
            # Rank-local histories need not agree. Do not introduce collective
            # benchmark calls until the parent/candidate plan is synchronized.
            and self.kernel.env.process_group_name is None
        )

    def run_compiler_refinement(self, best: PopulationMember) -> PopulationMember:
        """Cross compiler options with the best measured structures, then verify.

        This phase spends the search budget; it never opens the final-selection
        reserve or records per-candidate checks as population generations. Its
        short-window winner still enters the ordinary shootout.
        """
        if not self._compiler_refinement_enabled():
            return best
        registry = self._terminal_refinement_members
        assert registry is not None

        def reusable(member: PopulationMember) -> bool:
            return (
                member.status in {"ok", "deduplicated"}
                and bool(member.perfs)
                and math.isfinite(member.perf)
                and member.config not in self._fallback_excluded_configs
            )

        def prior_outcome(
            config: Config,
        ) -> tuple[PopulationMember | None, str | None]:
            """Check prior results and current admission limits for one trial."""
            member = registry.get(config)
            if member is not None:
                return member, "reused" if reusable(member) else "prior_failure"
            if config in self._fallback_excluded_configs:
                return None, "quarantined"
            if self._autotune_budget_exceeded():
                return None, "budget_exhausted"
            if not self._backend_config_is_viable(config):
                return None, "invalid"
            return None, None

        trials: list[dict[str, object]] = []
        transcript: dict[str, object] = {
            **compiler_refinement_policy(self._compiler_refinement_keys),
            "initial_config_id": canonical_config_id(best.config),
            "trials": trials,
            "new_trials": 0,
            "budget_exhausted": False,
            "completed": False,
        }
        self._autotune_metrics.compiler_refinement = transcript
        try:
            history = sorted(
                (member for member in registry.values() if reusable(member)),
                key=performance,
            )
            # A lucky earlier sample must not displace the actual incumbent.
            # Rank other parents by the same latest timing used below to pick.
            parents = [best, *history] if reusable(best) else history
            candidates = compiler_refinement_candidates(
                self.config_gen,
                [member.config for member in parents],
                self._compiler_refinement_keys,
            )
            new_trials = 0
            for candidate in candidates:
                trial: dict[str, object] = {
                    "parent_config_id": canonical_config_id(candidate.parent),
                    "config_id": canonical_config_id(candidate.config),
                    "config": _copy_config_structure(candidate.config.config),
                }
                trials.append(trial)
                member, outcome = prior_outcome(candidate.config)
                if outcome is not None:
                    trial["outcome"] = outcome
                    if outcome == "budget_exhausted":
                        transcript["budget_exhausted"] = True
                    if member is None:
                        continue
                elif new_trials >= MAX_REFINEMENT_TRIALS:
                    trial["outcome"] = "trial_limit"
                    continue
                elif self._autotune_budget_exceeded():
                    trial["outcome"] = "budget_exhausted"
                    transcript["budget_exhausted"] = True
                    continue
                else:
                    # Admit one trial at a time: its compile/measurement spends
                    # the search budget before the next admission check, and
                    # an unkillable worker must stop all subsequent work.
                    member = self.make_unbenchmarked(candidate.flat_values)
                    if member is None or not same_refinement_value(
                        member.config.config, candidate.config.config
                    ):
                        trial["outcome"] = "invalid"
                        continue
                    if self._autotune_budget_exceeded():
                        trial["outcome"] = "budget_exhausted"
                        transcript["budget_exhausted"] = True
                        continue
                    new_trials += 1
                    transcript["new_trials"] = new_trials
                    trial["outcome"] = "benchmarking"
                    try:
                        self.benchmark_population(
                            [member],
                            desc="Compiler-option refinement",
                            raise_if_no_viable_config=False,
                        )
                    except BenchmarkWorkerUnkillable as error:
                        self._record_unkillable([member], error)
                        trial["outcome"] = "unkillable_worker"
                        raise
                    trial["outcome"] = member.status
                    if self._is_source_alias(member):
                        # Code an earlier candidate already produced: no
                        # trial of a new kernel, and its reading is that
                        # candidate's, an earlier sample that must not
                        # displace the incumbent.
                        new_trials -= 1
                        transcript["new_trials"] = new_trials
                        continue
                if reusable(member):
                    trial["perf"] = member.perf
                    if member.perf < best.perf:
                        best = member
            transcript["completed"] = True
            transcript["selected_config_id"] = canonical_config_id(best.config)
            return best
        finally:
            self.log.record_compiler_refinement(transcript)

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
        if (
            isinstance(
                getattr(self, "benchmark_provider", None),
                MultiShapeBenchmarkProvider,
            )
            or settings is None
            or config_spec is None
            or not settings.static_shapes
        ):
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

        # Every delta is measured against a fresh run of ``best``. Sampling
        # noise can make the incumbent's own pair nonzero, so the smallest
        # positive delta may belong to another candidate. Keep the incumbent
        # unless at least one paired measurement is actually faster.
        if best_member is not best and delta_by_slot[best_slot] >= 0:
            return best

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
        elif status not in {"error", "timeout", "skipped"}:
            # "skipped": never attempted (the autotune budget ran out).
            status = "error"
        status_counts[status] += 1
    if len(working) == 0:
        raise exc.NoConfigFound
    count_parts = [
        f"{label}={status_counts[label]}"
        for label in ("error", "timeout", "skipped", "ok")
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
