from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import datetime
import enum
import functools
from itertools import count
from itertools import starmap
import logging
import math
from math import inf
import os
from pathlib import Path
import pprint
import random
import re
import sys
import tempfile
import time
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Literal
from typing import NamedTuple
from typing import NoReturn
from typing import Protocol
from typing import cast
from unittest.mock import patch
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map_only
from torch.utils._pytree import tree_unflatten

from .. import exc
from .._compat import extract_device
from .._compat import get_device_name
from ..runtime.config import Config
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import already_compiled_fail
from ..runtime.precompile_shim import make_precompiler
from ..runtime.settings import is_pallas_interpret
from .benchmarking import do_bench
from .benchmarking import interleaved_bench
from .logger import SUPPRESSED_TRITON_CODE_MSG
from .logger import AutotuneLogEntry
from .logger import AutotuningLogger
from .logger import _get_failure_dump_dir
from .logger import capture_output
from .logger import classify_triton_exception
from .logger import format_triton_compile_failure
from .logger import log_generated_triton_code_debug
from .logger import match_unrecoverable_runtime_error
from .logger import maybe_dump_triton_failure
from .metrics import AutotuneMetrics
from .metrics import _run_post_autotune_hooks
from .precompile_future import PrecompileFuture as PrecompileFuture
from .precompile_future import _ExtractedLaunchArgs
from .progress_bar import iter_with_progress
from .utils import safe_globals
from helion._dist_utils import _clone_symm_mem_tensor
from helion._dist_utils import all_gather_object
from helion._dist_utils import get_signal_pad_ptrs_dev
from helion._dist_utils import is_master_rank
from helion._dist_utils import is_symm_mem_tensor
from helion._dist_utils import sync_object

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import CompiledConfig
    from ..runtime.settings import Settings
    from . import ConfigSpec
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig
    from .local_cache import SavedBestConfig
    from helion.autotuner.effort_profile import AutotuneEffortProfile


def _synchronize_device() -> None:
    """Synchronize the accelerator unless running in Pallas interpret mode on CPU."""
    if not is_pallas_interpret() or torch.accelerator.is_available():
        torch.accelerator.synchronize()


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


class BenchmarkResult(NamedTuple):
    """Result tuple returned by parallel_benchmark."""

    config: Config
    fn: Callable[..., object]
    perf: float
    status: Literal["ok", "error", "timeout", "peer_compilation_fail"]
    compile_time: float | None


_FP8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
}


def _assert_close(actual: object, expected: object, atol: float, rtol: float) -> None:
    """Like torch.testing.assert_close but handles fp8 and uses chunked comparison for large tensors."""

    def convert(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.uint8) if t.dtype in _FP8_DTYPES else t

    actual_flat, actual_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, actual)
    )
    expected_flat, expected_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, expected)
    )

    if actual_spec != expected_spec:
        raise AssertionError(
            f"Output tree structure mismatch during autotuner accuracy check:\n"
            f"  actual:   {actual_spec} ({len(actual_flat)} leaves)\n"
            f"  expected: {expected_spec} ({len(expected_flat)} leaves)"
        )

    for a, e in zip(actual_flat, expected_flat, strict=True):
        if isinstance(a, torch.Tensor):
            _chunked_assert_close(a, e, atol=atol, rtol=rtol)
        elif isinstance(a, str):
            if not isinstance(e, str):
                raise AssertionError(f"Type mismatch {a} vs {e}")
            if a != e:
                raise AssertionError(f"string mismatch {a} vs {e}")
        else:
            torch.testing.assert_close(a, e, atol=atol, rtol=rtol)


def _chunked_assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    chunk_size: int = 2**22,  # ~4M elements per chunk
) -> None:
    """Memory-efficient assert_close for large tensors.

    Processes the comparison in chunks to avoid allocating multiple
    full-size temporary tensors.  Uses torch.testing.assert_close on
    each chunk so error messages retain full detail.
    """
    if actual.numel() <= chunk_size:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    a_flat = actual.reshape(-1)
    e_flat = expected.reshape(-1)
    for i in range(0, a_flat.numel(), chunk_size):
        a_chunk = a_flat[i : i + chunk_size]
        e_chunk = e_flat[i : i + chunk_size]
        torch.testing.assert_close(a_chunk, e_chunk, atol=atol, rtol=rtol)


def _clone_args(
    args: Sequence[object],
    process_group_name: str | None,
    idx_to_clone: Sequence[int] | None = None,
) -> Sequence[object]:
    """
    Clone the given arguments, but cloning only the tensors specified by
      idx_to_clone. If idx_to_clone is None, clone all tensors.
    """

    def _should_clone(idx: int) -> bool:
        return idx_to_clone is None or idx in idx_to_clone

    args_flat, tree_spec = tree_flatten(args)
    old_arg_to_new_arg = {}

    for i, arg in enumerate(args_flat):
        if _should_clone(i) and is_symm_mem_tensor(arg, process_group_name):
            new_arg = _clone_symm_mem_tensor(arg, process_group_name)
            old_arg_to_new_arg[get_signal_pad_ptrs_dev(arg, process_group_name)] = (
                get_signal_pad_ptrs_dev(new_arg, process_group_name)
            )
            old_arg_to_new_arg[arg] = new_arg  # pyrefly: ignore[unsupported-operation]

    for i, arg in enumerate(args_flat):
        if arg in old_arg_to_new_arg:
            args_flat[i] = old_arg_to_new_arg[arg]
            continue
        if not isinstance(arg, torch.Tensor):
            continue
        if _should_clone(i):
            clone = arg.detach().clone()
            clone.requires_grad_(arg.requires_grad)
            args_flat[i] = clone

    return tree_unflatten(args_flat, tree_spec)


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

    # Instance attributes intentionally NOT checkpointed (transient/derived).
    # Everything in __dict__ not listed here is auto-serialized by state_dict().
    # Tests declare the expected set of auto-serialized keys per algorithm;
    # adding a new attr without updating either _checkpoint_exclude or the test
    # will cause a test failure, forcing an explicit decision.
    _checkpoint_exclude: ClassVar[tuple[str, ...]] = (
        "kernel",
        "args",
        "settings",
        "config_spec",
        "log",
        "_original_args",
        "_precompile_tmpdir",
        "_precompile_args_path",
        "_precompile_result_counter",
        "_baseline_output",
        "_mutated_arg_indices",
        "_baseline_post_args",
        "_effective_atol",
        "_effective_rtol",
        "_jobs",
        "_prepared",
        "_skip_cache",
        "_autotune_metrics",
        "_stable_hash",
        "_crashed_config_strs",
    )

    @classmethod
    def _collect_checkpoint_exclude(cls) -> set[str]:
        """Collect all excluded attrs by walking MRO."""
        excluded: set[str] = set()
        for klass in cls.__mro__:
            excluded.update(getattr(klass, "_checkpoint_exclude", ()))
        return excluded

    _baseline_output: object
    _mutated_arg_indices: Sequence[int] = []
    _baseline_post_args: Sequence[object] | None
    _jobs: int
    _precompile_result_counter: count[int]
    _effective_atol: float
    _effective_rtol: float

    def __init__(self, kernel: _AutotunableKernel, args: Sequence[object]) -> None:
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
        self._current_generation = 0
        self.counters: collections.Counter[str] = collections.Counter()
        self._autotune_metrics: AutotuneMetrics = AutotuneMetrics(
            kernel_name="",
            input_shapes="",
            hardware="",
            random_seed=0,
            search_algorithm=type(self).__name__,
        )
        self._prepared = False
        self._precompile_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._precompile_args_path: str | None = None
        self._precompile_result_counter = count()
        self._crashed_config_strs: set[str] = set()

    def _prepare(self) -> None:
        """Some initialization deferred until autotuning actually runs.

        This is called at the start of autotune() so that cache hits skip it.
        """
        if self._prepared:
            return
        self._prepared = True
        seed = self.settings.autotune_random_seed
        random.seed(seed)
        self.log(f"Autotune random seed: {seed}")
        self._autotune_metrics: AutotuneMetrics = AutotuneMetrics(
            kernel_name=getattr(getattr(self.kernel, "kernel", None), "name", ""),
            input_shapes=str(
                [tuple(arg.shape) for arg in self.args if isinstance(arg, torch.Tensor)]
            ),
            hardware=get_device_name(extract_device(self.args)) or "",
            random_seed=self.settings.autotune_random_seed,
            search_algorithm=type(self).__name__,
        )
        (
            self._baseline_output,
            self._mutated_arg_indices,
            self._baseline_post_args,
        ) = self._compute_baseline()
        self._effective_atol, self._effective_rtol = (
            self._compute_effective_tolerances()
        )
        self._jobs = self._decide_num_jobs()

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

    def _next_precompile_result_path(self) -> str:
        assert self._precompile_tmpdir is not None
        return os.path.join(
            self._precompile_tmpdir.name,
            f"result_{next(self._precompile_result_counter)}.pkl",
        )

    def cleanup(self) -> None:
        if self._precompile_tmpdir is not None:
            self._precompile_tmpdir.cleanup()
            self._precompile_tmpdir = None
        self._precompile_args_path = None
        self._precompile_result_counter = count()

    _stable_hash: str | None = None

    def _get_stable_hash(self) -> str:
        """Get the full stable hash for this kernel's cache key (cached)."""
        if self._stable_hash is None:
            from .local_cache import LocalAutotuneCache

            self._stable_hash = LocalAutotuneCache(self)._generate_key().stable_hash()
        return self._stable_hash

    def _try_load_checkpoint(self) -> bool:
        """Attempt to load checkpoint from checkpoint dir. Returns True if successful."""
        checkpoint_dir_str = self.settings.autotune_checkpoint_dir
        if checkpoint_dir_str is None:
            return False

        checkpoint_dir = Path(checkpoint_dir_str)
        stable_hash = self._get_stable_hash()
        checkpoint_file = checkpoint_dir / f"{stable_hash}.pt"

        if not checkpoint_file.exists():
            return False  # No matching checkpoint; start fresh

        # Matching file exists, attempt to load
        self.log(f"Resuming from checkpoint: {checkpoint_file}")
        try:
            with safe_globals(Config, OrderedSet):
                state = torch.load(checkpoint_file, weights_only=True)
        except Exception as e:
            raise exc.CheckpointError(
                f"Failed to load checkpoint file '{checkpoint_file}': {e}\n"
                f"The file may be corrupted. Delete it to start fresh."
            ) from e

        # load_state_dict validates required keys and raises CheckpointError for issues
        self.load_state_dict(state)

        self.log(f"Resumed at generation {self._current_generation}")
        return True

    def _load_crashed_configs(self) -> None:
        """Load crashed configs from {hash}.crashed_configs (written by crash-recovery script)."""
        checkpoint_dir_str = self.settings.autotune_checkpoint_dir
        if checkpoint_dir_str is None:
            return
        crashed_configs_path = (
            Path(checkpoint_dir_str) / f"{self._get_stable_hash()}.crashed_configs"
        )
        if crashed_configs_path.exists():
            self._crashed_config_strs |= {
                line.strip()
                for line in crashed_configs_path.read_text().splitlines()
                if line.strip()
            }
        if self._crashed_config_strs:
            self.log(
                f"Loaded {len(self._crashed_config_strs)} crashed config(s) to skip"
            )

    def _get_pending_config_path(self) -> Path | None:
        """Get path for pending-config sentinel, or None if checkpointing disabled."""
        checkpoint_dir_str = self.settings.autotune_checkpoint_dir
        if checkpoint_dir_str is None:
            return None
        return Path(checkpoint_dir_str) / f"{self._get_stable_hash()}.pending_config"

    def _compute_baseline(
        self,
    ) -> tuple[object, Sequence[int], Sequence[object] | None]:
        """
        Compute baseline output for accuracy validation during autotuning.
        Also detect if the kernel mutates any of its input arguments.

        The baseline is computed in one of two ways:
        - If settings.autotune_baseline_fn is provided, use that custom function
        - Otherwise, run the kernel with the default config
        """
        new_args = _clone_args(self.args, self.kernel.env.process_group_name)

        # Use custom baseline function if provided
        if self.settings.autotune_baseline_fn is not None:
            try:
                baseline_output = self.settings.autotune_baseline_fn(*new_args)
                _synchronize_device()
            except Exception as e:
                raise exc.AutotuneError(
                    "Custom baseline function failed while computing baseline.\n"
                    f"Baseline function: {self.settings.autotune_baseline_fn}\n"
                ) from e
        else:
            # Use default config
            baseline_config = self.config_spec.default_config()
            try:
                baseline_output = self.kernel.compile_config(
                    baseline_config, allow_print=False
                )(*new_args)
                _synchronize_device()
            except Exception as e:
                decorator = self.kernel.format_kernel_decorator(
                    baseline_config, self.settings
                )
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    baseline_config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.kernel.maybe_log_repro(self.log.error, new_args, baseline_config)
                raise exc.InvalidConfig(
                    "Default config failed while computing baseline.\n"
                    f"Default config: {decorator}\n"
                    f"{SUPPRESSED_TRITON_CODE_MSG}\n"
                    "To work around this error, you could set `@helion.kernel(autotune_baseline_fn=...)` "
                    "to provide a custom baseline function (e.g. PyTorch eager implementation of your kernel)."
                ) from e

        original_args_flat, _ = tree_flatten(self.args)
        new_args_flat, _ = tree_flatten(new_args)
        mutated_tensor_idxs = []
        # we should only count tensors, since they won't be bound or removed
        tensor_idx = 0
        for old, new in zip(original_args_flat, new_args_flat, strict=False):
            if not (isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor)):
                continue
            try:
                equal = torch.equal(new, old)
            except RuntimeError:
                # torch.equal and device-to-host copies can fail on some
                # devices (e.g., TPU for large tensors).  Conservatively
                # assume the argument was not mutated.
                equal = True
            if not equal:
                mutated_tensor_idxs.append(tensor_idx)
            tensor_idx += 1
        baseline_post_args = _clone_args(
            new_args,
            self.kernel.env.process_group_name,
            idx_to_clone=mutated_tensor_idxs,
        )
        return baseline_output, mutated_tensor_idxs, baseline_post_args

    def _compute_effective_tolerances(self) -> tuple[float, float]:
        """
        Compute effective tolerances based on the dtypes in the baseline output.

        For low-precision dtypes (fp8), we need stricter tolerances to ensure
        bitwise comparison works correctly. This method automatically detects
        such dtypes and adjusts tolerances accordingly.

        Returns:
            A tuple of (atol, rtol) to use for accuracy validation.
        """
        # Default tolerance when not user-specified
        DEFAULT_TOL = 1e-2

        # Get user-specified or default tolerances
        atol = self.settings.autotune_baseline_atol
        rtol = self.settings.autotune_baseline_rtol

        # Collect all dtypes from baseline output and mutated args
        dtypes = set()

        def collect_dtypes(obj: object) -> object:
            if isinstance(obj, torch.Tensor):
                dtypes.add(obj.dtype)
            return obj

        tree_map_only(torch.Tensor, collect_dtypes, self._baseline_output)
        if len(self._mutated_arg_indices) > 0 and self._baseline_post_args is not None:
            tree_map_only(torch.Tensor, collect_dtypes, self._baseline_post_args)

        # Only apply strict tolerances if ALL dtypes are fp8
        # Mixed dtypes (fp8 + fp32) would be too strict with atol=0.0, rtol=0.0
        all_dtypes_are_fp8 = dtypes and all(dtype in _FP8_DTYPES for dtype in dtypes)

        if all_dtypes_are_fp8:
            # All dtypes are fp8 - use bitwise comparison
            # unless the user explicitly set either tolerance value (i.e., not None)
            if atol is None and rtol is None:
                self.log(
                    f"Detected fp8 dtype(s) in output: {dtypes}. "
                    "Using bitwise comparison (atol=0.0, rtol=0.0) for autotuning accuracy check."
                )
                return 0.0, 0.0

        # Use user-specified values or defaults
        return (
            atol if atol is not None else DEFAULT_TOL,
            rtol if rtol is not None else DEFAULT_TOL,
        )

    def _decide_num_jobs(self) -> int:
        if not self.settings.autotune_precompile:
            return 1

        jobs = self.settings.autotune_precompile_jobs
        if not jobs:
            jobs = os.cpu_count() or 1

        if self.settings.autotune_precompile != "spawn":
            return jobs

        memory_per_job = _estimate_tree_bytes(self.args) + _estimate_tree_bytes(
            self._baseline_output
        )
        memory_per_job *= 2  # safety factor
        if memory_per_job <= 0:
            return jobs

        device = self.kernel.env.device
        if device.type != "cuda":
            # TODO(jansel): support non-cuda devices
            return jobs

        available_memory, _ = torch.cuda.mem_get_info(device)
        jobs_by_memory = available_memory // memory_per_job
        if jobs_by_memory < jobs:
            gib_per_job = memory_per_job / (1024**3)
            available_gib = available_memory / (1024**3)
            if jobs_by_memory > 0:
                self.log.warning(
                    f"Reducing autotune precompile spawn jobs from {jobs} to {jobs_by_memory} "
                    f"due to limited GPU memory (estimated {gib_per_job:.2f} GiB per job, "
                    f"{available_gib:.2f} GiB free). "
                    f"Set HELION_AUTOTUNE_PRECOMPILE_JOBS={jobs_by_memory} "
                    "to make this lower cap persistent, "
                    'set HELION_AUTOTUNE_PRECOMPILE="fork" to disable spawning, or reduce GPU memory usage.'
                )
            else:
                raise exc.AutotuneError(
                    "Autotune precompile spawn mode requires at least one job, but estimated "
                    "memory usage exceeds available GPU memory."
                    f"Estimated {gib_per_job:.2f} GiB per job, but only "
                    f"{available_gib:.2f} GiB free. "
                    'Set HELION_AUTOTUNE_PRECOMPILE="fork" to disable spawning, or reduce GPU memory usage.'
                )
            jobs = jobs_by_memory

        return jobs

    def _validate_against_baseline(
        self, config: Config, output: object, args: Sequence[object]
    ) -> bool:
        try:
            custom_check = self.settings.autotune_baseline_accuracy_check_fn
            if custom_check is not None:
                custom_check(output, self._baseline_output)
                if len(self._mutated_arg_indices) > 0:
                    custom_check(args, self._baseline_post_args)
            else:
                _assert_close(
                    output,
                    self._baseline_output,
                    atol=self._effective_atol,
                    rtol=self._effective_rtol,
                )
                if os.getenv("CHECK_INPUT_ACCURACY", "1") == "1":
                    if len(self._mutated_arg_indices) > 0:
                        # For distributed kernel, group_name may also be a argument.
                        # torch.testing.assert_close does not handle str argument.
                        # Filter needed.
                        assert self._baseline_post_args is not None
                        _assert_close(
                            args,
                            self._baseline_post_args,
                            atol=self._effective_atol,
                            rtol=self._effective_rtol,
                        )
        except AssertionError as e:
            if not self.settings.autotune_ignore_errors:
                self.log.warning(
                    f"Skipping config with accuracy mismatch: {config!r}\n{e!s}\nUse HELION_AUTOTUNE_ACCURACY_CHECK=0 to disable this check.\n"
                )
            return False
        return True

    def benchmark_function(self, config: Config, fn: CompiledConfig) -> float:
        """
        Benchmark a compiled function.  This function is called by the autotuner to measure the
        performance of a specific configuration.

        Args:
            config: The configuration to benchmark.
            fn: A precompiled version of config.

        Returns:
            The performance of the configuration in ms.
        """
        # Skip configs that previously crashed the subprocess
        config_str = str(config)
        if config_str in self._crashed_config_strs:
            self.log.warning(f"Skipping known-crashed config: {config}")
            return inf

        self._autotune_metrics.num_configs_tested += 1
        self.counters["benchmark"] += 1
        self.log.debug(lambda: f"Running benchmark for {config!r}")
        _captured_output: list[str] = [""]
        _capture_ctx = (
            capture_output()
            if _get_failure_dump_dir()
            else contextlib.nullcontext(_captured_output)
        )

        if len(self._mutated_arg_indices) > 0:
            working_args = _clone_args(
                self.args,
                self.kernel.env.process_group_name,
                idx_to_clone=self._mutated_arg_indices,
            )
        else:
            working_args = self.args

        # precompile in the current process for distributed kernels.
        # The reason we need this is due to some tricky distributed kernels
        # like https://gist.github.com/shunting314/81f13ce00f835b21ab6466e21454b7c5 . We specialize the RANK argument for each GPU,
        # some rank may get out of resource errors while others don't
        # due to the specialization.
        #
        # Without precompilation here, some rank may fail and skip running
        # the kernel while outer ranks waiting for its peers. It
        # results in a stuck job.
        #
        # Precompiilation happening in child process is not enough because
        # CUDA is not available there. We can not check resource usage
        # like shared-memory, tmem, max-threads etc.
        #
        # This precompilation has overhead. Only do it if distributed is
        # initialized.

        if dist.is_initialized():
            # Trigger Triton JIT compilation before running the kernel
            compile_success = _triton_compile(fn, working_args, config, self.kernel)
            compile_success_all = all(
                all_gather_object(
                    compile_success,
                    process_group_name=self.kernel.env.process_group_name,
                )
            )

            if not compile_success_all:
                return inf

        try:
            # TODO(jansel): early exit with fewer trials if early runs are slow
            self.log.debug(lambda: f"Running {config} at {datetime.datetime.now()}")
            t0 = time.perf_counter()
            _synchronize_device()

            with _capture_ctx as _captured_output:
                output = fn(*working_args)  # make sure the kernel is compiled

            _synchronize_device()

            pass_accuracy_check = (
                not self.settings.autotune_accuracy_check
                or self._validate_against_baseline(config, output, working_args)
            )
            if not pass_accuracy_check:
                self._autotune_metrics.num_accuracy_failures += 1
            if not all(
                all_gather_object(
                    pass_accuracy_check,
                    process_group_name=self.kernel.env.process_group_name,
                )
            ):
                # for distributed kernels like matmul-reduce-scatter, different ranks compute
                # a different chunk. It's possible that some ranks pass the accuracy check while
                # others don't. Skip the config if any rank fails the accuracy check.
                # Without this synchronization, some ranks go on to call the benchmark function
                # while other ranks return immediately, this will cause stuck jobs!
                return inf

            bench_fn = self.kernel.bench_compile_config(config, allow_print=False)
            bench_fn(*working_args)  # warmup benchmark kernel

            t1 = time.perf_counter()
            _backend = getattr(getattr(self, "config_spec", None), "backend", None)
            _bench_fn = (
                _backend.get_do_bench() if _backend is not None else None
            ) or do_bench
            res = _bench_fn(
                functools.partial(bench_fn, *working_args),
                return_mode="median",
                warmup=1,  # we are already warmed up above
                rep=50,
                process_group_name=self.kernel.env.process_group_name,
            )
            res = sync_object(
                res, process_group_name=self.kernel.env.process_group_name
            )
            t2 = time.perf_counter()
            assert isinstance(res, float)

            self.log.debug(
                lambda: f"result: {res:.4f}ms (took {t1 - t0:.1f}s + {t2 - t1:.1f}s)",
            )
            if res < self.best_perf_so_far:
                self.best_perf_so_far = res
            return res
        except Exception as e:
            # e.__traceback__ holds references to all local variables in the call stack frames.
            # When a Triton kernel fails, the output tensors allocated by the Helion kernel function
            # were being held by the traceback, preventing them from being freed.
            e.__traceback__ = None
            maybe_dump_triton_failure(
                self.kernel,
                config,
                e,
                captured_output=_captured_output[0] or None,
            )
            if match_unrecoverable_runtime_error(e):
                self.kernel.maybe_log_repro(self.log.error, self.args, config)
                raise exc.TritonUnrecoverableRuntimeError(
                    reason=str(e),
                    decorator=self.kernel.format_kernel_decorator(
                        config, self.settings
                    ),
                    error=f"{type(e).__qualname__}: {e}",
                ) from e
            _backend = getattr(getattr(self, "config_spec", None), "backend", None)
            action = (
                _backend.classify_autotune_exception(e)
                if _backend is not None
                else None
            ) or classify_triton_exception(e)
            if self.settings.autotune_ignore_errors:
                pass
            elif action == "raise":
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.kernel.maybe_log_repro(self.log.error, self.args, config)
                raise exc.TritonError(
                    error=f"{type(e).__qualname__}: {e}",
                    decorator=decorator,
                    code=SUPPRESSED_TRITON_CODE_MSG,
                ) from e
            elif action == "warn":
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.log.warning(format_triton_compile_failure(config, e, self.kernel))
                self.kernel.maybe_log_repro(self.log.warning, self.args, config)
            else:
                decorator = self.kernel.format_kernel_decorator(config, self.settings)
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.log.debug(f"Benchmarking failed: {type(e).__name__}: {e}")
                self.kernel.maybe_log_repro(self.log.debug, self.args, config)

            self._autotune_metrics.num_compile_failures += 1
            return inf

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

    def create_precompile_future(
        self, config: Config, fn: CompiledConfig
    ) -> PrecompileFuture:
        """
        Create a subprocess that will precompile the kernel to detect hangs
        during compilation.  The subprocess is not started until the returned
        future is called or started explicitly.

        Args:
            config: The config that generated fn.
            fn: The function to be precompiled.

        Returns:
            A ``PrecompileFuture`` that resolves to True on success or False on
            failure/timeout when called.
        """
        if not self.settings.autotune_precompile:
            return PrecompileFuture.skip(self, config, True)
        mode = self.settings.autotune_precompile
        if mode not in {"fork", "spawn"}:
            raise exc.InvalidAPIUsage("autotune_precompile must be 'fork' or 'spawn'")
        if len(self._mutated_arg_indices) > 0:
            args = _clone_args(
                self.args,
                self.kernel.env.process_group_name,
                idx_to_clone=self._mutated_arg_indices,
            )
        else:
            args = self.args

        return PrecompileFuture.create(
            search=self,
            config=config,
            fn=fn,
            args=args,
            result_path=self._next_precompile_result_path(),
            args_path=self._precompile_args_path,
        )

    def _benchmark(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        """
        Internal benchmark implementation. Compiles in parallel and benchmarks configs.

        Args:
            configs: A list of configurations to benchmark.
            desc: Description for the progress bar.

        Returns:
            A list of BenchmarkResult entries containing the configuration, compiled
            callable, measured performance, status, and compilation time.
        """
        # Filter out known-crashed configs before compilation
        if self._crashed_config_strs:
            original_len = len(configs)
            configs = [c for c in configs if str(c) not in self._crashed_config_strs]
            skipped = original_len - len(configs)
            if skipped:
                self.log.warning(
                    f"Skipped {skipped} known-crashed config(s) before compilation"
                )
            if not configs:
                return []

        fns: list[Callable[..., object]] = []
        valid_configs: list[Config] = []
        futures: list[PrecompileFuture] | None = None
        pending_path = self._get_pending_config_path()
        for i, config in enumerate(configs):
            # Write sentinel before compile so a hard crash (SIGKILL /
            # CUDA IMA) leaves a trace the crash recovery script can find.
            if pending_path is not None:
                pending_path.write_text(str(config))
            try:
                fn = self.kernel.compile_config(config, allow_print=False)
            except Exception as e:
                if match_unrecoverable_runtime_error(e):
                    # Leave sentinel for crash recovery — CUDA context is
                    # corrupted and the process cannot continue.
                    raise
                if pending_path is not None:
                    pending_path.unlink(missing_ok=True)
                # If all configs failed, raise error
                if not valid_configs and i == len(configs) - 1:
                    raise
                self.log.warning(
                    "Skipping config that failed to compile: %s",
                    self.kernel.format_kernel_decorator(config, self.settings),
                    exc_info=True,
                )
                continue
            if pending_path is not None:
                pending_path.unlink(missing_ok=True)
            fns.append(fn)
            valid_configs.append(config)
        configs = valid_configs
        # NOTE: precompile runs in separate subprocesses with isolated CUDA
        # contexts; crashes there are caught via is_working checks, not
        # sentinels.
        if self.settings.autotune_precompile:
            futures = list(
                starmap(
                    self.create_precompile_future,
                    zip(configs, fns, strict=True),
                )
            )
            precompile_desc = (
                f"{desc} precompiling" if self.settings.autotune_progress_bar else None
            )
            is_workings = PrecompileFuture.wait_for_all(futures, desc=precompile_desc)
            precompile_status: list[Literal["ok", "error", "timeout"]] = []
            for future, ok in zip(futures, is_workings, strict=True):
                reason = future.failure_reason
                if ok:
                    precompile_status.append("ok")
                elif reason == "timeout":
                    precompile_status.append("timeout")
                else:
                    precompile_status.append("error")
        else:
            is_workings = [True] * len(configs)
            precompile_status = ["ok"] * len(configs)

        results: list[BenchmarkResult] = []

        # Render a progress bar only when the user requested it.
        iterator = iter_with_progress(
            enumerate(zip(fns, is_workings, precompile_status, strict=True)),
            total=len(configs),
            description=f"{desc} exploring neighbors",
            enabled=self.settings.autotune_progress_bar,
        )
        for index, (fn, is_working, reason) in iterator:
            config = configs[index]
            if futures is not None:
                future = futures[index]
                compile_time = (
                    future.elapsed
                    if future.process is not None and future.started
                    else None
                )
            else:
                compile_time = None
            status: Literal["ok", "error", "timeout", "peer_compilation_fail"]
            if all(
                all_gather_object(
                    is_working, process_group_name=self.kernel.env.process_group_name
                )
            ):
                # Log started before benchmarking to help identify hangs
                self.log.record_autotune_entry(
                    AutotuneLogEntry(
                        generation=self._autotune_metrics.num_generations,
                        status="started",
                        perf_ms=None,
                        compile_time=compile_time,
                        config=config,
                    )
                )
                # benchmark one-by-one to avoid noisy results
                # Write pending-config sentinel; cleared after benchmark.
                # On crash the file stays so the crash recovery script can
                # detect which config caused the failure.
                if pending_path is not None:
                    pending_path.write_text(str(config))
                perf = self.benchmark_function(config, fn)
                if pending_path is not None:
                    pending_path.unlink(missing_ok=True)
                status = "ok" if math.isfinite(perf) else "error"
                # Log completion after benchmarking
                self.log.record_autotune_entry(
                    AutotuneLogEntry(
                        generation=self._autotune_metrics.num_generations,
                        status=status,
                        perf_ms=perf if math.isfinite(perf) else None,
                        compile_time=compile_time,
                        config=config,
                    )
                )
                results.append(
                    BenchmarkResult(
                        config=config,
                        fn=fn,
                        perf=perf,
                        status=status,
                        compile_time=compile_time,
                    )
                )
            else:
                status = "timeout" if reason == "timeout" else "error"
                if is_working:
                    status = "peer_compilation_fail"
                results.append(
                    BenchmarkResult(
                        config=config,
                        fn=fn,
                        perf=inf,
                        status=status,
                        compile_time=compile_time,
                    )
                )
        return results

    def benchmark_batch(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        """
        Compile and benchmark a batch of configurations.

        This is the primary entry point for benchmarking. It compiles and
        benchmarks the given configs, then updates search-level metrics
        (configs tested, failures, best performance).

        Args:
            configs: A list of configurations to benchmark.
            desc: Description for the progress bar.

        Returns:
            A list of BenchmarkResult entries.
        """
        return self._benchmark(configs, desc=desc)

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
                exit_stack.enter_context(self.log.autotune_logging())
            self.log.reset()
            # Autotuner triggers bugs in remote triton compile service.
            # Skip storing Triton intermediate IRs (.ttir, .ttgir, .llir, etc.)
            # during autotuning to reduce cache size by ~40%. Only binaries and
            # metadata are needed for execution.
            env_overrides = {"TRITON_LOCAL_BUILD": "1"}
            if "TRITON_STORE_BINARY_ONLY" not in os.environ:
                env_overrides["TRITON_STORE_BINARY_ONLY"] = "1"
            exit_stack.enter_context(patch.dict(os.environ, env_overrides, clear=False))
            assert self._precompile_tmpdir is None
            tempdir = tempfile.TemporaryDirectory()
            self._precompile_tmpdir = tempdir
            if self.settings.autotune_precompile == "spawn":
                args_path = os.path.join(tempdir.name, "args.pt")
                torch.save(self.args, args_path)
                self._precompile_args_path = args_path
            exit_stack.callback(self.cleanup)

            if not self._try_load_checkpoint():
                self._init_search()
            self._load_crashed_configs()
            try:
                best = self._autotune()
                self._cleanup_checkpoint()
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
        cached_path = self.kernel.get_cached_path(best)
        if cached_path is not None and is_master_rank():
            self.log(f"Code of selected kernel: {cached_path}")
        self.kernel.maybe_log_repro(self.log.warning, self.args, best)
        if self.settings.print_output_code:
            triton_code = self.kernel.to_triton_code(best)
            if triton_code is not None:
                print(triton_code, file=sys.stderr)
        return best

    def _init_search(self) -> None:
        """
        Initialize the search state for a fresh autotuning run.

        This method is called when starting autotuning without a checkpoint.
        Subclasses should override this to set up initial population and state.
        After this method, _current_generation should be set to the generation
        that _autotune() should start its loop from.
        """

    def _autotune(self) -> Config:
        """
        Abstract method to perform the actual autotuning.

        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    def save_checkpoint(self) -> Path | None:
        """
        Save current autotuner state to checkpoint file.

        Only saves when autotune_checkpoint_dir is set (opt-in).
        Overwrites the same file each generation (keyed by stable hash).
        Returns None if checkpointing is disabled or not supported.

        Returns:
            Path to saved checkpoint file, or None if not saved
        """
        from ..runtime.kernel import BoundKernel

        # External kernels don't support caching/checkpointing
        if not isinstance(self.kernel, BoundKernel):
            return None

        if not self.kernel.is_cacheable():
            return None

        checkpoint_dir_str = self.settings.autotune_checkpoint_dir
        if checkpoint_dir_str is None:
            return None  # Opt-in: no dir set, no saving

        stable_hash = self._get_stable_hash()
        checkpoint_dir = Path(checkpoint_dir_str)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{stable_hash}.pt"

        state = self.state_dict()

        # Atomic write using temp file + rename
        tmp = checkpoint_dir / f".tmp.{stable_hash}.{uuid.uuid4().hex[:8]}"
        torch.save(state, tmp)
        os.replace(tmp, checkpoint_path)

        self.log(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def _cleanup_checkpoint(self) -> None:
        """Delete checkpoint file on successful autotune completion.

        Checkpoints are ephemeral in-progress state. Once autotuning
        completes successfully, the result is cached normally and the
        checkpoint is no longer needed.
        """
        checkpoint_dir_str = self.settings.autotune_checkpoint_dir
        if checkpoint_dir_str is None:
            return

        stable_hash = self._get_stable_hash()
        checkpoint_file = Path(checkpoint_dir_str) / f"{stable_hash}.pt"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            self.log(f"Checkpoint cleaned up: {checkpoint_file}")

        # Clean up crash-recovery artifacts
        for suffix in (".pending_config", ".crashed_configs"):
            artifact = Path(checkpoint_dir_str) / f"{stable_hash}{suffix}"
            if artifact.exists():
                artifact.unlink()

    @staticmethod
    def _serialize_numpy_rng_state(
        state: tuple[str, Any, int, int, float],
    ) -> dict[str, Any]:
        """Convert numpy RNG state to torch-serializable format.

        numpy.random.get_state() returns a tuple that contains a numpy array,
        which can't be serialized with torch.save(..., weights_only=True).
        This converts it to a dict with the array as a torch tensor.
        """
        return {
            "algorithm": state[0],  # str 'MT19937'
            "keys": torch.from_numpy(state[1].copy()),  # uint32[624] -> Tensor
            "pos": state[2],  # int
            "has_gauss": state[3],  # int
            "cached_gaussian": state[4],  # float
        }

    @staticmethod
    def _deserialize_numpy_rng_state(
        d: dict[str, Any],
    ) -> tuple[str, Any, int, int, float]:
        """Convert back to numpy RNG state tuple."""
        return (
            d["algorithm"],
            d["keys"].numpy().astype(np.uint32),
            d["pos"],
            d["has_gauss"],
            d["cached_gaussian"],
        )

    def state_dict(self) -> dict[str, Any]:
        """
        Return autotuner state as a dictionary.

        Auto-serializes all instance attrs not in ``_checkpoint_exclude``.
        Subclasses should override ``_save_custom_checkpoint_state``
        to add keys that need custom serialization (e.g. population).
        """
        rng_state: dict[str, Any] = {
            "random": random.getstate(),
            "numpy": self._serialize_numpy_rng_state(
                cast(
                    "tuple[str, Any, int, int, float]",
                    np.random.get_state(legacy=True),  # noqa: NPY002
                )
            ),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state()

        state: dict[str, Any] = {
            "algorithm": self.__class__.__name__,
            "cache_key_stable_hash": self._get_stable_hash(),
            "rng_state": rng_state,
        }

        # Auto-serialize __dict__ minus excluded attrs (skip callables like
        # patched methods that tests may add to the instance)
        excluded = type(self)._collect_checkpoint_exclude()
        for attr, val in self.__dict__.items():
            if attr in excluded or callable(val):
                continue
            if isinstance(val, enum.Enum):
                val = val.value
            elif isinstance(val, collections.Counter):
                val = dict(val)
            state[attr] = val

        self._save_custom_checkpoint_state(state)
        return state

    def _save_custom_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Hook for subclasses to add custom keys to state dict."""

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore autotuner state from a dictionary.

        Auto-restores all instance attrs not in ``_checkpoint_exclude``
        that are present in the state dict. Subclasses should override
        ``_load_custom_checkpoint_state`` to restore keys that need
        custom deserialization.
        """
        # Validate required metadata
        for key in ("algorithm", "cache_key_stable_hash", "rng_state"):
            if key not in state:
                raise exc.CheckpointError(
                    f"Checkpoint is missing required field: {key!r}. "
                    f"This may be from an incompatible Helion version."
                )

        if state.get("cache_key_stable_hash") != self._get_stable_hash():
            raise exc.CheckpointError(
                "State dict is incompatible: kernel, hardware, or input shapes may have changed"
            )

        # Validate that auto-serialized attrs are present
        excluded = type(self)._collect_checkpoint_exclude()
        expected = {
            k for k, v in self.__dict__.items() if k not in excluded and not callable(v)
        }
        missing = expected - state.keys()
        if missing:
            raise exc.CheckpointError(
                f"Checkpoint is missing required fields: {missing}. "
                f"This may be from an incompatible Helion version."
            )

        # Restore RNG state
        rng_state = state["rng_state"]
        random.setstate(rng_state["random"])
        np.random.set_state(  # noqa: NPY002
            self._deserialize_numpy_rng_state(rng_state["numpy"])
        )
        torch.random.set_rng_state(rng_state["torch"])
        if "torch_cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["torch_cuda"])

        # Auto-restore attrs from state dict
        for attr in expected:
            val = state[attr]
            current = getattr(self, attr, None)
            if isinstance(current, enum.Enum):
                val = type(current)(val)
            elif isinstance(current, collections.Counter):
                val = collections.Counter(val)
            setattr(self, attr, val)

        self._load_custom_checkpoint_state(state)

    def _load_custom_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Hook for subclasses to restore custom state + post-load actions."""

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
    status: Literal["ok", "error", "timeout", "peer_compilation_fail", "unknown"] = (
        "unknown"
    )
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


def _estimate_tree_bytes(obj: object) -> int:
    """Estimate the memory usage of a pytree of objects, counting shared storage only once."""
    total = 0
    seen_ptrs: set[int] = set()

    def _accumulate(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        size = tensor.element_size() * tensor.numel()
        try:
            storage = tensor.untyped_storage()
        except RuntimeError:
            pass
        else:
            ptr = storage.data_ptr()
            if ptr in seen_ptrs:
                return tensor
            seen_ptrs.add(ptr)
            size = storage.nbytes()
        total += size
        return tensor

    tree_map_only(torch.Tensor, _accumulate, obj)
    return total


class PopulationBasedSearch(BaseSearch):
    """
    Base class for search algorithms that use a population of configurations.

    Attributes:
        population (list[PopulationMember]): The current population of configurations.
        flat_spec (list[ConfigSpecFragment]): The flattened configuration specification.
    """

    _checkpoint_exclude: ClassVar[tuple[str, ...]] = (
        "config_gen",
        "finishing_rounds",
        "population",  # handled by _save/_load_custom_checkpoint_state
    )

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
        self.parallel_benchmark_population([member], desc="Benchmarking")
        return member

    def parallel_benchmark_flat(
        self, to_check: list[FlatConfig]
    ) -> list[PopulationMember]:
        """
        Benchmark multiple flat configurations in parallel.

        Args:
            to_check: A list of flat configurations to benchmark.

        Returns:
            A list of population members with the benchmark results.
        """
        result = [*map(self.make_unbenchmarked, to_check)]
        return self.parallel_benchmark_population(result)

    def make_unbenchmarked(self, flat_values: FlatConfig) -> PopulationMember:
        """
        Create a population member with unbenchmarked configuration.  You
        should pass the result of this to parallel_benchmark_population.

        Args:
            flat_values: The flat configuration values.

        Returns:
            A population member with undefined performance.
        """
        config = self.config_gen.unflatten(flat_values)
        return PopulationMember(_unset_fn, [], flat_values, config)

    def _get_current_hardware_and_specialization(
        self,
    ) -> tuple[str | None, str | None]:
        """
        Get the current hardware and specialization_key for matching cached configs.

        Returns:
            A tuple of (hardware, specialization_key) strings.
        """
        hardware = get_device_name(extract_device(self.args))

        inner_kernel = getattr(self.kernel, "kernel", None)
        if inner_kernel is None or not hasattr(inner_kernel, "specialization_key"):
            return hardware, None
        spec_key = inner_kernel.specialization_key(self.args)
        specialization_key = str(_normalize_spec_key(spec_key))

        return hardware, specialization_key

    def _find_similar_cached_configs(self, max_configs: int) -> list[SavedBestConfig]:
        """
        Find cached configs that match hardware, specialization_key, and
        structural fingerprint (config_spec_hash).

        Returns an empty list when cache is skipped (via HELION_SKIP_CACHE
        or the skip_cache parameter), so that "skip cache" consistently
        means no cache reads of any kind.

        Args:
            max_configs: Maximum number of configs to return.

        Returns:
            List of matching SavedBestConfig objects, sorted by file modification time (most recent first).
        """
        from .base_cache import should_skip_cache

        if self._skip_cache or should_skip_cache():
            return []

        from .local_cache import get_helion_cache_dir
        from .local_cache import iter_cache_entries

        current_hardware, current_spec_key = (
            self._get_current_hardware_and_specialization()
        )
        if current_hardware is None or current_spec_key is None:
            return []

        current_fingerprint_hash = self.config_spec.structural_fingerprint_hash()

        matching: list[SavedBestConfig] = []
        for entry in iter_cache_entries(
            get_helion_cache_dir(),
            max_scan=self.settings.autotune_best_available_max_cache_scan,
        ):
            if entry.hardware != current_hardware:
                continue
            if _normalize_spec_key_str(entry.specialization_key) != current_spec_key:
                continue
            # Skip entries without a matching structural fingerprint or flat_config.
            if entry.config_spec_hash != current_fingerprint_hash:
                continue
            if entry.flat_config is None:
                continue
            matching.append(entry)
            if len(matching) >= max_configs:
                break

        return matching

    def _generate_best_available_population_flat(self) -> list[FlatConfig]:
        """
        Generate initial population using default config plus cached configs.

        Always starts with the default configuration, then adds up to
        MAX_BEST_AVAILABLE_CONFIGS matching cached configs from previous runs.
        No random configs are added.  Duplicate configs are discarded.

        Returns:
            A list of unique FlatConfig values for the initial population.
            Minimum size is 1 (just default), maximum is 1 + autotune_best_available_max_configs setting.
        """
        # Always start with the default config
        default_flat = self.config_gen.default_flat()
        default_config = self.config_gen.unflatten(default_flat)
        seen: set[Config] = {default_config}
        result: list[FlatConfig] = [default_flat]
        self.log("Starting with default config")

        max_configs = self.settings.autotune_best_available_max_configs
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
            except (ValueError, TypeError, KeyError, AssertionError) as e:
                self.log(f"Failed to transfer cached config {i + 1}: {e}")
                continue

        if duplicates > 0:
            self.log.debug(f"Discarded {duplicates} duplicate config(s)")

        self.log(
            f"Initial population: 1 default + {len(result) - 1} unique cached = {len(result)} total"
        )

        return result

    def parallel_benchmark_population(
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
        if len(self._mutated_arg_indices) > 0:
            bench_args = _clone_args(
                self.args,
                self.kernel.env.process_group_name,
                idx_to_clone=self._mutated_arg_indices,
            )
        else:
            bench_args = self.args
        iterator = [functools.partial(m.fn, *bench_args) for m in members]
        _backend = getattr(getattr(self, "config_spec", None), "backend", None)
        _ib = (
            _backend.get_interleaved_bench() if _backend is not None else None
        ) or interleaved_bench
        bench_fn: Callable[..., list[float]] = (
            self.settings.autotune_benchmark_fn or _ib
        )
        if self.settings.autotune_progress_bar:
            new_timings = bench_fn(iterator, repeat=repeat, desc=desc)
        else:
            new_timings = bench_fn(iterator, repeat=repeat)
        new_timings = sync_object(
            new_timings, process_group_name=self.kernel.env.process_group_name
        )
        for m, t in zip(members, new_timings, strict=True):
            m.perfs.append(t)
            if t < self.best_perf_so_far:
                self.best_perf_so_far = t

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

    def set_generation(self, generation: int) -> None:
        if generation == self._current_generation:
            return
        self._current_generation = generation
        super().set_generation(generation)
        if generation > 0:
            self.save_checkpoint()

    def statistics(self) -> str:
        """
        Generate statistics for the current population.

        Returns:
            A string summarizing the population performance.
        """
        return population_statistics(self.population)

    def _save_custom_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._save_custom_checkpoint_state(state)
        population_state = []
        for member in self.population:
            population_state.append(
                {
                    "perfs": member.perfs,
                    "flat_values": member.flat_values,
                    "config": member.config,
                    "status": member.status,
                    "compile_time": member.compile_time,
                }
            )
        state["population"] = population_state

    def _load_custom_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_custom_checkpoint_state(state)
        self.population = []
        for member_state in state["population"]:
            member = PopulationMember(
                fn=_unset_fn,
                perfs=member_state["perfs"],
                flat_values=member_state["flat_values"],
                config=member_state["config"],
                status=member_state["status"],
                compile_time=member_state.get("compile_time"),
            )
            self.population.append(member)

        # Recompile kernel functions for all population members
        recompile_failures: list[tuple[PopulationMember, str]] = []
        for member in self.population:
            if member.fn is _unset_fn and member.status == "ok":
                try:
                    member.fn = self.kernel.compile_config(
                        member.config, allow_print=False
                    )
                except Exception as e:
                    member.fn = _unset_fn
                    member.status = "error"
                    member.perfs.append(inf)  # Ensure member won't be selected as best
                    recompile_failures.append((member, str(e)))

        if recompile_failures:
            self.log(
                f"Warning: {len(recompile_failures)} config(s) failed to recompile "
                f"and will be skipped. First failure: {recompile_failures[0][1]}"
            )

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

        for round_num in range(1, rounds + 1):
            simplified = False
            candidates: list[PopulationMember] = [current]

            # Generate candidates by resetting each parameter to its default
            for i in range(len(current.flat_values)):
                if current.flat_values[i] != default_flat[i]:
                    # Create a new config with this parameter reset to default
                    new_flat = [*current.flat_values]
                    new_flat[i] = default_flat[i]
                    candidate = self.make_unbenchmarked(new_flat)
                    # Only add if this produces a different config
                    if candidate.config != current.config:
                        candidates.append(candidate)

            if len(candidates) <= 1:
                self.log(f"Finishing round {round_num}: no more parameters to simplify")
                break

            # Benchmark the candidates
            unbenchmarked = [m for m in candidates if len(m.perfs) == 0]
            if unbenchmarked:
                self.set_generation(self._autotune_metrics.num_generations + 1)
                self.parallel_benchmark_population(
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
                if combined.config != current.config:
                    self.parallel_benchmark_population(
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
    parts: list[str] = []
    for label in ("error", "timeout", "ok"):
        count = status_counts.get(label, 0)
        if count:
            parts.append(f"{label}={count}")

    parts.extend(
        (
            f"min={working[0].perf:.4f}",
            f"mid={working[len(working) // 2].perf:.4f}",
            f"max={working[-1].perf:.4f}",
            f"best={pprint.pformat(dict(population[0].config), width=100, compact=True)}",
        )
    )
    return "\n" + "\n".join(parts)


def _triton_compile(
    fn: CompiledConfig,
    args: Sequence[object],
    config: Config,
    kernel: _AutotunableKernel,
) -> bool:
    """Trigger Triton JIT compilation without running the kernel.

    Extracts the Triton kernel and its launch arguments from fn, then
    invokes the precompiler so the compiled binary is cached before the
    actual benchmark run.

    The function requires the availability of CUDA.
    """

    def extract_launcher(
        triton_kernel: object,
        grid: tuple[int, ...],
        *launch_args: object,
        **launch_kwargs: object,
    ) -> NoReturn:
        raise _ExtractedLaunchArgs(triton_kernel, grid, launch_args, launch_kwargs)

    try:
        fn(*args, _launcher=extract_launcher)
        raise RuntimeError("Expected _ExtractedLaunchArgs to be raised")
    except _ExtractedLaunchArgs as extracted:
        precompiler = make_precompiler(
            cast("Any", extracted.kernel),
            config,
            cast("BoundKernel", kernel),
        )(*extracted.args, **extracted.kwargs)
        if precompiler is already_compiled:
            return True
        if precompiler is already_compiled_fail:
            return False
        return precompiler(False)  # pyrefly: ignore[bad-argument-count]
    except Exception:
        return False


def _unset_fn(*args: object) -> NoReturn:
    raise RuntimeError("Uninitialized function")
