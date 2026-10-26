"""Picklable benchmark job executed inside a ``BenchmarkWorker``."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.utils._pytree import tree_flatten

from .accuracy import assert_close
from .accuracy import poison_cached_device_memory
from .benchmark_worker import PHASE_COMPILED
from .benchmark_worker import PhasedJob
from .benchmarking import do_bench
from .benchmarking import do_bench_generic
from .benchmarking import synchronize_device
from .kernel_args import _clone_args
from .kernel_args import load_private_kernel_args
from .kernel_args import load_trusted_kernel_args
from .logger import capture_output
from .precompile_future import _load_compiled_fn
from .precompile_future import _unload_compiled_fn

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..runtime.kernel import CompiledConfig
    from .precompile_future import SerializedCompiledFunction


class CompiledFunctionLoadError(Exception):
    """The benchmark worker could not reconstruct a generated wrapper."""


def _load_compiled_fn_for_worker(
    fn_spec: SerializedCompiledFunction,
) -> CompiledConfig:
    try:
        return _load_compiled_fn(fn_spec)
    except Exception as error:
        raise CompiledFunctionLoadError(
            f"{type(error).__qualname__}: {error}"
        ) from error


@dataclasses.dataclass
class BenchmarkJob(PhasedJob):
    """Load, compile, run once and time one candidate in the worker.

    The first call (the backend JIT -- for CuTe the DSL compile of the
    generated wrapper -- plus one launch) is its own phase, reported to the
    parent as :data:`PHASE_COMPILED` before the timing starts, so the parent
    can hold the compile to a deadline of its own and give the timing the
    whole benchmark window.
    """

    fn_spec: SerializedCompiledFunction
    args_path: str
    warmup: int = 1
    rep: int = 50
    use_wall_clock: bool = False
    fixed_repetitions: int | None = None
    probe_long_kernel: bool = False
    # Finalist slot whose private argument copies this job reads (None shares
    # the cached arguments with every other job).
    args_slot: int | None = None
    # Arguments the kernel writes: the job times it on a fresh clone of them,
    # as the in-process path does, so the worker's cached arguments keep
    # their original values for every later job (an accuracy check among
    # them). Private finalist copies are already the finalist's own.
    mutated_arg_indices: tuple[int, ...] = ()
    # L2 regime of the timed calls (``call``, ``off`` or ``replay``); None
    # reads HELION_AUTOTUNE_L2_FLUSH in the worker. The final selection names
    # its own regime (``final_selection_l2_flush_mode``).
    l2_flush_mode: str | None = None

    def __call__(self, report_phase: Callable[[str], None] | None = None) -> float:
        # Subprocess inherits parent stderr; capture so Triton runtime
        # diagnostics don't leak to the user's terminal.
        with capture_output():
            fn = _load_compiled_fn_for_worker(self.fn_spec)
            try:
                if self.args_slot is None:
                    args = load_trusted_kernel_args(self.args_path)
                    if self.mutated_arg_indices:
                        args = _clone_args(
                            args, None, idx_to_clone=self.mutated_arg_indices
                        )
                else:
                    args = load_private_kernel_args(self.args_path, self.args_slot)
                bench = do_bench_generic if self.use_wall_clock else do_bench
                benchmark_fn = functools.partial(fn, *args)
                # Compile and run once; the timers then skip their setup call.
                benchmark_fn()
                synchronize_device()
                if report_phase is not None:
                    report_phase(PHASE_COMPILED)
                # return_mode="median" guarantees a float return.
                result = bench(
                    benchmark_fn,
                    return_mode="median",
                    warmup=self.warmup,
                    rep=self.rep,
                    fixed_repetitions=self.fixed_repetitions,
                    probe_long_kernel=self.probe_long_kernel,
                    l2_flush_mode=self.l2_flush_mode,
                    pre_warmed=True,
                )
                return cast(
                    "float",
                    result,
                )
            finally:
                _unload_compiled_fn(fn)


@functools.cache
def _load_trusted_baseline_output(path: str) -> object:
    return torch.load(path, weights_only=False)


def _cuda_device_of(args: object) -> torch.device | None:
    """The CUDA device of the first CUDA tensor in ``args`` (None without one)."""
    for leaf in tree_flatten(args)[0]:
        if isinstance(leaf, torch.Tensor) and leaf.device.type == "cuda":
            return leaf.device
    return None


@dataclasses.dataclass(frozen=True)
class AccuracyCheckResult:
    ok: bool
    message: str = ""


@dataclasses.dataclass
class AccuracyCheckJob:
    """Run one candidate on the trusted arguments and compare it to the baseline.

    A kernel that mutates arguments (``mutated_arg_indices``) runs on a fresh
    clone of those arguments, so every candidate starts from the original
    values; its post-call arguments are then compared against the baseline's
    (``baseline_post_args_path``), the same check the in-process path makes.
    """

    fn_spec: SerializedCompiledFunction
    args_path: str
    baseline_path: str
    atol: float
    rtol: float
    scale_atol: bool = False
    fp8_accuracy: str = "bitwise"
    mutated_arg_indices: tuple[int, ...] = ()
    baseline_post_args_path: str | None = None
    check_input_accuracy: bool = True

    def __call__(self) -> AccuracyCheckResult:
        # Keep compile/launch diagnostics out of the autotune progress stream.
        with capture_output():
            fn = _load_compiled_fn_for_worker(self.fn_spec)
            try:
                args = load_trusted_kernel_args(self.args_path)
                if self.mutated_arg_indices:
                    args = _clone_args(
                        args, None, idx_to_clone=self.mutated_arg_indices
                    )
                baseline_output = _load_trusted_baseline_output(self.baseline_path)
                # The worker ran the previous candidates on these arguments:
                # a fresh output must not inherit their (correct) values.
                poison_cached_device_memory(_cuda_device_of(args))
                output = fn(*args)
                synchronize_device()
            finally:
                _unload_compiled_fn(fn)

        try:
            assert_close(
                output,
                baseline_output,
                atol=self.atol,
                rtol=self.rtol,
                scale_atol_by_expected_rms=self.scale_atol,
                fp8_accuracy=self.fp8_accuracy,
            )
            if (
                self.mutated_arg_indices
                and self.check_input_accuracy
                and self.baseline_post_args_path is not None
            ):
                assert_close(
                    args,
                    _load_trusted_baseline_output(self.baseline_post_args_path),
                    atol=self.atol,
                    rtol=self.rtol,
                    scale_atol_by_expected_rms=self.scale_atol,
                    fp8_accuracy=self.fp8_accuracy,
                )
        except AssertionError as e:
            return AccuracyCheckResult(ok=False, message=str(e))
        return AccuracyCheckResult(ok=True)
