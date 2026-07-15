"""Picklable benchmark job executed inside a ``BenchmarkWorker``."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING
from typing import Literal

import torch

from .accuracy import assert_close
from .benchmarking import PerfStats
from .benchmarking import do_bench
from .benchmarking import do_bench_generic
from .benchmarking import synchronize_device
from .kernel_args import load_trusted_kernel_args
from .logger import capture_output
from .precompile_future import _load_compiled_fn

if TYPE_CHECKING:
    from .precompile_future import SerializedCompiledFunction


@dataclasses.dataclass
class BenchmarkJob:
    fn_spec: SerializedCompiledFunction
    args_path: str
    warmup: int = 1
    rep: int = 50
    use_wall_clock: bool = False
    return_mode: Literal["median", "stats"] = "median"

    def __call__(self) -> float | PerfStats:
        # Subprocess inherits parent stderr; capture so Triton runtime
        # diagnostics don't leak to the user's terminal.
        with capture_output():
            fn = _load_compiled_fn(self.fn_spec)
            args = load_trusted_kernel_args(self.args_path)
            bench = do_bench_generic if self.use_wall_clock else do_bench
            result = bench(
                functools.partial(fn, *args),
                return_mode=self.return_mode,
                warmup=self.warmup,
                rep=self.rep,
            )
            if self.return_mode == "stats":
                if not isinstance(result, PerfStats):
                    raise TypeError(
                        f"return_mode='stats' expected do_bench to return a "
                        f"PerfStats, got {type(result).__name__}"
                    )
                return result
            if not isinstance(result, float):
                raise TypeError(
                    f"return_mode='median' expected do_bench to return a "
                    f"float, got {type(result).__name__}"
                )
            return result


@functools.cache
def _load_trusted_baseline_output(path: str) -> object:
    return torch.load(path, weights_only=False)


@dataclasses.dataclass(frozen=True)
class AccuracyCheckResult:
    ok: bool
    message: str = ""


@dataclasses.dataclass
class AccuracyCheckJob:
    fn_spec: SerializedCompiledFunction
    args_path: str
    baseline_path: str
    atol: float
    rtol: float

    def __call__(self) -> AccuracyCheckResult:
        # Keep compile/launch diagnostics out of the autotune progress stream.
        with capture_output():
            fn = _load_compiled_fn(self.fn_spec)
            args = load_trusted_kernel_args(self.args_path)
            baseline_output = _load_trusted_baseline_output(self.baseline_path)
            output = fn(*args)
            synchronize_device()

        try:
            assert_close(output, baseline_output, atol=self.atol, rtol=self.rtol)
        except AssertionError as e:
            return AccuracyCheckResult(ok=False, message=str(e))
        return AccuracyCheckResult(ok=True)
