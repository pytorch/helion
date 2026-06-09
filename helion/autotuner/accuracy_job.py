"""Picklable accuracy-check job executed inside a ``BenchmarkWorker``."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import torch

from .accuracy import _assert_close
from .benchmarking import synchronize_device
from .kernel_args import load_trusted_kernel_args
from .logger import capture_output
from .precompile_future import _load_compiled_fn

if TYPE_CHECKING:
    from .precompile_future import SerializedCompiledFunction


@functools.cache
def _load_baseline(path: str) -> object:
    # Cached like load_trusted_kernel_args: every config in one autotune run
    # compares against the same baseline, so the worker reads it off disk once.
    return torch.load(path, weights_only=False)


@dataclasses.dataclass
class AccuracyCheckJob:
    fn_spec: SerializedCompiledFunction
    args_path: str
    baseline_path: str
    atol: float
    rtol: float

    def __call__(self) -> bool:
        # Subprocess inherits parent stderr; capture so Triton runtime
        # diagnostics don't leak to the user's terminal.
        with capture_output():
            fn = _load_compiled_fn(self.fn_spec)
            args = load_trusted_kernel_args(self.args_path)
            baseline_output = _load_baseline(self.baseline_path)
            output = fn(*args)
            synchronize_device(output)
            try:
                _assert_close(output, baseline_output, atol=self.atol, rtol=self.rtol)
            except AssertionError:
                return False
            return True
