"""Picklable benchmark job executed inside a ``BenchmarkWorker``."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING
from typing import cast

import torch

from .benchmarking import do_bench
from .precompile_future import _load_compiled_fn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .precompile_future import SerializedCompiledFunction


@functools.cache
def _load_args(path: str) -> Sequence[object]:
    # Cached so re-spawning configs don't re-read the same args off disk.
    return cast("Sequence[object]", torch.load(path))


@dataclasses.dataclass
class BenchmarkJob:
    fn_spec: SerializedCompiledFunction
    args_path: str
    warmup: int = 1
    rep: int = 50

    def __call__(self) -> float:
        fn = _load_compiled_fn(self.fn_spec)
        args = _load_args(self.args_path)
        # return_mode="median" guarantees a float return (not the tuple variant).
        return cast(
            "float",
            do_bench(
                functools.partial(fn, *args),
                return_mode="median",
                warmup=self.warmup,
                rep=self.rep,
            ),
        )
