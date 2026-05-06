"""Picklable benchmark jobs executed inside a ``BenchmarkWorker``."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from .benchmarking import do_bench
from .benchmarking import interleaved_bench
from .precompile_future import _load_compiled_fn

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from .precompile_future import SerializedCompiledFunction


@functools.lru_cache(maxsize=2)
def _load_args(path: str) -> Sequence[object]:
    """Load Helion-created benchmark args in a worker process.

    The cache is intentionally tiny: process-level pools see multiple shapes,
    but each worker should only retain the latest args. ``weights_only=False``
    is required because kernel args can include callables such as epilogues.
    """
    return cast("Sequence[object]", torch.load(path, weights_only=False))


@dataclasses.dataclass
class PrecompileJob:
    """Compile-only precompile in a worker. Runs host-side helion code with
    an extract_launcher that raises before any kernel launch, then triggers
    Triton's compile (CPU + ptxas, no kernel execution). The binary lands
    in Triton's on-disk cache for the benchmark phase to reuse.

    Mirrors fork-mode children, but inside a long-lived spawn worker so the
    parent never touches CUDA during prep."""

    fn_spec: SerializedCompiledFunction
    args_path: str

    def __call__(self) -> bool:
        from ..runtime.precompile_shim import already_compiled
        from ..runtime.precompile_shim import already_compiled_fail
        from ..runtime.precompile_shim import make_precompiler
        from .precompile_future import _ExtractedLaunchArgs

        fn = _load_compiled_fn(self.fn_spec)
        args = _load_args(self.args_path)

        captured: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def extract_launcher(
            triton_kernel: object,
            grid: tuple[int, ...],
            *launch_args: object,
            **launch_kwargs: object,
        ) -> object:
            captured.append((triton_kernel, launch_args, launch_kwargs))
            raise _ExtractedLaunchArgs(triton_kernel, grid, launch_args, launch_kwargs)

        with contextlib.suppress(_ExtractedLaunchArgs):
            fn(*args, _launcher=extract_launcher)  # pyrefly: ignore[bad-argument-type]
        if not captured:
            # No kernel launch in host code -> nothing to compile.
            return True

        triton_fn, launch_args, launch_kwargs = captured[0]
        precompiler = cast(
            "Callable[..., bool]",
            make_precompiler(cast("Any", triton_fn), None, None)(
                *launch_args, **launch_kwargs
            ),
        )
        if precompiler is already_compiled:
            return True
        if precompiler is already_compiled_fail:
            return False
        return precompiler(in_child_process=False)


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


@dataclasses.dataclass
class RebenchmarkJob:
    """Run interleaved rebenchmarking in the same isolated worker path."""

    fn_specs: list[SerializedCompiledFunction]
    args_path: str
    repeat: int

    def __call__(self) -> list[float]:
        args = _load_args(self.args_path)
        fns: list[Callable[[], object]] = [
            functools.partial(_load_compiled_fn(fn_spec), *args)
            for fn_spec in self.fn_specs
        ]
        return interleaved_bench(fns, repeat=self.repeat)
