"""Shared benchmark loop for pretuned kernel ``benchmark()`` functions.

Times the Helion kernel against one or more baselines (under CUDA graphs or
plain ``do_bench``), prints a per-shape table, and returns a metrics dict that
``pretuned_kernels/run.py`` records directly (no stdout parsing):

  {"helion_wins":.., "total":.., "geomean":.., "best_speedup":..,   # vs best baseline
   "baselines": {"<name>": {"wins":..,"total":..,"geomean":..,"best_speedup":..}}}

where "baselines" is helion's speedup over *each* baseline (powers the dashboard
dropdown). Kernels add their directory's parent to ``sys.path`` and
``import _bench`` so this works both under ``python pretuned_kernels/<k>/<k>.py``
and via run.py's importlib loader.
"""

from __future__ import annotations

import abc
import math
import statistics
import time
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import torch

from helion import exc
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.logger import classify_triton_exception
from helion.autotuner.logger import match_unrecoverable_runtime_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence

    import helion
    from helion.runtime.kernel import CompiledConfig


ShapeT = TypeVar("ShapeT")
OutputT = TypeVar("OutputT")


def geomean(values: Iterable[float]) -> float:
    pos = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in pos) / max(len(pos), 1))


def bench_cudagraph(call: Callable[[], object], rep: int = 100) -> float:
    """Median CUDA-graph latency (ms), clearing the L2 cache each iteration.

    Uses tritonbench's cudagraph timer, which zeroes the L2 cache before every
    replay and subtracts the clear cost. ``triton.testing.do_bench_cudagraph``
    does *not* clear L2, so a graph replaying the same inputs reuses cached data
    and under-reports latency. tritonbench is required (no fallback) so pretuned
    numbers are always measured with cache clearing -- install it before running
    a cudagraph kernel's ``main()`` (the nightly benchmark workflow does).
    """
    from tritonbench.components.do_bench.run import (  # pyrefly: ignore[missing-import]
        _do_bench_cudagraph_with_cache_clear,
    )

    return _do_bench_cudagraph_with_cache_clear(call, rep=rep, return_mode="median")


def capture_cuda_graph(
    call: Callable[[], OutputT],
    reset: Callable[[], object] | None = None,
) -> tuple[torch.cuda.CUDAGraph, OutputT]:
    """Warm up and capture one CUDA graph, optionally restoring mutable inputs."""
    capture_stream = torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            if reset is not None:
                reset()
            output = call()
        if reset is not None:
            reset()
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = call()
    torch.cuda.synchronize()
    return graph, output


def bench_pre_captured_cudagraphs(
    calls: Sequence[Callable[[], object]],
    rep: int = 100,
    resets: Sequence[Callable[[], object] | None] | None = None,
) -> list[float]:
    """Median graph device latencies (ms), with cold L2 and balanced ordering.

    Some external references must be captured before benchmarking so their
    graph nodes, including captured copies, are represented faithfully. CUDA
    does not allow a graph replay inside the outer capture used by
    ``bench_cudagraph``. Queue an L2 clear before every event pair, matching
    ``triton.testing.do_bench``.

    Do not synchronize between the clear and the start event. On an idle GPU,
    doing so lets the start event execute while Python is still submitting the
    graph, which incorrectly includes host-side graph wrapper work as idle time
    in the CUDA-event interval. The clear keeps the GPU busy while the next
    replay is submitted, so the events measure device execution consistently.
    Calls rotate through every position, then reverse the rotations, so no
    implementation always sees the hottest GPU clocks.
    """
    import triton

    if not calls:
        raise ValueError("calls must not be empty")
    if rep <= 0:
        raise ValueError("rep must be positive")
    if resets is None:
        resets = (None,) * len(calls)
    elif len(resets) != len(calls):
        raise ValueError("resets must have one entry per call")

    driver = triton.runtime.driver.active
    device_interface = driver.get_device_interface()  # pyrefly: ignore[missing-attribute]
    cache = driver.get_empty_cache_for_benchmark()  # pyrefly: ignore[missing-attribute]
    cycle = 2 * len(calls)
    repetitions = rep

    def call_order(sample: int) -> tuple[int, ...]:
        indices = tuple(range(len(calls)))
        rotation = sample % len(indices)
        order = indices[rotation:] + indices[:rotation]
        return order[::-1] if (sample // len(indices)) % 2 else order

    for sample in range(cycle):
        for index in call_order(sample):
            reset = resets[index]
            if reset is not None:
                reset()
            calls[index]()
    device_interface.synchronize()

    starts = [
        [device_interface.Event(enable_timing=True) for _ in range(repetitions)]
        for _ in calls
    ]
    ends = [
        [device_interface.Event(enable_timing=True) for _ in range(repetitions)]
        for _ in calls
    ]
    for sample in range(repetitions):
        for index in call_order(sample):
            reset = resets[index]
            if reset is not None:
                reset()
            driver.clear_cache(cache)  # pyrefly: ignore[missing-attribute]
            starts[index][sample].record()
            calls[index]()
            ends[index][sample].record()
    device_interface.synchronize()
    return [
        statistics.median(
            start.elapsed_time(end)
            for start, end in zip(call_starts, call_ends, strict=True)
        )
        for call_starts, call_ends in zip(starts, ends, strict=True)
    ]


def bench_pre_captured_cudagraph(call: Callable[[], object], rep: int = 100) -> float:
    """Median latency for one graph; see the balanced multi-graph timer."""
    return bench_pre_captured_cudagraphs([call], rep=rep)[0]


def thermal_warmup(duration_ms: int) -> None:
    """Raise GPU clocks with device work before a latency sweep."""
    if duration_ms <= 0:
        return
    value = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    end = time.monotonic() + duration_ms / 1000
    while time.monotonic() < end:
        for _ in range(50):
            value = value @ value
        torch.cuda.synchronize()


class CapturedCudagraphBenchmarkProvider(LocalBenchmarkProvider, abc.ABC):
    """Shared error handling and cold-L2 timing for captured-graph tuning."""

    def __init__(self, *args: object, repetitions: int, **kwargs: object) -> None:
        self._repetitions = repetitions
        super().__init__(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    @abc.abstractmethod
    def _capture_validated_replay(
        self, config: helion.Config, fn: CompiledConfig
    ) -> Callable[[], object] | None: ...

    def _benchmark_function(
        self,
        config: helion.Config,
        fn: CompiledConfig,
        *,
        effective_source_hash: str | None = None,
    ) -> float:
        self._autotune_metrics.num_configs_tested += 1
        try:
            replay = self._capture_validated_replay(config, fn)
            if replay is None:
                return float("inf")
            return bench_pre_captured_cudagraph(replay, rep=self._repetitions)
        except Exception as error:
            error.__traceback__ = None
            if match_unrecoverable_runtime_error(error):
                self.kernel.maybe_log_repro(self.log.error, self.args, config)
                raise exc.TritonUnrecoverableRuntimeError(
                    reason=str(error),
                    decorator=self.kernel.format_kernel_decorator(
                        config, self.settings
                    ),
                    error=f"{type(error).__qualname__}: {error}",
                ) from error
            backend = getattr(self.config_spec, "backend", None)
            action = (
                backend.classify_autotune_exception(error)
                if backend is not None
                else None
            ) or classify_triton_exception(error)
            if action == "raise" and not self.settings.autotune_ignore_errors:
                raise
            self.log.debug(
                f"Skipping captured-graph candidate after "
                f"{type(error).__name__}: {error}"
            )
            self._record_compile_failure(config)
            return float("inf")
        finally:
            self._clear_jit_fast_path_caches(fn)


def _bench(
    call: Callable[[], object], use_cudagraph: bool, warmup: int, rep: int
) -> float:
    if use_cudagraph:
        return bench_cudagraph(call, rep=rep)
    from triton.testing import do_bench

    return cast(
        "float",
        do_bench(call, warmup=warmup, rep=rep, return_mode="median"),
    )


def run_sweep(
    shapes: Iterable[ShapeT],
    make_calls: Callable[
        [ShapeT],
        tuple[
            Callable[[], object],
            list[tuple[str, Callable[[], object]]],
            str,
        ],
    ],
    *,
    use_cudagraph: bool,
    pre_captured_cudagraph: bool = False,
    interleave_pre_captured: bool = True,
    make_resets: Callable[[ShapeT], Sequence[Callable[[], object] | None]]
    | None = None,
    shape_header: str,
    warmup: int = 25,
    rep: int = 100,
    thermal_warmup_ms: int = 0,
    verbose: bool = True,
) -> dict:
    """Benchmark helion vs baselines over ``shapes``; return metrics (print if verbose).

    ``make_calls(shape)`` returns ``(helion_call, [(baseline_name, baseline_call)],
    shape_cells)`` where the calls are zero-arg closures over freshly built inputs
    and ``shape_cells`` is the preformatted leading column(s) for the table row.
    Set ``interleave_pre_captured=False`` to finish one cold-L2 graph benchmark
    before starting the next implementation.
    The metrics dict is always returned; the per-shape table is printed only when
    ``verbose``.
    """
    if use_cudagraph and pre_captured_cudagraph:
        raise ValueError(
            "use_cudagraph and pre_captured_cudagraph are mutually exclusive"
        )
    if make_resets is not None and not pre_captured_cudagraph:
        raise ValueError("make_resets requires pre_captured_cudagraph=True")

    def _p(*args: object) -> None:
        if verbose:
            print(*args)

    if verbose:
        _p(f"GPU: {torch.cuda.get_device_name()}")
    speedups_by_base: dict[str, list[float]] = {}
    best_speedups: list[float] = []
    helion_wins = 0
    best_speedup = 0.0
    header_printed = False
    for shape in shapes:
        helion_call, baseline_calls, shape_cells = make_calls(shape)
        names = [n for n, _ in baseline_calls]
        if not header_printed:
            for n in names:
                speedups_by_base[n] = []
            base_hdr = "  ".join(f"{n + ' (us)':>13s}" for n in names)
            _p(f"{shape_header}  {'helion (us)':>12s}  {base_hdr}  {'speedup':>8s}")
            header_printed = True

        if pre_captured_cudagraph:
            calls = [helion_call, *(call for _name, call in baseline_calls)]
            resets = (
                list(make_resets(shape))
                if make_resets is not None
                else [None] * len(calls)
            )
            if len(resets) != len(calls):
                raise ValueError("make_resets must return one entry per call")
            if interleave_pre_captured:
                thermal_warmup(thermal_warmup_ms)
                if make_resets is None:
                    timings = bench_pre_captured_cudagraphs(calls, rep=rep)
                else:
                    timings = bench_pre_captured_cudagraphs(
                        calls,
                        rep=rep,
                        resets=resets,
                    )
            else:
                timings = []
                for call, reset in zip(calls, resets, strict=True):
                    thermal_warmup(thermal_warmup_ms)
                    timings.extend(
                        bench_pre_captured_cudagraphs(
                            [call],
                            rep=rep,
                            resets=[reset],
                        )
                    )
            ms_helion, *baseline_timings = timings
            base_ms = dict(zip(names, baseline_timings, strict=True))
        else:
            helion_call()  # warmup / compile
            ms_helion = _bench(helion_call, use_cudagraph, warmup, rep)
            base_ms = {
                name: _bench(call, use_cudagraph, warmup, rep)
                for name, call in baseline_calls
            }
        for name in names:
            speedups_by_base[name].append(
                base_ms[name] / ms_helion if ms_helion > 0 else float("nan")
            )
        best_name = min(base_ms, key=lambda name: base_ms[name])
        speedup = base_ms[best_name] / ms_helion if ms_helion > 0 else float("nan")
        best_speedups.append(speedup)
        if speedup > 1.0:
            helion_wins += 1
        best_speedup = max(best_speedup, speedup)
        base_cols = "  ".join(f"{base_ms[n] * 1000:>13.2f}" for n in names)
        _p(
            f"{shape_cells}  {ms_helion * 1000:>12.2f}  {base_cols}  "
            f"{speedup:>7.2f}x  (vs {best_name})"
        )

    names = list(speedups_by_base)
    per_baseline = {
        n: {
            "wins": sum(1 for s in speedups_by_base[n] if s > 1.0),
            "total": len(speedups_by_base[n]),
            "geomean": round(geomean(speedups_by_base[n]), 4),
            "best_speedup": round(max(speedups_by_base[n], default=0.0), 4),
        }
        for n in names
    }
    for n in names:
        m = per_baseline[n]
        _p(
            f"vs {n}: wins={m['wins']}/{m['total']} "
            f"geomean={m['geomean']:.3f}x best={m['best_speedup']:.2f}x"
        )
    total = len(best_speedups)
    gm = geomean(best_speedups)
    _p(
        f"\nHelion faster on {helion_wins}/{total} shapes vs the best baseline; "
        f"geomean speedup {gm:.3f}x; best speedup {best_speedup:.2f}x."
    )
    # Metrics are helion vs the best (fastest) baseline per shape, plus the
    # per-baseline breakdown; returned to the caller (pretuned_kernels/run.py).
    return {
        "helion_wins": helion_wins,
        "total": total,
        "geomean": round(gm, 4),
        "best_speedup": round(best_speedup, 4),
        "baselines": per_baseline,
    }
