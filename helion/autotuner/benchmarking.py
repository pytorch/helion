from __future__ import annotations

import contextlib
import functools
import glob
import inspect
import logging
import math
import os
import statistics
import tempfile
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import TypeVar
from typing import cast

import torch

from ..runtime.settings import _env_get_bool
from ..runtime.settings import is_pallas_interpret
from .progress_bar import iter_with_progress
from helion import _compat
from helion._dist_utils import sync_object

if TYPE_CHECKING:
    from .logger import AutotuningLogger

T = TypeVar("T")

_log = logging.getLogger(__name__)
_BENCHMARK_CUDAGRAPH_ENV = "HELION_BENCHMARK_CUDAGRAPH"


def _bench_device_synchronize() -> None:
    """Synchronize the active device for wall-clock microbenchmarks.

    On Ascend, ``torch.accelerator.synchronize()`` can disagree with torch_npu's
    stream bookkeeping when multiple kernels were queued; use
    ``torch.npu.synchronize()`` when NPU is available.
    """
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    else:
        torch.accelerator.synchronize()



def _make_l2_cache_clearer() -> Callable[[], None]:
    """Return a callable that flushes the GPU L2 cache, or a no-op.

    The generic (wall-clock) bench used by the CuTe backend otherwise times
    kernels with the operands resident in L2 (warm), biasing the autotuner
    toward shallow-prefetch configs that starve the pipeline in the cold-L2 /
    streamed-once regime that deployment and tritonbench (which clears L2 by
    default) actually measure. Flushing L2 between timed calls makes the
    autotune regime match the deployment regime.

    Uses Triton's CUDA-only cache-clear primitive. Returns a no-op when not on
    CUDA (e.g. TPU/Pallas backends that also use the generic bench).
    """
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None:
        return lambda: None
    from triton import runtime

    active = runtime.driver.active  # type: ignore[attr-defined]
    cache = active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]

    def clear() -> None:
        active.clear_cache(cache)  # type: ignore[attr-defined]

    return clear


def _cudagraph_unavailable_reason() -> str | None:
    if getattr(torch.version, "hip", None) is not None:
        return "CUDA graph benchmarking is only enabled for NVIDIA CUDA"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if torch.cuda.is_current_stream_capturing():
        return "the current CUDA stream is already capturing"
    return None


def _make_cudagraph_replay(fn: Callable[[], T]) -> Callable[[], T]:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_output: list[T] = []
    with torch.cuda.graph(graph):
        static_output.append(fn())
    torch.cuda.synchronize()

    def replay() -> T:
        graph.replay()
        return static_output[0]

    return replay


def _maybe_cudagraph_replay(
    fn: Callable[[], T], *, default_enabled: bool = False
) -> Callable[[], T]:
    if not _env_get_bool(_BENCHMARK_CUDAGRAPH_ENV, default=default_enabled):
        return fn

    reason = _cudagraph_unavailable_reason()
    if reason is not None:
        _log.debug("Skipping CUDA graph benchmarking: %s", reason)
        return fn

    try:
        return _make_cudagraph_replay(fn)
    except Exception:
        _log.debug("CUDA graph benchmark capture failed; falling back", exc_info=True)
        return fn


def clear_jit_fast_path_caches(
    fn: Callable[..., object],
    log: logging.Logger | AutotuningLogger | None = None,
) -> None:
    """Clear Triton JIT fast-path caches for a generated Helion wrapper."""
    try:
        fn_name = getattr(fn, "__name__", None)
        fn_globals = getattr(fn, "__globals__", None)
        if fn_name is None or fn_globals is None:
            return
        triton_jit_fn = fn_globals.get(f"_helion_{fn_name}")
        clear = getattr(triton_jit_fn, "clear_fast_path_caches", None)
        if clear is not None:
            clear()
    except Exception:
        if log is not None:
            log.debug("Failed to clear Triton JIT fast-path cache.", exc_info=True)


def _get_tpu_tensors(result: object) -> list[torch.Tensor]:
    """Extract TPU tensors from a result that may be a tensor, tuple, or list."""
    if isinstance(result, torch.Tensor) and result.device.type == "tpu":
        return [result]
    if isinstance(result, (tuple, list)):
        tensors = []
        for v in result:
            if isinstance(v, torch.Tensor) and v.device.type == "tpu":
                tensors.append(v)
        return tensors
    return []


def synchronize_device(result: object = None) -> None:
    """Wait for device computation to complete."""
    if not is_pallas_interpret():
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.accelerator.is_available():
            torch.accelerator.synchronize()


def compute_repeat(
    fn: Callable[[], object],
    *,
    target_ms: float = 100.0,
    min_repeat: int = 10,
    max_repeat: int = 1000,
    estimate_runs: int = 5,
    default_cudagraph: bool = False,
) -> int:
    """
    Estimate how many repetitions are needed to collect a stable benchmark for a
    single function call, mirroring Triton's ``do_bench`` heuristic while
    clamping the result between ``min_repeat`` and ``max_repeat``.
    """
    from triton import runtime

    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]
    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]

    # Warm the pipeline once before collecting timing samples.
    fn()
    di.synchronize()
    benchmark_function = _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph)

    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(estimate_runs):
        _compat.safe_clear_cache()
        benchmark_function()
    end_event.record()
    di.synchronize()

    estimate_ms = start_event.elapsed_time(end_event) / max(estimate_runs, 1)
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        return max_repeat

    repeat = int(target_ms / estimate_ms)
    return max(min_repeat, min(max_repeat, max(1, repeat)))


def compute_repeat_generic(
    fn: Callable[[], object],
    *,
    target_ms: float = 100.0,
    min_repeat: int = 10,
    max_repeat: int = 1000,
    estimate_runs: int = 5,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
) -> int:
    """
    Estimate how many repetitions are needed using wall-clock timing.
    Used for backends that don't have Triton's event-based timing (e.g., Pallas/TPU).
    """
    # Warm the pipeline once before collecting timing samples.
    out = fn()
    synchronize_device(out)

    clear_l2 = _make_l2_cache_clearer()
    start = time.perf_counter()
    for _ in range(estimate_runs):
        clear_l2()
        out = fn()
    synchronize_device(out)
    end = time.perf_counter()

    estimate_ms = (end - start) * 1000 / max(estimate_runs, 1)
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        return max_repeat

    repeat = int(target_ms / estimate_ms)
    return max(min_repeat, min(max_repeat, max(1, repeat)))


def interleaved_bench(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,
) -> list[float]:
    """
    Benchmark multiple functions at once, interleaving their executions to reduce
    the impact of external factors (e.g., load, temperature) on the
    measurements.

    Args:
        fns: List of functions to benchmark
        repeat: Number of times to repeat each benchmark
        desc: Optional description for progress bar
    """
    from triton import runtime

    # warmup
    for fn in fns:
        fn()
    _compat.safe_clear_cache()
    di = runtime.driver.active.get_device_interface()  # type: ignore[attr-defined]
    start_events = [
        [di.Event(enable_timing=True) for _ in range(repeat)] for _ in range(len(fns))
    ]
    end_events = [
        [di.Event(enable_timing=True) for _ in range(repeat)] for _ in range(len(fns))
    ]

    di.synchronize()
    benchmark_functions = [
        _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph) for fn in fns
    ]

    # When a description is supplied we show a progress bar so the user can
    # track the repeated benchmarking loop.
    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    for i in iterator:
        for j in range(len(benchmark_functions)):
            _compat.safe_clear_cache()
            start_events[j][i].record()
            benchmark_functions[j]()
            end_events[j][i].record()
    di.synchronize()

    return [
        statistics.median(
            [
                s.elapsed_time(e)
                for s, e in zip(start_events[j], end_events[j], strict=True)
            ]
        )
        for j in range(len(fns))
    ]


def interleaved_bench_generic(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
) -> list[float]:
    """
    Benchmark multiple functions using wall-clock timing.
    Used for backends that don't have Triton's event-based timing (e.g., Pallas/TPU).
    """
    # warmup
    out: object = None
    for fn in fns:
        out = fn()
    synchronize_device(out)

    clear_l2 = _make_l2_cache_clearer()
    all_times: list[list[float]] = [[] for _ in range(len(fns))]

    iterator = iter_with_progress(
        range(repeat),
        total=repeat,
        description=desc,
        enabled=desc is not None,
    )
    for _i in iterator:
        for j in range(len(fns)):
            clear_l2()
            synchronize_device(out)
            start = time.perf_counter()
            out = fns[j]()
            synchronize_device(out)
            end = time.perf_counter()
            all_times[j].append((end - start) * 1000)  # convert to ms

    return [statistics.median(times) for times in all_times]


def paired_device_micros_bench(
    fns: list[Callable[..., object]],
    reference_fn: Callable[..., object],
    *,
    device_micros_fn: Callable[[Callable[[], object]], float],
    desc: str | None = None,
) -> list[tuple[float, float]]:
    """Paired device-µs timing for the final-pick re-rank.

    Times each candidate and ``reference_fn`` via ``device_micros_fn`` and returns
    ``(candidate_micros, candidate_micros - reference_micros)`` per candidate (negative delta
    = faster on-device). Unlike single-call wall-clock µs, which on small Pallas
    matmuls is ~96-98% dispatch overhead, ``device_micros_fn`` reports on-chip µs so
    configs differing by a few µs are separable. An unusable trace yields
    ``(inf, inf)`` so the caller deranks it.
    """
    iterator = iter_with_progress(
        range(len(fns)),
        total=len(fns),
        description=desc,
        enabled=desc is not None,
    )
    results: list[tuple[float, float]] = []
    for j in iterator:
        candidate_micros = device_micros_fn(fns[j])
        reference_micros = device_micros_fn(reference_fn)
        if math.isfinite(candidate_micros) and math.isfinite(reference_micros):
            results.append((candidate_micros, candidate_micros - reference_micros))
        else:
            results.append((math.inf, math.inf))
    return results


# 100 calls/trace keeps the per-event average stable to ~0.1µs (measured), well
# below the multi-µs deltas the re-rank separates.
_PALLAS_AUTOTUNE_DEVICE_MICROS_DEFAULT_N_CALLS = 100
_PALLAS_AUTOTUNE_DEVICE_MICROS_DEFAULT_N_WARMUP = 5
# Min event count for a ``/device:TPU:0`` line to count as a kernel sample: above
# the DVFS counter lines (<=~17 events), below the kernel band (>=~40). Per-call
# us divides by the actual count, so partial-flush traces stay unbiased.
_PALLAS_AUTOTUNE_DEVICE_MICROS_MIN_TRACE_EVENTS = 20


def _autotune_rank_by_device_micros() -> bool:
    """True unless ``HELION_AUTOTUNE_PALLAS_RANK_BY`` opts out of device-µs ranking.

    Default ``device_time``; any other value (e.g. ``wall_time``) falls back to the
    legacy wall-clock paired-sample ranking. Pallas-only (the device-µs path is
    jax.profiler-based); inert on other backends.
    """
    value = (
        os.environ.get("HELION_AUTOTUNE_PALLAS_RANK_BY", "device_time").strip().lower()
    )
    return value == "device_time"


def _pallas_device_micros_for_fn(
    fn: Callable[[], object],
    *,
    n_calls: int,
    n_warmup: int,
) -> float:
    """Per-call on-device µs for ``fn`` under a single ``jax.profiler`` trace.

    Wraps ``n_calls`` invocations in one ``start_trace``/``stop_trace`` window,
    parses the ``.xplane.pb``, and returns ``total_ns / count / 1000`` for the
    dominant ``/device:TPU:0`` event (count >=
    ``_PALLAS_AUTOTUNE_DEVICE_MICROS_MIN_TRACE_EVENTS``, which excludes the DVFS
    counter line). Returns ``+inf`` when the trace is unusable (no ``jax``, no
    xplane/TPU plane, too few events). Kernel exceptions from ``fn`` propagate.
    """
    try:
        import jax  # pyrefly: ignore[missing-module-attribute]
    except ImportError:
        return math.inf

    # Warmup outside the trace window so first-call compile doesn't pollute it.
    for _ in range(n_warmup):
        fn()

    with tempfile.TemporaryDirectory(prefix="helion_autotune_device_micros_") as td:
        jax.profiler.start_trace(td)
        last_out: object = None
        try:
            for _ in range(n_calls):
                last_out = fn()
        finally:
            # Block on the last output so stop_trace sees every device event.
            if last_out is not None:
                with contextlib.suppress(TypeError, AttributeError):
                    jax.block_until_ready(last_out)
            jax.profiler.stop_trace()

        matches = glob.glob(os.path.join(td, "**", "*.xplane.pb"), recursive=True)
        if not matches:
            return math.inf

        pd = jax.profiler.ProfileData.from_file(matches[0])
        best_total_ns = 0
        best_count = 0
        for plane in pd.planes:
            if plane.name != "/device:TPU:0":
                continue
            for line in plane.lines:
                totals: dict[str, int] = {}
                counts: dict[str, int] = {}
                for ev in line.events:
                    totals[ev.name] = totals.get(ev.name, 0) + ev.duration_ns
                    counts[ev.name] = counts.get(ev.name, 0) + 1
                for name, total in totals.items():
                    if (
                        counts[name] >= _PALLAS_AUTOTUNE_DEVICE_MICROS_MIN_TRACE_EVENTS
                        and total > best_total_ns
                    ):
                        best_total_ns, best_count = total, counts[name]
        if best_total_ns == 0:
            return math.inf
        return best_total_ns / best_count / 1000.0


def make_pallas_paired_device_micros_bench(
    *,
    n_calls: int = _PALLAS_AUTOTUNE_DEVICE_MICROS_DEFAULT_N_CALLS,
    n_warmup: int = _PALLAS_AUTOTUNE_DEVICE_MICROS_DEFAULT_N_WARMUP,
) -> Callable[..., list[tuple[float, float]]] | None:
    """Paired device-µs bench closure for the Pallas backend, or None.

    Returns None when the user opted out via ``HELION_AUTOTUNE_PALLAS_RANK_BY=wall_time``.
    Otherwise the closure has the :func:`paired_device_micros_bench` signature and
    captures ``n_calls`` / ``n_warmup``.
    """
    if not _autotune_rank_by_device_micros():
        return None

    def _bench(
        fns: list[Callable[..., object]],
        reference_fn: Callable[..., object],
        *,
        desc: str | None = None,
    ) -> list[tuple[float, float]]:
        def _device_micros_fn(fn: Callable[[], object]) -> float:
            return _pallas_device_micros_for_fn(fn, n_calls=n_calls, n_warmup=n_warmup)

        return paired_device_micros_bench(
            fns,
            reference_fn,
            device_micros_fn=_device_micros_fn,
            desc=desc,
        )

    return _bench


def _coerce_triton_timing(val: object) -> float | tuple[float, ...]:
    """When ``_summarize_statistics`` is given a Tensor, some builds return Tensor outputs."""
    if isinstance(val, torch.Tensor):
        return float(val.detach().cpu().item())
    if isinstance(val, tuple):
        return tuple(
            float(x.detach().cpu().item()) if isinstance(x, torch.Tensor) else float(x)
            for x in val
        )
    return float(val)


def _summarize_statistics_fallback(
    times: list[float] | object,
    quantiles: list[float] | None,
    return_mode: str,
) -> float | tuple[float, ...]:
    """Fallback statistics summarizer when triton.testing._summarize_statistics is unavailable.

    Handles both Python lists and torch tensors.
    """
    if isinstance(times, torch.Tensor):
        times_list = times.cpu().tolist()
    elif isinstance(times, list):
        times_list = times
    else:
        times_list = list(times)  # type: ignore[arg-type]

    if return_mode == "min":
        return min(times_list)
    if return_mode == "max":
        return max(times_list)
    if return_mode == "mean":
        return statistics.mean(times_list)
    if return_mode == "median":
        return statistics.median(times_list)
    # "all" mode
    if quantiles is not None:
        sorted_times = sorted(times_list)
        n = len(sorted_times)
        result = []
        for q in quantiles:
            idx = min(int(q * n), n - 1)
            result.append(sorted_times[idx])
        return tuple(result)
    return statistics.median(times_list)


# This function is copied from triton._testing.do_bench with modification
# to make sure different ranks run the benchmark for the same number
# of times.
def do_bench(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    process_group_name: str | None = None,
    *,
    default_cudagraph: bool = False,
) -> float | tuple[float, ...]:
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    from triton import runtime
    from triton.testing import _summarize_statistics

    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()  # pyrefly: ignore

    fn()
    di.synchronize()
    # Backward benchmarks mutate grad fields between iterations, so keep their
    # existing launch path.
    benchmark_function = (
        fn
        if grad_to_none is not None
        else _maybe_cudagraph_replay(fn, default_enabled=default_cudagraph)
    )

    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # pyrefly: ignore

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        _compat.safe_clear_cache()
        benchmark_function()
    end_event.record()
    di.synchronize()
    estimate_ms = sync_object(
        start_event.elapsed_time(end_event) / 5, process_group_name=process_group_name
    )

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        benchmark_function()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run if supported
        _compat.safe_clear_cache()
        # record time of `fn`
        start_event[i].record()
        benchmark_function()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event, strict=True)]
    # Ascend Triton expects a Tensor here; CUDA upstream returns plain floats from a list.
    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            raw = _summarize_statistics(
                torch.tensor(times), quantiles, return_mode
            )  # pyrefly: ignore
        except Exception:
            raw = _summarize_statistics(times, quantiles, return_mode)  # pyrefly: ignore
        return _coerce_triton_timing(raw)
    return _summarize_statistics(times, quantiles, return_mode)  # pyrefly: ignore


def do_bench_generic(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    process_group_name: str | None = None,
    *,
    default_cudagraph: bool = False,  # accepted for API symmetry; wall-clock timing doesn't use CG
) -> float | tuple[float, ...]:
    """
    Benchmark using wall-clock timing for backends without Triton event timing.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    out = fn()
    synchronize_device(out)

    clear_l2 = _make_l2_cache_clearer()

    # Estimate the runtime of the function
    synchronize_device(out)
    start = time.perf_counter()
    for _ in range(5):
        clear_l2()
        out = fn()
    synchronize_device(out)
    end = time.perf_counter()
    estimate_ms = sync_object(
        (end - start) * 1000 / 5, process_group_name=process_group_name
    )

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    times: list[float] = []
    for _i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        clear_l2()
        synchronize_device(out)
        t0 = time.perf_counter()
        out = fn()
        synchronize_device(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convert to ms
    return _summarize_statistics_fallback(times, quantiles, return_mode)


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


@contextlib.contextmanager
def _quiet_torch_npu_profiler_parse_info() -> Iterator[None]:
    """Drop Ascend profiler parse INFO lines (``Start parsing…`` / ``All profiling…``).

    Only runs when ``torch.npu.is_available()``; CUDA/CPU runs never apply this patch.
    Set ``HELION_SHOW_NPU_PROFILER_LOGS=1`` to see them again.
    """
    if _env_truthy("HELION_SHOW_NPU_PROFILER_LOGS"):
        yield
        return
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        yield
        return
    try:
        import torch_npu.profiler.analysis._profiling_parser as _prof_parser  # type: ignore[import-not-found]
        import torch_npu.profiler.analysis.prof_view.cann_parse._cann_export as _cann_export  # type: ignore[import-not-found]
    except ImportError:
        yield
        return

    def _noop(_message: str) -> None:
        return None

    prev_pp = _prof_parser.print_info_msg
    prev_ce = _cann_export.print_info_msg
    _prof_parser.print_info_msg = _noop  # type: ignore[assignment]
    _cann_export.print_info_msg = _noop  # type: ignore[assignment]
    try:
        yield
    finally:
        _prof_parser.print_info_msg = prev_pp
        _cann_export.print_info_msg = prev_ce


@functools.cache
def _npu_triton_do_bench_available() -> bool:
    """True if ``triton.testing.do_bench_npu`` is importable (triton-ascend profiler bench)."""
    try:
        from triton.testing import do_bench_npu as _triton_do_bench_npu  # noqa: F401
    except Exception:
        return False
    return True


@functools.cache
def _npu_triton_do_bench_returns_ms_already() -> bool:
    """True if ``triton.testing.do_bench_npu`` already returns ms (triton-ascend).

    Stock PyTorch Triton reads ``op_statistic.csv`` ``Avg Time(us)`` without scaling
    (microseconds); Helion divides by 1000. Ascend fork uses ``kernel_details`` and
    divides by ``active * 1e3`` (already ms). Detection: ``clear_l2_cache`` in
    signature or ``ascend`` in ``__module__``.
    """
    try:
        from triton.testing import do_bench_npu as triton_do_bench_npu
    except Exception:
        return False
    try:
        if "clear_l2_cache" in inspect.signature(triton_do_bench_npu).parameters:
            return True
    except (TypeError, ValueError):
        pass
    mod = getattr(triton_do_bench_npu, "__module__", "") or ""
    return "ascend" in mod


def _npu_profiler_scalar_to_milliseconds(raw: float) -> float:
    if _npu_triton_do_bench_returns_ms_already():
        return raw
    return raw / 1000.0


def _scalar_do_bench_npu_timing(result: object) -> float:
    """Normalize ``do_bench_npu`` output to **milliseconds** for Helion tables."""
    if isinstance(result, torch.Tensor):
        raw = float(result.detach().cpu().item())
    else:
        raw = float(result)
    return _npu_profiler_scalar_to_milliseconds(raw)


def npu_benchmark_results_timing_caption() -> str:
    """One-line stderr note for NPU ``run_example`` tables (empty if not NPU)."""
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return ""
    if not _npu_triton_do_bench_available():
        return "Timing: triton do_bench events (ms; triton.testing.do_bench_npu unavailable)."
    if _npu_triton_do_bench_returns_ms_already():
        return "Timing: triton do_bench_npu (ms; triton-ascend kernel_details)."
    return "Timing: triton do_bench_npu (ms; op_statistic Avg Time(us) ÷ 1000)."


def do_bench_npu(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
    process_group_name: str | None = None,
    *,
    default_cudagraph: bool = False,
) -> float | tuple[float, ...]:
    """NPU profiler bench via ``triton.testing.do_bench_npu``; return value in ms.

    Ascend fork: ``warmup``/``rep`` are profiler iteration counts. Stock: same names;
    raw ``Avg Time(us)`` is scaled to ms when needed (see
    :func:`_npu_triton_do_bench_returns_ms_already`). Fallback path uses wall-clock
    timing when ``triton.testing.do_bench_npu`` is unavailable.
    """
    try:
        from triton.testing import do_bench_npu as triton_do_bench_npu
    except ImportError:
        return _do_bench_npu_fallback(
            fn, warmup, rep, grad_to_none, quantiles, return_mode
        )

    if grad_to_none is not None:
        for x in grad_to_none:
            x.grad = None

    with _quiet_torch_npu_profiler_parse_info():
        result = triton_do_bench_npu(
            fn,
            warmup=warmup,
            active=rep,
        )

    if isinstance(result, list):
        if not result:
            return 0.0
        r = result[0]
    else:
        r = result
    return _scalar_do_bench_npu_timing(r)


def _do_bench_npu_fallback(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: str = "mean",
) -> float | tuple[float, ...]:
    """Wall-clock fallback for NPU benchmarking when ``triton.testing.do_bench_npu`` is unavailable."""
    from triton.testing import _summarize_statistics

    assert return_mode in ["min", "max", "mean", "median", "all"]

    fn()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(5):
        _compat.safe_clear_cache()
        fn()
    torch.npu.synchronize()
    end = time.perf_counter()
    estimate_ms = sync_object((end - start) * 1000 / 5)

    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    for _ in range(n_warmup):
        fn()

    times: list[float] = []
    for _i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        _compat.safe_clear_cache()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # convert to ms

    return _coerce_triton_timing(
        _summarize_statistics(torch.tensor(times), quantiles, return_mode)
    )


def _npu_profiler_iteration_params(repeat: int) -> tuple[int, int]:
    """Map ``interleaved_bench``-style repeat to Ascend ``do_bench_npu`` counts."""
    active = max(30, min(int(repeat), 500))
    warmup = max(5, min(active // 10, 50))
    return warmup, active


def interleaved_bench_npu(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,
) -> list[float]:
    """Benchmark multiple functions on Ascend NPU using ``triton.testing.do_bench_npu``.

    Upstream ``do_bench_npu`` takes a single callable, so this runs one profiler
    session per function. Falls back to wall-clock timing when unavailable.
    """
    if not fns:
        return []

    warmup_n, active_n = _npu_profiler_iteration_params(repeat)

    try:
        from triton.testing import do_bench_npu as triton_do_bench_npu
    except ImportError:
        times_fb: list[float] = []
        for fn in fns:
            try:
                times_fb.append(
                    cast(
                        "float",
                        _do_bench_npu_fallback(
                            cast("Callable[[], Any]", fn),
                            warmup=warmup_n,
                            rep=active_n,
                            return_mode="median",
                        ),
                    )
                )
            except Exception as e:
                _log.debug(
                    "interleaved_bench_npu fallback: callable failed (%s): %s",
                    type(e).__qualname__,
                    e,
                )
                times_fb.append(float("inf"))
        return times_fb

    times: list[float] = []
    iterator = iter_with_progress(
        range(len(fns)),
        total=len(fns),
        description=desc,
        enabled=desc is not None,
    )
    for j in iterator:
        try:
            with _quiet_torch_npu_profiler_parse_info():
                raw = triton_do_bench_npu(
                    fns[j],
                    warmup=warmup_n,
                    active=active_n,
                )
        except Exception as e:
            _log.debug(
                "interleaved_bench_npu: callable %s failed (%s): %s",
                j,
                type(e).__qualname__,
                e,
            )
            times.append(float("inf"))
            continue
        if isinstance(raw, list):
            if not raw:
                times.append(float("inf"))
                continue
            raw = raw[0]
        times.append(_scalar_do_bench_npu_timing(raw))
    return times


def default_do_bench() -> Callable[..., Any]:
    """Bench hook for autotuning: NPU uses profiler timing when available, else events.

    - **Ascend NPU** with ``triton.testing.do_bench_npu``: :func:`do_bench_npu`.
    - **Otherwise** (incl. NPU without the profiler fn): :func:`do_bench` (event-based).
    """
    if (
        hasattr(torch, "npu")
        and torch.npu.is_available()
        and _npu_triton_do_bench_available()
    ):
        return do_bench_npu
    return do_bench


def default_interleaved_bench() -> Callable[..., list[float]]:
    """Interleaved bench for autotune rebenchmarking; NPU uses profiler path when available."""
    if (
        hasattr(torch, "npu")
        and torch.npu.is_available()
        and _npu_triton_do_bench_available()
    ):
        return interleaved_bench_npu
    return interleaved_bench
