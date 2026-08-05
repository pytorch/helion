"""Benchmarking for Metal autotuning.

Neither shared benchmark works here.  The module-level ``do_bench`` is Triton's
CUDA-event timer and Triton is not installed on macOS, so every candidate
raises.  ``do_bench_generic`` runs, but it times one call between two
synchronizations, and dispatching a Metal kernel from Python costs an
appreciable fraction of what these kernels take -- so most of a sample is
overhead and configs that differ by 15% measure the same.  Each sample here
times a batch of back-to-back calls instead.

A cold kernel's first dispatches run several times slower than its steady
state, and it is the estimate that fixes the batch size for every sample after
it, so the kernel is warmed before it is probed (see ``_WARMUP_CALLS``).

Neither function resolves differences below a few percent, but they fail
differently.  :func:`do_bench_metal` scores every candidate as it is compiled
and the search's surrogate is fitted on those numbers, so a config it scores
badly is never explored again.  :func:`interleaved_bench_metal` re-ranks the
survivors with drift held common across them; it decides which one wins, but
never sees what the first function dropped.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import Any

from ...autotuner.benchmarking import _summarize_statistics_fallback
from ...autotuner.benchmarking import synchronize_device

if TYPE_CHECKING:
    from collections.abc import Callable

#: Target duration of one timed sample.
#:
#: A sample carries a fixed cost of roughly 0.25ms (two synchronizations plus
#: the first call after one) that ``_time_batch_ms`` divides across the batch
#: but never removes, so the target must sit past where a kernel's measured
#: time converges: 20ms lands a typical kernel on batch 60-90, within a couple
#: of percent of its batch-256 reading, for ~250ms per config against ~4.4s of
#: MSL compilation.
#:
#: This raises the ceiling on resolution; it does not make the harness exact.
#: Below a few percent, ranking is a coin flip (see
#: ``test_identical_candidates_measure_the_same``).
_SAMPLE_TARGET_MS = 20.0

#: Upper bound on calls per sample, so a pathologically cheap kernel cannot
#: make a single sample run away.
_MAX_BATCH = 512

#: Burn-in before estimating, as a call count and a time budget -- whichever is
#: smaller.  A cold probe reads several times high and would fix the batch size
#: too small for every sample after it.  The time bound keeps an expensive
#: kernel from spending seconds here for no benefit; it gets ``batch == 1``
#: regardless.
_WARMUP_CALLS = 32
_WARMUP_MS = 25.0

#: Floors on the number of samples, independent of the caller's budget.
#:
#: The autotuner asks for ``warmup=1, rep=50`` -- budgets written for
#: event-timed CUDA kernels measured one call at a time.  Divided by a 20ms
#: sample they round down to one warmup sample and two timed ones, which is
#: not enough to take a median over.  The budgets are honored when they ask
#: for *more* than this.
_MIN_WARMUP_SAMPLES = 2
_MIN_SAMPLES = 8

#: Floor on interleaved rounds, for the same reason (see
#: :func:`interleaved_bench_metal`, where ``repeat`` is a call budget).
_MIN_ROUNDS = 3


def _batch_size_for(per_call_ms: float) -> int:
    """Number of back-to-back calls that make up one timed sample."""
    if per_call_ms <= 0:
        return _MAX_BATCH
    return max(1, min(_MAX_BATCH, int(_SAMPLE_TARGET_MS / per_call_ms)))


def _time_batch_ms(fn: Callable[[], Any], batch: int) -> float:
    """Mean per-call time over ``batch`` back-to-back calls and one sync."""
    synchronize_device()
    start = time.perf_counter()
    output = None
    for _ in range(batch):
        output = fn()
    synchronize_device()
    del output
    return (time.perf_counter() - start) * 1000 / batch


def _warm_up(fn: Callable[[], Any]) -> None:
    """Bring ``fn`` to steady state before anything is measured from it."""
    fn()
    synchronize_device()
    # The first call reads high, being cold, which is exactly the right bias
    # for sizing a burn-in: it over-estimates the cost and so under-shoots the
    # budget rather than blowing through it.
    cold_ms = _time_batch_ms(fn, 1)
    calls = max(1, min(_WARMUP_CALLS, int(_WARMUP_MS / max(cold_ms, 1e-6))))
    _time_batch_ms(fn, calls)


def _estimate_per_call_ms(fn: Callable[[], Any], runs: int = 5) -> float:
    """Estimate the amortized per-call cost, to size the timed batch.

    Assumes ``fn`` is already warm (see :func:`_warm_up`); this is a sizing
    probe, not a measurement.  A short probe reads high, which picks too small
    a batch, which keeps it high, so each pass re-times at the implied batch
    and keeps the ``min`` (noise only adds time).
    """
    estimate = _time_batch_ms(fn, runs)
    batch = runs
    for _ in range(2):
        refined_batch = _batch_size_for(estimate)
        if refined_batch <= batch:
            break
        estimate = min(estimate, _time_batch_ms(fn, refined_batch))
        batch = refined_batch
    return estimate


def do_bench_metal(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: object = None,
    quantiles: list[float] | None = None,
    return_mode: str = "median",
    process_group_name: str | None = None,
    *,
    default_cudagraph: bool = False,  # accepted for API symmetry
) -> float | tuple[float, ...]:
    """Time ``fn`` on MPS, amortizing per-launch host overhead.

    ``warmup`` and ``rep`` are millisecond budgets, as in the shared
    benchmarks, but are spent in whole batches rather than single calls.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    _warm_up(fn)

    per_call_ms = _estimate_per_call_ms(fn)
    batch = _batch_size_for(per_call_ms)
    sample_ms = max(per_call_ms * batch, 1e-6)
    n_warmup = max(_MIN_WARMUP_SAMPLES, int(warmup / sample_ms))
    n_repeat = max(_MIN_SAMPLES, int(rep / sample_ms))

    for _ in range(n_warmup):
        _time_batch_ms(fn, batch)

    times = [_time_batch_ms(fn, batch) for _ in range(n_repeat)]
    return _summarize_statistics_fallback(times, quantiles, return_mode)


def interleaved_bench_metal(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    default_cudagraph: bool = False,  # accepted for API symmetry
    max_total_ms: float | None = None,
) -> list[float]:
    """Interleaved counterpart of :func:`do_bench_metal`.

    Interleaving holds slow drift (thermal, other GPU work) common across
    candidates; batching removes the per-launch overhead that would otherwise
    compress the differences between them.

    Each candidate gets its **own** batch size: ``_time_batch_ms`` already
    returns a per-call time, so a shared batch equalizes nothing and only
    drags every candidate down to the slowest one's overhead residual.

    The visit order is fixed; with per-candidate batches there is no
    measurable predecessor penalty.

    ``min`` is a fair summary across candidates: each is sampled once per
    round, so the order-statistic bias is common to them.
    """
    from ...autotuner.benchmarking import iter_with_progress

    estimates = []
    batches = []
    for fn in fns:
        _warm_up(fn)
        estimates.append(_estimate_per_call_ms(fn))
        batches.append(_batch_size_for(estimates[-1]))
    all_times: list[list[float]] = [[] for _ in fns]

    # ``repeat`` is a budget in *calls*, matching the shared benchmarks (one
    # call per candidate per round).  Here a round is a whole batch, so rounds
    # are ``repeat // max(batches)`` -- spending ``repeat`` rounds would
    # overshoot the budget by the batch size.  The cheapest candidate sets the
    # count; ``_MIN_ROUNDS`` bounds it from below.
    rounds = max(_MIN_ROUNDS, repeat // max(batches, default=1))
    if max_total_ms is not None:
        # The caller (``BaseSearch.rebenchmark``) caps the whole pass near
        # the sequential-window budget it replaced.  Without this the
        # keyword it always passes has nowhere to land.
        per_round_ms = sum(b * e for b, e in zip(batches, estimates, strict=True))
        rounds = max(
            _MIN_ROUNDS, min(rounds, int(max_total_ms / max(per_round_ms, 1e-6)))
        )

    iterator = iter_with_progress(
        range(rounds),
        total=rounds,
        description=desc,
        enabled=desc is not None,
    )
    for _ in iterator:
        for j, fn in enumerate(fns):
            all_times[j].append(_time_batch_ms(fn, batches[j]))

    return [min(times) for times in all_times]
