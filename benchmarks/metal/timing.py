"""Timing primitives for the Metal benchmark.

The headline, ``latency``, mirrors TritonBench's ``_do_bench_tpu`` at the CI
pin: wall clock around one warm call plus a device-wide completion, with
warmup and repeat counts sized from a 5-call estimate.  It is what the
dashboard's TPU column measures.

``kernel_ms`` is a diagnostic: GPU time over back-to-back calls.  A single
call's MPS event pair also spans any GPU idle time waiting on the host, so it
is only meaningful over a batch, and only when the host enqueues the batch
faster than the GPU runs it; ``enqueue_ms`` says whether it did.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from .providers import Bound

NS_TO_MS = 1e-6
# Retained batch outputs are capped by bytes so large cells cannot exhaust
# unified memory; the batch shrinks instead.
BATCH_BYTES_CAP = 2 << 30
BATCH_CALLS = 100
# TritonBench's counts when the estimate rounds to zero.
DEFAULT_N_WARMUP = 200
DEFAULT_N_REP = 1000


def resolve_warmup_and_rep(estimate_ms: float) -> tuple[int, int]:
    """TritonBench's warmup/rep budgets (ms) by estimated per-call time."""
    if estimate_ms <= 10:
        return 25, 100
    return 3000, 3000


def latency(bound: Bound) -> list[float]:
    """Per-call wall times (ms) of one warm call plus completion each."""

    def timed_call() -> float:
        t0 = time.perf_counter_ns()
        out = bound.call()
        bound.complete(out)
        t1 = time.perf_counter_ns()
        # ``out`` is bound so it stays alive through completion and is
        # released after t1, outside the timed region.
        return (t1 - t0) * NS_TO_MS

    timed_call()
    estimate_ms = sum(timed_call() for _ in range(5)) / 5
    warmup_ms, rep_ms = resolve_warmup_and_rep(estimate_ms)
    if estimate_ms == 0:
        n_warmup, n_repeat = DEFAULT_N_WARMUP, DEFAULT_N_REP
    else:
        n_warmup = max(1, int(warmup_ms / estimate_ms))
        n_repeat = max(1, int(rep_ms / estimate_ms))
    for _ in range(n_warmup):
        timed_call()
    return [timed_call() for _ in range(n_repeat)]


def batch_size(output_bytes: int) -> int:
    return max(1, min(BATCH_CALLS, BATCH_BYTES_CAP // max(1, output_bytes)))


@dataclass
class Batch:
    """Per-call means over one batch of back-to-back calls."""

    kernel_ms: float  # device time (MPS events) or wall time to completion
    enqueue_ms: float  # host time to issue the calls, before completion
    source: str  # "mps_events" or "wall_batched"


def kernel_ms(bound: Bound, output_bytes: int) -> Batch:
    """Time a batch of back-to-back calls.

    Outputs are retained until the batch completes so each call keeps its own
    buffers, as it would in a real program that uses its results.  An
    untimed batch runs first: in a fresh process the first batch also pays
    for first touching its outputs' pages, and the allocators keep those
    buffers for the timed batch.  When ``enqueue_ms`` is close to
    ``kernel_ms`` the GPU was waiting on the host, so ``kernel_ms`` is the
    dispatch rate, not the device time.
    """
    n = batch_size(output_bytes)
    bound.complete([bound.call() for _ in range(n)])
    events = bound.events() if bound.events is not None else None
    bound.sync()
    if events is not None:
        events[0].record()
    t0 = time.perf_counter_ns()
    outs = [bound.call() for _ in range(n)]
    t1 = time.perf_counter_ns()
    if events is not None:
        events[1].record()
    bound.complete(outs)
    t2 = time.perf_counter_ns()
    enqueue_ms = (t1 - t0) * NS_TO_MS / n
    if events is not None:
        return Batch(events[0].elapsed_time(events[1]) / n, enqueue_ms, "mps_events")
    return Batch((t2 - t0) * NS_TO_MS / n, enqueue_ms, "wall_batched")


def sync_floor(
    empty: Callable[[], object], complete: Callable[[object], None]
) -> float:
    """Median wall time (ms) of the smallest call plus completion."""
    samples = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        out = empty()
        complete(out)
        samples.append((time.perf_counter_ns() - t0) * NS_TO_MS)
    return median(samples)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
