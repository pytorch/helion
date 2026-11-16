from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import supported_activities

from helion._dist_utils import all_ranks_agree
from helion._testing import DEVICE
from helion._testing import skipIfRocm
from helion.autotuner import benchmark_job
from helion.autotuner import benchmarking
from helion.autotuner.benchmark_job import KernelTimerProbeJob
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_worker import BenchmarkTimeout
from helion.autotuner.benchmark_worker import BenchmarkWorker
from helion.autotuner.benchmark_worker import BenchmarkWorkerUnkillable

if TYPE_CHECKING:
    from collections.abc import Callable


def test_batched_replay_calls_fold_short_kernels_only() -> None:
    assert benchmarking._batched_replay_calls(math.inf) == 1
    assert benchmarking._batched_replay_calls(0.0) == 1
    assert benchmarking._batched_replay_calls(200.0) == 1
    assert benchmarking._batched_replay_calls(64.0) == 1
    assert benchmarking._batched_replay_calls(30.0) == 3
    # The single-replay floor (~6 us) still folds a bounded number of calls.
    assert benchmarking._batched_replay_calls(6.0) == 11
    assert (
        benchmarking._batched_replay_calls(0.5)
        == benchmarking._BATCHED_REPLAY_MAX_CALLS
    )


@skipIfRocm("CUDA graph benchmarking is NVIDIA-only")
@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA")
def test_kernels_below_the_launch_floor_are_timed_as_batched_flushed_graphs(
    monkeypatch,
) -> None:
    """A single graph replay per event pair reads the graph launch floor for
    any kernel shorter than it.  Such a kernel is timed as one graph holding
    ``calls`` x [L2 flush; call] against a graph of the flushes alone: every
    sample is one replay of each, and the per-call time is their difference
    over ``calls``.  The floor estimate is pinned so the batch is known, and
    the replays are counted rather than timed."""
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    # Every single replay is estimated at 4 us: ceil(64 / 4) = 16 calls.
    monkeypatch.setattr(
        benchmarking, "_single_replay_estimate_us", lambda *args, **kwargs: 4.0
    )
    make_batched = benchmarking._make_batched_cudagraph_replay
    batches: list[int] = []
    replays = {"kernel": 0, "flush": 0}

    def counted_batched(
        fn: Callable[[], object] | None,
        clear_cache: Callable[[], None],
        calls: int,
        keep_outputs: bool = False,
    ) -> Callable[[], None]:
        replay = make_batched(fn, clear_cache, calls, keep_outputs=keep_outputs)
        kind = "flush" if fn is None else "kernel"
        if fn is not None:
            batches.append(calls)

        def counting_replay() -> None:
            replays[kind] += 1
            replay()

        return counting_replay

    monkeypatch.setattr(benchmarking, "_make_batched_cudagraph_replay", counted_batched)
    small = torch.randn(1 << 14, device=DEVICE)
    outs = [torch.empty_like(small), torch.empty_like(small)]

    def first() -> None:
        torch.add(small, 1.0, out=outs[0])

    def second() -> None:
        torch.add(small, 2.0, out=outs[1])

    times = benchmarking.interleaved_bench([first, second], repeat=40)
    assert batches == [16, 16]
    # 40 samples of 16 calls are more than needed: the loop shrinks to 20
    # samples, each one batched replay per kernel and one flush replay.
    assert replays == {"kernel": 40, "flush": 20}
    assert len(times) == 2

    # do_bench batches the same way; the flush-only graph is timed once per
    # _BATCHED_FLUSH_EVERY samples, next to the samples it corrects.
    benchmarking.do_bench(first, fixed_repetitions=10, return_mode="median")
    assert batches == [16, 16, 16]
    assert replays == {"kernel": 50, "flush": 23}


@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA")
def test_resident_batches_are_sized_from_back_to_back_calls(monkeypatch) -> None:
    """A one-call graph replay of a 2.8 us kernel reads ~11 us (graph launch
    latency), which sizes a 6-call batch in which that latency still hides the
    kernels' differences. When a batch's calls run back to back (``off`` and
    ``replay`` modes) it is re-sized from a probe batch's per-call time, the
    deployed regime, so the short kernel gets the full batch; the per-call
    flushed mode keeps its sizing; ``replay`` also fills the batch towards the
    harness's burst for kernels the single-replay sizing leaves unbatched."""
    probes: list[int] = []

    def fake_batched(fns, clear_cache, calls, keep_outputs=False):
        probes.append(calls)
        return ([object()] * len(fns), None)

    def fake_estimate(replay, clear_cache, di, samples=3):
        # The probe batch runs ``calls`` back-to-back 2.8 us kernels.
        return 2.8 * probes[-1]

    monkeypatch.setattr(benchmarking, "_batched_flushed_replays", fake_batched)
    monkeypatch.setattr(benchmarking, "_single_replay_estimate_us", fake_estimate)
    assert benchmarking._batched_replay_calls(10.78) == 6
    assert (
        benchmarking._back_to_back_batch_calls(
            [object()], benchmarking._no_cache_clear, None, 6, mode="off"
        )
        == benchmarking._BATCHED_REPLAY_MAX_CALLS
    )
    assert probes == [6]
    assert (
        benchmarking._back_to_back_batch_calls(
            [object()], benchmarking._no_cache_clear, None, 6, mode="call"
        )
        == 6
    )
    assert (
        benchmarking._back_to_back_batch_calls(
            [object()], benchmarking._no_cache_clear, None, 1, mode="off"
        )
        == 1
    )
    assert probes == [6]
    # A 70 us kernel: single replay 78 us -> 1 call; ``replay`` probes it and
    # fills the batch to ceil(1024 / 70) = 15 calls.
    monkeypatch.setattr(
        benchmarking,
        "_single_replay_estimate_us",
        lambda replay, clear_cache, di, samples=3: 70.0 * probes[-1],
    )
    assert (
        benchmarking._back_to_back_batch_calls(
            [object()], benchmarking._no_cache_clear, None, 1, mode="replay"
        )
        == 15
    )
    assert probes == [6, 1]
    # A probe that cannot be captured keeps the single-replay sizing.
    monkeypatch.setattr(benchmarking, "_batched_flushed_replays", lambda *args: None)
    assert (
        benchmarking._back_to_back_batch_calls(
            [object()], benchmarking._no_cache_clear, None, 3, mode="off"
        )
        == 3
    )


def test_replay_batches_grow_to_the_long_graph_rule(monkeypatch) -> None:
    """``replay`` mode doubles the batch past the 16-call burst until one
    replay lasts 400 us (the catalogue's graph_long_ms rule), at most 256
    calls and within the memory guard; ``off`` keeps the 16-call cap."""
    probes: list[int] = []
    per_call = {"us": 3.0}

    def fake_batched(fns, clear_cache, calls, keep_outputs=False):
        probes.append(calls)
        return ([object()] * len(fns), None)

    monkeypatch.setattr(benchmarking, "_batched_flushed_replays", fake_batched)
    monkeypatch.setattr(
        benchmarking,
        "_single_replay_estimate_us",
        lambda replay, clear_cache, di, samples=3: per_call["us"] * probes[-1],
    )

    def sized(mode: str, calls: int = 16) -> int:
        return benchmarking._back_to_back_batch_calls(
            [object()], benchmarking._no_cache_clear, None, calls, mode=mode
        )

    # 3 us: 16 calls are 48 us -> doubled four times to the 256-call cap
    assert sized("replay") == 256
    assert sized("off") == 16
    # 10 us: 160 us -> 64 calls (640 us); 30 us: 480 us already -> 16
    per_call["us"] = 10.0
    assert sized("replay") == 64
    per_call["us"] = 30.0
    assert sized("replay") == 16
    # 70 us single-replay sizing: filled to 15 calls (1050 us), no growth
    per_call["us"] = 70.0
    assert sized("replay", calls=1) == 15
    assert sized("off") == 16


def _fake_builder_parts(monkeypatch, *, per_call_us: float, fail_above: int | None):
    """Fakes for ``_build_batched_replays``: records executions and
    agreements in order, fails captures above ``fail_above`` calls."""
    events: list[tuple[str, object]] = []

    def fake_estimate(replay, clear_cache, di, samples=3):
        events.append(("estimate", replay))
        return 11.0  # one single replay of a ~3 us kernel reads ~11 us

    def fake_back_to_back(fns, clear_cache, di, calls, *, mode):
        events.append(("probe", calls))
        return 256 if mode == "replay" else 16

    def fake_memory(fns, calls):
        events.append(("memory", calls))
        return calls

    def fake_batched(fns, clear_cache, calls, keep_outputs=False):
        events.append(("capture", (calls, keep_outputs, clear_cache)))
        if fail_above is not None and calls > fail_above:
            return None
        return ([("batch", calls)] * len(fns), None)

    def fake_sync(obj, process_group_name=None):
        events.append(("agree", obj))
        return obj

    monkeypatch.setattr(benchmarking, "_single_replay_estimate_us", fake_estimate)
    monkeypatch.setattr(benchmarking, "_back_to_back_batch_calls", fake_back_to_back)
    monkeypatch.setattr(benchmarking, "_batch_memory_calls", fake_memory)
    monkeypatch.setattr(benchmarking, "_batched_flushed_replays", fake_batched)
    monkeypatch.setattr(benchmarking, "sync_object", fake_sync)
    return events


def test_batch_sizes_are_agreed_before_anything_runs_on_them(monkeypatch) -> None:
    """Every rank agrees on a size before the probe, the memory guard or the
    capture run with it, so a kernel holding a collective is launched the
    same number of times everywhere."""
    events = _fake_builder_parts(monkeypatch, per_call_us=3.0, fail_above=None)
    fn = object()
    calls, replays, flush = benchmarking._build_batched_replays(
        [fn],
        [fn],
        benchmarking._no_cache_clear,
        None,
        mode="replay",
        process_group_name="pg",
    )
    assert calls == 256 and replays == [("batch", 256)] and flush is None
    kinds = [kind for kind, _ in events]
    # estimate -> agree(6) -> probe(6) -> agree(256) -> memory(256) -> agree -> capture
    assert kinds == [
        "estimate",
        "agree",
        "probe",
        "agree",
        "memory",
        "agree",
        "capture",
    ]
    assert events[1] == ("agree", 6)  # ceil(64 / 11) single-replay sizing
    assert events[2] == ("probe", 6)
    assert events[4] == ("memory", 256)
    assert events[6][1] == (256, True, benchmarking._no_cache_clear)


def test_call_regime_keeps_its_batches_light(monkeypatch) -> None:
    """The per-call flush regime neither keeps outputs nor runs the memory
    guard's extra call; its batch flushes between calls and keeps the
    flush-only graph for the subtraction."""
    events = _fake_builder_parts(monkeypatch, per_call_us=3.0, fail_above=None)
    fn = object()

    def clear() -> None:
        pass

    calls, _replays, _flush = benchmarking._build_batched_replays(
        [fn], [fn], clear, None, mode="call", process_group_name=None
    )
    assert calls == 16
    kinds = [kind for kind, _ in events]
    assert "memory" not in kinds
    assert events[-1] == ("capture", (16, False, clear))
    events.clear()
    calls, _replays, _flush = benchmarking._build_batched_replays(
        [fn], [fn], clear, None, mode="off", process_group_name=None
    )
    assert calls == 16
    assert [kind for kind, _ in events].count("memory") == 1
    # a resident batch keeps its outputs and never flushes inside
    assert events[-1] == ("capture", (16, True, benchmarking._no_cache_clear))


def test_failed_batch_captures_halve_loudly_then_fall_back(monkeypatch, caplog) -> None:
    events = _fake_builder_parts(monkeypatch, per_call_us=3.0, fail_above=40)
    fn = object()
    with caplog.at_level(logging.WARNING, logger=benchmarking._log.name):
        calls, replays, _flush = benchmarking._build_batched_replays(
            [fn],
            [fn],
            benchmarking._no_cache_clear,
            None,
            mode="replay",
            process_group_name=None,
        )
    assert calls == 32 and replays == [("batch", 32)]
    captures = [payload[0] for kind, payload in events if kind == "capture"]
    assert captures == [256, 128, 64, 32]
    assert [r.getMessage() for r in caplog.records] == [
        "Batched CUDA graph capture of 256 calls failed; retrying with 128",
        "Batched CUDA graph capture of 128 calls failed; retrying with 64",
        "Batched CUDA graph capture of 64 calls failed; retrying with 32",
    ]
    caplog.clear()
    events = _fake_builder_parts(monkeypatch, per_call_us=3.0, fail_above=1)
    single = object()
    with caplog.at_level(logging.WARNING, logger=benchmarking._log.name):
        calls, replays, flush = benchmarking._build_batched_replays(
            [fn],
            [single],
            benchmarking._no_cache_clear,
            None,
            mode="off",
            process_group_name=None,
        )
    assert (calls, replays, flush) == (1, [single], None)
    assert "timing single flushed replays instead" in caplog.records[-1].getMessage()


def test_batched_repetitions_follow_the_batch_time() -> None:
    # a 256-call batch of a 3 us kernel replays in ~0.8 ms: a 25+100 ms
    # window is 31 warm-up and 125 timed replays, not two and eight
    assert benchmarking._batched_repetitions(25, 100, 0.8) == (31, 125)
    assert benchmarking._batched_repetitions(1000, 500, 0.8) == (1250, 625)
    assert benchmarking._batched_repetitions(25, 100, 500.0) == (1, 1)
    assert benchmarking._batched_repetitions(25, 100, math.inf) == (1, 1)


def test_batched_replay_keeps_every_calls_outputs() -> None:
    """The batched graph holds one output per call (distinct buffers, as a
    real sequence of calls writes), not one buffer the pool hands to every
    call; the flush-only graph keeps nothing."""
    if DEVICE.type != "cuda":
        pytest.skip("requires CUDA")

    def fn() -> torch.Tensor:
        return torch.empty(1 << 20, dtype=torch.uint8, device=DEVICE)

    replay = benchmarking._make_batched_cudagraph_replay(
        fn, benchmarking._no_cache_clear, 4, keep_outputs=True
    )
    outputs = replay.outputs  # type: ignore[attr-defined]
    assert len(outputs) == 4
    assert len({t.data_ptr() for t in outputs}) == 4
    light = benchmarking._make_batched_cudagraph_replay(
        fn, benchmarking._no_cache_clear, 4
    )
    assert light.outputs == []  # type: ignore[attr-defined]
    # a batch that does not flush captures no flush-only graph
    batched = benchmarking._batched_flushed_replays(
        [fn], benchmarking._no_cache_clear, 4, keep_outputs=True
    )
    assert batched is not None and batched[1] is None


@skipIfRocm("CUDA graph benchmarking is NVIDIA-only")
@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA")
def test_batch_memory_guard_counts_what_a_call_keeps(monkeypatch) -> None:
    """The batch is capped so ``calls`` x one call's allocations stay within
    a quarter of the free device memory; a call that keeps nothing and a
    single call are not capped."""
    kept: list[torch.Tensor] = []

    def allocating() -> torch.Tensor:
        return torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)  # 16 MB

    def in_place() -> None:
        kept.append(torch.empty(0, device=DEVICE))

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (1 << 30, 1 << 31))
    # a quarter of 1 GB holds sixteen 16 MB outputs
    assert benchmarking._batch_memory_calls([allocating], 256) == 16
    assert benchmarking._batch_memory_calls([allocating], 8) == 8
    assert benchmarking._batch_memory_calls([allocating], 1) == 1
    # an interleaved population holds every candidate's batch at once
    assert benchmarking._batch_memory_calls([allocating] * 4, 256) == 4
    # a call that keeps nothing is not capped
    assert benchmarking._batch_memory_calls([in_place], 256) == 256


def test_l2_flush_mode_literals(monkeypatch) -> None:
    monkeypatch.delenv("HELION_AUTOTUNE_L2_FLUSH", raising=False)
    assert benchmarking._l2_flush_mode() == "call"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "0")
    assert benchmarking._l2_flush_mode() == "off"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "1")
    assert benchmarking._l2_flush_mode() == "call"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "replay")
    assert benchmarking._l2_flush_mode() == "replay"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", " Burst ")
    assert benchmarking._l2_flush_mode() == "replay"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "call")
    assert benchmarking._l2_flush_mode() == "call"
    # the kernel timer serves the final selection only
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "kernel")
    assert benchmarking._l2_flush_mode() == "call"


def test_final_selection_regime_follows_the_search_regime(monkeypatch) -> None:
    """The final selection keeps the per-call flush and replay regimes of
    the search, but a resident search (HELION_AUTOTUNE_L2_FLUSH=0) hands its
    finalists to the replay regime, where configs that tie L2-resident
    separate; HELION_AUTOTUNE_FINAL_L2_FLUSH pins the regime explicitly."""
    monkeypatch.delenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_L2_FLUSH", raising=False)
    assert benchmarking.final_selection_l2_flush_mode() == "call"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "1")
    assert benchmarking.final_selection_l2_flush_mode() == "call"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "replay")
    assert benchmarking.final_selection_l2_flush_mode() == "replay"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "0")
    assert benchmarking.final_selection_l2_flush_mode() == "replay"
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", "0")
    assert benchmarking.final_selection_l2_flush_mode() == "off"
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", "1")
    assert benchmarking.final_selection_l2_flush_mode() == "call"
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", "")
    assert benchmarking.final_selection_l2_flush_mode() == "replay"
    # the timers take the regime as an argument; the search regime is the default
    assert benchmarking._resolve_l2_flush_mode(None) == "off"
    assert benchmarking._resolve_l2_flush_mode("call") == "call"
    assert benchmarking._resolve_l2_flush_mode("kernel") == "kernel"
    with pytest.raises(ValueError, match="l2_flush_mode"):
        benchmarking._resolve_l2_flush_mode("flush")


@pytest.mark.cpu_only
def test_backend_default_replaces_only_the_default_search_regime(
    monkeypatch,
) -> None:
    """A backend's final-selection regime (CuTe: ``kernel``) replaces the
    default per-call flush only; an explicit search or final regime wins."""
    monkeypatch.delenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_L2_FLUSH", raising=False)
    assert benchmarking.final_selection_l2_flush_mode("kernel") == "kernel"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "1")
    assert benchmarking.final_selection_l2_flush_mode("kernel") == "call"
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "0")
    assert benchmarking.final_selection_l2_flush_mode("kernel") == "replay"
    monkeypatch.delenv("HELION_AUTOTUNE_L2_FLUSH")
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", "call")
    assert benchmarking.final_selection_l2_flush_mode("kernel") == "call"
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", "kernel")
    assert benchmarking.final_selection_l2_flush_mode() == "kernel"
    # a search variable naming the kernel timer reaches the final selection
    monkeypatch.delenv("HELION_AUTOTUNE_FINAL_L2_FLUSH")
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "kernel")
    assert benchmarking.final_selection_l2_flush_mode() == "kernel"
    # a per-call flushed regime takes no second (cold) column; a warm one's
    # cold column takes the backend's timer
    monkeypatch.delenv("HELION_AUTOTUNE_TARGET_REGIME", raising=False)
    assert benchmarking.final_selection_second_regime("kernel") is None
    assert benchmarking.final_selection_second_regime("replay") == "call"
    assert benchmarking.final_selection_second_regime("replay", "kernel") == "kernel"


_TRACE = (
    skipIfRocm("the CUDA activity trace is NVIDIA-only"),
    pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA"),
    pytest.mark.skipif(
        ProfilerActivity.CUDA not in supported_activities(),
        reason="requires CUDA activity tracing",
    ),
)


def _trace_test(fn: Callable[..., None]) -> Callable[..., None]:
    for mark in _TRACE:
        fn = mark(fn)
    return fn


@pytest.fixture
def traced_windows(monkeypatch) -> list[int]:
    """The call counts of the trace windows split during a test: the kernel
    timer's tests assert that the trace path ran, not the CUDA-event
    fallback."""
    windows: list[int] = []
    split = benchmarking._kernel_trace_samples

    def recorded(
        events: list[benchmarking._TraceEvent], count: int, per_flush: int
    ) -> list[tuple[int, float]] | None:
        windows.append(count)
        return split(events, count, per_flush)

    monkeypatch.setattr(benchmarking, "_kernel_trace_samples", recorded)
    return windows


@_trace_test
def test_kernel_timer_reads_each_calls_trace_durations(traced_windows) -> None:
    """``kernel`` samples are per-call spans of the call's kernels (graph
    replayed, as the final selection times them); a call that launches the
    flush's own memset still splits at the markers."""
    from triton import runtime

    a = torch.randn(512, 512, device=DEVICE)
    out = torch.empty_like(a)
    # int32, like Triton's flush buffer: the call's memset is the flush's own
    scratch = torch.empty(1 << 20, dtype=torch.int32, device=DEVICE)

    def call() -> None:
        scratch.zero_()
        torch.mm(a, a, out=out)

    samples = benchmarking.do_bench(
        call,
        return_mode="all",
        fixed_repetitions=7,
        default_cudagraph=True,
        l2_flush_mode="kernel",
    )
    assert isinstance(samples, list)
    assert len(samples) == 7
    assert traced_windows == [7]

    def durations_ms(run: Callable[[], object]) -> list[float]:
        def synchronized() -> None:
            run()
            torch.cuda.synchronize()

        return [
            (event.end - event.start) / 1e6
            for event in benchmarking._cuda_kernel_events(synchronized)
        ]

    cache = runtime.driver.active.get_empty_cache_for_benchmark()  # type: ignore[attr-defined]
    flush_ms = min(
        durations_ms(lambda: runtime.driver.active.clear_cache(cache))  # type: ignore[attr-defined]
    )
    mm_ms = min(durations_ms(lambda: torch.mm(a, a, out=out)))
    # each sample holds the call's memset and mm, never the flush
    assert all(mm_ms <= sample < flush_ms for sample in samples)


@_trace_test
def test_kernel_timer_sizes_windows_with_the_flush(traced_windows) -> None:
    """The flush runs before every warmup and timed call, so it sizes the
    windows with the call (a short call sized alone runs thousands of 256 MB
    flushes)."""
    a = torch.randn(64, 64, device=DEVICE)
    calls: list[None] = []

    def short_call() -> None:
        calls.append(None)
        torch.mm(a, a)

    benchmarking.do_bench(short_call, warmup=20, rep=20, l2_flush_mode="kernel")
    # a flush takes tens of microseconds: 40 ms hold about a thousand (sized
    # by the call alone, over ten thousand)
    assert len(calls) < 3000
    assert traced_windows


@_trace_test
def test_kernel_timer_probes_a_long_call_once(traced_windows) -> None:
    """A call longer than both windows is one traced call, estimate and
    sample at once (``do_bench``'s long-kernel probe)."""
    calls: list[None] = []

    def long_call() -> None:
        calls.append(None)
        torch.cuda._sleep(50_000_000)

    reading = benchmarking.do_bench(
        long_call, warmup=1, rep=1, probe_long_kernel=True, l2_flush_mode="kernel"
    )
    # the setup call, then the traced probe
    assert len(calls) == 2
    assert traced_windows == [1]
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    long_call()
    end.record()
    torch.cuda.synchronize()
    assert reading == pytest.approx(start.elapsed_time(end), rel=0.2)


@_trace_test
def test_kernel_timer_resets_gradients_before_every_call(traced_windows) -> None:
    x = torch.ones(4, device=DEVICE, requires_grad=True)
    seen: list[bool] = []

    def backward_call() -> None:
        seen.append(x.grad is None)
        x.sum().backward()

    benchmarking.do_bench(
        backward_call,
        grad_to_none=[x],  # pyrefly: ignore[bad-argument-type]
        fixed_repetitions=3,
        l2_flush_mode="kernel",
    )
    assert seen[-3:] == [True, True, True]
    assert traced_windows == [3]


@_trace_test
def test_kernel_timer_resizes_a_window_whose_estimate_ran_long(
    traced_windows,
) -> None:
    """The windows are sized by an estimate; when the calls then run far
    shorter (here: the estimate's calls sleep), the first session settles
    too few calls past the traced warmup, and a second one sized by the
    traced call period times them."""
    a = torch.randn(64, 64, device=DEVICE)
    calls = [0]

    def call() -> None:
        calls[0] += 1
        # the setup call and the estimate
        if calls[0] <= 1 + benchmarking._KERNEL_TIMER_ESTIMATE_CALLS:
            torch.cuda._sleep(5_000_000)
        torch.mm(a, a)

    reading = benchmarking.do_bench(call, warmup=20, rep=20, l2_flush_mode="kernel")
    assert reading < 1.0  # milliseconds: no sleeping call is timed
    # a second, longer session sized by the traced call period
    assert len(traced_windows) == 2
    assert traced_windows[1] > traced_windows[0]


@_trace_test
def test_kernel_timer_builds_its_marker_outside_the_trace(
    monkeypatch, traced_windows
) -> None:
    """The marker of a device the process has not timed on yet is built
    before the session: its buffer would otherwise put a kernel of its own
    between the first flush and its marker."""
    a = torch.randn(64, 64, device=DEVICE)
    monkeypatch.setattr(benchmarking, "_trace_markers", {})
    reading = benchmarking.do_bench(
        lambda: torch.mm(a, a), fixed_repetitions=3, l2_flush_mode="kernel"
    )
    assert reading < math.inf
    assert traced_windows == [3]


@_trace_test
def test_kernel_timer_yields_to_a_profiler_of_another_thread(monkeypatch) -> None:
    """A profiling session of any thread (the process-wide flag) keeps the
    process that times the windows off the trace (a second session would
    cancel the first one's); the parent's check, which starts no session,
    does not ask (a parent's profiler does not hold the worker's trace)."""
    monkeypatch.setattr(benchmarking.autograd_profiler, "_is_profiler_enabled", True)
    assert benchmarking.kernel_timer_unready_reason() is not None
    assert benchmarking.kernel_timer_available()


@_trace_test
def test_worker_probe_learns_the_trace() -> None:
    """The final selection asks the benchmark worker whether it traces
    (``KernelTimerProbeJob``) before it labels a column ``kernel``."""
    assert KernelTimerProbeJob(torch.cuda.current_device())() is None
    assert benchmarking._flush_trace


@_trace_test
@pytest.mark.slow
def test_worker_probe_runs_in_the_benchmark_worker() -> None:
    """The probe's profiling session runs in the killable worker, not here."""
    worker = BenchmarkWorker(device=None, log_warning=logging.getLogger().warning)
    try:
        assert worker.run(KernelTimerProbeJob(0), timeout=120.0) is None
    finally:
        worker.shutdown()


def _no_trace(run: Callable[[], None]) -> list[object]:
    """A trace that saw nothing: ``run`` runs, its activities are lost."""
    run()
    return []


@_trace_test
def test_kernel_timer_falls_back_to_events(monkeypatch, caplog) -> None:
    """A running profiler (a second session would cancel its trace) and an
    empty flush trace (another tool holds the activity API; remembered and
    said once for the process) time per-call flushed with CUDA events; a
    trace that does not split into the flushed calls reads infinite."""
    a = torch.randn(64, 64, device=DEVICE)

    def call() -> None:
        torch.mm(a, a)

    with profile(activities=[ProfilerActivity.CUDA]) as outer:
        reading = benchmarking.do_bench(
            call, fixed_repetitions=3, l2_flush_mode="kernel"
        )
        torch.cuda.synchronize()
    assert isinstance(reading, float)
    assert reading > 0.0
    assert any(event.device_type.name == "CUDA" for event in outer.events())

    monkeypatch.setattr(benchmarking, "_flush_trace", 1)
    monkeypatch.setattr(benchmarking, "_cuda_kernel_events", _no_trace)
    with caplog.at_level(logging.WARNING, logger=benchmarking.__name__):
        reading = benchmarking.do_bench(
            call, fixed_repetitions=3, l2_flush_mode="kernel"
        )
    # another scale than its rivals' would mislead: the window cannot win
    assert reading == math.inf
    assert "could not split" in caplog.text

    caplog.clear()
    monkeypatch.setattr(benchmarking, "_flush_trace", None)
    with caplog.at_level(logging.WARNING, logger=benchmarking.__name__):
        for _ in range(2):
            reading = benchmarking.do_bench(
                call, fixed_repetitions=3, l2_flush_mode="kernel"
            )
    assert isinstance(reading, float)
    assert reading < math.inf
    assert benchmarking._flush_trace == 0
    assert caplog.text.count("timing per-call flushed with CUDA events") == 1


@pytest.mark.cpu_only
def test_worker_probe_is_asked_once_and_explains_a_refusal() -> None:
    """``worker_kernel_timer_available`` runs the probe in the worker once,
    logs its reason (a raising probe as a warning), turns any failed probe
    into False and re-raises an unkillable worker."""

    def provider(run: Mock) -> LocalBenchmarkProvider:
        instance = object.__new__(LocalBenchmarkProvider)
        instance._worker_kernel_timer = None
        instance.args = ()
        instance.settings = SimpleNamespace(autotune_benchmark_timeout=60)
        instance.log = Mock()
        instance.steady_window_benchmark_available = Mock(return_value=True)  # type: ignore[method-assign]
        instance._get_benchmark_worker = Mock(return_value=Mock(run=run))  # type: ignore[method-assign]
        return instance

    ready = provider(Mock(return_value=None))
    assert ready.worker_kernel_timer_available()
    assert ready.worker_kernel_timer_available()
    ready._get_benchmark_worker().run.assert_called_once()  # type: ignore[attr-defined]
    held = provider(Mock(return_value="the CUDA activity trace is held"))
    assert not held.worker_kernel_timer_available()
    assert "is held" in held.log.debug.call_args.args[0]  # type: ignore[attr-defined]
    assert not provider(
        Mock(side_effect=BenchmarkTimeout("t"))
    ).worker_kernel_timer_available()
    # a torch nightly renaming a private profiler API the trace path reads:
    # a raising probe is said loudly, unlike a reason it returns
    failing = provider(Mock(side_effect=AttributeError("renamed")))
    assert not failing.worker_kernel_timer_available()
    assert "probe failed" in failing.log.warning.call_args.args[0]  # type: ignore[attr-defined]
    with pytest.raises(BenchmarkWorkerUnkillable):
        provider(
            Mock(side_effect=BenchmarkWorkerUnkillable("w"))
        ).worker_kernel_timer_available()


@_trace_test
def test_worker_job_times_a_kernel_window(monkeypatch) -> None:
    """A worker job asked for a ``kernel`` window times the candidate from the
    trace, under its arguments' device."""
    a = torch.randn(256, 256, device=DEVICE)
    monkeypatch.setattr(
        benchmark_job,
        "_load_compiled_fn_for_worker",
        lambda spec: lambda x: torch.mm(x, x),
    )
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
    monkeypatch.setattr(benchmark_job, "load_trusted_kernel_args", lambda path: (a,))
    job = benchmark_job.BenchmarkJob(
        fn_spec=None,  # pyrefly: ignore[bad-argument-type]
        args_path="",
        l2_flush_mode="kernel",
        fixed_repetitions=3,
    )
    reading = job()
    assert 0.0 < reading < math.inf


@pytest.mark.cpu_only
def test_worker_job_refuses_an_untraceable_kernel_window(monkeypatch) -> None:
    """A worker job asked for a ``kernel`` window it cannot trace fails
    rather than time with CUDA events under the ``kernel`` label."""
    monkeypatch.setattr(
        benchmark_job, "_load_compiled_fn_for_worker", lambda spec: lambda: None
    )
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
    monkeypatch.setattr(benchmark_job, "load_trusted_kernel_args", lambda path: ())
    monkeypatch.setattr(benchmark_job, "steady_graph_dispatch", lambda device: None)
    monkeypatch.setattr(benchmark_job, "synchronize_device", lambda: None)
    monkeypatch.setattr(
        benchmark_job, "kernel_timer_unready_reason", lambda: "the trace is held"
    )
    job = benchmark_job.BenchmarkJob(
        fn_spec=None,  # pyrefly: ignore[bad-argument-type]
        args_path="",
        l2_flush_mode="kernel",
    )
    with pytest.raises(RuntimeError, match="the trace is held"):
        job()


@pytest.mark.cpu_only
def test_kernel_trace_samples_split_calls_at_the_markers() -> None:
    """The marker after each flush delimits the calls: a call's own fill may
    share the flush's name and size, calls may launch different kernels, and
    each sample is the call's first start and its span (first start to last
    end), exact on epoch-nanosecond timestamps. A trace whose markers do not
    match the calls is refused."""

    marker = benchmarking._TRACE_MARKER

    def event(name: str, start_us: int, end_us: int) -> benchmarking._TraceEvent:
        epoch = 1_794_110_432_668_685_631  # Kineto's clock, in ns
        return benchmarking._TraceEvent(
            name, epoch + 1000 * start_us, epoch + 1000 * end_us + 7
        )

    trace = [
        event("memset", 0, 45),
        event(marker, 46, 47),
        event("memset", 48, 93),  # the call's own 256 MB fill
        event("mm", 94, 105),
        event("memset", 106, 151),
        event(marker, 152, 153),
        event("mm", 154, 160),
        event("mm", 158, 170),  # a dependent launch overlapping its primary
    ]
    samples = benchmarking._kernel_trace_samples(trace, 2, 1)
    assert samples is not None
    assert [ms for _start, ms in samples] == [0.057007, 0.016007]
    assert samples[1][0] - samples[0][0] == 106_000
    assert benchmarking._kernel_trace_samples(trace, 3, 1) is None
    assert benchmarking._kernel_trace_samples(trace[1:], 2, 1) is None
    assert benchmarking._kernel_trace_samples(trace[:6], 2, 1) is None


@skipIfRocm("CUDA graph benchmarking is NVIDIA-only")
@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA")
def test_timer_regime_argument_overrides_the_environment(monkeypatch) -> None:
    """``l2_flush_mode`` names a timer's regime regardless of
    HELION_AUTOTUNE_L2_FLUSH: the final selection times its finalists in the
    replay regime while the search runs resident."""
    from triton import runtime

    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "0")
    monkeypatch.setattr(
        benchmarking, "_single_replay_estimate_us", lambda *args, **kwargs: 4.0
    )
    make_batched = benchmarking._make_batched_cudagraph_replay
    seen: list[tuple[bool, int, Callable[[], None]]] = []

    def recording_batched(
        fn: Callable[[], object] | None,
        clear_cache: Callable[[], None],
        calls: int,
        keep_outputs: bool = False,
    ) -> Callable[[], None]:
        seen.append((fn is None, calls, clear_cache))
        return make_batched(fn, clear_cache, calls, keep_outputs=keep_outputs)

    monkeypatch.setattr(
        benchmarking, "_make_batched_cudagraph_replay", recording_batched
    )
    flushes = {"count": 0}
    real_clear = runtime.driver.active.clear_cache

    def counting_clear(cache):
        flushes["count"] += 1
        return real_clear(cache)

    monkeypatch.setattr(runtime.driver.active, "clear_cache", counting_clear)
    small = torch.randn(1 << 14, device=DEVICE)
    out = torch.empty_like(small)

    def fn() -> None:
        torch.add(small, 1.0, out=out)

    # resident search regime: no flush anywhere
    benchmarking.do_bench(fn, fixed_repetitions=10, return_mode="median")
    assert flushes["count"] == 0
    # the final selection's replay regime: flush once before every replay
    median = benchmarking.do_bench(
        fn, fixed_repetitions=10, return_mode="median", l2_flush_mode="replay"
    )
    assert isinstance(median, float) and 0 < median < 0.05
    assert seen and all(clear is benchmarking._no_cache_clear for _, _, clear in seen)
    assert flushes["count"] >= 10
    # per-call regime on request: a flush-only graph is captured and subtracted
    seen.clear()
    timings = benchmarking.interleaved_bench([fn], repeat=4, l2_flush_mode="call")
    assert len(timings) == 1 and timings[0] > 0
    assert any(flush_only for flush_only, _, _ in seen)
    assert all(clear is not benchmarking._no_cache_clear for _, _, clear in seen)


@skipIfRocm("CUDA graph benchmarking is NVIDIA-only")
@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA")
def test_replay_regime_flushes_once_per_back_to_back_batch(monkeypatch) -> None:
    """HELION_AUTOTUNE_L2_FLUSH=replay: the batch holds ``calls`` back-to-back
    calls with no flush inside it, no flush-only graph is subtracted, and L2
    is flushed once before every timed replay — the sweep harness's regime."""
    from triton import runtime

    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "replay")
    monkeypatch.setattr(
        benchmarking, "_single_replay_estimate_us", lambda *args, **kwargs: 4.0
    )
    make_batched = benchmarking._make_batched_cudagraph_replay
    seen: list[tuple[bool, int, Callable[[], None]]] = []

    def recording_batched(
        fn: Callable[[], object] | None,
        clear_cache: Callable[[], None],
        calls: int,
        keep_outputs: bool = False,
    ) -> Callable[[], None]:
        seen.append((fn is None, calls, clear_cache))
        return make_batched(fn, clear_cache, calls, keep_outputs=keep_outputs)

    monkeypatch.setattr(
        benchmarking, "_make_batched_cudagraph_replay", recording_batched
    )
    flushes = {"count": 0}
    real_clear = runtime.driver.active.clear_cache

    def counting_clear(cache):
        flushes["count"] += 1
        return real_clear(cache)

    monkeypatch.setattr(runtime.driver.active, "clear_cache", counting_clear)
    small = torch.randn(1 << 14, device=DEVICE)
    out = torch.empty_like(small)

    def fn() -> None:
        torch.add(small, 1.0, out=out)

    median = benchmarking.do_bench(fn, fixed_repetitions=10, return_mode="median")
    assert seen and all(clear is benchmarking._no_cache_clear for _, _, clear in seen)
    # the 16-call probe batch, then the batch grown to the 256-call cap
    assert {calls for _, calls, _ in seen} == {16, 256}
    assert isinstance(median, float) and 0 < median < 0.05
    # Every timed replay (10) was preceded by a flush, plus the estimates.
    assert flushes["count"] >= 10


@pytest.mark.cpu_only
def test_interleaved_bench_of_nothing_is_empty() -> None:
    # An empty benchmark needs neither a GPU driver nor a flush buffer.
    assert benchmarking.interleaved_bench([], repeat=3) == []


def test_positive_median_ignores_poisoned_samples() -> None:
    assert benchmarking._positive_median([2.0, 3.0, 4.0]) == 3.0
    assert benchmarking._positive_median([-5.0, -4.0, 1.0]) == 1.0
    assert benchmarking._positive_median([-1.0, -2.0]) == math.inf


def test_positive_mean_ignores_poisoned_samples() -> None:
    assert benchmarking._positive_mean([2.0, 3.0, 4.0]) == 3.0
    assert benchmarking._positive_mean([-5.0, 1.0, 3.0]) == 2.0
    assert benchmarking._positive_mean([-1.0, 0.0]) == math.inf


def test_a_single_rank_agrees_with_itself() -> None:
    assert all_ranks_agree(True) is True
    assert all_ranks_agree(False) is False


@skipIfRocm("CUDA graph benchmarking is NVIDIA-only")
@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA")
def test_batched_timer_agrees_its_batch_across_ranks(monkeypatch) -> None:
    """The batch size and the capture outcome pass through the distributed
    helpers with the caller's process group, so every rank launches the same
    number of kernels; a capture that failed on any rank drops the batch."""
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    x = torch.randn(1 << 14, device=DEVICE)
    out = torch.empty_like(x)

    def fn() -> None:
        torch.add(x, 1.0, out=out)

    synced: list[tuple[object, str | None]] = []
    agreed: list[tuple[bool, str | None]] = []

    def fake_sync(obj: object, process_group_name: str | None = None) -> object:
        synced.append((obj, process_group_name))
        return obj

    def fake_agree(flag: bool, process_group_name: str | None = None) -> bool:
        agreed.append((flag, process_group_name))
        return False  # another rank failed its capture

    monkeypatch.setattr(benchmarking, "sync_object", fake_sync)
    monkeypatch.setattr(benchmarking, "all_ranks_agree", fake_agree)

    timings = benchmarking.interleaved_bench([fn], repeat=4, process_group_name="pg")
    assert len(timings) == 1 and timings[0] > 0
    # the sizing steps are agreed through the group before anything runs on
    # them; the last agreed size is the batch the capture was attempted at
    assert all(group == "pg" for _obj, group in synced)
    calls, group = synced[-1]
    assert isinstance(calls, int) and calls > 1 and group == "pg"
    # every local capture succeeded; the other rank's failure halves the
    # batch until the single replays remain, one agreement per attempt
    assert agreed and all(entry == (True, "pg") for entry in agreed)
    assert len(agreed) == max(1, calls.bit_length() - 1)
    agreed.clear()

    timing = benchmarking.do_bench(fn, return_mode="median", process_group_name="pg")
    assert timing > 0
    calls, group = synced[-1]
    assert isinstance(calls, int) and calls > 1 and group == "pg"
    assert agreed and all(entry == (True, "pg") for entry in agreed)


@skipIfRocm("CUDA graph benchmarking is NVIDIA-only")
@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires CUDA")
def test_l2_resident_regime_batches_without_a_flush_graph(monkeypatch) -> None:
    """HELION_AUTOTUNE_L2_FLUSH=0 keeps the operands L2-resident: no cache
    clear between calls, the short-kernel batch is ``calls`` back-to-back
    replays with no flush-only graph captured or subtracted, and the per-call time is
    the batch time over ``calls``."""
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH", "0")
    monkeypatch.setattr(
        benchmarking, "_single_replay_estimate_us", lambda *args, **kwargs: 4.0
    )
    make_batched = benchmarking._make_batched_cudagraph_replay
    seen: list[tuple[bool, int, Callable[[], None]]] = []

    def recording_batched(
        fn: Callable[[], object] | None,
        clear_cache: Callable[[], None],
        calls: int,
        keep_outputs: bool = False,
    ) -> Callable[[], None]:
        seen.append((fn is None, calls, clear_cache))
        return make_batched(fn, clear_cache, calls, keep_outputs=keep_outputs)

    monkeypatch.setattr(
        benchmarking, "_make_batched_cudagraph_replay", recording_batched
    )
    small = torch.randn(1 << 14, device=DEVICE)
    out = torch.empty_like(small)

    def fn() -> None:
        torch.add(small, 1.0, out=out)

    times = benchmarking.interleaved_bench([fn], repeat=20)
    assert [(flush_only, calls) for flush_only, calls, _ in seen] == [(False, 16)]
    assert all(clear is benchmarking._no_cache_clear for _, _, clear in seen)
    assert len(times) == 1 and 0 < times[0] < 0.05  # per call, not per batch

    seen.clear()
    median = benchmarking.do_bench(fn, fixed_repetitions=10, return_mode="median")
    assert [(flush_only, calls) for flush_only, calls, _ in seen] == [(False, 16)]
    assert isinstance(median, float)
    assert 0 < median < 0.05


@pytest.mark.skipif(DEVICE.type != "cuda", reason="requires a GPU")
@pytest.mark.parametrize("hip", [True, False])
def test_do_bench_drains_the_launch_queue_on_hip(monkeypatch, hip: bool) -> None:
    """HIP segfaults once tens of thousands of launches are queued, so on ROCm
    do_bench synchronizes every ``_HIP_MAX_QUEUED_CALLS`` calls of its warm-up
    and timed loops (each loop counts its own, so the end of the warm-up and the
    start of the timing queue under twice that); elsewhere those loops never
    wait on the device."""
    monkeypatch.delenv("HELION_BENCHMARK_CUDAGRAPH", raising=False)
    monkeypatch.setattr(torch.version, "hip", "7.2" if hip else None)
    monkeypatch.setattr(benchmarking, "_HIP_MAX_QUEUED_CALLS", 10)
    trace: list[str] = []
    synchronize = torch.cuda.synchronize

    def traced_synchronize(*args: object, **kwargs: object) -> None:
        trace.append("sync")
        synchronize(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", traced_synchronize)
    x = torch.zeros(1024, device=DEVICE)

    def fn() -> None:
        trace.append("call")
        x.add_(1)

    benchmarking.do_bench(fn, warmup=20, rep=20, l2_flush_mode="off")
    longest = max(
        len(run)
        for run in "".join("c" if event == "call" else " " for event in trace).split()
    )
    assert trace.count("call") > 100
    if hip:
        assert longest < 2 * 10
    else:
        assert longest > 2 * 10
