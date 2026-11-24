"""``autotune_l2_flush_sweep``: the autotuner's timers follow every L2 flush
with the sweep of lines pinned with evict-last hints
(``benchmarking.l2_flush_sweep``)."""

from __future__ import annotations

import contextlib
import functools
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from helion._testing import DEVICE
from helion._testing.benchmark_timers import LoggedPrelude
from helion._testing.benchmark_timers import stand_in_event_timer
from helion._testing.benchmark_timers import stand_in_trace
from helion._testing.benchmark_timers import trace_test
from helion.autotuner import benchmark_job
from helion.autotuner import benchmark_provider
from helion.autotuner import benchmarking
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.base_cache import AutotuneCacheBase
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.benchmark_job import BenchmarkJob
from helion.autotuner.benchmark_job import KernelTimerProbeJob
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmarking import l2_flush_sweep
from helion.autotuner.benchmarking import l2_flush_sweep_enabled
from helion.autotuner.local_cache import _hash_search_policy
from helion.autotuner.local_cache import _search_policy_hash
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


@pytest.mark.cpu_only
def test_the_setting_is_off_by_default_and_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HELION_AUTOTUNE_L2_FLUSH_SWEEP", raising=False)
    assert Settings().autotune_l2_flush_sweep is False
    assert Settings(autotune_l2_flush_sweep=True).autotune_l2_flush_sweep is True
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH_SWEEP", "1")
    assert Settings().autotune_l2_flush_sweep is True
    assert Settings(autotune_l2_flush_sweep=False).autotune_l2_flush_sweep is False
    monkeypatch.setenv("HELION_AUTOTUNE_L2_FLUSH_SWEEP", "0")
    assert Settings().autotune_l2_flush_sweep is False


@pytest.mark.cpu_only
def test_the_block_nests_and_unwinds() -> None:
    assert not l2_flush_sweep_enabled()
    with l2_flush_sweep(True):
        assert l2_flush_sweep_enabled()
        with l2_flush_sweep(False):
            assert not l2_flush_sweep_enabled()
        assert l2_flush_sweep_enabled()
    assert not l2_flush_sweep_enabled()
    with pytest.raises(RuntimeError, match="boom"), l2_flush_sweep(True):
        raise RuntimeError("boom")
    assert not l2_flush_sweep_enabled()


@pytest.fixture
def event_timer(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The timers on a stand-in Triton driver: the log receives every flush,
    sweep (``_release_l2_pins``), event record and synchronization."""
    return stand_in_event_timer(monkeypatch)


@pytest.mark.cpu_only
def test_do_bench_sweeps_after_every_flush_inside_the_block(
    event_timer: list[str],
) -> None:
    def bench(sweep: bool, **kwargs: object) -> list[str]:
        event_timer.clear()
        with l2_flush_sweep(sweep):
            assert benchmarking.do_bench(
                lambda: event_timer.append("call"),
                fixed_repetitions=2,
                pre_warmed=True,
                return_mode="median",
                **kwargs,
            ) == pytest.approx(2.0)
        return list(event_timer)

    assert bench(False) == ["sync", *["flush", "event", "call", "event"] * 2, "sync"]
    # One sweep runs as the flush is built (compiled before any capture or
    # trace), then every flush is followed by one, outside the events.
    assert bench(True) == [
        "sync",
        "sweep",
        *["flush", "sweep", "event", "call", "event"] * 2,
        "sync",
    ]
    # Behind a prelude: flush, sweep, then the work, so the work's lines
    # stay for the call. (The window's own release precedes the calls.)
    assert bench(True, prelude=LoggedPrelude(event_timer)) == [
        "sync",
        "sweep",
        "sweep",
        *["flush", "sweep", "prelude", "event", "call", "event"] * 2,
        "sync",
    ]
    # A regime that never flushes does not sweep either.
    assert bench(True, l2_flush_mode="off") == [
        "sync",
        *["event", "call", "event"] * 2,
        "sync",
    ]


@pytest.mark.cpu_only
def test_interleaved_bench_sweeps_after_every_flush_inside_the_block(
    event_timer: list[str],
) -> None:
    def bench(sweep: bool) -> list[str]:
        event_timer.clear()
        with l2_flush_sweep(sweep):
            assert (
                benchmarking.interleaved_bench(
                    [lambda: event_timer.append("a"), lambda: event_timer.append("b")],
                    repeat=2,
                )
                == [pytest.approx(2.0)] * 2
            )
        return list(event_timer)

    def call(name: str, clear: list[str]) -> list[str]:
        return [*clear, "event", name, "event"]

    flush = ["flush"]
    assert bench(False) == [
        "a",
        "b",
        *flush,
        "sync",
        *(call("a", flush) + call("b", flush)) * 2,
        "sync",
    ]
    swept = ["flush", "sweep"]
    assert bench(True) == [
        "a",
        "b",
        "sweep",
        *swept,
        "sync",
        *(call("a", swept) + call("b", swept)) * 2,
        "sync",
    ]


@pytest.mark.cpu_only
def test_no_batch_subtracts_a_graph_that_holds_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short graph replays of the per-call flushed regime are batched as
    [flush, call] x N minus [flush] x N; with the sweep in both that reads
    hinted calls low, so under the sweep they stay single flushed replays,
    the flush and the sweep outside the events."""
    batched: list[object] = []

    def capture(fns: object, clear_cache: object, calls: int, **kwargs: object):
        batched.append(clear_cache)
        return [lambda: None for _ in fns], lambda: None  # pyrefly: ignore

    # every single replay reads 4 us: a batch of ceil(64 / 4) = 16 calls
    monkeypatch.setattr(
        benchmarking, "_single_replay_estimate_us", lambda *args, **kwargs: 4.0
    )
    monkeypatch.setattr(benchmarking, "_batched_flushed_replays", capture)
    replay = object()

    def flush() -> None:
        pass

    def build(mode: str) -> tuple[int, list[object], object]:
        calls, replays, flush_replay = benchmarking._build_batched_replays(
            [lambda: None],
            [replay],  # pyrefly: ignore[bad-argument-type]
            flush,
            None,
            mode=mode,
            process_group_name=None,
        )
        return calls, replays, flush_replay

    calls, _replays, flush_replay = build("call")
    assert calls == 16 and flush_replay is not None
    assert batched == [flush]
    batched.clear()
    with l2_flush_sweep(True):
        assert build("call") == (1, [replay], None)
        assert batched == []
        # ``replay`` flushes once before each replay, outside the events:
        # nothing is subtracted, so its batches stay
        assert build("replay")[0] > 1
        assert batched


@pytest.fixture
def fake_trace(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """``_do_bench_kernel_durations`` on a stand-in trace (``stand_in_trace``):
    ``window`` times a candidate of 7 us behind a flush, the sweep where
    asked (600 us), and a prelude's work; ``launch`` makes a kernel."""
    trace = stand_in_trace(monkeypatch)
    launch, device, flush = trace.launch, trace.device, trace.flush
    monkeypatch.setattr(
        benchmarking, "_release_l2_pins", lambda buffer: launch("sweep", 600_000)()
    )

    def window(*, swept: bool, prelude: object = None) -> object:
        clear_cache = flush
        if swept:
            clear_cache = functools.partial(
                benchmarking._flush_then_sweep,
                flush,
                "buffer",  # pyrefly: ignore[bad-argument-type]
            )
        if prelude is not None:
            clear_cache = functools.partial(
                benchmarking._flush_then_prelude,
                clear_cache,
                prelude,  # pyrefly: ignore[bad-argument-type]
            )
        return benchmarking._do_bench_kernel_durations(
            launch("candidate", 7_000),
            clear_cache,
            device,
            warmup=0,
            rep=1,
            quantiles=None,
            return_mode="median",
            fixed_repetitions=3,
            prelude=prelude,  # pyrefly: ignore[bad-argument-type]
        )

    return SimpleNamespace(window=window, launch=launch)


@pytest.mark.cpu_only
def test_the_trace_learns_the_sweep_apart_from_the_flush(
    fake_trace: SimpleNamespace,
) -> None:
    """Inside the block the trace splits each call off [flush, sweep] (two
    events before the marker), a footprint learned apart from the flush's
    one: a process can time windows of both kinds."""
    with l2_flush_sweep(True):
        assert fake_trace.window(swept=True) == pytest.approx(0.007)
    assert benchmarking._flush_sweep_trace == 2
    assert benchmarking._flush_trace is None
    assert fake_trace.window(swept=False) == pytest.approx(0.007)
    assert benchmarking._flush_trace == 1
    assert benchmarking._flush_sweep_trace == 2

    class Work:
        def run(self) -> None:
            fake_trace.launch("producer", 500_000)()

    # Behind a prelude: [flush, sweep, work] before the marker, or [flush,
    # work] without the sweep, each learned once per prelude.
    work = Work()
    with l2_flush_sweep(True):
        assert fake_trace.window(swept=True, prelude=work) == pytest.approx(0.007)
    assert benchmarking._prelude_flush_sweep_traces[work] == 3
    assert work not in benchmarking._prelude_flush_traces
    assert fake_trace.window(swept=False, prelude=work) == pytest.approx(0.007)
    assert benchmarking._prelude_flush_traces[work] == 2


@pytest.mark.cpu_only
def test_the_flush_of_the_wall_clock_and_graph_timers_sweeps_inside_the_block(
    event_timer: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_make_l2_cache_clearer`` is the flush of the wall-clock timers and
    of ``do_bench_cuda_graph``, which captures it before its start event."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    clear = benchmarking._make_l2_cache_clearer()
    clear()
    assert event_timer == ["flush"]
    event_timer.clear()
    with l2_flush_sweep(True):
        clear = benchmarking._make_l2_cache_clearer()
    assert event_timer == ["sweep"]
    clear()
    assert event_timer == ["sweep", "flush", "sweep"]


@pytest.mark.cpu_only
def test_worker_jobs_time_inside_the_block_they_carry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[bool] = []

    class Stop(Exception):
        pass

    def load(fn_spec: object) -> object:
        seen.append(l2_flush_sweep_enabled())
        raise Stop

    monkeypatch.setattr(benchmark_job, "_load_compiled_fn_for_worker", load)
    for sweep in (True, False):
        job = BenchmarkJob(
            fn_spec=None,  # pyrefly: ignore[bad-argument-type]
            args_path="unused",
            l2_flush_sweep=sweep,
        )
        with pytest.raises(Stop):
            job()
    assert seen == [True, False]
    assert BenchmarkJob(fn_spec=None, args_path="unused").l2_flush_sweep is False  # pyrefly: ignore[bad-argument-type]

    def probe() -> str | None:
        seen.append(l2_flush_sweep_enabled())
        return None

    seen.clear()
    monkeypatch.setattr(benchmark_job, "kernel_timer_unready_reason", probe)
    assert KernelTimerProbeJob(None, l2_flush_sweep=True)() is None
    assert KernelTimerProbeJob(None)() is None
    assert seen == [True, False]


@pytest.mark.cpu_only
def test_the_provider_queues_jobs_with_the_block_they_were_made_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[BenchmarkJob] = []
    provider = object.__new__(LocalBenchmarkProvider)
    provider._input_bank_count = 1
    provider._precompile_args_path = "unused"
    provider._subprocess_wrapper_unloadable = False
    provider.mutated_arg_indices = []
    provider.settings = Settings()
    provider._subprocess_benchmark_uses_wall_clock = lambda: False
    provider._probe_long_kernel = lambda: False
    provider._worker_compile_timeout = lambda: 30
    provider._get_benchmark_worker = lambda: SimpleNamespace(
        run=lambda job, **kwargs: captured.append(job) or 4.0
    )
    monkeypatch.setattr(benchmark_provider, "_serialize_compiled_fn", lambda fn: None)
    for sweep in (True, False):
        with l2_flush_sweep(sweep):
            assert (
                provider._run_subprocess_benchmark_job(lambda: None, warmup=1, rep=5)
                == 4.0
            )
    assert [job.l2_flush_sweep for job in captured] == [True, False]


@pytest.mark.cpu_only
def test_a_search_times_inside_the_block_of_its_setting() -> None:
    seen: list[bool] = []

    @contextlib.contextmanager
    def tracing(name: str) -> Iterator[None]:
        yield

    for sweep in (True, False):
        search = object.__new__(BaseSearch)
        search.settings = Settings(autotune_l2_flush_sweep=sweep)
        search.log = SimpleNamespace(  # pyrefly: ignore[bad-assignment]
            autotune_tracing=tracing,
            record_selected_config=lambda config, measured_as: None,
        )
        search._autotune_and_log = (  # pyrefly: ignore[bad-assignment]
            lambda skip_cache: seen.append(l2_flush_sweep_enabled()) or "config"
        )
        assert search.autotune() == "config"
    assert seen == [True, False]
    assert not l2_flush_sweep_enabled()


@pytest.mark.cpu_only
def test_best_of_k_rebench_and_aot_measure_time_inside_the_block() -> None:
    """Paths that time a search's configs outside its run: the best-of-K
    rebench round and AOT measurement follow the setting too."""
    seen: list[bool] = []

    def autotuner(sweep: bool) -> SimpleNamespace:
        return SimpleNamespace(
            settings=Settings(autotune_l2_flush_sweep=sweep),
            _prepare=lambda: None,
            benchmark_provider=SimpleNamespace(
                setup=lambda: None, cleanup=lambda: None
            ),
            benchmark_batch=lambda configs, desc: (
                seen.append(l2_flush_sweep_enabled()) or []
            ),
        )

    for sweep in (True, False):
        cache = SimpleNamespace(_autotuner_factory=functools.partial(autotuner, sweep))
        assert AutotuneCacheBase._run_rebench_round(cache, []) == []  # pyrefly: ignore[bad-argument-type]
        aot = SimpleNamespace(
            autotuner=autotuner(sweep),
            _measure_all_configs=lambda: seen.append(l2_flush_sweep_enabled()) or [],
        )
        assert AOTAutotuneCache.measure_all_configs(aot) == []  # pyrefly: ignore[bad-argument-type]
    assert seen == [True, True, False, False]
    assert not l2_flush_sweep_enabled()


@pytest.mark.cpu_only
def test_searches_with_the_sweep_keep_their_winners_apart() -> None:
    def key(sweep: bool, **kwargs: object) -> str:
        search = SimpleNamespace(
            settings=Settings(autotune_l2_flush_sweep=sweep),
            _benchmark_provider_cls=LocalBenchmarkProvider,
            _benchmark_prelude=None,
            _search_policy_cacheable=True,
        )
        return _search_policy_hash(search, cute_flash_search_enabled=False, **kwargs)

    # every key of a search without the sweep stays what it was
    assert key(False) == _hash_search_policy(b"")
    assert key(True) != key(False)
    assert key(True) == key(True)
    # an explicit AOT artifact keeps its stable identifier
    assert key(True, explicit_aot=True) == key(False, explicit_aot=True)


def _kernel_names(run: Callable[[], object]) -> list[str]:
    def synchronized() -> None:
        run()
        torch.cuda.synchronize()

    return [event.name for event in benchmarking._cuda_kernel_events(synchronized)]


_SWEEP = "helion_l2_sweep_evict_last"


@trace_test
def test_traced_windows_read_each_call_behind_the_flush_and_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmarking, "_flush_trace", None)
    monkeypatch.setattr(benchmarking, "_flush_sweep_trace", None)
    windows: list[list[str]] = []
    split = benchmarking._kernel_trace_samples

    def recorded(
        events: list[benchmarking._TraceEvent],
        count: int,
        per_flush: int,
        work_events: int = 0,
    ) -> list[tuple[int, float]] | None:
        windows.append([event.name for event in events])
        return split(events, count, per_flush, work_events)

    monkeypatch.setattr(benchmarking, "_kernel_trace_samples", recorded)
    x = torch.randn(1 << 20, device=DEVICE)
    assert benchmarking.kernel_timer_unready_reason() is None
    with l2_flush_sweep(True):
        assert benchmarking.kernel_timer_unready_reason() is None
    flush = benchmarking._flush_trace
    assert flush
    assert benchmarking._flush_sweep_trace == flush + 1

    def window() -> float:
        return benchmarking.do_bench(  # pyrefly: ignore[bad-return]
            lambda: torch.add(x, 1.0),
            fixed_repetitions=5,
            return_mode="median",
            l2_flush_mode="call",
            trace_kernel_durations=True,
        )

    def markers(names: list[str]) -> list[int]:
        return [i for i, name in enumerate(names) if name == benchmarking._TRACE_MARKER]

    # The traced window holds [flush, sweep, marker, call] per call: the sweep
    # precedes the marker, so the reading (the kernels after it) leaves it out.
    with l2_flush_sweep(True):
        assert 0.0 < window() < 1.0
    (names,) = windows
    assert len(markers(names)) == 5
    assert all(names[i - 1] == _SWEEP for i in markers(names))
    windows.clear()
    assert 0.0 < window() < 1.0
    (names,) = windows
    assert len(markers(names)) == 5
    assert _SWEEP not in names


@trace_test
def test_graph_timers_sweep_outside_the_timed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(1 << 16, device=DEVICE)
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    batched: list[int] = []
    capture = benchmarking._batched_flushed_replays

    def recorded(fns: object, clear_cache: object, calls: int, **kwargs: object):
        batched.append(calls)
        return capture(fns, clear_cache, calls, **kwargs)  # pyrefly: ignore

    monkeypatch.setattr(benchmarking, "_batched_flushed_replays", recorded)

    def short() -> float:
        return benchmarking.do_bench(  # pyrefly: ignore[bad-return]
            lambda: torch.add(x, 1.0),
            warmup=1,
            rep=5,
            return_mode="median",
            l2_flush_mode="call",
        )

    with l2_flush_sweep(True):
        assert 0.0 < short() < 1.0
    assert batched == []
    assert 0.0 < short() < 1.0
    assert batched
    # do_bench_cuda_graph captures [flush, sweep] before its start event
    with l2_flush_sweep(True):
        names = _kernel_names(
            lambda: benchmarking.do_bench_cuda_graph(
                lambda: torch.add(x, 1.0),
                fixed_repetitions=3,
                return_mode="median",
                l2_flush_mode="replay",
            )
        )
    assert names.count(_SWEEP) >= 3
