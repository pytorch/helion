"""A search's candidates timed behind the kernels that run before them
(``helion.autotuner.benchmark_prelude``): the sequence and its preludes, the
timers' treatment of a prelude, and the plumbing from ``timed_behind`` to
the benchmark worker's job. Without a device: compiled functions, the Triton
driver, the event clock and the trace are stand-ins."""

from __future__ import annotations

import functools
import importlib.util
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion import exc
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing.benchmark_timers import LoggedPrelude
from helion._testing.benchmark_timers import stand_in_event_timer
from helion._testing.benchmark_timers import stand_in_trace
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion.autotuner import benchmark_job
from helion.autotuner import benchmark_provider
from helion.autotuner import benchmarking
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.benchmark_prelude import BenchmarkPrelude
from helion.autotuner.benchmark_prelude import KernelSequence
from helion.autotuner.benchmark_prelude import PreludeFailed
from helion.autotuner.benchmark_prelude import PreludeOutput
from helion.autotuner.benchmark_prelude import PreludeRefused
from helion.autotuner.benchmark_prelude import PreludeSearchFailed
from helion.autotuner.benchmark_prelude import PreludeStep
from helion.autotuner.benchmark_prelude import PreludeUnsupported
from helion.autotuner.benchmark_prelude import active_prelude
from helion.autotuner.benchmark_prelude import load_prelude
from helion.autotuner.benchmark_prelude import timed_behind
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.local_cache import _hash_search_policy
from helion.autotuner.local_cache import _search_policy_hash
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.metrics import AutotuneMetrics
import helion.language as hl
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    import types

pytestmark = pytest.mark.cpu_only

_KERNELS = """
import torch


def scale(x, factor):
    return x * factor


def fill(buffer, value):
    buffer.fill_(value)
    return buffer


def pair(x):
    return x + 1, x + 2


def total(*tensors):
    return sum(t.sum() for t in tensors)
"""


@pytest.fixture
def kernels(tmp_path: Path) -> types.ModuleType:
    """Stand-ins for compiled kernel functions, in a module with a file: the
    benchmark worker loads a function from its module's source."""
    path = tmp_path / "prelude_kernels.py"
    path.write_text(_KERNELS)
    spec = importlib.util.spec_from_file_location("_test_prelude_kernels", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _sequence(kernels: types.ModuleType) -> tuple[KernelSequence, dict[str, object]]:
    """Three calls of a sequence: ``scale`` makes a tensor, ``fill`` writes
    a buffer the caller owns and returns it, ``pair`` makes two more."""
    x = torch.arange(12.0).reshape(3, 4)
    buffer = torch.empty(4)
    sequence = KernelSequence()
    scaled = kernels.scale(x, 2.0)
    sequence.record(kernels.scale, (x, 2.0), scaled)
    filled = kernels.fill(buffer, 3.0)
    sequence.record(kernels.fill, (buffer, 3.0), filled)
    pairs = kernels.pair(scaled[0])
    sequence.record(kernels.pair, (scaled[0],), pairs)
    return sequence, {"x": x, "buffer": buffer, "scaled": scaled, "pairs": pairs}


def test_a_sequence_stands_arguments_for_the_outputs_they_view(
    kernels: types.ModuleType,
) -> None:
    sequence, tensors = _sequence(kernels)
    scaled, buffer, x = tensors["scaled"], tensors["buffer"], tensors["x"]
    assert len(sequence) == 3
    # An argument that views an earlier call's output is that output with
    # the argument's view; a later output of the same call is its own leaf.
    prelude = sequence.prelude(
        (scaled.view(4, 3)[1:], tensors["pairs"][1], buffer, x, 7)
    )
    assert prelude.args[0] == PreludeOutput(0, 0, (3, 3), (3, 1), 3)
    assert prelude.args[1] == PreludeOutput(2, 1, (4,), (1,), 0)
    # The caller's own tensors stay themselves: the buffer a step filled
    # and returned is the caller's buffer, not that step's output.
    assert prelude.args[2] is buffer and prelude.args[3] is x and prelude.args[4] == 7
    # So do they among the steps' arguments, and a step's argument that an
    # earlier step made is a stand-in too.
    assert [step.fn for step in prelude.steps] == [
        kernels.scale,
        kernels.fill,
        kernels.pair,
    ]
    assert prelude.steps[0].args == (x, 2.0)
    assert prelude.steps[1].args[0] is buffer
    assert prelude.steps[2].args == (PreludeOutput(0, 0, (4,), (1,), 0),)
    # An argument that reads an output as another dtype has no stand-in:
    # the kernel about to be called gets no prelude.
    with pytest.raises(PreludeUnsupported, match=r"float32 output .* as torch\.int32"):
        sequence.prelude((scaled.view(torch.int32),))
    # (asking changed nothing: the calls so far can all be replayed)
    assert len(sequence.prelude((x,)).steps) == 3


def test_no_call_gets_a_prelude_after_one_that_cannot_be_replayed(
    kernels: types.ModuleType,
) -> None:
    """A call that read an earlier output as another dtype ran on that
    output; replayed, it would run on the copy it was given, and whatever
    it wrote there the later calls would never see. The sequence makes no
    prelude from then on, also for a call that reads other tensors."""
    x = torch.arange(4.0)
    sequence = KernelSequence()
    made = kernels.scale(x, 2.0)
    sequence.record(kernels.scale, (x, 2.0), made)
    bits = made.view(torch.int32)
    # the call writes the output through its view: zeros, in the sequence
    sequence.record(kernels.fill, (bits, 0), kernels.fill(bits, 0))
    assert torch.equal(made, torch.zeros(4))
    for args in ((made,), (x,), ()):
        with pytest.raises(
            PreludeUnsupported,
            match=r"an earlier kernel call read the torch\.float32 output .* "
            r"as torch\.int32",
        ):
            sequence.prelude(args)
    # (the sequence still takes calls: a harness records every call)
    sequence.record(kernels.total, (made,), kernels.total(made))
    assert len(sequence) == 3


def test_an_output_held_by_a_callback_has_no_stand_in(
    kernels: types.ModuleType,
) -> None:
    """A callback among the arguments is one pytree leaf, and the tensors
    in its closure cells and defaults are not: an earlier output held there
    would be replayed as the copy the call was given."""
    x = torch.arange(4.0)
    sequence = KernelSequence()
    made = kernels.scale(x, 2.0)
    sequence.record(kernels.scale, (x, 2.0), made)

    def epilogue(value: torch.Tensor, bias: torch.Tensor = made) -> torch.Tensor:
        return value + bias

    def reads_the_input(value: torch.Tensor) -> torch.Tensor:
        return value + x

    with pytest.raises(PreludeUnsupported, match="a callback or a record"):
        sequence.prelude((x, epilogue))
    # A callback that holds only the caller's own tensors is itself.
    prelude = sequence.prelude((made, reads_the_input))
    assert prelude.args[1] is reads_the_input
    assert prelude.args[0] == PreludeOutput(0, 0, (4,), (1,), 0)
    assert prelude.args[0].dtype == torch.float32
    # So it is when the same output is also an argument of its own: the
    # argument would become a stand-in for what the earlier call makes anew,
    # and the callback would still hold the copy.
    with pytest.raises(PreludeUnsupported, match="a callback or a record"):
        sequence.prelude((made, epilogue))
    # Recorded, such a call ends the preludes of the calls after it.
    sequence.record(kernels.total, (made, epilogue), None)
    with pytest.raises(PreludeUnsupported, match="an earlier kernel call's arguments"):
        sequence.prelude((x,))


def test_a_prelude_knows_whether_its_timed_call_is_a_searchs() -> None:
    """The worker runs every candidate on the prelude's arguments: a search
    takes a prelude only for the call it specialized its candidates on."""
    x = torch.zeros(3, 4)
    step = PreludeStep(len, ((),))
    made = PreludeOutput(0, 0, (4, 3), (3, 1), 0, torch.float32)
    prelude = BenchmarkPrelude((step,), (made, x, 2, len, [x[0]]))
    theirs = torch.ones(4, 3)
    assert prelude.mismatch((theirs, torch.ones(3, 4), 2, len, [x[1]])) is None
    # (a stand-in made without a dtype stands for any)
    undated = BenchmarkPrelude((step,), (PreludeOutput(0, 0, (4, 3), (3, 1), 0),))
    assert undated.mismatch((theirs.double(),)) is None

    def differs(*args: object) -> str:
        reason = prelude.mismatch(args)
        assert reason is not None
        return reason

    assert "another structure" in differs(theirs, x, 2, len)
    assert "another structure" in differs(theirs, x, 2, len, [x[0], x[1]])
    assert "tensor leaf 0" in differs(theirs.t(), x, 2, len, [x[0]])
    assert "tensor leaf 0" in differs(theirs.double(), x, 2, len, [x[0]])
    assert "tensor leaf 1" in differs(theirs, x[:, :2], 2, len, [x[0]])
    assert "leaf 1 is a tensor" in differs(theirs, 5, 2, len, [x[0]])
    assert "leaf 2 is no tensor" in differs(theirs, x, x, len, [x[0]])
    assert "leaf 2 is 2, the search's 3" in differs(theirs, x, 3, len, [x[0]])
    assert "leaf 3" in differs(theirs, x, 2, "len", [x[0]])


def test_a_prelude_makes_its_calls_in_order_and_hands_on_what_they_made(
    kernels: types.ModuleType,
) -> None:
    sequence, tensors = _sequence(kernels)
    scaled, buffer, x = tensors["scaled"], tensors["buffer"], tensors["x"]
    prelude = sequence.prelude((scaled.view(4, 3)[1:], tensors["pairs"][1], buffer))
    loaded = prelude.loaded()
    assert len(loaded) == 3 and loaded.args is None
    timed = loaded.timed(kernels.total)
    with pytest.raises(AssertionError, match="has not run"):
        timed()
    # The work runs on the caller's tensors as they are now: every run makes
    # the outputs anew from them, and the timed call reads the new ones.
    buffer.zero_()
    x.mul_(10.0)
    loaded.run()
    first = loaded.args
    assert first is not None and first[2] is buffer
    assert torch.equal(buffer, torch.full((4,), 3.0))
    assert torch.equal(first[0], (x * 2.0).view(4, 3)[1:])
    assert torch.equal(first[1], (x * 2.0)[0] + 2)
    assert first[0].untyped_storage().data_ptr() != scaled.untyped_storage().data_ptr()
    assert float(timed()) == float(first[0].sum() + first[1].sum() + 12.0)
    loaded.run()
    assert loaded.args is not first
    assert loaded.args[0].untyped_storage().data_ptr() != (
        first[0].untyped_storage().data_ptr()
    )


def test_a_saved_prelude_runs_alike_where_it_is_loaded(
    kernels: types.ModuleType, tmp_path: Path
) -> None:
    """The benchmark worker loads the steps' functions from their modules'
    source and all arguments from one tree: tensors that share storage
    across the steps and the timed call still share it."""
    sequence, tensors = _sequence(kernels)
    prelude = sequence.prelude((tensors["scaled"][1:], tensors["buffer"], tensors["x"]))
    path = str(tmp_path / "prelude.pt")
    prelude.save(path)
    loaded = load_prelude(path)
    assert load_prelude(path) is loaded
    assert len(loaded) == 3
    # (a stand-in crosses as what it was, the dtype it notes included)
    made, crossed = prelude.args[0], loaded._args[0]
    assert crossed is not made and crossed == made and hash(crossed) == hash(made)
    assert repr(crossed) == repr(made) and crossed.dtype == torch.float32
    loaded.run()
    args = loaded.args
    assert args is not None
    assert torch.equal(args[0], (tensors["x"] * 2.0)[1:])
    steps = loaded._steps
    # (the functions are the worker's copies, not this process's)
    assert steps[0][0] is not kernels.scale and steps[0][0].__name__ == "scale"

    def storage(tensor: torch.Tensor) -> int:
        return tensor.untyped_storage().data_ptr()

    assert storage(args[1]) == storage(steps[1][1][0])
    assert storage(args[2]) == storage(steps[0][1][0])
    assert storage(args[2]) != storage(tensors["x"])
    assert float(loaded.timed(kernels.total)()) == float(
        (tensors["x"] * 2.0)[1:].sum() + 12.0 + tensors["x"].sum()
    )


@helion.kernel(autotune_effort="none", static_shapes=True)
def _doubled(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * 2.0
    return out


@onlyBackends(["triton", "cute"])
@skipIfRefEager("ref mode compiles no module")
def test_a_compiled_kernels_step_keys_the_cache_alike_after_a_reload() -> None:
    # helion.inductor times a kernel behind the call before it as a step that
    # runs that call's compiled module (``bound.compile_config``). The search
    # caches its winner under the step's fingerprint, so the fingerprint must
    # name the module by its code alone: the same kernel at the same config,
    # compiled again in another process (here: a fresh bind, which cpu_bind
    # makes outside the kernel's caches, and a module loaded afresh from
    # PyCodeCache), keys the same.
    x = torch.randn(256, 64)

    def step() -> tuple[object, str | None]:
        with cpu_target():
            bound = cpu_bind(_doubled, (x,))
            with (
                patch.object(bound.env.backend, "setup_compile_cache_dir"),
                patch.object(bound.env.backend, "after_compile"),
            ):
                compiled = bound.compile_config(
                    bound.config_spec.default_config(), allow_print=False
                )
        prelude = BenchmarkPrelude((PreludeStep(compiled, (x,)),), ())
        return compiled, prelude.fingerprint()

    first, key = step()
    assert key is not None
    PyCodeCache.cache_clear()
    again, key_again = step()
    assert again is not first
    assert key_again == key


def test_a_prelude_that_cannot_cross_says_so(tmp_path: Path) -> None:
    def local(x: torch.Tensor) -> torch.Tensor:
        return x

    prelude = BenchmarkPrelude((PreludeStep(local, (torch.ones(2),)),), ())
    with pytest.raises(PreludeUnsupported, match="cannot reach the benchmark worker"):
        prelude.save(str(tmp_path / "prelude.pt"))


def test_timed_behind_names_a_prelude_inside_its_block_only() -> None:
    prelude = BenchmarkPrelude((PreludeStep(len, ((),)),), ())
    assert active_prelude() is None
    with timed_behind(prelude):
        assert active_prelude() is prelude
        # None and a prelude without steps name nothing.
        with timed_behind(None):
            assert active_prelude() is None
        with timed_behind(BenchmarkPrelude((), (1,))):
            assert active_prelude() is None
        assert active_prelude() is prelude
    assert active_prelude() is None
    with pytest.raises(RuntimeError, match="boom"), timed_behind(prelude):
        raise RuntimeError("boom")
    assert active_prelude() is None


@pytest.fixture
def event_timer(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``do_bench`` on a stand-in Triton driver: the log receives every
    flush, release of pinned lines, event record and synchronization."""
    return stand_in_event_timer(monkeypatch, sweep="release")


def test_do_bench_runs_the_prelude_between_the_flush_and_each_timed_call(
    event_timer: list[str],
) -> None:
    def call() -> None:
        event_timer.append("call")

    def bench(**kwargs: object) -> list[str]:
        event_timer.clear()
        assert benchmarking.do_bench(
            call, fixed_repetitions=2, pre_warmed=True, return_mode="median", **kwargs
        ) == pytest.approx(2.0)
        return list(event_timer)

    # Timed alone, as before the prelude existed: flush, then the call
    # between its two events.
    alone = ["sync", *["flush", "event", "call", "event"] * 2, "sync"]
    assert bench() == alone
    assert bench(prelude=None) == alone
    # Behind a prelude the window first displaces the lines earlier windows
    # pinned, and the work follows every flush, outside the events.
    behind = [
        "sync",
        "release",
        *["flush", "prelude", "event", "call", "event"] * 2,
        "sync",
    ]
    assert bench(prelude=LoggedPrelude(event_timer)) == behind
    assert bench(prelude=LoggedPrelude(event_timer), l2_flush_mode="call") == behind


def test_a_long_first_launch_behind_a_prelude_is_not_its_own_sample(
    event_timer: list[str],
) -> None:
    """With ``probe_long_kernel`` a first flushed launch longer than the
    windows is the window's one sample. Behind a prelude that launch holds
    the work, so one more flushed call is timed between its own events."""

    def bench(**kwargs: object) -> list[str]:
        event_timer.clear()
        # every event pair reads 2 ms: longer than both windows
        assert benchmarking.do_bench(
            lambda: event_timer.append("call"),
            warmup=1,
            rep=1,
            probe_long_kernel=True,
            pre_warmed=True,
            return_mode="median",
            **kwargs,
        ) == pytest.approx(2.0)
        return list(event_timer)

    probe = ["event", "flush", "call", "event", "sync"]
    assert bench() == ["sync", *probe]
    assert bench(prelude=LoggedPrelude(event_timer)) == [
        "sync",
        "release",
        *["event", "flush", "prelude", "call", "event", "sync"],
        *["flush", "prelude", "event", "call", "event"],
        "sync",
    ]


def test_do_bench_refuses_a_prelude_through_a_graph_replay(
    event_timer: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay runs the call on the arguments of its capture, and the
    prelude's work makes new ones before every call."""
    monkeypatch.setattr(
        benchmarking,
        "_maybe_cudagraph_replay",
        lambda fn, **_: lambda: fn(),
    )
    with pytest.raises(ValueError, match="CUDA graph replays"):
        benchmarking.do_bench(
            lambda: None,
            fixed_repetitions=1,
            pre_warmed=True,
            prelude=LoggedPrelude(event_timer),
        )
    assert "prelude" not in event_timer and "release" not in event_timer


def test_a_call_of_several_kernels_runs_eagerly_behind_a_prelude(
    event_timer: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call that launches several kernels (a split-K workspace schedule's
    producer and reducer) is timed as a graph replay alone; behind a prelude
    it runs eagerly, on the arguments the prelude's work has just made,
    instead of raising as an asked-for replay does."""
    replays: list[object] = []

    def replay_of(fn: object) -> object:
        replays.append(fn)
        return lambda: None

    monkeypatch.setattr(benchmarking, "_cudagraph_unavailable_reason", lambda: None)
    monkeypatch.setattr(benchmarking, "_make_cudagraph_replay", replay_of)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)

    def host() -> None:
        event_timer.append("call")

    host._helion_cute_launches = 2  # pyrefly: ignore
    assert benchmarking._maybe_cudagraph_replay(host, replay_launches=True) is not host
    replays.clear()
    benchmarking.do_bench(
        host, fixed_repetitions=1, pre_warmed=True, prelude=LoggedPrelude(event_timer)
    )
    assert replays == []
    assert event_timer[-6:] == ["flush", "prelude", "event", "call", "event", "sync"]


@pytest.mark.parametrize("regime", ["off", "replay"])
def test_do_bench_refuses_a_prelude_in_a_regime_without_a_flush_per_call(
    event_timer: list[str], regime: str
) -> None:
    with pytest.raises(ValueError, match="flushes L2 before every timed call"):
        benchmarking.do_bench(
            lambda: None,
            fixed_repetitions=1,
            pre_warmed=True,
            l2_flush_mode=regime,
            prelude=LoggedPrelude(event_timer),
        )
    # (and a backward benchmark, whose state the prelude would not reset:
    # the regime is not what is wrong there)
    with pytest.raises(ValueError, match="forward benchmark, not of a backward"):
        benchmarking.do_bench(
            lambda: None,
            fixed_repetitions=1,
            pre_warmed=True,
            grad_to_none=[torch.ones(1)],
            prelude=LoggedPrelude(event_timer),
        )


def test_a_traced_window_behind_a_prelude_reads_the_candidates_kernels_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trace of a window behind a prelude is [flush, work, marker, call]
    per call. The events before a marker are learned for the prelude, apart
    from the flush's own footprint, and a sample is the span of the kernels
    after the marker: the candidate's."""
    trace = stand_in_trace(monkeypatch)
    log, launch, device, flush = trace.log, trace.launch, trace.device, trace.flush
    traced = benchmarking._cuda_kernel_events

    class Work:
        def run(self) -> None:
            # two kernels, each far longer than the candidate's
            log.append(("producer", 500_000))
            log.append(("producer", 300_000))

    def window(clear_cache: Callable[[], None], prelude: object) -> object:
        return benchmarking._do_bench_kernel_durations(
            launch("candidate", 7_000),
            clear_cache,
            device,
            warmup=0,
            rep=1,
            quantiles=None,
            return_mode="median",
            fixed_repetitions=3,
            prelude=prelude,
        )

    work = Work()
    behind = functools.partial(benchmarking._flush_then_prelude, flush, work)
    assert window(behind, work) == pytest.approx(0.007)
    assert benchmarking._prelude_flush_traces[work] == 3
    # The flush alone keeps its own footprint, learned apart.
    assert benchmarking._flush_trace is None
    assert window(flush, None) == pytest.approx(0.007)
    assert benchmarking._flush_trace == 1
    # The footprint of a prelude is learned once: a second window traces
    # its own calls only.
    sessions: list[int] = []
    monkeypatch.setattr(
        benchmarking,
        "_cuda_kernel_events",
        lambda run: sessions.append(1) or traced(run),
    )
    assert window(behind, work) == pytest.approx(0.007)
    assert len(sessions) == 1

    # A search's traced candidate window releases the pinned lines itself
    # when it is timed alone; behind a prelude ``do_bench`` has done so.
    released: list[object] = []
    monkeypatch.setattr(benchmarking, "_release_l2_pins", released.append)

    def candidate_window(clear_cache: Callable[[], None], prelude: object) -> object:
        return benchmarking._traced_candidate_window(
            launch("candidate", 7_000),
            clear_cache,
            device,
            flush_buffer="buffer",
            warmup=0,
            rep=1,
            quantiles=None,
            return_mode="median",
            fixed_repetitions=3,
            process_group_name=None,
            prelude=prelude,
        )

    assert candidate_window(flush, None) == pytest.approx(0.007)
    assert released == ["buffer"]
    assert candidate_window(behind, work) == pytest.approx(0.007)
    assert released == ["buffer"]


def test_work_that_differs_from_run_to_run_has_no_trace_reading(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The trace finds a candidate's kernels by the number of events the
    flush and the work issue before its marker. Work whose runs differ has
    no such number: the worker asks before a traced window and gives no
    reading, where ``do_bench`` alone would fall back to CUDA events."""
    pytest.importorskip("triton")
    log: list[str] = []

    def traced(run: Callable[[], None]) -> list[benchmarking._TraceEvent]:
        log.clear()
        run()
        return [
            benchmarking._TraceEvent(name, index, index + 1)
            for index, name in enumerate(log)
        ]

    monkeypatch.setattr(
        benchmarking, "_trace_marker", lambda: log.append(benchmarking._TRACE_MARKER)
    )
    monkeypatch.setattr(benchmarking, "_cuda_kernel_events", traced)
    monkeypatch.setattr(
        benchmarking, "_make_l2_cache_clearer", lambda: lambda: log.append("flush")
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    class Work:
        def __init__(self, launches: list[int]) -> None:
            self.launches = launches

        def run(self) -> None:
            log.extend(["producer"] * self.launches.pop(0))

    steady = Work([2, 2])
    assert benchmarking.prelude_trace_unready_reason(steady) is None
    assert benchmarking._prelude_flush_traces[steady] == 3
    # (learned once: a second question traces nothing)
    assert benchmarking.prelude_trace_unready_reason(steady) is None
    varying = Work([2, 3])
    with caplog.at_level("WARNING", logger=benchmarking.__name__):
        reason = benchmarking.prelude_trace_unready_reason(varying)
    assert reason is not None and "not the same" in reason
    assert benchmarking._prelude_flush_traces[varying] == 0
    # The warning is the prelude's: its windows have no trace reading (the
    # flush's own says that windows are timed with CUDA events instead).
    (warning,) = caplog.messages
    assert "9 events" in warning and "have no trace reading" in warning
    caplog.clear()
    monkeypatch.setattr(benchmarking, "_flush_trace", None)
    foreign = iter([1, 2])
    with caplog.at_level("WARNING", logger=benchmarking.__name__):
        assert (
            benchmarking._learn_flush_trace(
                lambda: log.extend(["flush"] * next(foreign)), torch.cuda
            )
            == 0
        )
    (warning,) = caplog.messages
    assert "5 events for two flushed markers" in warning
    assert "timing per-call flushed with CUDA events" in warning


def _job_stand_ins(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., object], args: tuple
) -> list[dict[str, object]]:
    """A ``BenchmarkJob`` whose compiled function, arguments and timer are
    stand-ins; returns what the timer was called with."""
    calls: list[dict[str, object]] = []

    def bench(call: Callable[[], object], **kwargs: object) -> float:
        calls.append({"call": call, "result": call(), **kwargs})
        return 5.0

    monkeypatch.setattr(benchmark_job, "_load_compiled_fn_for_worker", lambda spec: fn)
    monkeypatch.setattr(benchmark_job, "_unload_compiled_fn", lambda fn: None)
    monkeypatch.setattr(benchmark_job, "load_trusted_kernel_args", lambda path: args)
    monkeypatch.setattr(benchmark_job, "steady_graph_dispatch", lambda **kwargs: None)
    monkeypatch.setattr(benchmark_job, "synchronize_device", lambda: None)
    monkeypatch.setattr(benchmark_job, "do_bench", bench)
    monkeypatch.delenv("HELION_AUTOTUNE_L2_FLUSH", raising=False)
    return calls


def test_a_benchmark_job_times_its_candidate_behind_the_saved_prelude(
    monkeypatch: pytest.MonkeyPatch, kernels: types.ModuleType, tmp_path: Path
) -> None:
    sequence, tensors = _sequence(kernels)
    trusted = (tensors["scaled"], tensors["buffer"])
    path = str(tmp_path / "job_prelude.pt")
    sequence.prelude(trusted).save(path)
    seen: list[tuple[int, ...]] = []

    def candidate(scaled: torch.Tensor, buffer: torch.Tensor) -> float:
        seen.append(tuple(t.untyped_storage().data_ptr() for t in (scaled, buffer)))
        return float(scaled.sum() + buffer.sum())

    calls = _job_stand_ins(monkeypatch, candidate, trusted)
    expected = float((tensors["x"] * 2.0).sum() + 12.0)

    # Without a prelude the job is the job it was: the candidate on the
    # trusted arguments, and a timer that is told of no prelude.
    assert benchmark_job.BenchmarkJob(None, "unused")() == 5.0
    (alone,) = calls
    assert "prelude" not in alone
    assert isinstance(alone["call"], functools.partial)
    assert alone["call"].args == trusted and alone["result"] == expected
    trusted_storages = tuple(t.untyped_storage().data_ptr() for t in trusted)
    assert seen == [trusted_storages] * 2

    # Behind one, the candidate takes what the prelude's work leaves it,
    # and the timer gets the prelude to run behind every flush.
    calls.clear()
    seen.clear()
    assert benchmark_job.BenchmarkJob(None, "unused", prelude_path=path)() == 5.0
    (behind,) = calls
    prelude = behind["prelude"]
    assert prelude is load_prelude(path) and behind["result"] == expected
    assert len(seen) == 2 and seen[0] == seen[1]
    assert set(seen[0]).isdisjoint(trusted_storages)
    prelude.run()
    assert behind["call"]() == expected
    assert seen[-1][0] != seen[0][0]


def test_a_prelude_that_fails_before_the_candidate_runs_says_so(
    monkeypatch: pytest.MonkeyPatch, kernels: types.ModuleType, tmp_path: Path
) -> None:
    """The prelude's own failure, in the worker: its file does not load, or
    a step raises on its first run. The candidate was not called."""
    called: list[int] = []
    _job_stand_ins(monkeypatch, lambda *args: called.append(1), ())
    missing = str(tmp_path / "no_prelude.pt")
    with pytest.raises(PreludeFailed, match="before the candidate's first call"):
        benchmark_job.BenchmarkJob(None, "unused", prelude_path=missing)()
    # (``total`` of a number: a step that raises)
    path = str(tmp_path / "failing_prelude.pt")
    BenchmarkPrelude((PreludeStep(kernels.total, (3,)),), ()).save(path)
    with pytest.raises(PreludeFailed, match="AttributeError"):
        benchmark_job.BenchmarkJob(None, "unused", prelude_path=path)()
    assert not called


def test_a_traced_job_behind_work_the_trace_cannot_split_gives_no_reading(
    monkeypatch: pytest.MonkeyPatch, kernels: types.ModuleType, tmp_path: Path
) -> None:
    sequence, tensors = _sequence(kernels)
    path = str(tmp_path / "untraced_prelude.pt")
    sequence.prelude((tensors["scaled"],)).save(path)
    calls = _job_stand_ins(monkeypatch, lambda scaled: None, ())
    asked: list[object] = []
    monkeypatch.setattr(benchmark_job, "kernel_timer_unready_reason", lambda: None)
    monkeypatch.setattr(
        benchmark_job,
        "prelude_trace_unready_reason",
        lambda prelude: asked.append(prelude) or "its work differs",
    )
    # The final selection's ``kernel`` window, and a search's traced one.
    with pytest.raises(RuntimeError, match="unavailable .*: its work differs"):
        benchmark_job.BenchmarkJob(
            None, "unused", prelude_path=path, l2_flush_mode="kernel"
        )()
    with pytest.raises(benchmarking.KernelTraceUnreadable, match="its work differs"):
        benchmark_job.BenchmarkJob(
            None, "unused", prelude_path=path, trace_kernel_durations=True
        )()
    assert asked == [load_prelude(path)] * 2 and not calls
    # An event window does not ask, and a job without a prelude never does.
    asked.clear()
    assert benchmark_job.BenchmarkJob(None, "unused", prelude_path=path)() == 5.0
    monkeypatch.setattr(benchmark_job, "load_trusted_kernel_args", lambda path: (1,))
    assert benchmark_job.BenchmarkJob(None, "unused", l2_flush_mode="kernel")() == 5.0
    assert not asked


def test_the_workers_trace_probe_asks_about_the_prelude_too(
    monkeypatch: pytest.MonkeyPatch, kernels: types.ModuleType, tmp_path: Path
) -> None:
    """The search asks the worker once whether it can time from the trace,
    before it moves its candidates there and before its final selection
    labels its windows ``kernel``. Behind a prelude the answer covers the
    prelude's work: a search whose prelude cannot be traced stays on CUDA
    events instead of asking for windows that give no reading."""
    sequence, tensors = _sequence(kernels)
    path = str(tmp_path / "probed_prelude.pt")
    sequence.prelude((tensors["scaled"],)).save(path)
    asked: list[object] = []
    monkeypatch.setattr(benchmark_job, "kernel_timer_unready_reason", lambda: None)
    monkeypatch.setattr(
        benchmark_job,
        "prelude_trace_unready_reason",
        lambda prelude: asked.append(prelude) or "its work differs",
    )
    assert benchmark_job.KernelTimerProbeJob()() is None
    assert not asked
    assert benchmark_job.KernelTimerProbeJob(prelude_path=path)() == "its work differs"
    assert asked == [load_prelude(path)] and asked[0].args is not None
    # (a worker that cannot trace at all says so first)
    monkeypatch.setattr(benchmark_job, "kernel_timer_unready_reason", lambda: "held")
    assert benchmark_job.KernelTimerProbeJob(prelude_path=path)() == "held"
    assert len(asked) == 1


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"l2_flush_mode": "replay"}, "flushes L2 before every timed call"),
        ({"l2_flush_mode": "off"}, "flushes L2 before every timed call"),
        ({"use_wall_clock": True}, "CUDA event or trace timer"),
    ],
)
def test_a_benchmark_job_refuses_a_prelude_it_cannot_time(
    monkeypatch: pytest.MonkeyPatch, fields: dict[str, object], message: str
) -> None:
    _job_stand_ins(monkeypatch, lambda: None, ())
    job = benchmark_job.BenchmarkJob(None, "unused", prelude_path="unused", **fields)
    with pytest.raises(ValueError, match=message):
        job()


def _provider(
    monkeypatch: pytest.MonkeyPatch, **settings: object
) -> LocalBenchmarkProvider:
    """A provider as its constructor finds it when it checks a prelude: the
    built-in worker on a CUDA kernel of a backend with the default timers."""
    provider = object.__new__(LocalBenchmarkProvider)
    provider.settings = Settings(**settings)
    provider.args = ()
    provider._benchmark_prelude = BenchmarkPrelude((PreludeStep(len, ((),)),), ())
    provider._input_bank_count = provider.settings.autotune_benchmark_input_banks
    provider.config_spec = SimpleNamespace(
        backend=SimpleNamespace(
            get_do_bench=lambda: None,
            prefers_wall_clock_bench=lambda: False,
            default_final_selection_regime=lambda: "kernel",
        ),
        cute_flash_search_enabled=False,
    )
    provider.kernel = SimpleNamespace(
        env=SimpleNamespace(device=torch.device("cuda")),
        supports_subprocess_benchmark=lambda: True,
    )
    for name in (
        "HELION_AUTOTUNE_L2_FLUSH",
        "HELION_AUTOTUNE_FINAL_L2_FLUSH",
        "HELION_AUTOTUNE_TARGET_REGIME",
        "HELION_BENCHMARK_CUDAGRAPH",
    ):
        monkeypatch.delenv(name, raising=False)
    # (an NVIDIA build, whatever build runs the test)
    monkeypatch.setattr(torch.version, "hip", None)
    return provider


def test_a_provider_takes_a_prelude_with_the_default_worker_and_regimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provider(monkeypatch)._check_benchmark_prelude()
    # The trace-timed final selection of a backend, and one on CUDA events.
    provider = _provider(monkeypatch)
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_L2_FLUSH", "1")
    assert benchmarking.final_selection_l2_flush_mode("kernel") == "call"
    provider._check_benchmark_prelude()


@pytest.mark.parametrize(
    "variable,value",
    [
        ("HELION_AUTOTUNE_L2_FLUSH", "0"),
        ("HELION_AUTOTUNE_L2_FLUSH", "replay"),
        ("HELION_AUTOTUNE_FINAL_L2_FLUSH", "replay"),
        ("HELION_AUTOTUNE_TARGET_REGIME", "0"),
    ],
)
def test_a_provider_refuses_a_prelude_in_a_regime_without_a_flush_per_call(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    provider = _provider(monkeypatch)
    monkeypatch.setenv(variable, value)
    with pytest.raises(PreludeRefused, match="must flush per call"):
        provider._check_benchmark_prelude()


@pytest.mark.parametrize(
    "settings",
    [
        {"autotune_benchmark_subprocess": False},
        {"autotune_benchmark_fn": lambda fns, repeat, desc=None: [0.0] * len(fns)},
        {"autotune_benchmark_input_banks": 2},
    ],
)
def test_a_provider_refuses_a_prelude_it_would_time_outside_the_worker(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, object]
) -> None:
    provider = _provider(monkeypatch, **settings)
    with pytest.raises(PreludeRefused, match="built-in benchmark worker"):
        provider._check_benchmark_prelude()


def test_a_provider_refuses_a_prelude_where_its_timers_do_not_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another device than an NVIDIA one, a search whose refinement times
    in this process, and candidates timed as CUDA graph replays."""
    provider = _provider(monkeypatch)
    provider.kernel.env.device = torch.device("cpu")
    # a PreludeRefused is still the InvalidAPIUsage it was
    with pytest.raises(exc.InvalidAPIUsage, match="NVIDIA device"):
        provider._check_benchmark_prelude()
    provider = _provider(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    with pytest.raises(PreludeRefused, match="NVIDIA device"):
        provider._check_benchmark_prelude()
    provider = _provider(monkeypatch)
    provider.config_spec.cute_flash_search_enabled = True
    with pytest.raises(PreludeRefused, match="in-process refinement"):
        provider._check_benchmark_prelude()
    provider = _provider(monkeypatch)
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "1")
    with pytest.raises(PreludeRefused, match="HELION_BENCHMARK_CUDAGRAPH"):
        provider._check_benchmark_prelude()
    monkeypatch.setenv("HELION_BENCHMARK_CUDAGRAPH", "0")
    provider._check_benchmark_prelude()


def test_a_provider_refuses_a_prelude_of_another_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A block that covers another kernel's search, or arguments that are
    not given as the compiled function takes them (a default left out)."""
    provider = _provider(monkeypatch)
    provider._benchmark_prelude = BenchmarkPrelude(
        (PreludeStep(len, ((),)),), (torch.zeros(2, 3),)
    )
    provider.args = (torch.zeros(2, 3),)
    provider._check_benchmark_prelude()
    for args in ((torch.zeros(3, 2),), (torch.zeros(2, 3), 1.0)):
        provider.args = args
        with pytest.raises(PreludeRefused, match="not of this search's call"):
            provider._check_benchmark_prelude()


def test_a_wrapper_the_worker_cannot_load_costs_one_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a prelude such a wrapper turns the worker off and the search
    goes on in this process. Behind a prelude only the worker can time, so
    the candidate is rejected and the worker stays for the others."""
    provider = _provider(monkeypatch)
    provider._precompile_args_path = "unused"
    provider._subprocess_wrapper_unloadable = False
    provider.mutated_arg_indices = []
    provider._subprocess_benchmark_uses_wall_clock = lambda: False
    provider._probe_long_kernel = lambda: False
    provider._worker_compile_timeout = lambda: 30
    disabled: list[object] = []
    provider._disable_unloadable_benchmark_worker = disabled.append

    def unloadable(job: object, **kwargs: object) -> float:
        raise benchmark_job.CompiledFunctionLoadError("ImportError: no module")

    provider._get_benchmark_worker = lambda: SimpleNamespace(run=unloadable)
    monkeypatch.setattr(benchmark_provider, "_serialize_compiled_fn", lambda fn: fn)
    with pytest.raises(benchmark_job.CompiledFunctionLoadError):
        provider._run_subprocess_benchmark_job(len, warmup=1, rep=50)
    assert not disabled
    provider._benchmark_prelude = None
    assert provider._run_subprocess_benchmark_job(len, warmup=1, rep=50) is None
    assert len(disabled) == 1


def test_a_candidate_whose_prelude_failed_has_no_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(monkeypatch)
    warnings: list[str] = []
    provider.log = SimpleNamespace(warning=warnings.append)
    provider._autotune_metrics = SimpleNamespace(
        num_worker_failures=0, num_compile_failures=0, num_configs_tested=3
    )
    provider._worker_failure_config_ids = []
    provider._timeout_retry_repetitions = lambda config, source_hash: None

    def failed(config: object, fn: object) -> float:
        raise PreludeFailed("the benchmark prelude failed: boom")

    assert provider.prelude_failure_cascade() is None
    provider._run_candidate_benchmark_job = failed
    assert provider._benchmark_function_subprocess("config", len) == float("inf")
    assert warnings == ["No reading of 'config': the benchmark prelude failed: boom"]
    # the prelude's failure is not a compile failure of the config
    assert provider._autotune_metrics.num_worker_failures == 1
    assert provider._autotune_metrics.num_compile_failures == 0
    # a search that ends without a config says the prelude took this reading
    assert provider.prelude_failure_cascade() == (
        "1 of the 3 tested configs had no reading because of the benchmark "
        "prelude they were timed behind"
    )


@pytest.mark.parametrize("took_readings", [False, True])
def test_a_search_the_prelude_took_readings_from_fails_as_such(
    took_readings: bool,
) -> None:
    """A search that ends without a config raises ``PreludeSearchFailed``
    (a ``NoConfigFound``) when the prelude took readings from it, so that a
    caller can tune the kernel alone; otherwise the ``NoConfigFound`` of a
    kernel without a working config, as alone."""

    def no_config() -> object:
        raise exc.NoConfigFound

    settings = Settings(autotune_log=False, autotune_log_details=False)
    search = object.__new__(BaseSearch)
    search.settings = settings
    search.log = AutotuningLogger(settings)
    search.args = ()
    search.kernel = Mock()
    search.kernel.get_cached_path.return_value = None
    search.benchmark_provider = Mock()
    search.benchmark_provider.compile_deadline_cascade.return_value = None
    detail = "2 of the 3 tested configs had no reading because of the prelude"
    search.benchmark_provider.prelude_failure_cascade.return_value = (
        detail if took_readings else None
    )
    search._prepare = Mock()
    search._autotune_metrics = AutotuneMetrics()
    search._finalize_autotune_metrics = Mock()
    search._autotune = no_config
    with pytest.raises(exc.NoConfigFound) as raised:
        search.autotune()
    assert isinstance(raised.value, PreludeSearchFailed) is took_readings
    assert (detail in str(raised.value)) is took_readings


def test_a_provider_hands_its_preludes_file_to_every_worker_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs: list[benchmark_job.BenchmarkJob] = []
    provider = _provider(monkeypatch)
    provider._precompile_args_path = "unused"
    provider._subprocess_wrapper_unloadable = False
    provider.mutated_arg_indices = []
    provider._subprocess_benchmark_uses_wall_clock = lambda: False
    provider._probe_long_kernel = lambda: False
    provider._worker_compile_timeout = lambda: 30
    provider._get_benchmark_worker = lambda: SimpleNamespace(
        run=lambda job, **kwargs: jobs.append(job) or 4.0
    )
    monkeypatch.setattr(benchmark_provider, "_serialize_compiled_fn", lambda fn: fn)
    # No prelude (the class's default, as outside ``timed_behind``): the job
    # names none.
    assert provider._run_subprocess_benchmark_job(len, warmup=1, rep=50) == 4.0
    provider._prelude_path = "prelude.pt"
    provider._run_subprocess_benchmark_job(
        len, warmup=1, rep=50, l2_flush_mode="kernel"
    )
    assert [job.prelude_path for job in jobs] == [None, "prelude.pt"]
    # Its finalists share the prelude's tensors: no private copies.
    provider._benchmark_prelude = BenchmarkPrelude((PreludeStep(len, ((),)),), ())
    provider._private_argument_copies_disabled = False
    assert not provider.private_argument_copies_affordable(4)


def test_a_search_behind_a_prelude_reads_in_the_worker(
    kernels: types.ModuleType,
) -> None:
    prelude = BenchmarkPrelude((PreludeStep(len, ((),)),), ())
    search = object.__new__(PopulationBasedSearch)
    search._input_bank_count = 1
    assert search._benchmark_prelude is None
    search._benchmark_prelude = prelude
    assert search._final_rebenchmark_use_isolated()
    # It runs inside the block it was made in, on the built-in provider:
    # the provider it makes there is the one that carries the prelude.
    search._prepared = False
    search._benchmark_provider_cls = LocalBenchmarkProvider
    with pytest.raises(exc.InvalidAPIUsage, match="block it was made in"):
        search._prepare()
    search._benchmark_provider_cls = object
    with (
        timed_behind(prelude),
        pytest.raises(exc.InvalidAPIUsage, match="built-in benchmark provider"),
    ):
        search._prepare()
    # (and a search made outside any block does not run inside one)
    outside = object.__new__(PopulationBasedSearch)
    outside._prepared = False
    with (
        timed_behind(prelude),
        pytest.raises(exc.InvalidAPIUsage, match="block it was made in"),
    ):
        outside._prepare()
    # Its members are never timed by this process's mirrored sweeps.
    search.benchmark_provider = SimpleNamespace()
    search.settings = Settings()
    with pytest.raises(exc.AutotuneError, match="behind a benchmark prelude"):
        search.mirrored_rebenchmark([object(), object()], desc="refinement")

    def policy(prelude: BenchmarkPrelude | None) -> tuple[str, bool]:
        stand_in = SimpleNamespace(
            settings=Settings(),
            _benchmark_provider_cls=LocalBenchmarkProvider,
            _benchmark_prelude=prelude,
            _search_policy_cacheable=True,
        )
        key = _search_policy_hash(stand_in, cute_flash_search_enabled=False)
        return key, stand_in._search_policy_cacheable

    # A search without a prelude keeps its key (the one-bank key of every
    # default search); one behind a prelude whose function has no source to
    # name it by (a builtin) gets a key of its own each time and stays out
    # of the automatic cache.
    assert policy(None) == (_hash_search_policy(b""), True)
    assert policy(prelude)[0] != policy(None)[0]
    assert policy(prelude)[0] != policy(prelude)[0]
    assert not policy(prelude)[1]
    # Behind a step with a source, the key names the step's code and the
    # layout of the arguments, which buffers they share included: the same
    # work, with other tensors of the same layout, finds the same key and
    # stays in the cache; other work, or other layouts, another key.
    x, y = torch.ones(4, 3), torch.ones(4, 3)

    def behind(step_fn: object, *step_args: object, timed: tuple = ()) -> str:
        key, cacheable = policy(
            BenchmarkPrelude((PreludeStep(step_fn, step_args),), timed)
        )
        assert cacheable
        return key

    key = behind(kernels.scale, x, 2.0, timed=(PreludeOutput(0, 0, (4, 3), (3, 1), 0),))
    assert key not in (policy(None)[0], policy(prelude)[0])
    assert key == behind(
        kernels.scale,
        torch.zeros(4, 3),
        2.0,
        timed=(PreludeOutput(0, 0, (4, 3), (3, 1), 0),),
    )
    assert key != behind(
        kernels.pair, x, timed=(PreludeOutput(0, 0, (4, 3), (3, 1), 0),)
    )
    assert key != behind(
        kernels.scale, x.t(), 2.0, timed=(PreludeOutput(0, 0, (4, 3), (3, 1), 0),)
    )
    assert behind(kernels.total, x, y, timed=(x,)) != behind(
        kernels.total, x, x, timed=(x,)
    )
    # Another function of the same module with the same layouts is other
    # work, and so is an equal view of the output at another offset.
    assert key != behind(
        kernels.fill, x, 2.0, timed=(PreludeOutput(0, 0, (4, 3), (3, 1), 0),)
    )
    assert key != behind(
        kernels.scale, x, 2.0, timed=(PreludeOutput(0, 0, (4, 3), (3, 1), 3),)
    )
    # A callback, or any argument but a tensor or a plain value, gives the
    # work no identity (two callbacks would share their type's name): such a
    # prelude stays out of the automatic cache.
    assert not policy(BenchmarkPrelude((PreludeStep(kernels.scale, (x, len)),), ()))[1]
    assert behind(kernels.scale, x, torch.float32, timed=(x,)) != behind(
        kernels.scale, x, torch.bfloat16, timed=(x,)
    )


@pytest.mark.parametrize("l2_flush_mode", ["call", "replay"])
def test_a_call_of_several_kernels_runs_eagerly_on_the_event_timers(
    event_timer: list[str], monkeypatch: pytest.MonkeyPatch, l2_flush_mode: str
) -> None:
    """The event timers read a graph replay on another scale than the eager
    calls of a candidate's rivals (a short replay batched with its launch
    latency subtracted; the ``replay`` regime's burst unflushed), so a call
    that launches several kernels is replayed only for the trace and
    ``kernel`` windows; elsewhere it runs eagerly, as its rivals do."""
    replays: list[object] = []

    def replay_of(fn: object) -> object:
        replays.append(fn)
        return lambda: None

    monkeypatch.setattr(benchmarking, "_cudagraph_unavailable_reason", lambda: None)
    monkeypatch.setattr(benchmarking, "_make_cudagraph_replay", replay_of)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)

    def host() -> None:
        event_timer.append("call")

    host._helion_cute_launches = 2  # pyrefly: ignore
    benchmarking.do_bench(
        host, fixed_repetitions=1, pre_warmed=True, l2_flush_mode=l2_flush_mode
    )
    assert replays == []
    assert "call" in event_timer
