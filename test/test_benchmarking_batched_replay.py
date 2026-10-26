from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from helion._dist_utils import all_ranks_agree
from helion._testing import DEVICE
from helion._testing import skipIfRocm
from helion.autotuner import benchmarking

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
        fn: Callable[[], object] | None, clear_cache: Callable[[], None], calls: int
    ) -> Callable[[], None]:
        replay = make_batched(fn, clear_cache, calls)
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
def test_interleaved_bench_of_nothing_is_empty() -> None:
    # even an empty bench allocates the driver's flush buffer
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
    calls, group = synced[-1]
    assert isinstance(calls, int) and calls > 1 and group == "pg"
    assert agreed == [(True, "pg")]

    timing = benchmarking.do_bench(fn, return_mode="median", process_group_name="pg")
    assert timing > 0
    calls, group = synced[-1]
    assert isinstance(calls, int) and calls > 1 and group == "pg"
    assert agreed[-1] == (True, "pg")
