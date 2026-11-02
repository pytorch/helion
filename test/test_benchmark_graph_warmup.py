"""``steady_graph_dispatch``: the benchmarking helpers drive a process into its
steady CUDA graph dispatch state once, before the first graph they capture,
so every candidate (the first two included) is timed in the state a
long-running process runs in.  The GPU half checks the warm-up really runs
and is idempotent; the CPU half checks the once-per-device contract, the
opt-out, that a failure is reported and retried rather than raised, that a
malformed opt-out value cannot stop a benchmark, and that every capture
helper calls it first."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from helion.autotuner import benchmarking

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def fresh_warmup_state() -> Iterator[None]:
    benchmarking._reset_graph_warmup_for_tests()
    yield
    benchmarking._reset_graph_warmup_for_tests()


def _cuda_available_without_a_gpu():
    """Patch the CUDA availability check and the device query so the
    warm-up's control flow runs on a CPU-only host; the launches themselves
    are patched by each test."""
    return (
        patch.object(benchmarking, "_cudagraph_unavailable_reason", return_value=None),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch, "ones", return_value=torch.empty(0)),
    )


def test_the_warmup_runs_once_per_device() -> None:
    launches = _cuda_available_without_a_gpu()
    with (
        launches[0],
        launches[1],
        launches[2] as ones,
        patch.object(
            benchmarking, "_graph_warmup_launch", return_value=object()
        ) as round_,
    ):
        first = benchmarking.steady_graph_dispatch()
        second = benchmarking.steady_graph_dispatch()
        assert first == second == benchmarking._GRAPH_WARMUP_NOTE
        assert round_.call_count == benchmarking._GRAPH_WARMUP_LAUNCHES
        assert ones.call_args.kwargs["device"] == torch.device("cuda", 0)
        # The arguments of a kernel may live on another device than the
        # current one; that device is warmed once too, and the first stays.
        assert (
            benchmarking.steady_graph_dispatch(device=torch.device("cuda", 1)) == first
        )
        assert ones.call_args.kwargs["device"] == torch.device("cuda", 1)
        assert round_.call_count == 2 * benchmarking._GRAPH_WARMUP_LAUNCHES
        assert (
            benchmarking.steady_graph_dispatch(device=torch.device("cuda:1")) == first
        )
        assert benchmarking.steady_graph_dispatch(device=torch.device("cuda")) == first
        assert round_.call_count == 2 * benchmarking._GRAPH_WARMUP_LAUNCHES
    assert (
        len(benchmarking._graph_warmup_kept) == 2 * benchmarking._GRAPH_WARMUP_LAUNCHES
    )
    assert benchmarking._graph_warmup_notes == {0: first, 1: first}


def test_the_warmup_is_skipped_without_graphs_or_when_opted_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        patch.object(
            benchmarking,
            "_cudagraph_unavailable_reason",
            return_value="CUDA is unavailable",
        ),
        patch.object(benchmarking, "_graph_warmup_launch") as round_,
    ):
        assert benchmarking.steady_graph_dispatch() == "skipped: CUDA is unavailable"
    round_.assert_not_called()
    monkeypatch.setenv(benchmarking._GRAPH_WARMUP_ENV, "0")
    with (
        patch.object(benchmarking, "_cudagraph_unavailable_reason", return_value=None),
        patch.object(benchmarking, "_graph_warmup_launch") as round_,
    ):
        assert (
            benchmarking.steady_graph_dispatch() == benchmarking._GRAPH_WARMUP_OPTED_OUT
        )
    round_.assert_not_called()
    assert benchmarking._graph_warmup_kept == []
    assert benchmarking._graph_warmup_notes == {}


def test_a_malformed_opt_out_value_warns_and_keeps_the_warmup_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The variable is read inside every graph capture and every worker job;
    # a ValueError there would silently drop CUDA-graph timing in-process
    # and fail every candidate in the worker.
    monkeypatch.setenv(benchmarking._GRAPH_WARMUP_ENV, "maybe")
    launches = _cuda_available_without_a_gpu()
    with (
        launches[0],
        launches[1],
        launches[2],
        patch.object(benchmarking, "_graph_warmup_launch", return_value=object()),
        caplog.at_level(logging.WARNING, logger=benchmarking._log.name),
    ):
        assert benchmarking.steady_graph_dispatch() == benchmarking._GRAPH_WARMUP_NOTE
    assert any(
        benchmarking._GRAPH_WARMUP_ENV in record.message
        and "'maybe'" in record.message
        and "stays on" in record.message
        for record in caplog.records
    )


def test_a_failed_warmup_is_reported_and_retried_not_raised() -> None:
    launches = _cuda_available_without_a_gpu()
    with (
        launches[0],
        launches[1],
        launches[2],
        patch.object(
            benchmarking,
            "_graph_warmup_launch",
            side_effect=[object(), RuntimeError("device busy")],
        ) as round_,
    ):
        assert (
            benchmarking.steady_graph_dispatch() == "failed: RuntimeError: device busy"
        )
        assert round_.call_count == 2
        # Nothing of the failed attempt is kept, and the next call tries again.
        assert benchmarking._graph_warmup_kept == []
        assert benchmarking._graph_warmup_notes == {}
        round_.side_effect = None
        round_.return_value = object()
        assert benchmarking.steady_graph_dispatch() == benchmarking._GRAPH_WARMUP_NOTE
        assert round_.call_count == 2 + benchmarking._GRAPH_WARMUP_LAUNCHES
    assert len(benchmarking._graph_warmup_kept) == benchmarking._GRAPH_WARMUP_LAUNCHES


class _WarmedUp(Exception):
    pass


def test_the_capture_helpers_warm_up_before_touching_cuda() -> None:
    with patch.object(benchmarking, "steady_graph_dispatch", side_effect=_WarmedUp):
        with pytest.raises(_WarmedUp):
            benchmarking._make_cudagraph_replay(lambda: None)
        with pytest.raises(_WarmedUp):
            benchmarking._make_batched_cudagraph_replay(
                lambda: None, benchmarking._no_cache_clear, 4
            )


def test_the_graph_timer_warms_up_before_touching_cuda() -> None:
    # The handoff evaluation's timer captures its own graph (and warms the
    # callable on a side stream first): same contract.  It summarizes its
    # samples with triton's helper, so it needs triton installed.
    pytest.importorskip("triton")
    with (
        patch.object(benchmarking, "steady_graph_dispatch", side_effect=_WarmedUp),
        patch.object(benchmarking, "_cudagraph_unavailable_reason", return_value=None),
        pytest.raises(_WarmedUp),
    ):
        benchmarking.do_bench_cuda_graph(lambda: None, reset=lambda: None)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA graphs",
)
def test_the_warmup_launches_and_keeps_its_outputs() -> None:
    note = benchmarking.steady_graph_dispatch()
    assert note == benchmarking._GRAPH_WARMUP_NOTE
    kept = benchmarking._graph_warmup_kept
    assert len(kept) == benchmarking._GRAPH_WARMUP_LAUNCHES
    torch.cuda.synchronize()
    assert all(torch.equal(y, torch.full_like(y, 2.0)) for y in kept)  # pyrefly: ignore
    assert benchmarking.steady_graph_dispatch() == note
    assert benchmarking.steady_graph_dispatch(device=kept[0].device) == note  # pyrefly: ignore
    assert len(benchmarking._graph_warmup_kept) == benchmarking._GRAPH_WARMUP_LAUNCHES
    assert benchmarking._graph_warmup_notes == {torch.cuda.current_device(): note}
