"""Cold-L2 timing of CUDA-graph replays, in microseconds, for the KDA verify
megakernel and its baselines.

The L2 is flushed by reading 256 MB before each timed call: unlike the zero fill
of ``_bench.bench_pre_captured_cudagraphs``, a read leaves no dirty lines for the
timed call to write back.
"""

from __future__ import annotations

import functools
import statistics
from typing import TYPE_CHECKING

from pretuned_kernels._bench import balanced_order
from pretuned_kernels._bench import capture_cuda_graph
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


@functools.cache
def _flush_buffers() -> tuple[torch.Tensor, torch.Tensor]:
    # 64 Mi fp32 values: 256 MB, past the L2, summed into a scalar
    return (
        torch.empty(64 * 1024 * 1024, device="cuda"),
        torch.empty((), device="cuda"),
    )


def _flush() -> None:
    source, sink = _flush_buffers()
    torch.sum(source, dim=0, out=sink)


def _event_pairs(
    count: int, rep: int
) -> tuple[list[list[torch.cuda.Event]], list[list[torch.cuda.Event]]]:
    def events() -> list[list[torch.cuda.Event]]:
        return [
            [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
            for _ in range(count)
        ]

    return events(), events()


def _trimmed_mean(values: list[float]) -> float:
    # Event timestamps are quantized (about 0.5 us): a trimmed mean resolves
    # finer than the median.
    values = sorted(values)
    cut = len(values) // 10
    values = values[cut : len(values) - cut]
    return sum(values) / len(values)


def bench_clean(replays: Sequence[Callable[[], object]], rep: int = 200) -> list[float]:
    """Each replay's time after a flush: the trimmed mean of ``rep`` samples, the
    replays interleaved in ``balanced_order``."""
    for _ in range(3):
        for replay in replays:
            _flush()
            replay()
    torch.cuda.synchronize()
    starts, ends = _event_pairs(len(replays), rep)
    for sample in range(rep):
        for index in balanced_order(sample, len(replays)):
            _flush()
            starts[index][sample].record()
            replays[index]()
            ends[index][sample].record()
    torch.cuda.synchronize()
    return [
        _trimmed_mean(
            [s.elapsed_time(e) * 1e3 for s, e in zip(starts[j], ends[j], strict=True)]
        )
        for j in range(len(replays))
    ]


def _median_replay(replays: Sequence[Callable[[], object]], rep: int) -> list[float]:
    """Each replay's median time over ``rep`` back-to-back samples (no flush)."""
    for replay in replays:
        replay()
    torch.cuda.synchronize()
    starts, ends = _event_pairs(len(replays), rep)
    for sample in range(rep):
        for index, replay in enumerate(replays):
            starts[index][sample].record()
            replay()
            ends[index][sample].record()
    torch.cuda.synchronize()
    return [
        statistics.median(
            s.elapsed_time(e) * 1e3 for s, e in zip(starts[j], ends[j], strict=True)
        )
        for j in range(len(replays))
    ]


def bench_steady(
    calls: Sequence[Callable[[], object]], reps_in_graph: int = 20, rep: int = 50
) -> tuple[list[float], float]:
    """Per-call time in steady state, and the flush's: one CUDA graph runs
    ``reps_in_graph`` x (flush, call); a graph of the flushes alone is
    subtracted and the difference divided by the count (the graph launch
    amortizes away, as in a whole-model graph)."""

    def flushed(call: Callable[[], object] | None) -> Callable[[], None]:
        def run() -> None:
            for _ in range(reps_in_graph):
                _flush()
                if call is not None:
                    call()

        return run

    graphs = [capture_cuda_graph(flushed(call))[0] for call in [*calls, None]]
    *times, base = _median_replay([graph.replay for graph in graphs], rep)
    return [(t - base) / reps_in_graph for t in times], base / reps_in_graph


def compare(calls: dict[str, Callable[[], object]], rounds: int = 3) -> None:
    """Print ``rounds`` measurements of each call: one replay of its CUDA graph
    after a flush (``bench_clean``, 400 samples) and the steady state
    (``bench_steady``)."""
    with torch.inference_mode():
        graphs = [capture_cuda_graph(call)[0] for call in calls.values()]
        for _ in range(rounds):
            single = bench_clean([graph.replay for graph in graphs], rep=400)
            steady, _ = bench_steady(list(calls.values()))
            print(
                "single replay:",
                "  ".join(
                    f"{n} {t:.2f} us" for n, t in zip(calls, single, strict=True)
                ),
                "  steady:",
                "  ".join(
                    f"{n} {t:.2f} us" for n, t in zip(calls, steady, strict=True)
                ),
            )
