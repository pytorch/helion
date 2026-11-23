"""``helion.autotuner.benchmark_prelude`` on a device: the timers leave a
prelude's work out of their readings, and a search started inside
``timed_behind`` times its candidates behind real kernels."""

from __future__ import annotations

import functools

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRocm
from helion.autotuner import benchmarking
from helion.autotuner.benchmark_prelude import BenchmarkPrelude
from helion.autotuner.benchmark_prelude import KernelSequence
from helion.autotuner.benchmark_prelude import PreludeOutput
from helion.autotuner.benchmark_prelude import PreludeStep
from helion.autotuner.benchmark_prelude import timed_behind
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
import helion.language as hl

# DEVICE names "cuda" without asking whether a device is visible, so ask.
pytestmark = pytest.mark.skipif(
    DEVICE.type != "cuda" or not torch.cuda.is_available(), reason="requires CUDA"
)


def _produce(source: torch.Tensor, rounds: int, runs: torch.Tensor) -> torch.Tensor:
    """Work of a millisecond or two (hundreds of launches over 16 MB), then
    the tensor the timed call reads: other values on every run, which
    ``runs`` counts in place."""
    scratch = source
    for _ in range(rounds):
        scratch = scratch * 1.0001
    runs += 1
    return scratch + runs


def _candidate(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return torch.add(x, 2.0, out=out)


def _bench_us(call: object, regime: str, **kwargs: object) -> float:
    return 1000.0 * float(
        benchmarking.do_bench(
            call,
            warmup=5,
            rep=60,
            return_mode="median",
            pre_warmed=True,
            l2_flush_mode=regime,
            **kwargs,
        )
    )


# The two timers a search behind a prelude reads its candidates with: one
# CUDA event pair per flushed call, and the kernels after the marker in the
# CUDA activity trace. (CUDA graph replays are refused behind a prelude;
# test_benchmark_prelude.py checks that.)
@skipIfRocm("the kernel trace is NVIDIA-only")
@pytest.mark.parametrize("timer", ["events", "trace"])
def test_a_preludes_work_stays_out_of_the_reading(
    monkeypatch: pytest.MonkeyPatch, timer: str
) -> None:
    for name in ("HELION_AUTOTUNE_L2_FLUSH", "HELION_BENCHMARK_CUDAGRAPH"):
        monkeypatch.delenv(name, raising=False)
    regime = "kernel" if timer == "trace" else "call"
    if timer == "trace" and not benchmarking.kernel_timer_available():
        pytest.skip("this build collects no CUDA activity")
    elements = 1 << 22
    source = torch.randn(elements, device=DEVICE)
    out = torch.empty(elements, device=DEVICE)
    runs = torch.zeros((), dtype=torch.int64, device=DEVICE)
    prelude = BenchmarkPrelude(
        (PreludeStep(_produce, (source, 400, runs)),),
        (PreludeOutput(0, 0, (elements,), (1,), 0), out),
    ).loaded()
    prelude.run()
    timed = prelude.timed(_candidate)
    timed()
    torch.cuda.synchronize()

    alone_us = _bench_us(
        functools.partial(_candidate, prelude.args[0].clone(), out), regime
    )
    runs_before = int(runs)
    behind_us = _bench_us(timed, regime, prelude=prelude)
    # The work ran before the timed calls, and the last of them read what
    # its latest run made: values no earlier run made. Checked here, before
    # anything else runs the work or writes ``out``.
    assert int(runs) > runs_before
    assert torch.equal(out, prelude.args[0] + 2.0)
    # The work alone, timed the same way: far longer than the candidate.
    work_us = _bench_us(prelude.run, regime)
    assert work_us > 10 * alone_us, (work_us, alone_us)
    # Behind it the reading is the candidate's, not the work's: within a
    # twentieth of the work's time of its reading alone.
    assert 0 < behind_us < alone_us + work_us / 20, (behind_us, alone_us, work_us)
    # It is the reading of a candidate that finds its 16 MB input in L2,
    # where the work left it (the sweep of pinned lines that opens the
    # window does not take it out): shorter than alone, where the input is
    # in memory (4.4 against 7.2 us on a B200). Checked in the trace, which
    # reads the kernel itself, on a device whose L2 holds the input and
    # the output twice over (a B200's 126.5 MiB does; the 32 MiB of the two
    # are no more than that of an L2 under 64 MiB, which is not asked).
    l2_bytes = torch.cuda.get_device_properties(DEVICE).L2_cache_size
    if timer == "trace" and l2_bytes >= 2 * (source.nbytes + out.nbytes):
        assert behind_us < alone_us, (behind_us, alone_us, l2_bytes)


@helion.kernel(static_shapes=True, autotune_effort="quick", force_autotune=True)
def _row_sums(x: torch.Tensor) -> torch.Tensor:
    rows, _columns = x.shape
    out = torch.empty([rows], dtype=torch.float32, device=x.device)
    for tile in hl.tile(rows):
        out[tile] = x[tile, :].to(torch.float32).sum(-1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick", force_autotune=True)
def _scale_rows(x: torch.Tensor, sums: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    out = torch.empty([rows, columns], dtype=x.dtype, device=x.device)
    for tile_r, tile_c in hl.tile([rows, columns]):
        out[tile_r, tile_c] = x[tile_r, tile_c] / (sums[tile_r][:, None] + 1.0)
    return out


@pytest.mark.slow
@skipIfRocm("the benchmark worker's CUDA timers")
def test_a_search_inside_timed_behind_times_its_candidates_behind_the_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two kernels as a harness runs them: the second searches inside
    ``timed_behind``, so every job of its benchmark worker carries the
    prelude (the first kernel at its config), and the search still returns
    a config that computes the right values."""
    for name in (
        "HELION_AUTOTUNE_L2_FLUSH",
        "HELION_AUTOTUNE_FINAL_L2_FLUSH",
        "HELION_AUTOTUNE_TARGET_REGIME",
    ):
        monkeypatch.delenv(name, raising=False)
    jobs: list[str | None] = []
    run_job = LocalBenchmarkProvider._run_subprocess_benchmark_job

    def recorded(
        self: LocalBenchmarkProvider, *args: object, **kwargs: object
    ) -> object:
        jobs.append(self._prelude_path)
        return run_job(self, *args, **kwargs)

    monkeypatch.setattr(
        LocalBenchmarkProvider, "_run_subprocess_benchmark_job", recorded
    )
    x = torch.rand(512, 256, device=DEVICE)
    sequence = KernelSequence()
    with timed_behind(sequence.prelude((x,))):
        sums = _row_sums(x)
    # The sequence's first kernel has no prelude: its jobs name none.
    assert jobs and set(jobs) == {None}
    bound = _row_sums.bind((x,))
    sequence.record(bound.compile_config(bound._config, allow_print=False), (x,), sums)
    jobs.clear()
    with timed_behind(sequence.prelude((x, sums))):
        scaled = _scale_rows(x, sums)
    assert jobs and None not in jobs and len(set(jobs)) == 1
    torch.testing.assert_close(
        scaled, x / (x.sum(-1)[:, None] + 1.0), rtol=1e-4, atol=1e-5
    )
