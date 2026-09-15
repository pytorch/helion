from __future__ import annotations

import functools
import math

import pytest
import torch

from helion.autotuner.benchmarking import interleaved_bench

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="Requires the HIP event and graph runtime",
)


def test_interleaved_bench_large_repeat() -> None:
    # This many live events crashed final autotune verification on ROCm 7.2.
    # An uneven repeat also exercises the last partially filled event batch.
    repeat = 20003
    counters = torch.zeros((8, 1024), device="cuda")
    functions = [functools.partial(row.add_, 1) for row in counters]
    timings = interleaved_bench(functions, repeat=repeat)
    assert len(timings) == len(functions)
    assert all(math.isfinite(value) and value > 0 for value in timings)
    # One initial warmup, followed by every requested repetition.
    torch.testing.assert_close(counters, torch.full_like(counters, repeat + 1))


def test_cold_cache_cudagraph_timing() -> None:
    pytest.importorskip("tritonbench.components.do_bench.run")
    from benchmarks.rocm_utils import do_bench_cudagraph_with_cache_clear

    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    output = torch.empty_like(a)
    expected = a @ b
    times = do_bench_cudagraph_with_cache_clear(
        lambda: torch.mm(a, b, out=output), rep=100, return_mode="all"
    )
    assert isinstance(times, list)
    assert len(times) == 10
    assert all(math.isfinite(value) and value > 0 for value in times)
    torch.testing.assert_close(output, expected)
