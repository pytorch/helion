"""Benchmark the Helion FP8 RowWise scaled_mm kernels vs vLLM's cutlass kernel.

Compares ``examples/scaled_mm.py`` against ``vllm._custom_ops.cutlass_scaled_mm``
on skinny-M FP8 (e4m3) decode shapes on a single GPU (developed on B200).

Two Helion paths are measured:
  (1) ``scaled_mm``       -- self-contained (allocates + zeros internally),
                             tuned by Helion's native autotuner.
  (2) ``scaled_mm_into``  -- accumulates into a caller pre-zeroed buffer, with no
                             internal allocation/memset, deployed with a
                             double-buffered memset overlap ("ping-pong"): two
                             output buffers and a dedicated zeroing stream so the
                             ~0.7-1.2us output-zeroing that atomics require is
                             hidden behind the previous call's compute. Configs
                             are picked by a deployment-aware search that ranks
                             candidates by their actual ping-pong cudagraph time
                             (the metric that matters and that Helion's
                             single-call autotuner cannot observe).

vLLM and Helion are measured interleaved (alternating each rep, median) inside
amortized cudagraphs (64 calls/graph), because vLLM's measured time swings with
clock/thermal state and interleaving cancels the common drift.

Run:  python benchmarks/scaled_mm_vllm_benchmark.py
"""

from __future__ import annotations

import functools
import logging
import statistics
from typing import TYPE_CHECKING

from examples.scaled_mm import scaled_mm
from examples.scaled_mm import scaled_mm_into
import torch

import helion

if TYPE_CHECKING:
    from collections.abc import Callable

# Helion's autotuner is very chatty ("Skipping config ..."); quiet it down.
logging.disable(logging.WARNING)

DEVICE = "cuda"

# Skinny-M FP8 decode shapes: (M, N, K).
SHAPES = [
    (1, 4096, 4096),
    (16, 4096, 4096),
    (1, 14336, 4096),
    (16, 14336, 4096),
]

# Deployment-aware candidate configs for the ping-pong path: (split_k, bn, bk).
INTO_CANDIDATES = [
    (sk, bn, bk) for sk in (8, 16, 32) for bn in (64, 128, 256) for bk in (128, 256)
]

Inputs = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def make_inputs(m: int, n: int, k: int) -> Inputs:
    a = torch.randn(m, k, device=DEVICE).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device=DEVICE).to(torch.float8_e4m3fn)
    y = b.t()  # [K, N] col-major view -- what _scaled_mm/vLLM consume
    scale_a = (torch.rand(m, 1, device=DEVICE) + 0.5).float()
    scale_b = (torch.rand(1, n, device=DEVICE) + 0.5).float()
    ref = torch._scaled_mm(a, b.t(), scale_a, scale_b, out_dtype=torch.bfloat16)
    return a, b, y, scale_a, scale_b, ref


def relerr(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (out.float() - ref.float()).abs().max().item() / (
        ref.float().abs().max().item() + 1e-6
    )


def time_graph(g: torch.cuda.CUDAGraph, n_calls: int, iters: int = 100) -> float:
    """Return us per call (graph replays `iters` times, `n_calls` kernels each)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / n_calls * 1000


def capture(
    fn: Callable[[], object], n_calls: int = 64
) -> tuple[torch.cuda.CUDAGraph, int]:
    """Warm up then capture `n_calls` invocations of `fn` into one cudagraph."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_calls):
            fn()
    torch.cuda.synchronize()
    return g, n_calls


def capture_pingpong(
    kern: Callable[[torch.Tensor], object],
    bufs: list[torch.Tensor],
    n_calls: int = 64,
) -> tuple[torch.cuda.CUDAGraph, int]:
    """Capture `n_calls` of `kern(buf)` with double-buffered memset overlap.

    While the kernel for buffer j runs on the main stream, buffer j is re-zeroed
    on a side stream for its *next* use, hiding the memset behind compute.
    """
    main = torch.cuda.current_stream()
    zero_stream = torch.cuda.Stream()
    ev_kern = [torch.cuda.Event() for _ in range(2)]
    ev_zero = [torch.cuda.Event() for _ in range(2)]
    for b in bufs:
        b.zero_()
    torch.cuda.synchronize()
    for i in range(5):
        kern(bufs[i % 2])
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(main)
    with torch.cuda.stream(s):
        for b in bufs:
            b.zero_()
        for i in range(3):
            kern(bufs[i % 2])
    main.wait_stream(s)
    for j in range(2):
        bufs[j].zero_()
        ev_zero[j].record(main)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_calls):
            j = i % 2
            main.wait_event(ev_zero[j])  # buffer j is zeroed
            kern(bufs[j])  # GEMM into buffer j (main stream)
            ev_kern[j].record(main)
            zero_stream.wait_event(ev_kern[j])
            with torch.cuda.stream(zero_stream):  # zero j for its NEXT use
                bufs[j].zero_()
                ev_zero[j].record(zero_stream)
    torch.cuda.synchronize()
    return g, n_calls


def into_config(sk: int, bn: int, bk: int) -> helion.Config:
    return helion.Config(
        atomic_indexing=["pointer"],
        block_sizes=[16, bn, bk],
        indexing=["pointer"] * 4,
        num_stages=2,
        num_warps=4,
        pid_type="flat",
        split_k=sk,
    )


def tune_into_pingpong(
    m: int,
    n: int,
    a: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    ref: torch.Tensor,
) -> tuple[float, tuple[int, int, int], object] | None:
    """Pick the scaled_mm_into config with the best ping-pong cudagraph time."""
    best: tuple[float, tuple[int, int, int], object] | None = None
    for sk, bn, bk in INTO_CANDIDATES:
        try:
            kern = helion.kernel(
                scaled_mm_into.fn, config=into_config(sk, bn, bk), static_shapes=True
            )
            tmp = torch.zeros(m, n, device=DEVICE, dtype=torch.bfloat16)
            kern(tmp, a, y, scale_a, scale_b)
            torch.cuda.synchronize()
            if relerr(tmp, ref) >= 0.05:
                continue
            bufs = [
                torch.zeros(m, n, device=DEVICE, dtype=torch.bfloat16) for _ in range(2)
            ]
            call = functools.partial(kern, x=a, y=y, scale_a=scale_a, scale_b=scale_b)
            g, nc = capture_pingpong(call, bufs)
            tm = statistics.median([time_graph(g, nc) for _ in range(8)])
            del g
            torch.cuda.empty_cache()
            if best is None or tm < best[0]:
                best = (tm, (sk, bn, bk), kern)
        except Exception:
            torch.cuda.empty_cache()
            continue
    return best


def main() -> None:
    from vllm import _custom_ops as vllm_ops

    print(
        f"{'shape':>22} | {'vLLM':>8} | {'scaled_mm (HL autotune)':>26} "
        f"| {'scaled_mm_into + ping-pong':>30}"
    )
    print("-" * 96)
    for m, n, k in SHAPES:
        a, b, y, scale_a, scale_b, ref = make_inputs(m, n, k)

        # (1) Helion native autotune of the self-contained kernel.
        scaled_mm.reset()
        out1 = scaled_mm(a, y, scale_a, scale_b)
        torch.cuda.synchronize()
        r1 = relerr(out1, ref)

        # (2) Deployment-aware tuning of the ping-pong path.
        best = tune_into_pingpong(m, n, a, y, scale_a, scale_b, ref)
        assert best is not None, f"no valid scaled_mm_into config for {(m, n, k)}"
        kern_best, cfg_best = best[2], best[1]

        # Build cudagraphs and time interleaved.
        bt, sbt = b.t(), scale_b.t()
        vllm_call = functools.partial(
            vllm_ops.cutlass_scaled_mm, a, bt, scale_a, sbt, out_dtype=torch.bfloat16
        )
        sm_call = functools.partial(scaled_mm, a, y, scale_a, scale_b)
        into_call = functools.partial(
            kern_best, x=a, y=y, scale_a=scale_a, scale_b=scale_b
        )
        gv, nv = capture(vllm_call)
        g1, n1 = capture(sm_call)
        bufs = [
            torch.zeros(m, n, device=DEVICE, dtype=torch.bfloat16) for _ in range(2)
        ]
        g2, n2 = capture_pingpong(into_call, bufs)

        tv_, t1_, t2_ = [], [], []
        for _ in range(15):
            tv_.append(time_graph(gv, nv))
            t1_.append(time_graph(g1, n1))
            t2_.append(time_graph(g2, n2))
        tv = statistics.median(tv_)
        t1 = statistics.median(t1_)
        t2 = statistics.median(t2_)

        def tag(t: float, ref_t: float) -> str:
            return "WIN" if t < ref_t else ("MATCH" if t <= ref_t * 1.05 else "")

        print(
            f"M={m:>2} N={n:>5} K={k:>5} | {tv:7.2f}us | "
            f"{t1:6.2f}us {t1 / tv:.2f}x {tag(t1, tv):5} [rel {r1:.0e}] | "
            f"{t2:6.2f}us {t2 / tv:.2f}x {tag(t2, tv):5} cfg={cfg_best}"
        )
        del gv, g1, g2
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
