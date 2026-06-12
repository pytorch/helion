"""fp8 rowwise-scaled matmul on a small-grid shape: bm=128 2-CTA family demo.

Shape M=512, K=6144, N=2048, fp8e4m3 -> bf16, rowwise scales:

    out[m, n] = scale_a[m] * scale_b[n] * sum_k(a[m, k] * b[k, n])

This is the validation benchmark for the bm=128 CtaGroup.TWO tcgen05 family.
The shape has only (512/128) x (2048/128) = 4 x 16 = 64 output tiles at the
(128, 128) MMA tiler, far fewer than the 148 SMs on a B200, so large-tile
configs badly underutilize the machine. The 2-CTA MMA at CTA tile 64x128 with
B multicast across each CTA pair is the fastest expressible Helion config on
this shape; it beats every 1-CTA config, the bm=256 2-CTA config, and aten
``torch._scaled_mm`` (see FINDINGS_512_SHAPE.md).

The config below is a *plain Helion config* -- no hand-edited generated code --
that codegens the bm=128 2-CTA family after the two optimization commits on
this branch (the 2-CTA-at-bm=128 MMA family and the pre-wait rowvec hoist).

Methodology: CUDA graphs with 50 back-to-back launches captured per graph,
median of 100 whole-graph replays via CUDA events, divided by 50. A thermal
warmup runs before timing. Wall-clock (non-graph) is dominated by ~25 us/launch
of Helion host overhead at this tiny kernel time, so always graph-time it.

Run:  HELION_BACKEND=cute python benchmarks/cute/fp8_scaled_mm_small_grid.py
"""

from __future__ import annotations

import os
from typing import Callable

os.environ.setdefault("HELION_BACKEND", "cute")

import torch

import helion
from helion import Config
import helion.language as hl

M, K, N = 512, 6144, 2048
FLOP = 2 * M * N * K


# ---------------------------------------------------------------------------
# Best config for this shape (bm=128 2-CTA, fp8). The non-shape knobs match
# the 4096^3-tuned tcgen05 fp8 config; the levers that matter here are
# block_sizes=[128, 128, 128] + tcgen05_cluster_m=2 (which select the new
# bm=128 CtaGroup.TWO family) and the static-persistent blocked schedule.
# ---------------------------------------------------------------------------
BEST_CONFIG = Config(
    indexing=["tensor_descriptor"] * 5,
    block_sizes=[128, 128, 128],
    pid_type="persistent_blocked",
    tcgen05_persistence_model="static_persistent",
    tcgen05_cluster_m=2,
    tcgen05_cluster_n=1,
    tcgen05_ab_stages=8,
    tcgen05_acc_stages=2,
    tcgen05_c_stages=2,
    tcgen05_l2_swizzle_size=8,
    tcgen05_num_epi_warps=4,
    tcgen05_strategy="role_local_monolithic",
)


@helion.kernel(backend="cute")
def scaled_mm(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a2d: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[tm, tk], y[tk, tn], acc=acc)
        out[tm, tn] = (acc * scale_a2d[tm, tn] * scale_b[tn]).to(torch.bfloat16)
    return out


def _make_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    x = (torch.randn(M, K, device="cuda") * 0.4).to(torch.float8_e4m3fn)
    # K-major (column-major / K-contiguous) B, as the tcgen05 fp8 TMA path
    # prefers for B.
    y = (torch.randn(N, K, device="cuda") * 0.4).to(torch.float8_e4m3fn).t()
    scale_a = (torch.rand(M, device="cuda") + 0.5).float()
    scale_b = (torch.rand(N, device="cuda") + 0.5).float()
    scale_a2d = scale_a[:, None].expand(M, N)
    return x, y, scale_a, scale_b, scale_a2d


def _graph_time_us(
    fn: Callable[[], object], iters: int = 50, replays: int = 100
) -> float:
    """Median per-launch time (us) over a captured CUDA graph of ``iters``.

    Each measurement event-times one whole-graph replay (``iters`` launches)
    with a single device sync per replay batch -- no per-launch sync inside
    the captured graph, so the measured time is the steady-state back-to-back
    kernel time, not launch latency.
    """
    # Warmup + capture.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(replays):
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / iters)  # ms per launch
    times.sort()
    return times[len(times) // 2] * 1e3  # -> us


def _sm_clock_mhz() -> tuple[int, int] | None:
    """Return (current_sm_mhz, max_sm_mhz), or None if nvidia-smi is absent.

    fp8 TFLOP/s scales ~linearly with SM clock; absolute numbers are only
    comparable to the FINDINGS reference (1540 TF/s helion / 1209 TF/s aten)
    when the GPU runs at its max SM clock. A clocked-down GPU (common on
    shared/locked hosts) inflates both kernels' us uniformly, so the *ratio*
    stays valid while the absolute TF/s is depressed. Printed so the reader
    can tell which regime they are in.
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,clocks.max.sm",
            "--format=csv,noheader,nounits",
        ]
    ).decode()
    cur, mx = (int(v) for v in out.strip().splitlines()[0].split(","))
    return cur, mx


def main() -> None:
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}")
    clocks = _sm_clock_mhz()
    if clocks is not None:
        cur, mx = clocks
        note = "" if cur >= int(0.95 * mx) else "  (clocked DOWN -> TF/s depressed)"
        print(f"SM clock: {cur}/{mx} MHz{note}")
    print(f"Shape M={M} K={K} N={N}, fp8e4m3 -> bf16, rowwise scales")

    x, y, scale_a, scale_b, scale_a2d = _make_inputs()

    bound = scaled_mm.bind((x, y, scale_a2d, scale_b))
    bound.set_config(BEST_CONFIG)
    helion_fn = bound.compile_config(BEST_CONFIG)

    out = helion_fn(x, y, scale_a2d, scale_b)
    torch.cuda.synchronize()
    ref = (x.float() @ y.float()) * scale_a2d.float() * scale_b.float()
    err = (out.float() - ref).abs().max().item()
    rel = ((out.float() - ref).abs() / (ref.abs() + 1e-6)).max().item()
    nan = int(out.float().isnan().sum().item())
    print(f"correctness: max_abs_err={err:.4f} max_rel_err={rel:.4f} nan={nan}")

    # Thermal warmup.
    warm = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(60):
        warm @ warm
    torch.cuda.synchronize()

    def _scaled_mm_aten() -> torch.Tensor:
        return torch._scaled_mm(
            x,
            y,
            scale_a=scale_a[:, None],
            scale_b=scale_b[None, :],
            out_dtype=torch.bfloat16,
        )

    helion_us = _graph_time_us(lambda: helion_fn(x, y, scale_a2d, scale_b))
    aten_us = _graph_time_us(_scaled_mm_aten)

    def tflops(us: float) -> float:
        return FLOP / (us * 1e-6) / 1e12

    print("\n50 kernels/graph, median of 100 replays")
    print(
        f"  helion bm=128 2-CTA   {helion_us:7.2f} us  ({tflops(helion_us):6.0f} TF/s)"
    )
    print(f"  aten _scaled_mm       {aten_us:7.2f} us  ({tflops(aten_us):6.0f} TF/s)")
    print(f"\n  speedup vs aten: {aten_us / helion_us:.2f}x")
    print(
        "\n  reference (max SM clock, FINDINGS_512_SHAPE.md):\n"
        "    helion bm=128 2-CTA      8.37 us  ( 1540 TF/s)\n"
        "    cute standalone 128x128  8.48 us  ( 1518 TF/s)\n"
        "    aten _scaled_mm         10.65 us  ( 1209 TF/s)"
    )


if __name__ == "__main__":
    main()
