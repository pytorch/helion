"""helion.tools.profiler — Integrated profiling utility for Helion GPU kernels.

Provides a :class:`KernelProfiler` that compares a ``@helion.kernel``-decorated
function against a PyTorch reference.  For each pair it reports:

* Achieved vs. peak memory bandwidth (GB/s and % of theoretical maximum).
* Compute utilisation (TFLOP/s, optional).
* A roofline-based bottleneck classification (memory-bound / compute-bound /
  latency-bound).
* The generated Triton code side-by-side with the Helion source.
* Mean / std / min / max latency for both implementations.

Timings use CUDA events inside a :func:`gc.disable` / :func:`gc.enable` bracket
and are preceded by a ``torch.cuda._sleep`` spin to stabilise GPU clocks before
measurement starts (the same technique used to reduce timing variance in Triton's
own benchmarking suite and referenced in helion issue #1187).

Quick-start::

    from helion.tools.profiler import profile
    import helion
    import helion.language as hl
    import torch

    @helion.kernel
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.shape):
            out[tile] = x[tile] + y[tile]
        return out

    def add_ref(x, y):
        return x + y

    profile(
        helion_fn=add,
        reference_fn=add_ref,
        input_generator=lambda: (
            torch.randn(4096, 4096, device="cuda"),
            torch.randn(4096, 4096, device="cuda"),
        ),
        kernel_name="add",
        bytes_read_fn=lambda a: sum(t.numel() * t.element_size() for t in a),
        bytes_written_fn=lambda a: a[0].numel() * a[0].element_size(),
    )

CLI usage::

    helion-profile \\
        --module examples.add \\
        --helion add \\
        --reference add_ref \\
        --input_generator make_inputs
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import io
import os
import statistics
import sys
from typing import TYPE_CHECKING
from typing import Callable

import torch

if TYPE_CHECKING:
    pass

__all__ = [
    "KernelProfiler",
    "profile",
]

# ---------------------------------------------------------------------------
# GPU hardware introspection
# ---------------------------------------------------------------------------

_WARMUP_SLEEP_CYCLES = 100_000_000  # ~50-70 ms on modern NVIDIA GPU


def _gpu_specs(device: int = 0) -> dict[str, object]:
    """Return key hardware properties for *device*."""
    p = torch.cuda.get_device_properties(device)
    # theoretical peak memory bandwidth: clock × 2 (DDR) × bus_width / 8
    peak_bw_GBps = (p.memory_clock_rate * 2 * p.memory_bus_width / 8) / 1e6
    # rough FP32 peak: SMs × threads/SM × 2 (FMA) × GHz
    peak_fp32_tflops = (
        p.multi_processor_count
        * p.max_threads_per_multi_processor
        * 2
        * (p.clock_rate / 1e6)
        / 1e3
    )
    return {
        "name": p.name,
        "sm_count": p.multi_processor_count,
        "smem_per_block_kb": p.shared_memory_per_block // 1024,
        "mem_clock_mhz": p.memory_clock_rate / 1e3,
        "mem_bus_bits": p.memory_bus_width,
        "peak_bw_GBps": peak_bw_GBps,
        "peak_fp32_tflops": peak_fp32_tflops,
        "total_mem_GB": p.total_memory / 1e9,
    }


# ---------------------------------------------------------------------------
# CUDA-event timing with GC isolation and clock stabilisation
# ---------------------------------------------------------------------------

def _time_kernel(
    fn: Callable[..., object],
    args: tuple[object, ...],
    *,
    warmup: int = 5,
    iterations: int = 50,
    device: int = 0,
) -> dict[str, float]:
    """Time *fn(*args)* on GPU using CUDA events.

    * Disables the Python GC for the duration of the hot loop to prevent
      GC pauses from inflating measured latency (see helion issue #1187).
    * Inserts a ``torch.cuda._sleep`` spin before measurement begins to
      bring the GPU to a stable clock frequency.
    """
    # warmup pass
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize(device)

    # clock stabilisation — spin the GPU so the driver boosts clocks before we
    # start recording.  torch.cuda._sleep is in CUDA cycles.
    if torch.cuda.is_available():
        torch.cuda._sleep(_WARMUP_SLEEP_CYCLES)
        torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings: list[float] = []

    gc.disable()
    try:
        for _ in range(iterations):
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
    finally:
        gc.enable()

    return {
        "mean_ms": statistics.mean(timings),
        "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


# ---------------------------------------------------------------------------
# Triton-code capture
# ---------------------------------------------------------------------------

def _capture_triton_code(
    helion_fn: Callable[..., object],
    args: tuple[object, ...],
) -> str:
    """Invoke *helion_fn* with ``HELION_PRINT_OUTPUT_CODE=1`` and return the
    generated Triton code that Helion writes to *stderr*."""
    prev = os.environ.get("HELION_PRINT_OUTPUT_CODE")
    os.environ["HELION_PRINT_OUTPUT_CODE"] = "1"
    buf = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = buf
    try:
        helion_fn(*args)
    except Exception:  # noqa: BLE001
        pass
    finally:
        sys.stderr = old_stderr
        if prev is None:
            os.environ.pop("HELION_PRINT_OUTPUT_CODE", None)
        else:
            os.environ["HELION_PRINT_OUTPUT_CODE"] = prev

    captured = buf.getvalue().strip()
    return captured or "(run with HELION_PRINT_OUTPUT_CODE=1 to see generated code)"


# ---------------------------------------------------------------------------
# Bottleneck classification (roofline)
# ---------------------------------------------------------------------------

def _classify(bw_util_pct: float, flop_util_pct: float | None) -> str:
    """Simple roofline classification.

    * > 60 % of peak memory bandwidth  → Memory-bound
    * > 60 % of peak FLOP throughput    → Compute-bound
    * otherwise                         → Latency-bound / under-utilised
    """
    if bw_util_pct > 60.0:
        return f"Memory-bound  ({bw_util_pct:.1f}% of peak BW)"
    if flop_util_pct is not None and flop_util_pct > 60.0:
        return f"Compute-bound  ({flop_util_pct:.1f}% of peak FLOP throughput)"
    hint = f"BW util {bw_util_pct:.1f}%"
    if flop_util_pct is not None:
        hint += f", FLOP util {flop_util_pct:.1f}%"
    return f"Latency-bound / under-utilised  ({hint})"


# ---------------------------------------------------------------------------
# KernelProfiler
# ---------------------------------------------------------------------------

class KernelProfiler:
    """Compare a ``@helion.kernel`` against a PyTorch reference implementation.

    Parameters
    ----------
    helion_fn:
        The ``@helion.kernel``-decorated function to profile.
    reference_fn:
        Equivalent PyTorch / eager reference.
    input_generator:
        Callable returning a ``tuple`` of input tensors.  Called once; the same
        args are reused for all measurements.
    kernel_name:
        Display label.  Defaults to ``helion_fn.__name__``.
    bytes_read_fn:
        ``(args) -> int`` — bytes read by the kernel.  Used to compute achieved
        memory bandwidth.
    bytes_written_fn:
        ``(args) -> int`` — bytes written by the kernel.
    flops_fn:
        ``(args) -> int`` — total FLOPs performed (optional; enables FLOP
        utilisation reporting and compute-bound classification).
    warmup:
        Number of un-timed warm-up iterations.
    iterations:
        Number of timed iterations.
    device:
        CUDA device index.
    """

    def __init__(
        self,
        helion_fn: Callable[..., object],
        reference_fn: Callable[..., object],
        input_generator: Callable[[], tuple[object, ...]],
        *,
        kernel_name: str = "",
        bytes_read_fn: Callable[[tuple[object, ...]], int] | None = None,
        bytes_written_fn: Callable[[tuple[object, ...]], int] | None = None,
        flops_fn: Callable[[tuple[object, ...]], int] | None = None,
        warmup: int = 5,
        iterations: int = 50,
        device: int = 0,
    ) -> None:
        self.helion_fn = helion_fn
        self.reference_fn = reference_fn
        self.input_generator = input_generator
        self.kernel_name = kernel_name or getattr(helion_fn, "__name__", "kernel")
        self.bytes_read_fn = bytes_read_fn
        self.bytes_written_fn = bytes_written_fn
        self.flops_fn = flops_fn
        self.warmup = warmup
        self.iterations = iterations
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, *, capture_triton: bool = True) -> dict[str, object]:
        """Execute the profiling run and return a results dict.

        Returns a dict with keys:
        ``gpu``, ``ref_timing``, ``helion_timing``,
        ``speedup``, ``bw_util_pct``, ``flop_util_pct``,
        ``classification``, ``helion_source``, ``triton_code``.
        """
        if not torch.cuda.is_available():
            msg = "CUDA not available — KernelProfiler requires a GPU."
            raise RuntimeError(msg)

        args = self.input_generator()
        gpu = _gpu_specs(self.device)

        ref_t = _time_kernel(
            self.reference_fn, args,
            warmup=self.warmup, iterations=self.iterations, device=self.device,
        )
        hl_t = _time_kernel(
            self.helion_fn, args,
            warmup=self.warmup, iterations=self.iterations, device=self.device,
        )

        speedup = ref_t["mean_ms"] / hl_t["mean_ms"] if hl_t["mean_ms"] > 0 else float("inf")

        bytes_transferred = 0
        if self.bytes_read_fn:
            bytes_transferred += self.bytes_read_fn(args)
        if self.bytes_written_fn:
            bytes_transferred += self.bytes_written_fn(args)

        achieved_bw = (bytes_transferred / 1e9) / (hl_t["mean_ms"] / 1e3) if hl_t["mean_ms"] > 0 else 0.0
        peak_bw: float = gpu["peak_bw_GBps"]  # type: ignore[assignment]
        bw_util_pct = (achieved_bw / peak_bw * 100) if peak_bw > 0 else 0.0

        flop_util_pct: float | None = None
        if self.flops_fn is not None and hl_t["mean_ms"] > 0:
            achieved_tflops = (self.flops_fn(args) / 1e12) / (hl_t["mean_ms"] / 1e3)
            peak_fp32: float = gpu["peak_fp32_tflops"]  # type: ignore[assignment]
            flop_util_pct = (achieved_tflops / peak_fp32 * 100) if peak_fp32 > 0 else 0.0

        classification = _classify(bw_util_pct, flop_util_pct)

        helion_source = ""
        try:
            helion_source = inspect.getsource(self.helion_fn)
        except (OSError, TypeError):
            helion_source = "(source unavailable)"

        triton_code = ""
        if capture_triton:
            triton_code = _capture_triton_code(self.helion_fn, args)

        return {
            "gpu": gpu,
            "ref_timing": ref_t,
            "helion_timing": hl_t,
            "speedup": speedup,
            "achieved_bw_GBps": achieved_bw,
            "bw_util_pct": bw_util_pct,
            "flop_util_pct": flop_util_pct,
            "classification": classification,
            "helion_source": helion_source,
            "triton_code": triton_code,
        }

    def print_report(self, results: dict[str, object] | None = None) -> None:
        """Print a human-readable report.  Calls :meth:`run` if *results* is
        ``None``."""
        if results is None:
            results = self.run()
        _print_report(self.kernel_name, results)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _print_report(kernel_name: str, r: dict[str, object]) -> None:
    gpu: dict[str, object] = r["gpu"]  # type: ignore[assignment]
    ref_t: dict[str, float] = r["ref_timing"]  # type: ignore[assignment]
    hl_t: dict[str, float] = r["helion_timing"]  # type: ignore[assignment]

    W = 58
    sep = "=" * W
    thin = "-" * W

    def row(label: str, value: str) -> None:
        print(f"  {label:<28}{value}")

    print(sep)
    print("  HELION KERNEL PROFILE".center(W))
    print(sep)
    print()
    row("Kernel:", kernel_name)
    row("GPU:", str(gpu["name"]))
    row("SMs:", f"{gpu['sm_count']}   smem/block: {gpu['smem_per_block_kb']} KB")
    row("VRAM:", f"{gpu['total_mem_GB']:.1f} GB   "
        f"bus: {gpu['mem_bus_bits']} bits @ {gpu['mem_clock_mhz']:.0f} MHz")
    print()
    print(thin)
    print("  LATENCY".center(W))
    print(thin)
    row("Reference mean:", f"{ref_t['mean_ms']:>8.3f} ms  ±{ref_t['std_ms']:.3f}")
    row("Helion mean:", f"{hl_t['mean_ms']:>8.3f} ms  ±{hl_t['std_ms']:.3f}")
    row("Speedup:", f"{r['speedup']:>8.2f}x")
    print()
    print(thin)
    print("  MEMORY BANDWIDTH".center(W))
    print(thin)
    row("Peak (theoretical):", f"{gpu['peak_bw_GBps']:>8.1f} GB/s")
    row("Achieved:", f"{r['achieved_bw_GBps']:>8.1f} GB/s")
    row("Utilisation:", f"{r['bw_util_pct']:>8.1f}%")
    if r["flop_util_pct"] is not None:
        row("FLOP utilisation:", f"{r['flop_util_pct']:>8.1f}%  "
            f"(peak {gpu['peak_fp32_tflops']:.1f} TFLOPS FP32)")
    print()
    print(thin)
    print("  BOTTLENECK".center(W))
    print(thin)
    print(f"  {r['classification']}")
    print()

    helion_source: str = r["helion_source"]  # type: ignore[assignment]
    if helion_source:
        print(sep)
        print("  HELION SOURCE".center(W))
        print(sep)
        print(helion_source)
        print()

    triton_code: str = r["triton_code"]  # type: ignore[assignment]
    if triton_code:
        print(sep)
        print("  GENERATED TRITON CODE".center(W))
        print(sep)
        print(triton_code)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def profile(
    helion_fn: Callable[..., object],
    reference_fn: Callable[..., object],
    input_generator: Callable[[], tuple[object, ...]],
    *,
    kernel_name: str = "",
    bytes_read_fn: Callable[[tuple[object, ...]], int] | None = None,
    bytes_written_fn: Callable[[tuple[object, ...]], int] | None = None,
    flops_fn: Callable[[tuple[object, ...]], int] | None = None,
    warmup: int = 5,
    iterations: int = 50,
    device: int = 0,
    capture_triton: bool = True,
) -> None:
    """Profile a Helion kernel and print a report.

    Convenience wrapper around :class:`KernelProfiler`.  See the class
    documentation for parameter descriptions.
    """
    profiler = KernelProfiler(
        helion_fn=helion_fn,
        reference_fn=reference_fn,
        input_generator=input_generator,
        kernel_name=kernel_name,
        bytes_read_fn=bytes_read_fn,
        bytes_written_fn=bytes_written_fn,
        flops_fn=flops_fn,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    profiler.print_report(profiler.run(capture_triton=capture_triton))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Entry point for the ``helion-profile`` command."""
    parser = argparse.ArgumentParser(
        prog="helion-profile",
        description=(
            "Profile a @helion.kernel function vs a PyTorch reference and "
            "report latency, memory-bandwidth utilisation, and generated "
            "Triton code."
        ),
    )
    parser.add_argument(
        "--module",
        required=True,
        metavar="MODULE",
        help="Python module containing the functions (e.g. examples.softmax).",
    )
    parser.add_argument(
        "--helion",
        required=True,
        metavar="FUNC",
        help="Name of the @helion.kernel function.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        metavar="FUNC",
        help="Name of the PyTorch reference function.",
    )
    parser.add_argument(
        "--input-generator",
        required=True,
        dest="input_generator",
        metavar="FUNC",
        help="Name of a zero-argument callable returning input tensors.",
    )
    parser.add_argument("--kernel-name", dest="kernel_name", default="")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--no-triton-code",
        dest="capture_triton",
        action="store_false",
        help="Skip capturing generated Triton code.",
    )

    args = parser.parse_args()
    mod = importlib.import_module(args.module)

    def _get(name: str) -> Callable[..., object]:
        fn = getattr(mod, name, None)
        if fn is None:
            parser.error(f"'{name}' not found in module '{args.module}'")
        return fn  # type: ignore[return-value]

    profile(
        helion_fn=_get(args.helion),
        reference_fn=_get(args.reference),
        input_generator=_get(args.input_generator),
        kernel_name=args.kernel_name or args.helion,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
        capture_triton=args.capture_triton,
    )


if __name__ == "__main__":
    _cli()
