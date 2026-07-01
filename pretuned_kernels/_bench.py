"""Shared benchmark loop for pretuned kernel ``benchmark()`` functions.

Times the Helion kernel against one or more baselines (under CUDA graphs or
plain ``do_bench``), prints a per-shape table, and returns a metrics dict that
``pretuned_kernels/run.py`` records directly (no stdout parsing):

  {"helion_wins":.., "total":.., "geomean":.., "best_speedup":..,   # vs best baseline
   "baselines": {"<name>": {"wins":..,"total":..,"geomean":..,"best_speedup":..}}}

where "baselines" is helion's speedup over *each* baseline (powers the dashboard
dropdown). Kernels add their directory's parent to ``sys.path`` and
``import _bench`` so this works both under ``python pretuned_kernels/<k>/<k>.py``
and via run.py's importlib loader.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import triton.testing as tt

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable


def geomean(values: Iterable[float]) -> float:
    pos = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in pos) / max(len(pos), 1))


def bench_cudagraph(call: Callable[[], object], rep: int = 100) -> float:
    """Median CUDA-graph latency (ms), clearing the L2 cache each iteration.

    Uses tritonbench's cudagraph timer, which zeroes the L2 cache before every
    replay and subtracts the clear cost. ``triton.testing.do_bench_cudagraph``
    does *not* clear L2, so a graph replaying the same inputs reuses cached data
    and under-reports latency. tritonbench is required (no fallback) so pretuned
    numbers are always measured with cache clearing -- install it before running
    a cudagraph kernel's ``main()`` (the nightly benchmark workflow does).
    """
    from tritonbench.components.do_bench.run import _do_bench_cudagraph_with_cache_clear

    return _do_bench_cudagraph_with_cache_clear(call, rep=rep, return_mode="median")


def _bench(
    call: Callable[[], object], use_cudagraph: bool, warmup: int, rep: int
) -> float:
    if use_cudagraph:
        return bench_cudagraph(call, rep=rep)
    return tt.do_bench(call, warmup=warmup, rep=rep, return_mode="median")


def run_sweep(
    shapes: Iterable[object],
    make_calls: Callable[[object], tuple],
    *,
    use_cudagraph: bool,
    shape_header: str,
    warmup: int = 25,
    rep: int = 100,
    verbose: bool = True,
) -> dict:
    """Benchmark helion vs baselines over ``shapes``; return metrics (print if verbose).

    ``make_calls(shape)`` returns ``(helion_call, [(baseline_name, baseline_call)],
    shape_cells)`` where the calls are zero-arg closures over freshly built inputs
    and ``shape_cells`` is the preformatted leading column(s) for the table row.
    The metrics dict is always returned; the per-shape table is printed only when
    ``verbose``.
    """

    def _p(*args: object) -> None:
        if verbose:
            print(*args)

    _p(f"GPU: {torch.cuda.get_device_name()}")
    speedups_by_base: dict[str, list[float]] = {}
    best_speedups: list[float] = []
    helion_wins = 0
    best_speedup = 0.0
    header_printed = False
    for shape in shapes:
        helion_call, baseline_calls, shape_cells = make_calls(shape)
        names = [n for n, _ in baseline_calls]
        if not header_printed:
            for n in names:
                speedups_by_base[n] = []
            base_hdr = "  ".join(f"{n + ' (us)':>13s}" for n in names)
            _p(f"{shape_header}  {'helion (us)':>12s}  {base_hdr}  {'speedup':>8s}")
            header_printed = True

        helion_call()  # warmup / compile
        ms_helion = _bench(helion_call, use_cudagraph, warmup, rep)
        base_ms: dict[str, float] = {}
        for name, call in baseline_calls:
            base_ms[name] = _bench(call, use_cudagraph, warmup, rep)
            speedups_by_base[name].append(
                base_ms[name] / ms_helion if ms_helion > 0 else float("nan")
            )
        best_name = min(base_ms, key=base_ms.get)
        speedup = base_ms[best_name] / ms_helion if ms_helion > 0 else float("nan")
        best_speedups.append(speedup)
        if speedup > 1.0:
            helion_wins += 1
        best_speedup = max(best_speedup, speedup)
        base_cols = "  ".join(f"{base_ms[n] * 1000:>13.2f}" for n in names)
        _p(
            f"{shape_cells}  {ms_helion * 1000:>12.2f}  {base_cols}  "
            f"{speedup:>7.2f}x  (vs {best_name})"
        )

    names = list(speedups_by_base)
    per_baseline = {
        n: {
            "wins": sum(1 for s in speedups_by_base[n] if s > 1.0),
            "total": len(speedups_by_base[n]),
            "geomean": round(geomean(speedups_by_base[n]), 4),
            "best_speedup": round(max(speedups_by_base[n], default=0.0), 4),
        }
        for n in names
    }
    for n in names:
        m = per_baseline[n]
        _p(
            f"vs {n}: wins={m['wins']}/{m['total']} "
            f"geomean={m['geomean']:.3f}x best={m['best_speedup']:.2f}x"
        )
    total = len(best_speedups)
    gm = geomean(best_speedups)
    _p(
        f"\nHelion faster on {helion_wins}/{total} shapes vs the best baseline; "
        f"geomean speedup {gm:.3f}x; best speedup {best_speedup:.2f}x."
    )
    # Metrics are helion vs the best (fastest) baseline per shape, plus the
    # per-baseline breakdown; returned to the caller (pretuned_kernels/run.py).
    return {
        "helion_wins": helion_wins,
        "total": total,
        "geomean": round(gm, 4),
        "best_speedup": round(best_speedup, 4),
        "baselines": per_baseline,
    }
