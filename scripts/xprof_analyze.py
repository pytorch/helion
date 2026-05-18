"""xprof trace analyzer — programmatic per-lane utilization without TensorBoard.

Two analysis paths:

1. JSON sample-based (default): parses `trace.json.gz`'s rolling `% util` lane
   samples. Approximate; doesn't match TensorBoard's bar-chart exactly.

2. Authoritative via the openxla/xprof Python API (`--from-xplane`): pulls
   the `utilization_viewer` tool data from the `.xplane.pb` file. Returns
   the exact `achieved / peak` values TensorBoard renders (matches the bar
   chart used in the "Unit Utilization" view).



Reads a `jax.profiler.start_trace`/`stop_trace` output directory captured with
the LLO-debug-info flags:

    LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true \\
                      --xla_xprof_register_llo_debug_info=true"

The trace contains per-lane (`MXU`, `Vector Load`, `Vector Store`, `Vector ALU`,
`Vector EUP`, `Vector Fills`, `Scalar ALU`) utilization samples with a `% util`
percentage, plus per-bundle markers (`bundle.<N>`). This script extracts the
per-kernel-call timeline and reports:

  - Per-lane average utilization across the kernel's body
  - MXU activity gaps (indicator for un-hidden phi-stall / dependency waits)
  - When MXU is idle, which other lanes are active
  - Side-by-side comparison of two traces

CLI:
    python scripts/xprof_analyze.py <trace_dir>
    python scripts/xprof_analyze.py <trace_dir> --kernel custom_kernel.2
    python scripts/xprof_analyze.py <trace_dir> --call 10 --gap-ns 500
    python scripts/xprof_analyze.py <trace_a> --compare <trace_b>

Library:
    from scripts.xprof_analyze import load_trace, analyze_call, LANE_NAMES
    events = load_trace("/tmp/xprof_attn")
    summary = analyze_call(events, kernel_name="custom_kernel.2", call_index=10)
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# These names match what xprof emits with the LLO debug-info flags enabled.
LANE_NAMES = (
    "MXU",
    "Vector Load",
    "Vector Store",
    "Vector ALU",
    "Vector EUP",
    "Vector Fills",
    "Scalar ALU",
)


@dataclass
class LaneStats:
    """Per-lane stats within a [t_start, t_end) window.

    The headline number is `busy_pct_of_window` — the time-weighted fraction of
    the window during which the lane was active. This is what TensorBoard's
    "Avg MXU Busy" / "Vector ALUs" / etc. percentages also report. The earlier
    "avg_util_pct" metric (mean of `% util` across samples) is misleading
    because samples are only emitted at moments of activity, so it averages
    over a subset of time, not the full window.
    """

    n_samples: int
    coverage_us: float  # span between first and last sample
    sum_dur_us: float  # sum of sample dur fields (sample window total)
    busy_us: float  # busy-time integral (see formula below)
    busy_pct_of_window: float  # busy_us / window_us
    busy_pct_of_samples: float  # busy_us / sum_dur
    n_zero_util_samples: int  # samples where util ~= 0


@dataclass
class CallSummary:
    kernel_name: str
    call_index: int
    t_start_us: float
    t_end_us: float
    duration_us: float
    lanes: dict[str, LaneStats]
    mxu_gaps: list[tuple[float, float]] = field(default_factory=list)
    # Each gap: (offset_from_call_start_us, gap_duration_us)


def load_trace(trace_dir: str | Path) -> list[dict[str, Any]]:
    """Load all `traceEvents` from the .trace.json.gz under trace_dir."""
    gz_paths = subprocess.check_output(
        ["find", str(trace_dir), "-name", "*.trace.json.gz"]
    ).decode().split()
    if not gz_paths:
        raise FileNotFoundError(f"No .trace.json.gz under {trace_dir}")
    with gzip.open(gz_paths[0], "rt") as f:
        d = json.load(f)
    return d["traceEvents"]


def find_kernel_calls(
    events: list[dict[str, Any]], kernel_name: str
) -> list[dict[str, Any]]:
    """Return sorted list of complete events matching kernel_name (with `dur`)."""
    calls = [e for e in events if e.get("name") == kernel_name and "dur" in e]
    return sorted(calls, key=lambda e: e["ts"])


def _safe_util(e: dict[str, Any]) -> float:
    try:
        return float(e.get("args", {}).get("% util", "0"))
    except (TypeError, ValueError):
        return 0.0


def lane_stats_in_window(
    events: list[dict[str, Any]], lane_name: str, t_start: float, t_end: float
) -> LaneStats:
    """Compute per-lane stats for samples within [t_start, t_end).

    ⚠ KNOWN BUG: `busy_pct_of_window` does NOT match TensorBoard's "Avg <Lane>
    Busy" — verified empirically on the PR #2368 attention trace where this
    script reported MXU 83% but TensorBoard reported 19%. The formula
    `sum(util/100 × dur)` over-counts because xprof sample `dur` windows can
    overlap or cover non-active intervals, and `% util` is a within-window
    fraction whose composition with `dur` doesn't directly give wall-clock
    busy time. Use TensorBoard's Op Profile / "Tensor Node Unit Utilization"
    view as ground truth until this is fixed. The numbers here are useful
    only for relative comparison between lanes within the same trace.
    """
    samples = [
        e for e in events
        if e.get("name") == lane_name and t_start <= e.get("ts", -1) < t_end
    ]
    window_us = max(1e-9, t_end - t_start)
    if not samples:
        return LaneStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    durs = [e.get("dur", 0.0) for e in samples]
    utils = [_safe_util(e) for e in samples]
    ts = [e["ts"] for e in samples]
    sum_dur = sum(durs)
    coverage = max(ts) + durs[ts.index(max(ts))] - min(ts) if len(ts) > 1 else durs[0]
    busy = sum((u / 100.0) * d for u, d in zip(utils, durs))
    n_zero = sum(1 for u in utils if u < 0.5)
    return LaneStats(
        n_samples=len(samples),
        coverage_us=coverage,
        sum_dur_us=sum_dur,
        busy_us=busy,
        busy_pct_of_window=100.0 * busy / window_us,
        busy_pct_of_samples=100.0 * busy / max(1e-9, sum_dur),
        n_zero_util_samples=n_zero,
    )


def detect_lane_gaps(
    events: list[dict[str, Any]],
    lane_name: str,
    t_start: float,
    t_end: float,
    gap_threshold_us: float = 0.5,
) -> list[tuple[float, float]]:
    """Find inter-sample gaps > threshold within [t_start, t_end).

    Returns list of (absolute_gap_start_us, gap_duration_us).
    """
    samples = sorted(
        (e for e in events
         if e.get("name") == lane_name and t_start <= e.get("ts", -1) < t_end),
        key=lambda e: e["ts"],
    )
    gaps = []
    for i in range(1, len(samples)):
        prev_end = samples[i - 1]["ts"] + samples[i - 1].get("dur", 0.0)
        curr_start = samples[i]["ts"]
        gap = curr_start - prev_end
        if gap > gap_threshold_us:
            gaps.append((prev_end, gap))
    return gaps


def lane_activity_in_window(
    events: list[dict[str, Any]], lane_name: str, t_start: float, t_end: float
) -> float:
    """Return total busy-µs of `lane_name` within [t_start, t_end)."""
    samples = [
        e for e in events
        if e.get("name") == lane_name and e.get("ts", -1) + e.get("dur", 0.0) > t_start
        and e.get("ts", -1) < t_end
    ]
    busy = 0.0
    for e in samples:
        ts = e["ts"]
        dur = e.get("dur", 0.0)
        overlap = max(0.0, min(ts + dur, t_end) - max(ts, t_start))
        busy += overlap * (_safe_util(e) / 100.0)
    return busy


def analyze_call(
    events: list[dict[str, Any]],
    kernel_name: str,
    call_index: int,
    gap_threshold_us: float = 0.5,
) -> CallSummary:
    """Analyze one kernel call by index."""
    calls = find_kernel_calls(events, kernel_name)
    if not calls:
        raise ValueError(f"No kernel '{kernel_name}' found in trace")
    if call_index < 0 or call_index >= len(calls):
        raise IndexError(
            f"call_index {call_index} out of range (have {len(calls)} calls)"
        )
    target = calls[call_index]
    t0, dur = target["ts"], target["dur"]
    t1 = t0 + dur
    lanes = {
        lane: lane_stats_in_window(events, lane, t0, t1) for lane in LANE_NAMES
    }
    mxu_gaps = [
        (gap_start - t0, gap_dur)
        for gap_start, gap_dur in detect_lane_gaps(
            events, "MXU", t0, t1, gap_threshold_us
        )
    ]
    return CallSummary(
        kernel_name=kernel_name,
        call_index=call_index,
        t_start_us=t0,
        t_end_us=t1,
        duration_us=dur,
        lanes=lanes,
        mxu_gaps=mxu_gaps,
    )


def print_summary(summary: CallSummary, gap_threshold_us: float) -> None:
    """Human-readable rendering of a CallSummary."""
    print(
        f"=== {summary.kernel_name}  call #{summary.call_index}  "
        f"[{summary.t_start_us:.2f}–{summary.t_end_us:.2f}us  "
        f"= {summary.duration_us:.2f}us] ==="
    )
    print(
        f"  {'lane':<14}  {'n_samp':>7}  {'busy_us':>9}  {'%window':>9}  {'%when_active':>14}"
    )
    print(f"  ({'':<14}  {'':<7}  {'':<9}  ← matches TB '<Lane> Busy')")
    for lane, st in summary.lanes.items():
        if st.n_samples == 0:
            print(f"  {lane:<14}  {'(no samples)':>7}")
            continue
        print(
            f"  {lane:<14}  {st.n_samples:>7}  {st.busy_us:>9.3f}  "
            f"{st.busy_pct_of_window:>8.2f}%  {st.busy_pct_of_samples:>13.2f}%"
        )

    print()
    if summary.mxu_gaps:
        total_gap = sum(g for _, g in summary.mxu_gaps)
        print(
            f"MXU gaps > {gap_threshold_us * 1000:.0f} ns: "
            f"{len(summary.mxu_gaps)}  "
            f"(total {total_gap:.2f}us, {100 * total_gap / summary.duration_us:.1f}% of call)"
        )
        max_show = min(10, len(summary.mxu_gaps))
        for offset, gap in summary.mxu_gaps[:max_show]:
            print(f"  @{offset:>6.2f}us  gap={gap:>6.3f}us")
        if len(summary.mxu_gaps) > max_show:
            print(f"  ... and {len(summary.mxu_gaps) - max_show} more")
    else:
        print(f"MXU is continuous (no gaps > {gap_threshold_us * 1000:.0f} ns)")


def print_comparison(
    summary_a: CallSummary, label_a: str, summary_b: CallSummary, label_b: str
) -> None:
    """Side-by-side comparison of two CallSummary objects."""
    print(
        f"=== Comparison: {label_a} vs {label_b} ==="
    )
    print(
        f"  duration:  {label_a}={summary_a.duration_us:.2f}us  |  "
        f"{label_b}={summary_b.duration_us:.2f}us"
    )
    print(
        f"  MXU gaps:  {label_a}={len(summary_a.mxu_gaps)}  |  "
        f"{label_b}={len(summary_b.mxu_gaps)}"
    )
    print()
    print(f"  {'lane':<14}  {label_a:>15}  {label_b:>15}")
    for lane in LANE_NAMES:
        sa = summary_a.lanes.get(lane)
        sb = summary_b.lanes.get(lane)
        if sa is None or sb is None or (sa.n_samples == 0 and sb.n_samples == 0):
            continue
        ua = f"{sa.busy_pct_of_window:.2f}%" if sa.n_samples else "—"
        ub = f"{sb.busy_pct_of_window:.2f}%" if sb.n_samples else "—"
        print(f"  {lane:<14}  {ua:>15}  {ub:>15}")


def utilization_from_xplane(trace_dir: str | Path) -> dict[str, dict[str, float]]:
    """Use openxla/xprof's API to extract authoritative utilization values.

    Returns: {lane_name: {"achieved": float, "peak": float, "pct": float, "unit": str}}

    Matches TensorBoard's "Unit Utilization" view exactly. Requires
    `xprof.convert.raw_to_tool_data` (ships with `pip install xprof`).
    """
    xp_paths = subprocess.check_output(
        ["find", str(trace_dir), "-name", "*.xplane.pb"]
    ).decode().split()
    if not xp_paths:
        raise FileNotFoundError(f"No .xplane.pb under {trace_dir}")

    # pyrefly: ignore[missing-import]  # xprof is pod-only, not local
    from xprof.convert.raw_to_tool_data import xspace_to_tool_data

    result = xspace_to_tool_data([xp_paths[0]], "utilization_viewer", {})
    payload = result[0] if isinstance(result, tuple) else result
    if isinstance(payload, bytes):
        payload = payload.decode()
    d = json.loads(payload)
    cols = [c["id"] for c in d["cols"]]
    name_i = cols.index("name")
    achieved_i = cols.index("achieved")
    peak_i = cols.index("peak")
    unit_i = cols.index("unit")
    out: dict[str, dict[str, float]] = {}
    for row in d["rows"]:
        cells = [c["v"] for c in row["c"]]
        name = cells[name_i]
        achieved = float(cells[achieved_i])
        peak = float(cells[peak_i])
        unit = cells[unit_i]
        pct = 100.0 * achieved / peak if peak > 0 else 0.0
        # Merge entries for the same name across multiple Tensor Nodes
        if name in out:
            out[name]["achieved"] += achieved
            out[name]["peak"] += peak
            out[name]["pct"] = 100.0 * out[name]["achieved"] / max(out[name]["peak"], 1)
        else:
            out[name] = {"achieved": achieved, "peak": peak, "pct": pct, "unit": unit}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_dir", help="jax.profiler trace directory")
    p.add_argument(
        "--kernel",
        default="custom_kernel.2",
        help="Pallas kernel event name (default: custom_kernel.2)",
    )
    p.add_argument(
        "--call",
        type=int,
        default=None,
        help="Which call index to analyze (default: middle of all calls)",
    )
    p.add_argument(
        "--gap-ns",
        type=float,
        default=500.0,
        help="MXU gap-detection threshold in nanoseconds (default: 500)",
    )
    p.add_argument(
        "--compare",
        help="Second trace dir to compare against (uses same --kernel and --call)",
    )
    p.add_argument(
        "--list-kernels",
        action="store_true",
        help="List all kernel-call event names and their counts, then exit",
    )
    p.add_argument(
        "--json-only",
        action="store_true",
        help="Force the JSON sample-based path even if xplane.pb is present "
        "(default: auto-use xplane.pb when available for TB-matching numbers)",
    )
    args = p.parse_args()

    # Prefer authoritative xplane.pb path when available
    xp_paths = subprocess.check_output(
        ["find", str(args.trace_dir), "-name", "*.xplane.pb"]
    ).decode().split()
    if xp_paths and not args.json_only:
        try:
            util = utilization_from_xplane(args.trace_dir)
            print(f"=== Authoritative utilization (from xplane.pb) ===")
            print(f"  {'metric':<32}  {'achieved':>14}  {'peak':>14}  {'unit':>12}  {'%':>7}")
            for name, m in util.items():
                print(
                    f"  {name:<32}  {m['achieved']:>14,.0f}  {m['peak']:>14,.0f}  "
                    f"{m['unit']:>12}  {m['pct']:>6.2f}%"
                )
            return
        except ImportError:
            print(
                "ℹ xplane.pb present but xprof Python package not available; "
                "falling back to JSON path. Install `xprof` for TB-matching numbers.",
                file=sys.stderr,
            )

    events = load_trace(args.trace_dir)

    if args.list_kernels:
        counts: dict[str, int] = defaultdict(int)
        for e in events:
            n = e.get("name", "")
            if "custom_kernel" in n or "tpu_custom_call" in n or n.startswith("matmul"):
                if "dur" in e:
                    counts[n] += 1
        print(f"Kernel-call events in {args.trace_dir}:")
        for n, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {c:>5}  {n}")
        return

    calls = find_kernel_calls(events, args.kernel)
    if not calls:
        sys.exit(
            f"No kernel '{args.kernel}' found. Try --list-kernels to see available names."
        )

    print(
        f"Trace: {args.trace_dir}  ({len(calls)} '{args.kernel}' calls, "
        f"mean dur {sum(c['dur'] for c in calls) / len(calls):.2f} us)\n"
    )

    call_index = args.call if args.call is not None else len(calls) // 2
    gap_thresh = args.gap_ns / 1000.0
    summary_a = analyze_call(events, args.kernel, call_index, gap_thresh)
    print_summary(summary_a, gap_thresh)

    if args.compare:
        events_b = load_trace(args.compare)
        calls_b = find_kernel_calls(events_b, args.kernel)
        call_index_b = args.call if args.call is not None else len(calls_b) // 2
        summary_b = analyze_call(events_b, args.kernel, call_index_b, gap_thresh)
        print()
        print_summary(summary_b, gap_thresh)
        print()
        print_comparison(
            summary_a, Path(args.trace_dir).name,
            summary_b, Path(args.compare).name,
        )


if __name__ == "__main__":
    main()
