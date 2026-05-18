"""HLO sidecar — derive runtime trip counts for kernels the LLO can't reveal.

The roofline predictor refuses to predict when an LLO dump shows deeply
nested loops without static trip-count annotations (RPA, splash SWA,
emit_pipeline kernels with data-dependent bounds). This sidecar derives
the missing trip count in one of two ways:

1. **Static trace mode** (`inspect_pallas_loops`): trace the kernel
   through JAX, recurse into `pallas_call` bodies, and report all
   `while`/`scan`/`fori_loop` primitives. Often XLA has already lowered
   even shape-determinable loops to `while_loop`, so static extraction
   succeeds only for kernels with literal `scan(length=K, ...)`. For
   most production kernels the sidecar will report `dynamic_bound=True`
   — useful as a confirmation but not enough to fix the prediction.

2. **Back-solve mode** (`back_solve_inner_loop_iters`): given one
   measured runtime and an LLO dump, compute K such that the predictor's
   formula reproduces the measurement. This is the practical fix for
   dynamic-bound kernels: measure once, persist K in the database entry's
   `meta.yaml`, and future predictions of the same kernel + shape use
   the stored K directly.

Usage:
    # Mode 1 — static trace
    python scripts/hlo_sidecar.py trace \\
        --kernel-import 'my_pkg.my_kernel:my_kernel_fn' \\
        --runner my_runner.py:make_inputs \\
        --runner-args '1024,4096,4096,8'

    # Mode 2 — back-solve from one measurement
    python scripts/hlo_sidecar.py back-solve \\
        --llo-dir llo/rpa_decode_S32x2048_q8kv1d128_bf16 \\
        --inputs 'bf16:32x8x128,bf16:65536x1x128,bf16:65536x1x128' \\
        --outputs 'bf16:32x8x128' \\
        --measured-us 211.45
    # → prints: derived inner_loop_iters: 39 (writes to meta.yaml if --persist)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import sys


@dataclass
class LoopInfo:
    primitive: str  # "while", "scan", "fori_loop", "pallas_call"
    depth: int
    trip_count: int | None  # None if dynamic
    note: str = ""


@dataclass
class HloSidecarReport:
    loops: list[LoopInfo] = field(default_factory=list)
    recommended_inner_loop_iters: int | None = None
    dynamic_bound_detected: bool = False
    notes: list[str] = field(default_factory=list)


def _extract_scan_length(eqn: object) -> int | None:
    """Read `length` param from a scan eqn; return None if not a concrete int."""
    length = getattr(eqn, "params", {}).get("length", None)
    if isinstance(length, int):
        return length
    return None


def _walk_jaxpr(jaxpr: object, depth: int, report: HloSidecarReport) -> None:
    """Recursively walk a jaxpr, recording while/scan/fori_loop primitives."""
    for eqn in jaxpr.eqns:  # type: ignore[attr-defined]
        name = eqn.primitive.name
        if name == "scan":
            length = _extract_scan_length(eqn)
            report.loops.append(
                LoopInfo(
                    primitive="scan",
                    depth=depth,
                    trip_count=length,
                    note="" if length is not None else "dynamic length",
                )
            )
            if length is None:
                report.dynamic_bound_detected = True
            # Recurse into the body
            body = eqn.params.get("jaxpr", None)
            if body is not None:
                _walk_jaxpr(getattr(body, "jaxpr", body), depth + 1, report)
        elif name == "while":
            # while_loop hides trip count entirely (loop condition is a function).
            # Some lowerings (e.g. fori_loop translated to while) carry a
            # `__fori_lower__` hint or use a counter inside body_jaxpr.
            report.loops.append(
                LoopInfo(
                    primitive="while",
                    depth=depth,
                    trip_count=None,
                    note="while_loop — trip count is a body invariant; not statically extractable",
                )
            )
            report.dynamic_bound_detected = True
            body = eqn.params.get("body_jaxpr", None)
            if body is not None:
                _walk_jaxpr(getattr(body, "jaxpr", body), depth + 1, report)
        elif name == "pallas_call":
            report.loops.append(
                LoopInfo(
                    primitive="pallas_call",
                    depth=depth,
                    trip_count=None,
                    note="pallas_call boundary — recurse into kernel body",
                )
            )
            inner = eqn.params.get("jaxpr", None)
            if inner is not None:
                _walk_jaxpr(getattr(inner, "jaxpr", inner), depth + 1, report)
        else:
            # Generic: still recurse if this primitive carries a sub-jaxpr
            for v in eqn.params.values():
                if hasattr(v, "eqns"):
                    _walk_jaxpr(v, depth + 1, report)
                elif hasattr(v, "jaxpr") and hasattr(v.jaxpr, "eqns"):
                    _walk_jaxpr(v.jaxpr, depth + 1, report)


def inspect_pallas_loops(jit_fn: object, *args: object) -> HloSidecarReport:
    """Trace `jit_fn(*args)` and inspect the resulting jaxpr for loops.

    Returns an HloSidecarReport with everything we could extract; if the
    trip count is purely runtime-dependent (typical of RPA/splash SWA),
    `recommended_inner_loop_iters` is None and `dynamic_bound_detected`
    is True — the caller should fall back to the measure-then-back-solve
    workflow.
    """
    import jax  # pyrefly: ignore[missing-import]

    report = HloSidecarReport()
    jaxpr = jax.make_jaxpr(jit_fn)(*args)
    _walk_jaxpr(jaxpr.jaxpr, 0, report)

    # Pick the deepest *concrete* scan as the recommendation. This works for
    # kernels like attention/softmax where the inner-K loop is shape-determined.
    concrete = [
        loop
        for loop in report.loops
        if loop.primitive == "scan" and loop.trip_count is not None
    ]
    if concrete:
        deepest = max(concrete, key=lambda lp: lp.depth)
        report.recommended_inner_loop_iters = deepest.trip_count

    if report.dynamic_bound_detected and report.recommended_inner_loop_iters is None:
        report.notes.append(
            "Dynamic trip count detected (e.g. RPA kv_lens, splash SWA mask). "
            "Use the measure-then-back-solve workflow: run the kernel once, "
            "then K = (measured_us - 6) / per_iter_us where per_iter_us is "
            "static_bundles_per_iter / 2.2 / 1000."
        )

    return report


def back_solve_inner_loop_iters(
    llo_dir: Path,
    measured_us: float,
    hbm_bytes: int,
) -> tuple[int, dict[str, object]]:
    """Back-solve K such that the predictor reproduces measured_us.

    Uses the full predictor (gated-dispatch simulator, per-lane realization,
    memory floor, overhead) via binary search over K in [1, 4096].

    Returns (K, debug_dict). If the measurement is overhead- or memory-bound,
    K is reported as 1 (the inner trip count doesn't affect the result).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from llo_parse import parse_llo_dump  # pyrefly: ignore[missing-import]
    from tpu_roofline import ADDITIVE_OVERHEAD_US  # pyrefly: ignore[missing-import]
    from tpu_roofline import HBM_EFFECTIVE_GBPS  # pyrefly: ignore[missing-import]
    from tpu_roofline import MIN_TIME_FLOOR_US  # pyrefly: ignore[missing-import]
    from tpu_roofline import predict  # pyrefly: ignore[missing-import]
    from tpu_roofline import stats_from_llo  # pyrefly: ignore[missing-import]

    stats = stats_from_llo(Path(llo_dir))
    parsed = parse_llo_dump(Path(llo_dir))
    base_static = stats.static_bundles
    # Apply the parser's dynamic bundle estimate (loop_factor from sphi
    # markers or heuristic). Back-solving uses this as the per-iter cost.
    if parsed.dynamic_bundles > stats.static_bundles:
        factor = parsed.dynamic_bundles / stats.static_bundles
        stats.dynamic_bundles = parsed.dynamic_bundles
        stats.lane_dynamic_busy = [int(b * factor) for b in stats.lane_static_busy]
        stats.loop_factor = factor
    base_dynamic = stats.dynamic_bundles
    memory_us = hbm_bytes / (HBM_EFFECTIVE_GBPS * 1000)

    headroom_us = measured_us - ADDITIVE_OVERHEAD_US
    if measured_us <= MIN_TIME_FLOOR_US + 1e-3:
        return 1, {
            "regime": "overhead-bound",
            "note": "measurement fits min-time floor; K cannot be derived",
        }
    if headroom_us <= memory_us + 1e-3:
        return 1, {
            "regime": "memory-bound",
            "note": "memory floor already covers measurement; K not compute-relevant",
            "memory_floor_us": memory_us,
        }

    def predict_with_k(k: int) -> float:
        total_dyn = base_dynamic * k
        # Mutate stats in place (caller doesn't reuse) — cheap path
        stats.dynamic_bundles = total_dyn
        stats.lane_dynamic_busy = [
            int(b * total_dyn / base_static) for b in stats.lane_static_busy
        ]
        stats.loop_factor = total_dyn / base_static
        return predict(stats, hbm_bytes)["predicted_us"]

    # Binary search K
    lo, hi = 1, 4096
    best_k, best_err = 1, float("inf")
    best_pred = 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        pred = predict_with_k(mid)
        if abs(pred - measured_us) < best_err:
            best_err = abs(pred - measured_us)
            best_k = mid
            best_pred = pred
        if pred < measured_us:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_k, {
        "regime": "compute-bound",
        "memory_floor_us": memory_us,
        "static_bundles": base_static,
        "dynamic_bundles_per_iter": base_dynamic,
        "best_predicted_us": best_pred,
        "abs_error_us": best_err,
    }


def _parse_shape_arg(spec: str) -> int:
    """Compute byte count from 'bf16:8x4096,bf16:8x4096' (sum of all elements × dtype-bytes)."""
    dtype_bytes = {
        "bf16": 2,
        "f16": 2,
        "f32": 4,
        "f8": 1,
        "i32": 4,
        "i64": 8,
        "i8": 1,
        "u8": 1,
    }
    total = 0
    for entry in spec.split(","):
        dtype_str, shape_str = entry.strip().split(":")
        elems = 1
        for x in shape_str.split("x"):
            elems *= int(x)
        total += elems * dtype_bytes[dtype_str]
    return total


def _cmd_trace(args: argparse.Namespace) -> None:
    """Mode 1: trace a JIT'd kernel and report loop structure."""
    mod_path, sym = args.kernel_import.split(":")
    import importlib

    kernel_mod = importlib.import_module(mod_path)
    kernel = getattr(kernel_mod, sym)

    runner_path, runner_sym = args.runner.split(":")
    sys.path.insert(0, str(Path(runner_path).parent.resolve()))
    import importlib.util

    spec = importlib.util.spec_from_file_location("runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_mod)
    make_inputs = getattr(runner_mod, runner_sym)
    raw_args = args.runner_args.split(",") if args.runner_args else []
    coerced = [int(x) if x.lstrip("-").isdigit() else x for x in raw_args]
    inputs = make_inputs(*coerced)

    import jax  # pyrefly: ignore[missing-import]

    report = inspect_pallas_loops(jax.jit(kernel), *inputs)

    print("=== HLO Sidecar — trace mode ===")
    print(f"Loops found: {len(report.loops)}")
    for loop in report.loops:
        indent = "  " * loop.depth
        trip = "?" if loop.trip_count is None else str(loop.trip_count)
        note = f"  -- {loop.note}" if loop.note else ""
        print(f"{indent}[{loop.depth}] {loop.primitive} trip={trip}{note}")
    print()
    print(f"Dynamic bound detected: {report.dynamic_bound_detected}")
    if report.recommended_inner_loop_iters is not None:
        print(f"Recommended --inner-loop-iters: {report.recommended_inner_loop_iters}")
    else:
        print("No static recommendation; use back-solve mode instead.")
    for note in report.notes:
        print(f"  • {note}")


def _cmd_back_solve(args: argparse.Namespace) -> None:
    """Mode 2: back-solve K from one measurement + LLO."""
    hbm_bytes = _parse_shape_arg(args.inputs) + _parse_shape_arg(args.outputs)
    k, debug = back_solve_inner_loop_iters(
        Path(args.llo_dir), args.measured_us, hbm_bytes
    )
    print("=== HLO Sidecar — back-solve mode ===")
    print(f"  measured_us:      {args.measured_us}")
    print(f"  hbm_bytes:        {hbm_bytes}")
    for key, value in debug.items():
        print(f"  {key}: {value}")
    print()
    print(f"  derived inner_loop_iters: {k}")
    if args.persist:
        meta = Path(args.llo_dir) / "meta.yaml"
        if meta.exists():
            text = meta.read_text()
            if "inner_loop_iters:" not in text:
                with meta.open("a") as f:
                    f.write(f"\ninner_loop_iters: {k}  # derived by hlo_sidecar\n")
                print(f"  → appended to {meta}")
            else:
                print(f"  ! {meta} already contains inner_loop_iters; not overwriting")
        else:
            meta.write_text(f"inner_loop_iters: {k}  # derived by hlo_sidecar\n")
            print(f"  → wrote {meta}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("trace", help="Static trace mode (mode 1)")
    pt.add_argument(
        "--kernel-import",
        required=True,
        help="'module:symbol' of the kernel function (e.g. "
        "'my_pkg.my_kernel:my_kernel_fn')",
    )
    pt.add_argument(
        "--runner",
        required=True,
        help="'path:symbol' of the input-builder function (e.g. "
        "'scripts/llo_runner_gmm.py:make_inputs')",
    )
    pt.add_argument(
        "--runner-args",
        default="",
        help="Comma-separated args for the input-builder (e.g. '1024,4096,4096,8')",
    )

    pb = sub.add_parser("back-solve", help="Back-solve K from one measurement (mode 2)")
    pb.add_argument("--llo-dir", required=True)
    pb.add_argument(
        "--inputs", required=True, help="e.g. 'bf16:32x8x128,bf16:65536x1x128'"
    )
    pb.add_argument("--outputs", required=True)
    pb.add_argument("--measured-us", type=float, required=True)
    pb.add_argument(
        "--persist",
        action="store_true",
        help="Append `inner_loop_iters: K` to the entry's meta.yaml",
    )

    args = p.parse_args()
    if args.cmd == "trace":
        _cmd_trace(args)
    else:
        _cmd_back_solve(args)


if __name__ == "__main__":
    main()
