"""Helion Metal benchmark: the dashboard's nightly kernels on Apple GPUs.

Compares Helion's Metal backend with torch eager, torch.compile and MLX on
the inputs the GPU nightly uses, and records where Helion is wrong and where
it is slower.  The suite is frozen in benchmarks/metal/manifest.py; each
kernel's inputs, reference and implementations are in benchmarks/metal/ops/.

A snapshot (--output OUT) runs in a supervised child process that
checkpoints OUT/results.json after every step.  A child that dies, or whose
results stop changing for --stall-timeout minutes, is restarted with
--resume, and the step it was in is recorded as a crash or timeout.
--resume continues a run by hand and refuses one whose code, manifest or
arguments changed.  Every run writes OUT/summary.md, OUT/dashboard.html and
OUT/helionbench.json.

A preflight refuses to start on battery, in Low Power Mode, under a thermal
limit or beside a busy process; --force runs anyway and marks the run
(--correctness-only is not gated).  A drift sentinel timed at the start and
end of a snapshot is reported as the run's stability against the noise
calibration.

--ab BASE CAND runs two commits' Helion, each from its own worktree, in
alternating blocks, either re-tuning both (--retune) or replaying one
autotune cache (--replay), and gives each cell a better/worse/neutral
verdict plus a fixed guard set.  It needs a noise calibration
(--calibrate, an A/B of HEAD against itself) taken with the same --blocks.
Machine state (the lock and the calibration) lives in
$HELION_METAL_BENCH_HOME, default ~/.cache/helion-metal-bench.

Usage:
    python benchmarks/run_metal.py --list
    python benchmarks/run_metal.py --output OUT                 # full snapshot
    python benchmarks/run_metal.py --output OUT --kernel softmax,rope --case 0
    python benchmarks/run_metal.py --output OUT --correctness-only
    python benchmarks/run_metal.py --output OUT --resume
    python benchmarks/run_metal.py --calibrate
    python benchmarks/run_metal.py --output OUT --ab BASE CAND --retune
    python benchmarks/run_metal.py --output OUT --ab BASE CAND --retune --base-tunes CACHE
    python benchmarks/run_metal.py --output OUT --ab BASE CAND --replay SNAPSHOT/helion-cache

The exit code is nonzero only on harness errors; cell failures are data.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.metal.manifest import PROVIDERS  # noqa: E402  (imports no torch)

# How often the supervisor checks that its run is still making progress.
POLL_S = 30


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list kernels and cases")
    parser.add_argument("--output", type=Path, help="result directory")
    parser.add_argument("--kernel", help="comma-separated kernel names")
    parser.add_argument("--case", help="comma-separated row ids")
    parser.add_argument(
        "--providers", default=",".join(PROVIDERS), help="comma-separated providers"
    )
    parser.add_argument("--blocks", type=int, default=5, help="repeat blocks per cell")
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue --output")
    parser.add_argument(
        "--force", action="store_true", help="run despite failed preflight checks"
    )
    parser.add_argument("--calibrate", action="store_true", help="measure noise")
    parser.add_argument("--ab", nargs=2, metavar=("BASE", "CAND"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--retune", action="store_true", help="A/B: both refs re-tune")
    mode.add_argument(
        "--replay", type=Path, metavar="CACHE", help="A/B: both refs replay CACHE"
    )
    parser.add_argument(
        "--base-tunes",
        type=Path,
        metavar="CACHE",
        help="A/B --retune: BASE starts from CACHE, an earlier retune's BASE configs",
    )
    parser.add_argument(
        "--autotune-effort", default="full", choices=("none", "quick", "full")
    )
    parser.add_argument("--autotune-seed", type=int, default=1)
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=120,
        metavar="MIN",
        help=(
            "kill and resume a run whose results have not changed for MIN "
            "minutes; must exceed the longest single autotune"
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cells", help=argparse.SUPPRESS)
    parser.add_argument("--supervised", action="store_true", help=argparse.SUPPRESS)
    # Set by the supervisor when it restarts a run that died.
    parser.add_argument(
        "--last-death", choices=("crash", "timeout"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if not (args.calibrate or args.list) and args.output is None:
        parser.error("--output is required")
    if args.ab and not (args.retune or args.replay):
        parser.error("--ab needs --retune or --replay")
    if args.base_tunes and not args.retune:
        parser.error("--base-tunes needs --ab ... --retune")
    unknown = set(args.providers.split(",")) - set(PROVIDERS)
    if unknown:
        parser.error(f"unknown providers {sorted(unknown)}")
    return args


def supervise(argv: list[str], results: Path, stall_s: float) -> None:
    """Run the benchmark in a child, resuming it until it finishes.

    A provider can kill the process (a Metal compiler crash, the OS killing an
    oversized allocation) or hang it (a deadlocked kernel, a stuck compile).
    A child that dies from a signal, or whose ``results`` file has not changed
    for ``stall_s`` seconds, is resumed; the resumed run records the step it
    was in as a crash or timeout and moves on, so every restart makes progress.
    """
    # The child is in its own session, so a terminal hangup or a SIGTERM no
    # longer reaches it; turn both into an exit that kills its group.
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _exit_on_signal)
    child = [sys.executable, __file__, *argv, "--supervised"]
    while True:
        # Its own process group, so a kill also reaches the processes it
        # forked (e.g. autotuning workers), including on Ctrl-C here.
        proc = subprocess.Popen(child, start_new_session=True)
        try:
            death = _watch(proc, results, stall_s)
        finally:
            _kill_group(proc)
        if death is None:
            sys.exit(proc.returncode)
        print(f"run ended by {death}; resuming", file=sys.stderr)
        child = [arg for arg in child if not arg.startswith("--last-death")]
        child.append(f"--last-death={death}")
        if "--resume" not in child:
            child.append("--resume")


def _exit_on_signal(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Kill whatever is left of ``proc``'s process group."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def _watch(proc: subprocess.Popen[bytes], results: Path, stall_s: float) -> str | None:
    """Wait for ``proc``; return how it died, or None if it exited."""
    last_change = time.monotonic()
    last_mtime = None
    while True:
        try:
            code = proc.wait(timeout=POLL_S)
        except subprocess.TimeoutExpired:
            mtime = results.stat().st_mtime if results.exists() else None
            if mtime != last_mtime:
                last_mtime, last_change = mtime, time.monotonic()
            elif time.monotonic() - last_change > stall_s:
                return "timeout"
            continue
        return "crash" if code < 0 else None


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if not (args.supervised or args.list or args.calibrate or args.ab):
        supervise(argv, args.output / "results.json", args.stall_timeout * 60)
    # Read when torch and Helion are imported, so set before importing them.
    # Cap MPS at the recommended working set so an oversized cell raises
    # out-of-memory instead of swapping; the low watermark keeps its default
    # proportion to the high one (1.4 / 1.7).
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
    os.environ["HELION_AUTOTUNE_EFFORT"] = args.autotune_effort
    os.environ["HELION_AUTOTUNE_RANDOM_SEED"] = str(args.autotune_seed)
    # A/B workers get their side's cache from the parent; calibration and the
    # A/B parent run no Helion themselves.
    if args.output is not None and not (args.worker or args.ab):
        os.environ["HELION_CACHE_DIR"] = str(args.output.resolve() / "helion-cache")

    from benchmarks.metal import harness
    from benchmarks.metal import manifest

    if args.list:
        for kernel in manifest.KERNELS:
            print(f"{kernel.name}  ({kernel.source})")
            for case in kernel.cases:
                print(f"  case {case.index:3d}  {case.label}  {case.dtype}")
        return

    kernels = args.kernel.split(",") if args.kernel else None
    cases = [int(c) for c in args.case.split(",")] if args.case else None
    if kernels is not None:
        known = {k.name for k in manifest.KERNELS}
        if set(kernels) - known:
            raise SystemExit(f"unknown kernels {sorted(set(kernels) - known)}")
    options = harness.Options(
        providers=tuple(args.providers.split(",")),
        blocks=args.blocks,
        correctness_only=args.correctness_only,
        order_seed=args.order_seed,
    )
    if args.worker:
        header = harness.run_header(vars_of(args), options, args.output)
        harness.snapshot(
            args.output,
            harness.cells_by_key(args.cells.split(",")),
            options,
            header,
            resume=args.resume,
            last_death=args.last_death,
            measure_drift=False,
        )
        return

    with harness.run_lock():
        state = harness.preflight()
        if state["problems"] and preflight_gates(args):
            raise SystemExit(
                "preflight failed (--force to run anyway):\n  "
                + "\n  ".join(state["problems"])
            )
        worker_args = ["--autotune-effort", args.autotune_effort]
        worker_args += ["--autotune-seed", str(args.autotune_seed)]
        worker_args += ["--stall-timeout", str(args.stall_timeout)]
        if args.calibrate:
            calibration = harness.calibrate(
                harness.bench_home() / "calibration-runs",
                worker_args,
                args.blocks,
                args.order_seed,
            )
            for mode, spreads in calibration["spreads"].items():
                print(f"calibrated {len(spreads)} {mode} cells")
            return
        chosen = harness.selected_cases(kernels, cases)
        if args.ab:
            report = harness.ab(
                args.output.resolve(),
                args.ab[0],
                args.ab[1],
                "retune" if args.retune else "replay",
                args.replay.resolve() if args.replay else None,
                worker_args,
                harness.ab_cells(chosen),
                args.blocks,
                args.order_seed,
                args.base_tunes.resolve() if args.base_tunes else None,
            )
            for key, entry in report["cells"].items():
                print(key, entry.get("verdict", "untimed"), entry["correctness"])
            return
        header = harness.run_header(vars_of(args), options, args.output)
        results = harness.snapshot(
            args.output,
            chosen,
            options,
            header,
            resume=args.resume,
            last_death=args.last_death,
            measure_drift=not args.correctness_only,
        )
        print(f"wrote {results.out_dir / 'summary.md'}")


def preflight_gates(args: argparse.Namespace) -> bool:
    """Whether failed preflight checks stop this run.

    A run the supervisor restarts is not gated again: it already passed (or
    was forced past) the check, and stopping would abandon it.  The resumed
    run records what the check found.
    """
    return not (args.force or args.correctness_only or args.last_death)


def vars_of(args: argparse.Namespace) -> dict[str, object]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


if __name__ == "__main__":
    main(sys.argv[1:])
