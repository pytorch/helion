"""Compare Helion CuTe/Triton with FLA and PyTorch on the linear-attention examples.

Usage::

    python benchmarks/cute/compare_linattn_backends.py run --gpu 3 --output DIR \\
        [--variants vanilla,delta_rule,gated_delta_rule] [--shapes s0,s1,s2] \\
        [--shape NAME=B,H,T,D,DV] [--impls helion-cute,helion-triton,fla,...] \\
        [--effort full|quick|none] [--seed N] [--pin-configs FILE.json] [--abab K] \\
        [--sustain-ms N]
    python benchmarks/cute/compare_linattn_backends.py report --input DIR \\
        [--output FILE.md]

Each (variant, shape, impl) job runs in a fresh subprocess with private compiler
caches and a recorded autotune seed.  The worker checks the forward output
against the fp32 recurrent reference and every gradient against the chunked
reference's autograd, then times forward and forward+backward with
``triton.testing.do_bench`` (the same timer for every implementation).  Helion
jobs record their autotune metrics and the configs of all bound engine kernels;
``report`` writes them as ``configs/*.json`` plus ``pin_configs.json``, which
``run --pin-configs`` replays without autotuning.  ``--abab K`` runs the impls K
times in round-robin order and reports per-impl medians.  ``--sustain-ms N``
runs each timed callable back to back for N ms before its do_bench series so
the GPU is at its power-capped steady-state clock when timing starts (a B200
steps down from boost 40-60 ms into a series at s2, swinging the means by
+-15 us between runs); 0 (the default) times exactly as before.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Sequence
from typing import cast

if TYPE_CHECKING:
    from examples.linear.linear_attention_harness import Inputs
    from examples.linear.linear_attention_harness import LinearAttentionExampleHarness
    import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "linattn-backend-comparison-v1"
INPUT_SEED = 20261011
ENGINE_MODULE = "examples.linear.linear_attention_engine"
VARIANTS = {
    "vanilla": "examples.linear.example_vanilla_linear_attn",
    "delta_rule": "examples.linear.example_delta_rule",
    "gated_delta_rule": "examples.linear.example_gated_delta_rule",
}
SHAPES: dict[str, tuple[int, ...]] = {
    "s0": (8, 8, 1024, 64, 64),
    "s1": (4, 16, 2048, 128, 128),
    "s2": (2, 16, 16384, 128, 128),
}
IMPLS = ("helion-cute", "helion-triton", "fla", "torch-compile", "aten-chunked")
HELION_IMPLS = {"helion-cute": "cute", "helion-triton": "triton"}
BASELINES = ("fla", "helion-triton", "torch-compile", "aten-chunked")
TIMER = (
    "triton.testing.do_bench(warmup=25, rep=100, return_mode=all) after "
    "gc.collect(); gc.freeze(); mean reported, median/min/std recorded"
)


def job_key(impl: str, variant: str, shape: str) -> str:
    return f"{impl}/{variant}/{shape}"


def split_list(value: str, allowed: Iterable[str], what: str) -> list[str]:
    names = value.split(",")
    if set(names) - set(allowed) or len(set(names)) != len(names):
        raise ValueError(f"Unknown or duplicate {what} in {value!r}")
    return names


def make_forward(
    impl: str, harness: LinearAttentionExampleHarness, inputs: Inputs
) -> Callable[[Inputs], torch.Tensor]:
    """Forward of one implementation in its native layout, taking an Inputs."""
    from examples.linear.linear_attention_harness import BENCH_C
    from examples.linear.linear_attention_utils import chunked_linear_attn_reference
    import torch

    if impl in HELION_IMPLS:

        def helion_forward(i: Inputs) -> torch.Tensor:
            return harness.helion_fwd(i, BENCH_C)

        return helion_forward
    if impl == "fla":

        def fla_forward(i: Inputs) -> torch.Tensor:
            return harness.fla_fwd(i, inputs.scale)

        return fla_forward
    assert inputs.g is not None
    decay, beta, scale = inputs.g, inputs.beta, inputs.scale

    # The chunked-reference baseline exactly as example_cases.py's LINEAR branch
    # builds it: q is pre-scaled, decay/beta are closed over.
    def ref_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return chunked_linear_attn_reference(q * scale, k, v, decay, beta, C=BENCH_C)

    fn = torch.compile(ref_fn, fullgraph=True) if impl == "torch-compile" else ref_fn

    def chunked_forward(i: Inputs) -> torch.Tensor:
        return fn(i.q, i.k, i.v)

    return chunked_forward


def worker(args: argparse.Namespace) -> None:
    from benchmarks.cute.compare_example_backends import write_json
    from examples.linear import linear_attention_harness as lah
    from examples.linear.linear_attention_utils import ACC_BWD_TOL
    from examples.linear.linear_attention_utils import ACC_FWD_TOL
    import torch
    import triton.testing

    import helion
    from helion.autotuner.metrics import register_post_autotune_hook

    dims = tuple(int(value) for value in args.dims.split(","))
    key = job_key(args.impl, args.variant, args.shape)
    result: dict[str, Any] = {
        "variant": args.variant,
        "shape": args.shape,
        "dims": list(dims),
        "impl": args.impl,
        "status": "error",
        "autotune_seed": args.seed,
        "input_seed": INPUT_SEED,
        "effort": os.environ["HELION_AUTOTUNE_EFFORT"],
        "pinned": [],
    }
    started = time.monotonic()
    try:
        torch.cuda.set_device(0)
        result["device"] = torch.cuda.get_device_name(0)
        harness = importlib.import_module(VARIANTS[args.variant]).HARNESS
        engine = importlib.import_module(ENGINE_MODULE)
        kernels = {
            obj.name: obj
            for obj in vars(engine).values()
            if isinstance(obj, helion.Kernel)
        }
        pinned: dict[str, dict[str, Any]] = {}
        if args.pin_configs is not None:
            pinned = json.loads(args.pin_configs.read_text()).get(key, {})
        for name, config in pinned.items():
            # A single decorator config with force_autotune off is returned by
            # BoundKernel._user_provided_config, so no search runs for it.
            kernels[name].configs = [helion.Config(**config)]
        result["pinned"] = sorted(pinned)
        autotunes: list[dict[str, Any]] = []
        register_post_autotune_hook(lambda metrics: autotunes.append(metrics.to_dict()))

        B, H, T, D, DV = dims
        torch.manual_seed(INPUT_SEED)
        inputs = harness.make_inputs(
            B, H, T, D, DV, dtype=lah.DTYPE, device="cuda", requires_grad=True
        )
        grad_out = torch.randn(B, H, T, DV, device="cuda", dtype=lah.DTYPE)
        forward = make_forward(args.impl, harness, inputs)
        time_first = args.impl == "fla"
        back = lah._htf if time_first else (lambda t: t)
        if time_first:
            bench_inputs, leaves = lah._fla_benchmark_inputs(harness, inputs)
            bench_grad = lah._htf(grad_out)
        else:
            bench_inputs = inputs
            leaves = [getattr(inputs, name) for name in harness.grad_tensors]
            bench_grad = grad_out

        # Correctness as run_accuracy checks it: forward vs the fp32 recurrent
        # reference, gradients vs the chunked reference's autograd.
        out = back(forward(bench_inputs).detach())
        with torch.no_grad():
            reference = harness.reference(inputs)
        fwd_error = lah._rel_error(out, reference)
        c_inputs, c_leaves = lah._grad_leaves(
            harness, lah._fla_inputs(inputs) if time_first else inputs
        )
        forward(c_inputs).backward(bench_grad)
        r_inputs, r_leaves = lah._grad_leaves(harness, inputs)
        harness.chunked_reference(r_inputs, lah.BENCH_C).backward(grad_out)
        bwd_errors = {
            name: lah._rel_error(back(c.grad), r.grad)
            for name, c, r in zip(harness.grad_tensors, c_leaves, r_leaves, strict=True)
        }
        result.update(
            fwd_error=fwd_error,
            fwd_pass=fwd_error < ACC_FWD_TOL,
            bwd_errors=bwd_errors,
            bwd_pass=all(error < ACC_BWD_TOL for error in bwd_errors.values()),
            fwd_tol=ACC_FWD_TOL,
            bwd_tol=ACC_BWD_TOL,
        )

        def fwd_only() -> None:
            forward(bench_inputs)

        def fwd_bwd() -> None:
            forward(bench_inputs).backward(bench_grad)

        def sustain(fn: Callable[[], None], grad_to_none: Sequence[Any] = ()) -> None:
            """Run the timed callable back to back for ``--sustain-ms`` so the
            GPU sits at its capped steady-state clock when the series starts
            (the same per-call grad reset do_bench applies); 0 skips."""
            if args.sustain_ms <= 0:
                return
            torch.cuda.synchronize()
            deadline = time.monotonic() + args.sustain_ms / 1000
            while time.monotonic() < deadline:
                for leaf in grad_to_none:
                    leaf.grad = None
                fn()
            torch.cuda.synchronize()

        # Compilation leaves ~1M tracked objects behind; a gen-2 collection
        # landing inside the timed loop (300+ ms) would dominate a mean of
        # 100 ms worth of samples.  Freeze them so only the loop's own
        # garbage is ever collected, for every implementation alike.
        gc.collect()
        gc.freeze()
        sustain(fwd_only)
        fwd_samples = cast(
            "list[float]",
            triton.testing.do_bench(fwd_only, return_mode="all"),
        )
        sustain(fwd_bwd, leaves)
        fb_samples = cast(
            "list[float]",
            triton.testing.do_bench(fwd_bwd, grad_to_none=leaves, return_mode="all"),
        )
        for field, samples in (("fwd_ms", fwd_samples), ("fb_ms", fb_samples)):
            # The mean stays the reported metric (same value do_bench's
            # default return_mode gives); the spread is recorded alongside.
            result[field] = statistics.mean(samples)
            result[f"{field}_median"] = statistics.median(samples)
            result[f"{field}_min"] = min(samples)
            result[f"{field}_std"] = statistics.pstdev(samples)
            result[f"{field}_samples"] = len(samples)
        result["timer"] = TIMER
        result["sustain_ms"] = args.sustain_ms
        result["autotunes"] = autotunes
        result["kernel_configs"] = {
            name: [
                dict(bound._config.config)
                for bound in obj._bound_kernels.values()
                if bound._config is not None
            ]
            for name, obj in kernels.items()
            if obj._bound_kernels
        }
        if args.impl in HELION_IMPLS:
            if any(
                obj.settings.backend != HELION_IMPLS[args.impl]
                for obj in kernels.values()
            ):
                raise RuntimeError("An engine kernel used a different Helion backend")
            for name, config in pinned.items():
                wanted = json.loads(json.dumps(helion.Config(**config).config))
                ran = json.loads(json.dumps(result["kernel_configs"].get(name, [])))
                if any(candidate != wanted for candidate in ran):
                    raise RuntimeError(f"Pinned config for {name} is not what ran")
            note = "" if autotunes else " (no autotuning ran)"
            print(
                f"pinned {len(pinned)} kernels; autotune runs: {len(autotunes)}{note}",
                flush=True,
            )
        result["status"] = (
            "ok" if result["fwd_pass"] and result["bwd_pass"] else "numerics"
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(args.output, result)


def run(args: argparse.Namespace) -> None:
    from benchmarks.cute.compare_attention_backends import _git_source_snapshot
    from benchmarks.cute.compare_attention_backends import _git_worktree_root_matches
    from benchmarks.cute.compare_example_backends import check_gpu_idle
    from benchmarks.cute.compare_example_backends import provenance
    from benchmarks.cute.compare_example_backends import worker_environment
    from benchmarks.cute.compare_example_backends import write_json

    variants = split_list(args.variants, VARIANTS, "variant")
    impls = split_list(args.impls, IMPLS, "impl")
    shapes = dict(SHAPES)
    for spec in args.shape:
        name, _, dims = spec.partition("=")
        values = tuple(int(value) for value in dims.split(","))
        if not name or len(values) != 5:
            raise ValueError(f"--shape expects NAME=B,H,T,D,DV, got {spec!r}")
        shapes[name] = values
    selected = (
        split_list(args.shapes, shapes, "shape")
        if args.shapes
        else [*SHAPES, *(name for name in shapes if name not in SHAPES)]
    )
    pins = json.loads(args.pin_configs.read_text()) if args.pin_configs else {}
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "raw"
    raw.mkdir()
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source.stdout.strip()
        if source.returncode == 0 and _git_worktree_root_matches(REPO_ROOT)
        else None,
        "source_snapshot": _git_source_snapshot(
            REPO_ROOT, ("helion", "examples", "benchmarks/cute")
        ),
        "variants": variants,
        "shapes": {name: list(shapes[name]) for name in selected},
        "impls": impls,
        "gpu": args.gpu,
        "gpu_uuid": check_gpu_idle(args.gpu),
        "effort": args.effort,
        "abab": args.abab,
        "sustain_ms": args.sustain_ms,
        "pin_configs": None if args.pin_configs is None else str(args.pin_configs),
        "autotune_seed": seed,
        "input_seed": INPUT_SEED,
        **provenance(args.gpu),
        "timer": TIMER,
        "jobs": [],
    }

    def execute(variant: str, shape: str, impl: str, round_index: int) -> None:
        check_gpu_idle(args.gpu)
        index = len(record["jobs"])
        worker_seed = (seed + index) % (1 << 63)
        key = job_key(impl, variant, shape)
        output = raw / f"{index:05d}.json"
        log = raw / f"{index:05d}.log"
        with tempfile.TemporaryDirectory(prefix="hbench-") as directory:
            env = worker_environment(
                args.gpu, HELION_IMPLS.get(impl, "triton"), Path(directory), worker_seed
            )
            env["HELION_AUTOTUNE_EFFORT"] = args.effort
            options = {
                "--variant": variant,
                "--shape": shape,
                "--dims": ",".join(str(value) for value in shapes[shape]),
                "--impl": impl,
                "--seed": str(worker_seed),
                "--output": str(output),
                "--sustain-ms": str(args.sustain_ms),
            }
            if key in pins:
                env["HELION_FORCE_AUTOTUNE"] = "0"
                options["--pin-configs"] = str(args.pin_configs)
            command = [sys.executable, str(Path(__file__).resolve()), "worker"]
            command += [token for item in options.items() for token in item]
            with log.open("w") as stream:
                process = subprocess.run(
                    command,
                    env=env,
                    cwd=REPO_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result: dict[str, Any] = (
                json.loads(output.read_text())
                if output.exists()
                else {
                    "variant": variant,
                    "shape": shape,
                    "impl": impl,
                    "status": "process_error",
                }
            )
            if process.returncode:
                result.update(status="process_error", returncode=process.returncode)
                write_json(output, result)
        record["jobs"].append(
            {
                "path": str(output.relative_to(args.output)),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "log": str(log.relative_to(args.output)),
                "status": result["status"],
                "variant": variant,
                "shape": shape,
                "impl": impl,
                "round": round_index,
                "pinned": key in pins,
                "autotune_seed": worker_seed,
            }
        )
        write_json(args.output / "results.json", record)
        print(f"{variant}/{shape} {impl} round {round_index}: {result['status']}")
        sys.stdout.flush()

    try:
        for variant in variants:
            for shape in selected:
                for round_index in range(args.abab or 1):
                    for impl in impls:
                        execute(variant, shape, impl, round_index)
        record["status"] = "complete"
    finally:
        write_json(args.output / "results.json", record)


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def summarize(record: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in record["variants"]:
        for shape in record["shapes"]:
            cells: dict[str, dict[str, Any]] = {}
            for impl in record["impls"]:
                runs = [
                    job
                    for job in jobs
                    if (job["variant"], job["shape"], job["impl"])
                    == (variant, shape, impl)
                ]
                timed = [job for job in runs if "fb_ms" in job]
                statuses = [job["status"] for job in runs]
                if not runs:
                    status = "missing"
                elif all(status == "ok" for status in statuses):
                    status = "ok"
                else:
                    status = "numerics" if "numerics" in statuses else statuses[-1]
                cells[impl] = {
                    "status": status,
                    "rounds": len(runs),
                    **{
                        field: statistics.median(j[field] for j in timed)
                        if timed
                        else None
                        for field in ("fwd_ms", "fb_ms")
                    },
                    "fwd_error": max((j["fwd_error"] for j in timed), default=None),
                    "bwd_errors": [j["bwd_errors"] for j in timed],
                }
            row: dict[str, Any] = {
                "variant": variant,
                "shape": shape,
                "dims": record["shapes"][shape],
                "impls": cells,
            }
            for metric in ("fb_ms", "fwd_ms"):
                baselines = {
                    impl: cells[impl][metric]
                    for impl in BASELINES
                    if impl in cells and cells[impl][metric] is not None
                }
                best = min(baselines, key=baselines.__getitem__) if baselines else None
                cute = cells.get("helion-cute", {}).get(metric)
                row[f"best_{metric}"] = best
                row[f"ratio_{metric}"] = (
                    baselines[best] / cute if best is not None and cute else None
                )
            rows.append(row)
    return {
        "rows": rows,
        "geomean_fb": geomean([r["ratio_fb_ms"] for r in rows if r["ratio_fb_ms"]]),
        "geomean_fwd": geomean([r["ratio_fwd_ms"] for r in rows if r["ratio_fwd_ms"]]),
        "sustain_ms": record.get("sustain_ms", 0),
    }


def cell_text(cell: dict[str, Any], metric: str) -> str:
    if cell[metric] is None:
        return cell["status"]
    return f"{cell[metric]:.3f}" + ("" if cell["status"] == "ok" else "!")


def markdown(record: dict[str, Any], summary: dict[str, Any]) -> str:
    impls = record["impls"]
    rounds = record["abab"] or 1
    lines = [
        (
            f"# Linear attention: {record['gpu_properties']['name']}, "
            f"effort={record['effort']}, rounds={rounds}, "
            f"sustain_ms={record.get('sustain_ms', 0)}, "
            f"commit={record['source_commit']}"
        ),
        "",
        (
            f"Timer: {record['timer']}; cells are medians over rounds in ms; "
            "`!` marks a correctness or run failure in at least one round "
            "(details in summary.json). ratio = best baseline / helion-cute "
            "(>1 means cute is faster)."
        ),
    ]
    for title, metric, key in (
        ("Forward + backward", "fb_ms", "geomean_fb"),
        ("Forward only", "fwd_ms", "geomean_fwd"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| variant | shape | " + " | ".join(impls) + " | best baseline | ratio |",
            "|" + "---|" * (len(impls) + 4),
        ]
        for row in summary["rows"]:
            cells = [cell_text(row["impls"][impl], metric) for impl in impls]
            best = row[f"best_{metric}"] or "n/a"
            ratio = row[f"ratio_{metric}"]
            lines.append(
                f"| {row['variant']} | {row['shape']} {tuple(row['dims'])} | "
                + " | ".join(cells)
                + f" | {best} | {'n/a' if ratio is None else f'{ratio:.3f}'} |"
            )
        value = summary[key]
        lines.append(
            "| geomean | | "
            + " | ".join("" for _ in impls)
            + f" | | {'n/a' if value is None else f'{value:.3f}'} |"
        )
    return "\n".join(lines) + "\n"


def report(args: argparse.Namespace) -> None:
    from benchmarks.cute.compare_example_backends import write_json

    record = json.loads((args.input / "results.json").read_text())
    jobs = []
    for entry in record["jobs"]:
        job = json.loads((args.input / entry["path"]).read_text())
        job["round"] = entry["round"]
        jobs.append(job)
    summary = summarize(record, jobs)
    write_json(args.input / "summary.json", summary)
    configs = args.input / "configs"
    configs.mkdir(exist_ok=True)
    pin: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if job["impl"] not in HELION_IMPLS or "kernel_configs" not in job:
            continue
        key = job_key(job["impl"], job["variant"], job["shape"])
        if key in pin:
            continue
        pin[key] = {}
        for name, found in job["kernel_configs"].items():
            distinct = {json.dumps(config, sort_keys=True) for config in found}
            if len(distinct) > 1:
                print(
                    f"warning: {key} {name} ran {len(distinct)} configs; pinning first"
                )
            if found:
                pin[key][name] = found[0]
        write_json(
            configs / f"{key.replace('/', '__')}.json",
            {
                "key": key,
                "round": job["round"],
                "status": job["status"],
                "fwd_ms": job["fwd_ms"],
                "fb_ms": job["fb_ms"],
                "fwd_ms_median": job.get("fwd_ms_median"),
                "fb_ms_median": job.get("fb_ms_median"),
                "pinned": job["pinned"],
                "kernel_configs": job["kernel_configs"],
                "autotunes": job["autotunes"],
            },
        )
    write_json(args.input / "pin_configs.json", pin)
    text = markdown(record, summary)
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="action", required=True)
    runner = commands.add_parser("run", help="Run the comparison on one GPU")
    runner.add_argument("--gpu", required=True)
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--variants", default=",".join(VARIANTS))
    runner.add_argument("--shapes", default="", help="Comma-separated shape names")
    runner.add_argument(
        "--shape", action="append", default=[], metavar="NAME=B,H,T,D,DV"
    )
    runner.add_argument("--impls", default=",".join(IMPLS))
    runner.add_argument("--effort", choices=["full", "quick", "none"], default="full")
    runner.add_argument("--seed", type=int, help="Run seed; default: fresh random")
    runner.add_argument(
        "--pin-configs", type=Path, help="pin_configs.json from a previous report"
    )
    runner.add_argument(
        "--abab", type=int, help="Interleave the impls K times; report medians"
    )
    runner.add_argument(
        "--sustain-ms",
        type=int,
        default=0,
        help="Run each timed callable for N ms before its do_bench series "
        "(GPU at its capped steady-state clock); 0 = time as before",
    )
    child = commands.add_parser("worker", help=argparse.SUPPRESS)
    child.add_argument("--variant", required=True, choices=list(VARIANTS))
    child.add_argument("--shape", required=True)
    child.add_argument("--dims", required=True)
    child.add_argument("--impl", required=True, choices=IMPLS)
    child.add_argument("--seed", required=True, type=int)
    child.add_argument("--output", required=True, type=Path)
    child.add_argument("--pin-configs", type=Path)
    child.add_argument("--sustain-ms", type=int, default=0)
    reporter = commands.add_parser(
        "report", help="Print the markdown tables; write summary.json and configs/"
    )
    reporter.add_argument("--input", required=True, type=Path)
    reporter.add_argument("--output", type=Path, help="Also write the markdown here")
    args = parser.parse_args()
    if args.action == "report":
        report(args)
    elif args.action == "worker":
        worker(args)
    else:
        if args.seed is not None and not 0 <= args.seed < (1 << 63):
            parser.error("--seed must be in [0, 2**63)")
        if args.abab is not None and args.abab < 1:
            parser.error("--abab must be positive")
        if args.sustain_ms < 0:
            parser.error("--sustain-ms must be non-negative")
        args.output = args.output.resolve()
        if args.pin_configs is not None:
            args.pin_configs = args.pin_configs.resolve()
        run(args)


if __name__ == "__main__":
    main()
