"""Plan a campaign run from the records of earlier ones.

Every point of the selection gets a cold CuTe autotune: the CuTe code
generation has moved too far for earlier CuTe configs to stand.  A Triton
point reuses the newest ok autotune record of the earlier runs (``--prior``,
newest first) when

- its activity was sampled and clean (``common.activity_issue``; with
  ``--reuse-unsampled`` a record made before the sampling existed, with no
  samples at all, counts as clean, never one whose samples miss its job),
- its searches ran in the regime the run's Triton autotunes will run in
  (``--regime``; unnamed, Triton's default, ``common.default_regime``),
  where its port has several kernels (``common.record_regime``: a record
  made before records named a regime was tuned alone), and
- its pinned kernels still generate the code they were tuned as, at this
  tree (``common.pin_roundtrip`` after one forward of the pinned port, as
  ``bench.py`` checks it; this needs a GPU, and ``--no-pin-check`` reuses
  nothing);

any other Triton point, one whose config this tree rejects included, is
autotuned again.  Every point is benched.  Run it on a GPU of your own::

    scripts/gpu_board.py run -n 1 --purpose "plan a better-benchmark run" -- \\
        python benchmarks/cute/better_benchmark/plan_run.py \\
        --run-dir artifacts/bb-new --prior artifacts/bb-old \\
        --prior artifacts/bb-older --write

It prints the jobs, why each Triton point is autotuned again, and the
GPU-hours they should take, from the durations (``wall_s``) of the earlier
records of the same jobs (the median of a kind's other jobs where a point
has none).  ``--write`` writes the plan to ``<run>/plan.json`` and copies
the reused records into ``<run>/configs``, each naming its source in
``reused_from``; ``campaign.py`` takes them for finished, so a campaign on
that run directory runs exactly the planned jobs.  It writes only into a run
directory that holds no records yet.

With ``--provisional-triton``, ``--write`` also copies the earlier record of
each Triton point to autotune again, marked ``provisional``: benches pin it
until the point's own autotune has run, and the campaign runs that autotune
all the same.  A run can then report its primary result before the Triton
autotunes: ``campaign.py --backends cute`` (every CuTe autotune and the
benches, with Helion-Triton timed on the provisional configs, which reports
flag), then ``campaign.py --backends triton`` (the Triton autotunes, each
point benched again after its own).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.cute.better_benchmark import common
import torch

import helion


def newest_record(
    dirs: list[Path], name: str
) -> tuple[Path, dict[str, Any]] | tuple[None, None]:
    """The first ok record called ``name`` in ``dirs``. A provisional record
    (an earlier run's copy standing in for an autotune still to run, as in
    a run stopped before its Triton phase) is no run's result: reused, it
    would stay provisional, and the campaign would autotune it all the same."""
    for directory in dirs:
        path = directory / name
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("status") == "ok" and not record.get("provisional"):
            return path, record
    return None, None


def pinned_code_matches(configs_dir: Path, repro: str, shape: str) -> dict[str, bool]:
    """``{kernel: whether it generates its recorded code}`` for the Triton
    port pinned to its autotune record in ``configs_dir``, after one forward
    on the point's inputs."""
    port, record = common.load_autotuned_port(configs_dir, repro, shape, "triton")
    with torch.no_grad():
        port.forward(common.make_inputs(repro, shape))
    torch.cuda.synchronize()
    return common.pin_roundtrip(port, record["kernels"])


def _cuda_broken() -> bool:
    """Whether a failed forward left a sticky CUDA error, which would fail
    every later check too."""
    if not torch.cuda.is_initialized():
        return False
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        return True
    return False


def triton_reuse(
    path: Path,
    record: dict[str, Any],
    *,
    reuse_unsampled: bool,
    pin_check: bool,
    regime: str | None = None,
) -> str | None:
    """Why the Triton autotune record at ``path`` cannot be reused, or None.
    A record of a port with several kernels stands only for the regime its
    searches ran in (``common.record_regime``: ``alone`` for every record
    made before ``autotune.py`` named one); a port with one kernel is tuned
    alone in every regime. ``regime`` is the one the run's Triton autotunes
    will run in (None: Triton's default, ``common.default_regime``, which is
    ``alone``, so the records of earlier runs stand)."""
    if issue := common.activity_issue(record, reuse_unsampled):
        return issue
    regime = regime or common.default_regime("triton")
    if len(record.get("kernels", ())) > 1 and common.record_regime(record) != regime:
        return f"tuned in the regime {common.record_regime(record)!r}, not {regime!r}"
    if not pin_check:
        return "pinned code not checked"
    try:
        matches = pinned_code_matches(path.parent, record["repro"], record["shape"])
    except Exception as error:
        message = next(iter(str(error).splitlines()), "")
        if _cuda_broken():
            raise RuntimeError(
                f"the pinned port of {path.name} left a sticky CUDA error, so "
                f"no later point can be checked: {message}"
            ) from error
        if isinstance(error, helion.exc.InvalidConfig):
            return f"config invalid at this tree: {message}"
        return f"pinned port failed: {type(error).__name__}: {message}"
    stale = sorted(name for name, same in matches.items() if not same)
    if stale:
        return f"other code at this tree: {', '.join(stale)}"
    return None


def plan(
    selection: dict[str, Any],
    priors: list[Path],
    *,
    reuse_unsampled: bool = False,
    pin_check: bool = True,
    provisional: bool = False,
    regime: str | None = None,
) -> dict[str, Any]:
    """The jobs of a run (autotunes with the reason for each, reused Triton
    records and benches), each job with an estimated duration in seconds.
    With ``provisional``, a Triton autotune names the earlier record to
    stand in for it until it has run."""
    configs = [run / "configs" for run in priors]
    benches = [run / "bench" for run in priors]
    autotunes: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    bench_jobs: list[dict[str, Any]] = []

    def estimate(record: dict[str, Any] | None) -> float | None:
        return None if record is None else record.get("wall_s")

    for entry in selection["kernels"]:
        repro, shape = entry["repro"], entry["shape"]
        for backend in ("cute", "triton"):
            name = common.config_record_path(Path(), repro, shape, backend).name
            path, record = newest_record(configs, name)
            job = {"repro": repro, "shape": shape, "backend": backend}
            if backend == "cute":
                reason = "every CuTe point is autotuned"
            elif path is None or record is None:
                reason = "no ok earlier record"
            else:
                reason = triton_reuse(
                    path,
                    record,
                    reuse_unsampled=reuse_unsampled,
                    pin_check=pin_check,
                    regime=regime,
                )
                if reason is None:
                    reused.append({**job, "record": str(path)})
                    continue
                if provisional:
                    job["provisional"] = str(path)
            autotunes.append({**job, "reason": reason, "estimate_s": estimate(record)})
        name = common.bench_record_path(Path(), repro, shape).name
        bench_jobs.append(
            {
                "repro": repro,
                "shape": shape,
                "estimate_s": estimate(newest_record(benches, name)[1]),
            }
        )
    # A job without an earlier record of its own takes the median of its kind.
    for kind in (
        [job for job in autotunes if job["backend"] == "cute"],
        [job for job in autotunes if job["backend"] == "triton"],
        bench_jobs,
    ):
        known = [job["estimate_s"] for job in kind if job["estimate_s"] is not None]
        for job in kind:
            if job["estimate_s"] is None:
                job["estimate_s"] = statistics.median(known) if known else 0.0
                job["estimate_from"] = "median"

    def hours(jobs: list[dict[str, Any]]) -> float:
        return sum(job["estimate_s"] for job in jobs) / 3600

    triton = [job for job in autotunes if job["backend"] == "triton"]
    estimated = {
        "cute_autotune": hours([job for job in autotunes if job["backend"] == "cute"]),
        "triton_autotune": hours(triton),
        "bench": hours(bench_jobs),
    }
    estimated["total"] = sum(estimated.values())
    if provisional:
        # In two phases (campaign.py --backends cute, then triton), the
        # second benches each point with a Triton autotune again.
        tuned_again = {(job["repro"], job["shape"]) for job in triton}
        rebench = hours(
            [b for b in bench_jobs if (b["repro"], b["shape"]) in tuned_again]
        )
        estimated["phase1"] = estimated["cute_autotune"] + estimated["bench"]
        estimated["phase2"] = estimated["triton_autotune"] + rebench
        estimated["total"] += rebench
    return {
        "tree": common.git_revision(common.REPO_ROOT),
        "priors": [str(run) for run in priors],
        # The regime the plan reused Triton records for: the run's campaign
        # must tune its Triton ports in it (``campaign.py`` checks).
        "triton_regime": regime or common.default_regime("triton"),
        "autotune": autotunes,
        "reuse": reused,
        "bench": bench_jobs,
        "estimated_gpu_hours": {
            key: round(value, 2) for key, value in estimated.items()
        },
    }


def summary(result: dict[str, Any]) -> list[str]:
    """The plan's counts, the reason for each Triton autotune and the
    estimated GPU-hours."""
    autotunes = result["autotune"]
    triton = [job for job in autotunes if job["backend"] == "triton"]
    hours = result["estimated_gpu_hours"]
    lines = [
        (
            f"autotunes: {len(autotunes)} ({len(autotunes) - len(triton)} CuTe, "
            f"{len(triton)} Triton); Triton records reused: {len(result['reuse'])}"
        ),
        *(
            f"  Triton {job['repro']}__{job['shape']}: {job['reason']}"
            for job in triton
        ),
        f"benches: {len(result['bench'])}",
        (
            f"estimated GPU-hours: {hours['cute_autotune']} CuTe autotune + "
            f"{hours['triton_autotune']} Triton autotune + {hours['bench']} bench"
        ),
    ]
    if "phase1" in hours:
        provisional = sum(1 for job in triton if "provisional" in job)
        lines += [
            (
                "provisional Triton configs for the benches before their "
                f"autotunes: {provisional}"
            ),
            (
                f"in two phases: {hours['phase1']} (campaign.py --backends cute: "
                f"CuTe autotunes and benches), then {hours['phase2']} (--backends "
                "triton: Triton autotunes and their points benched again)"
            ),
        ]
    lines.append(f"total: {hours['total']} GPU-hours")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selection", type=Path, default=common.SELECTION_PATH)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--prior",
        type=Path,
        action="append",
        default=[],
        help="an earlier run directory, newest first (repeat the option)",
    )
    parser.add_argument(
        "--reuse-unsampled",
        action="store_true",
        help="reuse Triton records made before foreign activity was sampled",
    )
    parser.add_argument(
        "--no-pin-check",
        action="store_true",
        help="check no pinned code (no GPU needed): every point is autotuned",
    )
    parser.add_argument(
        "--provisional-triton",
        action="store_true",
        help="copy the earlier record of each Triton point to autotune again into "
        "the run as a provisional config, for the benches that run before the "
        "autotune (campaign.py --backends cute, then --backends triton)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write <run>/plan.json and copy the reused records into the run",
    )
    parser.add_argument(
        "--regime",
        choices=common.REGIMES,
        default=None,
        help="the regime the run's autotunes will time their candidates in "
        "(campaign.py --regime): an earlier Triton record of a port with "
        "several kernels is reused only when it was tuned in it (default: "
        "Triton's own, alone)",
    )
    args = parser.parse_args()
    # A record already there would pass for a finished job of the plan.
    if args.write and any(
        any((args.run_dir / kind).glob("*.json")) for kind in ("configs", "bench")
    ):
        parser.error(f"{args.run_dir} already holds records; plan a new run")
    # The campaign's jobs inherit no HELION_* settings (campaign.job_env);
    # neither does the code they are checked against.
    for key in [key for key in os.environ if key.startswith("HELION_")]:
        del os.environ[key]
    result = plan(
        json.loads(args.selection.read_text()),
        args.prior,
        reuse_unsampled=args.reuse_unsampled,
        pin_check=not args.no_pin_check,
        provisional=args.provisional_triton,
        regime=args.regime,
    )
    result["selection"] = str(args.selection)
    if args.write:
        configs = args.run_dir / "configs"
        configs.mkdir(parents=True, exist_ok=True)
        for job in result["reuse"]:
            source = Path(job["record"])
            record = json.loads(source.read_text())
            record["reused_from"] = str(source)
            common.write_json(configs / source.name, record)
        for job in result["autotune"]:
            if "provisional" in job:
                source = Path(job["provisional"])
                record = json.loads(source.read_text())
                record.update(provisional=True, reused_from=str(source))
                common.write_json(configs / source.name, record)
        common.write_json(args.run_dir / "plan.json", result)
    print("\n".join(summary(result)))


if __name__ == "__main__":
    main()
