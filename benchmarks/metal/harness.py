"""Cell execution, measurement discipline, calibration and A/B runs."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import datetime
import fcntl
import gc
import hashlib
import importlib
from itertools import starmap
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten

from . import manifest
from . import timing
from .correctness import DASHBOARD_TOL
from .correctness import Check
from .correctness import compare
from .inputs import generator
from .inputs import input_hash
from .providers import bind
from .providers import floor_call
from .providers import helion_evidence
from .providers import mlx_available
from .providers import runtime_of
from .results import Results
from .results import block_medians
from .results import block_stats
from .results import cell_key
from .results import write_reports

if TYPE_CHECKING:
    from types import ModuleType

    from .manifest import Case
    from .manifest import Kernel
    from .providers import Bound

REPO_ROOT = Path(__file__).resolve().parents[2]
# A change is real when it exceeds this multiple of the calibrated noise.
NOISE_MULTIPLIER = 2.0
# Steps that belong to the whole cell rather than to one provider; a process
# death in one of them ends the cell instead of being retried.
CELL_STEPS = ("setup", "sync_floor")


def bench_home() -> Path:
    """Machine-scoped state: the lock and the noise calibration."""
    return Path(
        os.environ.get(
            "HELION_METAL_BENCH_HOME", Path.home() / ".cache" / "helion-metal-bench"
        )
    )


@dataclass
class Options:
    providers: tuple[str, ...]
    blocks: int
    correctness_only: bool
    order_seed: int


def load_op(kernel: Kernel) -> ModuleType:
    return importlib.import_module(f"benchmarks.metal.ops.{kernel.module}")


def classify(error: BaseException) -> str:
    from helion import exc

    message = str(error).lower()
    if isinstance(error, torch.OutOfMemoryError) or "out of memory" in message:
        return "oom"
    if isinstance(
        error,
        (exc.BackendUnsupported, exc.BackendImplementationMissing, NotImplementedError),
    ):
        return "unsupported"
    if isinstance(error, (exc.Base, torch._dynamo.exc.TorchDynamoException)):
        return "compile_error"
    return "runtime_error"


def first_line(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".strip()
    return text.splitlines()[0][:400]


def output_bytes(tree: object) -> int:
    leaves, _ = tree_flatten(tree)
    return sum(t.numel() * t.element_size() for t in leaves)


def check_dict(check: Check | None) -> dict[str, Any] | None:
    return None if check is None else vars(check)


def leaf_dtypes(tree: object) -> list[torch.dtype]:
    leaves, _ = tree_flatten(tree)
    return [t.dtype for t in leaves]


def expected_dtypes(op: ModuleType, host: dict[str, torch.Tensor]) -> list[torch.dtype]:
    """Eager's output dtypes, from a run on fake MPS tensors.

    Costs no memory and needs no eager success, so every provider's output
    dtypes are checked even in a cell where eager itself fails.
    """
    with FakeTensorMode():
        fake = {
            name: torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="mps")
            for name, t in host.items()
        }
        return leaf_dtypes(bind("torch_eager", op, host, {"mps": fake}).call())


def _new_cell(kernel: Kernel, case: Case) -> dict[str, Any]:
    return {
        "kernel": kernel.name,
        "index": case.index,
        "label": case.label,
        "params": case.params,
        "state": "running",
        "in_progress": None,
        "providers": {},
        "block_orders": [],
        "sync_floor_ms": {},
    }


def _resume_cell(cell: dict[str, Any], death: str) -> None:
    """Restart an unfinished cell; the step the process ``death`` hit fails.

    ``death`` is ``"crash"`` or ``"timeout"``.  A death in a cell step ends
    the cell untimed; a death in a provider's step is that provider's status,
    and every other provider is checked again, so a resumed run never retries
    the step that killed it.
    """
    step = cell["in_progress"]
    if step in CELL_STEPS:
        cell.update(
            state="done",
            in_progress=None,
            error=f"process {death} during {step} in an earlier run",
        )
        return
    kept = {
        name: record
        for name, record in cell["providers"].items()
        if record["status"] in ("crash", "timeout")
    }
    if step is not None:
        kept[step.split(":")[0]] = {
            "status": death,
            "error": f"process {death} during {step} in an earlier run",
        }
    cell.update(providers=kept, in_progress=None, block_orders=[], sync_floor_ms={})


def release_memory() -> None:
    """Return freed device memory to the system.

    torch and MLX cache freed buffers in separate allocators; one runtime's
    cache would otherwise squeeze the other's next allocation.
    """
    gc.collect()
    torch.mps.empty_cache()
    if "mlx.core" in sys.modules:
        import mlx.core as mx

        mx.clear_cache()


def run_cell(
    results: Results, kernel: Kernel, case: Case, options: Options, death: str
) -> None:
    key = cell_key(kernel.name, case.index)
    cell = results.cells.get(key)
    if cell is None:
        cell = results.cells[key] = _new_cell(kernel, case)
    elif cell["state"] != "done":
        _resume_cell(cell, death)
        results.drop_samples(key)
    if cell["state"] == "done":
        results.save()
        return
    device: dict[str, Any] = {}
    bounds, sizes = _check_providers(results, cell, kernel, case, options, device)
    if not options.correctness_only and bounds:
        _time_cell(results, cell, key, bounds, sizes, options)
    cell["state"] = "done"
    cell["in_progress"] = None
    results.save()
    # Free this cell's device buffers before handing the caches back.
    del bounds, device
    release_memory()


def _check_providers(
    results: Results,
    cell: dict[str, Any],
    kernel: Kernel,
    case: Case,
    options: Options,
    device: dict[str, Any],
) -> tuple[dict[str, Bound], dict[str, int]]:
    """Build and check every provider; return the correct ones for timing.

    The host inputs, the reference and the eager output live only here, so
    they are freed before timing starts; ``device`` keeps the converted
    inputs the bound providers need.  A failure building the inputs or the
    reference fails the cell, not the run.
    """
    cell["in_progress"] = "setup"
    results.save()
    try:
        op = load_op(kernel)
        host = op.inputs(case, generator(case))
        cell["input_hash"] = input_hash(host)
        expected = op.reference(*host.values())
        dtypes = expected_dtypes(op, host)
    except Exception as error:
        cell.update(state="done", error=f"setup failed: {first_line(error)}")
        return {}, {}
    # The dashboard verdict compares against eager, as CI does.  If eager
    # fails, its own check below records why and the cell has no verdict.  A
    # death here is eager's, so a resumed run that recorded one skips it.
    eager_host = None
    if "torch_eager" not in cell["providers"]:
        cell["in_progress"] = "torch_eager:dashboard"
        results.save()
        with contextlib.suppress(Exception):
            eager = bind("torch_eager", op, host, device)
            eager_host = eager.to_host(eager.call())
    bounds: dict[str, Bound] = {}
    sizes: dict[str, int] = {}
    for provider in options.providers:
        if provider in cell["providers"]:
            continue
        if runtime_of(provider) == "mlx" and not mlx_available():
            cell["providers"][provider] = {
                "status": "unavailable",
                "error": "mlx is not installed",
            }
            continue
        release_memory()
        cell["in_progress"] = f"{provider}:build"
        results.save()
        rtol, atol = manifest.tolerance(kernel, provider, case.dtype)
        record, bound, size = _build_and_check(
            provider, op, host, device, expected, eager_host, dtypes, rtol, atol
        )
        cell["providers"][provider] = record
        if bound is not None and record["status"] == "ok":
            bounds[provider] = bound
            sizes[provider] = size
    cell["in_progress"] = None
    results.save()
    return bounds, sizes


def _build_and_check(
    provider: str,
    op: ModuleType,
    host: dict[str, torch.Tensor],
    device: dict[str, Any],
    expected: object,
    eager_host: object,
    dtypes: list[torch.dtype],
    rtol: float,
    atol: float,
) -> tuple[dict[str, Any], Bound | None, int]:
    """Bind, run once (compiling and autotuning), and check correctness.

    Returns the record, the bound call if it is correct, and its output size.
    """
    cache = _helion_cache_files() if provider == "helion" else set()
    t0 = time.perf_counter()
    try:
        bound = bind(provider, op, host, device)
        out = bound.call()
        bound.complete(out)
        host_out = bound.to_host(out)
    except Exception as error:
        return {"status": classify(error), "error": first_line(error)}, None, 0
    record: dict[str, Any] = {"setup_s": time.perf_counter() - t0, "error": None}
    if provider == "helion":
        kernels = helion_evidence(op)
        record["evidence"] = {
            "kernels": kernels,
            "config_source": _config_source(kernels, _helion_cache_files() - cache),
        }
        missing = {k.name for k in op.HELION_KERNELS} - {k["kernel"] for k in kernels}
        other = sorted({k["backend"] for k in kernels} - {"metal"})
        if not kernels or missing or other:
            record.update(
                status="runtime_error",
                error=(
                    "not every Helion kernel ran on Metal "
                    f"(not run: {sorted(missing)}, other backends: {other})"
                ),
            )
            return record, None, 0
    strict = _compare(host_out, expected, rtol, atol, dtypes)
    dashboard = (
        None
        if eager_host is None
        else _compare(host_out, eager_host, DASHBOARD_TOL, DASHBOARD_TOL, dtypes)
    )
    record["strict"] = check_dict(strict)
    record["dashboard"] = check_dict(dashboard)
    record["status"] = "ok" if strict.passed else "wrong_result"
    if not strict.passed:
        record["error"] = strict.reason
        return record, None, 0
    return record, bound, output_bytes(host_out)


def _config_source(kernels: list[dict[str, str]], new_cache_files: set[str]) -> str:
    """Where the Helion configs came from: default, tuned now, or the cache."""
    if all(k["autotune_effort"] == "none" for k in kernels):
        return "default"
    return "tuned" if new_cache_files else "cache"


def _compare(
    actual: object,
    expected: object,
    rtol: float,
    atol: float,
    dtypes: list[torch.dtype],
) -> Check:
    got = leaf_dtypes(actual)
    if got != dtypes:
        return Check(False, f"output dtypes {got} != {dtypes}")
    return compare(actual, expected, rtol, atol)


def _helion_cache_files() -> set[str]:
    from helion.autotuner.local_cache import get_helion_cache_dir

    directory = get_helion_cache_dir()
    if not directory.exists():
        return set()
    return {p.name for p in directory.glob("*.best_config")}


def _time_cell(
    results: Results,
    cell: dict[str, Any],
    key: str,
    bounds: dict[str, Bound],
    sizes: dict[str, int],
    options: Options,
) -> None:
    cell["in_progress"] = "sync_floor"
    results.save()
    for runtime in sorted({runtime_of(p) for p in bounds}):
        cell["sync_floor_ms"][runtime] = timing.sync_floor(*floor_call(runtime))
    rng = random.Random(f"{options.order_seed}:{key}")
    live = dict(bounds)
    # Compiled providers must not recompile while being timed.
    torch.compiler.set_stance("fail_on_recompile")
    try:
        for block in range(options.blocks):
            order = list(live)
            rng.shuffle(order)
            cell["block_orders"].append(order)
            for provider in order:
                record = cell["providers"][provider]
                cell["in_progress"] = f"{provider}:timing"
                results.save()
                try:
                    samples = timing.latency(live[provider])
                    batch = timing.kernel_ms(live[provider], sizes[provider])
                except Exception as error:
                    record.update(status=classify(error), error=first_line(error))
                    for field in ("latency", "kernel_ms", "enqueue_ms"):
                        record.pop(field, None)
                    del live[provider]
                    continue
                record.setdefault("latency", []).append(block_stats(samples))
                record.setdefault("kernel_ms", []).append(batch.kernel_ms)
                record.setdefault("enqueue_ms", []).append(batch.enqueue_ms)
                record["kernel_ms_source"] = batch.source
                results.append_samples(
                    {"cell": key, "provider": provider, "block": block, "ms": samples}
                )
    finally:
        torch.compiler.set_stance("default")


# --- environment and preflight -------------------------------------------


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


# The code a run depends on.  Untracked files elsewhere, such as the run's own
# output inside the checkout, are not part of it.
SOURCE_PATHS = ("helion", "examples", "benchmarks")


def source_state(exclude: tuple[Path, ...]) -> str | None:
    """A hash of uncommitted edits to the sources, or None when they are clean.

    Covers tracked changes and the contents of untracked files, except those
    under ``exclude``.
    """
    git = ["git", "-C", str(REPO_ROOT)]
    diff = _run([*git, "diff", "HEAD", "--", *SOURCE_PATHS])
    listed = _run(
        [*git, "ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS]
    )
    untracked = sorted(
        path
        for path in (REPO_ROOT / name for name in listed.splitlines())
        # Directory entries are untracked nested repos, not sources.
        if path.is_file() and not any(path.is_relative_to(e) for e in exclude)
    )
    if not diff and not untracked:
        return None
    digest = hashlib.sha256(diff.encode())
    for path in untracked:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def environment(exclude: tuple[Path, ...] = ()) -> dict[str, Any]:
    if mlx_available():
        import mlx.core as mx

        mlx_version = mx.__version__  # pyrefly: ignore[missing-attribute]
    else:
        mlx_version = None
    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]).strip()
    state = source_state(exclude)
    return {
        "chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
        "memory_gb": int(_run(["sysctl", "-n", "hw.memsize"]).strip()) / 2**30,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mlx": mlx_version,
        "helion_commit": commit[:12] + ("" if state is None else "-dirty"),
        # Identifies uncommitted edits, so a resumed run can tell they changed.
        "helion_diff": state,
    }


def identity(env: dict[str, Any]) -> dict[str, Any]:
    """What a noise calibration is valid for."""
    return {k: env[k] for k in ("chip", "macos", "torch", "mlx")}


def preflight() -> dict[str, Any]:
    """Machine state, plus the reasons a measurement run should not start."""
    therm = _run(["pmset", "-g", "therm"])
    power = _run(["pmset", "-g"])
    battery = _run(["pmset", "-g", "batt"])
    problems = []
    for line in therm.splitlines():
        lowered = line.lower()
        if ("limit" in lowered and "100" not in lowered) or (
            "warning level" in lowered and "no " not in lowered
        ):
            problems.append(f"thermal: {line.strip()}")
    for line in power.splitlines():
        parts = line.split()
        if parts[:1] == ["lowpowermode"] and parts[1:2] == ["1"]:
            problems.append("Low Power Mode is on")
    if "Battery Power" in battery:
        problems.append("running on battery")
    busy = []
    for line in _run(["ps", "-A", "-o", "pid=,%cpu=,comm="]).splitlines():
        pid, cpu, command = line.split(None, 2)
        if int(pid) != os.getpid() and float(cpu) >= 50:
            busy.append(f"{command.strip()} ({cpu}%)")
    if busy:
        problems.append("busy processes: " + ", ".join(busy))
    return {"therm": therm.strip(), "battery": battery.strip(), "problems": problems}


@contextlib.contextmanager
def run_lock() -> Iterator[None]:
    """One benchmark run per machine at a time."""
    home = bench_home()
    home.mkdir(parents=True, exist_ok=True)
    with open(home / "run.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                f"another benchmark run holds {home / 'run.lock'}"
            ) from None
        yield


# --- runs ------------------------------------------------------------------


def cells_by_key(keys: list[str]) -> list[tuple[Kernel, Case]]:
    return [
        manifest.case(name, int(index)) for name, index in (k.split("/") for k in keys)
    ]


def selected_cases(
    kernels: list[str] | None, cases: list[int] | None
) -> list[tuple[Kernel, Case]]:
    chosen = []
    for kernel in manifest.KERNELS:
        if kernels is not None and kernel.name not in kernels:
            continue
        for case in kernel.cases:
            if cases is None or case.index in cases:
                chosen.append((kernel, case))
    return chosen


def sentinel_ms() -> float:
    """Eager latency of the fixed drift-sentinel cell."""
    kernel, case = manifest.sentinel()
    op = load_op(kernel)
    bound = bind("torch_eager", op, op.inputs(case, generator(case)), {})
    return timing.median(timing.latency(bound))


def run_header(args: dict[str, Any], options: Options, out_dir: Path) -> dict[str, Any]:
    cache = Path(os.environ["HELION_CACHE_DIR"])
    state = preflight()
    return {
        "manifest": {"version": manifest.VERSION, "hash": manifest.manifest_hash()},
        "environment": environment((out_dir.resolve(), cache.resolve())),
        "preflight": state,
        "run": {
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "args": args,
            "providers": list(options.providers),
            "blocks": options.blocks,
            "order_seed": options.order_seed,
            "correctness_only": options.correctness_only,
            "autotune_effort": os.environ.get("HELION_AUTOTUNE_EFFORT", "full"),
            "autotune_seed": os.environ["HELION_AUTOTUNE_RANDOM_SEED"],
            "helion_cache_dir": os.environ["HELION_CACHE_DIR"],
            "force": bool(state["problems"]),
        },
    }


# Arguments that may differ between a run and its resumption.
RESUMABLE_ARGS = ("resume", "force", "supervised", "last_death", "stall_timeout")


def _same_run(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """How a resumed run differs from the one that wrote ``stored``."""
    problems = []
    if stored["manifest"] != current["manifest"]:
        problems.append("the manifest or operator code changed")
    for key in (*identity(current["environment"]), "helion_commit", "helion_diff"):
        if stored["environment"][key] != current["environment"][key]:
            problems.append(f"environment {key} changed")
    old, new = (
        {k: v for k, v in run["run"]["args"].items() if k not in RESUMABLE_ARGS}
        for run in (stored, current)
    )
    changed = {
        k: (old.get(k), new.get(k))
        for k in old.keys() | new.keys()
        if old.get(k) != new.get(k)
    }
    if changed:
        problems.append(f"arguments changed (old, new): {changed}")
    return problems


def snapshot(
    out_dir: Path,
    chosen: list[tuple[Kernel, Case]],
    options: Options,
    header: dict[str, Any],
    resume: bool,
    last_death: str | None,
    measure_drift: bool,
) -> Results:
    """Run every chosen cell, continuing ``out_dir`` when ``resume`` is set.

    ``last_death`` is how the supervisor saw the previous process end
    (``"crash"`` or ``"timeout"``; None for a resume by hand, which counts as
    a crash); the step it was in gets that status.
    """
    death = last_death or "crash"
    if resume and (out_dir / "results.json").exists():
        results = Results.load(out_dir)
        problems = _same_run(results.data, header)
        if problems:
            raise SystemExit(
                f"cannot resume {out_dir}: "
                + "; ".join(problems)
                + " (start a new run)"
            )
        results.data["run"].setdefault("resumes", []).append(
            {
                "at": datetime.datetime.now().isoformat(timespec="seconds"),
                "after": last_death,
                "preflight_problems": header["preflight"]["problems"],
            }
        )
    else:
        results = Results.create(out_dir, header)
    run = results.data["run"]
    if measure_drift:
        _measure_sentinel(results, "start", death)
    for i, (kernel, case) in enumerate(chosen):
        print(
            f"[{i + 1}/{len(chosen)}] {kernel.name} case {case.index} {case.label}",
            file=sys.stderr,
            flush=True,
        )
        run_cell(results, kernel, case, options, death)
        write_reports(results)
    if measure_drift:
        _measure_sentinel(results, "end", death)
    if "sentinel_end_ms" in run and "sentinel_start_ms" in run:
        drift = run["sentinel_end_ms"] / run["sentinel_start_ms"] - 1
        run["drift"] = drift
        calibration = load_calibration(results.data["environment"])
        if calibration is None:
            run["stability"] = f"uncalibrated ({drift:+.1%} drift)"
        else:
            threshold = NOISE_MULTIPLIER * calibration["sentinel_spread"]
            run["stability"] = (
                "stable"
                if abs(drift) <= threshold
                else f"unstable ({drift:+.1%} drift)"
            )
            run["drift_threshold"] = threshold
    run["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    results.save()
    write_reports(results)
    return results


def _measure_sentinel(results: Results, when: str, death: str) -> None:
    """Record the drift sentinel once; a process death while measuring it is
    recorded instead of retried."""
    run = results.data["run"]
    key = f"sentinel_{when}_ms"
    if key in run or f"sentinel_{when}_error" in run:
        return
    if run.get("in_progress") == key:
        run[f"sentinel_{when}_error"] = f"process {death} measuring the sentinel"
        run["in_progress"] = None
        results.save()
        return
    run["in_progress"] = key
    results.save()
    run[key] = sentinel_ms()
    run["in_progress"] = None
    results.save()


# --- noise calibration -------------------------------------------------------


# Bumped whenever the calibration file's layout changes.
CALIBRATION_FORMAT = 2


def calibration_path() -> Path:
    return bench_home() / "calibration.json"


def calibrate(
    out_dir: Path, worker_args: list[str], blocks: int, order_seed: int
) -> dict[str, Any]:
    """Measure the noise an A/B sees, per mode: A/Bs of HEAD against itself.

    The guard set runs exactly as ``ab`` runs it.  In re-tune mode each side
    tunes its own configs, so the spread includes autotuning; replay mode then
    gives both sides the configs the re-tune pass's base side found.  The
    drift sentinel is measured before, between and after the two passes, so
    its spread covers drift over the calibration's own duration.
    """
    keys = list(starmap(cell_key, manifest.GUARD_SET))
    sides = {"base": "HEAD", "cand": "HEAD"}
    sentinel = [sentinel_ms()]
    spreads: dict[str, dict[str, float]] = {}
    for mode in ("retune", "replay"):
        run_dir = out_dir / mode
        cache = out_dir / "retune" / "helion-cache-base" if mode == "replay" else None
        starts = dict.fromkeys(sides, cache)
        _ab_run(run_dir, sides, starts, worker_args, keys, blocks, order_seed)
        spreads[mode] = {}
        for key in keys:
            records = {
                side: _block_records(run_dir, side, blocks, key) for side in sides
            }
            # A replay pass that re-tuned a cell would measure tuning noise.
            if mode == "replay" and _replay_missed(records):
                continue
            ratios = _paired_ratios(records["base"], records["cand"])
            if ratios:
                spreads[mode][key] = ratio_spread(ratios)
        sentinel.append(sentinel_ms())
    calibration = {
        "format": CALIBRATION_FORMAT,
        "identity": identity(environment()),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "blocks": blocks,
        "spreads": spreads,
        "sentinel_spread": (max(sentinel) - min(sentinel)) / timing.median(sentinel),
    }
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=1))
    return calibration


def ratio_spread(ratios: list[float]) -> float:
    """The largest relative deviation of one block's ratio from 1."""
    return max(*ratios, 1 / min(ratios)) - 1


def load_calibration(env: dict[str, Any]) -> dict[str, Any] | None:
    """The stored calibration, if it was made on this machine and stack."""
    path = calibration_path()
    if not path.exists():
        return None
    calibration = json.loads(path.read_text())
    if calibration.get("format") != CALIBRATION_FORMAT:
        return None
    if calibration["identity"] != identity(env):
        return None
    return calibration


def noise_threshold(
    calibration: dict[str, Any], mode: str, key: str
) -> tuple[float | None, str]:
    """Relative A/B difference in ``mode`` beyond which a change is real.

    A cell without its own calibrated spread uses the median over calibrated
    cells; with none at all there is no threshold.
    """
    spreads = calibration["spreads"][mode]
    if key in spreads:
        return NOISE_MULTIPLIER * spreads[key], f"calibrated {key}"
    if not spreads:
        return None, "no calibrated cell"
    return NOISE_MULTIPLIER * timing.median(list(spreads.values())), "guard-set median"


# --- A/B ----------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _harness_tree(ref: str) -> str:
    return _git("rev-parse", f"{ref}:benchmarks/metal") + _git(
        "rev-parse", f"{ref}:benchmarks/run_metal.py"
    )


def ab_cells(chosen: list[tuple[Kernel, Case]]) -> list[str]:
    """The cells an A/B measures: the chosen ones plus the guard set.

    The guard set's verdicts are the check for collateral damage: a change
    aimed elsewhere should leave them correct and neutral.
    """
    keys = [cell_key(k.name, c.index) for k, c in chosen]
    return keys + [
        key for key in starmap(cell_key, manifest.GUARD_SET) if key not in keys
    ]


def ab(
    out_dir: Path,
    base_ref: str,
    cand_ref: str,
    mode: str,
    replay_cache: Path | None,
    worker_args: list[str],
    keys: list[str],
    blocks: int,
    order_seed: int,
    base_tunes: Path | None = None,
) -> dict[str, Any]:
    """Compare two refs' Helion on ``keys``, in alternating blocks.

    Both refs must carry this exact harness.  In ``replay`` mode both start
    from a copy of ``replay_cache``; in ``retune`` mode each tunes its own,
    except that BASE starts from a copy of ``base_tunes`` if given: configs
    an earlier retune A/B tuned for the same BASE with the same autotune
    seed, so BASE re-tunes only the cells missing there.
    """
    calibration = load_calibration(environment())
    if calibration is None:
        raise SystemExit(
            f"A/B needs a noise calibration for this machine at {calibration_path()}; "
            "run --calibrate first"
        )
    if calibration["blocks"] != blocks:
        raise SystemExit(
            f"the calibration used {calibration['blocks']} blocks; "
            f"run the A/B with --blocks {calibration['blocks']} or recalibrate"
        )
    current = _harness_tree("HEAD")
    for ref in (base_ref, cand_ref):
        if _harness_tree(ref) != current:
            raise SystemExit(f"{ref} carries a different benchmark harness than HEAD")
    sides = {"base": base_ref, "cand": cand_ref}
    starts = {"base": base_tunes} if mode == "retune" else {}
    if mode == "replay":
        starts = dict.fromkeys(sides, replay_cache)
    _ab_run(out_dir, sides, starts, worker_args, keys, blocks, order_seed)
    return ab_verdicts(out_dir, blocks, mode, sides, keys, calibration)


def _ab_run(
    out_dir: Path,
    sides: dict[str, str],
    starts: dict[str, Path | None],
    worker_args: list[str],
    keys: list[str],
    blocks: int,
    order_seed: int,
) -> None:
    """Run ``blocks`` alternating blocks of each side's Helion provider.

    Each block is a supervised worker process started from the side's own
    worktree, so each side imports its own Helion, and a candidate that kills
    or hangs its process is recorded, not fatal.  A cell that timed out on a
    side is not run again on that side.  A side's autotune cache starts as a
    copy of its entry in ``starts``, else empty.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trees = {}
    for side, ref in sides.items():
        tree = out_dir / f"worktree-{side}"
        if tree.exists():
            _git("worktree", "remove", "--force", str(tree))
        _git("worktree", "add", "--detach", str(tree), ref)
        trees[side] = tree
        cache = out_dir / f"helion-cache-{side}"
        shutil.rmtree(cache, ignore_errors=True)
        start = starts.get(side)
        if start is not None:
            shutil.copytree(start, cache)
        else:
            cache.mkdir()
        shutil.rmtree(out_dir / side, ignore_errors=True)
    rng = random.Random(order_seed)
    hung: dict[str, set[str]] = {side: set() for side in sides}
    try:
        for block in range(blocks):
            order = list(sides)
            rng.shuffle(order)
            for side in order:
                cells = [key for key in keys if key not in hung[side]]
                if not cells:
                    continue
                block_dir = out_dir / side / f"block{block}"
                env = {
                    **os.environ,
                    "HELION_CACHE_DIR": str(out_dir / f"helion-cache-{side}"),
                }
                env.pop("PYTHONPATH", None)
                subprocess.run(
                    [
                        sys.executable,
                        str(trees[side] / "benchmarks" / "run_metal.py"),
                        "--worker",
                        "--providers",
                        "helion",
                        "--blocks",
                        "1",
                        "--cells",
                        ",".join(cells),
                        "--output",
                        str(block_dir),
                        *worker_args,
                    ],
                    cwd=trees[side],
                    env=env,
                    check=False,
                )
                if (block_dir / "results.json").exists():
                    hung[side] |= {
                        key
                        for key, cell in Results.load(block_dir).cells.items()
                        if cell["providers"].get("helion", {}).get("status")
                        == "timeout"
                    }
    finally:
        for tree in trees.values():
            _git("worktree", "remove", "--force", str(tree))


def _block_records(
    out_dir: Path, side: str, blocks: int, key: str
) -> list[dict[str, Any]]:
    """One side's Helion record for ``key`` in each block.

    A cell skipped after timing out in an earlier block is a timeout; any
    other block whose worker left no finished cell (a harness error, or a
    death the supervisor could not get past) counts as a crash.
    """
    records: list[dict[str, Any]] = []
    for block in range(blocks):
        path = out_dir / side / f"block{block}"
        cell = None
        if (path / "results.json").exists():
            cell = Results.load(path).cells.get(key)
        if (
            cell is not None
            and cell["state"] == "done"
            and "helion" in cell["providers"]
        ):
            records.append(cell["providers"]["helion"])
        elif any(r["status"] == "timeout" for r in records):
            records.append(
                {"status": "timeout", "error": "skipped after an earlier timeout"}
            )
        else:
            records.append({"status": "crash", "error": "worker left no result"})
    return records


def _replay_missed(records: dict[str, list[dict[str, Any]]]) -> list[str]:
    """The sides whose configs did not all come from the replayed cache."""
    return [
        side
        for side, recs in records.items()
        if any(
            r["evidence"]["config_source"] != "cache"
            for r in recs
            if r["status"] == "ok"
        )
    ]


def _paired_ratios(
    base: list[dict[str, Any]], cand: list[dict[str, Any]]
) -> list[float]:
    """Per-block base / candidate latency, when every block of both timed."""
    if not all(r.get("latency") for r in (*base, *cand)):
        return []
    return [
        block_medians(b)[0] / block_medians(c)[0]
        for b, c in zip(base, cand, strict=True)
    ]


def ab_verdicts(
    out_dir: Path,
    blocks: int,
    mode: str,
    sides: dict[str, str],
    keys: list[str],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Per cell: correctness over every block, and a timing verdict.

    A side is correct only if it is ok in every block, so a flaky candidate
    regresses.  In replay mode a block that did not take its configs from the
    cache gets ``replay_miss`` instead of a verdict: a re-tuned side would be
    compared with a replayed one.
    """
    cells = {}
    for key in keys:
        records = {side: _block_records(out_dir, side, blocks, key) for side in sides}
        statuses = {side: [r["status"] for r in recs] for side, recs in records.items()}
        base_ok, cand_ok = (all(s == "ok" for s in statuses[side]) for side in sides)
        entry: dict[str, Any] = {
            "base_status": statuses["base"],
            "cand_status": statuses["cand"],
            "correctness": (
                "regressed"
                if base_ok and not cand_ok
                else "fixed"
                if cand_ok and not base_ok
                else "unchanged"
            ),
        }
        cells[key] = entry
        if mode == "replay":
            misses = _replay_missed(records)
            if misses:
                entry.update(verdict="replay_miss", replay_miss=misses)
                continue
        ratios = _paired_ratios(records["base"], records["cand"])
        if ratios:
            speedup = timing.median(ratios)
            threshold, source = noise_threshold(calibration, mode, key)
            entry.update(
                speedup=speedup,
                ratios=ratios,
                threshold=threshold,
                threshold_source=source,
                verdict="uncalibrated"
                if threshold is None
                else verdict(speedup, threshold),
            )
    report = {
        "base": sides["base"],
        "cand": sides["cand"],
        "mode": mode,
        "cells": cells,
    }
    (out_dir / "ab.json").write_text(json.dumps(report, indent=1))
    return report


def verdict(speedup: float, threshold: float) -> str:
    """Candidate vs base, where speedup = base latency / candidate latency."""
    if speedup > 1 + threshold:
        return "better"
    if speedup < 1 / (1 + threshold):
        return "worse"
    return "neutral"
