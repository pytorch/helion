#!/usr/bin/env python3
"""Run Helion's test lanes locally the way CI does, sharded across GPUs.

CI (.github/workflows/test.yml) runs the suite as lanes: the CuTe GPU jobs run
every test except the ``cpu_only`` modules (``-m "not cpu_only or cuda"``),
split across runners with ``HELION_TEST_SHARD=i/n`` and balanced by
test/shard_durations.json; a CPU runner runs those modules without a GPU
(``-m "cpu_only and not cuda"`` with ``HELION_TEST_LANE=cpu``); the other
backends' GPU jobs run the whole suite in one job; pull requests skip the
``slow`` tier. This script runs the same split here for any backend: the GPU
lane as one pytest per GPU, the CPU lane beside it without a GPU (together
they are the whole suite, so ``--lane all`` matches every CI job; ``--lane
gpu`` alone leaves out the cpu_only modules CI's non-CuTe jobs run), the PR
filter unless ``--slow``, and afterwards every failure once more, alone in a
fresh process, so order-dependent failures and flakes are told apart from
real ones.

``--multi-gpu`` instead runs one serial pytest process with all claimed GPUs
visible, for tests that launch their own distributed workers. It requires one
backend and ``--lane gpu``; both the initial job and isolation reruns retain
the full device set without test sharding.

    scripts/run_tests.py --backend cute --gpus 2,5       # CuTe lanes; GPU lane over two GPUs
    scripts/run_tests.py --backend triton --lane cpu     # Triton CPU lane only; no GPU needed
    scripts/run_tests.py --backend triton,cute -n 2      # both backends on 2 GPUs from the board
    scripts/run_tests.py --backend triton --lane gpu --multi-gpu -n 4 test/test_remote_copy.py::TestRemoteCopyGPU
    scripts/run_tests.py --changed                       # the modules touching the stack's files
    scripts/run_tests.py --changed --since HEAD~1        # ... touched by the last commit
    scripts/run_tests.py --slow test/test_autotuner.py   # one module, slow tier included
    scripts/run_tests.py --backend cute -- -k softmax    # extra pytest arguments after --

GPUs come from ``--gpus``, else from ``CUDA_VISIBLE_DEVICES``, else
``scripts/gpu_board.py`` picks ``-n`` free ones; the run holds a claim on the
board while it lasts (on a positively identified single-GPU machine, which is not
shared, the board is skipped and that GPU is used). GPU discovery requires
nvidia-smi. Failed or empty discovery aborts an invocation requesting GPU lanes
before any CPU or GPU test job starts; use ``--lane cpu`` for checks that do not
need NVIDIA discovery. GPUs named explicitly are refused when another
account holds a live claim on them, they are busy, or another account's
processes sit on them (``--shared`` allows idle ones, ``--force`` any). Test
subprocesses get this tree and its pytest-plugin directory on PYTHONPATH, so a
worktree tests its own helion and loads the runner's node-identity plugin.
A ``-m`` among the pytest options is combined with the lane's marker. Logs, junit
reports, the exact commands and a summary go to
``.logs/tests/<timestamp>-<tag>/``; the junit files can feed
``scripts/shard_durations.py``. Exit status: 0 when every test passed, 1 when
a failure reproduces alone or a job crashes, 2 when every failure passed
alone, 3 when nothing was selected, 4 on a GPU board security alert, 130 when
interrupted.

Explicit test paths are checked relative to this checkout before any job starts.
Pytest node-id suffixes are preserved; ``--pyargs`` package names are resolved
by pytest. Missing paths are errors, not empty backend selections.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import inspect
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
import xml.etree.ElementTree as ET

# scripts/gpu_board.py; run as a script, Python puts scripts/ on sys.path.
import gpu_board
from process_supervisor import CleanupError
from process_supervisor import OwnedProcess
from process_supervisor import ProcessError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
LOGS_DIR = REPO_ROOT / ".logs" / "tests"
TEST_DIR = "test"

# The lane and tier markers of .github/workflows/test.yml; test/test_agent_scripts.py
# checks that the workflow still uses these exact expressions.
GPU_LANE_MARK = "not cpu_only or cuda"
CPU_LANE_MARK = "cpu_only and not cuda"
PR_TIER_MARK = "not slow"

# Serial backends (CI runs them with -n0) and their environment.
SERIAL_BACKENDS = {"pallas-interpret", "metal"}
BACKEND_ENV = {
    "pallas-interpret": {"HELION_BACKEND": "pallas", "HELION_PALLAS_INTERPRET": "1"},
    "tileir": {"HELION_BACKEND": "tileir", "ENABLE_TILE": "1"},
}
# Variables this script sets per job; whatever the shell had in them must not
# leak into a job (a stale HELION_TEST_SHARD would silently halve a run).
RUNNER_ENV = (
    "HELION_BACKEND",
    "HELION_TEST_SHARD",
    "HELION_TEST_LANE",
    "HELION_PALLAS_INTERPRET",
    "ENABLE_TILE",
    "CUDA_VISIBLE_DEVICES",
)
# The CuTe shards and the slow runners get this in CI; under local contention
# every lane does.
TEST_TIMEOUT = 180
SERIAL_TIMEOUT = 300
RERUN_TIMEOUT = 300
# More failures than this are not a flake pattern; skip the isolation reruns.
MAX_RERUNS = 40
# A changed source module whose name appears in more than this share of the
# test modules is shared infrastructure: run the lanes, not a selection.
BROAD_FRACTION = 0.4
IGNORED_PATHS = ("test/test_examples_dist.py",)
CLAIM_TTL = gpu_board.parse_duration(gpu_board.CLAIM_TTL)


def lane_marker(lane: str, slow: bool, user: str = "") -> str:
    """The ``-m`` expression of a lane, with the PR filter unless ``slow``, and
    the user's own ``-m`` expression (pytest keeps only the last ``-m``, so
    the user's is combined with the lane's rather than passed along)."""
    base = {"gpu": GPU_LANE_MARK, "cpu": CPU_LANE_MARK}.get(lane, "")
    terms = [base, "" if slow else PR_TIER_MARK, user]
    if sum(bool(term) for term in terms) <= 1:
        return base or terms[1] or user
    # The same form as the workflow's: "(<lane>) and not slow".
    parts = [f"({base})" if base else "", terms[1], f"({user})" if user else ""]
    return " and ".join(part for part in parts if part)


@dataclass
class Job:
    name: str
    backend: str
    lane: str  # gpu | cpu | serial
    gpu: str | None
    shard: tuple[int, int] | None
    argv: list[str]
    env: dict[str, str]
    log: Path
    junit: Path
    # Jobs sharing a group run one at a time (the isolation reruns of a lane).
    group: str | None = None


def job_environment(job: Job, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of a job: the caller's, minus the variables this script
    controls, plus the job's own."""
    source = os.environ if base is None else base
    env = {key: value for key, value in source.items() if key not in RUNNER_ENV}
    env.update(job.env)
    # Subprocesses the tests start (python <script>, other cwds) import this
    # tree's helion, not the editable install's: they differ in a worktree.
    # Expose only the flat pytest plugin, not unrelated scripts that could
    # shadow dependencies. An unrelated package named scripts cannot hide it.
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT),
            str(REPO_ROOT / "scripts" / "pytest_plugins"),
            *filter(None, [source.get("PYTHONPATH")]),
        ]
    )
    return env


@dataclass
class Junit:
    counts: dict[str, int]
    outcomes: dict[str, str]  # node id -> passed | failed | error | skipped
    # Modules that failed to import or collect: nothing in them ran.
    collection_errors: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[str]:
        return [
            n for n, outcome in self.outcomes.items() if outcome in ("failed", "error")
        ]


@dataclass
class JobResult:
    job: Job
    returncode: int
    seconds: float
    junit: Junit | None = None

    @property
    def tests(self) -> int:
        return self.junit.counts["tests"] if self.junit else 0

    @property
    def failed(self) -> list[str]:
        return self.junit.failed if self.junit else []

    @property
    def crashed(self) -> bool:
        # 0: passed, 1: failures (listed in the junit), 5: nothing collected.
        return self.returncode not in (0, 1, 5) or (
            self.returncode == 1 and not self.failed
        )

    def describe(self) -> str:
        if self.crashed:
            return f"crashed (exit {self.returncode}), see {self.job.log.name}"
        if self.returncode == 5 or self.junit is None or self.tests == 0:
            return "no tests collected"
        counts = self.junit.counts
        parts = [
            f"{counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']} passed"
        ]
        if counts["failures"]:
            parts.append(f"{counts['failures']} failed")
        if counts["errors"]:
            parts.append(f"{counts['errors']} errors")
        if counts["skipped"]:
            parts.append(f"{counts['skipped']} skipped")
        if self.junit.collection_errors:
            parts.append(f"{len(self.junit.collection_errors)} collection errors")
        return ", ".join(parts)


def lane_env(
    backend: str, lane: str, gpu: str | None, shard: tuple[int, int] | None
) -> dict[str, str]:
    env = {"HELION_BACKEND": backend, "MKL_CBWR": "AVX2"}
    env.update(BACKEND_ENV.get(backend, {}))
    if lane == "gpu":
        assert gpu is not None
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # The indices are nvidia-smi's, which number GPUs in PCI bus order.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if shard is not None and shard[1] > 1:
            env["HELION_TEST_SHARD"] = f"{shard[0]}/{shard[1]}"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        if lane == "cpu":
            env["HELION_TEST_LANE"] = "cpu"
    return env


def pytest_argv(
    junit: Path,
    marker: str,
    paths: Sequence[str],
    extra: Sequence[str],
    serial: bool,
    *,
    timeout: int | None = None,
    overrides_last: bool = False,
    multi_gpu: bool = False,
    basetemp_subdir: str | None = None,
) -> list[str]:
    if timeout is None:
        timeout = SERIAL_TIMEOUT if serial else TEST_TIMEOUT
    timeout_args = [
        f"--timeout={timeout}",
        "--timeout-method=thread",
    ]
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "helion_pytest_nodeids",
        "-rf",
        *(timeout_args if not overrides_last else []),
        "-o",
        "faulthandler_timeout=120",
        "-o",
        "junit_duration_report=total",
    ]
    argv += [f"--ignore={path}" for path in IGNORED_PATHS]
    if serial and not (overrides_last or multi_gpu):
        argv.append("--helion-serial")
    if marker:
        argv += ["-m", marker]
    argv += list(paths) or [TEST_DIR]
    argv += list(extra)
    if serial and (overrides_last or multi_gpu):
        argv.append("--helion-serial")
    if overrides_last:
        argv += timeout_args
    # The runner reads this report to classify the job and select isolation runs.
    if basetemp_subdir is not None:
        argv += ["--helion-basetemp-subdir", basetemp_subdir]
    argv.append(f"--junitxml={junit}")
    return argv


def build_jobs(
    backends: Sequence[str],
    lane: str,
    gpus_by_backend: dict[str, list[str]],
    slow: bool,
    paths: Sequence[str],
    extra: Sequence[str],
    run_dir: Path,
    marker: str = "",
    *,
    multi_gpu: bool = False,
) -> list[Job]:
    jobs = []
    for backend in backends:
        if backend in SERIAL_BACKENDS:
            lanes: list[str] = ["serial"]
        else:
            lanes = ["gpu", "cpu"] if lane == "all" else [lane]
        for job_lane in lanes:
            if job_lane == "gpu":
                gpus = gpus_by_backend[backend]
                if multi_gpu:
                    gpus = [",".join(gpus)]
                for index, gpu in enumerate(gpus):
                    shard = None if multi_gpu else (index, len(gpus))
                    name = f"{backend}-multigpu" if multi_gpu else f"{backend}-gpu{gpu}"
                    jobs.append(
                        Job(
                            name=name,
                            backend=backend,
                            lane="gpu",
                            gpu=gpu,
                            shard=shard,
                            argv=[],
                            env=lane_env(backend, "gpu", gpu, shard),
                            log=run_dir / f"{name}.log",
                            junit=run_dir / f"{name}.xml",
                        )
                    )
            else:
                name = f"{backend}-{job_lane}"
                jobs.append(
                    Job(
                        name=name,
                        backend=backend,
                        lane=job_lane,
                        gpu=None,
                        shard=None,
                        argv=[],
                        env=lane_env(backend, job_lane, None, None),
                        log=run_dir / f"{name}.log",
                        junit=run_dir / f"{name}.xml",
                    )
                )
    for job in jobs:
        job.argv = pytest_argv(
            job.junit,
            lane_marker(job.lane, slow, marker),
            paths,
            extra,
            job.lane == "serial" or multi_gpu,
            multi_gpu=multi_gpu,
            basetemp_subdir=job.name if len(jobs) > 1 else None,
        )
    return jobs


# ---------------------------------------------------------------------------
# junit reports


def _module_of(classname: str, root: Path) -> str | None:
    """``test.test_foo.TestBar`` -> ``test/test_foo.py`` (the longest prefix that
    names a file under ``root``), as scripts/shard_durations.py does."""
    parts = classname.split(".")
    for end in range(len(parts), 0, -1):
        candidate = "/".join(parts[:end]) + ".py"
        if (root / candidate).is_file():
            return candidate
    return None


def node_id(classname: str, name: str, root: Path) -> str:
    module = _module_of(classname, root)
    if module is None:
        return f"{classname}::{name}"
    rest = classname.split(".")[module.count("/") + 1 :]
    return "::".join([module, *rest, name])


def read_junit(path: Path, root: Path = REPO_ROOT) -> Junit:
    """Test counts and the outcome of every test, by node id. A node reported
    more than once (subtests) keeps its worst outcome."""
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    outcomes: dict[str, str] = {}
    collection_errors: list[str] = []
    severity = {"passed": 0, "skipped": 1, "failed": 2, "error": 2}
    for case in ET.parse(path).getroot().iter("testcase"):
        classname = case.get("classname", "")
        tags = {child.tag for child in case}
        properties = {
            prop.get("name"): prop.get("value", "")
            for prop in case.findall("properties/property")
        }
        has_identity = "helion_run_tests_path" in properties
        if not classname and not has_identity:
            # A collection error (the module did not import) or a
            # session-level entry.
            if tags & {"error", "failure"}:
                collection_errors.append(case.get("name", "?"))
            continue
        counts["tests"] += 1
        if "failure" in tags:
            counts["failures"] += 1
            outcome = "failed"
        elif "error" in tags:
            counts["errors"] += 1
            outcome = "error"
        elif "skipped" in tags:
            counts["skipped"] += 1
            outcome = "skipped"
        else:
            outcome = "passed"
        if has_identity:
            # In mixed collections pytest can omit an external file from
            # its nodeid. The item's actual path remains authoritative.
            selector = properties["helion_run_tests_selector"]
            # Preserve external locations while keeping repository IDs short.
            source = Path(os.path.abspath(properties["helion_run_tests_path"]))
            if source.is_relative_to(root):
                source = source.relative_to(root)
            nodeid = f"{source}{selector}"
        else:
            # Reports the plugin never sees, such as xdist worker crashes,
            # lack item properties and need the legacy mapping.
            nodeid = node_id(classname, case.get("name", ""), root)
        if severity[outcome] >= severity.get(outcomes.get(nodeid, "passed"), 0):
            outcomes[nodeid] = outcome
    return Junit(counts, outcomes, collection_errors)


# ---------------------------------------------------------------------------
# --changed: the test modules a diff touches


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def changed_files(rev: str) -> list[str]:
    """Files changed since the merge base with ``rev``, plus the staged,
    unstaged and untracked ones; deleted files and both names of a rename
    count."""
    try:
        base = git_output("merge-base", rev, "HEAD").strip()
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"run_tests: --since {rev!r} is not a revision git knows: {e.stderr.strip()}"
        ) from None
    files = set(git_output("diff", "--name-only", "--no-renames", base).split())
    files.update(git_output("diff", "--name-only", "--no-renames").split())
    files.update(git_output("diff", "--name-only", "--no-renames", "--cached").split())
    files.update(git_output("ls-files", "--others", "--exclude-standard").split())
    return sorted(files)


def select_modules(
    files: Iterable[str], root: Path = REPO_ROOT
) -> tuple[list[str], list[str], bool]:
    """The test modules to run for the changed ``files``: the changed test
    modules and the owners of changed golden files (``test/test_x.expected*``),
    the ones naming a changed source module or script, and test_examples for
    examples. Returns (modules, notes, broad): ``broad`` means a change to
    shared infrastructure, where a selection is not a substitute for the lanes."""
    test_dir = root / TEST_DIR
    modules = sorted(str(p.relative_to(root)) for p in test_dir.glob("test_*.py"))
    sources: dict[str, str] = {}

    def text_of(module: str) -> str:
        if module not in sources:
            sources[module] = (root / module).read_text(
                encoding="utf-8", errors="replace"
            )
        return sources[module]

    selected: set[str] = set()
    notes: list[str] = []
    broad = False
    for file in files:
        path = Path(file)
        if file.startswith(f"{TEST_DIR}/") and ".expected" in path.name:
            module = f"{TEST_DIR}/{path.name.split('.expected')[0]}.py"
            if module in modules:
                selected.add(module)
            else:
                notes.append(f"{file}: golden file with no {module}")
            continue
        if path.suffix != ".py":
            continue
        if file.startswith(f"{TEST_DIR}/"):
            if path.name.startswith("test_") and file in modules:
                selected.add(file)
            elif path.name.startswith("test_") and not (root / file).exists():
                notes.append(f"{file}: deleted test module")
            elif path.name == "conftest.py":
                notes.append(f"{file}: shared test infrastructure")
                broad = True
            else:
                notes.append(
                    f"{file}: a test helper; the modules importing it are not tracked"
                )
                broad = True
            continue
        if file.startswith("helion/_testing/"):
            notes.append(f"{file}: shared test infrastructure")
            broad = True
            continue
        if not file.startswith(("helion/", "examples/", "scripts/")):
            continue
        stem = path.stem
        if stem.startswith("__"):
            notes.append(f"{file}: package module; its importers are not tracked")
            broad = True
            continue
        pattern = re.compile(rf"\b{re.escape(stem)}\b")
        hits = [m for m in modules if pattern.search(text_of(m))]
        if file.startswith("examples/"):
            hits.append(f"{TEST_DIR}/test_examples.py")
        if len(hits) > BROAD_FRACTION * len(modules):
            notes.append(f"{file}: named in {len(hits)} of {len(modules)} test modules")
            broad = True
            continue
        if not hits:
            notes.append(f"{file}: no test module names it")
        selected.update(
            h for h in hits if h in modules or h.endswith("test_examples.py")
        )
    return sorted(selected), notes, broad


# ---------------------------------------------------------------------------
# GPUs


def explicit_gpus(args: argparse.Namespace) -> list[str]:
    """``--gpus``, else ``CUDA_VISIBLE_DEVICES``, as sorted unique nvidia-smi
    indices; empty when neither is set."""
    if args.gpus:
        return args.gpus
    text = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not text.strip():
        return []
    try:
        return [str(g) for g in gpu_board.parse_gpu_list(text)]
    except argparse.ArgumentTypeError:
        raise SystemExit(
            f"run_tests: CUDA_VISIBLE_DEVICES={text!r} is not a list of GPU indices "
            "(UUIDs are not supported); pass --gpus"
        ) from None


def acquire_gpus(
    args: argparse.Namespace,
    purpose: str,
    stack: contextlib.ExitStack,
    alerts: list[gpu_board.SecurityAlert] | None = None,
) -> list[str]:
    """The GPUs for the GPU lanes, with a claim on the board held until
    ``stack`` closes: the explicit GPUs (refused when another account claims
    them, they are busy, or another account's processes sit on them without
    ``--shared``, unless ``--force``), else ``-n`` free GPUs picked and
    claimed under the board lock."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not args.gpus and visible is not None and not visible.strip():
        raise SystemExit(
            "run_tests: CUDA_VISIBLE_DEVICES disables all GPUs; no tests started. "
            "Use --lane cpu for CPU-only checks, or --gpus to select GPUs explicitly"
        )
    explicit = explicit_gpus(args)
    if explicit and args.no_board:
        return explicit
    gpus = gpu_board.sample_gpus()
    if gpus is None:
        raise SystemExit(
            "run_tests: GPU discovery failed (nvidia-smi unavailable or failed); "
            "no tests started. Use --lane cpu for CPU-only checks; "
            "GPU lanes require working NVIDIA discovery"
        )
    if not gpus:
        raise SystemExit(
            "run_tests: nvidia-smi reported no GPUs; no tests started. "
            "Use --lane cpu for CPU-only checks"
        )
    if args.multi_gpu and len(gpus) < 2:
        raise SystemExit("run_tests: --multi-gpu needs at least two GPUs")
    if len(gpus) == 1:
        # Only positive single-device discovery permits the unshared path.
        available = str(gpus[0].index)
        if explicit and explicit != [available]:
            raise SystemExit(
                f"run_tests: nvidia-smi reported only GPU {available}; "
                f"requested {', '.join(explicit)}"
            )
        return [available]
    try:
        if explicit:
            board, claim, problems = gpu_board.claim_gpus(
                [int(g) for g in explicit],
                purpose,
                CLAIM_TTL,
                force=args.force,
                shared=args.shared,
            )
            for gpu, why in sorted(problems.items()):
                print(
                    f"run_tests: --force: using GPU {gpu}, although it {why}",
                    file=sys.stderr,
                )
        else:
            board, claim = gpu_board.pick_and_claim(
                args.count, args.shared, [], purpose, CLAIM_TTL
            )
    except gpu_board.SecurityAlert as e:
        print(f"run_tests: {e.report()}", file=sys.stderr)
        print(
            "run_tests: report this to the user; --no-board --gpus N runs without "
            "the board once they say so",
            file=sys.stderr,
        )
        raise SystemExit(gpu_board.SECURITY_EXIT) from None
    except gpu_board.GpuConflict as e:
        hint = (
            "Stop and inspect remaining processes before reusing these GPUs."
            if e.has_unconfirmed_cleanup
            else "Let the board choose (drop --gpus or CUDA_VISIBLE_DEVICES "
            "and pass -n), ask on the board, or pass --force."
        )
        raise SystemExit(f"run_tests: {e}. {hint}") from None
    except gpu_board.NotEnoughGpus as e:
        print(
            gpu_board.render_status(
                e.board.views,
                e.board.claims,
                [],
                e.board.me,
                e.board.now,
                e.board.truth,
            ),
            file=sys.stderr,
        )
        raise SystemExit(
            f"run_tests: only {len(e.chosen)} of {e.count} GPUs are free; pass --gpus, "
            "--shared, or wait (ask on the board with scripts/gpu_board.py say)"
        ) from None
    except gpu_board.LockTimeout as e:
        raise SystemExit(f"run_tests: GPU board lock busy: {e}") from None
    except (gpu_board.BoardError, OSError) as e:
        raise SystemExit(
            f"run_tests: GPU board unavailable ({e}); "
            "stop and report this GPU board failure to the user"
        ) from None
    stack.enter_context(
        gpu_board.hold_claim(
            board.dir, claim, CLAIM_TTL, alerts, retain_on_error=(CleanupError,)
        )
    )
    return [str(g) for g in claim.gpus]


def split_gpus(
    backends: Sequence[str], gpus: Sequence[str], need_gpu: bool = True
) -> dict[str, list[str]]:
    """Each backend's GPU lane gets its own GPUs, dealt round-robin; a run
    without a GPU lane (``need_gpu`` false) gets none."""
    needing = [b for b in backends if b not in SERIAL_BACKENDS] if need_gpu else []
    if needing and len(gpus) < len(needing):
        raise SystemExit(
            f"run_tests: {len(needing)} backends need a GPU each but only "
            f"{len(gpus)} given ({', '.join(gpus) or 'none'}); pass more with --gpus or -n"
        )
    result: dict[str, list[str]] = {b: [] for b in backends}
    for index, gpu in enumerate(gpus):
        if needing:
            result[needing[index % len(needing)]].append(gpu)
    return result


# ---------------------------------------------------------------------------
# Running


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def shell_line(job: Job) -> str:
    # Match the runner's controlled variables and import path without printing
    # unrelated (potentially private) values from the inherited environment.
    assignments = {**job.env, "PYTHONPATH": job_environment(job)["PYTHONPATH"]}
    return shlex.join(
        [
            "env",
            *(arg for key in RUNNER_ENV for arg in ("-u", key)),
            *(f"{key}={value}" for key, value in sorted(assignments.items())),
            *job.argv,
        ]
    )


class Runner:
    """Runs jobs concurrently, one at a time within a group, and reads their
    junit reports as they finish."""

    def __init__(self, heartbeat: float) -> None:
        self.heartbeat = heartbeat
        self.procs: dict[str, OwnedProcess] = {}
        self.interrupted = False
        self.results: dict[str, JobResult] = {}

    def _interrupt(self, signum: int, _frame: object) -> None:
        self.interrupted = True
        for proc in self.procs.values():
            proc.request_stop(signum)

    def _start(self, job: Job) -> tuple[OwnedProcess, float]:
        with job.log.open("wb") as log:
            proc = OwnedProcess(
                job.argv,
                cwd=REPO_ROOT,
                env=job_environment(job),
                stdout=log,
                label=job.name,
            )
            # Closing the parent's log can fail after the child has started.
            # Register ownership before leaving the context so run() retires it.
            self.procs[job.name] = proc
        if self.interrupted:
            proc.request_stop()
        where = f"GPU {job.gpu}" if job.gpu is not None else "no GPU"
        shard = (
            f", shard {job.shard[0] + 1} of {job.shard[1]}"
            if job.shard and job.shard[1] > 1
            else ""
        )
        if job.group is None:
            print(f"[{job.name}] started on {where}{shard}: {job.log}", flush=True)
        return proc, time.monotonic()

    @staticmethod
    def _finish(job: Job, code: int, started: float) -> JobResult:
        result = JobResult(job, code, time.monotonic() - started)
        if job.junit.exists():
            with contextlib.suppress(ET.ParseError, OSError):
                result.junit = read_junit(job.junit)
        return result

    def run(self, jobs: Sequence[Job], quiet: bool = False) -> dict[str, JobResult]:
        """Run ``jobs``; returns the result of every job that ran, by name
        (after an interrupt, the jobs not yet started are missing)."""
        signums = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        previous = {
            signum: signal.signal(signum, self._interrupt) for signum in signums
        }
        pending_error: BaseException | None = None
        try:
            return self._run(jobs, quiet)
        except BaseException as error:
            pending_error = error
            raise
        finally:
            # Even a log/JUnit/start failure must retire every already-started
            # job before the outer board context can release its claim.
            errors: list[tuple[str, BaseException]] = []
            try:
                for name, proc in self.procs.items():
                    try:
                        proc.close()
                    except BaseException as error:
                        errors.append((name, error))
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
            if errors:
                # Preserve every affected job in diagnostics, and retain the
                # claim if any cleanup failed to confirm process retirement.
                error_type = (
                    CleanupError
                    if isinstance(pending_error, CleanupError)
                    or any(
                        isinstance(error, CleanupError)
                        or not isinstance(error, ProcessError)
                        for _, error in errors
                    )
                    else ProcessError
                )
                details = [
                    f"[{name}] {str(error).removeprefix(f'[{name}] ')}"
                    for name, error in errors
                ]
                if pending_error is not None:
                    details.insert(0, f"original runner error: {pending_error}")
                raise error_type("; ".join(details)) from (
                    pending_error if pending_error is not None else errors[0][1]
                )

    def _run(self, jobs: Sequence[Job], quiet: bool) -> dict[str, JobResult]:
        pending = list(jobs)
        running: dict[str, tuple[Job, OwnedProcess, float]] = {}
        results = self.results
        last_beat = time.monotonic()
        while pending or running:
            busy_groups = {job.group for job, _, _ in running.values() if job.group}
            for job in list(pending):
                if self.interrupted:
                    pending.remove(job)
                elif job.group is None or job.group not in busy_groups:
                    pending.remove(job)
                    proc, started = self._start(job)
                    running[job.name] = (job, proc, started)
                    if job.group:
                        busy_groups.add(job.group)
            for name, (job, proc, started) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                del running[name]
                result = self._finish(job, code, started)
                results[name] = result
                if not quiet:
                    print(
                        f"[{name}] finished in {format_seconds(result.seconds)}: "
                        f"{result.describe()}",
                        flush=True,
                    )
            if running:
                if self.heartbeat and time.monotonic() - last_beat >= self.heartbeat:
                    names = [
                        f"{name} ({format_seconds(time.monotonic() - started)})"
                        for name, (_, _, started) in running.items()
                    ]
                    print(f"still running: {', '.join(names)}", flush=True)
                    last_beat = time.monotonic()
                time.sleep(1)
        return results


FailureKey = tuple[str, str, str]  # backend, lane, node id


def failure_keys(results: Iterable[JobResult]) -> list[FailureKey]:
    keys: list[FailureKey] = []
    for result in results:
        for nodeid in result.failed:
            key = (result.job.backend, result.job.lane, nodeid)
            if key not in keys:
                keys.append(key)
    return keys


def rerun_failures(
    results: Sequence[JobResult],
    run_dir: Path,
    heartbeat: float,
    pytest_extra: Sequence[str],
    *,
    runners: list[Runner] | None = None,
) -> tuple[dict[FailureKey, str], bool]:
    """Run every failed test again, each alone in a fresh pytest process, one
    at a time per backend and lane (on the first GPU that lane used). Returns
    the outcome per failure: passed, failed, error, skipped, crashed (the
    process died) or missing (nothing collected), and whether the reruns were
    interrupted."""
    first_gpu: dict[tuple[str, str], str | None] = {}
    for result in results:
        first_gpu.setdefault((result.job.backend, result.job.lane), result.job.gpu)
    jobs = []
    keys = failure_keys(results)
    key_of_job: dict[str, FailureKey] = {}
    for index, (backend, lane, nodeid) in enumerate(keys):
        name = f"rerun-{backend}-{lane}-{index}"
        junit = run_dir / f"{name}.xml"
        # Pytest keeps the last value for these options. Put isolation's
        # overrides after caller/config options without parsing unknown plugins'
        # argument arity or discarding their settings.
        extra = [
            *pytest_extra,
            "-k",
            "",
            "-m",
            "",
            "--helion-isolated-nodeid",
            nodeid,
            f"--basetemp={run_dir / (name + '-tmp')}",
        ]
        gpu = first_gpu[(backend, lane)]
        argv = pytest_argv(
            junit,
            "",
            [nodeid],
            extra,
            serial=True,
            timeout=RERUN_TIMEOUT,
            overrides_last=True,
        )
        key_of_job[name] = (backend, lane, nodeid)
        jobs.append(
            Job(
                name,
                backend,
                lane,
                gpu,
                None,
                argv,
                lane_env(backend, lane, gpu, None),
                run_dir / f"{name}.log",
                junit,
                group=f"{backend}-{lane}",
            )
        )
    runner = Runner(heartbeat)
    if runners is not None:
        runners.append(runner)
    outcomes: dict[FailureKey, str] = {}
    for name, result in runner.run(jobs, quiet=True).items():
        key = key_of_job[name]
        if result.crashed:
            outcomes[key] = "crashed"
        elif result.junit is None:
            outcomes[key] = "missing"
        else:
            outcomes[key] = result.junit.outcomes.get(key[2], "missing")
    return outcomes, runner.interrupted


RERUN_NOTES = {
    "passed": "passed in isolation (order-dependent or flaky)",
    "failed": "fails in isolation",
    "error": "errors in isolation",
    "skipped": "skipped when run alone",
    "crashed": "crashed when run alone",
    "missing": "no result when run alone",
}


def summarize(
    run_dir: Path,
    results: Sequence[JobResult],
    outcomes: dict[FailureKey, str] | None,
    skipped_rerun: str | None,
) -> tuple[str, int]:
    lines = [f"run directory: {run_dir}"]
    gpu_width = max([5, *(len(r.job.gpu or "-") for r in results)])
    lines.append(f"{'job':<22} {'GPU':<{gpu_width}} {'shard':<6} {'time':<8} result")
    for r in results:
        shard = (
            f"{r.job.shard[0] + 1}/{r.job.shard[1]}"
            if r.job.shard and r.job.shard[1] > 1
            else "-"
        )
        lines.append(
            f"{r.job.name:<22} {r.job.gpu or '-':<{gpu_width}} {shard:<6} "
            f"{format_seconds(r.seconds):<8} {r.describe()}"
        )
    keys = failure_keys(results)
    crashed = [r.job.name for r in results if r.crashed]
    several_backends = len({r.job.backend for r in results}) > 1
    # Modules that never imported: nothing ran, so there is nothing to rerun.
    uncollected = [
        f"[{r.job.backend}] {name}" if several_backends else name
        for r in results
        if r.junit is not None
        for name in r.junit.collection_errors
    ]
    flaky = [k for k in keys if outcomes is not None and outcomes.get(k) == "passed"]
    real = [k for k in keys if k not in flaky]
    if keys:
        lines.append(f"failures ({len(keys)}):")
        for key in keys:
            backend, _lane, nodeid = key
            label = f"[{backend}] {nodeid}" if several_backends else nodeid
            if outcomes is None:
                note = skipped_rerun or "not rerun"
            else:
                note = RERUN_NOTES.get(outcomes.get(key, "missing"), "no result")
            lines.append(f"  {label}  -> {note}")
    if uncollected:
        lines.append(
            f"collection errors ({len(uncollected)}; these modules did not run):"
        )
        lines.extend(f"  {name}" for name in dict.fromkeys(uncollected))
    if crashed:
        lines.append(f"crashed jobs: {', '.join(crashed)}")
    if crashed or real or uncollected:
        verdict = f"VERDICT: FAIL ({len(real)} failing"
        verdict += f", {len(flaky)} passed in isolation" if flaky else ""
        verdict += f", {len(uncollected)} collection errors" if uncollected else ""
        verdict += f", {len(crashed)} crashed" if crashed else ""
        verdict += ")"
        code = 1
    elif flaky:
        verdict = (
            f"VERDICT: FLAKY ({len(flaky)} failures passed in isolation; "
            "fix the order dependence)"
        )
        code = 2
    elif not any(r.tests for r in results):
        verdict = "VERDICT: NOTHING RAN (no tests collected)"
        code = 3
    else:
        verdict = "VERDICT: PASS"
        code = 0
    lines.append(verdict)
    return "\n".join(lines), code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("HELION_BACKEND", "triton"),
        help="backend(s), comma-separated: triton, cute, tileir, pallas-interpret "
        "(default: HELION_BACKEND or triton)",
    )
    parser.add_argument(
        "--lane",
        choices=("all", "gpu", "cpu"),
        default="all",
        help="all (default): the GPU lane and the CPU lane, which together are the "
        "whole suite; gpu: the GPU lane only (CI's non-CuTe jobs also run the "
        "cpu_only modules); cpu: the cpu_only modules without a GPU",
    )
    parser.add_argument("--gpus", help="comma-separated GPU indices for the GPU lanes")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="one serial pytest process using all selected GPUs; requires one backend "
        "and --lane gpu (for tests that launch their own distributed workers)",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=None,
        help="GPUs to pick from the board when --gpus and CUDA_VISIBLE_DEVICES are "
        "unset (default 2 with --multi-gpu, otherwise 1)",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="let the board pick idle GPUs with other accounts' processes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="use --gpus / CUDA_VISIBLE_DEVICES even when another account claims them",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="include the slow tier (pull requests skip it)",
    )
    parser.add_argument(
        "--changed",
        action="store_true",
        help="run the test modules that the changes since --since touch",
    )
    parser.add_argument(
        "--since",
        default="origin/main",
        metavar="REV",
        help="the revision --changed compares against (default origin/main)",
    )
    parser.add_argument(
        "--no-rerun", action="store_true", help="do not rerun failures in isolation"
    )
    parser.add_argument(
        "--no-board", action="store_true", help="do not claim the GPUs on the board"
    )
    parser.add_argument(
        "--tag", help="name for the run directory (default: backend and lane)"
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=300.0,
        help="seconds between progress lines (0: none)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the jobs and exit"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="test files or directories (default: the whole suite); pytest options after --",
    )
    args = parser.parse_args(argv)
    args.backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    if args.gpus:
        try:
            args.gpus = [str(g) for g in gpu_board.parse_gpu_list(args.gpus)]
        except argparse.ArgumentTypeError as e:
            parser.error(f"--gpus: {e}")
    if args.count is None:
        args.count = 2 if args.multi_gpu else 1
    serial = [b for b in args.backends if b in SERIAL_BACKENDS]
    if args.multi_gpu and (args.lane != "gpu" or len(args.backends) != 1 or serial):
        parser.error("--multi-gpu requires one GPU backend and --lane gpu")
    if serial and args.lane != "all":
        parser.error(
            f"{', '.join(serial)} runs the whole suite serially; --lane does not apply"
        )
    # argparse leaves a leading -- in paths when it separates pytest options.
    extra: list[str] = []
    paths: list[str] = []
    for part in args.paths:
        if part == "--":
            continue
        (extra if part.startswith("-") or extra else paths).append(part)
    # A -m among the pytest options is combined with the lane's marker.
    args.marker = ""
    while "-m" in extra:
        index = extra.index("-m")
        if index + 1 >= len(extra):
            parser.error("-m needs a marker expression")
        expression = extra[index + 1]
        args.marker = (
            f"({args.marker}) and ({expression})" if args.marker else expression
        )
        del extra[index : index + 2]
    args.paths = paths
    args.extra = extra
    if args.no_board and not args.gpus and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("--no-board needs --gpus (or CUDA_VISIBLE_DEVICES)")
    return args


def unused_run_dir(tag: str, create: bool) -> Path:
    """A fresh ``.logs/tests/<timestamp>-<tag>`` directory (with a numeric
    suffix when another run took the name in the same second)."""
    stem = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{tag}"
    if not create:
        return LOGS_DIR / stem
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(1000):
        run_dir = LOGS_DIR / (stem if attempt == 0 else f"{stem}-{attempt}")
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise SystemExit(f"run_tests: cannot create a run directory under {LOGS_DIR}")


def pytest_uses_pyargs(paths: Sequence[str], extra: Sequence[str]) -> bool:
    """Read pytest's effective addopts without importing tests or plugins."""
    parser = argparse.ArgumentParser(
        add_help=False, allow_abbrev=False, exit_on_error=False
    )
    parser.add_argument("--pyargs", action="store_true")
    parser.add_argument("-o", "--override-ini", action="append", default=[])
    parser.add_argument("-c", "--config-file", dest="inifilename")
    parser.add_argument("--rootdir")
    try:
        options = [*shlex.split(os.environ.get("PYTEST_ADDOPTS", "")), *extra]
        parsed, _ = parser.parse_known_args(options)
    except (ValueError, argparse.ArgumentError) as error:
        raise SystemExit(f"run_tests: invalid pytest options: {error}") from error
    if parsed.pyargs:
        return True
    addopts: object = None
    for override in parsed.override_ini:
        name, separator, value = override.partition("=")
        if separator and name == "addopts":
            addopts = value  # Last override wins, including an empty value.
    if addopts is None:
        # Lazy: filesystem selections and --help need no pytest import. This
        # helper only reads config files; Config.parse would load user plugins.
        from _pytest.config.exceptions import UsageError
        from _pytest.config.findpaths import determine_setup

        setup: dict[str, Any] = {
            "inifile": str(REPO_ROOT / parsed.inifilename)
            if parsed.inifilename is not None
            else None,
            "args": [str(REPO_ROOT / path.split("::", 1)[0]) for path in paths],
            "rootdir_cmd_arg": str(REPO_ROOT / os.path.expandvars(parsed.rootdir))
            if parsed.rootdir is not None
            else None,
            "invocation_dir": REPO_ROOT,
        }
        # pytest 9.2 added this keyword and wrapped config values; earlier
        # supported pytest versions return plain strings/lists instead.
        if "override_ini" in inspect.signature(determine_setup).parameters:
            setup["override_ini"] = []
        try:
            config = determine_setup(**setup)[2]
        except UsageError as error:
            raise SystemExit(f"run_tests: invalid pytest options: {error}") from error
        addopts = config.get("addopts", "")
        addopts = getattr(addopts, "value", addopts)
    try:
        tokens = shlex.split(addopts) if isinstance(addopts, str) else addopts
        return parser.parse_known_args([*tokens, *options])[0].pyargs
    except (ValueError, argparse.ArgumentError) as error:
        raise SystemExit(f"run_tests: invalid pytest options: {error}") from error


@contextlib.contextmanager
def record_runner_error(run_dir: Path, runners: Sequence[Runner]) -> Iterator[None]:
    try:
        yield
    except Exception as error:
        lines = [f"run directory: {run_dir}", "Completed jobs before runner error:"]
        lines += [
            f"{name}: {result.describe()}"
            for runner in runners
            for name, result in runner.results.items()
        ]
        lines += [f"VERDICT: RUNNER ERROR ({error})"]
        summary = "\n".join(lines) + "\n"
        try:
            (run_dir / "summary.txt").write_text(summary)
        except OSError as write_error:
            print(f"cannot save partial summary: {write_error}", file=sys.stderr)
        print(summary, file=sys.stderr, flush=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = list(args.paths)
    pyargs = None
    for selector in paths:
        path = selector.split("::", 1)[0]
        if path.startswith("@"):
            path = path[1:]  # pytest argument file, not a literal test path
        elif not (REPO_ROOT / path).exists() and all(
            part.isidentifier() for part in path.split(".")
        ):
            if pyargs is None:
                pyargs = pytest_uses_pyargs(paths, args.extra)
            if pyargs:
                continue  # Package discovery belongs to pytest, without importing here.
        if not (REPO_ROOT / path).exists():
            raise SystemExit(f"run_tests: test path does not exist: {REPO_ROOT / path}")
    if args.changed:
        modules, notes, broad = select_modules(changed_files(args.since))
        for note in notes:
            print(f"--changed: {note}")
        if broad:
            print(
                "--changed: shared code changed; running the full lanes instead of a selection"
            )
            paths = []
        elif not modules and paths:
            print(
                "--changed: no test module matched the changed files; running the paths given"
            )
        elif not modules:
            print(
                "--changed: no test module matched the changed files; pass paths or run the lanes"
            )
            print("VERDICT: NOTHING SELECTED")
            return 3
        else:
            print(f"--changed: {len(modules)} modules: {' '.join(modules)}")
            paths += modules

    tag = args.tag or f"{'-'.join(args.backends)}-{args.lane}"
    need_gpu = args.lane != "cpu" and any(
        b not in SERIAL_BACKENDS for b in args.backends
    )
    needing = sum(b not in SERIAL_BACKENDS for b in args.backends) if need_gpu else 0
    # Validate multi-device requests before discovery or creating a claim.
    if args.multi_gpu:
        selected = explicit_gpus(args)
        if (selected and len(selected) < 2) or (not selected and args.count < 2):
            raise SystemExit("run_tests: --multi-gpu needs at least two GPUs")
    # Each backend's GPU lane needs a GPU of its own.
    args.count = max(args.count, needing)
    run_dir = unused_run_dir(tag, create=not args.dry_run)
    # Security alerts raised while the run's claim was renewed.
    alerts: list[gpu_board.SecurityAlert] = []
    with contextlib.ExitStack() as stack:
        if args.dry_run:
            # Placeholder GPUs: a preview never touches the board.
            gpus = args.gpus or [str(i) for i in range(args.count)]
        elif need_gpu:
            gpus = acquire_gpus(
                args, f"pytest {','.join(args.backends)} ({tag})", stack, alerts
            )
        else:
            gpus = []
        gpus_by_backend = split_gpus(args.backends, gpus, need_gpu)
        jobs = build_jobs(
            args.backends,
            args.lane,
            gpus_by_backend,
            args.slow,
            paths,
            args.extra,
            run_dir,
            args.marker,
            multi_gpu=args.multi_gpu,
        )
        if args.dry_run:
            for job in jobs:
                print(f"# {job.name}\n{shell_line(job)}")
            return 0

        (run_dir / "commands.sh").write_text(
            f"# cwd: {REPO_ROOT}\n"
            + "".join(f"# {job.name}\n{shell_line(job)}\n" for job in jobs)
        )
        runner = Runner(args.heartbeat)
        runners = [runner]
        stack.enter_context(record_runner_error(run_dir, runners))
        by_name = runner.run(jobs)
        if runner.interrupted:
            print(f"interrupted; partial logs in {run_dir}")
            return 130
        results = [by_name[job.name] for job in jobs]
        keys = failure_keys(results)
        outcomes: dict[FailureKey, str] | None = None
        skipped = None
        if keys and not args.no_rerun and len(keys) <= MAX_RERUNS:
            print(
                f"rerunning {len(keys)} failures, each alone in a fresh process",
                flush=True,
            )
            outcomes, interrupted = rerun_failures(
                results, run_dir, args.heartbeat, args.extra, runners=runners
            )
            if interrupted:
                print(f"interrupted during the reruns; partial logs in {run_dir}")
                return 130
        elif keys:
            skipped = (
                "--no-rerun"
                if args.no_rerun
                else f"more than {MAX_RERUNS} failures, not rerun"
            )
    summary, code = summarize(run_dir, results, outcomes, skipped)
    (run_dir / "summary.txt").write_text(summary + "\n")
    print(summary)
    if alerts:
        print(
            "run_tests: the GPU board raised a SECURITY ALERT during the run "
            "(printed above); report it to the user",
            file=sys.stderr,
        )
        return gpu_board.SECURITY_EXIT
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ProcessError as error:
        print(f"run_tests: {error}", file=sys.stderr)
        sys.exit(1)
