# ruff: noqa: ANN401
"""Autotune and benchmark a selection of better-benchmark ports across GPUs.

Every job runs in a fresh process on one GPU, one job per GPU at a time. A
GPU waits while another account's processes compute on it or a process of
this account that the campaign did not start runs there; other accounts'
idle resident processes do not hold it up. GPUs come from the machine's GPU
board or from the command line:

- ``--board-gpus N``: claim free GPUs with ``scripts/gpu_board.py pick``
  while jobs can start and fewer than N are held (never shared GPUs:
  autotuning and timing need idle ones), renew the claims every third of
  ``--board-ttl`` while running, and hand a GPU back whenever no job can
  start on it, such as while the remaining benches wait for autotunes
  running elsewhere. ``gpu_board.py renew`` extends every claim of the
  board session, the campaign's and any other the session holds (run the
  campaign under a ``GPU_BOARD_SESSION`` of its own to keep others out).
  The campaign does not read board messages; stop it to give back GPUs it
  is using.
- ``--gpus``: GPUs you hold on the board. The campaign renews no claims in
  this mode, so hold these for its whole run, for example by starting it
  under ``scripts/gpu_board.py run --gpus 0,1 --ttl 2h -- python
  campaign.py --gpus 0,1 ...``, which renews its claim while the command
  runs.

The jobs:

- ``autotune``: a cold full autotune (``HELION_AUTOTUNE_EFFORT=full``, forced,
  fresh random seed, private cache directories) of one port on one backend,
  written to ``<run>/configs/<repro>__<shape>__<backend>.json``.
- ``bench``: ``bench.py`` for one (repro, shape) once its autotunes have
  finished, whether or not they succeeded (a backend without a usable
  autotune record fails on its own; the other implementations are timed),
  written to ``<run>/bench/<repro>__<shape>.json``.

A ``provisional`` autotune record, an earlier run's Triton record that
``plan_run.py --provisional-triton`` copied into the run, stands in for an
autotune still to run: benches pin it, but the campaign still runs the
autotune. A run can thus come in two phases: ``--backends cute`` runs every
CuTe autotune and the benches, which time Helion-Triton with the provisional
configs; then ``--backends triton`` runs the Triton autotunes and benches
each point again after its own.

A job during which a process outside it computed on its GPU (its record's
``foreign_gpu_activity`` reaches ``common.CONTAMINATION_SM_PCT``, an
autotune's for long enough: ``common.foreign_activity``), or whose
activity samples do not cover it (``common.activity_coverage_issue``: the
sampling failed or stalled), runs again, up to ``--retry-contaminated`` more
times, on another GPU when one is free. Each such attempt's record stays as
``<record>.attempt<k>``, and its log as ``<log>.attempt<k>``. The first clean,
covered attempt's record is the job's, listing the attempts before it
(``contaminated_attempts``); if every attempt was contaminated or not covered,
or a retry fails (times out, crashes), the best (a covered one first, then by
``common.contamination_rank``) is kept with that list and ``contaminated:
true``, and reports leave it out.

Finished jobs are skipped when the campaign is restarted, so it can be
interrupted and resumed; ``--retune``/``--rebench`` redo them, and a point is
benched again when one of its autotunes runs again or is newer than its bench
record. Starting a job moves its previous record to ``<record>.prev`` (and its
log to ``<log>.prev``), so a rerun that fails, times out or is stopped leaves
no record that passes for finished (or that a bench would pin). A restart
resumes a job the campaign stopped between contaminated attempts with the
attempts so far (the ``.attempt<k>`` files newer than its ``.prev``), and
settles a record the campaign stopped before settling, the last allowed
attempt's included; a job without a record whose attempts have spent the
retries keeps the cleanest of them.
However the campaign stops (done, an error, SIGINT, SIGTERM or SIGHUP), it
kills its running jobs -- each job's processes, found by parent pid (the
autotuner's workers start sessions of their own) and by a marker in their
environment (a worker whose parent has exited is init's child), and their
process groups -- reaps them, removes their cache directories and releases
its board claims; only SIGKILL leaves jobs behind. A job whose leader exits
on its own has what is left of it killed the same way before its cache
directory goes.
A stop signal that arrives while a job is being started or a GPU claimed
takes effect once the job or claim is recorded. A board failure (an error
from ``pick``, a failed ``renew`` or ``release``) stops the campaign; a
SECURITY ALERT from the board stops it with exit status 4 and leaves the
claims to expire.

Example::

    python benchmarks/cute/better_benchmark/campaign.py \\
        --selection benchmarks/cute/better_benchmark/selection.json \\
        --run-dir artifacts/bb-sweep --board-gpus 4
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    import types

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.cute.better_benchmark import common

REPO_ROOT = common.REPO_ROOT
PACKAGE_DIR = common.PACKAGE_DIR


@dataclasses.dataclass
class Job:
    kind: str  # "autotune" | "bench"
    repro: str
    shape: str
    backend: str | None = None
    deps: tuple[str, ...] = ()
    # A rerun after contaminated attempts: its number, the GPU of the last
    # one (avoided when another GPU is free) and their kept records.
    attempt: int = 1
    avoid_gpu: str | None = None
    contaminated: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        suffix = f"__{self.backend}" if self.backend else ""
        return f"{self.kind}:{self.repro}__{self.shape}{suffix}"


@dataclasses.dataclass
class Running:
    job: Job
    proc: subprocess.Popen
    cache: Path
    started: float
    log: Any
    marker: str


def gpu_uuid(gpu: str) -> str | None:
    """The UUID of GPU ``gpu`` (an index), or None when nvidia-smi fails or
    hangs (``common._nvidia_smi`` times out)."""
    out = common._nvidia_smi("-i", gpu, "--query-gpu=uuid", "--format=csv,noheader")
    return (out or "").strip() or None


# Utilization (percent) at which another account's processes on a GPU count
# as running work rather than resident idle workers: the SM utilization that
# contaminates a job's record (common.CONTAMINATION_SM_PCT), so that no job
# starts beside work that would make it run again (gpu_board.py's ``idle``
# state allows up to 10%).
FOREIGN_BUSY_UTILIZATION = common.CONTAMINATION_SM_PCT


def rerun_reason(record: dict[str, Any]) -> list[str] | None:
    """Why a finished job runs again: the processes outside it that computed
    on its GPU (``common.foreign_activity``), or why its activity samples do
    not cover it (``common.activity_coverage_issue``); None when its record
    stands, as one made without the sampling (``BB_FOREIGN_ACTIVITY=0``, or
    before the sampling existed) does, having no samples to judge."""
    found = common.foreign_activity(record)
    if found:
        return found
    if found is None and record.get("foreign_gpu_activity") is not None:
        return [f"samples miss the job: {common.activity_coverage_issue(record)}"]
    return None


def attempt_rank(record: dict[str, Any]) -> tuple[bool, float]:
    """The order in which a job's attempts are kept when every one ran again:
    those whose samples cover the job first, then the least contaminated
    (``common.contamination_rank``)."""
    return (
        common.activity_coverage_issue(record) is not None,
        common.contamination_rank(record),
    )


def busy_uuids() -> set[str] | None:
    """GPUs a new job must not start on: any process of this account that the
    campaign did not start (it tracks its own jobs), or another account's
    processes while the GPU computes; None when nvidia-smi fails or hangs
    (``common._nvidia_smi`` times out), and no job starts until it answers.
    Resident, idle processes of other accounts (benchmark workers that keep
    their contexts) do not block a GPU claimed for the campaign; each job
    records the foreign processes it saw and how much they computed
    (``foreign_gpu_activity``), and the campaign runs a contaminated job
    again (``--retry-contaminated``)."""
    apps = common._nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"
    )
    usage = common._nvidia_smi(
        "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits"
    )
    if apps is None or usage is None:
        return None
    utilization = {
        uuid.strip(): int(value) if value.strip().isdigit() else 100
        for uuid, value in (
            line.split(",", 1) for line in usage.splitlines() if "," in line
        )
    }
    busy: set[str] = set()
    for line in apps.splitlines():
        if "," not in line:
            continue
        uuid, pid = (field.strip() for field in line.split(",", 1))
        try:
            mine = os.stat(f"/proc/{pid}").st_uid == os.getuid()
        except OSError:
            mine = False
        if mine or utilization.get(uuid, 100) >= FOREIGN_BUSY_UTILIZATION:
            busy.add(uuid)
    return busy


BOARD = REPO_ROOT / "scripts" / "gpu_board.py"


def _load_gpu_board() -> types.ModuleType:
    """scripts/gpu_board.py as a module, for the limits and exit statuses the
    campaign shares with it.  scripts/ is no package, so it is loaded by path
    (as test_agent_scripts.py loads it), and registered under its name, by
    which its dataclasses resolve their module."""
    if (module := sys.modules.get("gpu_board")) is not None:
        return module
    spec = importlib.util.spec_from_file_location("gpu_board", BOARD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_board"] = module
    spec.loader.exec_module(module)
    return module


gpu_board = _load_gpu_board()
BOARD_TIMEOUT_S = 120
# Seconds between scheduler passes and between board picks after one found
# no free GPU.  Claims are renewed every third of --board-ttl.
POLL_S = 2.0
PICK_INTERVAL_S = 120.0
BOARD_SECURITY_EXIT = gpu_board.SECURITY_EXIT
# What ``gpu_board.py pick`` prints (with exit status 1) when not enough GPUs
# are free (its NotEnoughGpus); it exits 1 for its own errors too.
_NOT_ENOUGH_GPUS = re.compile(
    r"^gpu_board: only \d+ of the \d+ requested GPUs are ", re.MULTILINE
)


class BoardFailure(Exception):
    """The GPU board failed: the campaign stops."""


class BoardSecurityAlert(BoardFailure):
    """The board raised a SECURITY ALERT: the campaign stops without touching
    the board again."""


class Interrupted(BaseException):
    """A signal asked the campaign to stop (a BaseException, like
    KeyboardInterrupt, so no handler on the way swallows it)."""

    def __init__(self, signum: int) -> None:
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
# While a job or a board claim is being made and recorded, the stop signal
# handler only notes the signal; stop_signals_deferred raises it afterwards.
_defer_stop = False
_deferred_stop: int | None = None


def _raise_interrupted(signum: int, frame: object) -> None:
    global _deferred_stop
    # Later signals must not cut the cleanup short.
    for stop_signal in STOP_SIGNALS:
        signal.signal(stop_signal, signal.SIG_IGN)
    if _defer_stop:
        _deferred_stop = signum
        return
    raise Interrupted(signum)


@contextlib.contextmanager
def stop_signals_deferred() -> Iterator[None]:
    """Hold a stop signal off until the block has finished, so a job process
    or a board claim is never made without being recorded for the cleanup.
    (Blocking the signals with ``pthread_sigmask`` instead would leave them
    blocked in the jobs, which inherit the signal mask.)"""
    global _defer_stop
    _defer_stop = True
    try:
        yield
    finally:
        _defer_stop = False
    if _deferred_stop is not None:
        raise Interrupted(_deferred_stop)


def _board(*args: str) -> subprocess.CompletedProcess[str]:
    command = f"gpu_board.py {' '.join(args)}"
    try:
        proc = subprocess.run(
            [sys.executable, str(BOARD), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=BOARD_TIMEOUT_S,
            # Out of the terminal's process group: a Ctrl-C must not stop a
            # claim halfway.
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        raise BoardFailure(f"{command} timed out after {BOARD_TIMEOUT_S}s") from None
    if proc.returncode == BOARD_SECURITY_EXIT:
        raise BoardSecurityAlert(f"{command} stopped with:\n{proc.stderr}")
    return proc


def board_pick(purpose: str, ttl: str) -> str | None:
    """Claim one free (unshared) GPU on the board; None when none is free.
    Any other failure of ``pick`` is the board's and raises BoardFailure."""
    proc = _board("pick", "-n", "1", "--purpose", purpose, "--ttl", ttl)
    if proc.returncode == 1 and _NOT_ENOUGH_GPUS.search(proc.stderr):
        return None
    picked = proc.stdout.split()
    if proc.returncode != 0 or len(picked) != 1 or not picked[0].isdigit():
        raise BoardFailure(
            f"gpu_board.py pick failed (exit status {proc.returncode}):\n"
            f"{proc.stderr}{proc.stdout}"
        )
    return picked[0]


def board_call(*args: str) -> None:
    """A board command that must succeed (``renew``, ``release``)."""
    proc = _board(*args)
    if proc.returncode != 0:
        raise BoardFailure(
            f"gpu_board.py {' '.join(args)} failed (exit status "
            f"{proc.returncode}):\n{proc.stderr}"
        )


def _job_processes(root: int | None, marker: str | None) -> set[int]:
    """A job's processes: its leader ``root`` (None once reaped: its pid may
    be another process's by then), those carrying its ``marker``, and their
    descendants."""
    roots = set() if marker is None else common.marked_processes(marker)
    if root is not None:
        roots.add(root)
    return common.process_tree(*roots) if roots else set()


def _stop_job(root: int | None, marker: str | None) -> set[int]:
    """Stop (SIGSTOP) a job's processes (``_job_processes``) until no new one
    appears, so none starts a process the kill would miss; the stopped
    processes."""
    stopped: set[int] = set()
    while new := _job_processes(root, marker) - stopped:
        for pid in new:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGSTOP)
        stopped |= new
    return stopped


def _lives_on(tree: set[int], groups: set[int]) -> bool:
    """Whether a process of this account in ``tree`` or in one of
    ``groups`` is alive (a zombie only awaits its reaper)."""
    return any(
        state != "Z" and (pid in tree or int(group) in groups)
        for pid, (state, _ppid, group, *_) in common.proc_stats(os.getuid())
    )


def parse_duration(text: str) -> float:
    """A claim duration in seconds, as ``gpu_board.py --ttl`` reads it
    (``parse_ttl``: ``90s``, ``45m``, ``2h``, ``1d`` or a bare number of
    minutes, up to its ``MAX_TTL_SECONDS``); ValueError otherwise."""
    try:
        return gpu_board.parse_ttl(text)
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from None


def kill_job(
    proc: subprocess.Popen, marker: str | None = None, grace_s: float = 30.0
) -> bool:
    """Kill a job -- every process of its tree (``common.process_tree``),
    every process carrying its ``marker`` (``job_env``) and their process
    groups -- and reap the job; True once none of them is left within
    ``grace_s``, the leader's reap included (one that cannot be reaped in
    time, stuck in uninterruptible I/O, is a survivor).  The job's
    own group misses the autotuner's benchmark worker and its precompile and
    render children, which start sessions of their own; the compilers those
    children start share their groups.  A process whose parent exits
    becomes init's child, out of the tree but still marked: the workers a
    leader leaves behind when it exits on its own."""
    tree = _stop_job(proc.pid if proc.returncode is None else None, marker)
    groups = set()
    for pid in tree:
        with contextlib.suppress(ProcessLookupError):
            groups.add(os.getpgid(pid))
    # Never the campaign's own group (the jobs start sessions of their own).
    groups.discard(os.getpgrp())
    for group in groups:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(group, signal.SIGKILL)
    for pid in tree:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + grace_s
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        # A leader in uninterruptible I/O (D state, after a GPU fault, say)
        # dies only when that returns: a survivor, not a stalled scheduler.
        return False
    while _lives_on(tree, groups) or (
        marker is not None and bool(common.marked_processes(marker))
    ):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def job_env(gpu: str, cache: Path, seed: int, marker: str) -> dict[str, str]:
    # Do not inherit an unrelated experiment's search budget, forced config
    # or backend; optional package paths still work.
    env = {k: v for k, v in os.environ.items() if not k.startswith("HELION_")}
    env.update(
        # The board and nvidia-smi number GPUs in PCI bus order.
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=gpu,
        HELION_AUTOTUNE_EFFORT="full",
        HELION_FORCE_AUTOTUNE="1",
        HELION_AUTOTUNE_RANDOM_SEED=str(seed),
        HELION_CACHE_DIR=str(cache / "helion"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        CUTE_DSL_CACHE_DIR=str(cache / "cute"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
    )
    env[common.JOB_MARKER] = marker
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    gpu_source = parser.add_mutually_exclusive_group(required=True)
    gpu_source.add_argument(
        "--gpus",
        help="GPUs you hold on the board for the whole campaign (it does not "
        "renew their claims), e.g. 0,1, or @FILE to re-read the list from FILE "
        "between jobs",
    )
    gpu_source.add_argument(
        "--board-gpus",
        type=int,
        help="claim up to this many free GPUs on the GPU board as jobs need them",
    )
    parser.add_argument("--board-purpose", default="better-benchmark campaign")
    parser.add_argument(
        "--board-ttl",
        default="2h",
        help="expiry of the board claims (gpu_board.py --ttl), renewed every "
        "third of it",
    )
    parser.add_argument("--backends", default="cute,triton")
    parser.add_argument("--phases", default="autotune,bench")
    parser.add_argument("--only", nargs="*", default=None, help="repro ids")
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--rebench", action="store_true")
    parser.add_argument("--autotune-timeout-s", type=float, default=4 * 3600)
    parser.add_argument("--bench-timeout-s", type=float, default=3600)
    parser.add_argument("--bench-impls", default=None)
    parser.add_argument(
        "--retry-contaminated",
        type=int,
        default=2,
        help="run a job again this many times while a foreign process computes "
        "on its GPU during it (common.foreign_activity) or its activity "
        "samples do not cover it (common.activity_coverage_issue)",
    )
    args = parser.parse_args()
    try:
        renew_interval_s = parse_duration(args.board_ttl) / 3
    except ValueError as error:
        parser.error(f"--board-ttl: {error}")

    selection = json.loads(args.selection.read_text())
    entries = [
        e for e in selection["kernels"] if args.only is None or e["repro"] in args.only
    ]
    backends = [b for b in args.backends.split(",") if b]
    phases = set(args.phases.split(","))
    run_dir = args.run_dir.resolve()
    configs_dir = run_dir / "configs"
    bench_dir = run_dir / "bench"
    logs_dir = run_dir / "logs"
    for d in (configs_dir, bench_dir, logs_dir / "autotune", logs_dir / "bench"):
        d.mkdir(parents=True, exist_ok=True)

    def config_file(repro: str, shape: str, backend: str) -> Path:
        return common.config_record_path(configs_dir, repro, shape, backend)

    def record_path(job: Job) -> Path:
        if job.kind == "autotune":
            assert job.backend is not None
            return config_file(job.repro, job.shape, job.backend)
        return common.bench_record_path(bench_dir, job.repro, job.shape)

    def set_aside(job: Job, suffix: str) -> Path:
        """Move a job's record to ``<record><suffix>`` and its log along to
        ``<log><suffix>`` (each attempt keeps both, where it has them); the
        record's new path."""
        out = record_path(job)
        log = log_path(job)
        if log.exists():
            log.replace(log.with_name(f"{log.name}{suffix}"))
        kept = out.with_name(f"{out.name}{suffix}")
        if out.exists():
            out.replace(kept)
        return kept

    def next_attempt(job: Job) -> str:
        """The suffix that sets a job's record aside as its next attempt."""
        out = record_path(job)
        taken = {p.name for p in out.parent.glob(f"{out.name}.attempt*")}
        index = next(
            k for k in itertools.count(1) if f"{out.name}.attempt{k}" not in taken
        )
        return f".attempt{index}"

    def log_path(job: Job) -> Path:
        return logs_dir / job.kind / f"{record_path(job).stem}.log"

    def read(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def ok(path: Path) -> bool:
        """Whether the record at ``path`` is a finished job's (a provisional
        record stands in for a job still to run)."""
        record = read(path)
        return (
            record is not None
            and record.get("status") == "ok"
            and not record.get("provisional")
        )

    def newer_than_all(path: Path, others: list[Path]) -> bool:
        mtime = path.stat().st_mtime
        return all(not p.exists() or p.stat().st_mtime <= mtime for p in others)

    pending: list[Job] = []
    events = (run_dir / "campaign.jsonl").open("a")

    def event(**fields: Any) -> None:
        fields["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        events.write(json.dumps(fields) + "\n")
        events.flush()
        print(json.dumps(fields), flush=True)

    def settle(job: Job, gpu: str | None) -> bool:
        """Whether a finished job's record stands; a contaminated one, or one
        whose samples do not cover the job (``rerun_reason``), is set aside as
        ``<record>.attempt<k>`` and the job queued again, or, out of retries,
        the best attempt (``attempt_rank``) is kept and flagged."""
        out = record_path(job)
        record = read(out)
        if record is None:
            return True
        found = rerun_reason(record)
        if not found:
            if job.contaminated:
                record["contaminated_attempts"] = list(job.contaminated)
                common.write_json(out, record)
            return True
        kept = set_aside(job, next_attempt(job))
        attempts = (*job.contaminated, str(kept))
        if job.attempt <= args.retry_contaminated:
            pending.insert(
                0,
                dataclasses.replace(
                    job, attempt=job.attempt + 1, avoid_gpu=gpu, contaminated=attempts
                ),
            )
            event(
                kind="job_contaminated",
                job=job.key,
                gpu=gpu,
                attempt=job.attempt,
                foreign=found,
                record=str(kept),
                action="retry",
            )
            return False
        event(
            kind="job_contaminated",
            job=job.key,
            gpu=gpu,
            attempt=job.attempt,
            foreign=found,
            record=str(kept),
            action="kept_cleanest",
            kept=keep_cleanest(job, attempts),
        )
        return True

    def keep_cleanest(job: Job, attempts: tuple[str, ...]) -> str:
        """Keep the best of a job's attempts (``attempt_rank``) as its
        record, flagged contaminated; the attempt kept."""
        cleanest = min(
            attempts, key=lambda path: attempt_rank(json.loads(Path(path).read_text()))
        )
        record = json.loads(Path(cleanest).read_text())
        record.update(contaminated=True, contaminated_attempts=list(attempts))
        common.write_json(record_path(job), record)
        return cleanest

    def resumed(job: Job) -> Job:
        """``job`` resuming the contaminated attempts a stopped campaign set
        aside for it: the ``<record>.attempt<k>`` files newer than
        ``<record>.prev``, the record of the job's previous run."""
        out = record_path(job)
        prev = out.with_name(f"{out.name}.prev")
        since = prev.stat().st_mtime_ns if prev.exists() else -1
        found = []
        for path in out.parent.glob(f"{out.name}.attempt*"):
            index = path.name.removeprefix(f"{out.name}.attempt")
            if index.isdigit() and path.stat().st_mtime_ns > since:
                found.append((int(index), str(path)))
        attempts = tuple(path for _, path in sorted(found))
        if not attempts:
            return job
        return dataclasses.replace(
            job, attempt=len(attempts) + 1, contaminated=attempts
        )

    def queue(job: Job, afresh: bool) -> bool:
        """Queue ``job`` unless it has finished (always when ``afresh``);
        whether it is queued.  A job resumes the contaminated attempts left
        by a campaign that stopped while it ran again, and a record that
        campaign stopped before settling is settled now: a contaminated one
        is set aside, or, the retries spent, the cleanest attempt is kept;
        a clean one lists the attempts before it.  A job without a record
        whose attempts have spent the retries (the campaign stopped before
        keeping the cleanest, or runs with fewer retries now) keeps the
        cleanest instead of running again."""
        if afresh:
            pending.append(job)
            return True
        job = resumed(job)
        if not ok(record_path(job)):
            if (
                job.attempt > args.retry_contaminated + 1
                and read(record_path(job)) is None
            ):
                event(
                    kind="job_contaminated",
                    job=job.key,
                    gpu=None,
                    attempt=job.attempt - 1,
                    action="kept_cleanest",
                    kept=keep_cleanest(job, job.contaminated),
                )
                return False
            pending.append(job)
            return True
        record = read(record_path(job)) or {}
        if "contaminated" not in record and (
            rerun_reason(record)
            or (job.contaminated and "contaminated_attempts" not in record)
        ):
            return not settle(job, None)
        return False

    for e in entries:
        repro, shape = e["repro"], e["shape"]
        deps = []
        autotune_pending = False
        for backend in backends:
            job = Job("autotune", repro, shape, backend)
            deps.append(job.key)
            if "autotune" in phases and queue(job, args.retune):
                autotune_pending = True
        bench = Job("bench", repro, shape, None, tuple(deps))
        bench_out = record_path(bench)
        stale = ok(bench_out) and not newer_than_all(
            bench_out, [config_file(repro, shape, b) for b in backends]
        )
        if "bench" in phases:
            queue(bench, args.rebench or autotune_pending or stale)

    # Autotunes this campaign does not run count as finished (whether they
    # succeeded or not), so their benches can start.
    done: dict[str, bool] = {}
    for e in entries:
        for backend in backends:
            key = Job("autotune", e["repro"], e["shape"], backend).key
            done[key] = ok(config_file(e["repro"], e["shape"], backend))
    for job in pending:
        done.pop(job.key, None)

    def gpu_list() -> list[str]:
        spec = args.gpus
        if spec.startswith("@"):
            spec = Path(spec[1:]).read_text().replace("\n", ",")
        return [g.strip() for g in spec.split(",") if g.strip()]

    use_board = args.board_gpus is not None
    gpus = [] if use_board else gpu_list()
    # None until nvidia-smi tells a GPU's UUID (asked again before a start).
    uuids: dict[str, str | None] = {gpu: gpu_uuid(gpu) for gpu in gpus}
    running: dict[str, Running] = {}

    def ready(job: Job) -> bool:
        return all(dep in done for dep in job.deps)

    def launch(job: Job, gpu: str) -> None:
        seed = secrets.randbits(31)
        marker = secrets.token_hex(8)
        fields: dict[str, Any] = {}
        out = record_path(job)
        log = log_path(job)
        if job.kind == "autotune":
            assert job.backend is not None
            cmd = [
                sys.executable,
                str(PACKAGE_DIR / "autotune.py"),
                job.repro,
                "--shape",
                job.shape,
                "--backend",
                job.backend,
                "--output",
                str(out),
            ]
        else:
            cmd = [
                sys.executable,
                str(PACKAGE_DIR / "bench.py"),
                job.repro,
                "--shape",
                job.shape,
                "--configs-dir",
                str(configs_dir),
                "--output",
                str(out),
            ]
            if args.bench_impls:
                cmd += ["--impls", args.bench_impls]
            fields["failed_autotunes"] = [d for d in job.deps if not done[d]]
        # A rerun's old record must not outlive a rerun that fails or is
        # stopped: a restart would take it for finished and a bench would pin
        # the superseded config. Its log goes along.
        if out.exists():
            set_aside(job, ".prev")
        # The cleanup finds the job's process, cache and log through
        # ``running``: no signal between creating them and recording them.
        with stop_signals_deferred():
            cache = Path(tempfile.mkdtemp(prefix=f"bbjob-{job.repro[:20]}-"))
            handle = log.open("w")
            handle.write(f"# {' '.join(cmd)}\n# gpu={gpu} seed={seed}\n")
            handle.flush()
            proc = subprocess.Popen(
                cmd,
                env=job_env(gpu, cache, seed, marker),
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                start_new_session=True,
            )
            running[gpu] = Running(job, proc, cache, time.monotonic(), handle, marker)
        event(kind="job_start", job=job.key, gpu=gpu, seed=seed, log=str(log), **fields)

    def end_job(
        gpu: str,
        timed_out: bool = False,
        reason: str | None = None,
        survivors: bool = False,
    ) -> None:
        """Close out the job on ``gpu`` once its process has exited; its cache
        directory stays while ``survivors`` of its processes may use it."""
        job = running.pop(gpu)
        job.log.close()
        if not survivors:
            shutil.rmtree(job.cache, ignore_errors=True)
        code = job.proc.returncode
        event(
            kind="job_end" if reason is None else "job_killed",
            job=job.job.key,
            gpu=gpu,
            returncode=code,
            timed_out=timed_out,
            seconds=round(time.monotonic() - job.started, 1),
            **({} if reason is None else {"reason": reason}),
            **({"cache_kept": str(job.cache)} if survivors else {}),
        )
        if reason is not None:
            return
        if code == 0:
            if settle(job.job, gpu):
                done[job.job.key] = True
        elif job.job.contaminated:
            # A retry that failed (timed out, ran out of memory) leaves the
            # cleanest attempt before it standing, flagged, as when every
            # attempt ran again: its bench pins that rather than nothing.
            failed = set_aside(job.job, next_attempt(job.job))
            event(
                kind="job_contaminated",
                job=job.job.key,
                gpu=gpu,
                attempt=job.job.attempt,
                failed=str(failed) if failed.exists() else None,
                action="kept_cleanest",
                kept=keep_cleanest(job.job, job.job.contaminated),
            )
            done[job.job.key] = True
        else:
            done[job.job.key] = False

    def schedule() -> None:
        last_pick = 0.0
        last_renew = time.monotonic()
        while pending or running:
            for gpu, job in list(running.items()):
                timeout = (
                    args.autotune_timeout_s
                    if job.job.kind == "autotune"
                    else args.bench_timeout_s
                )
                if job.proc.poll() is None:
                    if time.monotonic() - job.started <= timeout:
                        continue
                    timed_out = True
                else:
                    timed_out = False
                # A leader that exits on its own can leave workers behind
                # (in sessions of their own): kill what is left of the job
                # before end_job removes its cache directory.
                gone = kill_job(job.proc, job.marker)
                end_job(gpu, timed_out=timed_out, survivors=not gone)
            startable = [j for j in pending if ready(j)]
            free = [g for g in gpus if g not in running]
            # nvidia-smi only when a job could start: these queries cover
            # every GPU, those whose jobs are timing included. (Each job
            # samples its own GPU throughout instead, to see foreign work
            # during its timing: common.watch_foreign_gpu_activity.)
            usable: list[str] = []
            answered = True
            if free and startable:
                for gpu in free:
                    if uuids.get(gpu) is None:
                        uuids[gpu] = gpu_uuid(gpu)
                busy = busy_uuids()
                # Without an answer from nvidia-smi nothing starts this time.
                answered = busy is not None and all(uuids.get(g) for g in free)
                usable = [
                    gpu
                    for gpu in free
                    if busy is not None
                    and uuids.get(gpu) is not None
                    and uuids[gpu] not in busy
                ]
            if use_board:
                now = time.monotonic()
                if not startable and free:
                    # Nothing can start (what is left waits on running
                    # autotunes, or nothing is left): hand idle GPUs back.
                    for gpu in free:
                        with stop_signals_deferred():
                            board_call("release", gpu)
                            gpus.remove(gpu)
                    event(kind="board_release", gpus=free)
                elif (
                    startable
                    and not usable
                    and answered
                    and len(gpus) < args.board_gpus
                    and now - last_pick > PICK_INTERVAL_S
                ):
                    # No claimed GPU can take a job: each runs one, or
                    # another account's work (or what is left of a job)
                    # keeps it busy. The new claim takes one next time.
                    # The cleanup releases the claims in ``gpus``.
                    with stop_signals_deferred():
                        picked = board_pick(args.board_purpose, args.board_ttl)
                        if picked is not None:
                            gpus.append(picked)
                    if picked is None:
                        # Nothing free: wait before asking again. After a
                        # successful pick the next one may follow at once.
                        last_pick = now
                    else:
                        uuids[picked] = gpu_uuid(picked)
                        event(kind="board_pick", gpu=picked, gpus=gpus)
                if gpus and now - last_renew > renew_interval_s:
                    last_renew = now
                    board_call("renew", "--ttl", args.board_ttl)
            elif args.gpus.startswith("@"):
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    listed = gpu_list()
                    for gpu in listed:
                        if gpu not in uuids:
                            uuids[gpu] = gpu_uuid(gpu)
                    if listed != gpus:
                        event(kind="gpus", gpus=listed)
                    gpus[:] = listed
                usable = [g for g in usable if g in gpus]
            for gpu in usable:
                if not startable:
                    break
                # A contaminated job's rerun takes another GPU when one is
                # free, else this one.
                others = [g for g in usable if g != gpu and g not in running]
                job = next(
                    (j for j in startable if j.avoid_gpu != gpu or not others),
                    None,
                )
                if job is None:
                    continue
                startable.remove(job)
                pending.remove(job)
                launch(job, gpu)
            time.sleep(POLL_S)

    def stop(outcome: str, board_usable: bool) -> list[BoardFailure]:
        """Kill and reap every running job and release the board claims;
        returns the board failures."""
        for gpu, job in list(running.items()):
            gone = kill_job(job.proc, job.marker)
            end_job(
                gpu,
                reason=outcome if gone else f"{outcome}; processes still alive",
                survivors=not gone,
            )
        problems: list[BoardFailure] = []
        if use_board and gpus and board_usable:
            for gpu in list(gpus):
                try:
                    board_call("release", gpu)
                    gpus.remove(gpu)
                except BoardSecurityAlert as alert:
                    problems.append(alert)
                    break
                except BoardFailure as failure:
                    problems.append(failure)
        if use_board:
            event(
                kind="board_release", gpus_left=gpus, failures=list(map(str, problems))
            )
        return problems

    event(kind="campaign_start", jobs=len(pending), gpus=gpus, run_dir=str(run_dir))
    for signum in STOP_SIGNALS:
        signal.signal(signum, _raise_interrupted)
    error: BaseException | None = None
    try:
        schedule()
    except BaseException as e:
        error = e
        raise
    finally:
        for signum in STOP_SIGNALS:
            signal.signal(signum, signal.SIG_IGN)
        outcome = "done" if error is None else f"{type(error).__name__}: {error}"
        problems = stop(outcome, board_usable=not isinstance(error, BoardSecurityAlert))
        event(
            kind="campaign_end",
            outcome=outcome,
            board_failures=list(map(str, problems)),
        )
        events.close()
        alert = next((p for p in problems if isinstance(p, BoardSecurityAlert)), None)
        if alert is not None:
            # Exit status 4 however the campaign came to stop.
            if error is not None:
                traceback.print_exception(error)
            for problem in problems:
                if problem is not alert:
                    print(f"campaign: also: {problem}", file=sys.stderr)
            raise alert
        if error is None and problems:
            raise BoardFailure("\n".join(map(str, problems)))
        for problem in problems:
            print(f"campaign: also: {problem}", file=sys.stderr)


def cli() -> None:
    try:
        main()
    except Interrupted as interrupted:
        print(f"campaign: stopped by {interrupted}", file=sys.stderr)
        sys.exit(128 + interrupted.signum)
    except BoardSecurityAlert as alert:
        print(
            f"campaign: GPU board SECURITY ALERT; claims left to expire:\n{alert}",
            file=sys.stderr,
        )
        sys.exit(BOARD_SECURITY_EXIT)
    except BoardFailure as failure:
        print(f"campaign: GPU board failure:\n{failure}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
