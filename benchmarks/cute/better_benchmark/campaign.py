# ruff: noqa: ANN401
"""Autotune and benchmark a selection of better-benchmark ports across GPUs.

Every job runs in a fresh process on one GPU, one job per GPU at a time. A
GPU waits while another account's processes compute on it or a process of
this account that the campaign did not start runs there; other accounts'
idle resident processes do not hold it up. GPUs come from the command line
or from the machine's GPU board:

- ``--gpus``: GPUs you hold on the board, by number. The campaign renews no
  claims in this mode, so hold these for its whole run by starting it under
  ``scripts/gpu_board.py run --gpus 0,1 -- python campaign.py --gpus 0,1
  ...``, which renews its claim for as long as the command runs, however long
  that is: the claim's ``--ttl`` (one hour unless given) is only how long it
  can outlive a ``run`` that was killed. ``run`` does not wait for the GPUs:
  it exits when one of them is busy, claimed or holds another account's idle
  processes, before the campaign starts. This is the mode for a machine where
  other accounts' jobs come and go without claims on the board: the campaign
  runs on the GPUs named and on no other.
- ``--board-gpus N``: claim free GPUs with ``scripts/gpu_board.py pick``
  while jobs can start and fewer than N are held (never shared GPUs:
  autotuning and timing need idle ones), renew the claims every third of
  ``--board-ttl`` while running, and hand a GPU back whenever no job can
  start on it, such as while the remaining benches wait for autotunes
  running elsewhere. ``pick`` takes any GPU that is free when it asks,
  whichever account uses it otherwise: one that another account's jobs
  come back to contaminates the jobs the campaign runs there, and nothing
  here keeps the campaign to a set of GPUs. ``gpu_board.py renew`` extends
  every claim of the board session, the campaign's and any other the
  session holds (run the campaign under a ``GPU_BOARD_SESSION`` of its own
  to keep others out). The campaign does not read board messages; stop it
  to give back GPUs it is using.

The jobs:

- ``autotune``: a cold full autotune (``HELION_AUTOTUNE_EFFORT=full``, forced,
  a fresh random seed or, with ``--seeds-from``, another run's, private cache
  directories) of one port on one backend, written to
  ``<run>/configs/<repro>__<shape>__<backend>.json``. The autotuner logs
  every config it benchmarks to
  ``<run>/logs/search/<repro>__<shape>__<backend>/search.csv`` (``SEARCH_LOG``;
  ``--no-search-logs`` keeps none), so a campaign's searches can be analysed
  afterwards without replaying them.
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
``<record>.attempt<k>``, its log as ``<log>.attempt<k>`` and its search logs
as ``<run>/logs/search/<record name>.attempt<k>``. The first clean,
covered attempt's record is the job's, listing the attempts before it
(``contaminated_attempts``); if every attempt was contaminated or not covered,
or a retry fails (times out, crashes), the best (a covered one first, then by
``common.contamination_rank``) is kept with that list and ``contaminated:
true``, and reports leave it out. The record kept is a copy of that
attempt's: its log and search logs stay under the attempt's names (the
``job_contaminated`` event names the attempt as ``kept``). A record and a
``job_start`` event name the files the job wrote while it ran (``log``,
``search_log``, the record's ``env.HELION_AUTOTUNE_LOG``), not where they
are once the job is set aside: add the record's suffix (``.attempt<k>``,
``.prev``) to the log and to the search logs' directory.

Finished jobs are skipped when the campaign is restarted, so it can be
interrupted and resumed; ``--retune``/``--rebench`` redo them, and a point is
benched again when one of its autotunes runs again, is newer than its bench
record, or is another record than the one that bench record's port was timed
on. The last is judged by what the records hold, not by their dates
(``common.timed_on_record``: a bench record names the kernels each port was
pinned to), so it also holds for a record that was copied, removed or dated
oddly, and for a bench that left out a port of ``--backends`` which the
benches now time and the point has an ok autotune record of (it ran with
other ``--bench-impls``). Starting a job moves its previous record to
``<record>.prev`` (and its log and search logs to ``<log>.prev`` and
``<search logs>.prev``; where the previous run left none, as with
``--no-search-logs``, an older ``.prev`` of them goes, since it is not that
record's), so a rerun that fails, times out or is stopped leaves no record
that passes for finished (or that a bench would pin). A restart resumes a job
the campaign stopped between contaminated attempts with the attempts so far
(the ``.attempt<k>`` files newer than its ``.prev``; of a bench, those timed
on the autotune records its point has now), and settles a record the campaign
stopped before settling, the last allowed attempt's included; a job without a
record whose attempts have spent the retries keeps the cleanest of them.
However the campaign stops (done, an error, SIGINT, SIGTERM or SIGHUP), it
kills its running jobs -- each job's processes, found by parent pid (the
autotuner's workers start sessions of their own) and by a marker in their
environment (a worker whose parent has exited is init's child), and their
process groups -- reaps them, removes their cache directories and releases
its board claims; only SIGKILL leaves jobs behind. A job whose leader exits
on its own has what is left of it killed the same way before its cache
directory goes.
A stop signal that arrives while a job is being started or a GPU claimed
takes effect once the job or claim is recorded, and one that arrives while a
job's record, log and search logs are set aside once all of them have moved
(also as the campaign starts, when it settles what a stopped one left). A
board failure (an error from ``pick``, a failed ``renew`` or ``release``)
stops the campaign; a SECURITY ALERT from the board stops it with exit status
4 and leaves the claims to expire.

``--configs-from DIR`` benches the picks of a finished run again without
autotuning anything: it copies the autotune records of the selected points
and ``--backends`` from DIR (that run's ``configs``) into this run's
``configs``, byte for byte, and runs the benches on them, so with a new
``--run-dir`` the records, logs and events land beside the earlier run and
nothing of it changes. Of the Helion ports its benches time those of
``--backends`` only, beside the other implementations (``bench.py --impls``
without the other backends' ports; ``--bench-impls`` narrows the list
further and may not name such a port): a bench pins no autotune record but
those the option has made DIR's. With ``--rotate-inputs N`` the benches also
time every implementation on N private input sets of its own (``bench.py``'s
``rot_us``); together they add that timing to a run's picks::

    scripts/gpu_board.py run --gpus 0,1 \\
        --purpose "better-benchmark rot_us" -- \\
        python benchmarks/cute/better_benchmark/campaign.py \\
        --selection benchmarks/cute/better_benchmark/selection.json \\
        --run-dir artifacts/bb-sweep-rot \\
        --configs-from artifacts/bb-sweep/configs --phases bench \\
        --rotate-inputs 8 --gpus 0,1

Started again, it copies the picks this run lacks and those that differ from
the copy it made of them (a pick DIR's run has autotuned again since, or
another DIR's), and removes the copies of picks that DIR does not have: the
selected records of this run are then DIR's, byte for byte, and none where
DIR has none (records it does not select, of other points or backends, stay
as they are, and no bench it starts pins one). A point whose bench record
was timed on other picks than these, or left out the port of one of them
(it was benched with other ``--backends``), is benched again, whenever the
copy was made and whatever stopped the campaign since (the rule above); a
port without a pick has its error in the bench record. The option replaces
and removes only its own copies, which
``<run>/configs_copied.json`` lists by digest (an entry is written before
its copy is made and names the copy it replaces too, so a campaign that dies
in between still knows the record it holds as its copy, and the next start
makes the entry that of the copy the run holds, or drops it where the run
holds none): where another
record of this run is not DIR's (one the run autotuned or planned, or a copy
that an autotune replaced since, that differs from DIR's or that DIR lacks),
it stops before changing anything, and so it does when DIR holds none of the
selected records (``RUN`` given for ``RUN/configs``, say). The
``configs_copied`` event names the records copied, those of them that
replaced a copy (``replaced``), the copies removed (``removed``), as
``missing`` the picks DIR does not have, and the implementations its benches
time (``bench_impls``). A finished bench does not run again for
``--rotate-inputs``: given for a run whose benches have finished, with
another N than their records have, or where the pass on N sets failed, it
starts nothing for them and names them in a ``rotate_inputs_not_added``
event (``--rebench`` runs them again).

``--seeds-from RUN`` pairs this run's searches with another run's, for a
before/after comparison of two trees. On a large kernel the pick of a cold
autotune follows the initial population its random seed draws: one seed
starts the searches of two trees from the same population and ends them at
the same level of pick, where searches from different seeds ended 4 to 10%
apart (the README has the measurements). That holds where both trees build
the same search space for the port: the seed draws the population from that
space, so a tree that adds or changes a tunable, a seed config or the way the
population is drawn starts from another one on the same seed. With the option
each autotune takes the seed of the autotune record that stands in
``RUN/configs`` for the same (repro, shape, backend)
(``common.standing_seed``): the seed in that record's ``env``, whichever
attempt made it (the first clean one, or the cleanest kept after contaminated
attempts) and whichever run (one that ``plan_run.py`` reused names its own
search's). The job searches from that seed on every start: a rerun after a
contaminated attempt and a restart of a stopped campaign with the same
arguments repeat it, where a job otherwise draws a fresh seed each time. Its
autotune record (``seed_from``) and its ``job_start`` event name RUN, and the
``seeds_from`` event lists the seeds of the autotunes the campaign is to run
and, as ``fresh``, the ones left on a fresh seed, with why RUN names none (no
ok record there; a provisional one, which is no search of that run; a search
that was given no seed). Autotunes the campaign does not run take none:
finished ones, and those whose records ``plan_run.py`` reused. The campaign
stops before changing anything when RUN names no seed for any selected
autotune (another directory than a run's, say). ``report.py --baseline RUN``
says how many of this run's autotunes searched from RUN's seeds::

    scripts/gpu_board.py run --gpus 0,1,2,3 \\
        --purpose "better-benchmark campaign" -- \\
        python benchmarks/cute/better_benchmark/campaign.py \\
        --selection benchmarks/cute/better_benchmark/selection.json \\
        --run-dir artifacts/bb-after --seeds-from artifacts/bb-before \\
        --gpus 0,1,2,3

Example, on GPUs 0 to 3 held on the board for the whole run::

    scripts/gpu_board.py run --gpus 0,1,2,3 \\
        --purpose "better-benchmark campaign" -- \\
        python benchmarks/cute/better_benchmark/campaign.py \\
        --selection benchmarks/cute/better_benchmark/selection.json \\
        --run-dir artifacts/bb-sweep --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
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
from typing import NamedTuple

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


# The base name of the files that the autotuner logs an autotune job's
# searches to (``HELION_AUTOTUNE_LOG``), in a directory of the job's own
# (``<run>/logs/search/<record name>``). The autotuner replaces the name's
# suffix: ``search.csv`` (two rows per benchmarked config, ``started`` and
# its outcome, in the order they happened: ``config_id``, ``generation``,
# ``status``, ``perf_ms``, ``compile_time_s`` and the config, under the
# ``run_id`` of the kernel's search), ``search.log`` (the search's messages,
# which the job's log has too) and, where a search records the digests of
# the code its configs generate, ``search.sources.csv``.
SEARCH_LOG = "search"
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


def job_env(
    gpu: str,
    cache: Path,
    seed: int,
    marker: str,
    seed_from: str | None = None,
    search_log: Path | None = None,
) -> dict[str, str]:
    """The environment of a job on ``gpu`` with the random seed ``seed``;
    ``seed_from`` names the run that ``--seeds-from`` took it from (for the
    autotune record), ``search_log`` the base path of the files the
    autotuner logs its searches to (``HELION_AUTOTUNE_LOG``)."""
    # Do not inherit an unrelated experiment's search budget, forced config
    # or backend, nor another campaign's word on the seed; optional package
    # paths still work.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("HELION_") and k != common.SEED_FROM
    }
    env.update(
        # The board and nvidia-smi number GPUs in PCI bus order.
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=gpu,
        HELION_AUTOTUNE_EFFORT="full",
        HELION_FORCE_AUTOTUNE="1",
        HELION_CACHE_DIR=str(cache / "helion"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        CUTE_DSL_CACHE_DIR=str(cache / "cute"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
    )
    env[common.SEED_VARIABLE] = str(seed)
    env[common.JOB_MARKER] = marker
    if seed_from is not None:
        env[common.SEED_FROM] = seed_from
    if search_log is not None:
        env["HELION_AUTOTUNE_LOG"] = str(search_log)
    return env


# What ``--configs-from`` has copied into a run, in its run directory: the
# name of each record with where it came from (``source``), the SHA-256 of
# the copy (``sha256``) and, where it took the place of an earlier copy,
# that one's SHA-256 (``replaces``). An entry is written before its copy is
# made, so a campaign that dies in between finds the record it holds under
# one of the two digests. The option replaces or removes a record only
# while the record is such a copy. The next start sets right the entry of
# a copy that was not made, or of one that was removed (the campaign died
# before the entry went): the entries of the selected records are then
# those of the copies the run holds (``ConfigCopies.entries``).
COPIED_CONFIGS = common.COPIED_CONFIGS


class ConfigCopies(NamedTuple):
    """What ``--configs-from`` does about the selected records
    (``config_copies``)."""

    # To copy from the other run, with their bytes: the run has none, or has
    # the copy that the option made of another state of the record.
    copy: dict[str, bytes]
    # Copies the option made that the other run does not have (another
    # run's pick, or one that run has set aside since): removed.
    remove: list[str]
    # The picks the other run does not have.
    missing: list[str]
    # Records of the run's own that are not the other run's (they differ
    # from its record, or it has none): the option leaves them alone, and
    # does not bench the other run's picks beside them.
    own: list[str]
    # Entries of ``COPIED_CONFIGS`` that name a copy the run does not hold,
    # where no record is to change, with what to make of each. The record
    # that the run holds instead is the other run's, byte for byte: where
    # it is the copy that the entry's was to replace, the entry is that
    # record's again; where it is a record of the run's own, or the run
    # holds none and the other run has none either, the entry goes (None).
    entries: dict[str, dict[str, str] | None]


def config_copies(
    source_dir: Path, configs_dir: Path, names: list[str], copied: dict[str, Any]
) -> ConfigCopies:
    """What ``--configs-from`` does about the records ``names`` of
    ``configs_dir``, given those of ``source_dir`` and what the option has
    copied so far (``copied``, the run's ``COPIED_CONFIGS``)."""
    found = ConfigCopies({}, [], [], [], {})
    for name in names:
        source, target = source_dir / name, configs_dir / name
        data = source.read_bytes() if source.exists() else None
        held = target.read_bytes() if target.exists() else None
        entry = copied.get(name)
        if data is None:
            found.missing.append(name)
        if held is None:
            if data is not None:
                found.copy[name] = data
            elif entry is not None:
                found.entries[name] = None
            continue
        digest = hashlib.sha256(held).hexdigest()
        if held == data:
            if entry is not None and entry["sha256"] != digest:
                found.entries[name] = (
                    {"source": str(source), "sha256": digest}
                    if entry.get("replaces") == digest
                    else None
                )
            continue
        if not common.holds_copy(entry, digest):
            found.own.append(name)
        elif data is None:
            found.remove.append(name)
        else:
            found.copy[name] = data
    return found


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
    parser.add_argument(
        "--bench-impls",
        default=None,
        help="the implementations the benches time (bench.py --impls; "
        "default: all of them, with --configs-from the Helion ports of "
        "--backends only)",
    )
    parser.add_argument(
        "--configs-from",
        type=Path,
        default=None,
        metavar="DIR",
        help="copy the selected points' autotune records from DIR, another "
        "run's configs, into this run and bench on them (needs --phases "
        "bench: nothing is autotuned)",
    )
    parser.add_argument(
        "--seeds-from",
        type=Path,
        default=None,
        metavar="RUN",
        help="give each autotune the random seed that RUN's autotune of the "
        "same port had (the record standing in RUN/configs names it), on "
        "every attempt; a port RUN names no seed for gets a fresh one",
    )
    parser.add_argument(
        "--search-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep the autotuner's per-config log of every search "
        "(HELION_AUTOTUNE_LOG) under <run>/logs/search",
    )
    parser.add_argument(
        "--rotate-inputs",
        type=int,
        default=0,
        metavar="N",
        help="benches also time every implementation on N private input sets "
        "(bench.py --rotate-inputs: rot_us)",
    )
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
    # The implementations the benches time (bench.py --impls; None: all).
    bench_impls: str | None = args.bench_impls or None
    # --configs-from: the names of the selected records, what to do about
    # them (config_copies), the copies among them that replace a copy, and
    # what the option has copied into this run so far (COPIED_CONFIGS).
    # Everything is checked here, before the run directory is made or
    # changed.
    names: list[str] = []
    copies = ConfigCopies({}, [], [], [], {})
    replaced: list[str] = []
    copied: dict[str, Any] = {}
    if args.configs_from is not None:
        if phases != {"bench"}:
            parser.error("--configs-from benches another run's picks: --phases bench")
        # Its benches time the Helion ports of --backends only. A port of
        # another backend would be pinned to whatever record of it the run
        # holds, which the option neither checks nor replaces: the copy an
        # earlier start made of another run's pick, say, timed beside the
        # picks of this start's DIR.
        ports = {f"helion_{backend}" for backend in backends}
        given = [impl for impl in (bench_impls or "").split(",") if impl]
        unselected = [
            impl for impl in given if impl.startswith("helion_") and impl not in ports
        ]
        if unselected:
            parser.error(
                f"--configs-from: --bench-impls names {', '.join(unselected)}, "
                f"and --backends {args.backends} takes no pick of that port "
                f"from {args.configs_from}: a bench would pin whatever record "
                "of it this run holds"
            )
        bench_impls = ",".join(
            impl
            for impl in given or common.ALL_IMPLS
            if not impl.startswith("helion_") or impl in ports
        )
        if not args.configs_from.is_dir():
            parser.error(f"--configs-from {args.configs_from}: no such directory")
        if args.configs_from.resolve() == configs_dir:
            parser.error("--configs-from names this run's own configs")
        if (run_dir / COPIED_CONFIGS).exists():
            copied = json.loads((run_dir / COPIED_CONFIGS).read_text())
        names = [
            common.config_record_path(configs_dir, e["repro"], e["shape"], backend).name
            for e in entries
            for backend in backends
        ]
        copies = config_copies(args.configs_from.resolve(), configs_dir, names, copied)
        if names and len(copies.missing) == len(names):
            # Another directory than a run's configs (the run itself, say):
            # every bench would time Inductor and record an error for each
            # port, and the copies this run has would go.
            parser.error(
                f"--configs-from {args.configs_from} holds none of the "
                f"{len(names)} selected autotune records ({names[0]}, say): "
                "give a run's configs directory (RUN/configs)"
            )
        if copies.own:
            # Nothing is changed: a run's own autotune record, hours of a
            # GPU, is not replaced in passing, nor benched beside another
            # run's picks as if it were one of them.
            several = len(copies.own) > 1
            parser.error(
                f"--configs-from: {len(copies.own)} autotune "
                f"record{'s' if several else ''} of this run that it did not "
                f"copy ({', '.join(copies.own)}) {'are' if several else 'is'} "
                f"not {args.configs_from}'s (it has another, or none), and it "
                "replaces and removes only its own copies: bench another "
                "run's picks in a run directory of their own (--run-dir)"
            )
        replaced = [name for name in copies.copy if (configs_dir / name).exists()]
    # --seeds-from: the seed of each selected autotune that the other run
    # names one for, by job key; for the rest, why it names none
    # (common.standing_seed). Checked before the run directory is made, too.
    seeds: dict[str, int] = {}
    unseeded: dict[str, str] = {}
    seeds_dir = None if args.seeds_from is None else args.seeds_from.resolve()
    if seeds_dir is not None:
        if "autotune" not in phases:
            parser.error("--seeds-from gives autotunes their seeds: --phases autotune")
        if seeds_dir == run_dir:
            # A job that starts again moves its record aside, so a restart
            # would no longer find the seed of a job that was running.
            parser.error("--seeds-from names this run: take another run's seeds")
        if not (seeds_dir / "configs").is_dir():
            parser.error(
                f"--seeds-from {args.seeds_from} has no configs directory: give "
                "a run directory (RUN, not RUN/configs)"
            )
        for e in entries:
            for backend in backends:
                key = Job("autotune", e["repro"], e["shape"], backend).key
                seed, why = common.standing_seed(
                    seeds_dir, e["repro"], e["shape"], backend
                )
                if seed is None:
                    assert why is not None
                    unseeded[key] = why
                else:
                    seeds[key] = seed
        if not seeds:
            # Another run directory than the one meant, or its configs: every
            # autotune would search from a fresh seed, as without the option.
            reason = next(
                (f" ({key}: {why}, say)" for key, why in unseeded.items()), ""
            )
            parser.error(
                f"--seeds-from {args.seeds_from} names no seed for any of the "
                f"{len(unseeded)} selected autotunes{reason}"
            )
    if args.rotate_inputs < 0 or args.rotate_inputs == 1:
        parser.error(f"--rotate-inputs {args.rotate_inputs}: at least 2 input sets")
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
        """Move a job's record to ``<record><suffix>``, its log along to
        ``<log><suffix>`` and an autotune's search logs to
        ``<directory><suffix>`` (each attempt keeps them all, where it has
        them); the record's new path.

        The log and the search logs that had those names go, also where the
        job has none to put there (a start with ``--no-search-logs``, a
        record that ``plan_run.py`` wrote or that was kept as the cleanest
        attempt's), so the files of one suffix are one start's. They are
        those of the run before the previous one (``.prev``) or of a retry
        that failed without a record, which left its ``.attempt<k>`` to the
        next attempt.

        A stop signal waits until all of them have moved: a campaign stopped
        after the log and before the record would, when it starts again,
        find the record in place, set the job aside once more and remove
        the log that had just been kept."""
        out = record_path(job)
        log = log_path(job)
        kept = out.with_name(f"{out.name}{suffix}")
        with stop_signals_deferred():
            kept_log = log.with_name(f"{log.name}{suffix}")
            kept_log.unlink(missing_ok=True)
            if log.exists():
                log.replace(kept_log)
            if job.kind == "autotune":
                search = search_dir(job)
                kept_search = search.with_name(f"{search.name}{suffix}")
                if kept_search.exists():
                    shutil.rmtree(kept_search)
                if search.exists():
                    search.replace(kept_search)
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

    def search_dir(job: Job) -> Path:
        """The directory of an autotune job's search logs (``--search-logs``):
        the autotuner appends every search of the job to the ``search.*``
        files there (``SEARCH_LOG``)."""
        return logs_dir / "search" / record_path(job).stem

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

    def timed_on_current(job: Job, record: dict[str, Any]) -> bool:
        """Whether ``record``, a bench record of ``job``'s point or an
        attempt at one, timed its ports on the autotune records the point
        has now (``common.timed_on_record``), those of ``--backends``: the
        records this campaign makes or copies, and under ``--configs-from``
        the only ones a bench started now would pin. ``bench.py`` records
        an entry for every implementation it is given, so a port without
        one was left out of that bench. Where a bench started now would
        leave it out too (``bench_impls``) there is nothing to compare;
        where it would time the port, the record names no autotune record
        for it, like the entry of a port that found none: it stands only
        while the point has no ok record of that port."""
        timed = None if bench_impls is None else bench_impls.split(",")
        for backend in backends:
            port = f"helion_{backend}"
            info = record["impls"].get(port)
            if info is None:
                if timed is not None and port not in timed:
                    continue
                info = {}
            tuned = read(config_file(job.repro, job.shape, backend))
            if tuned is not None and tuned.get("status") != "ok":
                tuned = None
            if not common.timed_on_record(info, tuned):
                return False
        return True

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
        ``<record>.prev``, the record of the job's previous run. An attempt
        at a bench on other autotune records than the point has now
        (``timed_on_current``) is none of this job's: its point's picks were
        replaced since, and the cleanest of such attempts must not come to
        stand for a bench on the new ones."""
        out = record_path(job)
        prev = out.with_name(f"{out.name}.prev")
        since = prev.stat().st_mtime_ns if prev.exists() else -1
        found = []
        for path in out.parent.glob(f"{out.name}.attempt*"):
            index = path.name.removeprefix(f"{out.name}.attempt")
            if not index.isdigit() or path.stat().st_mtime_ns <= since:
                continue
            attempt = read(path) if job.kind == "bench" else None
            if attempt is None or timed_on_current(job, attempt):
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
                # What a start that the campaign stopped left of itself (its
                # log, its search logs) is that start's, not the kept
                # attempt's: it goes aside like a retry that failed. Only
                # where there is something: a campaign stopped right after
                # this step repeats it, and must keep what it set aside.
                if log_path(job).exists() or (
                    job.kind == "autotune" and search_dir(job).exists()
                ):
                    set_aside(job, next_attempt(job))
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

    if args.configs_from is not None:
        # The records byte for byte. Each copy is in COPIED_CONFIGS before
        # it is made, with the digest of the copy it replaces, and a removed
        # copy leaves it after it is gone: wherever the campaign dies, the
        # records the option wrote are known as its own on the next start,
        # which also sets right the entries that such a death left without
        # their copies (copies.entries).
        # A pick the other run does not have is named: its port has nothing
        # to bench on here (the bench records that implementation's error).
        # Nothing here queues a bench: a point whose bench record was timed
        # on other records than it has now is benched again by what the
        # records hold (timed_on_current), on this start or a later one.
        for name in names:
            # What a start that died while it made a copy left of it.
            partial = configs_dir / f"{name}.tmp"
            if partial.exists():
                partial.unlink()
        for name, entry in copies.entries.items():
            if entry is None:
                del copied[name]
            else:
                copied[name] = entry
        for name, data in copies.copy.items():
            copied[name] = {
                "source": str(args.configs_from.resolve() / name),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            if name in replaced:
                copied[name]["replaces"] = hashlib.sha256(
                    (configs_dir / name).read_bytes()
                ).hexdigest()
        if copies.copy or copies.entries:
            common.write_json(run_dir / COPIED_CONFIGS, copied)
        for name, data in copies.copy.items():
            partial = configs_dir / f"{name}.tmp"
            partial.write_bytes(data)
            partial.replace(configs_dir / name)
        for name in copies.remove:
            (configs_dir / name).unlink()
            del copied[name]
        if copies.remove:
            common.write_json(run_dir / COPIED_CONFIGS, copied)
        event(
            kind="configs_copied",
            source=str(args.configs_from),
            records=list(copies.copy),
            replaced=replaced,
            removed=copies.remove,
            missing=copies.missing,
            bench_impls=(bench_impls or "").split(","),
        )

    # From here on a stop signal raises Interrupted, and waits where it is
    # deferred: queueing the jobs already sets files aside (set_aside).
    for signum in STOP_SIGNALS:
        signal.signal(signum, _raise_interrupted)
    # Finished benches that --rotate-inputs does not run again, though their
    # records lack the timing on that many sets: they have none, one on
    # another count of sets, or a pass on that count that failed.
    unrotated = []
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
        # A finished bench stands while none of the point's autotune records
        # is newer than it or another record than its port was timed on: one
        # that an autotune or --configs-from replaced, added or removed
        # since, which the dates need not show (a copy of an older record, a
        # clock that stepped back).
        finished = read(bench_out) if ok(bench_out) else None
        stale = finished is not None and not (
            newer_than_all(bench_out, [config_file(repro, shape, b) for b in backends])
            and timed_on_current(bench, finished)
        )
        if "bench" in phases:
            queued = queue(bench, args.rebench or autotune_pending or stale)
            if args.rotate_inputs and not queued:
                decision = (read(bench_out) or {}).get("rotate_inputs") or {}
                if (
                    decision.get("sets") != args.rotate_inputs
                    or decision["status"] == "error"
                ):
                    unrotated.append(bench_out.name)
    if unrotated:
        event(
            kind="rotate_inputs_not_added",
            sets=args.rotate_inputs,
            records=unrotated,
            reason="these benches have finished and do not run again, so their "
            "records keep what they have: --rebench runs them again (what they "
            "have moves to <record>.prev), a new --run-dir with --configs-from "
            "leaves this run as it is",
        )

    if seeds_dir is not None:
        # Which of the autotunes to run search from the other run's seeds,
        # and which from a fresh one, the other run naming none. Autotunes
        # that do not run (finished ones, records plan_run.py reused) take
        # none.
        queued = [job.key for job in pending if job.kind == "autotune"]
        event(
            kind="seeds_from",
            source=str(seeds_dir),
            seeds={key: seeds[key] for key in queued if key in seeds},
            fresh={key: unseeded[key] for key in queued if key in unseeded},
        )

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
        # The other run's seed (--seeds-from) on every start of the job, a
        # rerun after a contaminated attempt and a restart included: only
        # then is the search the one that run's pick came from. Any other
        # job draws a fresh seed each time.
        taken = seeds.get(job.key)
        seed = secrets.randbits(31) if taken is None else taken
        seed_from = None if taken is None else str(seeds_dir)
        marker = secrets.token_hex(8)
        fields: dict[str, Any] = {}
        if seed_from is not None:
            fields["seed_from"] = seed_from
        search_log = None
        out = record_path(job)
        log = log_path(job)
        if job.kind == "autotune":
            assert job.backend is not None
            if args.search_logs:
                search_log = search_dir(job) / SEARCH_LOG
                fields["search_log"] = str(search_log)
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
            if bench_impls:
                cmd += ["--impls", bench_impls]
            if args.rotate_inputs:
                cmd += ["--rotate-inputs", str(args.rotate_inputs)]
            fields["failed_autotunes"] = [d for d in job.deps if not done[d]]
        # A rerun's old record must not outlive a rerun that fails or is
        # stopped: a restart would take it for finished and a bench would pin
        # the superseded config. Its log goes along.
        if out.exists():
            set_aside(job, ".prev")
        if job.kind == "autotune":
            # The search logs of a start that left no record (stopped,
            # killed): the autotuner appends to the files it finds, and this
            # start's log replaces that one's as well.
            if search_dir(job).exists():
                shutil.rmtree(search_dir(job))
        # The cleanup finds the job's process, cache and log through
        # ``running``: no signal between creating them and recording them.
        with stop_signals_deferred():
            cache = Path(tempfile.mkdtemp(prefix=f"bbjob-{job.repro[:20]}-"))
            handle = log.open("w")
            source = "" if seed_from is None else f" seed_from={seed_from}"
            handle.write(f"# {' '.join(cmd)}\n# gpu={gpu} seed={seed}{source}\n")
            handle.flush()
            proc = subprocess.Popen(
                cmd,
                env=job_env(gpu, cache, seed, marker, seed_from, search_log),
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
