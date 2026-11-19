from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import itertools
import json
import math
import operator
import os
import pathlib
from pathlib import Path
import re
import secrets
import signal
import statistics
import subprocess
import sys
import threading
import time
import types
from typing import TYPE_CHECKING
from typing import cast
import warnings

from benchmarks.cute.better_benchmark import audit_activity
from benchmarks.cute.better_benchmark import autotune
from benchmarks.cute.better_benchmark import bench
from benchmarks.cute.better_benchmark import campaign
from benchmarks.cute.better_benchmark import common
from benchmarks.cute.better_benchmark import ncu_profile
from benchmarks.cute.better_benchmark import plan_run
from benchmarks.cute.better_benchmark import report
from benchmarks.cute.better_benchmark import select_kernels
import pytest
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion.autotuner import benchmarking
import helion.runtime

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.cpu_only

# Process trees, job markers and the owners of GPU processes come from /proc,
# as on the Linux machines the harness runs on (campaign.py looks for what a
# job left behind whenever one ends); the macOS CI job has none.
_requires_proc = pytest.mark.skipif(
    sys.platform != "linux", reason="reads processes from /proc"
)


def leaf(
    candidate: list[float],
    baseline: list[float],
    reference: list[float],
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, object], bool]:
    """``common._anchored_leaf`` on small CPU tensors (the reference in
    float64, like the gate's)."""
    return common._anchored_leaf(
        torch.tensor(candidate, dtype=dtype),
        torch.tensor(baseline, dtype=dtype),
        torch.tensor(reference, dtype=torch.float64),
    )


def test_threshold_is_three_times_the_baseline_error() -> None:
    reference = [1.0, 2.0, 3.0]
    baseline = [1.0, 2.0, 3.25]
    fields, passed = leaf([1.5, 2.0, 3.0], baseline, reference)
    assert passed and fields["threshold"] == 0.75
    fields, passed = leaf([1.0, 2.0, 4.0], baseline, reference)
    assert not passed and fields["err_candidate"] == 1.0


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_baseline_not_finite_where_the_reference_is_fails(bad: float) -> None:
    # An infinite baseline error would make the threshold infinite and let
    # any candidate pass ([1e20] included); a NaN one would fail every one.
    for candidate in ([1.0, 1e20], [1.0, 2.0]):
        fields, passed = leaf(candidate, [1.0, bad], [1.0, 2.0])
        assert not passed
        assert fields["baseline_nonfinite"] == 1


def test_non_finite_reference_elements_must_match() -> None:
    reference = [1.0, float("inf"), float("nan")]
    baseline = [1.0, float("inf"), float("nan")]
    _, passed = leaf([1.0, float("inf"), float("nan")], baseline, reference)
    assert passed
    _, passed = leaf([1.0, 3e38, float("nan")], baseline, reference)
    assert not passed
    fields, passed = leaf([1.0, float("inf"), 0.0], baseline, reference)
    assert not passed and "NaN" in str(fields["error"])


def test_one_bf16_ulp_is_allowed_when_the_baseline_is_exact() -> None:
    # Inductor rounds exactly like the reference, so the threshold is the
    # 1e-5 floor; a candidate one bf16 unit in the last place off passes.
    fields, passed = leaf([1.0078125], [1.0], [1.0], dtype=torch.bfloat16)
    assert passed and fields["one_ulp_flips"] == 1
    _, passed = leaf([1.015625], [1.0], [1.0], dtype=torch.bfloat16)
    assert not passed


def test_the_gate_falls_back_to_eager_with_its_reason_but_not_when_out_of_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A float64 run that fails falls back to the eager reference, which runs
    # once the failure is handled (its traceback holds the float64 copies),
    # and the verdict says why; running out of memory is no reason.
    monkeypatch.setattr(common, "bb_path", lambda: None)
    harness = types.SimpleNamespace(detect_stochastic_outputs=lambda *args: [])
    monkeypatch.setitem(sys.modules, "oracle_harness", harness)
    monkeypatch.setattr(common, "make_inputs", lambda *args, **kwargs: [torch.ones(3)])
    monkeypatch.setattr(common, "flatten_outputs", lambda value: [value])

    handling: list[object] = []

    def repro(error: Exception | None) -> torch.nn.Module:
        class Repro(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if error is not None and x.dtype == torch.float64:
                    raise error
                handling.append(sys.exc_info()[1])
                return x * 2

        return Repro()

    def gate(error: Exception | None) -> common.AnchoredGate:
        return common.AnchoredGate("r", "s", repro(error), lambda x: x * 2)

    def candidate(inputs: list[torch.Tensor]) -> torch.Tensor:
        return inputs[0] * 2

    verdict = gate(None).check(candidate)
    assert verdict["ref_precision"] == "f64_inputs" and "ref_fallback" not in verdict
    handling.clear()
    verdict = gate(RuntimeError("'Double' not implemented")).check(candidate)
    assert verdict["pass"] and verdict["ref_precision"] == "eager"
    assert verdict["ref_fallback"] == "RuntimeError: 'Double' not implemented"
    assert handling and not any(handling)
    with pytest.raises(torch.OutOfMemoryError):
        gate(torch.OutOfMemoryError("out of memory"))


def test_the_gate_judges_complex_outputs_by_their_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A complex output is gated by its real and imaginary parts, like any
    # float output, not skipped as an integer one is.
    monkeypatch.setattr(common, "bb_path", lambda: None)
    harness = types.SimpleNamespace(detect_stochastic_outputs=lambda *args: [])
    monkeypatch.setitem(sys.modules, "oracle_harness", harness)
    monkeypatch.setattr(common, "make_inputs", lambda *args, **kwargs: [torch.ones(3)])
    monkeypatch.setattr(common, "flatten_outputs", lambda value: [value])

    class Repro(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.complex(x, -2 * x)

    gate = common.AnchoredGate("r", "s", Repro(), lambda x: torch.complex(x, -2 * x))
    right = gate.check(lambda inputs: torch.complex(inputs[0], -2 * inputs[0]))
    assert right["pass"] and right["per_output"][0]["ref_dtype"] == "complex128"
    wrong = gate.check(lambda inputs: torch.complex(inputs[0], 2 * inputs[0]))
    assert not wrong["pass"]


@pytest.mark.parametrize("shape", [(), (7,), (5, 3), (2, 9), (3, 0), (2, 3, 4)])
def test_slices_cover_each_element_once_within_the_budget(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    values = torch.arange(math.prod(shape)).reshape(shape)
    for budget in (1, 2, 4, 5, 100):
        monkeypatch.setattr(common, "CHECK_SLICE_ELEMENTS", budget)
        parts = [values[part].flatten() for part in common._slices(shape)]
        assert all(part.numel() <= budget for part in parts)
        assert torch.equal(torch.cat([values.new_empty(0), *parts]), values.flatten())


class LargestTensor(TorchDispatchMode):
    """Records the most elements of any tensor an op returns, views (which
    allocate nothing) aside."""

    def __init__(self) -> None:
        super().__init__()
        self.largest = 0

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        out = func(*args, **(kwargs or {}))
        if not func.is_view:
            for value in pytree.tree_leaves(out):
                if isinstance(value, torch.Tensor):
                    self.largest = max(self.largest, value.numel())
        return out


@pytest.mark.parametrize("budget", [7, 25])
def test_numerics_checks_work_slice_by_slice(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    # sum_b2873b4c9052's output has 2**31 elements, and a float64 copy of a
    # whole output would take 16 GiB. Sliced (budgets of 7 and 25 elements
    # split the 24 x 10 outputs within and across rows), the checks give the
    # same results, and no op allocates a larger tensor.
    monkeypatch.setattr(common, "flatten_outputs", list)
    torch.manual_seed(0)
    reference = torch.randn(24, 10, dtype=torch.float64)
    reference[3, 2] = float("inf")
    reference[17, 9] = float("nan")
    baseline = reference.to(torch.bfloat16)
    candidate = baseline.clone()
    candidate.view(torch.int16)[20, 1] += 1  # one bf16 unit in the last place
    candidate = candidate.t().contiguous().t()  # another layout
    wrong = candidate.clone()
    wrong[9, 4] += 1
    infinite = wrong.where(wrong < 1, torch.inf)
    not_a_number = wrong.where(wrong < 1, torch.nan)
    masks = (candidate > 0, baseline > 0.25)

    def checks() -> list[object]:
        results: list[object] = [
            common._anchored_leaf(candidate, baseline, reference),
            common._anchored_leaf(wrong, baseline, reference),
            common._anchored_leaf(candidate, infinite, reference),
            common._anchored_leaf(not_a_number, baseline, reference),
            common._stochastic_leaf(*masks),
            common.output_errors(
                [wrong, not_a_number, masks[0]], [reference] * 2 + [masks[1]]
            ),
        ]
        for actual, expected in ((wrong, candidate), masks):
            with pytest.raises(AssertionError) as error:
                common.check_outputs([actual], [expected], check_strides=False)
            # The message's statistics; a sliced output's omits the indices.
            found = re.search(
                r"Mismatched elements: \d+ / \d+|\d+ of \d+ elements", str(error.value)
            )
            results += [type(error.value), found and found.group()]
        return results

    whole = checks()
    monkeypatch.setattr(common, "CHECK_SLICE_ELEMENTS", budget)
    with LargestTensor() as mode:
        sliced = checks()
    assert str(sliced) == str(whole)
    assert mode.largest <= budget


@pytest.fixture
def fake_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """capture_graph without CUDA: the capture context yields a stand-in."""

    @contextlib.contextmanager
    def cute_cuda_graph() -> object:
        yield "graph"

    monkeypatch.setattr(helion.runtime, "cute_cuda_graph", cute_cuda_graph)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)


def test_capture_graph_keeps_the_collector_off(fake_capture: None) -> None:
    # A collection during a capture can release an earlier capture's CUDA
    # state, which invalidates the capture.
    assert gc.isenabled()
    seen: list[bool] = []
    graph, result = common.capture_graph(lambda: seen.append(gc.isenabled()) or 7)
    assert (graph, result, seen) == ("graph", 7, [False])
    assert gc.isenabled()

    def fail() -> None:
        raise ValueError("capture failed")

    with pytest.raises(ValueError, match="capture failed"):
        common.capture_graph(fail)
    assert gc.isenabled()


def test_a_failed_capture_restores_the_callers_stream(
    fake_capture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not the default stream, which the caller may not have been on.
    streams = ["caller's"]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: streams[-1])
    monkeypatch.setattr(torch.cuda, "set_stream", streams.append)

    def fail() -> None:
        streams.append("capture")
        raise ValueError("capture failed")

    with pytest.raises(ValueError, match="capture failed"):
        common.capture_graph(fail)
    assert streams[-1] == "caller's"


def test_gc_paused_restores_the_outer_state() -> None:
    try:
        with common.gc_paused():
            with common.gc_paused():
                assert not gc.isenabled()
            assert not gc.isenabled()
        assert gc.isenabled()
        gc.disable()
        with common.gc_paused():
            pass
        assert not gc.isenabled()
    finally:
        gc.enable()


# A job as the campaign starts one, in a session of its own, with a worker in
# another session (the autotuner's benchmark worker and precompile and render
# children call os.setsid), which starts a compiler in its group.  The worker
# writes its and the compiler's pids to argv[1].
_JOB = """
import os, subprocess, sys, time
if os.fork() == 0:
    os.setsid()
    compiler = subprocess.Popen(["sleep", "60"])
    with open(sys.argv[1], "w") as f:
        f.write(f"{os.getpid()} {compiler.pid}")
    time.sleep(60)
    os._exit(0)
time.sleep(60)
"""


def _worker_pids(path: Path) -> list[int]:
    deadline = time.monotonic() + 30
    while len(pids := (path.read_text().split() if path.exists() else [])) < 2:
        assert time.monotonic() < deadline, "the job did not start its worker"
        time.sleep(0.05)
    return [int(pid) for pid in pids]


@pytest.mark.slow
@_requires_proc
def test_kill_job_kills_the_workers_in_sessions_of_their_own(tmp_path: Path) -> None:
    # The job's workers start sessions of their own, out of its process
    # group: kill_job finds them by parent pid and kills them with their
    # groups.
    job = subprocess.Popen(
        [sys.executable, "-c", _JOB, str(tmp_path / "pids")], start_new_session=True
    )
    worker, compiler = _worker_pids(tmp_path / "pids")
    try:
        assert {job.pid, worker, compiler} <= common.process_tree(job.pid)
        assert campaign.kill_job(job, grace_s=10)
        assert job.returncode == -signal.SIGKILL
        assert not campaign._lives_on({worker, compiler}, set())
    finally:
        for pid in (job.pid, worker, compiler):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


# A job leader that exits on its own, leaving a worker behind in a session of
# its own (init's child once its parent exits); the worker's pid goes to
# argv[1].
_LEAVER = """
import os, subprocess, sys
if os.fork() == 0:
    os.setsid()
    worker = subprocess.Popen(["sleep", "60"])
    with open(sys.argv[1], "w") as f:
        f.write(str(worker.pid))
    os._exit(0)
os.wait()
"""


@pytest.mark.slow
@_requires_proc
def test_kill_job_kills_the_workers_a_leader_left_behind(tmp_path: Path) -> None:
    # Out of the reaped leader's tree, the worker carries the job's marker.
    marker = secrets.token_hex(8)
    job = subprocess.Popen(
        [sys.executable, "-c", _LEAVER, str(tmp_path / "pid")],
        env={**os.environ, common.JOB_MARKER: marker},
        start_new_session=True,
    )
    job.wait()
    worker = int((tmp_path / "pid").read_text())
    try:
        assert common.marked_processes(marker) == {worker}
        assert campaign.kill_job(job, marker, grace_s=10)
        assert not campaign._lives_on({worker}, set())
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(worker, signal.SIGKILL)


@_requires_proc
def test_kill_job_reports_a_leader_it_cannot_reap_in_time() -> None:
    # A leader in uninterruptible I/O (D state) dies only when that returns:
    # kill_job reports it as a survivor after its grace period, so the
    # scheduler goes on enforcing timeouts and renewing claims.
    child = subprocess.Popen(["sleep", "60"], start_new_session=True)

    class Stuck:
        pid = child.pid
        returncode = None

        def wait(self, timeout: float | None = None) -> int:
            if timeout is None:
                raise AssertionError("an unbounded wait blocks the scheduler")
            raise subprocess.TimeoutExpired("sleep", timeout)

    try:
        stuck = cast("subprocess.Popen[bytes]", Stuck())
        assert not campaign.kill_job(stuck, grace_s=0.5)
    finally:
        child.kill()
        child.wait()


@_requires_proc
def test_a_jobs_own_workers_are_not_foreign(monkeypatch: pytest.MonkeyPatch) -> None:
    # The job's benchmark worker, in a session of its own, holds a context on
    # the job's GPU: it is the job's own, not a foreign process.
    worker = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        apps = f"GPU-a, {worker.pid}\nGPU-a, {os.getppid()}\nGPU-b, 1\n"
        monkeypatch.setattr(common, "gpu_info", lambda: {"uuid": "GPU-a"})
        monkeypatch.setattr(common, "_nvidia_smi", lambda *args: apps)
        owner = common._process_owner(os.getppid())
        assert common.foreign_gpu_processes() == [f"{owner}:{os.getppid()}"]
    finally:
        worker.kill()
        worker.wait()


_PMON = """\
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    6    {own}     C     90     30      -      -      -      -    python
    6    {other}     C     {sm}     18      -      -      -      -    python
"""


def test_pmon_samples_read_each_processes_sm_utilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {
        "GPU-a": _PMON.format(own=11, other=22, sm="-"),
        # A GPU without processes; a GPU nvidia-smi fails on; a row cut short,
        # read as an unknown sample.
        "GPU-b": "# gpu pid type sm command\n# Idx # C/G % name\n 0 - - - -\n",
        "GPU-c": None,
        "GPU-d": _PMON.format(own=11, other=22, sm="-").rsplit("\n", 2)[0]
        + "\n 6 33\n",
    }
    monkeypatch.setattr(common, "_nvidia_smi", lambda *args: outputs[args[2]])
    assert common._pmon_sample("GPU-a") == [(11, 90), (22, 0)]
    assert common._pmon_sample("GPU-b") == []
    assert common._pmon_sample("GPU-c") is None
    assert common._pmon_sample("GPU-d") is None


@_requires_proc
def test_foreign_gpu_activity_tells_computing_from_idle_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A resident foreign process idles ("-"), then computes; the record keeps
    # its peak SM utilization and active time, leaves the job's own process
    # out and counts the failed sample as unknown.
    own, other = os.getpid(), os.getppid()
    outputs = [_PMON.format(own=own, other=other, sm=sm) for sm in ("-", "3", "40")]
    outputs.append(None)
    sampled = threading.Semaphore(0)

    def pmon(*args: str) -> str | None:
        out = outputs.pop(0) if outputs else _PMON.format(own=own, other=other, sm="-")
        sampled.release()
        return out

    owner = common._process_owner
    looked_up = []

    def process_owner(pid: int) -> str:
        looked_up.append(pid)
        return owner(pid)

    monkeypatch.setattr(common, "gpu_info", lambda: {"uuid": "GPU-a"})
    monkeypatch.setattr(common, "_nvidia_smi", pmon)
    monkeypatch.setattr(common, "_process_owner", process_owner)
    monkeypatch.setattr(common, "FOREIGN_ACTIVITY_INTERVAL_S", 0.1)
    record: dict[str, object] = {}
    with common.watch_foreign_gpu_activity(record):
        for _ in range(4):
            assert sampled.acquire(timeout=30)
    activity = record["foreign_gpu_activity"]
    assert isinstance(activity, dict)
    assert activity["samples"] >= 5 and activity["unknown_samples"] == 1
    (name, entry), *rest = activity["processes"].items()
    assert not rest and name == f"{owner(other)}:{other}"
    # The 3% sample is active, but below the 5% that busy_s counts.
    assert entry["max_sm"] == 40 and entry["active_s"] > entry["busy_s"] > 0
    # Once, not at each of its samples: each lookup costs an os.stat and a
    # user database query.
    assert looked_up == [other]

    # Switched off, nothing samples.
    monkeypatch.setenv("BB_FOREIGN_ACTIVITY", "0")
    outputs.append(None)
    with common.watch_foreign_gpu_activity(record):
        pass
    assert record["foreign_gpu_activity"] is None and outputs == [None]


@_requires_proc
def test_foreign_gpu_activity_leaves_out_processes_gone_before_their_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A worker of the job that pmon saw computing but that exited before the
    # job's processes were read is left out, not taken for a foreign process.
    gone = subprocess.Popen(["true"])
    gone.wait()
    other = os.getppid()
    sampled = threading.Semaphore(0)

    def pmon(*args: str) -> str:
        sampled.release()
        return _PMON.format(own=gone.pid, other=other, sm="-")

    monkeypatch.setattr(common, "gpu_info", lambda: {"uuid": "GPU-a"})
    monkeypatch.setattr(common, "_nvidia_smi", pmon)
    monkeypatch.setattr(common, "FOREIGN_ACTIVITY_INTERVAL_S", 0.1)
    record: dict[str, object] = {}
    with common.watch_foreign_gpu_activity(record):
        assert sampled.acquire(timeout=30)
    activity = record["foreign_gpu_activity"]
    assert isinstance(activity, dict)
    assert list(activity["processes"]) == [f"{common._process_owner(other)}:{other}"]
    assert common.foreign_activity(record) == []


@pytest.mark.slow
@_requires_proc
def test_the_workers_a_leader_left_behind_are_not_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Out of the job's tree, such a worker carries the job's marker, which
    # makes it the job's own in both lists.
    monkeypatch.setenv(common.JOB_MARKER, secrets.token_hex(8))
    leader = subprocess.Popen(
        [sys.executable, "-c", _LEAVER, str(tmp_path / "pid")],
        start_new_session=True,
    )
    leader.wait()
    worker = int((tmp_path / "pid").read_text())
    try:
        assert worker not in common.process_tree(os.getpid())
        other = os.getppid()
        sampled = threading.Semaphore(0)

        def nvidia_smi(*args: str) -> str:
            if args[0] == "pmon":
                sampled.release()
                return _PMON.format(own=worker, other=other, sm="40")
            return f"GPU-a, {worker}\nGPU-a, {other}\n"

        monkeypatch.setattr(common, "gpu_info", lambda: {"uuid": "GPU-a"})
        monkeypatch.setattr(common, "_nvidia_smi", nvidia_smi)
        monkeypatch.setattr(common, "FOREIGN_ACTIVITY_INTERVAL_S", 0.1)
        name = f"{common._process_owner(other)}:{other}"
        assert common.foreign_gpu_processes() == [name]
        record: dict[str, object] = {}
        with common.watch_foreign_gpu_activity(record):
            assert sampled.acquire(timeout=30)
        activity = record["foreign_gpu_activity"]
        assert isinstance(activity, dict)
        assert list(activity["processes"]) == [name]
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(worker, signal.SIGKILL)


@_requires_proc
def test_foreign_gpu_activity_survives_a_failing_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A sample that raises counts as unknown and the sampling goes on, so no
    # record reads "sampled and clean" for a job nothing watched; the record
    # tells how long the job ran and its longest stretch without a sample.
    failed = threading.Event()
    sampled = threading.Semaphore(0)

    def pmon(uuid: str) -> list[tuple[int, int]]:
        sampled.release()
        if not failed.is_set():
            failed.set()
            raise IndexError("list index out of range")
        return []

    monkeypatch.setattr(common, "gpu_info", lambda: {"uuid": "GPU-a"})
    monkeypatch.setattr(common, "_pmon_sample", pmon)
    monkeypatch.setattr(common, "FOREIGN_ACTIVITY_INTERVAL_S", 0.05)
    record: dict[str, object] = {}
    with common.watch_foreign_gpu_activity(record):
        for _ in range(3):
            assert sampled.acquire(timeout=30)
    activity = record["foreign_gpu_activity"]
    assert isinstance(activity, dict)
    assert activity["samples"] >= 4 and activity["unknown_samples"] == 1
    assert 0 < activity["max_gap_s"] <= activity["watched_s"] < 30
    assert common.activity_coverage_issue(record) is None
    assert common.foreign_activity(record) == []


@_requires_proc
def test_foreign_gpu_activity_finds_its_gpu_after_a_failed_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # nvidia-smi does not answer the first two lookups of the job's GPU: the
    # sampling asks again instead of leaving every sample of the job unknown.
    lookups = [{}, {}]
    sampled = threading.Semaphore(0)

    def pmon(uuid: str) -> list[tuple[int, int]]:
        assert uuid == "GPU-a"
        sampled.release()
        return [(os.getppid(), 40)]

    monkeypatch.setattr(
        common, "gpu_info", lambda: lookups.pop() if lookups else {"uuid": "GPU-a"}
    )
    monkeypatch.setattr(common, "_pmon_sample", pmon)
    monkeypatch.setattr(common, "FOREIGN_ACTIVITY_INTERVAL_S", 0.05)
    record: dict[str, object] = {}
    with common.watch_foreign_gpu_activity(record):
        for _ in range(2):
            assert sampled.acquire(timeout=30)
    activity = record["foreign_gpu_activity"]
    assert isinstance(activity, dict)
    assert activity["unknown_samples"] == 2 and activity["samples"] >= 4
    assert len(activity["processes"]) == 1


def _activity_record(
    samples: int, unknown: int, *, wall_s: float = 60.0, **coverage: float
) -> dict[str, object]:
    return {
        "wall_s": wall_s,
        "foreign_gpu_activity": {
            "interval_s": 1.0,
            "samples": samples,
            "unknown_samples": unknown,
            **coverage,
            "processes": {"eve:7": {"max_sm": 50, "active_s": 2.0}},
        },
    }


@pytest.mark.parametrize(
    ("record", "issue"),
    [
        ({"wall_s": 60.0}, "no activity samples"),
        (_activity_record(40, 40), "none of 40 samples read the GPU"),
        (_activity_record(40, 3), "3 of 40 samples failed"),
        (_activity_record(400, 20, wall_s=600.0), None),
        (
            _activity_record(40, 0, watched_s=60.0, max_gap_s=45.0),
            "45 s without a sample",
        ),
        (_activity_record(40, 0, watched_s=60.0, max_gap_s=3.0), None),
        # Recorded before the gaps were: one sample per 6 s at least.
        (_activity_record(10, 0, wall_s=100.0), "10 samples over 100 s"),
        (_activity_record(20, 1, wall_s=100.0), None),
    ],
)
def test_activity_samples_must_cover_the_job(
    record: dict[str, object], issue: str | None
) -> None:
    # A record whose samples do not cover its job cannot tell whether a
    # foreign process computed: too many failed samples, a long gap or too
    # few samples for its wall time leave it unsampled, not clean.
    assert common.activity_coverage_issue(record) == issue
    found = common.foreign_activity(record)
    assert found is None if issue else found == ["eve:7 (50% SM, 2.0 s)"]


@pytest.mark.parametrize(
    ("watched_s", "busy_s", "contaminated"),
    [
        # 1% of an hour, 36 s, is the bar; ten minutes' 6 s is below 10 s.
        (3600.0, 35.9, False),
        (3600.0, 36.0, True),
        (600.0, 9.9, False),
        (600.0, 10.0, True),
    ],
)
def test_an_autotune_record_needs_foreign_compute_over_a_share_of_its_job(
    watched_s: float, busy_s: float, contaminated: bool
) -> None:
    # The seconds at 5% SM or more count, summed over the processes, against
    # 1% of the job and 10 s: one busy sample does not void a long autotune.
    # A bench record keeps the one-sample rule, and so does a record made
    # before busy_s.
    activity = _activity_record(400, 0, watched_s=watched_s, max_gap_s=3.0)
    processes = {
        "eve:7": {"max_sm": 50, "active_s": 60.0, "busy_s": busy_s / 2},
        "bob:8": {"max_sm": 90, "active_s": 60.0, "busy_s": busy_s / 2},
        "eve:9": {"max_sm": 3, "active_s": 600.0, "busy_s": 0.0},
    }
    activity["foreign_gpu_activity"]["processes"] = processes
    found = common.foreign_activity(activity)
    assert found == (
        ["eve:7 (50% SM, 60.0 s)", "bob:8 (90% SM, 60.0 s)"] if contaminated else []
    )
    assert common.contamination_rank(activity) == (
        pytest.approx(84.0) if contaminated else 0
    )
    assert len(common.foreign_activity({**activity, "impls": {}}) or []) == 2
    for entry in processes.values():
        del entry["busy_s"]
    assert len(common.foreign_activity(activity) or []) == 2


def test_the_campaign_waits_for_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without an answer from nvidia-smi (common._nvidia_smi: a failure or a
    # timeout), the scheduler's queries give None and nothing starts.
    monkeypatch.setattr(common, "_nvidia_smi", lambda *args: None)
    assert campaign.gpu_uuid("0") is None
    assert campaign.busy_uuids() is None


def test_no_job_starts_beside_work_that_would_contaminate_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Another account's processes keep a GPU busy at the 5% that would
    # contaminate a job's record. They are pid 1's on GPUs a, b and c, which
    # is no process of this account's whoever runs the test (CI runs as
    # root): no uid owns it.
    monkeypatch.setattr(os, "getuid", lambda: -1)
    usage = "GPU-a, 4\nGPU-b, 5\nGPU-c, 9\nGPU-d, 0\n"
    apps = "GPU-a, 1\nGPU-b, 1\nGPU-c, 1\n"
    outputs = {"--query-compute-apps=gpu_uuid,pid": apps}
    outputs["--query-gpu=uuid,utilization.gpu"] = usage
    monkeypatch.setattr(common, "_nvidia_smi", lambda *args: outputs[args[0]])
    assert campaign.busy_uuids() == {"GPU-b", "GPU-c"}


def test_the_audit_flags_records_whose_samples_miss_their_jobs(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    covered = _activity_record(40, 0, watched_s=60.0, max_gap_s=3.0)
    covered["foreign_gpu_activity"]["processes"] = {}
    _toy_record(run, "configs", "a__s__cute.json", **covered)
    _toy_record(
        run, "configs", "b__s__cute.json", **_activity_record(10, 0, wall_s=100.0)
    )
    _toy_record(run, "bench", "a__s.json", **covered)
    _toy_record(run, "bench", "c__s.json")
    # The thread's traceback in the job's log flags a record that reads
    # covered.
    log = run / "logs" / "bench" / "a__s.log"
    log.parent.mkdir(parents=True)
    log.write_text("Exception in thread Thread-1 (poll):\nTraceback ...\n")
    # So does an attempt's log, set aside with its record.
    _toy_record(run, "bench", "a__s.json.attempt1", **covered)
    log.with_name("a__s.log.attempt1").write_text(log.read_text())
    # A foreign process at 50% SM for 2 s contaminates a bench record, not an
    # autotune record, as the campaign has it.
    blip = _activity_record(40, 0, watched_s=600.0, max_gap_s=3.0)
    blip["foreign_gpu_activity"]["processes"]["eve:7"]["busy_s"] = 2.0
    _toy_record(run, "configs", "d__s__cute.json", **blip)
    _toy_record(run, "bench", "d__s.json", impls={}, **blip)
    issues = {
        os.path.basename(row["record"]): row["issue"]
        for row in audit_activity.audit(run)
    }
    assert issues == {
        "a__s__cute.json": None,
        "b__s__cute.json": "10 samples over 100 s",
        "d__s__cute.json": None,
        "a__s.json": "the sampling thread died",
        "a__s.json.attempt1": "the sampling thread died",
        "c__s.json": "no activity samples",
        "d__s.json": "contaminated: eve:7 (50% SM, 2.0 s)",
    }


def test_the_audit_fails_only_on_standing_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The attempts a campaign set aside were contaminated or uncovered, which
    # is why they were set aside: only a flagged standing record fails the
    # audit, and --all lists the attempts too.
    run = tmp_path / "run"
    covered = _activity_record(40, 0, watched_s=60.0, max_gap_s=3.0)
    covered["foreign_gpu_activity"]["processes"] = {}
    _toy_record(run, "configs", "a__s__cute.json", **covered)
    _toy_record(run, "configs", "a__s__cute.json.attempt1", **_activity_record(9, 9))

    def audit(*flags: str) -> tuple[object, str]:
        monkeypatch.setattr(sys, "argv", ["audit_activity.py", str(run), *flags])
        with pytest.raises(SystemExit) as exited:
            audit_activity.main()
        return exited.value.code, capsys.readouterr().out

    code, out = audit()
    assert code == 0 and "attempt1" not in out
    code, out = audit("--all")
    assert code == 0 and "a__s__cute.json.attempt1: none of 9 samples" in out
    _toy_record(run, "configs", "b__s__cute.json", **_activity_record(9, 9))
    code, out = audit()
    assert code == 1 and "b__s__cute.json: none of 9 samples" in out


def test_only_significant_strides_must_match_eagers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without better-benchmark's harness, which flattens nested outputs.
    monkeypatch.setattr(common, "flatten_outputs", lambda value: [value])
    expected = torch.zeros(4, 1, 3)
    # A size-1 dim's stride carries no meaning, for the errors recorded too.
    size_one_apart = torch.zeros(12).as_strided((4, 1, 3), (3, 7, 1))
    common.check_outputs(size_one_apart, expected)
    (row,) = common.output_errors(size_one_apart, expected)
    assert row["stride_match"]
    permuted = torch.zeros(3, 1, 4).permute(2, 1, 0)
    with pytest.raises(common.StructureMismatch, match=r"\(dims \[0, 2\]\)"):
        common.check_outputs(permuted, expected)
    (row,) = common.output_errors(permuted, expected)
    assert not row["stride_match"]


def test_the_errors_recorded_against_eager_see_every_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The recorded errors see infinite expected elements and complex outputs'
    # imaginary parts, and a different output count raises StructureMismatch.
    monkeypatch.setattr(common, "flatten_outputs", list)
    inf = float("inf")
    expected = [torch.tensor([1.0, inf, -inf]), torch.tensor([1 + 1j, 2 - 1j])]
    rows = common.output_errors(
        [torch.tensor([1.0, inf, -inf]), torch.tensor([1 + 1j, 2 - 1j])], expected
    )
    assert [row["max_abs"] for row in rows] == [0.0, 0.0]
    assert not any(row["inf_mismatch"] for row in rows)
    for actual in ([1.0, 0.0, -inf], [1.0, -inf, -inf], [1.0, inf, inf]):
        (row, _) = common.output_errors([torch.tensor(actual), expected[1]], expected)
        assert row["inf_mismatch"] and row["max_abs"] == 0.0
    (_, row) = common.output_errors(
        [expected[0], torch.tensor([1 + 1j, 2 + 1j])], expected
    )
    assert row["max_abs"] == 2.0 and row["max_rel"] == 1.0
    with pytest.raises(common.StructureMismatch, match="1 outputs, expected 2"):
        common.output_errors([expected[0]], expected)


def test_the_better_benchmark_revision_is_checked_once_wherever_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The check runs once, also when the checkout is on sys.path already
    # (PYTHONPATH), and a checkout with uncommitted changes at the pinned
    # commit is at that commit.
    pinned = json.loads(common.SELECTION_PATH.read_text())["better_benchmark_commit"]
    (tmp_path / "repro_harness.py").write_text("")
    monkeypatch.setattr(common, "BB_ROOT", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    revisions = [f"{pinned}-dirty", "0" * 40]
    monkeypatch.setattr(common, "git_revision", lambda path: revisions.pop(0))
    common._check_bb_revision.cache_clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            common.bb_path()
            common.bb_path()
        assert not caught and revisions == ["0" * 40]
        common._check_bb_revision.cache_clear()
        with pytest.warns(UserWarning, match="were made against"):
            common.bb_path()
    finally:
        common._check_bb_revision.cache_clear()


def test_the_selection_and_the_ports_match() -> None:
    # Every measured point has its port and every port its point: 48 picked
    # by select_kernels.py, then the extras.
    kernels = json.loads(common.SELECTION_PATH.read_text())["kernels"]
    ports = {path.stem for path in common.KERNELS_DIR.glob("*.py")}
    assert sorted(k["repro"] for k in kernels) == sorted(ports)
    assert [k["reason"] == "extra" for k in kernels] == [False] * 48 + [True] * 3


def test_a_selection_pins_only_a_committed_better_benchmark(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Uncommitted repros cannot be pinned: the selection refuses the checkout
    # before reading the sweep.
    monkeypatch.setattr(common, "git_revision", lambda path: "0" * 40 + "-dirty")
    argv = ["select_kernels.py", "--sweep", "nowhere.json", "--output", "out.json"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        select_kernels.main()
    assert "uncommitted changes" in capsys.readouterr().err


# A small kernel's cold rounds as a campaign recorded them before the order
# balancing (var_mean_f4578c128d74's Helion+CuTe): the forward-order rounds
# (0, 2, 4) read about a microsecond above the reverse-order ones.
_RECORDED_COLD_ROUNDS = [11.27, 10.23, 11.26, 10.39, 11.27]


def test_order_balanced_averages_each_orders_statistic() -> None:
    cold, orders = common.order_balanced(_RECORDED_COLD_ROUNDS, statistics.median)
    assert orders == [11.27, pytest.approx(10.31)]
    assert cold == pytest.approx(10.79)
    # The median over all five rounds was a forward-order reading.
    assert statistics.median(_RECORDED_COLD_ROUNDS) == 11.26
    assert common.order_balanced([3.0, 1.0, 2.0, 5.0], min) == (1.5, [2.0, 1.0])
    assert common.order_balanced([4.0], statistics.median) == (4.0, [4.0])


def test_bench_takes_only_positive_even_round_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Checked before anything compiles: no rounds would fail only after every
    # implementation ran its checks, and an odd count leans to one order.
    for flags in (["--rounds", "0"], ["--cold-rounds", "3"]):
        argv = ["bench.py", "r", "--configs-dir", "c", "--output", "o", *flags]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            bench.main()
        assert "a positive, even count is needed" in capsys.readouterr().err
    with pytest.raises(ValueError, match="at least one round"):
        common.order_balanced([], min)


def test_report_rebalance_recomputes_the_timings_from_the_rounds() -> None:
    record = {
        "impls": {
            "helion_cute": {
                "cold_us": 11.26,
                "cold_runs_us": list(_RECORDED_COLD_ROUNDS),
                "bb_us": 13.98,
                "bb_runs_us": [14.11, 14.08, 14.05, 13.98, 14.08, 14.02],
            },
            "memcopy": {"cold_us": 8.0, "bb_us": 11.0},
        }
    }
    info = report.rebalance(record)["impls"]["helion_cute"]
    assert info["cold_us"] == pytest.approx(10.79)
    assert info["cold_order_medians_us"] == [11.27, pytest.approx(10.31)]
    assert info["bb_us"] == pytest.approx(14.015)
    assert info["bb_order_mins_us"] == [14.05, 13.98]
    assert info["cold_runs_us"] == _RECORDED_COLD_ROUNDS
    # Without recorded rounds a timing stays as recorded.
    assert record["impls"]["memcopy"] == {"cold_us": 8.0, "bb_us": 11.0}


def test_time_impls_averages_the_two_round_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fns = {"first": lambda: None, "second": lambda: None}
    names = {id(fn): name for name, fn in fns.items()}

    class Graph:
        def __init__(self, name: str) -> None:
            self.name = name

        def replay(self) -> None:
            pass

    monkeypatch.setattr(
        common, "capture_graph", lambda fn: (Graph(names[id(fn)]), fn())
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(bench, "steady_graph_dispatch", lambda: "steady")
    monkeypatch.setattr(
        bench, "_profile_kernels", lambda replay: {"kernel_us": 1.0, "kernels": []}
    )
    # Each implementation reads 1 us more in the cold rounds' forward order,
    # and its bb reading alternates between two values round by round.
    cold = {"first": 0.005, "second": 0.007}

    def interleaved_bench(
        order: list[Callable[[], object]], **kwargs: object
    ) -> list[float]:
        # The rounds whose sample count rot_us takes over (rotating_samples).
        assert kwargs == {
            "repeat": bench.COLD_REPEAT,
            "default_cudagraph": True,
            "l2_flush_mode": "call",
        }
        forward = names[id(order[0])] == "first"
        return [cold[names[id(fn)]] + (0.001 if forward else 0.0) for fn in order]

    bb = {"first": iter([0.010, 0.012] * 3), "second": iter([0.020, 0.021] * 3)}

    def do_bench(replay: Callable[[], None], **kwargs: object) -> float:
        return next(bb[replay.__self__.name])  # pyrefly: ignore[missing-attribute]

    monkeypatch.setattr(bench, "interleaved_bench", interleaved_bench)
    monkeypatch.setattr(bench, "do_bench", do_bench)
    out = bench.time_impls(fns, rounds=6, rep_ms=1, cold_rounds=5)
    first = out["first"]
    assert first["cold_runs_us"] == pytest.approx([6.0, 5.0, 6.0, 5.0, 6.0])
    assert first["cold_order_medians_us"] == pytest.approx([6.0, 5.0])
    assert first["cold_us"] == pytest.approx(5.5)
    assert first["bb_order_mins_us"] == pytest.approx([10.0, 12.0])
    assert first["bb_us"] == pytest.approx(11.0)
    assert out["second"]["cold_us"] == pytest.approx(7.5)
    assert out["second"]["bb_us"] == pytest.approx(20.5)


def test_bench_takes_two_or_more_rotating_input_sets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # One set would be cold_us again under another name.
    base = ["bench.py", "r", "--configs-dir", "c", "--output", "o"]
    for flags, message in (
        (["--rotate-inputs", "1"], "at least 2 input sets are needed"),
        (["--rotate-inputs", "-8"], "at least 2 input sets are needed"),
        (["--rotate-inputs", "8", "--rotate-max-l2", "0"], "a positive factor"),
    ):
        monkeypatch.setattr(sys, "argv", [*base, *flags])
        with pytest.raises(SystemExit):
            bench.main()
        assert message in capsys.readouterr().err


def test_bench_rotates_inputs_only_on_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Without the option the record and the summary are what they were;
    # with it the summary shows rot_us, or why a port has none.
    timed = {"status": "ok", "bb_us": 12.0, "cold_us": 9.5}
    seen: list[tuple[int, float]] = []

    def run(args: object) -> dict[str, object]:
        count = args.rotate_inputs  # pyrefly: ignore[missing-attribute]
        seen.append((count, args.rotate_max_l2))  # pyrefly: ignore[missing-attribute]
        record: dict[str, object] = {
            "status": "ok",
            "shape": "s1",
            "impls": {"inductor": dict(timed), "oracle": {"status": "error"}},
        }
        if count == 8:
            record["impls"]["inductor"]["rot_us"] = 10.75  # pyrefly: ignore[bad-index]
            record["rotate_inputs"] = {"status": "ok"}
        elif count:
            record["rotate_inputs"] = {"status": "skipped", "skipped": "too large"}
        return record

    monkeypatch.setattr(bench, "run", run)
    monkeypatch.setenv("BB_FOREIGN_ACTIVITY", "0")
    out = tmp_path / "record.json"
    argv = ["bench.py", "r", "--configs-dir", "c", "--output", str(out)]
    summary = "ok r[s1] {'inductor': '12.00/9.50us', 'oracle': 'error'}\n"
    monkeypatch.setattr(sys, "argv", argv)
    bench.main()
    assert capsys.readouterr().out == summary
    assert json.loads(out.read_text()) == {
        "status": "ok",
        "shape": "s1",
        "impls": {"inductor": timed, "oracle": {"status": "error"}},
        "foreign_gpu_activity": None,
    }
    monkeypatch.setattr(sys, "argv", [*argv, "--rotate-inputs", "8"])
    bench.main()
    assert capsys.readouterr().out == summary.replace("9.50us", "9.50/10.75us")
    monkeypatch.setattr(
        sys, "argv", [*argv, "--rotate-inputs", "4", "--rotate-max-l2", "1"]
    )
    bench.main()
    assert capsys.readouterr().out == summary + "rot_us skipped: too large\n"
    assert seen == [(0, bench.ROTATE_MAX_L2_MULTIPLE), (8, 2.0), (4, 1.0)]


# A B200: ``torch.cuda.get_device_properties(0).L2_cache_size`` and its
# free memory with nothing else on it.
_B200_L2 = 132_644_864
_B200_FREE = 189_000_000_000


def test_rotation_takes_the_small_ports_that_cold_us_times_in_batches() -> None:
    def reason(input_bytes: int, **given: object) -> str:
        # 8 sets for each of 6 implementations, the fastest one at 10 us.
        fields: dict[str, object] = {
            "fastest_impl": "memcopy",
            "fastest_us": 10.0,
            "l2_bytes": _B200_L2,
            "free_bytes": _B200_FREE,
        }
        fields.update(given)
        found = bench.rotation_skip_reason(
            input_bytes,
            48 * input_bytes,
            **fields,  # pyrefly: ignore[bad-argument-type]
        )
        return found or "applies"

    # The selection's points around the cut-off, twice the L2: 7291 (7 MB),
    # 0dd15 (105 MB), c9e1 (127 MB), e232 (201 MB) and 979a (268 MB).
    assert [reason(mb * 1_000_000) == "applies" for mb in (7, 105, 127, 201, 268)] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert reason(2 * _B200_L2) == "applies"
    assert reason(2 * _B200_L2 + 1) == (
        "inputs of 265.3 MB exceed 2 x the 132.6 MB L2 (265.3 MB), the limit for "
        "private input sets (--rotate-max-l2)"
    )
    assert reason(201_000_000, max_l2_multiple=1.0).startswith(
        "inputs of 201.0 MB exceed 1 x the 132.6 MB L2 (132.6 MB)"
    )
    # cold_us times single replays from 64 us for the fastest implementation
    # (e232's memcopy takes 65.5 us): batches would not be its cross-check.
    assert reason(128_000_000, fastest_us=41.0) == "applies"
    assert reason(201_000_000, fastest_us=63.9) == "applies"
    assert reason(201_000_000, fastest_us=65.5) == (
        "the fastest implementation, memcopy, takes 65.5 us a call: from 64 us "
        "cold_us times single replays, not the batches of flushed calls that "
        "rot_us would time"
    )
    # Without the memcopy the fastest port decides (bbcd's takes 86 us).
    assert reason(128_000_000, fastest_impl="oracle", fastest_us=86.4).startswith(
        "the fastest implementation, oracle, takes 86.4 us a call"
    )
    assert reason(7_000_000, fastest_us=bench.ROTATE_BELOW_US) != "applies"
    # The sets must also fit a quarter of what the device has free.
    assert reason(105_000_000, free_bytes=4 * 48 * 105_000_000) == "applies"
    assert reason(105_000_000, free_bytes=20_000_000_000) == (
        "the private input sets need 5.04 GB, over 25% of the 20.00 GB of free "
        "device memory"
    )
    # A device that reports no L2 takes no port: the first rule leaves it out.
    assert reason(7_000_000, l2_bytes=0).startswith(
        "inputs of 7.0 MB exceed 2 x the 0.0 MB L2 (0.0 MB)"
    )


def test_input_bytes_count_each_storage_once() -> None:
    table = torch.zeros(100, 8, dtype=torch.float32)
    ids = torch.zeros(5, dtype=torch.int64)
    # Two views of one buffer, an expanded scalar, a shape and a scalar.
    inputs = [table[:10], table[50:, ::2], ids, torch.zeros(1).expand(1000), [3, 4], 2]
    assert common.input_bytes(inputs) == 3200 + 40 + 4
    assert common.input_bytes([]) == 0


class _FakeClock:
    """CUDA events without a device: ``Event.record`` reads a clock that the
    test's replays advance, in milliseconds."""

    def __init__(self) -> None:
        self.now = 0.0
        clock = self

        class Event:
            def __init__(self, enable_timing: bool = False) -> None:
                assert enable_timing
                self.at: float | None = None

            def record(self) -> None:
                self.at = clock.now

            def elapsed_time(self, end: Event) -> float:
                assert self.at is not None and end.at is not None
                return end.at - self.at

        self.Event = Event


def test_rotating_bench_subtracts_the_flush_graph_sample_by_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(torch.cuda, "Event", clock.Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    # Per sample: what the 4 flushes take, and each function's 4 calls on
    # top of them. The first flush sample is an outlier, and so is the last
    # sample of the first function.
    flushes = [0.400, 0.010, 0.010]
    kernels = {"first": [0.032, 0.032, 0.320], "second": [0.060, 0.064, 0.068]}
    sample = {"first": 0, "second": 0, "flush": 0}
    captured: list[tuple[list[str], str, int]] = []

    def replay_of(name: str) -> Callable[[], None]:
        def replay() -> None:
            k = sample[name]
            sample[name] += 1
            clock.now += flushes[k] + (0.0 if name == "flush" else kernels[name][k])

        return replay

    def batched(
        fns: list[Callable[[], str]], clear_cache: Callable[[], str], calls: int
    ) -> tuple[list[Callable[[], None]], Callable[[], None]]:
        captured.append(([fn() for fn in fns], clear_cache(), calls))
        return [replay_of(fn()) for fn in fns], replay_of("flush")

    monkeypatch.setattr(bench, "_batched_flushed_replays", batched)
    fns = [lambda: "second", lambda: "first"]
    timings = bench.rotating_bench(fns, lambda: "flush", calls=4, repeat=3)
    assert captured == [(["second", "first"], "flush", 4)]
    # In the order given, per call: the median of each sample's own
    # difference (unpaired medians would read 0.080 - 0.010 for the first).
    assert timings == pytest.approx([0.016, 0.008])
    # A capture that fails fails the round: nothing else would be the timer.
    monkeypatch.setattr(bench, "_batched_flushed_replays", lambda *args: None)
    with pytest.raises(RuntimeError, match="graphs of 4 flushed calls"):
        bench.rotating_bench(fns, lambda: "flush", calls=4, repeat=3)


def test_time_rotating_reads_each_private_set_once_a_batch_in_both_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(bench, "_make_l2_cache_clearer", lambda: "the flush")
    read: dict[str, list[str]] = {"first": [], "second": []}
    calls = {
        name: (lambda inputs, name=name: read[name].append(inputs[0])) for name in read
    }
    sets = {name: [[f"{name}{i}"] for i in range(4)] for name in read}
    cold = {"first": 0.005, "second": 0.007}
    batches: list[dict[str, list[str]]] = []
    # A clock that reads 0.2 s more each time: two untimed rounds pass
    # before the GPU has run these batches for half a second.
    clock = itertools.count()
    monkeypatch.setattr(bench.time, "perf_counter", lambda: 0.2 * next(clock))

    def rotating_bench(
        fns: list[Callable[[], object]],
        clear_cache: object,
        *,
        calls: int,
        repeat: int,
    ) -> list[float]:
        # Every set was read once, eagerly, before the first round.
        if not batches:
            assert read == {name: [s[0] for s in sets[name]] for name in read}
        assert clear_cache == "the flush" and (calls, repeat) == (4, 25)
        order = []
        batch = {}
        for fn in fns:
            before = {name: len(seen) for name, seen in read.items()}
            # As the capture runs it: once on a side stream, then the batch.
            fn()
            (name,) = [n for n, seen in read.items() if len(seen) > before[n]]
            for _ in range(calls):
                fn()
            batch[name] = read[name][-calls:]
            order.append(name)
        batches.append(batch)
        # The untimed rounds read 9 us more; in the timed ones each
        # implementation reads 1 us more in the forward order.
        forward = order[0] == "first"
        slower = 0.009 if len(batches) <= 2 else 0.001 if forward else 0.0
        return [cold[name] + slower for name in order]

    monkeypatch.setattr(bench, "rotating_bench", rotating_bench)
    out = bench.time_rotating(calls, sets, rounds=4)
    # Every batch reads each of the implementation's own sets exactly once.
    assert len(batches) == 2 + 4
    for batch in batches:
        for name, seen in batch.items():
            assert sorted(seen) == [s[0] for s in sets[name]]
    first = out["first"]
    assert first["rot_runs_us"] == pytest.approx([6.0, 5.0, 6.0, 5.0])
    assert first["rot_order_medians_us"] == pytest.approx([6.0, 5.0])
    assert first["rot_us"] == pytest.approx(5.5)
    assert first["rot_sets"] == 4
    assert out["second"]["rot_us"] == pytest.approx(7.5)
    # A round's samples for batches of 2 to 16 calls: cold_us's 100 over the
    # batch size, and at least 20.
    assert [bench.rotating_samples(calls) for calls in (2, 3, 4, 5, 8, 16)] == [
        50,
        34,
        25,
        20,
        20,
        20,
    ]


@pytest.mark.cuda
@skipIfNotCUDA()
def test_each_captured_call_of_a_rotating_batch_reads_a_set_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the device, what the fakes above assume of the capture: in the
    # graphs a round replays, every call of an implementation reads another
    # of its private sets, and none reads another implementation's.
    monkeypatch.setattr(bench, "SETTLE_S", 0.0)
    names = ("first", "second")
    sets = {
        name: [[torch.zeros(1024, device=DEVICE)] for _ in range(3)] for name in names
    }
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    calls = {
        name: (lambda inputs, name=name: outputs[name].append(inputs[0] + 1))
        for name in names
    }
    rounds = []
    capture = bench._batched_flushed_replays

    def recording(
        fns: list[Callable[[], object]], clear_cache: Callable[[], None], count: int
    ) -> object:
        rounds.append(capture(fns, clear_cache, count))
        return rounds[-1]

    monkeypatch.setattr(bench, "_batched_flushed_replays", recording)
    with common.gc_paused():
        out = bench.time_rotating(calls, sets, rounds=2)
    assert [out[name]["rot_sets"] for name in names] == [3, 3]
    assert all(out[name]["rot_us"] > 0 for name in names)
    # The last round's graphs, replayed on sets that each hold a value of
    # their own: an implementation's three captured calls (its last three)
    # give its three sets' values plus one.
    assert len(rounds) == 2 and rounds[-1] is not None
    replays, flush_replay = rounds[-1]
    value = {name: [100.0 * i + k for k in range(3)] for i, name in enumerate(names)}
    for name in names:
        for inputs, filled in zip(sets[name], value[name], strict=True):
            inputs[0].fill_(filled)
    for replay in (*replays, flush_replay):
        replay()
    torch.cuda.synchronize()
    for name in names:
        assert len(outputs[name]) == (3 + 2 * (1 + 3))
        read = sorted(float(result[0]) for result in outputs[name][-3:])
        assert read == [filled + 1 for filled in value[name]]


@pytest.mark.cuda
@skipIfNotCUDA()
def test_a_rotating_round_takes_the_samples_of_a_cold_round_on_such_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # rot_us is to average as many executions as cold_us. interleaved_bench
    # decides its sample count inside, by the batch it sizes, so bench.py
    # restates that rule (rotating_samples): here against the samples a cold
    # round takes on the device, counted as replays of its flush-only graph.
    source = torch.zeros(1024, device=DEVICE)
    out = torch.empty_like(source)
    capture = benchmarking._make_batched_cudagraph_replay
    batches: list[int] = []
    samples = 0

    def counted(
        fn: Callable[[], object] | None,
        clear_cache: Callable[[], None],
        calls: int,
        keep_outputs: bool = False,
    ) -> Callable[[], None]:
        replay = capture(fn, clear_cache, calls, keep_outputs=keep_outputs)
        if fn is not None:
            batches.append(calls)
            return replay

        def flush_replay() -> None:
            nonlocal samples
            samples += 1
            replay()

        return flush_replay

    monkeypatch.setattr(benchmarking, "_make_batched_cudagraph_replay", counted)
    for calls in (2, 4, 8, 16):
        # A single replay estimated at 64 / calls us: batches of that many.
        monkeypatch.setattr(
            benchmarking,
            "_single_replay_estimate_us",
            lambda *args, calls=calls, **kwargs: 64.0 / calls,
        )
        batches.clear()
        samples = 0
        with common.gc_paused():
            (ms,) = bench.cold_round([lambda: torch.add(source, 1.0, out=out)])
        assert ms > 0 and batches == [calls]
        assert samples == bench.rotating_samples(calls)


def test_a_ports_bench_entry_names_its_autotune_record_before_the_port_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = tmp_path / "configs"
    _toy_tuned(tmp_path, "sum_000000000001", "s1", "cute")
    path = configs / "sum_000000000001__s1__cute.json"
    tuned = json.loads(path.read_text())
    loads: list[object] = []

    def load_port(repro: str, backend: str, pinned: dict[str, str]) -> object:
        loads.append((repro, backend, pinned))
        if len(loads) == 1:
            raise helion.exc.InvalidConfig(_TOY_INVALID)
        return "port"

    monkeypatch.setattr(common, "load_port", load_port)
    # A pick that this tree cannot load: the entry still says what the port
    # was pinned to, so the campaign takes the bench for one on that record.
    named = {
        "autotune_record": str(path),
        "decorators": {"k": "@helion.kernel(version=1)"},
        "autotune_code_sha256": {"k": "cute-1"},
        "autotune_provisional": False,
        "autotune_reused_from": None,
    }
    info: dict[str, object] = {"status": "error"}
    with pytest.raises(helion.exc.InvalidConfig):
        bench.pinned_port(info, configs, "sum_000000000001", "s1", "cute")
    assert info == {"status": "error", **named}
    assert loads == [("sum_000000000001", "cute", named["decorators"])]
    assert common.timed_on_record(info, tuned)
    info = {}
    assert bench.pinned_port(info, configs, "sum_000000000001", "s1", "cute") == (
        "port",
        tuned,
    )
    assert info == named
    # Without an ok record the entry names none: its path only.
    for backend, error in (("triton", FileNotFoundError), ("cute", RuntimeError)):
        path.write_text(json.dumps({"status": "error"}))
        info = {}
        with pytest.raises(error):
            bench.pinned_port(info, configs, "sum_000000000001", "s1", backend)
        assert list(info) == ["autotune_record"] and len(loads) == 2
        assert common.timed_on_record(info, None)
        assert not common.timed_on_record(info, tuned)


def test_a_bench_entry_is_of_the_record_whose_kernels_it_names() -> None:
    tuned = {"status": "ok", "kernels": _toy_kernels("cute")}
    again = {"status": "ok", "kernels": _toy_kernels("cute", 2)}
    info = common.timed_record_fields(tuned)
    assert common.timed_on_record(info, tuned)
    assert common.timed_on_record(info, {**tuned, "autotune_s": 9.0})
    assert not common.timed_on_record(info, again)
    assert not common.timed_on_record(info, None)
    # A provisional record that an autotune replaces holds a result the
    # bench has not timed, whatever kernels the autotune picks.
    provisional = {**tuned, "provisional": True, "reused_from": "run5"}
    stood_in = common.timed_record_fields(provisional)
    assert common.timed_on_record(stood_in, provisional)
    assert not common.timed_on_record(stood_in, tuned)
    assert not common.timed_on_record(info, provisional)
    # An entry of a port that found no ok record names none.
    assert common.timed_on_record({"status": "error"}, None)
    assert not common.timed_on_record({"status": "error"}, tuned)
    # Records made before the code digests name the decorators, and do not
    # say whether their record was provisional: the kernels tell.
    before = {"decorators": {"k": "@helion.kernel(version=1)"}}
    assert common.timed_on_record(before, tuned)
    assert common.timed_on_record(before, provisional)
    assert not common.timed_on_record(before, again)
    assert not common.timed_on_record(before, None)


def _rotation_args(count: int = 8) -> object:
    return types.SimpleNamespace(
        rotate_inputs=count,
        rotate_max_l2=bench.ROTATE_MAX_L2_MULTIPLE,
        cold_rounds=6,
    )


def test_the_rotating_pass_records_rot_us_beside_the_other_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = types.SimpleNamespace(L2_cache_size=_B200_L2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (_B200_FREE, 2 * _B200_FREE)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    made: dict[str, int] = {"port": 0, "memcopy": 0}

    def fresh(name: str) -> Callable[[], list[object]]:
        def make() -> list[object]:
            made[name] += 1
            return [f"{name}{made[name]}"]

        return make

    rotating = {
        "helion_cute": bench.Rotating(lambda inputs: None, fresh("port")),
        "inductor": bench.Rotating(lambda inputs: None, fresh("port")),
        "memcopy": bench.Rotating(lambda inputs: None, fresh("memcopy"), 3_000_000),
    }
    timed: list[tuple[dict[str, list[list[object]]], int]] = []

    def time_rotating(
        calls: dict[str, object], sets: dict[str, list[list[object]]], *, rounds: int
    ) -> dict[str, dict[str, object]]:
        assert list(calls) == list(rotating)
        timed.append((sets, rounds))
        return {name: {"rot_us": 4.0, "rot_sets": len(sets[name])} for name in calls}

    monkeypatch.setattr(bench, "time_rotating", time_rotating)

    def record() -> dict[str, object]:
        return {
            "total_bytes": 8_000_000,
            "impls": {name: {"cold_us": 5.0} for name in rotating},
        }

    # A port of 7 MB: 8 private sets per implementation, timed, recorded.
    found = record()
    bench.rotate_inputs(_rotation_args(), found, rotating, 7_000_000)
    assert found["rotate_inputs"] == {
        "sets": 8,
        "rounds": 6,
        "input_bytes": 7_000_000,
        "fastest_impl": "helion_cute",
        "fastest_cold_us": 5.0,
        "sets_bytes": 8 * (7_000_000 + 7_000_000 + 3_000_000),
        "l2_cache_bytes": _B200_L2,
        "max_l2_multiple": 2.0,
        "free_bytes": _B200_FREE,
        "status": "ok",
    }
    [(sets, rounds)] = timed
    assert rounds == 6 and made == {"port": 16, "memcopy": 8}
    private = [s[0] for group in sets.values() for s in group]
    assert len(private) == len(set(private)) == 24
    assert found["impls"] == {
        name: {"cold_us": 5.0, "rot_us": 4.0, "rot_sets": 8, "rot_gbps": 2000.0}
        for name in rotating
    }
    # A port of 4.3 GB: no set is made, the record says why, and its
    # implementations keep the fields they had.
    found = record()
    bench.rotate_inputs(_rotation_args(), found, rotating, 4_295_000_000)
    decision = cast("dict[str, str]", found["rotate_inputs"])
    assert decision["status"] == "skipped"
    assert decision["skipped"] == (
        "inputs of 4295.0 MB exceed 2 x the 132.6 MB L2 (265.3 MB), the limit for "
        "private input sets (--rotate-max-l2)"
    )
    assert made == {"port": 16, "memcopy": 8} and len(timed) == 1
    assert found["impls"] == record()["impls"]
    # Nor does a port that cold_us timed call by call: its fastest
    # implementation takes 64 us or more.
    found = record()
    slow = cast("dict[str, dict[str, float]]", found["impls"])
    for name, us in zip(rotating, (90.0, 150.0, 65.5), strict=True):
        slow[name]["cold_us"] = us
    bench.rotate_inputs(_rotation_args(), found, rotating, 7_000_000)
    decision = cast("dict[str, object]", found["rotate_inputs"])
    assert (decision["status"], decision["fastest_cold_us"]) == ("skipped", 65.5)
    assert decision["fastest_impl"] == "memcopy"
    assert str(decision["skipped"]).startswith(
        "the fastest implementation, memcopy, takes 65.5 us a call"
    )
    assert len(timed) == 1 and "rot_us" not in json.dumps(found["impls"])
    # Nor is there anything to rotate when no implementation was timed.
    found = record()
    bench.rotate_inputs(_rotation_args(), found, {}, 7_000_000)
    decision = cast("dict[str, str]", found["rotate_inputs"])
    assert (decision["status"], decision["skipped"], decision["fastest_impl"]) == (
        "skipped",
        "no implementation was timed",
        None,
    )

    # A pass that fails is recorded; the other timings stand.
    def failing(*args: object, **kwargs: object) -> None:
        raise RuntimeError("could not capture graphs of 8 flushed calls")

    monkeypatch.setattr(bench, "time_rotating", failing)
    found = record()
    bench.rotate_inputs(_rotation_args(), found, rotating, 7_000_000)
    decision = cast("dict[str, str]", found["rotate_inputs"])
    assert decision["status"] == "error"
    assert "could not capture graphs of 8 flushed calls" in decision["error"]
    assert found["impls"] == record()["impls"]

    # So is a device query that fails before any set is made: the record is
    # not lost to it either.
    def lost(*args: object) -> None:
        raise RuntimeError("CUDA error: unspecified launch failure")

    for query in ("get_device_properties", "mem_get_info"):
        monkeypatch.setattr(torch.cuda, query, lost)
        found = record()
        bench.rotate_inputs(_rotation_args(), found, rotating, 7_000_000)
        decision = cast("dict[str, str]", found["rotate_inputs"])
        assert (decision["status"], decision["sets"]) == ("error", 8)
        assert "unspecified launch failure" in decision["error"]
        assert found["impls"] == record()["impls"]


def test_report_takes_only_finite_positive_timings() -> None:
    record = {
        "impls": {
            impl: {"status": "ok", "numerics_pass": True, "cold_us": value}
            for impl, value in (
                ("inductor", math.inf),
                ("helion_cute", 2.0),
                ("helion_triton", 0.0),
            )
        }
    }
    assert report.impl_time(record, "inductor", "cold_us") is None
    assert report.impl_time(record, "helion_cute", "cold_us") == 2.0
    assert report.impl_time(record, "helion_triton", "cold_us") is None


def test_report_lists_records_where_foreign_processes_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idle = {"bob:7": {"max_sm": 0, "active_s": 0.0}}
    busy = {
        **idle,
        "eve:8": {"max_sm": 70, "active_s": 12.0},
        "eve:9": {"max_sm": 3, "active_s": 1.0},
    }

    def record(name: str, processes: dict[str, object], **fields: object) -> None:
        repro, shape, *backend = name.split("__")
        activity = {"interval_s": 1.0, "samples": 9, "unknown_samples": 0}
        path = tmp_path / ("configs" if backend else "bench") / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        rec = {"repro": repro, "shape": shape, "status": "ok", "impls": {}}
        if backend:
            rec.update(backend=backend[0], autotune_s=1.0)
        rec["foreign_gpu_activity"] = {**activity, "processes": processes, **fields}
        path.write_text(json.dumps(rec))

    record("sum_000000000001__s1", idle)
    record("sum_000000000002__s2", busy)
    record("sum_000000000003__s3", busy, unknown_samples=9)
    record("sum_000000000002__s2__cute", busy)
    monkeypatch.setattr(sys, "argv", ["report.py", str(tmp_path)])
    report.main()
    found = json.loads((tmp_path / "report.json").read_text())["contaminated"]
    assert found == {
        "bench": {"sum_000000000002__s2": ["eve:8 (70% SM, 12.0 s)"]},
        "autotune": {"sum_000000000002__s2__cute": ["eve:8 (70% SM, 12.0 s)"]},
    }
    text = (tmp_path / "report.md").read_text()
    assert (
        "Records without activity samples (left out of the aggregates):\n\n- none\n"
    ) in text
    assert (
        "Records whose activity samples do not cover their jobs (left out of "
        "the aggregates; run them again):\n\n"
        "- bench sum_000000000003__s3: none of 9 samples read the GPU\n"
    ) in text
    assert "- bench sum_000000000002__s2: eve:8 (70% SM, 12.0 s)" in text


def test_ncu_units_convert_or_raise() -> None:
    entry = {
        "dram__bytes_read.sum": 1.5,
        "dram__bytes_read.sum__unit": "Gbyte",
        "gpu__time_duration.sum": 2.0,
        "gpu__time_duration.sum__unit": "msecond",
        "lts__t_bytes.sum": 1.0,
        "lts__t_bytes.sum__unit": "Pbyte",
    }
    assert ncu_profile._scaled(entry, "dram__bytes_read.sum", "Mbyte") == 1500.0
    assert ncu_profile._scaled(entry, "gpu__time_duration.sum", "us") == pytest.approx(
        2000.0
    )
    with pytest.raises(ValueError, match="Pbyte"):
        ncu_profile._scaled(entry, "lts__t_bytes.sum", "Mbyte")
    with pytest.raises(ValueError, match="not a unit of"):
        ncu_profile._scaled(entry, "gpu__time_duration.sum", "Mbyte")


def test_board_claims_are_renewed_every_third_of_the_ttl() -> None:
    assert campaign.parse_duration("90s") == 90
    assert campaign.parse_duration("45") == 45 * 60
    assert campaign.parse_duration("2h") == 7200
    with pytest.raises(ValueError, match="not a duration"):
        campaign.parse_duration("2 hours")
    # The board takes no longer claim, so the campaign refuses one before its
    # first pick.
    assert campaign.parse_duration("12h") == 12 * 3600
    with pytest.raises(ValueError, match="12h maximum"):
        campaign.parse_duration("13h")


# A stand-in for autotune.py and bench.py: writes an ok record whose foreign
# process peaks at the SM share planned for this attempt of this output
# (``BB_STUB_PLAN``: {record name: [max SM per attempt]}; a negative share
# leaves every sample unknown, and None fails the attempt without a record).
# With ``BB_STUB_LEAVE`` it leaves a worker behind in a session of its own and
# writes its pid there. Like autotune.py it records its HELION_* environment
# and the run its seed was taken from, and like the autotuner it appends to
# the CSV file of ``HELION_AUTOTUNE_LOG`` (here: its attempt).
_STUB_JOB = """
import json, os, subprocess, sys
from pathlib import Path

args = sys.argv[1:]
out = Path(args[args.index("--output") + 1])
plan = json.loads(os.environ["BB_STUB_PLAN"])
count = Path(os.environ["BB_STUB_DIR"], out.name + ".count")
attempt = int(count.read_text()) + 1 if count.exists() else 1
count.write_text(str(attempt))
sm = plan.get(out.name, [0])[min(attempt, len(plan.get(out.name, [0]))) - 1]
if sm is None:
    # The attempt fails (a timeout or an out-of-memory error, say).
    sys.exit(1)
record = {
    "repro": args[0],
    "shape": args[args.index("--shape") + 1],
    "status": "ok",
    "attempt": attempt,
    "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
    "autotune_s": 1.0,
    "impls": {},
    "env": {k: v for k, v in os.environ.items() if k.startswith("HELION_")},
    "foreign_gpu_activity": {
        "interval_s": 1.0,
        "samples": 9,
        "unknown_samples": 0,
        "processes": {"eve:7": {"max_sm": sm, "active_s": float(attempt)}},
    },
}
if "BB_SEED_FROM" in os.environ:
    record["seed_from"] = os.environ["BB_SEED_FROM"]
if "HELION_AUTOTUNE_LOG" in os.environ:
    search = Path(os.environ["HELION_AUTOTUNE_LOG"]).with_suffix(".csv")
    search.parent.mkdir(parents=True, exist_ok=True)
    with search.open("a") as rows:
        rows.write(f"attempt {attempt}\\n")
if sm < 0:
    # The sampling failed throughout: no sample read the GPU.
    record["foreign_gpu_activity"].update(unknown_samples=9, processes={})
if "--backend" in args:
    record["backend"] = args[args.index("--backend") + 1]
if "--configs-dir" in args:
    # A bench: an entry for every implementation it is given, a port's
    # naming the autotune record it is pinned to, as bench.py's do.
    configs = Path(args[args.index("--configs-dir") + 1])
    impls = os.environ["BB_STUB_IMPLS"]
    if "--impls" in args:
        impls = args[args.index("--impls") + 1]
    for impl in impls.split(","):
        entry = {"status": "ok"}
        if impl.startswith("helion_"):
            name = f"{args[0]}__{record['shape']}__{impl.removeprefix('helion_')}.json"
            tuned = {}
            if (configs / name).exists():
                tuned = json.loads((configs / name).read_text())
            if tuned.get("status") == "ok":
                entry["autotune_code_sha256"] = {
                    kernel: fields["code_sha256"]
                    for kernel, fields in tuned.get("kernels", {}).items()
                }
                entry["autotune_provisional"] = bool(tuned.get("provisional"))
            else:
                entry["status"] = "error"
        record["impls"][impl] = entry
if "--rotate-inputs" in args:
    sets = int(args[args.index("--rotate-inputs") + 1])
    record["rotate_inputs"] = {"sets": sets, "status": "ok"}
out.write_text(json.dumps(record))
if "BB_STUB_LEAVE" in os.environ:
    if os.fork() == 0:
        os.setsid()
        worker = subprocess.Popen(["sleep", "60"])
        leave = Path(os.environ["BB_STUB_LEAVE"], out.name + ".worker")
        leave.write_text(str(worker.pid))
        os._exit(0)
    os.wait()
"""


def _stub_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, list[int]],
    *flags: str,
    kernels: list[dict[str, str]] | None = None,
    backends: str = "triton",
    gpu_uuid: Callable[[str], str | None] | None = None,
    busy_uuids: Callable[[], set[str] | None] | None = None,
    gpu_source: tuple[str, ...] = ("--gpus", "0,1"),
) -> tuple[Path, list[dict[str, object]]]:
    """Run ``campaign.py`` with stub jobs on GPUs 0 and 1 (or ``gpu_source``)
    into ``tmp_path / "run"`` (nvidia-smi stubbed: each GPU's UUID
    ``GPU-<i>`` and none busy, unless ``gpu_uuid`` and ``busy_uuids`` say
    otherwise); the run directory and this run's events."""
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    for script in ("autotune.py", "bench.py"):
        (jobs / script).write_text(_STUB_JOB)
    if kernels is None:
        kernels = [{"repro": "sum_000000000001", "shape": "s1"}]
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"kernels": kernels}))
    monkeypatch.setenv("BB_STUB_PLAN", json.dumps(plan))
    monkeypatch.setenv("BB_STUB_DIR", str(tmp_path))
    monkeypatch.setenv("BB_STUB_IMPLS", ",".join(common.ALL_IMPLS))
    monkeypatch.setattr(campaign, "PACKAGE_DIR", jobs)
    monkeypatch.setattr(campaign, "POLL_S", 0.01)
    monkeypatch.setattr(campaign, "gpu_uuid", gpu_uuid or (lambda gpu: f"GPU-{gpu}"))
    monkeypatch.setattr(campaign, "busy_uuids", busy_uuids or set)
    run = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign.py",
            "--selection",
            str(selection),
            "--run-dir",
            str(run),
            *gpu_source,
            "--backends",
            backends,
            *flags,
        ],
    )
    log = run / "campaign.jsonl"
    before = len(log.read_text().splitlines()) if log.exists() else 0
    # The campaign ignores the stop signals once it ends; give them back.
    handlers = {signum: signal.getsignal(signum) for signum in campaign.STOP_SIGNALS}
    try:
        campaign.main()
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    events = [json.loads(line) for line in log.read_text().splitlines()[before:]]
    return run, events


@pytest.mark.slow
@_requires_proc
def test_a_contaminated_job_runs_again_on_another_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The autotune's first attempt has a foreign process at 83% SM, its
    # second none: the second record stands, listing the first, and the
    # bench waits for it.
    name = "sum_000000000001__s1__triton.json"
    run, events = _stub_campaign(tmp_path, monkeypatch, {name: [83, 0]})
    record = json.loads((run / "configs" / name).read_text())
    first = run / "configs" / f"{name}.attempt1"
    assert record["attempt"] == 2 and "contaminated" not in record
    assert record["contaminated_attempts"] == [str(first)]
    assert json.loads(first.read_text())["gpu"] != record["gpu"]
    # Each attempt keeps its log beside its record.
    log = run / "logs" / "autotune" / f"{name[:-5]}.log"
    first_log = log.with_name(f"{log.name}.attempt1")
    assert f"gpu={record['gpu']} " not in first_log.read_text()
    assert f"gpu={record['gpu']} " in log.read_text()
    retries = [e for e in events if e["kind"] == "job_contaminated"]
    assert [e["action"] for e in retries] == ["retry"]
    starts = [e["job"] for e in events if e["kind"] == "job_start"]
    assert starts[-1].startswith("bench:") and len(starts) == 3


@pytest.mark.slow
@_requires_proc
def test_a_campaign_waits_out_nvidia_smi_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # nvidia-smi fails for a while (a hung query times out instead of
    # freezing the scheduler): GPU 1's UUID comes on its third ask and the
    # GPUs' business on the fourth.  Nothing starts before, the campaign
    # does not crash, and both jobs run.
    asks = {"0": 0, "1": 0}

    def uuid(gpu: str) -> str | None:
        asks[gpu] += 1
        return None if gpu == "1" and asks[gpu] < 3 else f"GPU-{gpu}"

    answers: list[set[str] | None] = [None, None, None]

    def busy() -> set[str] | None:
        return answers.pop() if answers else set()

    run, events = _stub_campaign(
        tmp_path, monkeypatch, {}, gpu_uuid=uuid, busy_uuids=busy
    )
    starts = [e for e in events if e["kind"] == "job_start"]
    assert [e["job"].split(":")[0] for e in starts] == ["autotune", "bench"]
    assert not answers and asks["1"] >= 3
    assert (run / "configs" / "sum_000000000001__s1__triton.json").exists()


# A stand-in for scripts/gpu_board.py: logs its arguments to
# $BB_STUB_BOARD_LOG, picks GPU 0, and raises a SECURITY ALERT (exit status
# 4) for the command in $BB_STUB_BOARD_ALERT.
_STUB_BOARD = """
import json, os, sys
with open(os.environ["BB_STUB_BOARD_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == os.environ.get("BB_STUB_BOARD_ALERT"):
    print("gpu_board: SECURITY ALERT: a planted file", file=sys.stderr)
    sys.exit(4)
if sys.argv[1] == "pick":
    print("0")
"""


@pytest.mark.slow
@_requires_proc
def test_a_board_campaign_claims_its_gpus_through_the_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The board commands as the campaign runs them; a SECURITY ALERT stops
    # the campaign without touching the board again.
    board = tmp_path / "gpu_board.py"
    board.write_text(_STUB_BOARD)
    monkeypatch.setattr(campaign, "BOARD", board)
    log = tmp_path / "board.log"
    monkeypatch.setenv("BB_STUB_BOARD_LOG", str(log))
    run, events = _stub_campaign(
        tmp_path, monkeypatch, {}, gpu_source=("--board-gpus", "1")
    )
    assert [e["gpu"] for e in events if e["kind"] == "job_start"] == ["0", "0"]
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    purpose = ["--purpose", "better-benchmark campaign", "--ttl", "2h"]
    assert calls == [["pick", "-n", "1", *purpose], ["release", "0"]]
    log.unlink()
    monkeypatch.setenv("BB_STUB_BOARD_ALERT", "pick")
    (tmp_path / "alert").mkdir()
    with pytest.raises(campaign.BoardSecurityAlert, match="a planted file"):
        _stub_campaign(
            tmp_path / "alert", monkeypatch, {}, gpu_source=("--board-gpus", "1")
        )
    assert [json.loads(line)[0] for line in log.read_text().splitlines()] == ["pick"]


@pytest.mark.slow
@_requires_proc
def test_a_board_campaign_claims_another_gpu_while_its_gpu_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Another account's work keeps the claimed GPU 0 busy while the board has
    # GPU 1 free: the campaign claims GPU 1 rather than wait for GPU 0.
    picks = ["0", "1"]
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(campaign, "board_pick", lambda *args: picks.pop(0))
    monkeypatch.setattr(campaign, "board_call", lambda *args: calls.append(args))
    # GPU 0 busy for the first 300 queries (a scheduler pass or so each).
    queries = itertools.count()
    run, events = _stub_campaign(
        tmp_path,
        monkeypatch,
        {},
        busy_uuids=lambda: {"GPU-0"} if next(queries) < 300 else set(),
        gpu_source=("--board-gpus", "2"),
    )
    starts = [e["gpu"] for e in events if e["kind"] == "job_start"]
    assert starts == ["1", "1"]
    assert [e["gpu"] for e in events if e["kind"] == "board_pick"] == ["0", "1"]
    assert sorted(calls) == [("release", "0"), ("release", "1")]


@pytest.mark.slow
@_requires_proc
def test_a_job_whose_samples_miss_it_runs_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No sample of the autotune's first attempt read the GPU, so its record
    # cannot tell whether another process computed: the job runs again, and
    # the covered attempt stands.
    name = "sum_000000000001__s1__triton.json"
    run, events = _stub_campaign(tmp_path, monkeypatch, {name: [-1, 0]})
    record = json.loads((run / "configs" / name).read_text())
    assert record["attempt"] == 2 and "contaminated" not in record
    retries = [e for e in events if e["kind"] == "job_contaminated"]
    assert [e["action"] for e in retries] == ["retry"]
    assert retries[0]["foreign"] == [
        "samples miss the job: none of 9 samples read the GPU"
    ]
    # With every attempt contaminated or not covered, a covered one stands.
    (tmp_path / "again").mkdir()
    run, _ = _stub_campaign(
        tmp_path / "again",
        monkeypatch,
        {name: [-1, 40, -1]},
        "--retry-contaminated",
        "2",
    )
    record = json.loads((run / "configs" / name).read_text())
    assert record["attempt"] == 2 and record["contaminated"] is True


@pytest.mark.slow
@_requires_proc
def test_a_job_contaminated_every_time_keeps_its_cleanest_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "sum_000000000001__s1__triton.json"
    run, events = _stub_campaign(
        tmp_path, monkeypatch, {name: [83, 40, 90]}, "--retry-contaminated", "2"
    )
    record = json.loads((run / "configs" / name).read_text())
    # Attempt 2: 40% SM for 2 s ranks below 83% for 1 s and 90% for 3 s.
    assert record["attempt"] == 2 and record["contaminated"] is True
    assert len(record["contaminated_attempts"]) == 3
    actions = [e["action"] for e in events if e["kind"] == "job_contaminated"]
    assert actions == ["retry", "retry", "kept_cleanest"]
    assert common.foreign_activity(record) == ["eve:7 (40% SM, 2.0 s)"]


@pytest.mark.slow
@_requires_proc
def test_a_retry_that_fails_keeps_the_cleanest_attempt_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The retry of a contaminated autotune fails without a record (a timeout
    # or an out-of-memory error, say): the attempt before it stands, flagged,
    # and the bench pins it rather than run without the port's config.
    name = "sum_000000000001__s1__triton.json"
    run, events = _stub_campaign(tmp_path, monkeypatch, {name: [83, None]})
    configs = run / "configs"
    record = json.loads((configs / name).read_text())
    assert record["attempt"] == 1 and record["contaminated"] is True
    assert record["contaminated_attempts"] == [str(configs / f"{name}.attempt1")]
    settled = [e for e in events if e["kind"] == "job_contaminated"]
    assert [e["action"] for e in settled] == ["retry", "kept_cleanest"]
    assert settled[-1]["failed"] is None
    # The failed attempt's log stays beside the others.
    logs = run / "logs" / "autotune"
    assert (logs / f"{name[:-5]}.log.attempt2").exists()
    bench = [e for e in events if e["kind"] == "job_start" and "bench" in e["job"]]
    assert len(bench) == 1 and bench[0]["failed_autotunes"] == []


@pytest.mark.slow
@_requires_proc
def test_a_job_whose_leader_exits_leaves_no_worker_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each job's leader writes its record, leaves a worker running and
    # exits: the campaign kills the worker before end_job removes the job's
    # cache directory, and none outlives the campaign.
    monkeypatch.setenv("BB_STUB_LEAVE", str(tmp_path))
    try:
        _, events = _stub_campaign(tmp_path, monkeypatch, {})
        workers = {int(path.read_text()) for path in tmp_path.glob("*.worker")}
        assert len(workers) == 2
        assert not campaign._lives_on(workers, set())
        ends = [e for e in events if e["kind"] == "job_end"]
        assert len(ends) == 2 and not any("cache_kept" in e for e in ends)
    finally:
        for path in tmp_path.glob("*.worker"):
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(path.read_text()), signal.SIGKILL)


def _toy_record(run: Path, kind: str, name: str, **fields: object) -> None:
    path = run / kind / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "ok", **fields}))


def _toy_activity(max_sm: int) -> dict[str, object]:
    return {
        "interval_s": 1.0,
        "samples": 9,
        "unknown_samples": 0,
        "max_gap_s": 2.0,
        "processes": {"eve:7": {"max_sm": max_sm, "active_s": 3.0}},
    }


def _toy_priors(tmp_path: Path) -> tuple[dict[str, list[dict[str, str]]], Path, Path]:
    """Six points with earlier records in run5 (newest) and run3: point 1's
    Triton record is clean, 2's contaminated, 3's unsampled, 4's and 5's
    clean (their pins fail in ``_toy_pins``) and 6 has no ok one."""
    run5, run3 = tmp_path / "run5", tmp_path / "run3"
    kernels = [{"repro": f"sum_00000000000{i}", "shape": "s1"} for i in range(1, 7)]
    for i, point in enumerate(kernels[:5], 1):
        name = f"{point['repro']}__s1"
        _toy_record(run5, "configs", f"{name}__cute.json", wall_s=1000.0 * i)
        fields: dict[str, object] = {"wall_s": 100.0 * i, **point}
        if i != 3:
            fields["foreign_gpu_activity"] = _toy_activity(83 if i == 2 else 0)
        _toy_record(run5, "configs", f"{name}__triton.json", **fields)
    _toy_record(run3, "configs", "sum_000000000006__s1__cute.json", wall_s=600.0)
    (run3 / "configs" / "sum_000000000006__s1__triton.json").write_text(
        json.dumps({"status": "error", "wall_s": 9999.0})
    )
    # Older than run5's record of the same point, so never read.
    _toy_record(
        run3,
        "configs",
        "sum_000000000001__s1__triton.json",
        wall_s=7777.0,
        foreign_gpu_activity=_toy_activity(0),
        **kernels[0],
    )
    _toy_record(run5, "bench", "sum_000000000001__s1.json", wall_s=30.0)
    _toy_record(run3, "bench", "sum_000000000002__s1.json", wall_s=50.0)
    return {"kernels": kernels}, run5, run3


_TOY_INVALID = "Too many values for config['block_sizes'], expected 4, got 5"


def _toy_pins(calls: list[str]) -> Callable[[Path, str, str], dict[str, bool]]:
    def pinned_code_matches(
        configs_dir: Path, repro: str, shape: str
    ) -> dict[str, bool]:
        calls.append(f"{configs_dir.parent.name}/{repro}")
        if repro.endswith("4"):
            raise helion.exc.InvalidConfig(_TOY_INVALID)
        return {"k1": True, "k2": not repro.endswith("5")}

    return pinned_code_matches


def test_a_run_plan_reuses_clean_triton_records_whose_pinned_code_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, run5, run3 = _toy_priors(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(plan_run, "pinned_code_matches", _toy_pins(calls))
    result = plan_run.plan(selection, [run5, run3])
    assert result["reuse"] == [
        {
            "repro": "sum_000000000001",
            "shape": "s1",
            "backend": "triton",
            "record": str(run5 / "configs" / "sum_000000000001__s1__triton.json"),
        }
    ]
    assert calls == [
        "run5/sum_000000000001",
        "run5/sum_000000000004",
        "run5/sum_000000000005",
    ]
    triton = {
        job["repro"][-1]: (job["reason"], job["estimate_s"])
        for job in result["autotune"]
        if job["backend"] == "triton"
    }
    assert triton == {
        "2": ("contaminated: eve:7 (83% SM, 3.0 s)", 200.0),
        "3": ("no activity samples", 300.0),
        "4": (f"config invalid at this tree: {_TOY_INVALID}", 400.0),
        "5": ("other code at this tree: k2", 500.0),
        # The median of the other Triton autotunes.
        "6": ("no ok earlier record", 350.0),
    }
    cute = [job["estimate_s"] for job in result["autotune"] if job["backend"] == "cute"]
    assert cute == [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 600.0]
    assert [job["estimate_s"] for job in result["bench"]] == [30, 50, 40, 40, 40, 40]
    assert result["estimated_gpu_hours"] == {
        "cute_autotune": 4.33,
        "triton_autotune": 0.49,
        "bench": 0.07,
        "total": 4.89,
    }
    # Unsampled records may be reused; without the pin check none is.
    result = plan_run.plan(selection, [run5, run3], reuse_unsampled=True)
    assert [job["repro"] for job in result["reuse"]] == [
        "sum_000000000001",
        "sum_000000000003",
    ]
    calls.clear()
    result = plan_run.plan(selection, [run5, run3], pin_check=False)
    assert result["reuse"] == [] and calls == []
    reasons = [
        job["reason"] for job in result["autotune"] if job["backend"] == "triton"
    ]
    assert reasons.count("pinned code not checked") == 3


def test_a_run_plan_takes_no_provisional_record_for_an_earlier_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run stopped before its Triton phase still holds provisional copies,
    # which are no results: a plan neither reuses them nor has them stand in
    # again.
    selection, run5, run3 = _toy_priors(tmp_path)
    monkeypatch.setattr(plan_run, "pinned_code_matches", _toy_pins([]))
    stopped = tmp_path / "run6"
    for i in (1, 6):
        name = f"sum_00000000000{i}__s1__triton.json"
        record = {"repro": f"sum_00000000000{i}", "shape": "s1", "provisional": True}
        record["foreign_gpu_activity"] = _toy_activity(0)
        _toy_record(stopped, "configs", name, **record)
    result = plan_run.plan(selection, [stopped, run5, run3], provisional=True)
    assert [job["record"] for job in result["reuse"]] == [
        str(run5 / "configs" / "sum_000000000001__s1__triton.json")
    ]
    triton = {
        job["repro"][-1]: job
        for job in result["autotune"]
        if job["backend"] == "triton"
    }
    assert triton["6"]["reason"] == "no ok earlier record"
    assert "provisional" not in triton["6"]


def _plan_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *flags: str
) -> tuple[dict[str, list[dict[str, str]]], Path, list[str]]:
    """``plan_run.py --write`` on ``_toy_priors`` into ``tmp_path / "run"``:
    the selection, run5 and the command line."""
    selection, run5, run3 = _toy_priors(tmp_path)
    monkeypatch.setattr(plan_run, "pinned_code_matches", _toy_pins([]))
    # main() drops the caller's HELION_* settings; give them back afterwards.
    for key in [key for key in os.environ if key.startswith("HELION_")]:
        monkeypatch.delenv(key)
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    argv = [
        "plan_run.py",
        "--selection",
        str(tmp_path / "selection.json"),
        "--run-dir",
        str(tmp_path / "run"),
        "--prior",
        str(run5),
        "--prior",
        str(run3),
        "--write",
        *flags,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    plan_run.main()
    return selection, run5, argv


@pytest.mark.slow
@_requires_proc
def test_a_campaign_on_a_written_plan_runs_exactly_its_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selection, run5, argv = _plan_main(tmp_path, monkeypatch)
    run = tmp_path / "run"
    out = capsys.readouterr().out
    assert "autotunes: 11 (6 CuTe, 5 Triton); Triton records reused: 1" in out
    assert (
        "estimated GPU-hours: 4.33 CuTe autotune + 0.49 Triton autotune + 0.07 bench"
        "\ntotal: 4.89 GPU-hours\n"
    ) in out
    source = run5 / "configs" / "sum_000000000001__s1__triton.json"
    copied = json.loads((run / "configs" / source.name).read_text())
    assert copied == {**json.loads(source.read_text()), "reused_from": str(source)}
    planned = json.loads((run / "plan.json").read_text())
    expected = {
        f"autotune:{job['repro']}__{job['shape']}__{job['backend']}"
        for job in planned["autotune"]
    } | {f"bench:{job['repro']}__{job['shape']}" for job in planned["bench"]}
    _, events = _stub_campaign(
        tmp_path, monkeypatch, {}, kernels=selection["kernels"], backends="cute,triton"
    )
    started = [event["job"] for event in events if event["kind"] == "job_start"]
    assert sorted(started) == sorted(expected) and len(started) == 17
    # Planning into it again would leave the copies standing for jobs.
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        plan_run.main()
    assert "already holds records" in capsys.readouterr().err


def test_a_campaign_on_another_runs_picks_only_benches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Checked before any job starts or any record is copied.
    picks = tmp_path / "before" / "configs"
    picks.mkdir(parents=True)
    bench_only = ("--phases", "bench")
    for flags, message in (
        (["--configs-from", str(picks)], "--phases bench"),
        (
            ["--configs-from", str(picks), "--phases", "autotune,bench"],
            "--phases bench",
        ),
        (["--configs-from", str(tmp_path / "none"), "--phases", "bench"], "no such"),
        (
            ["--configs-from", str(tmp_path / "run" / "configs"), "--phases", "bench"],
            "no such",
        ),
        (["--rotate-inputs", "1"], "at least 2 input sets"),
    ):
        with pytest.raises(SystemExit):
            _stub_campaign(tmp_path, monkeypatch, {}, *flags)
        assert message in capsys.readouterr().err
    assert not (tmp_path / "run").exists() and not list(picks.iterdir())
    # Nor is a directory that holds none of the selected picks another run's
    # configs: the run itself given for its configs, say.
    _toy_tuned(tmp_path / "before", "sum_000000000002", "s2", "triton")
    for given in (picks, picks.parent):
        with pytest.raises(SystemExit):
            _stub_campaign(
                tmp_path, monkeypatch, {}, "--configs-from", str(given), *bench_only
            )
        error = capsys.readouterr().err
        assert f"{given} holds none of the 1 selected autotune records" in error
        assert "sum_000000000001__s1__triton.json" in error and "RUN/configs" in error
    assert not (tmp_path / "run").exists()
    (tmp_path / "run" / "configs").mkdir(parents=True)
    with pytest.raises(SystemExit):
        _stub_campaign(
            tmp_path,
            monkeypatch,
            {},
            "--configs-from",
            str(tmp_path / "run" / "configs"),
            "--phases",
            "bench",
        )
    assert "this run's own configs" in capsys.readouterr().err


@pytest.mark.slow
@_requires_proc
def test_a_campaign_adds_rotating_timings_to_another_runs_picks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The earlier run: a Triton pick for each of two points, a bench record
    # of the first, and a CuTe pick that --backends leaves out.
    before = tmp_path / "before"
    kernels = [
        {"repro": "sum_000000000001", "shape": "s1"},
        {"repro": "sum_000000000002", "shape": "s2"},
    ]
    for k in kernels:
        _toy_tuned(before, k["repro"], k["shape"], "triton")
    _toy_tuned(before, "sum_000000000001", "s1", "cute")
    _toy_bench(before, "sum_000000000001", "s1", {"inductor": 2.0})
    old = time.time() - 3600
    kept = {}
    for path in sorted(before.rglob("*.json")):
        os.utime(path, (old, old))
        kept[path] = (path.read_bytes(), path.stat().st_mtime_ns)
    flags = ("--configs-from", str(before / "configs"), "--phases", "bench")
    run, events = _stub_campaign(
        tmp_path, monkeypatch, {}, *flags, "--rotate-inputs", "8", kernels=kernels
    )
    names = [f"{k['repro']}__{k['shape']}" for k in kernels]
    # Only benches ran, on copies of the picks as they are, with the option.
    starts = [e["job"] for e in events if e["kind"] == "job_start"]
    assert sorted(starts) == [f"bench:{name}" for name in names]
    copies = [f"{name}__triton.json" for name in names]
    assert events[0]["kind"] == "configs_copied" and events[0]["records"] == copies
    assert events[0]["missing"] == events[0]["removed"] == []
    assert sorted(path.name for path in (run / "configs").iterdir()) == copies
    assert events[0]["replaced"] == []
    for name in copies:
        source, copy = before / "configs" / name, run / "configs" / name
        assert copy.read_bytes() == source.read_bytes()
        # A record that is new to this run, whenever the other run made it.
        assert copy.stat().st_mtime > old + 1800
    for name in names:
        assert (
            json.loads((run / "bench" / f"{name}.json").read_text())["status"] == "ok"
        )
        command = (run / "logs" / "bench" / f"{name}.log").read_text().splitlines()[0]
        assert f"--configs-dir {run / 'configs'} " in command
        # Of the Helion ports, the selected backend's only.
        assert command.endswith(
            "--impls inductor,inductor_cd,oracle,helion_triton,memcopy "
            "--rotate-inputs 8"
        )
    # The earlier run is as it was.
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(before.rglob("*"))
        if path.is_file()
    } == kept
    # Started again, it has nothing to copy or run; a pick that changed in
    # the earlier run since is copied and its point benched again.
    _, events = _stub_campaign(
        tmp_path, monkeypatch, {}, *flags, "--rotate-inputs", "8", kernels=kernels
    )
    assert [e["kind"] for e in events] == [
        "configs_copied",
        "campaign_start",
        "campaign_end",
    ]
    assert events[0]["records"] == []
    first_copy = (run / "configs" / copies[1]).read_bytes()
    _toy_tuned(before, "sum_000000000002", "s2", "triton", version=2)
    _, events = _stub_campaign(
        tmp_path, monkeypatch, {}, *flags, "--rotate-inputs", "8", kernels=kernels
    )
    assert events[0]["records"] == events[0]["replaced"] == [copies[1]]
    starts = [e["job"] for e in events if e["kind"] == "job_start"]
    assert starts == [f"bench:{names[1]}"]
    # A finished bench does not run again for another count of sets: the
    # campaign names the records that keep 8 sets.
    _, events = _stub_campaign(
        tmp_path, monkeypatch, {}, *flags, "--rotate-inputs", "4", kernels=kernels
    )
    assert [e["kind"] for e in events] == [
        "configs_copied",
        "rotate_inputs_not_added",
        "campaign_start",
        "campaign_end",
    ]
    assert events[0]["records"] == []
    assert (events[1]["sets"], events[1]["records"]) == (
        4,
        [f"{name}.json" for name in names],
    )
    assert "--rebench" in str(events[1]["reason"])
    # With the CuTe picks as well: the first point's is new to this run, so
    # the point is benched again, on 4 sets; the second point's, which the
    # earlier run lacks, is named, and its bench keeps what it has.
    _, events = _stub_campaign(
        tmp_path,
        monkeypatch,
        {},
        *flags,
        "--rotate-inputs",
        "4",
        kernels=kernels,
        backends="cute,triton",
    )
    assert events[0]["records"] == [f"{names[0]}__cute.json"]
    assert events[0]["replaced"] == []
    assert events[0]["missing"] == [f"{names[1]}__cute.json"]
    assert events[1]["kind"] == "rotate_inputs_not_added"
    assert events[1]["records"] == [f"{names[1]}.json"]
    starts = [e["job"] for e in events if e["kind"] == "job_start"]
    assert starts == [f"bench:{names[0]}"]
    command = (run / "logs" / "bench" / f"{names[0]}.log").read_text().splitlines()[0]
    assert f"--impls {','.join(common.ALL_IMPLS)} --rotate-inputs 4" in command
    for name, sets in zip(names, (4, 8), strict=True):
        record = json.loads((run / "bench" / f"{name}.json").read_text())
        assert record["rotate_inputs"] == {"sets": sets, "status": "ok"}
    # What the option copied, by the digest of each copy, and the copy that
    # the second point's replaced.
    manifest = json.loads((run / campaign.COPIED_CONFIGS).read_text())
    assert manifest.pop(copies[1])["replaces"] == hashlib.sha256(first_copy).hexdigest()
    assert manifest == {
        name: {
            "source": str(before / "configs" / name),
            "sha256": hashlib.sha256((run / "configs" / name).read_bytes()).hexdigest(),
        }
        for name in (copies[0], f"{names[0]}__cute.json")
    }


def _stopped() -> set[str] | None:
    """In place of ``campaign.busy_uuids``: the campaign is told to stop as
    it looks for a GPU for its first job."""
    raise campaign.Interrupted(signal.SIGTERM)


def _stopped_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kernels: list[dict[str, str]],
    backends: str,
    *flags: str,
) -> list[dict[str, object]]:
    """Start a campaign in ``tmp_path / "run"`` with ``flags`` that is
    stopped before a job starts: its events, the last two of them its
    ``campaign_start`` (with the number of jobs it queued) and its end."""
    log = tmp_path / "run" / "campaign.jsonl"
    before = len(log.read_text().splitlines()) if log.exists() else 0
    with contextlib.suppress(campaign.Interrupted):
        _stub_campaign(
            tmp_path,
            monkeypatch,
            {},
            *flags,
            kernels=kernels,
            backends=backends,
            busy_uuids=_stopped,
        )
    events = [json.loads(line) for line in log.read_text().splitlines()[before:]]
    assert [event["kind"] for event in events[-2:]] == [
        "campaign_start",
        "campaign_end",
    ]
    return events


def _copied_and_queued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    kernels: list[dict[str, str]],
    backends: str = "triton",
    *more: str,
) -> tuple[dict[str, object], object]:
    """Start a campaign in ``tmp_path / "run"`` on the picks of ``source``
    (another run's configs), with ``more`` flags, stopped before a job
    starts: its ``configs_copied`` event and the number of jobs it queued."""
    flags = ("--configs-from", str(source), "--phases", "bench", *more)
    events = _stopped_campaign(tmp_path, monkeypatch, kernels, backends, *flags)
    assert events[0]["kind"] == "configs_copied"
    return events[0], events[-2]["jobs"]


def _timed_on(backend: str, version: int) -> dict[str, object]:
    """What a bench entry of a port says it was pinned to: that version of
    ``_toy_kernels``."""
    return common.timed_record_fields({"kernels": _toy_kernels(backend, version)})


def test_a_point_is_benched_again_on_picks_its_bench_was_not_timed_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Earlier runs with a pick of the point each, all made an hour ago.
    kernels = [{"repro": "sum_000000000001", "shape": "s1"}]
    name = "sum_000000000001__s1__triton.json"
    run = tmp_path / "run"
    bench_record = run / "bench" / "sum_000000000001__s1.json"
    old = time.time() - 3600

    def pick(version: int) -> Path:
        source = tmp_path / f"picks{version}"
        _toy_tuned(
            source,
            "sum_000000000001",
            "s1",
            "triton",
            kernels=_toy_kernels("triton", version),
        )
        os.utime(source / "configs" / name, (old, old))
        return source / "configs"

    def queued(source: Path) -> tuple[object, object]:
        copied, jobs = _copied_and_queued(tmp_path, monkeypatch, source, kernels)
        assert (run / "configs" / name).read_bytes() == (source / name).read_bytes()
        return copied["records"], jobs

    def benched(version: int, when: float, attempt: int = 0) -> str:
        """The point's bench record, timed on that version of the pick and
        dated ``when``; with ``attempt``, a contaminated attempt at it."""
        _toy_bench(
            run,
            "sum_000000000001",
            "s1",
            {"inductor": 2.0, "helion_triton": 1.0},
            max_sm=83 if attempt else 0,
            helion_triton=_timed_on("triton", version),
        )
        os.utime(bench_record, (when, when))
        if attempt:
            kept = bench_record.with_name(f"{bench_record.name}.attempt{attempt}")
            bench_record.replace(kept)
            return str(kept)
        return str(bench_record)

    assert queued(pick(1)) == ([name], 1)
    # The bench finishes; a campaign started again has nothing to do.
    benched(1, time.time())
    assert queued(pick(1)) == ([], 0)
    # Another run's pick was made before that bench: it replaces the first
    # copy all the same, and the point is benched again on it.
    assert queued(pick(2)) == ([name], 1)
    # That campaign stopped before the bench started, and the bench record of
    # the pick before is dated after the copy (a clock that stepped back,
    # say). The campaigns after it copy nothing and still bench the point:
    # the record says which kernels it timed.
    benched(1, time.time() + 3600)
    assert queued(pick(2)) == ([], 1)
    assert queued(pick(2)) == ([], 1)
    benched(2, time.time() + 3600)
    assert queued(pick(2)) == ([], 0)
    # A campaign on that pick was contaminated at every attempt and stopped
    # before it kept the cleanest. Then a third pick is copied: the attempts
    # are no attempts at a bench on it, and neither that campaign nor, once
    # it has stopped too, the next one lets one of them stand for it.
    bench_record.unlink()
    stale = [benched(2, time.time(), attempt) for attempt in (1, 2, 3)]
    assert queued(pick(3)) == ([name], 1)
    assert queued(pick(3)) == ([], 1)
    assert not bench_record.exists()
    # The attempts on the pick the point has now do count: with the retries
    # spent, the cleanest of them is kept.
    spent = [benched(3, time.time(), attempt) for attempt in (4, 5, 6)]
    assert queued(pick(3)) == ([], 0)
    kept = json.loads(bench_record.read_text())
    assert kept["contaminated"] is True and kept["contaminated_attempts"] == spent
    assert kept["impls"]["helion_triton"]["autotune_code_sha256"] == {"k": "triton-3"}
    assert all(os.path.exists(path) for path in stale)


def test_configs_from_benches_only_the_ports_whose_picks_it_takes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A run with a CuTe and a Triton pick of the point, and a second run
    # with another Triton pick of it.
    kernels = [{"repro": "sum_000000000001", "shape": "s1"}]
    cute, triton = (f"sum_000000000001__s1__{b}.json" for b in ("cute", "triton"))
    first, second, run = (tmp_path / d for d in ("first", "second", "run"))
    for backend in ("cute", "triton"):
        _toy_tuned(first, "sum_000000000001", "s1", backend)
    _toy_tuned(
        second, "sum_000000000001", "s1", "triton", kernels=_toy_kernels("triton", 2)
    )

    def start(source: Path, backends: str, *more: str) -> tuple[object, ...]:
        copied, jobs = _copied_and_queued(
            tmp_path, monkeypatch, source / "configs", kernels, backends, *more
        )
        return copied["records"], ",".join(copied["bench_impls"]), jobs

    # Both backends: every implementation, in bench.py's order.
    every = "inductor,inductor_cd,oracle,helion_triton,helion_cute,memcopy"
    assert every == ",".join(common.ALL_IMPLS)
    assert start(first, "cute,triton") == ([cute, triton], every, 1)
    times = {"inductor": 2.0, "helion_cute": 1.0, "helion_triton": 1.5}
    _toy_bench(run, "sum_000000000001", "s1", times)
    assert start(first, "cute,triton")[2] == 0
    # The second run's Triton pick replaces the copy of the first run's, and
    # the copy of the first run's CuTe pick stays, since CuTe is not
    # selected: the bench that the point gets leaves the CuTe port out, so it
    # does not time the first run's pick beside the second run's.
    without_cute = every.replace("helion_cute,", "")
    assert start(second, "triton") == ([triton], without_cute, 1)
    held = {path.name: path.read_bytes() for path in (run / "configs").iterdir()}
    assert held == {
        cute: (first / "configs" / cute).read_bytes(),
        triton: (second / "configs" / triton).read_bytes(),
    }
    # --bench-impls narrows the list and may not widen it to a port whose
    # pick the option does not take.
    narrowed = "inductor,helion_triton"
    assert start(second, "triton", "--bench-impls", narrowed) == ([], narrowed, 1)
    kept = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit):
        start(second, "triton", "--bench-impls", "inductor,helion_cute,helion_triton")
    error = capsys.readouterr().err
    assert "--bench-impls names helion_cute, and --backends triton takes no" in error
    assert kept == {
        path: path.read_bytes() for path in run.rglob("*") if path.is_file()
    }
    # That bench finishes, on the second run's Triton pick alone and dated an
    # hour from now, so no date makes it stale.
    bench_record = run / "bench" / "sum_000000000001__s1.json"

    def benched(**ports: dict[str, object]) -> None:
        times = {"inductor": 2.0, **dict.fromkeys(ports, 1.5)}
        _toy_bench(run, "sum_000000000001", "s1", times, **ports)
        os.utime(bench_record, (time.time() + 3600,) * 2)

    benched(helion_triton=_timed_on("triton", 2))
    assert start(second, "triton") == ([], without_cute, 0)
    # It left the CuTe port out. That is nothing to bench the point again
    # for while the port has no pick: on the second run's picks of both
    # backends the copy of the first run's CuTe pick goes.
    assert start(second, "cute,triton") == ([], every, 0)
    assert not (run / "configs" / cute).exists()
    # On the first run's CuTe pick the point is benched again, by the start
    # that copies the pick and by one after it, which copies nothing: the
    # bench record has no timing of the port. The record stands for a start
    # whose benches leave the port out as well.
    without_triton = every.replace("helion_triton,", "")
    assert start(first, "cute") == ([cute], without_triton, 1)
    assert start(first, "cute") == ([], without_triton, 1)
    assert start(first, "cute", "--bench-impls", "inductor") == ([], "inductor", 0)
    benched(helion_cute=_timed_on("cute", 1))
    assert start(first, "cute") == ([], without_triton, 0)
    # So it is without --configs-from: a bench with fewer --bench-impls is
    # no bench of the ports that a restart without them times.
    plain = ("--phases", "bench")
    stopped = _stopped_campaign(tmp_path, monkeypatch, kernels, "cute,triton", *plain)
    assert stopped[-2]["jobs"] == 1
    stopped = _stopped_campaign(
        tmp_path, monkeypatch, kernels, "cute,triton", *plain, "--bench-impls", "oracle"
    )
    assert stopped[-2]["jobs"] == 0
    stopped = _stopped_campaign(tmp_path, monkeypatch, kernels, "cute", *plain)
    assert stopped[-2]["jobs"] == 0


def test_a_campaign_names_the_finished_benches_without_the_rotating_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Five finished benches: a pass on 8 sets that timed the port, one that
    # left it out by rule, one that failed, a pass on 4 sets, and no pass.
    run = tmp_path / "run"
    passes: list[dict[str, object] | None] = [
        {"sets": 8, "status": "ok"},
        {"sets": 8, "status": "skipped", "skipped": "inputs of 268.4 MB exceed"},
        {"sets": 8, "status": "error", "error": "could not capture graphs"},
        {"sets": 4, "status": "ok"},
        None,
    ]
    kernels = [
        {"repro": f"sum_00000000000{index}", "shape": "s1"}
        for index in range(1, len(passes) + 1)
    ]
    for k, decision in zip(kernels, passes, strict=True):
        _toy_tuned(run, k["repro"], k["shape"], "triton")
        _toy_bench(run, k["repro"], k["shape"], {"inductor": 2.0, "helion_triton": 1.0})
        if decision is not None:
            _add_fields(run, k["repro"], k["shape"], rotate_inputs=decision)
    kept = {path: path.read_bytes() for path in (run / "bench").iterdir()}
    flags = ("--phases", "bench", "--rotate-inputs", "8")
    events = _stopped_campaign(tmp_path, monkeypatch, kernels, "triton", *flags)
    # None of them runs again; the three that lack the timing are named.
    assert [event["kind"] for event in events[:-2]] == ["rotate_inputs_not_added"]
    assert events[0]["sets"] == 8 and events[-2]["jobs"] == 0
    assert events[0]["records"] == [
        f"{k['repro']}__{k['shape']}.json" for k in kernels[2:]
    ]
    assert kept == {path: path.read_bytes() for path in (run / "bench").iterdir()}
    # Without the option there is nothing to name.
    events = _stopped_campaign(tmp_path, monkeypatch, kernels, "triton", *flags[:2])
    assert len(events) == 2 and events[0]["jobs"] == 0


def test_configs_from_replaces_only_the_records_it_copied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The run autotuned the first point itself, and the other run's pick of
    # it differs: nothing is copied, the second point's pick neither, and
    # the run is as it was.
    kernels = [
        {"repro": "sum_000000000001", "shape": "s1"},
        {"repro": "sum_000000000002", "shape": "s2"},
    ]
    before, run = tmp_path / "before", tmp_path / "run"
    for k in kernels:
        _toy_tuned(
            before, k["repro"], k["shape"], "triton", kernels=_toy_kernels("triton", 2)
        )
    _toy_tuned(run, "sum_000000000001", "s1", "triton")
    own = run / "configs" / "sum_000000000001__s1__triton.json"

    def files() -> dict[Path, bytes]:
        return {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}

    def refused(backends: str = "triton") -> str:
        kept = files()
        flags = ("--configs-from", str(before / "configs"), "--phases", "bench")
        with pytest.raises(SystemExit):
            _stub_campaign(
                tmp_path, monkeypatch, {}, *flags, kernels=kernels, backends=backends
            )
        assert files() == kept
        return capsys.readouterr().err

    error = refused()
    assert f"1 autotune record of this run that it did not copy ({own.name})" in error
    assert "--run-dir" in error
    # With the record out of the way the picks are copied. A copy that an
    # autotune of this run then replaces is the run's own record too.
    own.unlink()
    copied, jobs = _copied_and_queued(
        tmp_path, monkeypatch, before / "configs", kernels
    )
    assert copied["records"] == [own.name, "sum_000000000002__s2__triton.json"]
    assert jobs == 2
    _toy_tuned(run, "sum_000000000001", "s1", "triton", kernels=_toy_kernels("own"))
    _toy_tuned(
        before, "sum_000000000001", "s1", "triton", kernels=_toy_kernels("triton", 3)
    )
    assert f"did not copy ({own.name})" in refused()
    # So is a record of the run's own that the other run has no pick for: a
    # bench would time it beside the other run's picks. The copies stay.
    own.write_bytes((before / "configs" / own.name).read_bytes())
    _toy_tuned(run, "sum_000000000002", "s2", "cute")
    error = refused("cute,triton")
    assert "did not copy (sum_000000000002__s2__cute.json) is not" in error
    # The run's record of the first point is the other run's now, byte for
    # byte, and still no copy of the option's: the entry of the copy that it
    # replaced goes, and the record stays when the other run's pick changes.
    (run / "configs" / "sum_000000000002__s2__cute.json").unlink()
    copied, _ = _copied_and_queued(tmp_path, monkeypatch, before / "configs", kernels)
    assert copied["records"] == []
    assert list(json.loads((run / campaign.COPIED_CONFIGS).read_text())) == [
        "sum_000000000002__s2__triton.json"
    ]
    _toy_tuned(
        before, "sum_000000000001", "s1", "triton", kernels=_toy_kernels("triton", 4)
    )
    assert f"did not copy ({own.name})" in refused()


def test_a_copy_goes_when_its_source_has_no_such_pick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # One run with a CuTe and a Triton pick of the point; another with that
    # Triton pick, byte for byte, and no CuTe pick.
    kernels = [{"repro": "sum_000000000001", "shape": "s1"}]
    cute, triton = (f"sum_000000000001__s1__{b}.json" for b in ("cute", "triton"))
    both, one, run = (tmp_path / d for d in ("both", "one", "run"))
    for backend in ("cute", "triton"):
        _toy_tuned(both, "sum_000000000001", "s1", backend)
    _toy_tuned(one, "sum_000000000001", "s1", "triton")
    bench_record = run / "bench" / "sum_000000000001__s1.json"

    def start(source: Path) -> tuple[object, object, object, object]:
        copied, jobs = _copied_and_queued(
            tmp_path, monkeypatch, source / "configs", kernels, "cute,triton"
        )
        return copied["records"], copied["removed"], copied["missing"], jobs

    def benched(**ports: dict[str, object]) -> None:
        """The point's bench record, dated an hour from now: no date makes
        it stale."""
        times = {"inductor": 2.0, "helion_cute": 1.0, "helion_triton": 1.5}
        _toy_bench(run, "sum_000000000001", "s1", times, **ports)
        os.utime(bench_record, (time.time() + 3600,) * 2)

    assert start(both) == ([cute, triton], [], [], 1)
    benched()
    assert start(both) == ([], [], [], 0)
    # On the other run's picks, this run's copy of the first run's CuTe pick
    # goes (a bench would time it beside the other run's Triton pick), and
    # the point is benched again: its record has a timing of that pick.
    assert start(one) == ([], [cute], [cute], 1)
    assert sorted(path.name for path in (run / "configs").iterdir()) == [triton]
    assert list(json.loads((run / campaign.COPIED_CONFIGS).read_text())) == [triton]
    assert start(one) == ([], [], [cute], 1)
    # That bench has an error for the CuTe port, which names no record.
    no_record = dict.fromkeys(_timed_on("cute", 1))
    benched(helion_cute={"status": "error", "cold_us": None, **no_record})
    assert start(one) == ([], [], [cute], 0)
    # Back on the first run's picks its CuTe pick is copied again, and the
    # point benched again for it, also by a campaign after the one that
    # copied it.
    assert start(both) == ([cute], [], [], 1)
    assert start(both) == ([], [], [], 1)
    # A directory without any of the picks (the run, not its configs) is
    # refused before the copies would go.
    kept = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit):
        _copied_and_queued(tmp_path, monkeypatch, one, kernels, "cute,triton")
    assert "holds none of the 2 selected autotune records" in capsys.readouterr().err
    assert kept == {
        path: path.read_bytes() for path in run.rglob("*") if path.is_file()
    }


class _Died(BaseException):
    """The campaign's process is gone (SIGKILL, a power cut)."""


def test_configs_from_knows_its_copies_wherever_a_copy_died(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = [
        {"repro": "sum_000000000001", "shape": "s1"},
        {"repro": "sum_000000000002", "shape": "s2"},
    ]

    def picks(root: str, version: int, points: int = 2) -> Path:
        for k in kernels[:points]:
            _toy_tuned(
                tmp_path / root,
                k["repro"],
                k["shape"],
                "triton",
                kernels=_toy_kernels("triton", version),
            )
        return tmp_path / root / "configs"

    def start(root: Path, source: Path, die_at: int = 0) -> int:
        """A campaign in ``root / "run"`` on the picks of ``source`` that
        dies before its ``die_at``-th change to the run's files, each a file
        put in place or removed (0: it lives, stopped before a job starts);
        the changes it made."""
        root.mkdir(exist_ok=True)
        changes = 0

        def counted(change: Callable[..., object]) -> Callable[..., object]:
            def step(*args: object) -> object:
                nonlocal changes
                if changes + 1 == die_at:
                    raise _Died
                changes += 1
                return change(*args)

            return step

        with monkeypatch.context() as patched:
            for change in ("replace", "unlink"):
                patched.setattr(
                    pathlib.Path, change, counted(getattr(pathlib.Path, change))
                )
            if die_at:
                flags = ("--configs-from", str(source), "--phases", "bench")
                with pytest.raises(_Died):
                    _stub_campaign(root, patched, {}, *flags, kernels=kernels)
            else:
                _copied_and_queued(root, patched, source, kernels)
        return changes

    # The run holds one run's picks. A campaign on other picks dies before
    # its first, second, ... change: to its record of the copies, or to a
    # copy. The next campaign, on those picks, a third run's or the first
    # one's again, knows every record the run then holds as a copy and
    # finishes the job: the records and their entries are those of its
    # picks, and no entry is left of a copy that was not made or is gone.
    first, second, third = picks("first", 1), picks("second", 2), picks("third", 3)
    lacking = picks("lacking", 2, 1)
    for held_first, dying, expected in (
        # Both picks replaced; one replaced and the copy of the other, which
        # the dying campaign's run lacks, removed; one replaced and one new.
        (first, second, 3),
        (first, lacking, 4),
        (lacking, first, 3),
    ):
        case = f"{held_first.parent.name}-{dying.parent.name}"
        start(tmp_path / f"{case}-whole", held_first)
        steps = start(tmp_path / f"{case}-whole", dying)
        assert steps == expected
        for die_at in range(1, steps + 1):
            for after in (dying, third, held_first):
                root = tmp_path / f"{case}-{die_at}-{after.parent.name}"
                start(root, held_first)
                assert start(root, dying, die_at) == die_at - 1
                start(root, after)
                held = sorted(
                    path.name for path in (root / "run" / "configs").iterdir()
                )
                assert held == sorted(path.name for path in after.iterdir())
                manifest = json.loads(
                    (root / "run" / campaign.COPIED_CONFIGS).read_text()
                )
                assert sorted(manifest) == held
                for name in held:
                    copy = (root / "run" / "configs" / name).read_bytes()
                    assert copy == (after / name).read_bytes()
                    assert manifest[name]["source"] == str(after / name)
                    assert manifest[name]["sha256"] == hashlib.sha256(copy).hexdigest()


_SEED = common.SEED_VARIABLE


def test_a_runs_seed_for_a_port_is_the_one_its_standing_record_names(
    tmp_path: Path,
) -> None:
    run = tmp_path / "before"

    def seed_of(index: int, *started: int, **fields: object) -> tuple[object, object]:
        """The seed the run names for a port whose campaign job started
        with the seeds ``started`` and left a record with ``fields`` (none
        without them)."""
        repro = f"sum_00000000000{index}"
        if fields:
            _toy_tuned(run, repro, "s1", "cute", **fields)
        run.mkdir(exist_ok=True)
        with (run / "campaign.jsonl").open("a") as log:
            for seed in started:
                start = {"job": f"autotune:{repro}__s1__cute", "seed": seed}
                log.write(json.dumps({"kind": "job_start", **start}) + "\n")
        return common.standing_seed(run, repro, "s1", "cute")

    # The seed is in the record's environment, whichever of the job's
    # attempts made it: here the second of three, the cleanest, which the
    # campaign kept when the last one was contaminated too.
    configs = run / "configs"
    attempts = [
        str(configs / f"sum_000000000001__s1__cute.json.attempt{k}") for k in (1, 2, 3)
    ]
    kept = {"contaminated": True, "contaminated_attempts": attempts}
    assert seed_of(1, 7, 11, 9, env={_SEED: "11"}, **kept) == (11, None)
    # So is a reused record's: its own search's, in the run it came from.
    assert seed_of(2, env={_SEED: "22"}, reused_from="run3/x.json") == (22, None)
    # No search of this run to repeat: no record, a failed autotune, a
    # provisional record, a search that was given no seed (autotune.py by
    # hand) or a record without its environment, whatever seed the campaign
    # started the job with.
    assert seed_of(3, 33) == (None, "no autotune record")
    assert seed_of(4, 44, env={_SEED: "44"}, status="error") == (
        None,
        "its autotune record is not ok",
    )
    found = seed_of(5, env={_SEED: "55"}, provisional=True, reused_from="run3/y.json")
    assert found[0] is None and "provisional" in str(found[1])
    no_seed = (None, "its autotune record names no seed")
    assert seed_of(6, 66, env={"HELION_FORCE_AUTOTUNE": "1"}) == no_seed
    assert seed_of(7, 77, 77, autotune_s=2.0) == no_seed
    # A blank seed is none to Helion, which then draws one from the clock,
    # and a padded one is the number.
    assert seed_of(8, env={_SEED: ""}) == no_seed
    assert seed_of(9, env={_SEED: "   "}) == no_seed
    assert seed_of(9, env={_SEED: " 99 "}) == (99, None)


def test_an_autotune_record_names_its_seed_and_the_run_it_was_taken_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The GPU's and the packages' identities are other tests'.
    monkeypatch.setattr(common, "gpu_info", dict)
    monkeypatch.setattr(common, "versions", dict)
    # The job as a campaign starts it on a fresh seed: its record holds the
    # HELION_* environment, the seed and the search log's file among it,
    # and nothing says the seed is another run's.
    monkeypatch.delenv(common.SEED_FROM, raising=False)
    monkeypatch.setenv(_SEED, "1234")
    monkeypatch.setenv("HELION_AUTOTUNE_LOG", "/run/logs/search/x/search")
    point = ("sum_000000000001", "s1", "cute")
    record = autotune.new_record(*point)
    assert (record["repro"], record["shape"], record["backend"]) == point
    assert record["env"][_SEED] == "1234" and common.record_seed(record) == 1234
    assert record["env"]["HELION_AUTOTUNE_LOG"] == "/run/logs/search/x/search"
    assert all(name.startswith("HELION_") for name in record["env"])
    assert "seed_from" not in record
    # With --seeds-from the campaign tells the job where the seed is from.
    monkeypatch.setenv(common.SEED_FROM, "/runs/before")
    assert autotune.new_record(*point) == {**record, "seed_from": "/runs/before"}

    # An autotune that fails, here for want of a device, records them too.
    def no_device(index: int) -> None:
        raise RuntimeError("no device for this test")

    monkeypatch.setattr(torch.cuda, "set_device", no_device)
    job = argparse.Namespace(repro_id=point[0], shape="s1", backend="cute")
    failed = autotune.run(job)
    assert failed["status"] == "error" and "no device for this test" in failed["error"]
    assert failed["env"] == record["env"] and failed["seed_from"] == "/runs/before"


def _seeded_and_queued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    kernels: list[dict[str, str]],
    *flags: str,
    backends: str = "cute,triton",
) -> tuple[dict[str, int], dict[str, str], object]:
    """Start a campaign in ``tmp_path / "run"`` on the seeds of the run
    ``source``, stopped before a job starts: the seeds it took and the
    autotunes it left on a fresh seed (by repro digit and backend, from its
    ``seeds_from`` event) and the number of jobs it queued."""
    events = _stopped_campaign(
        tmp_path, monkeypatch, kernels, backends, "--seeds-from", str(source), *flags
    )
    assert [event["kind"] for event in events] == [
        "seeds_from",
        "campaign_start",
        "campaign_end",
    ]
    assert set(events[0]) == {"kind", "source", "seeds", "fresh", "time"}
    assert events[0]["source"] == str(source)

    def short(jobs: dict[str, object]) -> dict[str, object]:
        return {
            "".join((key.split("__")[0][-1], *key.rsplit("__", 1)[1:])): value
            for key, value in jobs.items()
        }

    seeds, fresh = short(events[0]["seeds"]), short(events[0]["fresh"])
    return (
        cast("dict[str, int]", seeds),
        cast("dict[str, str]", fresh),
        events[1]["jobs"],
    )


def test_a_campaign_takes_another_runs_seeds_for_the_autotunes_it_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The run before: the first point's records name their seeds (the
    # Triton one is reused, from the run before that), the second's CuTe
    # record is provisional, and the third has no ok record.
    before, run = tmp_path / "before", tmp_path / "run"
    kernels = [{"repro": f"sum_00000000000{i}", "shape": f"s{i}"} for i in (1, 2, 3)]
    one, two, three = (k["repro"] for k in kernels)
    earlier = {"env": {_SEED: "102"}, "reused_from": "run3/configs/x.json"}
    _toy_tuned(before, one, "s1", "cute", env={_SEED: "101"})
    _toy_tuned(before, one, "s1", "triton", **earlier)
    _toy_tuned(before, two, "s2", "cute", env={_SEED: "201"}, provisional=True)
    _toy_tuned(before, two, "s2", "triton", env={_SEED: "202"})
    _toy_tuned(before, three, "s3", "triton", env={_SEED: "302"}, status="error")
    # Refused before anything is made: a run's configs for the run, a run
    # that names no seed for any selected autotune, benches only.
    for flags, message in (
        (["--seeds-from", str(before / "configs")], "RUN, not RUN/configs"),
        (
            ["--seeds-from", str(before), "--only", three],
            (
                "names no seed for any of the 2 selected autotunes "
                f"(autotune:{three}__s3__cute: no autotune record, say)"
            ),
        ),
        (["--seeds-from", str(before), "--phases", "bench"], "--phases autotune"),
    ):
        with pytest.raises(SystemExit):
            _stub_campaign(
                tmp_path,
                monkeypatch,
                {},
                *flags,
                kernels=kernels,
                backends="cute,triton",
            )
        assert message in capsys.readouterr().err
    assert not run.exists()
    # This run as plan_run.py leaves it: the first point's Triton record
    # reused, which is no job, and the second's provisional, whose autotune
    # is still to run.
    _toy_tuned(run, one, "s1", "triton", **earlier)
    _toy_tuned(run, two, "s2", "triton", env={_SEED: "9"}, provisional=True)
    fresh = {
        "2cute": "its autotune record is provisional: not a search of that run",
        "3cute": "no autotune record",
        "3triton": "its autotune record is not ok",
    }
    # Each autotune to run takes the seed of the other run's record of its
    # port; the reused record's takes none, and a restart takes them again.
    for _ in range(2):
        queued = _seeded_and_queued(tmp_path, monkeypatch, before, kernels)
        assert queued == ({"1cute": 101, "2triton": 202}, fresh, 8)
    assert _seeded_and_queued(
        tmp_path, monkeypatch, before, kernels, "--only", one, two, backends="cute"
    ) == ({"1cute": 101}, {"2cute": fresh["2cute"]}, 4)
    # Run again, the reused record's port searches from its record's seed.
    assert _seeded_and_queued(tmp_path, monkeypatch, before, kernels, "--retune") == (
        {"1cute": 101, "1triton": 102, "2triton": 202},
        fresh,
        9,
    )
    # A run's own records move aside as its jobs start.
    with pytest.raises(SystemExit):
        _seeded_and_queued(tmp_path, monkeypatch, run, kernels)
    assert "names this run" in capsys.readouterr().err
    # A job is told of a seed's source and of a search log only by its own
    # campaign.
    monkeypatch.setenv(common.SEED_FROM, "another campaign's")
    monkeypatch.setenv("HELION_AUTOTUNE_LOG", "the caller's")
    env = campaign.job_env("0", tmp_path, 7, "marker")
    assert env[_SEED] == "7" and not {common.SEED_FROM, "HELION_AUTOTUNE_LOG"} & set(
        env
    )
    env = campaign.job_env("0", tmp_path, 7, "marker", "run9", tmp_path / "search")
    assert env[common.SEED_FROM] == "run9"
    assert env["HELION_AUTOTUNE_LOG"] == str(tmp_path / "search")


@pytest.mark.slow
@_requires_proc
def test_a_paired_autotune_searches_from_its_seed_on_every_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The run before names a seed for the first point only. Both points'
    # autotunes are contaminated once and run again.
    before = tmp_path / "before"
    kernels = [{"repro": f"sum_00000000000{i}", "shape": "s1"} for i in (1, 2)]
    _toy_tuned(before, "sum_000000000001", "s1", "triton", env={_SEED: "1234"})
    names = [f"{k['repro']}__s1__triton" for k in kernels]
    plan = {f"{name}.json": [83, 0] for name in names}
    drawn = itertools.count(1)
    monkeypatch.setattr(campaign.secrets, "randbits", lambda bits: next(drawn))
    flags = ("--seeds-from", str(before))
    run, events = _stub_campaign(tmp_path, monkeypatch, plan, *flags, kernels=kernels)

    def starts(events: list[dict[str, object]], name: str) -> list[object]:
        return [
            (event["seed"], event.get("seed_from"))
            for event in events
            if event["kind"] == "job_start" and event["job"] == f"autotune:{name}"
        ]

    # The paired job has the other run's seed on both attempts, and its
    # event, its log and each attempt's record say where it is from; the
    # other draws a seed per attempt, and nothing calls it paired.
    assert starts(events, names[0]) == [(1234, str(before))] * 2
    fresh = starts(events, names[1])
    assert len({seed for seed, _ in fresh}) == 2 and {s for _, s in fresh} == {None}
    # A bench is no search: each start draws a seed, which nothing calls
    # another run's, and has no search log.
    benches = [
        event
        for event in events
        if event["kind"] == "job_start" and str(event["job"]).startswith("bench:")
    ]
    assert len({event["seed"] for event in benches} - {1234}) == len(benches) == 2
    assert not any({"seed_from", "search_log"} & set(event) for event in benches)
    configs = run / "configs"
    for suffix in ("", ".attempt1"):
        record = json.loads((configs / f"{names[0]}.json{suffix}").read_text())
        assert record["env"][_SEED] == "1234" and record["seed_from"] == str(before)
        log = run / "logs" / "autotune" / f"{names[0]}.log{suffix}"
        assert f" seed=1234 seed_from={before}\n" in log.read_text()
        assert "seed_from" not in (configs / f"{names[1]}.json{suffix}").read_text()
    # Each attempt keeps the log of its own search, in the file that its
    # start's event names for the job.
    search = run / "logs" / "search"
    logged = {
        (event["job"], event["search_log"])
        for event in events
        if event["kind"] == "job_start" and "search_log" in event
    }
    assert logged == {
        (f"autotune:{name}", str(search / name / "search")) for name in names
    }
    rows = {path.parent.name: path.read_text() for path in search.rglob("*.csv")}
    assert rows == {
        **dict.fromkeys(names, "attempt 2\n"),
        **{f"{name}.attempt1": "attempt 1\n" for name in names},
    }
    assert record["env"]["HELION_AUTOTUNE_LOG"] == str(search / names[0] / "search")
    # Run again without search logs, it searches from that seed again. The
    # search logs of the run before move aside with its record; those of a
    # start that left no record go, as its log does (the autotuner would
    # append the next search to them).
    (configs / f"{names[1]}.json").unlink()
    flags = (*flags, "--retune", "--no-search-logs")
    run, events = _stub_campaign(tmp_path, monkeypatch, {}, *flags, kernels=kernels)
    assert starts(events, names[0]) == [(1234, str(before))]
    assert not any("search_log" in event for event in events)
    record = json.loads((configs / f"{names[0]}.json").read_text())
    assert record["env"][_SEED] == "1234" and "HELION_AUTOTUNE_LOG" not in record["env"]
    assert sorted(path.name for path in search.iterdir()) == [
        f"{names[0]}.attempt1",
        f"{names[0]}.prev",
        f"{names[1]}.attempt1",
    ]
    assert (search / f"{names[0]}.prev" / "search.csv").read_text() == "attempt 2\n"


@pytest.mark.slow
@_requires_proc
def test_the_files_a_job_sets_aside_under_one_name_are_one_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An autotune that runs eight times: the campaign's k-th start draws the
    # seed k, which its log's header names, and the stub job writes k to its
    # record and to its search log.
    name = "sum_000000000001__s1__triton"
    logs = tmp_path / "run" / "logs"
    record = tmp_path / "run" / "configs" / f"{name}.json.prev"
    log = logs / "autotune" / f"{name}.log.prev"
    rows = logs / "search" / f"{name}.prev" / "search.csv"
    drawn = itertools.count(1)
    monkeypatch.setattr(campaign.secrets, "randbits", lambda bits: next(drawn))
    # Its sixth start is contaminated and the seventh, the retry, fails
    # without a record, so the sixth's record is kept for the job while its
    # log and search logs stay the attempt's.
    plan = {f"{name}.json": [0, 0, 0, 0, 0, 83, None, 0]}
    seed = re.compile(r" seed=(\d+)$", re.MULTILINE)
    # After each campaign, the start whose record, log and search logs are
    # the job's .prev ones (None: no such file). They are the previous
    # run's, each of them or none: an older run's do not stay where the
    # previous one kept no search logs (the fourth start) or left its log
    # and search logs with an attempt (the sixth).
    for flags, previous in (
        ((), (None, None, None)),
        (("--retune",), (1, 1, 1)),
        (("--retune",), (2, 2, 2)),
        (("--retune", "--no-search-logs"), (3, 3, 3)),
        (("--retune",), (4, 4, None)),
        (("--retune",), (5, 5, 5)),
        (("--retune",), (6, None, None)),
    ):
        _stub_campaign(tmp_path, monkeypatch, plan, "--phases", "autotune", *flags)
        assert previous == (
            json.loads(record.read_text())["attempt"] if record.exists() else None,
            int(cast("re.Match[str]", seed.search(log.read_text()))[1])
            if log.exists()
            else None,
            int(rows.read_text().split()[1]) if rows.exists() else None,
        )
    kept = json.loads(record.read_text())
    assert kept["contaminated"] and len(kept["contaminated_attempts"]) == 1
    assert sorted(path.name for path in (logs / "search").iterdir()) == [
        name,
        f"{name}.attempt1",
    ]
    assert sorted(path.name for path in (logs / "autotune").iterdir()) == [
        f"{name}.log",
        f"{name}.log.attempt1",
        f"{name}.log.attempt2",
    ]


def _toy_kernels(backend: str, version: int = 1) -> dict[str, dict[str, str]]:
    """The kernels of a toy autotune record."""
    return {
        "k": {
            "decorator": f"@helion.kernel(version={version})",
            "code_sha256": f"{backend}-{version}",
        }
    }


def _toy_bench(
    run: Path,
    repro: str,
    shape: str,
    times: dict[str, float],
    max_sm: int | None = 0,
    **impls: dict[str, object],
) -> None:
    """A bench record with these cold_us (and doubled bb_us) timings and a
    foreign process at ``max_sm`` (None: no activity samples), each port
    timed on the kernels of ``_toy_tuned``'s record; ``impls`` overrides an
    implementation's fields (None drops one)."""

    def pinned(impl: str) -> dict[str, object]:
        if impl not in report.PORTS:
            return {}
        return common.timed_record_fields({"kernels": _toy_kernels(report.PORTS[impl])})

    record: dict[str, object] = {
        "repro": repro,
        "shape": shape,
        "status": "ok",
        "impls": {
            impl: {
                field: entry
                for field, entry in {
                    "status": "ok",
                    "numerics_pass": True,
                    "cold_us": value,
                    "bb_us": 2 * value,
                    **pinned(impl),
                    **impls.get(impl, {}),
                }.items()
                if entry is not None
            }
            for impl, value in times.items()
        },
    }
    if max_sm is not None:
        record["foreign_gpu_activity"] = _toy_activity(max_sm)
    path = run / "bench" / f"{repro}__{shape}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _toy_tuned(
    run: Path,
    repro: str,
    shape: str,
    backend: str,
    max_sm: int | None = 0,
    **fields: object,
) -> None:
    record = {
        "repro": repro,
        "shape": shape,
        "backend": backend,
        "autotune_s": 1.0,
        "kernels": _toy_kernels(backend),
    }
    if max_sm is not None:
        record["foreign_gpu_activity"] = _toy_activity(max_sm)
    record.update(fields)
    _toy_record(run, "configs", f"{repro}__{shape}__{backend}.json", **record)


def test_the_final_tables_count_only_clean_sampled_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, before = tmp_path / "run6", tmp_path / "run3"
    t5 = "sum_sum_sum_d2c97d17f3dc"
    times = {"inductor": 1000.0, "helion_cute": 50.0, "helion_triton": 400.0}
    _toy_bench(run, t5, "t5", {**times, "memcopy": 20.0})
    times = {"inductor": 100.0, "inductor_cd": 90.0, "helion_cute": 60.0}
    times.update(helion_triton=120.0, oracle=80.0, memcopy=30.0)
    # Its gate's float64 reference failed (common.AnchoredGate).
    eager = {"pass": True, "ref_precision": "eager", "ref_fallback": "RuntimeError: x"}
    _toy_bench(run, "sum_000000000002", "s2", times, helion_cute={"anchored": eager})
    times = {"inductor": 40.0, "helion_cute": 50.0, "helion_triton": 40.0}
    _toy_bench(run, "amax_000000000003", "s3", {**times, "memcopy": 20.0})
    # A foreign process computed during this bench, and nothing tells for
    # the next; the cute port of the last failed its numerics.
    _toy_bench(
        run, "sum_000000000004", "s4", {"inductor": 10.0, "helion_cute": 5.0}, 70
    )
    times = {"inductor": 10.0, "helion_cute": 4.0, "helion_triton": 5.0}
    _toy_bench(run, "amax_000000000005", "s5", times, None)
    times = {"inductor": 30.0, "helion_cute": 10.0, "helion_triton": 20.0}
    _toy_bench(
        run, "sum_000000000006", "s6", times, helion_cute={"numerics_pass": False}
    )
    # The kernels the selection lists, with their model shares.
    listed = [
        (t5, "t5", 1.0),
        ("sum_000000000002", "s2", 2.0),
        ("amax_000000000003", "s3", 1.0),
    ]
    for repro, shape, _ in listed:
        _toy_tuned(run, repro, shape, "cute")
    _toy_tuned(run, t5, "t5", "triton")
    _toy_tuned(run, "sum_000000000002", "s2", "triton", reused_from="run5/x.json")
    # Made before the sampling: its port stays out of the aggregates.
    _toy_tuned(run, "amax_000000000003", "s3", "triton", None)
    _toy_tuned(run, "sum_000000000004", "s4", "cute")
    _toy_tuned(run, "amax_000000000005", "s5", "cute")
    _toy_tuned(run, "amax_000000000005", "s5", "triton")
    _toy_tuned(run, "sum_000000000006", "s6", "triton")
    attempts = ["a.attempt1", "a.attempt2", "a.attempt3"]
    _toy_tuned(
        run,
        "sum_000000000006",
        "s6",
        "cute",
        40,
        contaminated=True,
        contaminated_attempts=attempts,
    )
    # The baseline run: its T5 bench was contaminated, and so was its CuTe
    # autotune of s3, which leaves s3 out of the comparison too.
    _toy_bench(before, t5, "t5", {"inductor": 1000.0, "helion_cute": 100.0}, 90)
    _toy_bench(
        before, "sum_000000000002", "s2", {"inductor": 100.0, "helion_cute": 200.0}
    )
    _toy_bench(
        before, "amax_000000000003", "s3", {"inductor": 40.0, "helion_cute": 100.0}
    )
    for repro, shape, _ in listed:
        _toy_tuned(before, repro, shape, "cute", 90 if shape == "s3" else 0)
    kernels = [{"repro": r, "shape": s, "model_share": w} for r, s, w in listed]
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"kernels": kernels}))
    argv = [
        "report.py",
        str(run),
        "--selection",
        str(selection),
        "--baseline",
        str(before),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    report.main()
    found = json.loads((run / "report.json").read_text())
    every, without = found["aggregates"]["cold_us"].values()
    # T5 20x, s2 1.5x (against inductor_cd), s3 0.8x.
    assert every["kernels"] == 3 and every["at_or_above_parity"] == 2
    assert every["cute_vs_inductor"] == pytest.approx(24 ** (1 / 3))
    assert every["time_weighted"] == pytest.approx(1130 / 160)
    assert every["model_weighted"] == pytest.approx(4 / (1 / 20 + 2 / 1.5 + 1 / 0.8))
    assert every["inductor_of_sol"] == pytest.approx((0.02 * 30 / 90 * 0.5) ** (1 / 3))
    assert every["cute_of_sol"] == pytest.approx(0.08 ** (1 / 3))
    # Triton counts on T5, s2 and s6 (2.5x, 0.75x, 1.5x), not s3.
    assert every["triton_vs_inductor"] == pytest.approx(2.8125 ** (1 / 3))
    assert every["triton_vs_inductor_kernels"] == 3
    assert every["cute_vs_triton"] == pytest.approx(4.0)
    assert every["cute_vs_oracle"] == pytest.approx(80 / 60)
    assert every["baseline"] == {
        "kernels": 1,
        "baseline": pytest.approx(0.5),
        "this_run": pytest.approx(1.5),
    }
    assert without["kernels"] == 2
    assert without["cute_vs_inductor"] == pytest.approx(1.2**0.5)
    assert without["time_weighted"] == pytest.approx(130 / 110)
    assert without["model_weighted"] == pytest.approx(3 / (2 / 1.5 + 1 / 0.8))
    families = found["families"]["cold_us"]
    assert {name: f["kernels"] for name, f in families.items()} == {
        "sum_sum_sum": 1,
        "sum": 1,
        "amax": 1,
    }
    assert found["reused"] == ["sum_000000000002__s2__triton"]
    assert found["unsampled"] == {
        "bench": ["amax_000000000005__s5"],
        "autotune": ["amax_000000000003__s3__triton"],
    }
    assert found["contaminated"]["autotune"] == {
        "sum_000000000006__s6__cute": [
            "eve:7 (40% SM, 3.0 s)",
            "the cleanest of 3 attempts",
        ]
    }
    text = (run / "report.md").read_text()
    assert f"| | all kernels | without {t5} |" in text
    assert "| Helion+CuTe vs Inductor, geomean | 2.884x | 1.095x |" in text
    assert (
        "| Helion+CuTe vs Inductor, run3 -> this run, same kernels "
        "| 0.500x -> 1.500x (1) | 0.500x -> 1.500x (1) |"
    ) in text
    assert "| sum_000000000002 | s2 | 0.0 | 90.00 | 60.00 | 1.500 | 0.500 |" in text
    assert "| bench contaminated |" in text
    assert "| triton autotune no activity samples |" in text
    assert "Autotune records reused from earlier runs (plan_run.py): 1." in text
    assert found["eager_reference"] == {"sum_000000000002__s2": "RuntimeError: x"}
    assert "- sum_000000000002__s2: RuntimeError: x\n" in text
    # Records made before the sampling count on request.
    monkeypatch.setattr(sys, "argv", [*argv, "--include-unsampled"])
    report.main()
    found = json.loads((run / "report.json").read_text())
    every = found["aggregates"]["cold_us"]["all kernels"]
    assert every["kernels"] == 4 and every["triton_vs_inductor_kernels"] == 5


def test_the_report_says_which_autotunes_searched_from_the_baselines_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, before, third = (tmp_path / name for name in ("after", "before", "third"))

    def tuned(
        where: Path, index: int, backend: str, seed: int | None, **fields: object
    ) -> str:
        repro = f"sum_00000000000{index}"
        env = {} if seed is None else {_SEED: str(seed)}
        _toy_tuned(where, repro, "s1", backend, env=env, **fields)
        return f"{repro}__s1__{backend}"

    _toy_bench(run, "sum_000000000001", "s1", {"inductor": 100.0, "helion_cute": 50.0})
    # The baseline's seeds; it has autotuned the fourth port again since
    # this run took its seed, and has no record of the sixth and seventh.
    for index, seed in enumerate((10, 20, 30, 41, 50), 1):
        tuned(before, index, "cute", seed)
    reused = {
        "reused_from": str(before / "configs" / "sum_000000000001__s1__triton.json")
    }
    tuned(before, 1, "triton", 11)
    tuned(before, 3, "triton", 31)
    paired = [
        # Given the baseline's seed, or the seed that both runs took from a
        # third: the records name one seed either way.
        tuned(run, 1, "cute", 10, seed_from=str(before)),
        tuned(run, 2, "cute", 20, seed_from=str(third)),
        tuned(run, 3, "triton", 31, seed_from=str(before)),
    ]
    unpaired = {
        tuned(run, 3, "cute", 99): "a seed of its own",
        tuned(run, 4, "cute", 40, seed_from=str(before)): (
            "a seed taken from before, which names another now"
        ),
        tuned(run, 5, "cute", 55, seed_from=str(third)): "a seed taken from third",
        tuned(run, 6, "cute", 60, seed_from=str(before)): (
            "a seed taken from before, which names none now (no autotune record)"
        ),
        # A search that was given no seed (autotune.py by hand) drew its own:
        # no seed on either side is not the same seed.
        tuned(run, 7, "cute", None): "a seed of its own",
    }
    # No searches of this run: a record reused from the baseline, which has
    # its seed, a provisional one, and a copy that --configs-from made of the
    # baseline's (until an autotune of this run replaces the copy).
    tuned(run, 1, "triton", 11, **reused)
    tuned(run, 2, "triton", 21, provisional=True, **reused)
    copy = run / "configs" / f"{tuned(run, 5, 'triton', 51)}.json"
    digests = [hashlib.sha256(copy.read_bytes()).hexdigest(), "an earlier copy's"]
    tuned(before, 5, "triton", 51)
    argv = ["report.py", str(run), "--selection", str(tmp_path / "none.json")]

    def report_on(*flags: str) -> tuple[dict[str, object], str]:
        monkeypatch.setattr(sys, "argv", [*argv, *flags])
        report.main()
        found = json.loads((run / "report.json").read_text())
        return found, (run / "report.md").read_text()

    def copied(**entry: str) -> None:
        manifest = {copy.name: {"source": "before", **entry}}
        (run / campaign.COPIED_CONFIGS).write_text(json.dumps(manifest))

    copied(sha256=digests[1])
    found, text = report_on("--baseline", str(before))
    assert found["seed_pairing"]["paired"] == sorted([*paired, copy.stem])
    assert "cute 2 of 7, triton 2 of 2." in text
    # The record is a copy by the digest of the copy that was made, and by
    # that of the copy it was to replace (the campaign died before it did).
    for entry in (
        {"sha256": digests[0]},
        {"sha256": digests[1], "replaces": digests[0]},
    ):
        copied(**entry)
        found, text = report_on("--baseline", str(before))
        assert found["seed_pairing"] == {
            "baseline": str(before),
            "paired": paired,
            "unpaired": unpaired,
        }
    assert (
        "Autotune seeds: 3 of the 8 autotunes this run made searched from the "
        "seed of before's autotune of the same port (`campaign.py --seeds-from`): "
        "cute 2 of 7, triton 1 of 1."
    ) in text
    own = [name for name, why in unpaired.items() if why == "a seed of its own"]
    assert f"\n- a seed of its own: 2 ({', '.join(own)})\n" in text
    # Paired by seed is not yet the same initial population: the note says so.
    same_space = "only where both trees build the same search space for a port"
    assert same_space in text
    for name, why in unpaired.items():
        assert name in own or f"\n- {why}: 1 ({name})\n" in text
    # Against a run whose seeds it shares nowhere (it has no records), and
    # without a baseline.
    found, text = report_on("--baseline", str(third))
    assert found["seed_pairing"]["paired"] == []
    assert (
        "Autotune seeds: none of the 8 autotunes this run made searched from the "
        "seed of third's autotune of the same port"
    ) in text
    assert (
        "Their seeds:\n\n- a seed of its own: 2\n"
        "- a seed taken from before: 4\n"
        "- a seed taken from third, which names none now (no autotune record): 2\n"
    ) in text
    assert same_space not in text
    found, text = report_on()
    assert "seed_pairing" not in found and "Autotune seeds" not in text
    # A run of copies alone (--configs-from) made no autotune.
    for path in (run / "configs").iterdir():
        if path != copy:
            path.unlink()
    found, text = report_on("--baseline", str(before))
    assert found["seed_pairing"]["paired"] == []
    assert found["seed_pairing"]["unpaired"] == {}
    assert "Autotune seeds: this run made no autotune of its own" in text


def _add_fields(run: Path, repro: str, shape: str, **fields: object) -> None:
    """Record-level fields for a toy bench record."""
    path = run / "bench" / f"{repro}__{shape}.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


def test_the_report_adds_rot_us_beside_cold_us_and_changes_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    skipped = "inputs of 4295.0 MB exceed 2 x the 132.6 MB L2"

    def bench_records(rotated: bool) -> None:
        """Five kernels; with ``rotated``, as ``bench.py --rotate-inputs 8``
        records them: three timed, one too large, and one bench that a
        foreign process contaminated."""

        def rot(**times: float) -> dict[str, dict[str, object]]:
            return (
                {impl: {"rot_us": us} for impl, us in times.items()} if rotated else {}
            )

        def decided(repro: str, shape: str, **decision: object) -> None:
            if rotated:
                _add_fields(run, repro, shape, rotate_inputs={"sets": 8, **decision})

        times = {"inductor": 12.0, "inductor_cd": 10.0, "helion_cute": 8.0}
        # Inductor's hinted loads read 2 us warm in cold_us, CuTe's 1 us.
        fields = rot(inductor=13.5, inductor_cd=12.0, helion_cute=9.0, memcopy=5.0)
        _toy_bench(run, "mean_000000000001", "s1", {**times, "memcopy": 5.0}, **fields)
        decided("mean_000000000001", "s1", status="ok")
        # CuTe's hints are the ones that kept its inputs resident here.
        fields = rot(inductor=20.5, helion_cute=25.0)
        _toy_bench(
            run,
            "var_000000000002",
            "s2",
            {"inductor": 20.0, "helion_cute": 16.0},
            **fields,
        )
        decided("var_000000000002", "s2", status="ok")
        _toy_bench(
            run, "sum_000000000003", "s3", {"inductor": 900.0, "helion_cute": 600.0}
        )
        decided("sum_000000000003", "s3", status="skipped", skipped=skipped)
        fields = rot(inductor=30.0, helion_cute=10.0)
        _toy_bench(
            run,
            "sum_000000000004",
            "s4",
            {"inductor": 30.0, "helion_cute": 15.0},
            70,
            **fields,
        )
        decided("sum_000000000004", "s4", status="ok")
        # Its rotating pass failed: the other timings stand.
        _toy_bench(
            run, "amax_000000000005", "s5", {"inductor": 40.0, "helion_cute": 50.0}
        )
        decided(
            "amax_000000000005",
            "s5",
            status="error",
            error="Traceback\nRuntimeError: x\n",
        )

    for index in range(1, 6):
        repro = ["mean", "var", "sum", "sum", "amax"][
            index - 1
        ] + f"_00000000000{index}"
        _toy_tuned(run, repro, f"s{index}", "cute")
    monkeypatch.setattr(sys, "argv", ["report.py", str(run)])
    bench_records(rotated=False)
    report.main()
    plain = json.loads((run / "report.json").read_text())
    plain_text = (run / "report.md").read_text()
    assert "rot" not in plain_text and "rot_us" not in json.dumps(plain)
    bench_records(rotated=True)
    report.main()
    found = json.loads((run / "report.json").read_text())
    text = (run / "report.md").read_text()
    # s1 1.25x by cold_us and 12/9 by rot_us, s2 1.25x and 0.82x; the
    # contaminated s4 counts in neither, s3 and s5 have no rot_us.
    check = found["aggregates"]["cold_us"]["all kernels"].pop("rot_us")
    assert check == {
        "kernels": 2,
        "cute_vs_inductor": pytest.approx((12 / 9 * 20.5 / 25) ** 0.5),
        "same_kernels_by_this_timer": pytest.approx(1.25),
    }
    assert found["families"]["cold_us"]["mean"].pop("rot_us")["kernels"] == 1
    for family in ("var", "sum", "amax"):
        kernels = found["families"]["cold_us"][family].pop("rot_us")["kernels"]
        assert kernels == (family == "var")
    rot_rows = {row["repro"]: row for row in found["rows"].pop("rot_us")}
    assert sorted(rot_rows) == [
        "mean_000000000001",
        "sum_000000000004",
        "var_000000000002",
    ]
    assert rot_rows["mean_000000000001"]["inductor"] == 12.0
    assert rot_rows["sum_000000000004"]["counted"] == []
    assert found.pop("rotate_inputs") == {
        "mean_000000000001__s1": {"sets": 8, "status": "ok"},
        "var_000000000002__s2": {"sets": 8, "status": "ok"},
        "sum_000000000003__s3": {"sets": 8, "status": "skipped", "skipped": skipped},
        "sum_000000000004__s4": {"sets": 8, "status": "ok"},
        "amax_000000000005__s5": {
            "sets": 8,
            "status": "error",
            "error": "Traceback\nRuntimeError: x\n",
        },
    }
    # Every other number is the one of the report without rot_us.
    assert found == plain
    assert (
        "| Helion+CuTe vs Inductor by rot_us, private inputs, geomean (kernels) "
        "| 1.046x (2) |\n"
        "| the same kernels by cold_us, geomean | 1.250x |\n"
    ) in text
    assert "| mean_000000000001 | s1 | 0.0 | 10.00 | 8.00 | 1.250 | 1.333 | - |" in text
    assert "| sum_000000000004 | s4 | 0.0 | 30.00 | 15.00 | 2.000 | 3.000 | - |" in text
    assert "| sum_000000000003 | s3 | 0.0 | 900.00 | 600.00 | 1.500 | - | - |" in text
    assert (
        "## Rotating inputs (rot_us)\n\n"
        "`bench.py --rotate-inputs` timed 3 of 5 records on private input sets "
        "(8 per implementation), one per call of a flushed batch; not timed:\n\n"
        "- amax_000000000005__s5: failed: RuntimeError: x\n"
        f"- sum_000000000003__s3: {skipped}\n"
        f"\n{report.ROTATION_CAVEAT}\n"
    ) in text
    assert "bracket the hint-free one only where the CuTe config has no such hints" in (
        report.ROTATION_CAVEAT
    )
    # Without the rot_us rows, column and section it reads as before.
    start = text.index("## Rotating inputs")
    kept = []
    for line in text[:start].splitlines():
        cells = line.split(" | ")
        if "rot_us" in line or "the same kernels by cold_us" in line:
            continue
        if line.startswith("| ") and len(cells) == 15:
            del cells[6]
        kept.append(" | ".join(cells))
    kept = "\n".join(kept).replace(f"|{'---|' * 15}", f"|{'---|' * 14}", 1) + "\n"
    assert kept == plain_text


def test_the_goal_file_takes_only_the_ratios_the_aggregates_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The goal file takes only the ratios the aggregates count, so
    # check_goal.py cannot pass a goal on timings the report leaves out.
    run = tmp_path / "run"
    times = {"inductor": 100.0, "helion_cute": 50.0}
    # A foreign process's SM % during each point's bench and CuTe autotune
    # (None: no activity samples).
    points = {"clean": (0, 0), "bench": (70, 0), "unsampled": (None, 0)}
    points["autotune"] = (0, 40)
    for shape, (bench_sm, autotune_sm) in points.items():
        _toy_bench(run, "sum_000000000001", shape, times, bench_sm)
        _toy_tuned(run, "sum_000000000001", shape, "cute", autotune_sm)
    goal = tmp_path / "goal.json"

    def ratios(*flags: str) -> dict[str, object]:
        argv = ["report.py", str(run), "--goal", str(goal), *flags]
        monkeypatch.setattr(sys, "argv", argv)
        report.main()
        shapes = json.loads(goal.read_text())["shapes"]
        return {shape["name"]: shape["ratio"] for shape in shapes}

    assert ratios() == {
        "sum_000000000001[clean]": 2.0,
        "sum_000000000001[bench]": None,
        "sum_000000000001[unsampled]": None,
        "sum_000000000001[autotune]": None,
    }
    # Records made before the sampling count on request.
    assert ratios("--include-unsampled")["sum_000000000001[unsampled]"] == 2.0


def test_the_unsampled_flags_admit_only_records_made_without_the_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --include-unsampled (and the baseline comparison, and --reuse-unsampled)
    # admit a record made before the sampling existed. One whose samples miss
    # its job stays out, even where they saw a foreign process compute, and so
    # does one the campaign kept flagged contaminated.
    run = tmp_path / "run"
    times = {"inductor": 100.0, "helion_cute": 50.0}
    uncovered = {
        "interval_s": 1.0,
        "samples": 100,
        "unknown_samples": 20,
        "max_gap_s": 2.0,
        "processes": {"eve:7": {"max_sm": 90, "active_s": 60.0, "busy_s": 60.0}},
    }
    unread = {**uncovered, "unknown_samples": 100, "processes": {}}
    kept = {"contaminated": True, "contaminated_attempts": ["a1", "a2", "a3"]}
    for shape, activity, fields in (
        ("clean", _toy_activity(0), {}),
        ("old", None, {}),
        ("uncovered", uncovered, {}),
        ("kept", unread, kept),
    ):
        _toy_bench(run, "sum_000000000001", shape, times, None)
        path = run / "bench" / f"sum_000000000001__{shape}.json"
        record = json.loads(path.read_text())
        if activity is not None:
            record["foreign_gpu_activity"] = activity
        path.write_text(json.dumps({**record, **fields}))
        _toy_tuned(run, "sum_000000000001", shape, "cute")
    monkeypatch.setattr(sys, "argv", ["report.py", str(run), "--include-unsampled"])
    report.main()
    found = json.loads((run / "report.json").read_text())
    counted = {
        row["shape"]: "helion_cute" in row["counted"]
        for row in found["rows"]["cold_us"]
    }
    assert counted == {"clean": True, "old": True, "uncovered": False, "kept": False}
    assert found["unsampled"]["bench"] == ["sum_000000000001__old"]
    assert found["uncovered"]["bench"] == {
        "sum_000000000001__uncovered": "20 of 100 samples failed"
    }
    assert found["contaminated"]["bench"] == {
        "sum_000000000001__kept": ["the cleanest of 3 attempts"]
    }
    assert set(report.baseline_speedups(run, "cold_us")) == {
        ("sum_000000000001", "clean"),
        ("sum_000000000001", "old"),
    }
    record = json.loads(
        (run / "bench" / "sum_000000000001__uncovered.json").read_text()
    )
    reason = plan_run.triton_reuse(
        tmp_path / "x.json", record, reuse_unsampled=True, pin_check=False
    )
    assert reason == "20 of 100 samples failed"


def test_a_failing_inductor_baseline_leaves_the_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bench.py gates the default Inductor compile too (against itself it
    # fails only an output not finite where the reference is): when it
    # failed, the coordinate-descent compile stands for Inductor, and the
    # numerics column says so.
    assert "inductor" in bench.GATED_IMPLS
    run = tmp_path / "run"
    times = {"inductor": 100.0, "inductor_cd": 90.0, "helion_cute": 50.0}
    failed = {"numerics_pass": False}
    _toy_bench(run, "sum_000000000001", "s1", times, inductor=failed)
    _toy_tuned(run, "sum_000000000001", "s1", "cute")
    monkeypatch.setattr(sys, "argv", ["report.py", str(run)])
    report.main()
    (row,) = json.loads((run / "report.json").read_text())["rows"]["cold_us"]
    assert row["inductor"] == 90.0 and row["numerics"]["inductor"] == "FAIL"


@pytest.mark.slow
@_requires_proc
def test_a_two_phase_run_benches_on_provisional_triton_configs_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selection, run5, _ = _plan_main(tmp_path, monkeypatch, "--provisional-triton")
    configs = tmp_path / "run" / "configs"
    out = capsys.readouterr().out
    assert "provisional Triton configs for the benches before their autotunes: 4" in out
    assert "in two phases: 4.4 (campaign.py --backends cute" in out
    assert "then 0.54 (--backends triton" in out and "total: 4.94 GPU-hours" in out
    # Points 2-5 have earlier records to stand in; 1 is reused, 6 has none.
    name = "sum_000000000002__s1__triton.json"
    copied = json.loads((configs / name).read_text())
    assert copied["provisional"] is True
    assert copied["reused_from"] == str(run5 / "configs" / name)
    kernels = selection["kernels"]
    _, events = _stub_campaign(
        tmp_path, monkeypatch, {}, kernels=kernels, backends="cute"
    )
    started = [event["job"] for event in events if event["kind"] == "job_start"]
    assert len(started) == 12 and all(":" in key for key in started)
    assert not any(key.endswith("__triton") for key in started)
    assert json.loads((configs / name).read_text())["provisional"] is True
    _, events = _stub_campaign(
        tmp_path, monkeypatch, {}, kernels=kernels, backends="triton"
    )
    started = sorted(event["job"] for event in events if event["kind"] == "job_start")
    points = [f"sum_00000000000{i}__s1" for i in range(2, 7)]
    assert started == sorted(
        [f"autotune:{p}__triton" for p in points] + [f"bench:{p}" for p in points]
    )
    assert not any(
        json.loads(path.read_text()).get("provisional")
        for path in configs.glob("*.json")
    )
    assert json.loads((configs / f"{name}.prev").read_text())["provisional"] is True


@pytest.mark.slow
@_requires_proc
def test_a_restart_resumes_the_contaminated_attempts_of_a_stopped_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "sum_000000000001__s1__triton.json"
    configs = tmp_path / "run" / "configs"
    configs.mkdir(parents=True)
    # An attempt of the job's previous run (whose record is now .prev), then
    # one of this run, set aside before the campaign stopped.
    record = {"repro": "sum_000000000001", "shape": "s1", "status": "ok"}
    stale, prev, current = (
        configs / f"{name}.attempt1",
        configs / f"{name}.prev",
        configs / f"{name}.attempt2",
    )
    for age, path in enumerate((stale, prev, current)):
        busy = {"processes": {"eve:7": {"max_sm": 90, "active_s": 5.0}}}
        activity = {**_toy_activity(90), **busy}
        path.write_text(json.dumps({**record, "foreign_gpu_activity": activity}))
        os.utime(path, ns=(age * 10**9, age * 10**9))
    # One retry allowed, which the stopped run used: the resumed attempt is
    # the last, so the cleaner of the two stands.
    run, events = _stub_campaign(
        tmp_path, monkeypatch, {name: [83]}, "--retry-contaminated", "1"
    )
    kept = json.loads((configs / name).read_text())
    assert kept["contaminated"] is True
    assert kept["contaminated_attempts"] == [
        str(current),
        str(configs / f"{name}.attempt3"),
    ]
    assert common.foreign_activity(kept) == ["eve:7 (83% SM, 1.0 s)"]
    actions = [e["action"] for e in events if e["kind"] == "job_contaminated"]
    assert actions == ["kept_cleanest"]


@pytest.mark.slow
@_requires_proc
def test_a_restart_settles_a_contaminated_record_left_unsettled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The campaign stopped between the job's end and its settling.
    name = "sum_000000000001__s1__triton.json"
    configs = tmp_path / "run" / "configs"
    configs.mkdir(parents=True)
    record = {"repro": "sum_000000000001", "shape": "s1", "status": "ok"}
    record["foreign_gpu_activity"] = _toy_activity(83)
    (configs / name).write_text(json.dumps(record))
    run, events = _stub_campaign(tmp_path, monkeypatch, {name: [0]})
    final = json.loads((configs / name).read_text())
    assert final["contaminated_attempts"] == [str(configs / f"{name}.attempt1")]
    settled = [e for e in events if e["kind"] == "job_contaminated"]
    assert [(e["action"], e["gpu"]) for e in settled] == [("retry", None)]
    started = [e["job"] for e in events if e["kind"] == "job_start"]
    assert started == [f"autotune:{name[:-5]}", "bench:sum_000000000001__s1"]


@pytest.mark.slow
@_requires_proc
def test_a_restart_settles_the_last_allowed_attempt_and_a_clean_ones_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The campaign stopped between a job's end and its settling, after two
    # contaminated attempts were set aside, so the third was the last the
    # two retries allow. Contaminated, it is kept flagged; clean, its record
    # lists the attempts before it.
    name = "sum_000000000001__s1__triton.json"
    record = {"repro": "sum_000000000001", "shape": "s1", "status": "ok"}

    def restart(
        where: Path, last: tuple[int, float]
    ) -> tuple[list[str], dict[str, object], list[dict[str, object]]]:
        configs = where / "run" / "configs"
        configs.mkdir(parents=True)
        attempts = [str(configs / f"{name}.attempt{k}") for k in (1, 2, 3)]
        for path, (sm, seconds) in zip(
            [
                configs / f"{name}.attempt1",
                configs / f"{name}.attempt2",
                configs / name,
            ],
            [(90, 5.0), (40, 1.0), last],
            strict=True,
        ):
            activity = _toy_activity(sm)
            activity["processes"] = {"eve:7": {"max_sm": sm, "active_s": seconds}}
            path.write_text(json.dumps({**record, "foreign_gpu_activity": activity}))
        _, events = _stub_campaign(where, monkeypatch, {})
        started = [e["job"] for e in events if e["kind"] == "job_start"]
        assert started == ["bench:sum_000000000001__s1"]
        return attempts, json.loads((configs / name).read_text()), events

    attempts, kept, events = restart(tmp_path / "contaminated", (83, 3.0))
    assert kept["contaminated"] is True and kept["contaminated_attempts"] == attempts
    # Attempt 2: 40% SM for 1 s ranks below 90% for 5 s and 83% for 3 s.
    assert common.foreign_activity(kept) == ["eve:7 (40% SM, 1.0 s)"]
    settled = [
        (e["action"], e["gpu"]) for e in events if e["kind"] == "job_contaminated"
    ]
    assert settled == [("kept_cleanest", None)]
    attempts, kept, _ = restart(tmp_path / "clean", (0, 3.0))
    assert "contaminated" not in kept and kept["contaminated_attempts"] == attempts[:2]


@pytest.mark.slow
@_requires_proc
def test_a_restart_keeps_the_cleanest_attempt_once_the_retries_are_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The campaign stopped after setting the last allowed attempt aside but
    # before keeping the cleanest, or restarts with fewer retries: the
    # cleanest is kept, and the job does not run again.
    name = "sum_000000000001__s1__triton.json"
    record = {"repro": "sum_000000000001", "shape": "s1", "status": "ok"}
    for where, activity, flags in (
        ("stopped", [(90, 5.0), (40, 1.0), (83, 3.0)], []),
        ("fewer", [(90, 5.0), (40, 1.0)], ["--retry-contaminated", "1"]),
    ):
        configs = tmp_path / where / "run" / "configs"
        configs.mkdir(parents=True)
        attempts = [configs / f"{name}.attempt{k}" for k in (1, 2, 3)]
        attempts = attempts[: len(activity)]
        for path, (sm, seconds) in zip(attempts, activity, strict=True):
            sampled = _toy_activity(sm)
            sampled["processes"] = {"eve:7": {"max_sm": sm, "active_s": seconds}}
            path.write_text(json.dumps({**record, "foreign_gpu_activity": sampled}))
        # The campaign that runs with fewer retries now had been stopped
        # while the job ran a third time: what that start left of itself,
        # its log and search logs, is no file of the attempt that is kept,
        # and goes aside as the next attempt's. The one that stopped before
        # keeping the cleanest had started the job no further time, so
        # nothing is set aside for it.
        left = "a start that was stopped\n" if where == "fewer" else None
        logs = tmp_path / where / "run" / "logs"
        log = logs / "autotune" / f"{Path(name).stem}.log"
        search = logs / "search" / Path(name).stem / "search.csv"
        if left is not None:
            for path in (log, search):
                path.parent.mkdir(parents=True)
                path.write_text(left)
        _, events = _stub_campaign(tmp_path / where, monkeypatch, {}, *flags)
        aside = f".attempt{len(attempts) + 1}"
        assert not log.exists() and not search.parent.exists()
        set_aside = (
            log.with_name(log.name + aside),
            search.parent.with_name(search.parent.name + aside) / "search.csv",
        )
        found = [p.read_text() if p.exists() else None for p in set_aside]
        assert found == [left, left]
        started = [e["job"] for e in events if e["kind"] == "job_start"]
        assert started == ["bench:sum_000000000001__s1"]
        kept = json.loads((configs / name).read_text())
        assert kept["contaminated"] is True
        assert kept["contaminated_attempts"] == [str(path) for path in attempts]
        assert common.foreign_activity(kept) == ["eve:7 (40% SM, 1.0 s)"]
        settled = [
            (e["action"], e["gpu"]) for e in events if e["kind"] == "job_contaminated"
        ]
        assert settled == [("kept_cleanest", None)]
        # A campaign stopped between the two steps repeats them when it
        # starts again, and what it set aside the first time stays (the
        # bench runs again for the record that was kept anew).
        (configs / name).unlink()
        _, events = _stub_campaign(tmp_path / where, monkeypatch, {}, *flags)
        assert [e["job"] for e in events if e["kind"] == "job_start"] == started
        assert json.loads((configs / name).read_text())["contaminated"] is True
        found = [p.read_text() if p.exists() else None for p in set_aside]
        assert found == [left, left]


def _stop_once_moved(monkeypatch: pytest.MonkeyPatch, moved: Path) -> None:
    """Have ``Path.replace`` send this process SIGTERM right after it has
    moved the file ``moved``, while ``monkeypatch`` holds."""
    replace = pathlib.Path.replace

    def stopped_after(path: Path, target: Path) -> Path:
        result = replace(path, target)
        if path == moved:
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(pathlib.Path, "replace", stopped_after)


def test_a_stop_signal_at_startup_waits_for_a_jobs_files_as_well(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stopped campaign left a contaminated bench record unsettled, with
    # its log. The next campaign sets both aside while it queues its jobs,
    # and is told to stop when the log has moved and the record has not.
    kernels = [{"repro": "sum_000000000001", "shape": "s1"}]
    run, name = tmp_path / "run", "sum_000000000001__s1"
    _toy_tuned(run, "sum_000000000001", "s1", "triton")
    times = {"inductor": 2.0, "helion_triton": 1.0}
    _toy_bench(run, "sum_000000000001", "s1", times, max_sm=83)
    record = run / "bench" / f"{name}.json"
    log = run / "logs" / "bench" / f"{name}.log"
    log.parent.mkdir(parents=True)
    log.write_text("the contaminated bench\n")
    # A signal that the campaign does not handle yet reaches this handler.
    unhandled: list[int] = []
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: unhandled.append(1))
    monkeypatch.setattr(campaign, "_deferred_stop", None)
    try:
        with monkeypatch.context() as patched:
            _stop_once_moved(patched, log)
            with pytest.raises(campaign.Interrupted):
                _stub_campaign(
                    tmp_path, patched, {}, "--phases", "bench", kernels=kernels
                )
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert unhandled == []
    # The campaign stopped before it had a job to start, with both files of
    # the attempt set aside.
    assert "campaign_start" not in (run / "campaign.jsonl").read_text()
    assert not record.exists() and not log.exists()
    assert record.with_name(f"{record.name}.attempt1").exists()
    kept_log = log.with_name(f"{log.name}.attempt1")
    assert kept_log.read_text() == "the contaminated bench\n"


def test_a_stop_signal_waits_until_a_jobs_files_have_moved_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A finished autotune with its log and search logs. A campaign runs it
    # again and is told to stop just as it has moved the log aside.
    kernels = [{"repro": "sum_000000000001", "shape": "s1"}]
    run, name = tmp_path / "run", "sum_000000000001__s1__triton"
    _toy_tuned(run, "sum_000000000001", "s1", "triton")
    log = run / "logs" / "autotune" / f"{name}.log"
    search = run / "logs" / "search" / name / "search.csv"
    for path in (log, search):
        path.parent.mkdir(parents=True)
        path.write_text("the first search\n")
    monkeypatch.setattr(campaign, "_deferred_stop", None)
    with monkeypatch.context() as patched:
        _stop_once_moved(patched, log)
        with pytest.raises(campaign.Interrupted):
            _stub_campaign(
                tmp_path,
                patched,
                {},
                "--retune",
                "--phases",
                "autotune",
                kernels=kernels,
            )
    # It stopped before the job started, and with the record and the search
    # logs moved as well: the three are still one start's.
    events = (run / "campaign.jsonl").read_text()
    assert "job_start" not in events and "campaign_end" in events
    kept = [
        run / "configs" / f"{name}.json.prev",
        log.with_name(f"{log.name}.prev"),
        search.parent.with_name(f"{name}.prev") / "search.csv",
    ]
    assert all(path.exists() for path in kept)
    assert not (run / "configs" / f"{name}.json").exists()
    assert not log.exists() and not search.parent.exists()
    # The campaign that starts next leaves them as they are.
    monkeypatch.setattr(campaign, "_deferred_stop", None)
    _stopped_campaign(tmp_path, monkeypatch, kernels, "triton", "--phases", "autotune")
    assert [path.read_text() for path in kept[1:]] == ["the first search\n"] * 2


def test_ports_timed_on_provisional_configs_show_but_do_not_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    times = {"inductor": 100.0, "helion_cute": 50.0, "helion_triton": 80.0}
    # The provisional config generates other code than when it was tuned.
    _toy_bench(
        run,
        "sum_000000000001",
        "s1",
        times,
        helion_triton={"code_matches_autotune": False},
    )
    _toy_tuned(run, "sum_000000000001", "s1", "cute")
    _toy_tuned(
        run,
        "sum_000000000001",
        "s1",
        "triton",
        None,
        provisional=True,
        reused_from="run5/x.json",
    )
    monkeypatch.setattr(sys, "argv", ["report.py", str(run)])
    report.main()
    found = json.loads((run / "report.json").read_text())
    row = found["rows"]["cold_us"][0]
    assert row["helion_triton"] == 80.0 and row["triton_vs_inductor"] == 1.25
    assert row["issues"] == {"triton autotune": "provisional: run5/x.json"}
    every = found["aggregates"]["cold_us"]["all kernels"]
    assert every["kernels"] == 1 and every["triton_vs_inductor_kernels"] == 0
    assert found["provisional"] == ["sum_000000000001__s1__triton"]
    assert found["unsampled"] == {"bench": [], "autotune": []}
    text = (run / "report.md").read_text()
    assert "Provisional autotune records: 1." in text
    assert "| triton autotune provisional |" in text


def test_a_port_counts_only_while_its_autotune_record_holds_the_timed_kernels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An autotune that runs after the bench moves the point's record aside
    # and then replaces it, so a port counts only while the record holds
    # the kernels the bench timed: neither a timing on a provisional config
    # nor an old timing counts for a new record.
    run = tmp_path / "run"
    provisional = {"autotune_provisional": True, "autotune_reused_from": "run5/x.json"}
    # A bench record made before the digests names the pinned decorators.
    named = {
        "autotune_code_sha256": None,
        "decorators": {"k": "@helion.kernel(version=1)"},
    }
    points: list[tuple[dict[str, object], int | None]] = [
        # How the bench timed Helion-Triton, and the version of the record
        # that stands now (None: no ok record).
        (provisional, None),  # its autotune runs, the record set aside
        (provisional, 2),  # its autotune has replaced the record
        ({}, 2),  # autotuned again after the bench
        ({}, None),  # autotuning again, or that failed
        ({}, 1),
        (named, 1),
        (named, 2),
    ]
    times = {"inductor": 100.0, "helion_cute": 50.0, "helion_triton": 80.0}
    for i, (timed, version) in enumerate(points, 1):
        repro = f"sum_00000000000{i}"
        _toy_bench(run, repro, "s1", times, helion_triton=timed)
        _toy_tuned(run, repro, "s1", "cute")
        if version is not None:
            kernels = _toy_kernels("triton", version)
            _toy_tuned(run, repro, "s1", "triton", kernels=kernels)
    monkeypatch.setattr(sys, "argv", ["report.py", str(run)])
    report.main()
    found = json.loads((run / "report.json").read_text())
    rows = sorted(found["rows"]["cold_us"], key=operator.itemgetter("repro"))
    replaced = "replaced: autotuned again after the bench"
    assert [row["issues"].get("triton autotune") for row in rows] == [
        "provisional: run5/x.json",
        "provisional: run5/x.json",
        replaced,
        "missing: no ok autotune record",
        None,
        None,
        replaced,
    ]
    # Each shows, flagged where it does not count.
    assert all(row["helion_triton"] == 80.0 for row in rows)
    every = found["aggregates"]["cold_us"]["all kernels"]
    assert every["kernels"] == 7 and every["triton_vs_inductor_kernels"] == 2
