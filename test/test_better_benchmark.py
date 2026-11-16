from __future__ import annotations

import contextlib
import gc
import itertools
import json
import math
import operator
import os
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
import helion.runtime

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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
# writes its pid there.
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
    "foreign_gpu_activity": {
        "interval_s": 1.0,
        "samples": 9,
        "unknown_samples": 0,
        "processes": {"eve:7": {"max_sm": sm, "active_s": float(attempt)}},
    },
}
if sm < 0:
    # The sampling failed throughout: no sample read the GPU.
    record["foreign_gpu_activity"].update(unknown_samples=9, processes={})
if "--backend" in args:
    record["backend"] = args[args.index("--backend") + 1]
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
        _, events = _stub_campaign(tmp_path / where, monkeypatch, {}, *flags)
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
