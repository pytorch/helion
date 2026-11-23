from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import operator
import os
import pathlib
from pathlib import Path
import re
import runpy
import secrets
import signal
import statistics
import subprocess
import sys
import threading
import time
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import warnings
import weakref

from benchmarks.cute.better_benchmark import audit_activity
from benchmarks.cute.better_benchmark import autotune
from benchmarks.cute.better_benchmark import bench
from benchmarks.cute.better_benchmark import campaign
from benchmarks.cute.better_benchmark import check_port
from benchmarks.cute.better_benchmark import common
from benchmarks.cute.better_benchmark import inductor_code
from benchmarks.cute.better_benchmark import inductor_pretuned
from benchmarks.cute.better_benchmark import ncu_profile
from benchmarks.cute.better_benchmark import plan_run
from benchmarks.cute.better_benchmark import report
from benchmarks.cute.better_benchmark import select_kernels
import pytest
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

import helion
from helion._compat import requires_torch_version
from helion._testing import DEVICE
from helion._testing import _has_cute_dsl
from helion._testing import cpu_render_target
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEagerMark
from helion._testing.cute_codegen import codegen_skip_reason
from helion.autotuner import benchmark_prelude
from helion.autotuner import benchmarking
import helion.runtime
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing_extensions import Self

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


def test_an_exact_output_inductor_does_not_reproduce_must_equal_inductors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A sort permutation and a mask that eager's implementation and
    # Inductor's kernels settle differently (the order of topk's tied values
    # in pointwise_59acdb823e92), beside a float output.
    monkeypatch.setattr(common, "flatten_outputs", list)
    eager = [torch.tensor([0, 1, 2, 3]), torch.tensor([True, False]), torch.ones(2)]
    inductor = [torch.tensor([0, 2, 1, 3]), torch.tensor([False, True]), torch.ones(2)]
    asked: list[object] = []

    def checked(*outputs: torch.Tensor) -> list[dict[str, object]]:
        judged: list[dict[str, object]] = []
        common.check_outputs(
            list(outputs),
            eager,
            inductor=lambda: asked.append(outputs) or inductor,
            vs_inductor=judged,
        )
        return judged

    def entry(index: int, elements: int) -> dict[str, object]:
        return {
            "idx": index,
            "reference": "inductor",
            "elements": elements,
            "mismatches_vs_eager": 2,
            "inductor_mismatches_vs_eager": 2,
        }

    # Eager's values pass as before, and nothing asks for Inductor's.
    assert checked(*eager) == [] and not asked
    # Inductor's pass too, and the entries say why they were the reference;
    # an output that equals eager's is compared with eager only.
    assert checked(*inductor) == [entry(0, 4), entry(1, 2)]
    assert checked(inductor[0], eager[1], eager[2]) == [entry(0, 4)]
    # Values that are neither's fail, whatever they have in common with
    # Inductor's.
    with pytest.raises(
        common.ExactMismatch,
        match="output 0: 2 of 4 elements differ from eager's and 3 from the "
        "default Inductor compile's, which differs from eager's in 2",
    ):
        checked(torch.tensor([1, 0, 2, 3]), *eager[1:])
    # Inductor's float output is no reference: only the gate judges floats.
    wrong = torch.full([2], 3.0)
    judged: list[dict[str, object]] = []
    with pytest.raises(common.FloatMismatch):
        common.check_outputs(
            [inductor[0], eager[1], wrong],
            eager,
            inductor=lambda: [*inductor[:2], wrong],
            vs_inductor=judged,
        )
    assert judged == [entry(0, 4)]
    # Without Inductor's outputs eager is the only reference (check_port.py
    # --no-anchored), as it is for an output in which Inductor reproduces
    # eager and for one where Inductor's has another structure.
    as_today = r"output 0: 2 of 4 elements differ$"
    with pytest.raises(common.ExactMismatch, match=as_today):
        common.check_outputs(inductor, eager)
    for reference in (eager, [inductor[0].view(2, 2), *eager[1:]], inductor[:2]):
        with pytest.raises(common.ExactMismatch, match=as_today):
            common.check_outputs(inductor, eager, inductor=lambda r=reference: r)


def test_inductor_compiles_once_when_its_outputs_are_first_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiles: list[object] = []

    def compile_inductor(
        repro_id: str, shape: str, instance: object, *, impl: str = "inductor"
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        compiles.append((repro_id, shape, instance, impl))
        return lambda x: x + 1

    monkeypatch.setattr(common, "compile_inductor", compile_inductor)
    monkeypatch.setattr(common, "make_inputs", lambda *args, **kwargs: [torch.ones(3)])
    outputs = common.inductor_outputs("r", "s", "instance")
    assert not compiles
    first = outputs()
    assert torch.equal(first, torch.full([3], 2.0)) and outputs() is first
    assert compiles == [("r", "s", "instance", "inductor")]


# pointwise_59acdb823e92 in miniature: a float output and a sort permutation
# in which eager orders two tied values one way and Inductor's kernels the
# other, with the entry a check records for an implementation it judged
# against Inductor's.
_EAGER_ORDER = [0, 1, 2, 3]
_INDUCTOR_ORDER = [0, 2, 1, 3]
_NEITHER_ORDER = [1, 0, 2, 3]
_VS_INDUCTOR = {
    "idx": 1,
    "reference": "inductor",
    "elements": 4,
    "mismatches_vs_eager": 2,
    "inductor_mismatches_vs_eager": 2,
}
_NEITHER = (
    "ExactMismatch: output 1: 2 of 4 elements differ from eager's and 3 from "
    "the default Inductor compile's, which differs from eager's in 2"
)


def _ordering(order: list[int]) -> types.SimpleNamespace:
    """A port of the stand-in repro whose permutation is ``order``."""
    return types.SimpleNamespace(
        forward=lambda inputs: (inputs[0] * 2, torch.tensor(order))
    )


@pytest.fixture
def tie_order_repro(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand-ins for that repro, its inputs and its Inductor compiles on a
    machine without a device or better-benchmark; returns the ``impl`` of
    each compile made."""

    class Repro(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return _ordering(_EAGER_ORDER).forward([x])

    compiles: list[str] = []

    def compile_inductor(
        repro_id: str, shape: str, instance: object, *, impl: str = "inductor"
    ) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        compiles.append(impl)
        return lambda x: _ordering(_INDUCTOR_ORDER).forward([x])

    module = types.SimpleNamespace(Repro=Repro)
    harness = types.SimpleNamespace(detect_stochastic_outputs=lambda *args: [])
    monkeypatch.setitem(sys.modules, "oracle_harness", harness)
    monkeypatch.setattr(common, "bb_path", lambda: None)
    monkeypatch.setattr(common, "load_repro_module", lambda repro_id: module)
    monkeypatch.setattr(common, "make_inputs", lambda *args, **kwargs: [torch.ones(3)])
    monkeypatch.setattr(common, "flatten_outputs", list)
    monkeypatch.setattr(common, "compile_inductor", compile_inductor)
    monkeypatch.setattr(common, "gpu_info", dict)
    monkeypatch.setattr(common, "versions", lambda triton=None: {})
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return compiles


def test_check_port_reports_the_outputs_it_judged_against_inductor(
    tie_order_repro: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def check(order: list[int], *flags: str) -> tuple[object, str]:
        monkeypatch.setattr(common, "load_port", lambda *args: _ordering(order))
        argv = ["check_port.py", "r", "--shape", "s", "--backends", "cute", *flags]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exit_:
            check_port.main()
        return exit_.value.code, capsys.readouterr().out

    # A port that returns Inductor's permutation passes, and the check says
    # which output it compared with Inductor's and why.
    code, out = check(_INDUCTOR_ORDER)
    assert code == 0 and out.endswith("PASS\n")
    assert (
        "[cute]   exact output vs Inductor, whose own differs from eager's: "
        f"{_VS_INDUCTOR}\n"
    ) in out
    # One that returns eager's passes as before.
    code, out = check(_EAGER_ORDER)
    assert code == 0 and out.endswith("PASS\n") and "vs Inductor" not in out
    # One that returns neither fails.
    code, out = check(_NEITHER_ORDER)
    assert code == 1 and out.endswith("FAIL\n") and _NEITHER in out
    # Without the gate nothing is compiled, and eager is the only reference.
    tie_order_repro.clear()
    code, out = check(_INDUCTOR_ORDER, "--no-anchored")
    assert code == 1 and "ExactMismatch: output 1: 2 of 4 elements differ\n" in out
    assert not tie_order_repro


def test_an_autotune_record_names_the_outputs_it_judged_against_inductor(
    tie_order_repro: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = _toy_kernels("cute")
    monkeypatch.setattr(autotune, "register_post_autotune_hook", lambda hook: None)
    monkeypatch.setattr(common, "bound_kernel_records", lambda port: kernels)
    monkeypatch.setattr(common, "pin_roundtrip", lambda port, kernels: {"k": True})

    def tuned(order: list[int]) -> dict[str, object]:
        monkeypatch.setattr(
            common, "load_port", lambda *args, **kwargs: _ordering(order)
        )
        tie_order_repro.clear()
        job = argparse.Namespace(
            repro_id="r", shape="s", backend="cute", rtol=2e-2, atol=2e-2, regime=None
        )
        return autotune.run(job)

    # The tuned run and the pinned one are both judged against the one
    # default compile, which only a port that differs from eager needs.
    record = tuned(_INDUCTOR_ORDER)
    assert record["status"] == "ok" and record["exact_vs_inductor"] == [_VS_INDUCTOR]
    assert record["errors_vs_eager"][1]["mismatches"] == 2  # pyrefly: ignore[bad-index]
    assert tie_order_repro == ["inductor"]
    record = tuned(_EAGER_ORDER)
    assert record["status"] == "ok" and "exact_vs_inductor" not in record
    assert not tie_order_repro
    record = tuned(_NEITHER_ORDER)
    assert record["status"] == "error" and _NEITHER in str(record["error"])
    assert "kernels" not in record and "exact_vs_inductor" not in record


def test_a_bench_record_names_the_outputs_it_judged_against_inductor(
    tie_order_repro: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default compile is its own reference, the coordinate-descent
    # compile and a port of Inductor's kernels reproduce it, an oracle that
    # calls ATen reproduces eager, and a port that reproduces neither is an
    # error that is not timed.
    ports = {"triton": _INDUCTOR_ORDER, "cute": _NEITHER_ORDER}
    accounting = types.SimpleNamespace(count_bytes_effective=lambda *args: 24)
    monkeypatch.setitem(sys.modules, "byte_accounting", accounting)
    monkeypatch.setattr(
        common, "load_oracle", lambda *args: _ordering(_EAGER_ORDER).forward
    )
    monkeypatch.setattr(
        bench,
        "pinned_port",
        lambda info, configs, repro, shape, backend: (
            _ordering(ports[backend]),
            {"kernels": {}},
        ),
    )
    monkeypatch.setattr(common, "pin_roundtrip", lambda port, kernels: {})
    monkeypatch.setattr(bench, "_replayed", lambda call, fresh_inputs: call)
    monkeypatch.setattr(common, "foreign_gpu_processes", list)
    monkeypatch.setattr(common, "gpu_temperature", lambda: None)
    timed: list[str] = []

    def time_impls(
        fns: dict[str, Callable[[], object]], **rounds: int
    ) -> dict[str, dict[str, float]]:
        timed.extend(fns)
        return {impl: {"bb_us": 2.0, "cold_us": 1.0} for impl in fns}

    monkeypatch.setattr(bench, "time_impls", time_impls)
    impls = ["inductor", "inductor_cd", "oracle", "helion_triton", "helion_cute"]
    record = bench.run(
        argparse.Namespace(
            repro_id="r",
            shape="s",
            impls=",".join(impls),
            configs_dir=tmp_path,
            rounds=2,
            rep_ms=1,
            cold_rounds=2,
            cold_timer="events",
            rotate_inputs=0,
            placement_seed=None,
            fixed_placement=True,
        ),
        None,
    )
    assert record["status"] == "ok" and timed == impls[:4]
    entries = record["impls"]
    for impl in ("inductor", "inductor_cd", "helion_triton"):
        assert entries[impl]["status"] == "ok" and entries[impl]["numerics_pass"]
        assert entries[impl]["exact_vs_inductor"] == [_VS_INDUCTOR]
        assert entries[impl]["errors_vs_eager"][1]["mismatches"] == 2
    assert entries["oracle"]["status"] == "ok" and entries["oracle"]["numerics_pass"]
    assert "exact_vs_inductor" not in entries["oracle"]
    assert entries["helion_cute"]["status"] == "error"
    assert _NEITHER in entries["helion_cute"]["error"]
    assert "exact_vs_inductor" not in entries["helion_cute"]
    # The two Inductor compiles name their kernels' configs (these stand-ins
    # launch none), and the record its placement (none drawn here).
    assert [impl for impl in impls if "inductor_configs" in entries[impl]] == [
        "inductor",
        "inductor_cd",
    ]
    assert entries["inductor_cd"]["inductor_configs"] == []
    placement = record["placement"]
    assert placement["seed"] is None and placement["spacer_pages"] == []
    # Where each timed implementation's inputs lie: one storage each.
    assert list(placement["timed_inputs"]) == impls[:4]
    assert all(len(addresses) == 1 for addresses in placement["timed_inputs"].values())
    # What ran right behind the timed port in the rounds of each order: the
    # oracle, and nothing where the port ends the order: the next sample
    # starts there, with the default compile.
    assert record["behind_helion"] == {
        "forward": {"helion_triton": None},
        "reverse": {"helion_triton": "oracle"},
    }
    assert record["carryover_risk"] == [
        "inductor behind helion_triton at the start of the next sample in forward rounds"
    ]
    # No baseline two ways here: the stand-ins launch no kernel, so which
    # compiles are one code is not known.
    assert record["fastest_inductor"] is None


def test_only_the_standard_listing_keeps_inductor_from_behind_a_port() -> None:
    # common.ALL_IMPLS: the memcopy and the oracle follow the ports, and the
    # two ports each other.
    listing = list(common.ALL_IMPLS)
    assert bench.behind_helion(listing) == {
        "forward": {"helion_triton": "helion_cute", "helion_cute": "memcopy"},
        "reverse": {"helion_cute": "helion_triton", "helion_triton": "oracle"},
    }
    # helion.inductor's code runs Helion kernels too (common.runs_helion_code):
    # listed after the ports and before the memcopy (README, helion.inductor),
    # it keeps every Inductor compile from behind Helion code ...
    pinned = [*listing[:-1], "inductor_helion_pretuned", "memcopy"]
    assert bench.carryover_risk(pinned) == []
    # ... and an Inductor compile listed right after it is flagged.
    assert bench.carryover_risk(["inductor_helion_pretuned", "inductor_cd"]) == [
        "inductor_cd behind inductor_helion_pretuned in forward rounds",
        (
            "inductor_cd behind inductor_helion_pretuned at the start of the next "
            "sample in reverse rounds"
        ),
    ]
    assert bench.carryover_risk(listing) == []
    study = [*common.INDUCTOR_IMPLS, "oracle", "helion_cute", "memcopy"]
    assert bench.carryover_risk(study) == []
    # Without the oracle, or with the port listed among the compiles.
    assert bench.carryover_risk([*common.INDUCTOR_IMPLS, "helion_cute", "memcopy"]) == [
        "inductor_max behind helion_cute in reverse rounds"
    ]
    assert bench.carryover_risk(["inductor", "helion_cute", "inductor_cd"]) == [
        "inductor_cd behind helion_cute in forward rounds",
        "inductor behind helion_cute in reverse rounds",
    ]
    # A port that is last in an order has the first of the order behind
    # it, at the start of the next sample: the study's set without the
    # memcopy puts the default compile there.
    assert bench.behind_helion(["helion_cute", "inductor"]) == {
        "forward": {"helion_cute": "inductor"},
        "reverse": {"helion_cute": None},
    }
    assert bench.carryover_risk(["helion_cute", "inductor"]) == [
        "inductor behind helion_cute in forward rounds",
        "inductor behind helion_cute at the start of the next sample in reverse rounds",
    ]
    assert bench.carryover_risk([*common.INDUCTOR_IMPLS, "oracle", "helion_cute"]) == [
        "inductor behind helion_cute at the start of the next sample in forward rounds"
    ]
    # Not an Inductor compile there: no risk.
    assert bench.carryover_risk(["oracle", "inductor", "memcopy", "helion_cute"]) == []
    assert bench.carryover_risk(["helion_cute"]) == []


def test_bench_warns_of_an_inductor_compile_behind_a_port(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The study's listing: both fields in the record, nothing to warn of.
    study = [*common.INDUCTOR_IMPLS, "oracle", "helion_cute", "memcopy"]
    record: dict[str, Any] = {}
    bench.note_carryover(record, study)
    assert record == {
        "behind_helion": {
            "forward": {"helion_cute": "memcopy"},
            "reverse": {"helion_cute": "oracle"},
        },
        "carryover_risk": [],
    }
    assert capsys.readouterr().err == ""
    # Without the oracle between them an Inductor compile runs right behind
    # the port in every round of one order: the record names it, and so
    # does the log, once.
    bench.note_carryover(record, [*common.INDUCTOR_IMPLS, "helion_cute", "memcopy"])
    assert record["behind_helion"]["reverse"] == {"helion_cute": "inductor_max"}
    assert record["carryover_risk"] == [
        "inductor_max behind helion_cute in reverse rounds"
    ]
    warned = capsys.readouterr().err
    assert warned.count("[carryover] inductor_max behind helion_cute in reverse") == 1
    assert "List the oracle between the Inductor compiles and the Helion" in warned
    # The rounds are over the implementations that were timed: one whose
    # graph could not be captured is in none, and its neighbours close up.
    timings = {impl: {"cold_us": 10.0} for impl in study}
    timings["oracle"] = {"timing_error": "Traceback ..."}
    timed = bench.timed_implementations(timings)
    assert timed == [*common.INDUCTOR_IMPLS, "helion_cute", "memcopy"]
    bench.note_carryover(record, timed)
    assert record["behind_helion"] == {
        "forward": {"helion_cute": "memcopy"},
        "reverse": {"helion_cute": "inductor_max"},
    }
    assert record["carryover_risk"] == [
        "inductor_max behind helion_cute in reverse rounds"
    ]
    assert capsys.readouterr().err.count("[carryover]") == 1


def test_an_aggregate_keeps_one_cold_timer(tmp_path: Path) -> None:
    # The aggregate names its processes' cold timer (events for records made
    # before records named it), and a point whose processes timed cold_us
    # two ways has no median: the timers read one call apart.
    processes = _three_processes(tmp_path)
    digests = dict.fromkeys(processes, "")
    assert common.aggregate_bench_records(processes, digests)["cold_timer"] == "events"  # type: ignore[arg-type]
    for record in processes.values():
        record["cold_timer"] = "trace"  # type: ignore[index]
    assert common.aggregate_bench_records(processes, digests)["cold_timer"] == "trace"  # type: ignore[arg-type]
    del next(iter(processes.values()))["cold_timer"]  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="different cold timers"):
        common.aggregate_bench_records(processes, digests)  # type: ignore[arg-type]


def test_a_report_refuses_a_run_of_two_cold_timers(tmp_path: Path) -> None:
    # The bench records of one run directory share their cold timer (a
    # campaign forwards one to every bench: campaign.py --cold-timer); one
    # made before records named it was timed by events.
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    for name, timer in (("a__s1.json", "trace"), ("b__s1.json", "trace")):
        (bench_dir / name).write_text(json.dumps({"status": "ok", "cold_timer": timer}))
    assert len(report.ok_records(tmp_path, "bench")) == 2
    (bench_dir / "c__s1.json").write_text(json.dumps({"status": "ok"}))
    with pytest.raises(ValueError, match="different cold timers"):
        report.ok_records(tmp_path, "bench")


def test_an_aggregate_keeps_what_ran_behind_a_port(tmp_path: Path) -> None:
    # Three processes of the study's listing. The second could not capture
    # its oracle, so there inductor_max ran right behind the port in the
    # reverse rounds: the aggregate keeps each process's followers and
    # names the compile.
    processes = _three_processes(tmp_path)
    names = list(processes)
    digests = dict.fromkeys(names, "")
    study = [*common.INDUCTOR_IMPLS, "oracle", "helion_cute", "memcopy"]
    whole: dict[str, Any] = {}
    bench.note_carryover(whole, study)
    without: dict[str, Any] = {}
    bench.note_carryover(without, [impl for impl in study if impl != "oracle"])
    risk = "inductor_max behind helion_cute in reverse rounds"
    assert whole["carryover_risk"] == [] and without["carryover_risk"] == [risk]
    for name, fields in zip(names, (whole, without, whole), strict=True):
        processes[name].update(fields)  # type: ignore[attr-defined]
    record = common.aggregate_bench_records(processes, digests)  # type: ignore[arg-type]
    assert record["behind_helion_per_process"] == [
        whole["behind_helion"],
        without["behind_helion"],
        whole["behind_helion"],
    ]
    assert "behind_helion" not in record
    assert record["carryover_risk"] == [risk]
    # A first process recorded before the fields hides no later one's.
    del processes[names[0]]["behind_helion"], processes[names[0]]["carryover_risk"]  # type: ignore[attr-defined]
    record = common.aggregate_bench_records(processes, digests)  # type: ignore[arg-type]
    assert record["behind_helion_per_process"] == [
        None,
        without["behind_helion"],
        whole["behind_helion"],
    ]
    assert record["carryover_risk"] == [risk]
    # Processes that all predate the fields leave the aggregate without.
    for name in names[1:]:
        del processes[name]["behind_helion"], processes[name]["carryover_risk"]  # type: ignore[attr-defined]
    record = common.aggregate_bench_records(processes, digests)  # type: ignore[arg-type]
    assert "behind_helion_per_process" not in record
    assert "carryover_risk" not in record


def test_the_report_lists_the_benches_with_a_compile_behind_a_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two points benched in one process each: one with the study's
    # neighbours, one whose listing put inductor_max right behind the port.
    run = tmp_path / "run"
    risk = "inductor_max behind helion_cute in reverse rounds"
    for shape, risks in (("s1", []), ("s2", [risk])):
        _toy_bench(
            run, "sum_000000000001", shape, {"inductor": 50.0, "helion_cute": 25.0}
        )
        _toy_tuned(run, "sum_000000000001", shape, "cute")
        path = common.bench_record_path(run / "bench", "sum_000000000001", shape)
        record = json.loads(path.read_text())
        record["carryover_risk"] = risks
        path.write_text(json.dumps(record))
    monkeypatch.setattr(sys, "argv", ["report.py", str(run)])
    report.main()
    result = json.loads((run / "report.json").read_text())
    rows = {row["shape"]: row for row in result["rows"]["cold_us"]}
    assert rows["s1"]["carryover_risk"] == [] and rows["s2"]["carryover_risk"] == [risk]
    text = (run / "report.md").read_text()
    assert "## Inductor compiles behind a Helion port\n" in text
    assert f"\n- sum_000000000001[s2]: {risk}\n" in text
    assert "- sum_000000000001[s1]:" not in text
    # A run without such a bench has no such section: records made before
    # the field, and those of the study's listing.
    path = common.bench_record_path(run / "bench", "sum_000000000001", "s2")
    record = json.loads(path.read_text())
    del record["carryover_risk"]
    path.write_text(json.dumps(record))
    report.main()
    assert "behind a Helion port" not in (run / "report.md").read_text()


def _unwritten(
    values: torch.Tensor, skipped: slice, byte: int, written: bool = False
) -> torch.Tensor:
    """What a call that never writes ``skipped`` returns for ``values`` on
    memory filled with ``byte``; with ``written``, a call that writes all."""
    out = values.clone()
    if not written:
        size = out.element_size()
        integer = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
        pattern = int.from_bytes(bytes([byte]) * size, "little", signed=size > 1)
        out.view(integer[size])[skipped] = pattern
    return out


def test_the_fill_is_of_free_blocks_and_of_frees_that_wait_for_a_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a device: the allocator's blocks as its snapshot lists them.
    # A block whose free waits for another stream's work (record_stream) is
    # handed out by the next allocation once that work is done, with
    # nothing taken from the driver, so it is filled with the free blocks,
    # after the device has gone idle. Blocks that tensors hold, and another
    # device's, are left alone.
    steps: list[object] = []

    def segment(device: int, *blocks: tuple[int, int, str]) -> dict[str, object]:
        return {
            "device": device,
            "blocks": [
                {"address": address, "size": size, "state": state}
                for address, size, state in blocks
            ],
        }

    def snapshot() -> list[dict[str, object]]:
        steps.append("snapshot")
        return [
            segment(0, (16, 4, "active_allocated"), (32, 8, "inactive")),
            segment(0, (64, 2, "active_pending_free"), (80, 1, "inactive")),
            segment(1, (96, 16, "inactive"), (128, 16, "active_pending_free")),
        ]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: steps.append("idle"))
    monkeypatch.setattr(
        common, "_fill_device_bytes", lambda *block: steps.append(block)
    )
    assert common.poison_free_memory() == 8 + 2 + 1
    fill = common.POISON_BYTE
    assert steps == [
        "idle",
        "snapshot",
        (32, 8, fill),
        (64, 2, fill),
        (80, 1, fill),
        "idle",
    ]
    assert common.allocated_blocks() == {16}


# One case per kind of dtype the outputs have, each with values that the
# fill bytes also spell (a NaN is not among them, -1, 0, True and False are).
@pytest.mark.parametrize(
    "values",
    [
        torch.tensor([1.5, 0.0, -2.0, 0.0], dtype=torch.bfloat16),
        torch.tensor([3, -1, 0, -1]),
        torch.tensor([True, False, True, False]),
        torch.tensor([0, 255, 7, 0], dtype=torch.uint8),
    ],
    ids=lambda values: str(values.dtype),
)
def test_an_output_part_that_follows_the_fill_was_never_written(
    monkeypatch: pytest.MonkeyPatch, values: torch.Tensor
) -> None:
    monkeypatch.setattr(common, "flatten_outputs", list)
    reruns: list[int] = []

    def check(skipped: slice, written: bool = False) -> None:
        def run(byte: int) -> list[torch.Tensor]:
            reruns.append(byte)
            return [values[:2], _unwritten(values, skipped, byte, written)]

        reruns.clear()
        outputs = [values[:2], _unwritten(values, skipped, common.POISON_BYTE, written)]
        common.check_written(outputs, run)

    # Every element written: the values that spell the first fill byte (the
    # integer -1, the byte 255) take one more call, on the second byte, and
    # do not follow it.
    check(slice(0, 0), written=True)
    spelled = values.dtype in (torch.int64, torch.uint8)
    assert reruns == ([common.POISON_RECHECK_BYTE] if spelled else [])
    # Unwritten elements follow the fill whatever their right values are (a
    # zero among them, which the second byte spells), and the error counts
    # them.
    for skipped, count in ((slice(1, 2), 1), (slice(0, 4), 4)):
        with pytest.raises(common.UnwrittenOutput, match=f"output 1: {count} of 4"):
            check(skipped)
        assert reruns == [common.POISON_RECHECK_BYTE]


def test_checks_take_outputs_slice_by_slice_and_through_any_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fill is read in the elements' own bytes: a transposed bool output
    # (a copy would make its bytes 0 or 1), a complex one, one without
    # elements and a scalar, within the slice budget.
    monkeypatch.setattr(common, "CHECK_SLICE_ELEMENTS", 5)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask.view(torch.uint8)[1, 2:5] = 0xFF
    scalar = torch.zeros((), dtype=torch.float32)
    scalar.view(torch.int32).fill_(-1)
    both = torch.zeros(3, dtype=torch.complex64)
    torch.view_as_real(both).view(torch.int32)[0] = -1
    torch.view_as_real(both).view(torch.int32)[1, 0] = -1
    with LargestTensor() as mode:
        counts = [
            common._poison_left(tensor, 0xFF)
            for tensor in (mask.t(), scalar, both, torch.zeros(0, 3))
        ]
    assert counts == [3, 1, 1, 0] and mode.largest <= 5
    zeros = torch.zeros(6, 4, dtype=torch.bool)
    assert common._poison_left(mask.t(), 0xFF, (zeros, 0x00)) == 3


@pytest.fixture
def filled_memory(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Without a device: the bytes that the checks fill free memory with,
    in order (the last is what a stand-in call finds in memory that it does
    not write)."""
    fills: list[int] = []
    monkeypatch.setattr(
        common, "poison_free_memory", lambda byte=common.POISON_BYTE: fills.append(byte)
    )
    return fills


def _skipping_port(fills: list[int], skipped: slice) -> types.SimpleNamespace:
    """A port variant of the stand-in repro that never writes ``skipped`` of
    its first output."""
    return types.SimpleNamespace(
        forward=lambda inputs: (
            _unwritten(inputs[0] * 2, skipped, fills[-1]),
            torch.tensor(_EAGER_ORDER),
        )
    )


def test_check_port_fails_a_port_variant_that_skips_a_slice_of_its_output(
    tie_order_repro: list[str],
    filled_memory: list[int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def check(skipped: slice, *flags: str) -> tuple[object, str]:
        port = _skipping_port(filled_memory, skipped)
        monkeypatch.setattr(common, "load_port", lambda *args: port)
        argv = ["check_port.py", "r", "--shape", "s", "--backends", "cute", *flags]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exit_:
            check_port.main()
        return exit_.value.code, capsys.readouterr().out

    # The negative control: the port leaves one of three elements as it
    # finds it. Every call it is judged by starts from filled memory (the
    # baseline's, the port's, the gate's), and the one after the output
    # showed the first byte from the second.
    code, out = check(slice(2, 3))
    assert code == 1 and out.endswith("FAIL\n")
    assert "UnwrittenOutput: output 0: 1 of 3 elements were never written" in out
    assert filled_memory == [common.POISON_BYTE] * 2 + [common.POISON_RECHECK_BYTE]
    # Without the gate too, where nothing else looks at the values' bytes.
    code, out = check(slice(0, 1), "--no-anchored")
    assert code == 1 and "UnwrittenOutput: output 0: 1 of 3" in out
    # The same port writing everything passes; the gate's own call of it
    # starts from filled memory as well.
    filled_memory.clear()
    code, out = check(slice(0, 0))
    assert code == 0 and out.endswith("PASS\n")
    assert filled_memory == [common.POISON_BYTE] * 3


def test_check_port_checks_the_pinned_config_of_a_campaign(
    tie_order_repro: list[str],
    filled_memory: list[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # --configs-dir: the port pinned to its autotune record is checked like
    # the default config, and a pick that skips part of its output fails
    # the port; a directory without a record of the port says so. What
    # would check nothing without saying why fails: a record that cannot be
    # read, and a directory that is not there.
    loaded: list[object] = []

    def load_port(repro_id: str, backend: str, pinned: object = None) -> object:
        loaded.append(pinned)
        skipped = slice(1, 2) if pinned == {"k": "@broken"} else slice(0, 0)
        return _skipping_port(filled_memory, skipped)

    monkeypatch.setattr(common, "load_port", load_port)
    monkeypatch.setattr(common, "pin_roundtrip", lambda port, kernels: {"k": True})

    def check(
        decorator: str | None, text: str | None = None, configs: Path = tmp_path
    ) -> tuple[object, str]:
        path = common.config_record_path(tmp_path, "r", "s", "cute")
        path.unlink(missing_ok=True)
        if decorator is not None:
            kernels = {"k": {"decorator": decorator, "code_sha256": "0"}}
            text = json.dumps({"status": "ok", "kernels": kernels})
        if text is not None:
            path.write_text(text)
        argv = ["check_port.py", "r", "--shape", "s", "--backends", "cute"]
        monkeypatch.setattr(sys, "argv", [*argv, "--configs-dir", str(configs)])
        loaded.clear()
        with pytest.raises(SystemExit) as exit_:
            check_port.main()
        captured = capsys.readouterr()
        return exit_.value.code, captured.out + captured.err

    code, out = check("@good")
    assert code == 0 and loaded == [None, {"k": "@good"}]
    assert "pinned config" in out and "pinned anchored gate pass=True" in out
    code, out = check("@broken")
    assert code == 1 and "[cute] pinned_config_failure" in out
    assert "UnwrittenOutput: output 0: 1 of 3" in out
    code, out = check(None)
    assert code == 0 and loaded == [None] and "__cute.json): missing" in out
    # The record of an autotune that failed is no pick either.
    code, out = check(None, json.dumps({"status": "error"}))
    assert code == 0 and loaded == [None] and "__cute.json): missing" in out
    code, out = check(None, '{"status": "ok", "kernels"')
    assert code == 1 and loaded == [None] and "__cute.json): error" in out
    assert "[cute] pinned_config_failure" in out and "JSONDecodeError" in out
    code, out = check(None, configs=tmp_path / "absent")
    assert code == 2 and not loaded and "absent: not a directory" in out


def test_bench_does_not_time_an_implementation_that_skips_part_of_its_output(
    tie_order_repro: list[str],
    filled_memory: list[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The CuTe port leaves an element of its output unwritten, the Triton
    # port writes all of it: the replay that the checks judge starts from
    # filled memory, on the second byte for the call that tells an
    # unwritten element from a value.
    ports = {"triton": slice(0, 0), "cute": slice(1, 2)}
    accounting = types.SimpleNamespace(count_bytes_effective=lambda *args: 24)
    monkeypatch.setitem(sys.modules, "byte_accounting", accounting)
    monkeypatch.setattr(
        bench,
        "pinned_port",
        lambda info, configs, repro, shape, backend: (
            _skipping_port(filled_memory, ports[backend]),
            {"kernels": {}},
        ),
    )
    monkeypatch.setattr(common, "pin_roundtrip", lambda port, kernels: {})

    def replayed(
        call: Callable[[list[torch.Tensor]], object], fresh_inputs: object
    ) -> Callable[..., object]:
        def run(inputs: list[torch.Tensor], poison: int = common.POISON_BYTE) -> object:
            common.poison_free_memory(poison)
            return call(inputs)

        return run

    monkeypatch.setattr(bench, "_replayed", replayed)
    monkeypatch.setattr(common, "foreign_gpu_processes", list)
    monkeypatch.setattr(common, "gpu_temperature", lambda: None)
    timed: list[str] = []

    def time_impls(
        fns: dict[str, Callable[[], object]], **rounds: int
    ) -> dict[str, dict[str, float]]:
        timed.extend(fns)
        return {impl: {"bb_us": 2.0, "cold_us": 1.0} for impl in fns}

    monkeypatch.setattr(bench, "time_impls", time_impls)
    record = bench.run(
        argparse.Namespace(
            repro_id="r",
            shape="s",
            impls="helion_triton,helion_cute",
            configs_dir=tmp_path,
            rounds=2,
            rep_ms=1,
            cold_rounds=2,
            cold_timer="events",
            rotate_inputs=0,
            placement_seed=None,
            fixed_placement=True,
        ),
        None,
    )
    assert record["status"] == "ok" and timed == ["helion_triton"]
    entries = record["impls"]
    assert entries["helion_triton"]["status"] == "ok"
    assert entries["helion_cute"]["status"] == "error"
    assert "UnwrittenOutput: output 0: 1 of 3" in entries["helion_cute"]["error"]
    assert "errors_vs_eager" in entries["helion_cute"]
    assert common.POISON_RECHECK_BYTE in filled_memory


def test_an_autotune_fails_a_pick_that_skips_part_of_its_output(
    tie_order_repro: list[str],
    filled_memory: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pick, rebuilt from its decorators as bench.py will pin it, is
    # called on filled memory: one that leaves an element unwritten fails
    # the record, whatever the search's calls left in that memory (its
    # values alone are only a warning against eager).
    kernels = _toy_kernels("cute")
    monkeypatch.setattr(autotune, "register_post_autotune_hook", lambda hook: None)
    monkeypatch.setattr(common, "bound_kernel_records", lambda port: kernels)
    monkeypatch.setattr(common, "pin_roundtrip", lambda port, kernels: {"k": True})

    def tuned(skipped: slice) -> dict[str, object]:
        def load_port(
            repro_id: str, backend: str, pinned: object = None
        ) -> types.SimpleNamespace:
            if pinned is None:
                return _ordering(_EAGER_ORDER)
            return _skipping_port(filled_memory, skipped)

        monkeypatch.setattr(common, "load_port", load_port)
        filled_memory.clear()
        job = argparse.Namespace(
            repro_id="r", shape="s", backend="cute", rtol=2e-2, atol=2e-2, regime=None
        )
        return autotune.run(job)

    record = tuned(slice(1, 2))
    assert record["status"] == "error"
    assert "UnwrittenOutput: output 0: 1 of 3" in str(record["error"])
    assert filled_memory == [common.POISON_BYTE, common.POISON_RECHECK_BYTE]
    record = tuned(slice(0, 0))
    assert record["status"] == "ok" and filled_memory == [common.POISON_BYTE]


class _SequencedKernel:
    """A stand-in for one ``helion.Kernel`` of a port: it records the
    prelude its call would be tuned behind (the one a search started inside
    the call reads) and hands out a compiled function of its own."""

    def __init__(
        self,
        fn: Callable[..., object],
        seen: list[tuple[str, benchmark_prelude.BenchmarkPrelude | None]],
    ) -> None:
        self.fn = fn
        self.seen = seen
        self.compiled = lambda *args: fn(*args)
        # how often the harness asked for the function the call ran with
        self.bound = 0
        # the kernel call the harness named during each call
        self.calls: list[dict[str, object] | None] = []

    def normalize_args(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        assert not kwargs
        return args

    def bind(self, args: tuple[object, ...]) -> types.SimpleNamespace:
        self.bound += 1
        return types.SimpleNamespace(compile_config=lambda allow_print: self.compiled)

    def __call__(self, *args: object) -> object:
        self.seen.append((self.fn.__name__, benchmark_prelude.active_prelude()))
        # (what a search started here would be noted with)
        self.calls.append(common.searching_call.get())
        return self.fn(*args)


def _sequenced_port(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[types.SimpleNamespace, list, dict[str, _SequencedKernel]]:
    """A port of three kernels: ``partials`` reads the input, ``column_sum``
    runs twice on views of what it wrote, and ``scale`` reads the input and
    both sums."""
    seen: list[tuple[str, benchmark_prelude.BenchmarkPrelude | None]] = []

    def partials(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(2, 2, 3).sum(dim=1)

    def column_sum(rows: torch.Tensor) -> torch.Tensor:
        return rows.sum()

    def scale(x: torch.Tensor, first: torch.Tensor, second: torch.Tensor) -> object:
        return x * (first + second)

    kernels = {
        fn.__name__: _SequencedKernel(fn, seen) for fn in (partials, column_sum, scale)
    }
    port = types.SimpleNamespace(**kernels)

    def forward(inputs: list[torch.Tensor]) -> tuple[object, ...]:
        (x,) = inputs
        rows = port.partials(x)
        return (port.scale(x, port.column_sum(rows[0]), port.column_sum(rows[1])),)

    port.forward = forward
    monkeypatch.setattr(
        common,
        "port_kernels",
        lambda module: {
            name: value
            for name, value in vars(module).items()
            if isinstance(value, _SequencedKernel)
        },
    )
    return port, seen, kernels


def test_a_ports_later_kernels_are_tuned_behind_the_earlier_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, seen, kernels = _sequenced_port(monkeypatch)
    x = torch.arange(12.0)
    regimes: dict[str, dict[str, object]] = {}
    with common.kernels_in_sequence(port, "port", regimes):
        assert port.partials is not kernels["partials"]
        (out,) = port.forward([x])
    assert torch.equal(out, x * 66.0)
    # The block over, the module holds its kernels again.
    assert {name: getattr(port, name) for name in kernels} == kernels
    names = [name for name, _prelude in seen]
    assert names == ["partials", "column_sum", "column_sum", "scale"]
    preludes = [prelude for _name, prelude in seen]
    # The port's first kernel call has nothing before it: timed alone.
    assert preludes[0] is None
    # Every later call is given the calls before it, at the functions they
    # ran with, and its own arguments as views of what those calls made.
    first = preludes[1]
    assert first is not None
    assert [step.fn for step in first.steps] == [kernels["partials"].compiled]
    assert first.steps[0].args == (x,)
    assert first.args == (benchmark_prelude.PreludeOutput(0, 0, (3,), (1,), 0),)
    last = preludes[3]
    assert last is not None
    assert [step.fn for step in last.steps] == [
        kernels["partials"].compiled,
        kernels["column_sum"].compiled,
        kernels["column_sum"].compiled,
    ]
    assert last.steps[2].args == (benchmark_prelude.PreludeOutput(0, 0, (3,), (1,), 3),)
    assert last.args[0] is x
    assert last.args[1:] == (
        benchmark_prelude.PreludeOutput(1, 0, (), (), 0),
        benchmark_prelude.PreludeOutput(2, 0, (), (), 0),
    )
    # Run as the benchmark worker runs it, the prelude hands the kernel
    # what the port handed it.
    loaded = last.loaded()
    loaded.run()
    assert torch.equal(loaded.timed(kernels["scale"].compiled)(), out)
    # The record says how each kernel's first call, the one that searches,
    # was treated.
    assert regimes == {
        "partials": {
            "regime": "alone",
            "behind": [],
            "reason": "the port's first kernel call: nothing runs before it",
        },
        "column_sum": {
            "regime": "port",
            "behind": ["partials"],
            "reason": "runs behind 1 earlier kernel call(s) of the port",
        },
        "scale": {
            "regime": "port",
            "behind": ["partials", "column_sum", "column_sum"],
            "reason": "runs behind 3 earlier kernel call(s) of the port",
        },
    }
    # During every call the harness names it, by the port's name for the
    # kernel, for the search it may start: also a kernel's second call.
    assert kernels["partials"].calls == [
        {"kernel": "partials", "regime": "alone", "behind": 0}
    ]
    assert kernels["column_sum"].calls == [
        {"kernel": "column_sum", "regime": "port", "behind": 1},
        {"kernel": "column_sum", "regime": "port", "behind": 2},
    ]
    assert kernels["scale"].calls == [
        {"kernel": "scale", "regime": "port", "behind": 3}
    ]
    assert common.searching_call.get() is None


def test_the_regime_alone_tunes_every_kernel_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, seen, kernels = _sequenced_port(monkeypatch)
    regimes: dict[str, dict[str, object]] = {}
    with common.kernels_in_sequence(port, "alone", regimes):
        port.forward([torch.arange(12.0)])
    assert [prelude for _name, prelude in seen] == [None] * 4
    assert {name: entry["regime"] for name, entry in regimes.items()} == dict.fromkeys(
        kernels, "alone"
    )
    assert regimes["scale"]["reason"] == "--regime alone"
    assert regimes["scale"]["behind"] == ["partials", "column_sum", "column_sum"]
    assert regimes["partials"]["reason"].startswith("the port's first kernel call")
    assert kernels["scale"].calls == [
        {"kernel": "scale", "regime": "alone", "behind": 3}
    ]
    # Nothing of a call is kept or asked of the kernels: they run and
    # search as outside the block.
    assert [kernel.bound for kernel in kernels.values()] == [0, 0, 0]
    with (
        pytest.raises(ValueError, match="regime 'resident'"),
        common.kernels_in_sequence(port, "resident", {}),
    ):
        pass


def test_a_kernel_without_a_prelude_is_tuned_alone_and_the_port_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, seen, kernels = _sequenced_port(monkeypatch)

    def forward(inputs: list[torch.Tensor]) -> object:
        (x,) = inputs
        rows = port.partials(x)
        # A kernel that reads an earlier output as another dtype has no
        # prelude (the sequence follows tensors, not their bits), and
        # neither has a kernel after it: that call cannot be replayed.
        port.column_sum(rows.view(torch.int32))
        port.scale(x, rows[0, 0], rows[1, 0])
        raise RuntimeError("the port failed")

    port.forward = forward
    regimes: dict[str, dict[str, object]] = {}
    with (
        pytest.raises(RuntimeError, match="the port failed"),
        common.kernels_in_sequence(port, "port", regimes),
    ):
        port.forward([torch.arange(12.0)])
    assert seen[1] == ("column_sum", None)
    assert regimes["column_sum"]["regime"] == "alone"
    assert str(regimes["column_sum"]["reason"]).startswith(
        "no prelude: an argument views the torch.float32 output"
    )
    assert seen[2] == ("scale", None)
    assert regimes["scale"]["regime"] == "alone"
    assert regimes["scale"]["behind"] == ["partials", "column_sum"]
    assert str(regimes["scale"]["reason"]).startswith(
        "no prelude: an earlier kernel call read the torch.float32 output"
    )
    assert {name: getattr(port, name) for name in kernels} == kernels


def test_an_autotune_record_names_how_each_kernel_was_tuned(
    tie_order_repro: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    port, seen, _kernels = _sequenced_port(monkeypatch)
    monkeypatch.setattr(common, "make_inputs", lambda *args, **kw: [torch.arange(12.0)])
    monkeypatch.setattr(
        common,
        "load_repro_module",
        lambda repro_id: types.SimpleNamespace(
            Repro=lambda: lambda x: (x * 66.0,)  # what the port computes
        ),
    )
    monkeypatch.setattr(common, "stochastic_outputs", lambda *args: set())
    monkeypatch.setattr(common, "inductor_outputs", lambda *args: None)
    monkeypatch.setattr(common, "load_port", lambda *args, **kwargs: port)
    monkeypatch.setattr(
        common, "bound_kernel_records", lambda port: _toy_kernels("cute")
    )
    monkeypatch.setattr(common, "pin_roundtrip", lambda port, kernels: {"k": True})
    monkeypatch.setattr(
        common,
        "forward_on_poisoned_memory",
        lambda port, *args: port.forward([torch.arange(12.0)]),
    )
    monkeypatch.setattr(common, "check_written", lambda outputs, rerun: None)
    hooks: list[Callable[[object], None]] = []
    monkeypatch.setattr(autotune, "register_post_autotune_hook", hooks.append)

    def tuned(backend: str = "cute", regime: str | None = None) -> dict[str, object]:
        seen.clear()
        job = argparse.Namespace(
            repro_id="r",
            shape="s",
            backend=backend,
            rtol=2e-2,
            atol=2e-2,
            regime=regime,
        )
        return autotune.run(job)

    # The CuTe backend's default regime: every kernel behind the port's
    # earlier ones.
    record = tuned()
    assert record["status"] == "ok", record.get("error")
    assert record["regime"] == "port"
    assert {name: entry["regime"] for name, entry in record["regimes"].items()} == {
        "partials": "alone",
        "column_sum": "port",
        "scale": "port",
    }
    assert [prelude is not None for _name, prelude in seen[:4]] == [
        False,
        True,
        True,
        True,
    ]
    # The old one, for a comparison with records made before.
    record = tuned(regime="alone")
    assert record["status"] == "ok" and record["regime"] == "alone"
    assert all(entry["regime"] == "alone" for entry in record["regimes"].values())
    assert all(prelude is None for _name, prelude in seen)
    # It is Triton's default (its event-timed searches were not measured
    # behind earlier kernels), and the flag names one regime for either.
    assert common.default_regime("cute") == "port"
    assert common.default_regime("triton") == "alone"
    assert record["regimes"]["scale"]["reason"] == "--regime alone"
    triton = tuned(backend="triton")
    assert triton["status"] == "ok" and triton["regime"] == "alone"
    assert all(prelude is None for _name, prelude in seen)
    # (the reason says who chose: no flag was given)
    assert triton["regimes"]["scale"]["reason"] == (
        "the triton backend's default regime"
    )
    forced = tuned(backend="triton", regime="port")
    assert forced["regime"] == "port" and seen[1][1] is not None
    # A bench names the regime of the record it pinned, also of one made
    # before records named any.
    fields = common.timed_record_fields(record)
    assert fields["autotune_regime"] == "alone"
    assert fields["autotune_regimes"] == dict.fromkeys(record["regimes"], "alone")
    assert common.record_regime({"kernels": {}}) == "alone"
    # A kernel that forward runs other than through its name in the port's
    # module is not followed: the kernels after it would be tuned behind
    # preludes without its calls, so the port regime fails the record.
    toy = _toy_kernels("cute")["k"]
    monkeypatch.setattr(
        common,
        "bound_kernel_records",
        lambda port: dict.fromkeys(("partials", "column_sum", "scale"), toy),
    )
    assert tuned()["status"] == "ok"
    aliased = port.partials

    def forward(inputs: list[torch.Tensor]) -> tuple[object, ...]:
        (x,) = inputs
        rows = aliased(x)
        return (port.scale(x, port.column_sum(rows[0]), port.column_sum(rows[1])),)

    port.forward = forward
    failed = tuned()
    assert failed["status"] == "error"
    assert "forward ran ['partials'] other than through" in failed["error"]
    # (tuned alone, nothing depends on the calls before a kernel)
    assert tuned(regime="alone")["status"] == "ok"


def test_a_search_must_have_run_the_prelude_its_call_was_given() -> None:
    def regimes() -> dict[str, dict[str, object]]:
        return {
            "first": {"regime": "alone", "behind": []},
            "second": {"regime": "port", "behind": ["first"]},
            "third": {"regime": "alone", "behind": ["first", "second"]},
        }

    def metrics(
        kernel: str, regime: str, behind: int, **selection: object
    ) -> dict[str, object]:
        # A search's metrics name its kernel by its function, which several
        # kernels of a port may share; ``port_call`` is the call it ran in.
        return {
            "kernel_name": "_shared_function",
            "port_call": {"kernel": kernel, "regime": regime, "behind": behind},
            "final_selection": selection,
        }

    # What each search's final selection says it ran between the flush and
    # the timed calls goes into the record ...
    searched = autotune.searched_regimes(
        regimes(),
        [
            metrics("first", "alone", 0, rounds=4),
            metrics("second", "port", 1, prelude_steps=1),
            # (no final selection: a quick search, or too few finalists)
            {**metrics("third", "alone", 2), "final_selection": None},
            # (a search outside the port's forward names no call)
            {"kernel_name": "elsewhere", "final_selection": {"prelude_steps": 2}},
        ],
    )
    assert searched["first"]["prelude_steps"] == 0
    assert searched["second"]["prelude_steps"] == 1
    assert "prelude_steps" not in searched["third"]
    # ... and a search that ran another prelude than its call was given, or
    # none, fails the record.
    with pytest.raises(RuntimeError, match="second was to be tuned behind 1"):
        autotune.searched_regimes(regimes(), [metrics("second", "port", 1, rounds=4)])
    with pytest.raises(RuntimeError, match="third was to be tuned behind 0"):
        autotune.searched_regimes(
            regimes(), [metrics("third", "alone", 2, prelude_steps=2)]
        )
    # Every search is held to its own call. A kernel called with a second
    # signature searches again, behind more calls, whether or not its first
    # search had a final selection; the entry keeps the first count.
    for first in ({"prelude_steps": 1}, None):
        searched = autotune.searched_regimes(
            regimes(),
            [
                {**metrics("second", "port", 1), "final_selection": first},
                metrics("second", "port", 3, prelude_steps=3),
            ],
        )
        assert searched["second"]["prelude_steps"] == (1 if first else 3)
    with pytest.raises(RuntimeError, match="second was to be tuned behind 3"):
        autotune.searched_regimes(
            regimes(),
            [
                metrics("second", "port", 1, prelude_steps=1),
                metrics("second", "port", 3, prelude_steps=1),
            ],
        )


def test_the_report_says_how_the_runs_kernels_were_tuned() -> None:
    def record(
        point: str, backend: str = "cute", **fields: object
    ) -> dict[str, object]:
        return {
            "repro": point,
            "shape": "s",
            "backend": backend,
            "kernels": {"a": {}, "b": {}},
            **fields,
        }

    behind = {"regime": "port", "behind": ["a"], "reason": "runs behind 1"}
    alone = {"regime": "alone", "behind": [], "reason": "first"}
    refused = {"regime": "alone", "behind": ["a"], "reason": "no prelude: a view"}
    default = {
        "regime": "alone",
        "behind": ["a"],
        "reason": "the triton backend's default regime",
    }
    found = report.tuning_regimes(
        [
            record("p1", regime="port", regimes={"a": alone, "b": behind}),
            record("p2", regime="port", regimes={"a": alone, "b": refused}),
            # a record made before autotune.py named a regime
            record("p3"),
            record("p4", provisional=True),
            # a Triton record in its backend's default regime: every later
            # kernel is tuned alone, which is the record's regime and no
            # exception within it
            record("p1", "triton", regime="alone", regimes={"a": alone, "b": default}),
        ]
    )
    assert found == {
        "records": {"cute": {"port": 2, "alone": 1}, "triton": {"alone": 1}},
        "kernels": {
            "cute": {"port": 1, "alone": 3, "unrecorded": 2},
            "triton": {"port": 0, "alone": 2, "unrecorded": 0},
        },
        "later_alone": {"p2__s__cute.b": "no prelude: a view"},
    }
    text = "\n".join(report.tuning_regime_lines(found))
    assert "## Tuning regime" in text
    assert (
        "- cute: 1 record(s) `alone`, 2 record(s) `port`. Kernels: 1 tuned behind"
        in text
    )
    assert "2 in records that name no regime (tuned alone)" in text
    assert "- triton: 1 record(s) `alone`. Kernels: 0 tuned behind" in text
    assert "- p2__s__cute.b: no prelude: a view" in text
    assert "triton.b" not in text
    assert report.tuning_regime_lines(report.tuning_regimes([])) == []


def test_a_run_plan_reuses_a_several_kernel_record_only_in_its_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common, "activity_issue", lambda record, reuse: None)

    def reason(
        kernels: int, regime: str | None, tuned: str | None = None
    ) -> str | None:
        record: dict[str, object] = {"kernels": dict.fromkeys(range(kernels))}
        if tuned is not None:
            record["regime"] = tuned
        return plan_run.triton_reuse(
            Path("unused"),
            record,
            reuse_unsampled=False,
            pin_check=False,
            regime=regime,
        )

    # (with the pin check off every record is autotuned: the reason given
    # is the first that holds)
    unchecked = "pinned code not checked"
    assert reason(2, "port") == "tuned in the regime 'alone', not 'port'"
    assert reason(2, "port", tuned="port") == unchecked
    assert reason(2, "alone") == unchecked
    assert reason(2, "alone", tuned="port") == "tuned in the regime 'port', not 'alone'"
    # A port with one kernel is tuned alone in every regime.
    assert reason(1, "port") == unchecked
    # Unnamed, the run's Triton autotunes run in Triton's default regime,
    # alone: the records of earlier runs stand.
    assert reason(2, None) == unchecked
    assert reason(2, None, tuned="port") == "tuned in the regime 'port', not 'alone'"


def test_a_call_that_takes_fresh_memory_runs_again_or_cannot_be_judged(
    filled_memory: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without a device: how many times the allocator has taken memory from
    # the driver, as each try finds the count before and after its call. The
    # bytes it has reserved never grow here: an allocator that released its
    # cached blocks to the driver and asked again holds no more than before
    # (under memory pressure, or because the call emptied the cache).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 1 << 20)
    made: list[weakref.ref[torch.Tensor]] = []
    alive: list[int] = []

    def fresh_inputs() -> list[torch.Tensor]:
        # The inputs of the try before this one are gone: the try starts
        # from the allocator state of the one before it.
        alive.append(sum(made_before() is not None for made_before in made))
        tensor = torch.ones(3)
        made.append(weakref.ref(tensor))
        return [tensor]

    def called(allocations: list[int]) -> torch.Tensor:
        made.clear()
        alive.clear()
        filled_memory.clear()
        found = iter(allocations)
        monkeypatch.setattr(
            torch.cuda, "memory_stats", lambda: {"num_device_alloc": next(found)}
        )
        return common.call_on_poisoned_memory(
            lambda inputs: inputs[0] * 2, fresh_inputs
        )

    # No memory taken: one call. Taken in the first call: a second one, on
    # the blocks the first left, filled.
    assert called([7, 7]).tolist() == [2.0] * 3 and len(made) == 1
    assert called([7, 8, 8, 8]).tolist() == [2.0] * 3 and len(made) == 2
    assert filled_memory == [common.POISON_BYTE] * 2 and alive == [0, 0]
    # Taken in every call: where the outputs lie was never filled.
    with pytest.raises(common.UnfilledMemory, match="in each of 3 calls"):
        called([7, 8, 8, 10, 10, 11])
    assert len(made) == 3 and alive == [0, 0, 0]
    # For a random config of check_port.py that is an error, not a config
    # that the backend's classification lets the autotuner skip.
    backend = types.SimpleNamespace(classify_autotune_exception=lambda error: "warn")
    assert check_port.classify_failure(RuntimeError("launch"), backend) == "warn"
    unjudged = common.UnfilledMemory("in each of 3 calls")
    assert check_port.classify_failure(unjudged, backend) == "error"


def _writes_all_but(skipped: int) -> Callable[[list[torch.Tensor]], torch.Tensor]:
    """An implementation of ``x * 2`` that allocates its output with
    ``torch.empty`` and never writes its last ``skipped`` elements."""

    def call(inputs: list[torch.Tensor]) -> torch.Tensor:
        (x,) = inputs
        out = torch.empty_like(x)
        written = x.numel() - skipped
        torch.mul(x[:written], 2, out=out[:written])
        return out

    return call


def _tensors_of(value: object) -> list[object]:
    """Stands in for ``common.flatten_outputs``, which needs the
    better-benchmark checkout: the calls below return a tensor or a list."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


# What an earlier test of the process may have left free in the allocator,
# in MiB: nothing, a block that holds any output of the test below, and
# blocks around the sizes it allocates.
@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("left_free", [(), (64,), (3, 16, 3)], ids=str)
def test_a_call_cannot_pass_on_what_an_earlier_call_left_in_its_memory(
    left_free: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(common, "flatten_outputs", _tensors_of)
    # On the device, with PyTorch's allocator: the hole, then the fill.
    # 3 MB outputs, a size of their own in the allocator's large pool.
    n = 786432
    torch.cuda.empty_cache()
    earlier = [
        torch.empty(size << 20, dtype=torch.uint8, device=DEVICE) for size in left_free
    ]
    del earlier

    def inputs() -> list[torch.Tensor]:
        return [torch.arange(n, device=DEVICE, dtype=torch.float32)]

    expected = inputs()[0] * 2
    good, broken = _writes_all_but(0), _writes_all_but(n // 8)
    # The hole: torch.empty hands the broken call the block that the good
    # call's output was just freed from, values and all.
    left = good(inputs())
    address = left.data_ptr()
    del left
    stale = broken(inputs())
    assert stale.data_ptr() == address and torch.equal(stale, expected)
    del stale
    # Filled, the block is NaN wherever a call does not write it, however
    # the allocator holds it (here: free, then recycled).
    out = common.call_on_poisoned_memory(broken, inputs)
    assert out.data_ptr() == address
    assert bool(out[-n // 8 :].isnan().all()) and torch.equal(
        out[: n - n // 8], expected[: n - n // 8]
    )
    with pytest.raises(common.UnwrittenOutput, match=f"{n // 8} of {n} elements"):
        common.check_written(
            [out], lambda byte: [common.call_on_poisoned_memory(broken, inputs, byte)]
        )
    del out
    out = common.call_on_poisoned_memory(good, inputs)
    common.check_written([out], lambda byte: pytest.fail("a second call"))
    assert torch.equal(out, expected)
    # A call that the allocator has no free block for takes its memory from
    # the driver, which the fill did not reach: it runs again on that block.
    # Its output is larger than all the memory the allocator holds free,
    # whatever earlier calls and tests left there.
    held = inputs()
    free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    count = free // 4 + 5 * n
    calls: list[int] = []

    def larger(inputs: list[torch.Tensor]) -> torch.Tensor:
        calls.append(torch.cuda.memory_reserved())
        out = torch.empty(count, device=DEVICE)
        out[:n] = inputs[0]
        return out

    grown = common.call_on_poisoned_memory(larger, lambda: held)
    assert len(calls) == 2 and calls[1] > calls[0]
    assert bool(grown[n:].isnan().all())
    del grown
    # A call that takes fresh memory every time (it keeps its outputs, each
    # larger than what is free) cannot be judged: its last output lies in
    # memory that was never filled, which reads as zeros here, not as NaN.
    kept: list[torch.Tensor] = []

    def keeps_its_outputs(inputs: list[torch.Tensor]) -> torch.Tensor:
        free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        kept.append(torch.empty(free // 4 + 5 * n, device=DEVICE))
        return kept[-1]

    with pytest.raises(common.UnfilledMemory, match="in each of 3 calls"):
        common.call_on_poisoned_memory(keeps_its_outputs, lambda: held)
    assert len(kept) == 3
    # Nor can a call during which the allocator returns its cached blocks to
    # the driver and takes memory again (what it does when it finds no room,
    # and what a CUDA graph capture does first): its output lies in fresh
    # memory although the allocator holds no more bytes than before the call.
    kept.clear()
    torch.cuda.empty_cache()
    # What stays free now lies in segments that live tensors share, which
    # the allocator cannot give back. A cached block that it can, larger
    # than what the call below takes:
    free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    spare = torch.empty(2 * free + (64 << 20), dtype=torch.uint8, device=DEVICE)
    del spare
    reserved: list[tuple[int, int]] = []

    def empties_the_cache(inputs: list[torch.Tensor]) -> torch.Tensor:
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        # More than is left free, so from the driver.
        free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        out = torch.empty(free // 4 + 5 * n, device=DEVICE)
        reserved.append((before, torch.cuda.memory_reserved()))
        return out

    with pytest.raises(common.UnfilledMemory, match="in each of 3 calls"):
        common.call_on_poisoned_memory(empties_the_cache, lambda: held)
    assert len(reserved) == 3
    assert all(after <= before for before, after in reserved)


@pytest.mark.cuda
@skipIfNotCUDA()
def test_the_fill_reaches_a_block_whose_free_waits_for_a_stream() -> None:
    # A tensor that another stream used (record_stream) is not free when its
    # last reference goes: the allocator keeps the block until that stream's
    # work is done and its next allocation notices, and that allocation can
    # be handed the block with what the tensor held.
    n = 786432
    waiting = torch.full((n,), 7.0, device=DEVICE)
    waiting.record_stream(torch.cuda.Stream())
    address, size = waiting.data_ptr(), waiting.untyped_storage().nbytes()
    del waiting
    assert address in common._allocator_blocks("active_pending_free")
    common.poison_free_memory()
    # The next allocation of that size, in whichever block the allocator
    # picks, reads the fill, and so does the block that was waiting.
    out = torch.empty(n, device=DEVICE)
    block = torch.as_tensor(common._DeviceBytes(address, size), device=DEVICE)
    assert bool((block == common.POISON_BYTE).all())
    assert bool(out.isnan().all())


@pytest.mark.cuda
@skipIfNotCUDA()
def test_a_captured_replay_starts_from_filled_outputs_and_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common, "flatten_outputs", _tensors_of)
    # What bench.py judges and times is a replay of a captured call, whose
    # outputs and buffers are in a pool of the capture's own, fresh from the
    # driver (zeros on a B200's driver, which pass for values that are zero).
    n = 786432

    def inputs() -> list[torch.Tensor]:
        return [torch.arange(n, device=DEVICE, dtype=torch.float32)]

    def half_buffer(inputs: list[torch.Tensor]) -> torch.Tensor:
        # Writes half of a buffer that it frees, and reads all of it.
        buffer = torch.empty(n, device=DEVICE)
        buffer[: n // 2] = 1.0
        return buffer + inputs[0]

    def in_place(inputs: list[torch.Tensor]) -> torch.Tensor:
        return inputs[0].mul_(2)

    expected = inputs()[0] * 2
    with common.gc_paused():
        broken = bench._replayed(_writes_all_but(n // 8), inputs)
        out = broken(inputs())
        assert bool(out[-n // 8 :].isnan().all())
        with pytest.raises(common.UnwrittenOutput, match=f"{n // 8} of {n} elements"):
            common.check_written([out], lambda byte: [broken(inputs(), byte)])
        again = broken(inputs(), common.POISON_RECHECK_BYTE)
        assert not bool(again.isnan().any()) and float(again[-1]) == 0.0
        good = bench._replayed(_writes_all_but(0), inputs)
        assert torch.equal(good(inputs()), expected)
        # A buffer the captured call freed is filled where it lies.
        out = bench._replayed(half_buffer, inputs)(inputs())
        assert not bool(out[: n // 2].isnan().any()) and bool(
            out[n // 2 :].isnan().all()
        )
        # An output in memory from before the call is the call's input.
        assert torch.equal(bench._replayed(in_place, inputs)(inputs()), expected)


def _triton_kernel(
    name: str, meta: dict[str, object], *configs: tuple[dict[str, int], int, bool]
) -> types.SimpleNamespace:
    """A stand-in for one of Inductor's ``CachingAutotuner`` objects with
    these launchers: each config's kwargs, its warps and whether coordinate
    descent found it."""
    launchers = []
    for kwargs, warps, tuned in configs:
        config = types.SimpleNamespace(kwargs=kwargs, num_warps=warps, num_stages=1)
        if tuned:
            config.found_by_coordesc = True
        launchers.append(
            types.SimpleNamespace(
                config=config,
                n_regs=40,
                n_spills=0,
                shared=512,
                cache_hash=f"{name}-{len(launchers)}".upper(),
            )
        )
    return types.SimpleNamespace(
        inductor_meta=meta,
        triton_meta={"device": 0},
        fn=types.SimpleNamespace(__name__=name),
        launchers=launchers,
    )


def _toy_triton_cache(
    cache: Path, key: str, name: str, backend: str = "cuda", **metadata: str
) -> None:
    """An entry of a Triton cache directory as ``triton.compile`` leaves it:
    the kernel's metadata, its group file and a file per stage."""
    entry = cache / key
    entry.mkdir(parents=True)
    target = {"backend": backend, "arch": 100, "warp_size": 32}
    (entry / f"{name}.json").write_text(json.dumps({"target": target, **metadata}))
    (entry / f"__grp__{name}.json").write_text(json.dumps({"child_paths": {}}))
    for kind in (
        ("ttir", "tileir", "cubin") if backend == "tileir" else ("ptx", "cubin")
    ):
        (entry / f"{name}.{kind}").write_text("")


def test_a_bench_record_names_the_config_each_inductor_kernel_ended_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The binary behind the first kernel's launcher, in Triton's cache; the
    # cache holds none of the others'.
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    _toy_triton_cache(
        tmp_path, "TRITON_POI_0-0", "triton_poi_fused_add_0", triton_version="3.9.0"
    )
    pointwise = _triton_kernel(
        "triton_poi_0",
        {
            "kernel_name": "triton_poi_fused_add_0",
            "grid_type": "Grid1D",
            "backend_hash": "37875C9E2A70",
        },
        ({"XBLOCK": 2048}, 2, True),
    )
    # A mix-order reduction: its split size is a block size of the config
    # and what the kernel was generated for.
    mix_order = _triton_kernel(
        "triton_per_1",
        {"grid_type": "MixOrderReductionGrid", "RSPLIT_SIZE": 16},
        ({"XBLOCK": 1, "RSPLIT_SIZE": 16, "NUM_STAGES": 1}, 8, False),
    )
    untuned = _triton_kernel(
        "triton_red_2", {}, ({"XBLOCK": 1}, 4, False), ({"XBLOCK": 2}, 4, False)
    )
    first, second, third = common.inductor_kernel_configs(
        [pointwise, mix_order, untuned]
    )
    assert first == {
        "name": "triton_poi_fused_add_0",
        "grid_type": "Grid1D",
        # The Triton build and backend Inductor generated the kernel for.
        "backend_hash": "37875C9E2A70",
        "block_sizes": {"XBLOCK": 2048},
        "num_warps": 2,
        "num_stages": 1,
        "found_by_coordesc": True,
        "n_regs": 40,
        "n_spills": 0,
        "shared": 512,
        "artifact": {
            "backend": "cuda",
            "triton_version": "3.9.0",
            "meta_ws": None,
            "kinds": ["cubin", "ptx"],
        },
        # The binary's name in Triton's cache.
        "cache_hash": "TRITON_POI_0-0",
    }
    assert second["artifact"] is None and second["backend_hash"] is None
    assert second["name"] == "triton_per_1" and second["rsplit_size"] == 16
    assert second["block_sizes"]["RSPLIT_SIZE"] == 16
    assert second["num_warps"] == 8 and not second["found_by_coordesc"]
    # A kernel that has not settled on one launcher lists them all.
    assert [c["block_sizes"] for c in third["launchers"]] == [
        {"XBLOCK": 1},
        {"XBLOCK": 2},
    ]
    assert report.pick_text([first, second]) == (
        "triton_poi_fused_add_0: XBLOCK=2048 warps=2 stages=1; "
        "triton_per_1: XBLOCK=1 RSPLIT_SIZE=16 NUM_STAGES=1 warps=8 stages=1"
    )


def test_the_kernels_an_inductor_compile_launches_are_taken_at_their_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    monkeypatch.setattr(
        CachingAutotuner,
        "run",
        lambda self, *args, stream, benchmark_run=False: (self, args, stream),
    )
    stub = CachingAutotuner.run
    first, second, candidate = object(), object(), object()
    with common.inductor_kernels_launched() as launched:
        # What the compiled code calls: each kernel's run, with its result.
        assert CachingAutotuner.run(second, 1, stream=0) == (second, (1,), 0)
        # The first call compiles: a mix-order reduction's compile runs the
        # variants it benchmarks its split sizes with, as benchmark runs.
        CachingAutotuner.run(candidate, stream=0, benchmark_run=True)
        CachingAutotuner.run(first, stream=0)
        CachingAutotuner.run(second, 2, stream=0)
    assert launched == [second, first]
    assert CachingAutotuner.run is stub
    with pytest.raises(RuntimeError, match="stop"), common.inductor_kernels_launched():
        raise RuntimeError("stop")
    assert CachingAutotuner.run is stub


def _span(name: str, start: float, end: float) -> types.SimpleNamespace:
    """A device event of a profile."""
    return types.SimpleNamespace(
        name=name,
        device_type=torch.autograd.DeviceType.CUDA,
        time_range=types.SimpleNamespace(start=start, end=end),
    )


def test_a_dependent_launchs_time_is_its_step_not_its_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pointwise_06377502b8f5's first two kernels: the BatchNorm is launched
    # as a dependent launch 2.3 us in, while the max-pool runs to 51.3 us,
    # and ends at 67.1 us. A span that ends inside an earlier one adds
    # nothing, and a kernel that starts after the latest end keeps its span.
    spans = [(0.0, 51.3), (2.3, 67.1), (10.0, 60.0), (70.0, 75.5)]
    assert bench.kernel_steps(spans) == pytest.approx([51.3, 15.8, 0.0, 5.5])

    class Profile:
        def __init__(self, activities: object) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def events(self) -> list[types.SimpleNamespace]:
            # Two replays, each the max-pool and the BatchNorm inside it,
            # in another order than they started; a host event is no kernel.
            host = types.SimpleNamespace(
                name="host", device_type=torch.autograd.DeviceType.CPU
            )
            return [
                _span("batchnorm", 102.3, 167.1),
                host,
                _span("max_pool", 0.0, 51.3),
                _span("max_pool", 100.0, 151.3),
                _span("batchnorm", 2.3, 67.1),
            ]

    monkeypatch.setattr(bench, "profile", Profile)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    replays: list[int] = []
    profiled = bench._profile_kernels(lambda: replays.append(1), calls=2)
    assert len(replays) == 3
    assert profiled["kernels"] == [
        {
            "name": "max_pool",
            "us": pytest.approx(51.3),
            "span_us": pytest.approx(51.3),
            "per_call": 1.0,
        },
        {
            "name": "batchnorm",
            "us": pytest.approx(15.8),
            "span_us": pytest.approx(64.8),
            "per_call": 1.0,
        },
    ]
    assert profiled["kernel_us"] == pytest.approx(67.1)
    assert profiled["kernel_span_us"] == pytest.approx(116.1)
    assert profiled["kernel_spans_overlap"] is True
    # Kernels in a row: the steps are the spans.
    monkeypatch.setattr(
        Profile, "events", lambda self: [_span("a", 0.0, 4.0), _span("b", 4.0, 9.0)]
    )
    profiled = bench._profile_kernels(lambda: None, calls=1)
    assert profiled["kernel_us"] == profiled["kernel_span_us"] == 9.0
    assert profiled["kernel_spans_overlap"] is False


def test_a_process_draws_the_placement_of_its_timed_buffers_from_a_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(1))
    monkeypatch.setattr(
        torch, "empty", lambda size, dtype, device: torch.zeros(size, dtype=dtype)
    )
    monkeypatch.setattr(bench, "PLACEMENT_PAGE_BYTES", 16)

    def pages(seed: int | None, shifts: int = 6) -> list[int]:
        placement = bench.Placement(seed)
        for _ in range(shifts):
            placement.shift()
        record = placement.record()
        assert record["seed"] == seed
        # What the draw came to: where each spacer lies.
        assert len(set(record["spacer_addresses"])) == len(record["spacer_pages"])
        return record["spacer_pages"]

    # Without a seed nothing is held and nothing released: the buffers lie
    # where the allocation order puts them.
    assert pages(None) == [] and not released
    # A seed's spacers: the allocator's cache released before each, their
    # sizes whole pages, the same for the same seed and others for another.
    drawn = pages(7)
    assert len(drawn) == 6 == len(released)
    assert all(1 <= count <= bench.PLACEMENT_MAX_PAGES for count in drawn)
    assert pages(7) == drawn and pages(8) != drawn
    # And where the tensors that an implementation's timed call reads lie:
    # each storage once, with or without a draw.
    x, y = torch.ones(4), torch.ones(4)
    for seed in (None, 7):
        placement = bench.Placement(seed)
        placement.placed("helion_cute", [x, 3, x[1:], y])
        assert placement.record()["timed_inputs"] == {
            "helion_cute": [x.data_ptr(), y.data_ptr()]
        }


def test_bench_draws_a_fresh_placement_unless_told(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int | None] = []

    class Placement:
        def __init__(self, seed: int | None) -> None:
            seeds.append(seed)
            raise RuntimeError("stop here")

    monkeypatch.setattr(bench, "Placement", Placement)
    monkeypatch.setattr(common, "gpu_info", dict)
    monkeypatch.setattr(common, "versions", lambda triton=None: {})

    def run(**placement: object) -> None:
        args = argparse.Namespace(repro_id="r", shape="s", impls="", **placement)
        with pytest.raises(RuntimeError, match="stop here"):
            bench.run(args, None)

    run(placement_seed=None, fixed_placement=True)
    run(placement_seed=11, fixed_placement=False)
    run(placement_seed=None, fixed_placement=False)
    run(placement_seed=None, fixed_placement=False)
    assert seeds[:2] == [None, 11]
    assert None not in seeds[2:] and seeds[2] != seeds[3]


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
    # by select_kernels.py, then the extras, then the next 99 by model share
    # (``--after``), each repro once.
    kernels = json.loads(common.SELECTION_PATH.read_text())["kernels"]
    ports = {path.stem for path in common.KERNELS_DIR.glob("*.py")}
    assert sorted(k["repro"] for k in kernels) == sorted(ports)
    assert len(ports) == len(kernels) == 150
    assert [k["reason"] == "extra" for k in kernels] == (
        [False] * 48 + [True] * 3 + [False] * 99
    )
    assert {k["reason"] for k in kernels[51:]} == {"next model share"}


def test_every_port_loads_on_both_backends_without_a_config() -> None:
    # Every port, not a sample: each is a file of its own, and the rest of
    # the suite loads only a few of them (the test above compares file
    # names), so an edit that keeps a port from loading, or that breaks a
    # rule of PORTING.md for its decorators, showed only in check_port.py on
    # a GPU. load_port raises for a port without a kernel and for a kernel
    # that names a backend ("Leave the backend alone"), hence both backends;
    # "Let the autotuner tile" is static shapes and no config. No kernel is
    # bound, so the 150 ports load in seconds without a GPU.
    for point in json.loads(common.SELECTION_PATH.read_text())["kernels"]:
        for backend in ("cute", "triton"):
            port = common.load_port(point["repro"], backend)
            assert callable(port.forward), point["repro"]
            for name, kernel in common.port_kernels(port).items():
                assert kernel.settings.static_shapes and not kernel.configs, (
                    f"{point['repro']}.{name} on {backend}"
                )


# helion.inductor (common.inductor_source_hash(), the pinned files) needs the
# Inductor of torch 2.16; the module's other tests run on older torch.
needs_helion_inductor = pytest.mark.skipif(
    not requires_torch_version("2.16"), reason="helion.inductor needs torch 2.16"
)


def test_every_pinned_decorator_writes_fast_math_and_dot_precision() -> None:
    # The files replay the settings they were written and checked under,
    # whatever HELION_FAST_MATH or TRITON_F32_DEFAULT a bench sets
    # (inductor_pretuned.with_settings).
    decorators = [
        line
        for path in sorted(common.INDUCTOR_PRETUNED_DIR.glob("*.py"))
        for match in inductor_pretuned.SOURCE_RE.finditer(path.read_text())
        for line in match.group(2).split("\n")
        if line.startswith("@helion.kernel(")
    ]
    assert decorators
    for line in decorators:
        assert "fast_math=" in line and "dot_precision=" in line, line[:80]


@needs_helion_inductor
def test_every_selected_point_has_pinned_inductor_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # One helion.inductor output module per point of the study's 48 and the
    # three extras (the next 99 have none yet), which loads at that point's
    # shape on CPU with every Helion kernel pinned to one config
    # (common.load_inductor_pretuned), and has bench.py's entry.
    from helion.inductor.kernel_sequence import SequencedKernel

    kernels = [
        k
        for k in json.loads(common.SELECTION_PATH.read_text())["kernels"]
        if k["reason"] != "next model share"
    ]
    files = {p.stem for p in common.INDUCTOR_PRETUNED_DIR.glob("*.py")}
    assert len(kernels) == 51
    assert sorted(files) == sorted(k["repro"] for k in kernels)
    if not _has_cute_dsl():
        pytest.skip("the pinned kernels' decorators are CuTe's")
    # PyCodeCache writes each kernel's module under the Inductor cache.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    for kernel in kernels:
        try:
            module = common.load_inductor_pretuned(kernel["repro"], kernel["shape"])
        except (ImportError, AttributeError) as error:
            # A torch whose Inductor no longer has a wrapper helper the files
            # take from torch._C._dynamo.guards (README.md, helion.inductor):
            # they need regenerating. Any other error is the files' own.
            if "torch._C._dynamo.guards" not in str(error):
                raise
            pytest.skip(f"{kernel['repro']}: written for another torch ({error})")
        assert callable(module.forward), kernel["repro"]
        # (a module of helion.inductor's tune_in_sequence wraps its kernels
        # until their first call: helion.inductor.kernel_sequence)
        assert any(
            isinstance(v.kernel if isinstance(v, SequencedKernel) else v, helion.Kernel)
            for v in vars(module).values()
        )


_OUTPUT_MODULE = '''\
helion_fused_a_0_source = """\\
import helion

@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def kernel(in0, out0):
    out0[0] = in0[0]
"""
helion_fused_a_0 = PyCodeCache.load(helion_fused_a_0_source).kernel
helion_fused_b_1_source = """\\
import helion

@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def kernel(in0, out0):
    out0[1] = in0[1]
"""
helion_fused_b_1 = PyCodeCache.load(helion_fused_b_1_source, '0123456789abcdef').kernel
'''


def test_inductor_pretuned_pins_the_decorator_of_each_kernel_source() -> None:
    # Only the decorator line of each source string changes, to the one the
    # kernel's own source picks; decorator_of gets the key the module loads
    # it under (a kernel tuned behind the one before it has its own).
    extras: list[str] = []

    def decorator_of(source: str, extra: str) -> str:
        extras.append(extra)
        return f"@helion.kernel(config=C{int('out0[1]' in source)})"

    code, count = inductor_pretuned.pin(_OUTPUT_MODULE, decorator_of)
    assert count == 2
    assert extras == ["", "0123456789abcdef"]
    assert code == _OUTPUT_MODULE.replace(
        '@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")',
        "@helion.kernel(config=C0)",
        1,
    ).replace(
        '@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")',
        "@helion.kernel(config=C1)",
    )


def test_pinned_outputs_compare_bits_and_layout() -> None:
    # inductor_pretuned.py's check compares the written module's outputs with
    # the compiled call's bits and layout: -0.0 is not 0.0, a NaN matches a
    # NaN of its bits, and a transposed copy of equal values is not the same.
    x = torch.tensor([[0.0, float("nan")], [1.5, -2.0]])
    assert inductor_pretuned.identical(x, x.clone())
    assert not inductor_pretuned.identical(x, torch.where(x == 0, -0.0, x))
    assert not inductor_pretuned.identical(x, x.t().contiguous().t())
    assert inductor_pretuned.identical(x.bool(), x.clone().bool())


def test_pinned_outputs_off_by_their_own_spread_keep_their_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An output the compiled call does not reproduce itself (an atomic
    # accumulation's order) takes a tolerance on its values, not on its
    # layout. These outputs are flat lists already, so better-benchmark's
    # flattening (common.flatten_outputs) is not needed.
    monkeypatch.setattr(common, "flatten_outputs", list)
    e = torch.rand(2, 2)
    out = Path("out.py")
    inductor_pretuned.check_reproduced(out, [e], [e + 1e-7], [e.clone()])
    with pytest.raises(RuntimeError, match="does not reproduce the layout"):
        inductor_pretuned.check_reproduced(
            out, [e], [e + 1e-7], [e.t().contiguous().t()]
        )


def test_inductor_pretuned_reads_the_kernel_source_unescaped() -> None:
    # helion.inductor escapes a kernel source's backslashes in its literal;
    # pin() hands decorator_of the source as Python reads the literal (the
    # one PyCodeCache loaded) and escapes the decorator it writes back.
    source = (
        'import helion\n@helion.kernel(backend="cute")\ndef kernel():\n    s = "a\\b"\n'
    )
    literal = source.replace("\\", "\\\\")
    module = f'helion_k_0_source = """\\\n{literal}"""\n'
    seen: list[str] = []

    def decorator_of(text: str, extra: str) -> str:
        seen.append(text)
        return '@helion.kernel(backend="cute", config="x\\y")'

    code, count = inductor_pretuned.pin(module, decorator_of)
    assert (count, seen) == (1, [source])
    namespace: dict[str, object] = {}
    exec(code, namespace)
    assert namespace["helion_k_0_source"] == source.replace(
        '@helion.kernel(backend="cute")',
        '@helion.kernel(backend="cute", config="x\\y")',
    )


def test_inductor_pretuned_writes_the_fast_math_and_dot_precision_it_ran_with() -> None:
    # format_kernel_decorator() leaves out fast_math and dot_precision; the
    # pinned decorator writes the values the kernel ran with, False and the
    # default precision included, so that HELION_FAST_MATH at the time of
    # writing does not decide them for every later bench.
    decorator = (
        "@helion.kernel(config=helion.Config(block_sizes=[64]), static_shapes=True)"
    )
    ran = helion.Settings(fast_math=True, dot_precision="ieee")
    assert inductor_pretuned.with_settings(decorator, ran) == (
        "@helion.kernel(fast_math=True, dot_precision='ieee', "
        "config=helion.Config(block_sizes=[64]), static_shapes=True)"
    )
    ran = helion.Settings(fast_math=False, dot_precision="tf32")
    assert inductor_pretuned.with_settings(decorator, ran).startswith(
        "@helion.kernel(fast_math=False, dot_precision='tf32', config="
    )
    # A setting the decorator writes already stays as written.
    written = "@helion.kernel(config=helion.Config(), fast_math=True)"
    assert inductor_pretuned.with_settings(written, ran) == (
        "@helion.kernel(dot_precision='tf32', config=helion.Config(), fast_math=True)"
    )


def test_inductor_pretuned_matches_a_config_on_the_kernel_and_its_inputs() -> None:
    def entry(config: dict[str, object], **fields: str) -> dict[str, object]:
        key = {"kernel_source_hash": "k", "specialization_key": "s", "hardware": "B200"}
        key |= {"extra_cache_key": "", "config_spec_hash": "x"} | fields
        return {"key": {"fields": key}, "config": json.dumps(config)}

    want = {"kernel_source_hash": "k", "specialization_key": "s", "hardware": "B200"}
    want["extra_cache_key"] = ""
    # The config spec's hash, which covers the seed configs, may differ.
    picked = entry({"block_sizes": [4]}, config_spec_hash="other seeds")
    assert inductor_pretuned.matching_entry([picked], want) is picked
    for field in ("kernel_source_hash", "specialization_key", "hardware"):
        assert (
            inductor_pretuned.matching_entry([entry({}, **{field: "?"})], want) is None
        )
    # Two entries of one kernel that pick different configs: no guess.
    with pytest.raises(ValueError, match="2 different configs"):
        inductor_pretuned.matching_entry(
            [entry({"block_sizes": [4]}), entry({"block_sizes": [8]})], want
        )


@needs_helion_inductor
def test_configs_from_refuses_a_skipped_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Under HELION_SKIP_CACHE (any value Helion's should_skip_cache() reads as
    # set, "true" included) no kernel looks its --configs-from entry up, and
    # each would autotune: inductor_pretuned.py refuses before compiling.
    monkeypatch.setenv("HELION_SKIP_CACHE", "true")
    args = argparse.Namespace(
        repro_id="var_mean_f4578c128d74",
        shape=None,
        configs_from=tmp_path,
        require_configs=False,
        configs_note=None,
        output=tmp_path / "out.py",
    )
    with pytest.raises(SystemExit, match="HELION_SKIP_CACHE skips its lookups"):
        inductor_pretuned.write(args)


@needs_helion_inductor
def test_pinned_inductor_code_of_another_shape_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "sum_000000000001.py"
    source_hash = common.inductor_source_hash()
    path.write_text(
        f"REPRO = 'sum_000000000001'\nSHAPE = 's1'\nSOURCE_HASH = {source_hash!r}\n"
    )
    assert common.load_inductor_pretuned("sum_000000000001", "s1", path).SHAPE == "s1"
    with pytest.raises(ValueError, match="generated at shape s1, not s2"):
        common.load_inductor_pretuned("sum_000000000001", "s2", path)


@needs_helion_inductor
def test_a_sequenced_kernel_of_pinned_inductor_code_must_be_pinned(
    tmp_path: Path,
) -> None:
    # helion.inductor's code wraps each kernel for its first call
    # (helion_in_sequence): the pin check reads the kernel in the wrapper.
    path = tmp_path / "sum_000000000001.py"
    path.write_text(
        "import helion\n"
        "from helion.inductor.kernel_sequence import in_sequence as helion_in_sequence\n"
        f"REPRO = 'sum_000000000001'\nSHAPE = 's1'\n"
        f"SOURCE_HASH = {common.inductor_source_hash()!r}\n"
        "@helion.kernel(backend='triton')\n"
        "def k(x):\n"
        "    return x\n"
        "k = helion_in_sequence(k, globals(), 'k')\n"
    )
    with pytest.raises(RuntimeError, match="k is not pinned to one config"):
        common.load_inductor_pretuned("sum_000000000001", "s1", path)


@needs_helion_inductor
def test_pinned_inductor_code_of_another_helion_inductor_warns(tmp_path: Path) -> None:
    # A file written by another helion.inductor (SOURCE_HASH) still loads, with
    # a warning that its code may not be what helion.inductor generates now.
    path = tmp_path / "sum_000000000001.py"
    path.write_text("REPRO = 'sum_000000000001'\nSHAPE = 's1'\nSOURCE_HASH = '0'\n")
    with pytest.warns(UserWarning, match="holds the code of another helion.inductor"):
        common.load_inductor_pretuned("sum_000000000001", "s1", path)


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


def _git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    return subprocess.run(
        ["git", "-C", str(repo), *identity, "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _toy_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, times: dict[str, float]
) -> tuple[Path, Path]:
    """A committed better-benchmark checkout in which one model runs each of
    ``times``' repros once (at shape ``s0``), and the all-shapes sweep that
    times them: the checkout and the sweep, for ``select_kernels.main``."""
    repo = tmp_path / "bb"
    for repro in times:
        (repo / "repros" / "canonical" / repro).mkdir(parents=True)
        (repo / "repros" / "canonical" / repro / "repro.py").write_text("# captured\n")
    (repo / "repro_harness.py").write_text("")
    accounting = repo / "benchmarks" / "model_accounting" / "b200"
    accounting.mkdir(parents=True)
    fusible = {common.pattern_hash(repro): {"s0": 1} for repro in times}
    model = {"identity": "suite/infer/model", "model": {"fusible": fusible}}
    (accounting / "model.json").write_text(json.dumps(model))
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "repros")
    sweep = {
        f"repros/canonical/{repro}/repro.py": {
            "point_s0": {"compiled_us": us, "n_kernels": 1}
        }
        for repro, us in times.items()
    }
    (tmp_path / "sweep.json").write_text(json.dumps(sweep))
    monkeypatch.setattr(common, "BB_ROOT", repo)
    monkeypatch.setattr(common, "_check_bb_revision", lambda: None)
    monkeypatch.syspath_prepend(str(repo))
    oracle_harness = types.SimpleNamespace(has_stochastic_ops=lambda path: False)
    monkeypatch.setitem(sys.modules, "oracle_harness", oracle_harness)
    return repo, tmp_path / "sweep.json"


def _select(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["select_kernels.py", *args])
    select_kernels.main()


def test_a_selection_grows_after_an_earlier_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --after keeps the earlier selection's points first, as they are, and
    # adds the next repros by share alone (a first selection gives the
    # second family its turn): none that the earlier one holds (its extra
    # among them), none below --min-us, and no more than there are.
    times = {f"sum_{index:012x}": us for index, us in enumerate([60, 50, 40, 30, 20])}
    times |= {"sum_00000000000f": 5, "mean_00000000000a": 14, "mean_00000000000b": 13}
    sums, means = list(times)[:5], list(times)[6:]
    _, sweep = _toy_checkout(tmp_path, monkeypatch, times)
    original = sweep.read_text()
    out = tmp_path / "selection.json"
    base = ("--sweep", str(sweep), "--output", str(out))
    _select(monkeypatch, *base, "--count", "2", "--extra", f"{sums[4]}:s0")
    first = json.loads(out.read_text())
    assert [(k["repro"], k["reason"]) for k in first["kernels"]] == [
        (sums[0], "top model share"),
        (means[0], "family mean"),
        (sums[4], "extra"),
    ]
    capsys.readouterr()

    _select(monkeypatch, *base, "--after", str(out), "--count", "2")
    grown = json.loads(out.read_text())
    assert grown["kernels"][:3] == first["kernels"]
    assert [(k["repro"], k["reason"]) for k in grown["kernels"][3:]] == [
        (sums[1], "next model share"),
        (sums[2], "next model share"),
    ]
    total = sum(times.values())
    assert grown["selected_model_share"] == pytest.approx((60 + 14 + 50 + 40) / total)
    assert grown["extra_model_share"] == pytest.approx(20 / total)
    assert grown["better_benchmark_commit"] == first["better_benchmark_commit"]
    # The file says what it grew from: its own step, then the earlier one's.
    assert grown["params"] == {**first["params"], "extra": [], "top_fraction": 1.0} | {
        "after": str(out)
    }
    assert first["params"]["top_fraction"] == 0.6
    assert grown["earlier_selection"] == {
        key: first[key] for key in ("sweep", "better_benchmark_commit", "params")
    }
    captured = capsys.readouterr()
    assert captured.out.startswith(
        "4 kernels, 0.71 of 1.0 model-shares, and 1 extras\n"
    )
    assert captured.err == ""

    _select(monkeypatch, *base, "--after", str(out), "--count", "9")
    again = json.loads(out.read_text())
    assert [k["repro"] for k in again["kernels"][5:]] == [sums[3], means[1]]
    assert again["earlier_selection"]["earlier_selection"] == grown["earlier_selection"]
    capsys.readouterr()
    for refused, reason in (
        (("--extra", f"{means[1]}:s0"), "it takes none"),
        (("--top-fraction", "0.6"), "no --top-fraction"),
    ):
        with pytest.raises(SystemExit):
            _select(monkeypatch, *base, "--after", str(out), *refused)
        assert reason in capsys.readouterr().err

    # Another sweep may grow the selection: the earlier points keep their
    # times and shares, and the tool says how many of them the sweep gives
    # others. One more microsecond for any kernel of the model, also the 5 us
    # one that no selection holds, moves every share; twice the time for
    # every kernel moves none of them.
    timed = json.loads(original)
    one_selected = f"repros/canonical/{sums[0]}/repro.py"
    not_selected = "repros/canonical/sum_00000000000f/repro.py"
    for factor, slower in ((1, [one_selected]), (1, [not_selected]), (2, list(timed))):
        retimed = json.loads(original)
        for repro in slower:
            entry = retimed[repro]["point_s0"]
            entry["compiled_us"] = factor * entry["compiled_us"] + (factor == 1)
        sweep.write_text(json.dumps(retimed))
        _select(monkeypatch, *base, "--after", str(out), "--count", "1")
        assert json.loads(out.read_text())["kernels"] == again["kernels"]
        assert f"gives 7 of the 7 points of {out} another time or model share" in (
            capsys.readouterr().err
        )


def test_a_grown_selection_pins_a_commit_that_has_the_earlier_repros_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The ports of the earlier points read their repros' inputs as the
    # commit of the earlier selection has them, and the grown selection pins
    # the checkout's commit for every point: a checkout that moved is taken
    # only while those repros are the same in it.
    names = [f"sum_{index:012x}" for index in range(3)]
    times = dict(zip(names, [60.0, 50.0, 40.0], strict=True))
    repo, sweep = _toy_checkout(tmp_path, monkeypatch, times)
    out = tmp_path / "selection.json"
    _select(monkeypatch, "--sweep", str(sweep), "--output", str(out), "--count", "1")
    first = json.loads(out.read_text())
    assert first["better_benchmark_commit"] == _git(repo, "rev-parse", "HEAD")
    assert [k["repro"] for k in first["kernels"]] == [names[0]]
    grow = ("--sweep", str(sweep), "--output", str(tmp_path / "grown.json"))
    grow = (*grow, "--after", str(out), "--count", "1")

    def recapture(repro: str) -> str:
        path = repo / "repros" / "canonical" / repro / "repro.py"
        path.write_text(path.read_text() + "# captured again\n")
        _git(repo, "commit", "-q", "-am", f"recapture {repro}")
        return _git(repo, "rev-parse", "HEAD")

    moved = recapture(names[1])
    _select(monkeypatch, *grow)
    grown = json.loads((tmp_path / "grown.json").read_text())
    assert grown["better_benchmark_commit"] == moved != first["better_benchmark_commit"]
    assert [k["repro"] for k in grown["kernels"]] == names[:2]

    recapture(names[0])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        _select(monkeypatch, *grow)
    error = capsys.readouterr().err
    assert f"1 repros of {out} changed" in error
    assert names[0] in error and names[1] not in error

    # An earlier selection without points has no repro that could differ.
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({**first, "kernels": []}))
    _select(monkeypatch, *grow[:4], "--after", str(empty), "--count", "1")
    grown = json.loads((tmp_path / "grown.json").read_text())
    assert [k["repro"] for k in grown["kernels"]] == [names[0]]

    # A pinned commit that the checkout does not have cannot be compared.
    out.write_text(json.dumps({**first, "better_benchmark_commit": "0" * 40}))
    with pytest.raises(SystemExit):
        _select(monkeypatch, *grow)
    assert "cannot compare better-benchmark commit" in capsys.readouterr().err


def _without_autotune_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HELION_AUTOTUNE_EFFORT", "HELION_FORCE_AUTOTUNE"):
        # Set first, so that the variable is also removed after the test
        # when the test's code sets it.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.mark.parametrize(
    ("env", "effort"),
    [
        ({}, "none"),
        ({"HELION_AUTOTUNE_EFFORT": " "}, "none"),
        ({"HELION_FORCE_AUTOTUNE": "0"}, "none"),
        ({"HELION_FORCE_AUTOTUNE": "off"}, "none"),
        ({"HELION_FORCE_AUTOTUNE": "1"}, "full"),
        ({"HELION_FORCE_AUTOTUNE": "true"}, "full"),
        ({"HELION_AUTOTUNE_EFFORT": "quick"}, "quick"),
        ({"HELION_AUTOTUNE_EFFORT": "full", "HELION_FORCE_AUTOTUNE": "0"}, "full"),
    ],
)
def test_an_unpinned_port_searches_only_when_the_environment_asks(
    env: dict[str, str], effort: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A script that loads a port and calls it gets the default configs, not
    # a search, unless the environment names an effort or forces autotuning,
    # as Helion's settings read the two variables.
    _without_autotune_variables(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)
    port = common.load_port("pointwise_0872313c7e7a", "triton")
    kernels = common.port_kernels(port)
    assert [k.settings.autotune_effort for k in kernels.values()] == [effort]
    assert dict(os.environ) == before


def test_a_port_loads_whatever_backend_the_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The kernels are made under load_port's own HELION_BACKEND, and so is
    # its reading of the two variables: a value the caller's shell holds
    # for another purpose (or a misspelled one) is not looked at.
    _without_autotune_variables(monkeypatch)
    monkeypatch.setenv("HELION_BACKEND", "no-such-backend")
    port = common.load_port("pointwise_0872313c7e7a", "triton")
    (kernel,) = common.port_kernels(port).values()
    assert (kernel.settings.backend, kernel.settings.autotune_effort) == (
        "triton",
        "none",
    )
    assert os.environ["HELION_BACKEND"] == "no-such-backend"


def test_a_pinned_port_takes_its_settings_from_its_decorators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default-config override is for unpinned ports: a pinned kernel is
    # rebuilt from its decorator under the caller's environment, as before.
    _without_autotune_variables(monkeypatch)
    decorator = "@helion.kernel(config=helion.Config(block_sizes=[64]))"
    port = common.load_port(
        "pointwise_0872313c7e7a", "triton", pinned={"leaky_relu": decorator}
    )
    (kernel,) = common.port_kernels(port).values()
    assert kernel.settings.autotune_effort == "full"
    assert [dict(config) for config in kernel.configs] == [{"block_sizes": [64]}]


def test_a_standalone_autotune_asks_for_a_cold_full_search(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # autotune.py run as a script names both variables unless the
    # environment does, so that load_port's default configs are not what a
    # run that allows the autotune cache (HELION_FORCE_AUTOTUNE=0) records.
    assert autotune.__file__ is not None
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["autotune.py", "--help"])
    for env, expected in (
        ({}, ("full", "1")),
        ({"HELION_FORCE_AUTOTUNE": "0"}, ("full", "0")),
        ({"HELION_AUTOTUNE_EFFORT": " ", "HELION_FORCE_AUTOTUNE": "0"}, ("full", "0")),
        (
            {"HELION_AUTOTUNE_EFFORT": "quick", "HELION_FORCE_AUTOTUNE": ""},
            ("quick", "1"),
        ),
    ):
        _without_autotune_variables(monkeypatch)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(SystemExit):
            runpy.run_path(autotune.__file__, run_name="__main__")
        assert (
            os.environ["HELION_AUTOTUNE_EFFORT"],
            os.environ["HELION_FORCE_AUTOTUNE"],
        ) == expected
    assert "--backend" in capsys.readouterr().out


def test_a_standalone_port_check_names_no_search(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # check_port.py run as a script names the effort none unless the
    # environment names one. An empty value names none: kept beside
    # HELION_FORCE_AUTOTUNE=1, it would have load_port's unpinned kernels
    # search (the settings read an empty effort as the default, full).
    assert check_port.__file__ is not None
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["check_port.py", "--help"])
    for env, expected in (
        ({}, "none"),
        ({"HELION_AUTOTUNE_EFFORT": " ", "HELION_FORCE_AUTOTUNE": "1"}, "none"),
        ({"HELION_AUTOTUNE_EFFORT": "quick"}, "quick"),
    ):
        _without_autotune_variables(monkeypatch)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(SystemExit):
            runpy.run_path(check_port.__file__, run_name="__main__")
        assert os.environ["HELION_AUTOTUNE_EFFORT"] == expected
    assert "--random-configs" in capsys.readouterr().out
    # What the kernels of an unpinned port then run with, forced or not.
    monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "none")
    monkeypatch.setenv("HELION_FORCE_AUTOTUNE", "1")
    port = common.load_port("pointwise_0872313c7e7a", "triton")
    (kernel,) = common.port_kernels(port).values()
    assert kernel.settings.autotune_effort == "none"


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


def _compile(
    rounds: list[float], xblock: int = 1, **fields: object
) -> dict[str, object]:
    """An Inductor compile's entry in a bench record: its cold rounds, the
    timing bench.py takes of them, and the config of its one kernel."""
    return {
        "status": "ok",
        "numerics_pass": True,
        "inductor_configs": [
            {"name": "k", "block_sizes": {"XBLOCK": xblock}, "num_warps": 1}
        ],
        "cold_us": common.order_balanced(rounds, statistics.median)[0],
        "cold_order_medians_us": common.order_balanced(rounds, statistics.median)[1],
        "cold_runs_us": rounds,
        **fields,
    }


def test_debiased_fastest_chooses_on_other_rounds_than_it_times() -> None:
    steady = [10.0] * 6
    # One candidate: its own value, bench.py's statistic of the rounds.
    assert common.debiased_fastest([[9.8, 10.4, 10.0, 10.2, 10.2, 10.6]]) == (
        pytest.approx(10.2)
    )
    # A candidate that two pairs of rounds read fastest and the third slow:
    # the minimum of the two values is its 9.5, but the rounds that choose
    # it are not the rounds that time it.
    lucky = [9.5, 9.5, 9.5, 9.5, 12.0, 12.0]
    assert common.order_balanced(lucky, statistics.median)[0] == 9.5
    assert common.debiased_fastest([steady, lucky]) == pytest.approx(10.0)
    assert common.debiased_fastest([lucky, steady]) == pytest.approx(10.0)
    # A candidate that every round reads faster is the fastest either way.
    faster = [9.0, 9.2, 9.0, 9.2, 9.0, 9.2]
    assert common.debiased_fastest([steady, faster]) == pytest.approx(9.1)
    # Rounds for a single pair: nothing to choose on, the fastest's value.
    assert common.debiased_fastest([[10.0, 10.0], [9.0, 9.4]]) == pytest.approx(9.2)


def test_fastest_inductor_pools_the_compiles_that_ended_on_one_code() -> None:
    # Three compiles of one code, 10 us in every round but for 0.2 us of
    # noise that falls elsewhere for each: the fastest reads 9.8, the three
    # together 10.
    impls = {
        "inductor": _compile([9.8, 9.8, 10.0, 10.0, 10.2, 10.2]),
        "inductor_cd": _compile(
            [10.2, 10.2, 10.4, 10.4, 10.0, 10.0],
            inductor_configs=[
                {
                    "name": "k",
                    "block_sizes": {"XBLOCK": 1},
                    "num_warps": 1,
                    # Who found a config does not make it another code.
                    "found_by_coordesc": True,
                }
            ],
        ),
        "inductor_max": _compile([10.0, 10.0, 9.6, 9.6, 9.8, 9.8]),
        "helion_cute": {"status": "ok", "cold_us": 10.0, "cold_runs_us": [10.0] * 6},
    }
    assert [impls[impl]["cold_us"] for impl in common.INDUCTOR_IMPLS] == [
        pytest.approx(value) for value in (10.0, 10.2, 9.8)
    ]
    assert common.fastest_inductor(impls) == {
        "same_code": [["inductor", "inductor_cd", "inductor_max"]],
        "cold_us": {"headline": pytest.approx(9.8), "debiased": pytest.approx(10.0)},
    }
    # Another code is another candidate; the two of one code stay pooled.
    impls["inductor_cd"] = _compile([9.0] * 6, xblock=2)
    assert common.fastest_inductor(impls) == {
        "same_code": [["inductor", "inductor_max"], ["inductor_cd"]],
        "cold_us": {"headline": pytest.approx(9.0), "debiased": pytest.approx(9.0)},
    }
    # A compile that failed its gate is no baseline, as in the headline.
    impls["inductor_cd"]["numerics_pass"] = False
    assert common.fastest_inductor(impls) == {
        "same_code": [["inductor", "inductor_max"]],
        "cold_us": {"headline": pytest.approx(9.8), "debiased": pytest.approx(9.9)},
    }
    # Both cold timers, where the compiles have them; one that only some
    # of them have is of those, as its headline is. bb_us, a minimum by
    # design, gets no such figure.
    for impl, bb in (("inductor", [14.0, 14.4] * 3), ("inductor_max", [14.2] * 6)):
        impls[impl].update(bb_us=min(bb), bb_runs_us=bb, bb_order_mins_us=[min(bb)] * 2)
    rot = {
        "rot_us": 11.0,
        "rot_runs_us": [11.0] * 6,
        "rot_order_medians_us": [11.0] * 2,
    }
    impls["inductor"].update(rot)
    found = common.fastest_inductor(impls)
    assert found is not None and sorted(found) == ["cold_us", "rot_us", "same_code"]
    assert found["rot_us"] == {"headline": 11.0, "debiased": 11.0}
    for field in rot:
        del impls["inductor"][field]
    # A compile without a finite, positive timing is not among those the
    # headline is the fastest of, nor here.
    lost = {
        "inductor": _compile([10.0] * 6),
        "inductor_cd": _compile([9.0] * 6, xblock=2, cold_us=math.inf),
    }
    assert common.fastest_inductor(lost) == {
        "same_code": [["inductor"], ["inductor_cd"]],
        "cold_us": {"headline": 10.0, "debiased": 10.0},
    }
    # A record made before the order-balanced timings: its cold_us is the
    # median of five rounds, another statistic than the figure's, so it has
    # none until its timings are recomputed (report.py --rebalance).
    rounds = [10.0, 11.0, 10.0, 11.0, 10.0]
    legacy: dict[str, Any] = {
        "repro": "sum_000000000001",
        "shape": "s1",
        "impls": {
            "inductor": _compile(rounds, cold_us=statistics.median(rounds)),
            "helion_cute": {"status": "ok", "cold_us": 10.0, "cold_runs_us": rounds},
        },
    }
    del legacy["impls"]["inductor"]["cold_order_medians_us"]
    assert common.fastest_inductor(legacy["impls"]) == {"same_code": [["inductor"]]}
    assert report.summarize(legacy, "cold_us")["cute_vs_inductor_debiased"] is None
    rebalanced = report.rebalance(legacy)
    assert rebalanced["impls"]["helion_cute"]["cold_us"] == 10.5
    assert common.fastest_inductor(rebalanced["impls"]) == {
        "same_code": [["inductor"]],
        "cold_us": {"headline": 10.5, "debiased": 10.5},
    }
    assert report.summarize(rebalanced, "cold_us")["cute_vs_inductor_debiased"] == 1.0
    # Compiles that do not name their kernels' configs (records made before
    # bench.py noted them): which are one code is unknown, and without the
    # pooling the figure would keep most of the bias, so there is none.
    del impls["inductor_max"]["inductor_configs"]
    assert common.fastest_inductor(impls) is None
    impls["inductor_max"]["inductor_configs"] = []
    assert common.fastest_inductor(impls) is None
    # No usable compile, no baseline.
    assert common.fastest_inductor({"helion_cute": impls["helion_cute"]}) is None


def test_fastest_inductor_holds_two_binaries_apart_by_their_hash() -> None:
    # Three compiles whose kernels agree in all that a record said of them
    # before it named each binary (cache_hash, Triton's hash of what it
    # compiled): such a record's compiles are one code, as before.
    rounds = {
        "inductor": [9.8, 9.8, 10.0, 10.0, 10.2, 10.2],
        "inductor_cd": [10.2, 10.2, 10.4, 10.4, 10.0, 10.0],
        "inductor_max": [10.0, 10.0, 9.6, 9.6, 9.8, 9.8],
    }
    kernel = {"name": "k", "block_sizes": {"XBLOCK": 1}, "num_warps": 1, "n_regs": 40}

    def found(**hashes: str) -> dict[str, Any] | None:
        return common.fastest_inductor(
            {
                impl: _compile(
                    readings,
                    inductor_configs=[
                        {**kernel, "cache_hash": hashes[impl]} if hashes else kernel
                    ],
                )
                for impl, readings in rounds.items()
            }
        )

    pooled = {
        "same_code": [["inductor", "inductor_cd", "inductor_max"]],
        "cold_us": {"headline": pytest.approx(9.8), "debiased": pytest.approx(10.0)},
    }
    assert found() == pooled
    # One binary under the three compiles is one code as well.
    assert found(inductor="A", inductor_cd="A", inductor_max="A") == pooled
    # Another binary with the same config and resource counts is another
    # candidate: its timings are not averaged with the other two's.
    assert found(inductor="A", inductor_cd="A", inductor_max="B") == {
        "same_code": [["inductor", "inductor_cd"], ["inductor_max"]],
        "cold_us": {"headline": pytest.approx(9.8), "debiased": pytest.approx(9.8)},
    }


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


def test_flags_that_would_override_one_another_are_usage_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A seed with --fixed-placement would draw nothing, and --no-cd would
    # replace the compile that --impl names: neither goes unsaid.
    files = ["--configs-dir", "c", "--output", "o"]
    for main, flags in (
        (bench.main, [*files, "--placement-seed", "3", "--fixed-placement"]),
        (inductor_code.main, ["--impl", "inductor_max", "--no-cd"]),
    ):
        monkeypatch.setattr(sys, "argv", ["script.py", "r", *flags])
        with pytest.raises(SystemExit) as exit_:
            main()
        assert exit_.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err


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

    def run(args: object, backend: object) -> dict[str, object]:
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
    # The checks before a bench compiles have tests of their own.
    monkeypatch.setattr(bench, "preflight", lambda backend: {})
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
        "cache_directories": {},
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


def _toy_backend(name: str = "stock", **fields: object) -> dict[str, object]:
    """What ``common.triton_backend`` says of a Triton build: stock Triton,
    ``fbtriton`` (with ``+meta_ws``: its switch in effect) or the Tile IR
    fork (``tileir``; ``tileir-fork-ptx`` with its switch off)."""
    fbtriton = name.startswith("fbtriton")
    fork = name.startswith("tileir")
    version = "3.9.0.dev20261113+fb.git305df91c" if fbtriton else "3.9.0"
    backend: dict[str, object] = {
        "name": name,
        "distribution": "fbtriton" if fbtriton else "triton",
        "distribution_version": version.split("+")[0] + ("" if fbtriton else "+gitb7d"),
        "version": version,
        "switches": {
            "TRITON_USE_META_WS": "1" if name == "fbtriton+meta_ws" else None,
            "ENABLE_TILE": "1" if name == "tileir" else None,
        },
        "has_meta_ws": fbtriton,
        "meta_ws": name == "fbtriton+meta_ws",
        "has_tileir": fork,
        "tile": name == "tileir",
        "tileiras": sys.executable if fork else None,
        "tileiras_error": None,
        "environment": {},
        "static_cuda_launcher": not fork,
    }
    backend.update(fields)
    return backend


def _switches(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    """Only these of the switch variables in the environment."""
    others = ("TRITON_TILEIR_RUNTIME_FALLBACK", "TRITON_DEFAULT_BACKEND")
    for name in (*common.TRITON_SWITCHES, *others):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_a_switch_that_cannot_be_in_effect_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No switch asked for: any build passes, also none at all. A spelling
    # of off asks for nothing.
    _switches(monkeypatch)
    for backend in (None, _toy_backend(), _toy_backend("fbtriton")):
        common.check_triton_switches(backend)
    _switches(monkeypatch, TRITON_USE_META_WS="0", ENABLE_TILE="false")
    common.check_triton_switches(_toy_backend())
    # Meta's warp specialization: only fbtriton has the knob. Stock Triton
    # and the Tile IR fork would compile as without the variable.
    _switches(monkeypatch, TRITON_USE_META_WS="1")
    common.check_triton_switches(_toy_backend("fbtriton+meta_ws"))
    with pytest.raises(common.TritonSwitchError, match="no Triton is installed"):
        common.check_triton_switches(None)
    for backend in (_toy_backend(), _toy_backend("tileir-fork-ptx")):
        with pytest.raises(
            common.TritonSwitchError,
            match="TRITON_USE_META_WS=1 is set.*no Meta warp specialization",
        ):
            common.check_triton_switches(backend)
    with pytest.raises(common.TritonSwitchError, match="does not read that value"):
        common.check_triton_switches(_toy_backend("fbtriton"))
    # Tile IR: only the fork has the backend, it reads ENABLE_TILE=1 alone,
    # needs its assembler, and must not fall back to PTX at launch.
    _switches(monkeypatch, ENABLE_TILE="1")
    common.check_triton_switches(_toy_backend("tileir"))
    for backend in (_toy_backend(), _toy_backend("fbtriton")):
        with pytest.raises(common.TritonSwitchError, match="no Tile IR backend"):
            common.check_triton_switches(backend)
    with pytest.raises(common.TritonSwitchError, match="assembler is missing"):
        common.check_triton_switches(_toy_backend("tileir", tileiras=None))
    with pytest.raises(common.TritonSwitchError, match="assembler is missing"):
        common.check_triton_switches(_toy_backend("tileir", tileiras="/no/tileiras"))
    monkeypatch.setenv("TRITON_TILEIR_RUNTIME_FALLBACK", "1")
    with pytest.raises(common.TritonSwitchError, match="through PTX instead"):
        common.check_triton_switches(_toy_backend("tileir"))
    _switches(monkeypatch, ENABLE_TILE="true")
    with pytest.raises(common.TritonSwitchError, match="does not select its Tile"):
        common.check_triton_switches(_toy_backend("tileir-fork-ptx"))
    # Tile IR in effect without ENABLE_TILE (the fork's main reads
    # TRITON_DEFAULT_BACKEND first): the same two checks of a run through it.
    _switches(monkeypatch)
    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "tileir")
    common.check_triton_switches(_toy_backend("tileir"))
    lost = _toy_backend("tileir", tileiras=None, tileiras_error="tileiras not found")
    with pytest.raises(
        common.TritonSwitchError,
        match=r"in effect \(TRITON_DEFAULT_BACKEND=tileir\).*tileiras not found",
    ):
        common.check_triton_switches(lost)
    monkeypatch.setenv("TRITON_TILEIR_RUNTIME_FALLBACK", "1")
    with pytest.raises(common.TritonSwitchError, match="through PTX instead"):
        common.check_triton_switches(_toy_backend("tileir"))
    # The fork with its backend not selected is not held to them.
    common.check_triton_switches(_toy_backend("tileir-fork-ptx", tileiras=None))
    # A build without the backend has nothing to select.
    for backend in (_toy_backend(), _toy_backend("fbtriton")):
        with pytest.raises(
            common.TritonSwitchError, match="no Tile IR backend to select"
        ):
            common.check_triton_switches(backend)


def test_the_record_names_the_distribution_that_provides_triton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    triton = pytest.importorskip("triton")
    backend = common.triton_backend()
    assert backend is not None and backend["version"] == triton.__version__
    assert backend["distribution"] in backend["distributions"]
    assert common.versions()["triton"] == backend["distribution_version"]
    assert set(backend["switches"]) == set(common.TRITON_SWITCHES)
    assert not any(name.endswith("_CACHE_DIR") for name in backend["environment"])
    assert common.triton_identity(backend)["name"] == backend["name"]
    # fbtriton installed over the triton that torch requires: both
    # distributions claim the package. Where neither records the size of
    # its library, the version tells whose it is.
    versions = {"triton": "3.9.0+gitb7ddcf86", "fbtriton": "3.9.0.dev20261113"}
    monkeypatch.setattr(
        importlib.metadata,
        "packages_distributions",
        lambda: {"triton": ["triton", "fbtriton", "triton"]},
    )
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(importlib.metadata, "files", lambda name: None)
    monkeypatch.setattr(triton, "__version__", "3.9.0.dev20261113+fb.git305df91c")
    backend = common.triton_backend()
    assert backend is not None and backend["distributions"] == versions
    assert backend["distribution"] == "fbtriton"
    assert backend["distribution_version"] == "3.9.0.dev20261113"
    assert backend["commit"] == "305df91c"
    # The fork's main as triton_tileir over that triton: both versions
    # start triton.__version__, 3.9.0, alike, and the one whose record of
    # the library is the installed file's size is the build, whichever
    # was installed last.
    library = tmp_path / "libtriton.so"
    sizes = {"triton": 5, "triton_tileir": 9}
    versions = {"triton": "3.9.0+gitb7ddcf86", "triton_tileir": "3.9.0+git1dd8e2f2"}

    def files(name: str) -> list[types.SimpleNamespace]:
        other = types.SimpleNamespace(name="__init__.py", size=1, locate=lambda: "")
        recorded = types.SimpleNamespace(
            name="libtriton.so", size=sizes[name], locate=lambda: str(library)
        )
        return [other, recorded]

    monkeypatch.setattr(importlib.metadata, "files", files)
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(triton, "__version__", "3.9.0")
    for order in (["triton", "triton_tileir"], ["triton_tileir", "triton"]):
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda order=order: {"triton": order},
        )
        for installed, commit in (
            ("triton_tileir", "1dd8e2f2"),
            ("triton", "b7ddcf86"),
        ):
            library.write_bytes(b"x" * sizes[installed])
            backend = common.triton_backend()
            assert backend is not None and backend["distribution"] == installed
            assert backend["distribution_version"] == versions[installed]
            assert backend["commit"] == commit
            assert common.triton_identity(backend)["distribution"] == installed
    # The fork's main names itself triton_tileir; stock Triton's version
    # carries the commit only in the distribution's.
    versions = {"triton_tileir": "3.9.0+git1dd8e2f2"}
    monkeypatch.setattr(
        importlib.metadata, "packages_distributions", lambda: {"triton": [*versions]}
    )
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(triton, "__version__", "3.9.0")
    backend = common.triton_backend()
    assert backend is not None and backend["distribution"] == "triton_tileir"
    assert backend["commit"] == "1dd8e2f2"


def test_a_switched_run_takes_only_cache_directories_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inductor, triton_cache = tmp_path / "inductor", tmp_path / "triton"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(inductor))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton_cache))
    directories = {
        "TORCHINDUCTOR_CACHE_DIR": str(inductor),
        "TRITON_CACHE_DIR": str(triton_cache),
    }
    stamps = [inductor / common.CACHE_STAMP, triton_cache / common.CACHE_STAMP]
    # A run without a switch leaves no stamp, in empty directories or in
    # ones it has filled.
    assert common.claim_caches(_toy_backend("fbtriton")) == directories
    assert not inductor.exists() and not triton_cache.exists()
    (inductor / "fxgraph").mkdir(parents=True)
    assert common.claim_caches(_toy_backend("fbtriton")) == directories
    assert not any(stamp.exists() for stamp in stamps)
    # A run with the switch on would be handed that run's cubins by
    # Inductor's FX-graph cache: it takes no directory it cannot tell.
    switched = _toy_backend("fbtriton+meta_ws")
    with pytest.raises(common.TritonSwitchError, match="TORCHINDUCTOR_CACHE_DIR"):
        common.claim_caches(switched)
    (inductor / "fxgraph").rmdir()
    # So with a kernel in Triton's cache; the launcher modules that a Triton
    # build leaves there as it is imported are no kernels. The bench that is
    # refused leaves no stamp in the Inductor directory, which it could have
    # taken: a run without the switch still goes there.
    (triton_cache / "DRIVER").mkdir(parents=True)
    (triton_cache / "DRIVER" / "tileir_utils.so").write_text("")
    _toy_triton_cache(triton_cache, "KEY", "k")
    with pytest.raises(common.TritonSwitchError, match="TRITON_CACHE_DIR"):
        common.claim_caches(switched)
    assert not any(stamp.exists() for stamp in stamps)
    assert common.claim_caches(_toy_backend("fbtriton")) == directories
    for path in sorted((triton_cache / "KEY").iterdir()):
        path.unlink()
    (triton_cache / "KEY").rmdir()
    assert common.claim_caches(switched) == directories
    identity = common.triton_identity(switched)
    assert [json.loads(stamp.read_text()) for stamp in stamps] == [identity] * 2
    # Its own directories again, filled: the same build and switch state.
    (inductor / "fxgraph").mkdir()
    _toy_triton_cache(triton_cache, "KEY", "k")
    assert common.claim_caches(switched) == directories
    # Any other build or switch state is refused there, switched or not.
    for other in ("fbtriton", "stock", "tileir"):
        with pytest.raises(common.TritonSwitchError, match="another Triton build"):
            common.claim_caches(_toy_backend(other))
    with pytest.raises(common.TritonSwitchError, match="another Triton build"):
        common.claim_caches(
            _toy_backend("fbtriton+meta_ws", environment={"TRITON_DEBUG": "1"})
        )
    # Without TRITON_CACHE_DIR Inductor keeps Triton's cache inside its own.
    monkeypatch.delenv("TRITON_CACHE_DIR")
    assert common.claim_caches(switched) == {"TORCHINDUCTOR_CACHE_DIR": str(inductor)}
    nested = tmp_path / "nested"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(nested))
    (nested / "triton" / "0" / "DRIVER").mkdir(parents=True)
    (nested / "triton" / "0" / "DRIVER" / "cuda_utils.so").write_text("")
    common.claim_caches(_toy_backend("tileir"))
    (nested / common.CACHE_STAMP).unlink()
    _toy_triton_cache(nested / "triton" / "0", "KEY", "k")
    with pytest.raises(common.TritonSwitchError, match="names no Triton build"):
        common.claim_caches(_toy_backend("tileir"))
    # Benches started at once on one fresh directory: a process leaves its
    # stamp only where none stands, and is held against the one that does.
    raced = tmp_path / "raced"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(raced))
    # (The file that another process is leaving its stamp from is no kernel.)
    raced.mkdir()
    beside_file = raced / f"{common.CACHE_STAMP}.1.tmp"
    beside_file.write_text("{")
    common.claim_caches(_toy_backend("tileir"))
    beside_file.unlink()
    (raced / common.CACHE_STAMP).unlink()
    left: list[dict[str, object] | None] = []

    def beside(variable: str, path: Path) -> bool:
        """Another process leaves its stamp while this one looks."""
        path.mkdir(parents=True, exist_ok=True)
        (path / common.CACHE_STAMP).write_text(json.dumps(left[-1]))
        return False

    monkeypatch.setattr(common, "_holds_kernels", beside)
    left.append(common.triton_identity(_toy_backend("tileir")))
    common.claim_caches(_toy_backend("tileir"))
    (raced / common.CACHE_STAMP).unlink()
    left.append(identity)
    with pytest.raises(common.TritonSwitchError, match="another Triton build"):
        common.claim_caches(_toy_backend("tileir"))
    assert json.loads((raced / common.CACHE_STAMP).read_text()) == identity
    assert [entry.name for entry in raced.iterdir()] == [common.CACHE_STAMP]


def test_a_launched_kernel_must_be_the_builds_own_through_its_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What Triton's cache holds of four launched binaries: fbtriton's with
    # and without its switch, a Tile IR kernel and stock Triton's.
    fb = "3.9.0.dev20261113+fb.git305df91c"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    _toy_triton_cache(
        tmp_path, "WS1", "k", triton_version=fb, TRITON_USE_META_WS="true"
    )
    _toy_triton_cache(tmp_path, "WS0", "k", triton_version=fb)
    _toy_triton_cache(tmp_path, "TILE", "k", backend="tileir", triton_version="3.9.0")
    _toy_triton_cache(tmp_path, "PTX", "k", triton_version="3.9.0")
    assert common.compiled_artifact(0, "WS1") == {
        "backend": "cuda",
        "triton_version": fb,
        "meta_ws": "true",
        "kinds": ["cubin", "ptx"],
    }
    assert common.compiled_artifact(0, "TILE") == {
        "backend": "tileir",
        "triton_version": "3.9.0",
        "meta_ws": None,
        "kinds": ["cubin", "tileir", "ttir"],
    }
    assert common.compiled_artifact(0, "ABSENT") is None

    def launched(*keys: str) -> list[dict[str, object]]:
        """A compile's kernels by the cache keys of their binaries; the
        last has not settled on one launcher."""
        configs = [
            {"name": f"triton_{key}", "artifact": common.compiled_artifact(0, key)}
            for key in keys
        ]
        *settled, last = configs
        return [*settled, {"name": last["name"], "launchers": [last]}]

    switched, off = _toy_backend("fbtriton+meta_ws"), _toy_backend("fbtriton")
    tile, ptx = _toy_backend("tileir"), _toy_backend("tileir-fork-ptx")
    stock = _toy_backend()
    assert common.artifact_issue(None, launched("WS0")) is None
    assert common.artifact_issue(switched, []) is None
    for backend, keys in (
        (switched, ("WS1", "WS1")),
        (off, ("WS0", "WS0")),
        (tile, ("TILE",)),
        (ptx, ("PTX",)),
        (stock, ("PTX", "ABSENT")),
    ):
        assert common.artifact_issue(backend, launched(*keys)) is None
    # The cached kernel of a run without the switch, as Inductor's
    # FX-graph cache hands it back; of another Triton; of the PTX path.
    assert "without TRITON_USE_META_WS" in common.artifact_issue(
        switched, launched("WS1", "WS0")
    )
    assert "Triton 3.9.0 compiled" in common.artifact_issue(switched, launched("PTX"))
    assert f"Triton {fb} compiled" in common.artifact_issue(stock, launched("WS0"))
    assert "not through Tile IR" in common.artifact_issue(tile, launched("PTX"))
    # The other way round: a run without a switch, stock Triton or the
    # control of a baseline, that launches a kernel compiled under one.
    assert "compiled under TRITON_USE_META_WS=true, which is not in effect" in (
        common.artifact_issue(off, launched("WS0", "WS1"))
    )
    for backend in (ptx, stock):
        issue = common.artifact_issue(backend, launched("PTX", "TILE"))
        assert issue is not None and issue.startswith(
            "triton_TILE launches a binary compiled through Tile IR, and "
            "ENABLE_TILE is not in effect"
        )
    # With a switch in effect, a binary the cache cannot tell is no proof.
    for backend in (switched, tile):
        issue = common.artifact_issue(backend, launched("ABSENT"))
        assert issue is not None and "holds no metadata" in issue
    # Stock Triton 3.9.0 and the Tile IR fork's main both call themselves
    # 3.9.0, and Triton's metadata of a kernel names no other version: the
    # build is told by Inductor's hash of it, which every kernel it
    # generates carries and this process computes for its own.
    ours, theirs = "A" * 64, "B" * 64
    asked: list[str] = []
    monkeypatch.setattr(
        common, "inductor_backend_hash", lambda: asked.append(ours) or ours
    )

    def generated(*hashes: str | None) -> list[dict[str, object]]:
        return [
            {"backend_hash": named, **kernel}
            for named, kernel in zip(hashes, launched("PTX", "PTX"), strict=True)
        ]

    for backend in (stock, ptx):
        assert common.artifact_issue(backend, generated(ours, ours)) is None
        issue = common.artifact_issue(backend, generated(ours, theirs))
        assert issue == (
            "triton_PTX was generated for another Triton build or backend than "
            "this process's triton 3.9.0+gitb7d (Inductor's backend_hash "
            "BBBBBBBBBBBB, here AAAAAAAAAAAA)"
        )
    # A kernel of an Inductor that names no build is not compared, and the
    # process's own is asked (of the device's driver) only for one that does.
    assert common.artifact_issue(stock, generated(None, ours)) is None
    asked.clear()
    assert common.artifact_issue(stock, generated(None, None)) is None
    assert common.artifact_issue(stock, launched("PTX")) is None and not asked


def test_a_compile_that_launches_another_switch_states_kernel_is_not_timed(
    tie_order_repro: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fbtriton with its switch on, in a process whose coordinate-descent
    # compile came back from Inductor's cache with the binary of a run
    # without the switch: that compile is an error, the default one counts.
    # Its max-autotune compile came back with a kernel that Inductor
    # generated for another Triton build: an error as well.
    fb = "3.9.0.dev20261113+fb.git305df91c"
    compiled = {"backend": "cuda", "triton_version": fb, "kinds": ["cubin", "ptx"]}
    ours = "5DC21BC92D29" + "0" * 52
    kernels = [
        {"backend_hash": ours, "artifact": {**compiled, "meta_ws": "true"}},
        {"backend_hash": ours, "artifact": {**compiled, "meta_ws": None}},
        {"backend_hash": "37875C9E2A70", "artifact": {**compiled, "meta_ws": "true"}},
    ]
    monkeypatch.setattr(
        common,
        "inductor_kernel_configs",
        lambda launched: [{"name": "triton_k", **kernels.pop(0)}],
    )
    monkeypatch.setattr(common, "inductor_backend_hash", lambda: ours)
    accounting = types.SimpleNamespace(count_bytes_effective=lambda *args: 24)
    monkeypatch.setitem(sys.modules, "byte_accounting", accounting)
    monkeypatch.setattr(bench, "_replayed", lambda call, fresh_inputs: call)
    monkeypatch.setattr(common, "foreign_gpu_processes", list)
    monkeypatch.setattr(common, "gpu_temperature", lambda: None)
    timed: list[str] = []

    def time_impls(
        fns: dict[str, Callable[[], object]], **rounds: int
    ) -> dict[str, dict[str, float]]:
        timed.extend(fns)
        return {impl: {"bb_us": 2.0, "cold_us": 1.0} for impl in fns}

    monkeypatch.setattr(bench, "time_impls", time_impls)
    args = argparse.Namespace(
        repro_id="r",
        shape="s",
        impls="inductor,inductor_cd,inductor_max",
        configs_dir=tmp_path,
        rounds=2,
        rep_ms=1,
        cold_rounds=2,
        cold_timer="events",
        rotate_inputs=0,
        placement_seed=None,
        fixed_placement=True,
    )
    record = bench.run(args, _toy_backend("fbtriton+meta_ws"))
    assert record["status"] == "ok" and timed == ["inductor"]
    assert record["triton_backend"]["name"] == "fbtriton+meta_ws"
    entries = record["impls"]
    assert entries["inductor"]["status"] == "ok"
    assert entries["inductor"]["inductor_configs"][0]["artifact"]["meta_ws"] == "true"
    tuned = entries["inductor_cd"]
    assert tuned["status"] == "error" and "cold_us" not in tuned
    assert (
        tuned["error"]
        .rstrip()
        .endswith(
            "TritonSwitchError: triton_k launches a binary compiled without "
            "TRITON_USE_META_WS"
        )
    )
    # The entry still names what it launched.
    assert tuned["inductor_configs"][0]["artifact"]["meta_ws"] is None
    foreign = entries["inductor_max"]
    assert foreign["status"] == "error" and "cold_us" not in foreign
    assert (
        foreign["error"]
        .rstrip()
        .endswith(
            "TritonSwitchError: triton_k was generated for another Triton build "
            "or backend than this process's fbtriton 3.9.0.dev20261113 "
            "(Inductor's backend_hash 37875C9E2A70, here 5DC21BC92D29)"
        )
    )
    assert foreign["inductor_configs"][0]["backend_hash"] == "37875C9E2A70"
    # A default compile that is refused anchors nothing: the compile after
    # it launches its own build's kernel and is not judged by the other
    # switch state's, so nothing of the process is timed.
    kernels[:] = [
        {"backend_hash": ours, "artifact": {**compiled, "meta_ws": None}},
        {"backend_hash": ours, "artifact": {**compiled, "meta_ws": "true"}},
    ]
    timed.clear()
    args.impls = "inductor,inductor_cd"
    entries = bench.run(args, _toy_backend("fbtriton+meta_ws"))["impls"]
    assert timed == [] and entries["inductor"]["status"] == "error"
    after = entries["inductor_cd"]
    assert after["status"] == "error" and "cold_us" not in after
    assert after["inductor_configs"][0]["artifact"]["meta_ws"] == "true"
    assert (
        after["error"]
        .rstrip()
        .endswith(
            "TritonSwitchError: the numerics gate has no anchor: the default "
            "Inductor compile was refused (triton_k launches a binary compiled "
            "without TRITON_USE_META_WS)"
        )
    )
    # The anchor is held to its build before the gate is first built on
    # it, whatever the listing: with the default compile left out, or
    # listed behind the compile that needs the gate first. The tuned
    # compile's own kernel is this build's each time.
    no_anchor = after["error"].rstrip().splitlines()[-1]
    for listing in ("inductor_cd", "inductor_cd,inductor"):
        kernels[:] = [
            {"backend_hash": ours, "artifact": {**compiled, "meta_ws": "true"}},
            {"backend_hash": ours, "artifact": {**compiled, "meta_ws": None}},
            {"backend_hash": ours, "artifact": {**compiled, "meta_ws": None}},
        ]
        timed.clear()
        args.impls = listing
        entries = bench.run(args, _toy_backend("fbtriton+meta_ws"))["impls"]
        assert timed == [] and list(entries) == listing.split(",")
        assert entries["inductor_cd"]["status"] == "error"
        assert entries["inductor_cd"]["error"].rstrip().splitlines()[-1] == no_anchor
        if "inductor" in entries:
            assert (
                entries["inductor"]["error"]
                .rstrip()
                .endswith(
                    "TritonSwitchError: triton_k launches a binary compiled without "
                    "TRITON_USE_META_WS"
                )
            )
    # An anchor of this build: the compile is judged by it and timed.
    kernels[:] = [
        {"backend_hash": ours, "artifact": {**compiled, "meta_ws": "true"}}
    ] * 2
    timed.clear()
    args.impls = "inductor_cd"
    entries = bench.run(args, _toy_backend("fbtriton+meta_ws"))["impls"]
    assert timed == ["inductor_cd"] and entries["inductor_cd"]["status"] == "ok"
    assert kernels == []


def test_a_bench_fails_before_it_compiles_when_its_switch_cannot_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # TRITON_USE_META_WS=1 with stock Triton installed: the bench would
    # time stock kernels as fbtriton's. It ends before run, with a record
    # that says why it has no timings.
    _switches(monkeypatch, TRITON_USE_META_WS="1")
    monkeypatch.setattr(common, "triton_backend", _toy_backend)
    monkeypatch.setattr(common, "versions", lambda triton=None: {})
    monkeypatch.setattr(common, "gpu_info", lambda: {"uuid": "GPU-a"})
    monkeypatch.setattr(
        bench, "run", lambda args, backend: pytest.fail("the bench ran")
    )
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    out = tmp_path / "record.json"
    argv = [
        "bench.py",
        "r",
        "--shape",
        "s1",
        "--configs-dir",
        "c",
        "--output",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        bench.main()
    assert exit_info.value.code == 1
    record = json.loads(out.read_text())
    assert record["status"] == "error" and record["impls"] == {}
    assert (record["repro"], record["shape"]) == ("r", "s1")
    assert record["gpu"] == {"uuid": "GPU-a"} and record["versions"] == {}
    assert record["error"].startswith("TritonSwitchError: TRITON_USE_META_WS=1 is set")
    assert record["triton_backend"]["name"] == "stock"
    assert "no Meta warp specialization" in capsys.readouterr().err
    # Without --shape the record names the point a bench would have run.
    monkeypatch.setattr(common, "default_shape", lambda repro: f"{repro}-default")
    monkeypatch.setattr(sys, "argv", [*argv[:2], *argv[4:]])
    with pytest.raises(SystemExit):
        bench.main()
    assert json.loads(out.read_text())["shape"] == "r-default"
    monkeypatch.setattr(sys, "argv", argv)
    # So does a bench in a cache directory that another switch state filled.
    monkeypatch.setattr(
        common, "triton_backend", lambda: _toy_backend("fbtriton+meta_ws")
    )
    (tmp_path / "inductor" / "fxgraph").mkdir(parents=True)
    with pytest.raises(SystemExit):
        bench.main()
    assert "is not empty" in json.loads(out.read_text())["error"]
    # In directories of its own it runs, and the record names them. The
    # build is asked once, and the run gets what the checks went by.
    (tmp_path / "inductor" / "fxgraph").rmdir()
    asked: list[dict[str, object]] = []
    ran: list[object] = []

    def triton_backend() -> dict[str, object]:
        asked.append(_toy_backend("fbtriton+meta_ws"))
        return asked[-1]

    def run(args: object, backend: object) -> dict[str, object]:
        ran.append(backend)
        return {"status": "ok", "shape": "s1", "impls": {}}

    monkeypatch.setattr(common, "triton_backend", triton_backend)
    monkeypatch.setattr(bench, "run", run)
    monkeypatch.setenv("BB_FOREIGN_ACTIVITY", "0")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    bench.main()
    assert json.loads(out.read_text())["cache_directories"] == {
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor")
    }
    assert len(asked) == 1 and ran[0] is asked[0]


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
    # The events cold_us times single replays from 64 us for the fastest
    # implementation (e232's memcopy takes 65.5 us): batches would not be its
    # cross-check.
    assert reason(128_000_000, fastest_us=41.0) == "applies"
    assert reason(201_000_000, fastest_us=63.9) == "applies"
    assert reason(201_000_000, fastest_us=65.5) == (
        "the fastest implementation, memcopy, takes 65.5 us a call: from 64 us "
        "the events cold_us times single replays, not the batches of flushed "
        "calls that rot_us would time"
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


def test_a_trace_cold_round_times_each_replay_behind_the_pin_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --cold-timer trace: one kernel-timer window of COLD_REPEAT flushed calls
    # per replay, every flush followed by the prelude that releases pinned
    # L2 lines, so nothing is subtracted.
    first, second = (lambda: None), (lambda: None)
    readings = {first: 0.004, second: 0.006}
    seen: list[dict[str, object]] = []

    def fake_do_bench(fn: Callable[[], object], **kwargs: object) -> float:
        seen.append(kwargs)
        return readings[fn]

    monkeypatch.setattr(bench, "benchmark_do_bench", fake_do_bench)
    prelude = object.__new__(bench.ReleaseL2Pins)
    assert bench.cold_trace_round([second, first], prelude) == [0.006, 0.004]
    window = {
        "fixed_repetitions": bench.COLD_REPEAT,
        "return_mode": "median",
        "l2_flush_mode": "call",
        "trace_kernel_durations": True,
        "prelude": prelude,
    }
    assert seen == [window, window]


@pytest.mark.cuda
@skipIfNotCUDA()
def test_a_trace_cold_round_reads_a_graph_from_the_kernel_trace() -> None:
    source = torch.randn(1 << 20, device=DEVICE)
    out = torch.empty_like(source)
    graph, _ = common.capture_graph(lambda: torch.add(source, 1.0, out=out))
    prelude = bench.cold_trace_prelude()
    (ms,) = bench.cold_trace_round([graph.replay], prelude)
    # a 4 MB add reads 2-4 us on a B200 or GB300; anything near zero would
    # be a reading with the flush or the sweep subtracted
    assert 0.0005 < ms < 0.1


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
        # (a record made before autotune.py named a regime: tuned alone)
        "autotune_regime": "alone",
        "autotune_regimes": {},
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


def _digest_record(whole: str, nosrc: str | None = None) -> dict[str, Any]:
    """An ok autotune record of one kernel with these digests of its code:
    of the whole text and, unless the record is one made before records
    held it (None), of the text without source-line comments."""
    kernel = {"decorator": "@helion.kernel(version=1)", "code_sha256": whole}
    if nosrc is not None:
        kernel["code_nosrc_sha256"] = nosrc
    return {"status": "ok", "kernels": {"k": kernel}}


def test_a_bench_entry_is_of_a_record_with_its_code_but_for_source_comments() -> None:
    tuned = _digest_record("whole-1", "nosrc-1")
    info = common.timed_record_fields(tuned)
    assert info["autotune_code_sha256"] == {"k": "whole-1"}
    assert info["autotune_code_nosrc_sha256"] == {"k": "nosrc-1"}
    assert common.timed_on_record(info, tuned)
    # The port autotuned again after an edit of its comments, to the same
    # statements: the record holds the kernels the bench timed. After an
    # edit of its statements it does not.
    comments = _digest_record("whole-2", "nosrc-1")
    statements = _digest_record("whole-3", "nosrc-3")
    assert common.timed_on_record(info, comments)
    assert report.timed_record_issue(info, comments) is None
    assert not common.timed_on_record(info, statements)
    assert report.timed_record_issue(info, statements) == (
        "replaced: autotuned again after the bench"
    )
    # (the second digest decides alone where both hold it)
    assert not common.timed_on_record(info, _digest_record("whole-1", "nosrc-3"))
    # A record made before the second digest holds the first alone, and so
    # does the entry of a bench that pinned it. The whole text decides for
    # them as it did, also against an entry or a record that holds both.
    before = _digest_record("whole-1")
    named = common.timed_record_fields(before)
    assert named["autotune_code_sha256"] == {"k": "whole-1"}
    assert "autotune_code_nosrc_sha256" not in named
    assert common.timed_on_record(named, before)
    assert common.timed_on_record(named, tuned)
    assert not common.timed_on_record(named, comments)
    assert common.timed_on_record(info, before)
    assert not common.timed_on_record(info, _digest_record("whole-2"))
    # A record in which one kernel lacks the second digest (the harness
    # writes none; one put together by hand) is not one that holds it: the
    # entry names the digests of the whole text alone, and those decide.
    mixed = _digest_record("whole-1", "nosrc-1")
    edited = _digest_record("whole-2", "nosrc-1")
    for record in (mixed, edited):
        record["kernels"]["j"] = before["kernels"]["k"]
    partly = common.timed_record_fields(mixed)
    assert partly["autotune_code_sha256"] == {"k": "whole-1", "j": "whole-1"}
    assert "autotune_code_nosrc_sha256" not in partly
    assert common.timed_on_record(partly, mixed)
    assert not common.timed_on_record(partly, edited)


def test_only_source_comments_between_statements_leave_the_digested_code() -> None:
    quote = "# src[port.py:12]: y = x + 1"
    # The lines of a module, and which of them are source-line comments:
    # those that begin as the printer's do and stand between two logical
    # lines, where they have their line to themselves.
    lines = [
        (f"{quote}\n", True),
        ("import torch\n", False),
        ("\n", False),
        ('QUOTED = """\n', False),
        (f"{quote}\n", False),  # a line of a string
        ('"""\n', False),
        ("\n", False),
        ("def kernel(x):\n", False),
        (f"    {quote}\n", True),
        ("    # src[port.py:12-19]: ...\n", True),
        (f"    y = x + 1  {quote}\n", False),  # behind code
        ("    # a note that quotes nothing\n", False),
        (f"    # as the port has it: {quote}\n", False),  # begins otherwise
        ("    #src[port.py:12]: y = x + 1\n", False),
        ("    if y:\n", False),
        ("        # src[unknown]: [source unavailable]\n", True),
        ("        y = (\n", False),
        (f"            {quote}\n", False),  # inside brackets
        ("            y)\n", False),
        ('    z = f"""{\n', False),
        (f"{quote}\n", False),  # in a field, whose value has its line break
        ('y = }"""\n', False),
        (f"    {quote}\n", True),
        ("    return z\n", False),
        ("    \\\n", False),
        (f"{quote}\n", False),  # joined to the backslash's line
        (f"{quote}\n", True),
        ("kernel(1)\n", False),
        (quote, True),  # the module's last line, without a line break
    ]
    code = "".join(line for line, _ in lines)
    kept = "".join(line for line, comment in lines if not comment)
    assert common.without_source_comments(code) == kept
    assert common.without_source_comments(kept) == kept
    # The same lines with CRLF ends are told apart the same way. A carriage
    # return of its own ends a line for Python and not for the lines read
    # here, where the comment's line would take the statement behind it:
    # such a text stays whole.
    crlf = code.replace("\n", "\r\n")
    assert common.without_source_comments(crlf) == kept.replace("\n", "\r\n")
    alone = f"x = 1\n{quote}\rimport os\n"
    assert [type(node) for node in ast.parse(alone).body] == [ast.Assign, ast.Import]
    assert common.without_source_comments(alone) == alone
    if sys.version_info >= (3, 12):
        # (a comment in an f-string's field parses from Python 3.12 on)
        program = ast.dump(ast.parse(code))
        assert ast.dump(ast.parse(kept)) == program
        # The lines that stay at the start of a line are of the program:
        # without the one in the string, the one in the field (a line break
        # of z's value) or the one behind the backslash (which would join
        # the call to the kernel's body), the module is another.
        for index, entry in enumerate(lines):
            if entry == (f"{quote}\n", False):
                less = "".join(
                    line
                    for at, (line, comment) in enumerate(lines)
                    if not comment and at != index
                )
                assert ast.dump(ast.parse(less)) != program


_TOY_PORT = "scale_000000000001"
_TOY_PORT_SOURCE = '''"""A toy port: one kernel, which doubles its input."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def scale_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)  # the row, in fp32
        out[tile_m, :] = (row * 2.0).to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    return (scale_rows(inputs[0]),)
'''
# The port as a porter edits it: a reworded comment in the kernel, a line
# added above the kernel (the number of every line below it moves) and a
# statement that computes something else.
_TOY_PORT_EDITS = {
    "unedited": _TOY_PORT_SOURCE,
    "comment": _TOY_PORT_SOURCE.replace("# the row, in fp32", "# each row, as fp32"),
    "inserted": _TOY_PORT_SOURCE.replace(
        "@helion.kernel(", "# Inductor's one kernel.\n@helion.kernel("
    ),
    "statement": _TOY_PORT_SOURCE.replace("row * 2.0", "row * 3.0"),
}
# Where the toy port is rendered: under a backend the harness has ports for.
# (Not ``skipUnlessBackends``, which takes ``tileir`` for ``triton``: the
# harness loads no port on that backend.)
_harness_backend = pytest.mark.skipif(
    _get_backend() not in ("triton", "cute"),
    reason="the harness's ports are for the Triton and CuTe backends",
)


@pytest.fixture
def toy_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, tuple[types.ModuleType, str]]:
    """The toy port under each of ``_TOY_PORT_EDITS`` on the lane's backend:
    the port with its kernel bound and set to its default config on this CPU
    process (what a call of ``forward`` leaves, the launch aside), and the
    kernel's code as the printer writes it without source-line comments."""
    backend = _get_backend()
    if reason := codegen_skip_reason(backend):
        pytest.skip(reason)
    ports = {}
    with cpu_render_target():
        for edit, source in _TOY_PORT_EDITS.items():
            # A directory for each edit: the printer reads the lines it
            # quotes from the port's file, which linecache keeps by path.
            kernels = tmp_path / edit
            kernels.mkdir()
            (kernels / f"{_TOY_PORT}.py").write_text(source)
            monkeypatch.setattr(common, "KERNELS_DIR", kernels)
            port = common.load_port(_TOY_PORT, backend)
            rows = torch.zeros(8, 64, dtype=torch.bfloat16)
            bound = port.scale_rows.bind((rows,))
            # Setting the config renders it and loads the module, which
            # takes no GPU. The backend would name its compile cache in the
            # environment first, for every test after this one too.
            monkeypatch.setattr(
                bound.env.backend, "setup_compile_cache_dir", lambda device: None
            )
            bound.set_config(bound.config_spec.default_config())
            plain = bound.to_code(bound._config, output_origin_lines=False)
            ports[edit] = (port, plain)
    return ports


# One test for the pins and the digests: the port is rendered once under
# each edit.
@_harness_backend
@skipIfRefEagerMark("ref eager mode generates no code")
def test_a_pin_holds_through_comment_edits_unless_its_record_has_the_whole_text_only(
    toy_ports: dict[str, tuple[types.ModuleType, str]],
) -> None:
    ports = {edit: port for edit, (port, _) in toy_ports.items()}
    recorded = common.bound_kernel_records(ports["unedited"])
    # The record as autotune.py made it before the second digest.
    before = {
        name: {
            field: value
            for field, value in kernel.items()
            if field != "code_nosrc_sha256"
        }
        for name, kernel in recorded.items()
    }
    held = {
        edit: (common.pin_roundtrip(port, recorded), common.pin_roundtrip(port, before))
        for edit, port in ports.items()
    }
    yes, no = {"scale_rows": True}, {"scale_rows": False}
    assert held == {
        "unedited": (yes, yes),
        "comment": (yes, no),
        "inserted": (yes, no),
        "statement": (no, no),
    }
    (kernel,) = recorded.values()
    assert list(kernel) == ["decorator", "config", "code_sha256", "code_nosrc_sha256"]
    assert common.pinning_digest(kernel) == "code_nosrc_sha256"
    assert common.pinning_digest(before["scale_rows"]) == "code_sha256"
    # (which decides alone: the entry's digest of the whole text still matches)
    other = {"scale_rows": {**kernel, "code_nosrc_sha256": ""}}
    assert common.pin_roundtrip(ports["unedited"], other) == no
    # A kernel the port no longer has generates no code to compare.
    assert common.pin_roundtrip(ports["unedited"], {"gone": kernel}) == {"gone": False}

    whole, nosrc = {}, {}
    for edit, (port, plain) in toy_ports.items():
        bound = common._bound_kernel("scale_rows", port.scale_rows)
        code = common.generated_code(bound)
        # The code that was compiled quotes the port. Without those lines it
        # is, to the character, the code the printer writes without them.
        assert f"{common.SOURCE_COMMENT}{_TOY_PORT}.py:" in code
        assert common.without_source_comments(code) == plain
        (digests,) = common.code_digests(port).values()
        whole[edit] = digests["code_sha256"]
        nosrc[edit] = digests["code_nosrc_sha256"]
        assert whole[edit] == hashlib.sha256(code.encode()).hexdigest()
        assert nosrc[edit] == hashlib.sha256(plain.encode()).hexdigest()
    # Every edit changes the text: the line it quotes, the numbers of the
    # lines it quotes, or a statement.
    assert len(set(whole.values())) == len(_TOY_PORT_EDITS)
    # Only the changed statement changes the code that runs.
    assert nosrc["comment"] == nosrc["inserted"] == nosrc["unedited"]
    assert nosrc["statement"] != nosrc["unedited"]


def _rotation_args(count: int = 8, cold_timer: str = "events") -> object:
    return types.SimpleNamespace(
        rotate_inputs=count,
        rotate_max_l2=bench.ROTATE_MAX_L2_MULTIPLE,
        cold_rounds=6,
        cold_timer=cold_timer,
    )


def test_the_trace_timer_takes_no_rotating_pass() -> None:
    # rot_us is the events timer's regime, gated by that timer's batch
    # threshold: a record timed by the trace has no events cold_us for it to
    # cross-check, so the pass is skipped, saying why.
    found: dict[str, Any] = {"impls": {"inductor": {"cold_us": 9.5}}}
    bench.rotate_inputs(_rotation_args(cold_timer="trace"), found, {}, 7_000_000)
    assert found["rotate_inputs"]["status"] == "skipped"
    assert "the trace timer's" in found["rotate_inputs"]["skipped"]


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
    triton: str | None = "stock",
    switches: dict[str, str] | None = None,
) -> tuple[Path, list[dict[str, object]]]:
    """Run ``campaign.py`` with stub jobs on GPUs 0 and 1 (or ``gpu_source``)
    into ``tmp_path / "run"`` (nvidia-smi stubbed: each GPU's UUID
    ``GPU-<i>`` and none busy, unless ``gpu_uuid`` and ``busy_uuids`` say
    otherwise); the run directory and this run's events.

    The campaign sees the Triton build ``triton`` (a ``_toy_backend`` by
    name; None for an environment without a Triton) and that build's
    switches in the environment, or ``switches``: what the lane's own
    environment sets (``ENABLE_TILE=1`` on CI's Tile IR job) does not make
    a toy run directory a baseline's."""
    build = None if triton is None else _toy_backend(triton)
    monkeypatch.setattr(common, "triton_backend", lambda: build)
    if switches is None:
        asked = (
            {} if build is None else cast("dict[str, str | None]", build["switches"])
        )
        switches = {name: value for name, value in asked.items() if value}
    _switches(monkeypatch, **switches)
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
    assert result["triton_regime"] == "alone"
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
    named = plan_run.plan(selection, [run5, run3], pin_check=False, regime="port")
    assert named["triton_regime"] == "port" and calls == []
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
    # The plan names the regime it reused Triton records for (unnamed:
    # Triton's default), and a campaign that would tune Triton ports in
    # another one is refused: the copies would stand for the wrong regime.
    assert planned["triton_regime"] == "alone"
    with pytest.raises(SystemExit):
        _stub_campaign(
            tmp_path,
            monkeypatch,
            {},
            "--regime",
            "port",
            kernels=selection["kernels"],
            backends="cute,triton",
        )
    assert "reused earlier Triton records for the regime 'alone'" in (
        capsys.readouterr().err
    )
    # (a campaign that tunes no Triton port takes any regime: the CuTe
    # phase of a run planned in two phases)
    _stub_campaign(
        tmp_path,
        monkeypatch,
        {},
        "--regime",
        "port",
        kernels=selection["kernels"],
        backends="cute",
    )
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


def test_a_campaigns_jobs_inherit_the_switch_and_get_caches_of_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A baseline's environment script exports the switch and names cache
    # directories; every job compiles under the switch, in its own.
    _switches(monkeypatch, TRITON_USE_META_WS="1")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/shared/triton")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/shared/inductor")
    monkeypatch.setenv("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
    monkeypatch.setenv("HELION_BACKEND", "tileir")
    env = campaign.job_env("3", tmp_path, 7, "marker")
    assert env["TRITON_USE_META_WS"] == "1"
    assert env["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] == "0"
    assert env["TRITON_CACHE_DIR"] == str(tmp_path / "triton")
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor")
    assert "HELION_BACKEND" not in env and env["CUDA_VISIBLE_DEVICES"] == "3"


@pytest.mark.slow
@_requires_proc
def test_a_baseline_run_keeps_a_run_directory_to_one_triton_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The main run's CuTe pick, benched again under fbtriton with its
    # switch on, in two processes, into a run directory of its own.
    main = tmp_path / "main"
    _toy_tuned(main, "sum_000000000001", "s1", "cute")
    impls = "inductor,inductor_cd,inductor_max,helion_cute,memcopy"
    flags = ("--configs-from", str(main / "configs"), "--phases", "bench")
    flags += ("--bench-repeats", "2", "--bench-impls", impls)

    def baseline(triton: str) -> tuple[Path, list[dict[str, object]]]:
        return _stub_campaign(
            tmp_path, monkeypatch, {}, *flags, backends="cute", triton=triton
        )

    run, events = baseline("fbtriton+meta_ws")
    start = next(e for e in events if e["kind"] == "campaign_start")
    assert start["triton_backend"]["name"] == "fbtriton+meta_ws"
    assert start["python"] == sys.executable
    stamp = run / campaign.RUN_BACKEND
    assert json.loads(stamp.read_text()) == common.triton_identity(
        _toy_backend("fbtriton+meta_ws")
    )
    aggregate = json.loads((run / "bench" / "sum_000000000001__s1.json").read_text())
    assert list(aggregate["impls"]) == impls.split(",")
    assert aggregate["impls"]["helion_cute"]["status"] == "ok"
    kept = {path: path.stat().st_mtime_ns for path in run.rglob("*") if path.is_file()}
    # Started again in that environment, nothing is left to run.
    _, events = baseline("fbtriton+meta_ws")
    assert not [e for e in events if e["kind"] == "job_start"]
    # Started again from a shell without the baseline's environment (stock
    # Triton, no switch), or with the switch off, it would add stock
    # records to the baseline's: it stops before it changes anything.
    for other in ("stock", "fbtriton"):
        with pytest.raises(SystemExit):
            baseline(other)
        assert "another Triton build or switch state" in capsys.readouterr().err
    events_log = run / "campaign.jsonl"
    assert {
        path: path.stat().st_mtime_ns
        for path in run.rglob("*")
        if path.is_file() and path != events_log
    } == {path: mtime for path, mtime in kept.items() if path != events_log}
    # A campaign of another build that started in a fresh directory beside
    # this one left its stamp first: this one stops, and that stamp stands.
    (tmp_path / "raced").mkdir()
    theirs = common.triton_identity(_toy_backend("stock"))
    leave = common.leave_stamp

    def beside(stamp: Path, identity: dict[str, object]) -> None:
        leave(stamp, theirs)
        leave(stamp, identity)

    monkeypatch.setattr(common, "leave_stamp", beside)
    with pytest.raises(SystemExit):
        _stub_campaign(
            tmp_path / "raced",
            monkeypatch,
            {},
            *flags,
            backends="cute",
            triton="fbtriton+meta_ws",
        )
    assert "another Triton build or switch state" in capsys.readouterr().err
    raced = tmp_path / "raced" / "run"
    assert json.loads((raced / campaign.RUN_BACKEND).read_text()) == theirs
    assert not list((raced / "bench").iterdir())
    monkeypatch.setattr(common, "leave_stamp", leave)
    # A switch that the interpreter's Triton cannot put in effect stops a
    # campaign before it makes its run directory.
    (tmp_path / "tile").mkdir()
    with pytest.raises(SystemExit):
        _stub_campaign(
            tmp_path / "tile",
            monkeypatch,
            {},
            "--phases",
            "bench",
            switches={"ENABLE_TILE": "1"},
        )
    assert "no Tile IR backend" in capsys.readouterr().err
    assert not (tmp_path / "tile" / "run").exists()


@pytest.mark.slow
@_requires_proc
def test_a_baseline_without_helion_benches_the_listing_without_the_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An environment in which Helion+CuTe cannot run: no picks are copied
    # and the benches time the main run's listing without the port.
    impls = "inductor,inductor_cd,inductor_max,oracle,memcopy"
    flags = ("--phases", "bench", "--bench-impls", impls)

    def bench(triton: str | None) -> list[dict[str, object]]:
        _, events = _stub_campaign(
            tmp_path, monkeypatch, {}, *flags, backends="cute", triton=triton
        )
        return events

    tile = "tileir"
    run = tmp_path / "run"
    assert [e["job"] for e in bench(tile) if e["kind"] == "job_start"] == [
        "bench:sum_000000000001__s1"
    ]
    record = json.loads((run / "bench" / "sum_000000000001__s1.json").read_text())
    assert list(record["impls"]) == impls.split(",")
    assert list((run / "configs").iterdir()) == []
    # The record stands: a restart benches nothing.
    assert not [e for e in bench(tile) if e["kind"] == "job_start"]
    # A run directory with bench records that names no build (made before
    # the campaign named it) is a stock run's: a baseline takes another,
    # and so does a baseline's control with its switch off.
    (run / campaign.RUN_BACKEND).unlink()
    for other in (tile, "fbtriton+meta_ws", "fbtriton", "tileir-fork-ptx"):
        with pytest.raises(SystemExit):
            bench(other)
        assert "does not name the Triton build" in capsys.readouterr().err
        assert not (run / campaign.RUN_BACKEND).exists()
    # A campaign without a switch goes on in it and names its build there;
    # so does one in an environment without a Triton, which restarts.
    for stock in ("stock", None):
        (run / campaign.RUN_BACKEND).unlink(missing_ok=True)
        for _start in range(2):
            bench(stock)
        assert json.loads((run / campaign.RUN_BACKEND).read_text()) == (
            {} if stock is None else common.triton_identity(_toy_backend(stock))
        )
    # A record that names its build (a bench by hand, or a run whose stamp
    # went) says whose it is: only that build's campaign goes on beside it.
    (run / campaign.RUN_BACKEND).unlink()
    control = _toy_backend("fbtriton")
    _add_fields(run, "sum_000000000001", "s1", triton_backend=control)
    for other in ("stock", None, tile, "fbtriton+meta_ws"):
        with pytest.raises(SystemExit):
            bench(other)
        error = capsys.readouterr().err
        assert (
            "__s1.json is of another Triton build or switch state (fbtriton: " in error
        )
        assert not (run / campaign.RUN_BACKEND).exists()
    assert not [e for e in bench("fbtriton") if e["kind"] == "job_start"]
    assert json.loads(
        (run / campaign.RUN_BACKEND).read_text()
    ) == common.triton_identity(control)


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
            "--impls inductor,inductor_cd,inductor_max,oracle,helion_triton,memcopy "
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
    every = "inductor,inductor_cd,inductor_max,oracle,helion_triton,helion_cute,memcopy"
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
    # Nor does it start helion.inductor's own search (inductor_helion), which
    # no campaign times, nor bench the pinned helion.inductor code, which is
    # CuTe's, without cute among --backends.
    with pytest.raises(SystemExit):
        start(second, "triton", "--bench-impls", "inductor,inductor_helion")
    assert "names inductor_helion: a campaign job clears" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        start(second, "triton", "--bench-impls", "inductor,inductor_helion_pretuned")
    error = capsys.readouterr().err
    assert "names inductor_helion_pretuned, the pinned helion.inductor code" in error
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
    job = argparse.Namespace(repro_id=point[0], shape="s1", backend="cute", regime=None)
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
    """The kernels of a toy autotune record, as records held them before
    ``code_nosrc_sha256``: with the digest of the whole text only, by which
    the harness goes on comparing such records (``_digest_record`` makes
    the other kind)."""
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
    # The pinned helion.inductor code, timed at s2 only.
    times.update(inductor_helion_pretuned=45.0)
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
    assert every["helion_inductor_vs_inductor"] == pytest.approx(90 / 45)
    assert every["helion_inductor_vs_cute"] == pytest.approx(60 / 45)
    assert every["helion_inductor_vs_cute_kernels"] == 1
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
    # The pinned helion.inductor code's time, speedup and numerics verdict.
    assert "| 120.00 | 0.750 | 45.00 | 2.000 | 80.00 |" in text
    # The opt-in cold compile only where the bench timed it (here nowhere).
    assert "cute=ok,pretuned=ok |" in text
    assert "cold=" not in text
    assert (
        "| pinned helion.inductor vs Inductor, geomean (kernels) "
        "| 2.000x (1) | 2.000x (1) |"
    ) in text
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
        if line.startswith("| ") and len(cells) == 17:
            del cells[6]
        kept.append(" | ".join(cells))
    kept = "\n".join(kept).replace(f"|{'---|' * 17}", f"|{'---|' * 16}", 1) + "\n"
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


def _toy_process(
    run: Path,
    repeat: int,
    times: dict[str, float],
    max_sm: int | None = 0,
    shape: str = "s1",
    **impls: dict[str, object],
) -> dict[str, object]:
    """One process's record of a repeated bench of sum_000000000001
    (``_toy_bench``'s, as ``<point>.r<repeat>.json``), each Inductor compile
    naming a pick of that process; the record."""
    picks = {
        impl: {
            "inductor_configs": [
                {
                    "name": "triton_k",
                    "block_sizes": {"XBLOCK": 2**repeat},
                    "num_warps": 4,
                    "num_stages": 1,
                }
            ],
            **impls.get(impl, {}),
        }
        for impl in common.INDUCTOR_IMPLS
        if impl in times
    }
    _toy_bench(run, "sum_000000000001", shape, times, max_sm, **{**impls, **picks})
    plain = common.bench_record_path(run / "bench", "sum_000000000001", shape)
    path = common.bench_repeat_path(run / "bench", "sum_000000000001", shape, repeat)
    plain.replace(path)
    return json.loads(path.read_text())


def _three_processes(run: Path, **impls: dict[str, object]) -> dict[str, object]:
    """Three processes of a repeated bench whose Inductor picks differ: the
    faster compile is another one in each (80, 108 and 110 us), and the
    port's time differs too (100, 120 and 100 us), so the per-process
    ratios are 0.8, 0.9 and 1.1 while the medians give 108 / 100; their
    records by file name."""
    times = [(100.0, 80.0, 100.0), (108.0, 130.0, 120.0), (130.0, 110.0, 100.0)]
    return {
        f"sum_000000000001__s1.r{repeat}.json": _toy_process(
            run,
            repeat,
            {"inductor": default, "inductor_cd": tuned, "helion_cute": port},
            **impls,
        )
        for repeat, (default, tuned, port) in enumerate(times, start=1)
    }


def test_an_aggregate_keeps_an_implementation_every_process_skipped(
    tmp_path: Path,
) -> None:
    # bench.py skips the pinned helion.inductor code at a point without a
    # file, so in every process of a repeated bench: skipped, not an error.
    processes = _three_processes(tmp_path)
    for record in processes.values():
        record["impls"]["inductor_helion_pretuned"] = {  # type: ignore[index]
            "status": "skipped",
            "skipped": "no pinned helion.inductor code at x.py",
        }
    names = list(processes)
    record = common.aggregate_bench_records(processes, dict.fromkeys(names, ""))  # type: ignore[arg-type]
    pinned = record["impls"]["inductor_helion_pretuned"]
    assert pinned["status"] == "skipped" and "error" not in pinned
    assert record["status"] == "ok"


def test_an_aggregate_record_takes_medians_and_keeps_every_process(
    tmp_path: Path,
) -> None:
    processes = _three_processes(tmp_path)
    names = list(processes)
    digests = {name: f"sha-{name}" for name in names}
    record = common.aggregate_bench_records(processes, digests)  # type: ignore[arg-type]
    assert record["status"] == "ok" and "contaminated" not in record
    assert record["aggregate"]["processes"] == 3
    assert record["aggregate"]["counted"] == names
    assert record["aggregate"]["records"] == digests
    cd = record["impls"]["inductor_cd"]
    # The median, the process nearest to it, and every process's values.
    assert cd["cold_us"] == 110.0 and cd["bb_us"] == 220.0
    assert cd["representative"] == names[2]
    assert cd["per_process"]["cold_us"] == [80.0, 130.0, 110.0]
    assert cd["per_process"]["status"] == ["ok"] * 3
    assert [
        picks[0]["block_sizes"] for picks in cd["per_process"]["inductor_configs"]
    ] == [{"XBLOCK": 2}, {"XBLOCK": 4}, {"XBLOCK": 8}]
    assert cd["inductor_configs"][0]["block_sizes"] == {"XBLOCK": 8}
    assert cd["status"] == "ok" and cd["numerics_pass"] is True
    port = record["impls"]["helion_cute"]
    assert port["autotune_code_sha256"] == {"k": "cute-1"}
    # The samples of the three jobs read as one job's.
    assert record["foreign_gpu_activity"]["samples"] == 27
    assert common.activity_issue(record) is None

    # What one process does not reproduce is not the point's result: a
    # failed gate there fails the implementation, a process without it
    # leaves it without a timing.
    failing = dict(processes)
    failing[names[1]] = _toy_process(
        tmp_path,
        2,
        {"inductor": 90.0, "inductor_cd": 120.0, "helion_cute": 100.0},
        helion_cute={"numerics_pass": False},
        inductor_cd={"status": "error", "cold_us": None, "bb_us": None},
    )
    record = common.aggregate_bench_records(failing, digests)  # type: ignore[arg-type]
    assert record["impls"]["helion_cute"]["numerics_pass"] is False
    cd = record["impls"]["inductor_cd"]
    assert cd["status"] == "error" and names[1] in cd["error"]
    assert "cold_us" not in cd and cd["per_process"]["cold_us"] == [80.0, None, 110.0]

    # A contaminated process is left out; when every one is, they all count
    # and the aggregate is flagged like a job's cleanest attempt.
    dirty = dict(processes)
    dirty[names[0]] = _toy_process(
        tmp_path, 1, {"inductor": 1.0, "inductor_cd": 1.0, "helion_cute": 100.0}, 70
    )
    record = common.aggregate_bench_records(dirty, digests)  # type: ignore[arg-type]
    assert record["aggregate"]["counted"] == names[1:]
    assert "contaminated" in record["aggregate"]["left_out"][names[0]]
    assert record["impls"]["inductor_cd"]["cold_us"] == 120.0
    record = common.aggregate_bench_records({names[0]: dirty[names[0]]}, digests)  # type: ignore[arg-type]
    assert record["contaminated"] is True and record["aggregate"]["processes"] == 1
    assert common.activity_issue(record) is not None


def test_an_aggregates_samples_cover_it_when_each_processs_cover_that_process(
    tmp_path: Path,
) -> None:
    # Each process of a repeated bench has a job's allowance of failed
    # samples, here two of its nine. Added up, six of 27 are more than one
    # job may lose, and the aggregate still counts, as each process does.
    processes = cast("dict[str, dict[str, Any]]", _three_processes(tmp_path))
    names = list(processes)
    digests = dict.fromkeys(names, "sha")
    for process in processes.values():
        lost = common.MAX_UNKNOWN_SAMPLES
        process["foreign_gpu_activity"]["unknown_samples"] = lost
        assert common.activity_issue(process) is None
    record = common.aggregate_bench_records(processes, digests)
    assert record["aggregate"]["left_out"] == {}
    activity = record["foreign_gpu_activity"]
    assert (activity["samples"], activity["unknown_samples"]) == (27, 6)
    assert activity["unknown_allowed"] == 6
    assert common.activity_issue(record) is None
    row = report.summarize(record, "cold_us")
    assert "bench" not in row["issues"] and "inductor" in row["counted"]
    assert row["cute_vs_inductor"] == pytest.approx(0.9)
    # A process that lost more than a job may is left out, and takes its
    # allowance with it.
    processes[names[0]]["foreign_gpu_activity"]["unknown_samples"] = 3
    record = common.aggregate_bench_records(processes, digests)
    assert record["aggregate"]["left_out"] == {names[0]: "3 of 9 samples failed"}
    assert record["foreign_gpu_activity"]["unknown_allowed"] == 4
    assert common.activity_issue(record) is None
    # One job's samples keep one job's allowance.
    assert common.activity_coverage_issue(processes[names[0]]) is not None


# The kernel profile of a port whose kernels' spans overlap (a pick with
# dependent launches), as bench.py records it.
_OVERLAPPING = {
    "kernel_us": 156.0,
    "kernel_span_us": 206.0,
    "kernel_spans_overlap": True,
    "kernels": [{"name": "k", "us": 16.0, "span_us": 66.0, "per_call": 1.0}],
}
_FAILED = {"status": "error", "error": "out of memory", "cold_us": None, "bb_us": None}


def test_an_aggregates_overlap_flag_goes_with_its_two_kernel_totals(
    tmp_path: Path,
) -> None:
    # kernel_spans_overlap says that kernel_us and kernel_span_us differ:
    # in an aggregate it is of the two medians, and an implementation that
    # one process did not profile has neither total nor the flag (whichever
    # process the rest of its entry is taken from).
    times = {"inductor": 100.0, "inductor_cd": 90.0, "helion_cute": 95.0}

    def aggregated(*ports: dict[str, object]) -> dict[str, object]:
        processes = {
            f"sum_000000000001__s1.r{repeat}.json": _toy_process(
                tmp_path, repeat, times, helion_cute=port
            )
            for repeat, port in enumerate(ports, start=1)
        }
        digests = dict.fromkeys(processes, "sha")
        record = common.aggregate_bench_records(processes, digests)  # type: ignore[arg-type]
        return record["impls"]["helion_cute"]

    port = aggregated(_OVERLAPPING, {**_OVERLAPPING, "kernel_us": 150.0}, _OVERLAPPING)
    assert port["kernel_us"] == 156.0 and port["kernel_span_us"] == 206.0
    assert port["kernel_spans_overlap"] is True
    # Spans that one process read as overlapping by a hair: the medians do
    # not.
    hair = {**_OVERLAPPING, "kernel_us": 203.0}
    level = {**hair, "kernel_us": 205.0, "kernel_spans_overlap": False}
    assert aggregated(hair, level, level)["kernel_spans_overlap"] is False
    # Failed in the second process, or in the first: no totals, no flag.
    for ports in (
        (_OVERLAPPING, _FAILED, _OVERLAPPING),
        (_FAILED, _OVERLAPPING, _OVERLAPPING),
    ):
        port = aggregated(*ports)
        assert port["status"] == "error"
        assert not {"kernel_us", "kernel_span_us", "kernel_spans_overlap"} & set(port)
        assert port["per_process"]["kernel_us"].count(156.0) == 2


def test_the_report_takes_a_repeated_benchs_ratio_process_by_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gate's rule: in each process the faster of Inductor's two compiles
    # over the port's time (80 / 100, 108 / 120, 110 / 100), then the median
    # of those ratios, 0.9. The medians of the implementations would give
    # 108 / 100 = 1.08, a pass: no process timed that pair.
    run = tmp_path / "run"
    processes = _three_processes(run)
    digests = dict.fromkeys(processes, "sha")
    aggregate = common.aggregate_bench_records(processes, digests)  # type: ignore[arg-type]
    out = common.bench_record_path(run / "bench", "sum_000000000001", "s1")
    out.write_text(json.dumps(aggregate))
    _toy_tuned(run, "sum_000000000001", "s1", "cute")
    # Another point, benched in one process, beside it.
    _toy_bench(run, "sum_000000000001", "s2", {"inductor": 50.0, "helion_cute": 25.0})
    _toy_tuned(run, "sum_000000000001", "s2", "cute")
    goal = tmp_path / "goal.json"
    monkeypatch.setattr(sys, "argv", ["report.py", str(run), "--goal", str(goal)])
    report.main()
    result = json.loads((run / "report.json").read_text())
    rows = {row["shape"]: row for row in result["rows"]["cold_us"]}
    # The per-process records are no rows of their own.
    assert sorted(rows) == ["s1", "s2"]
    row = rows["s1"]
    assert row["processes"] == 3 and row["counted"] == ["inductor", "helion_cute"]
    assert row["inductor"] == 108.0 and row["helion_cute"] == 100.0
    assert row["cute_vs_inductor"] == pytest.approx(0.9)
    assert row["cute_vs_inductor_per_process"] == pytest.approx([0.8, 0.9, 1.1])
    assert row["inductor_missing"] == {} and row["overlapping_spans"] == {}
    assert rows["s2"]["processes"] == 1 and rows["s2"]["cute_vs_inductor"] == 2.0
    aggregates = result["aggregates"]["cold_us"]["all kernels"]
    assert aggregates["cute_vs_inductor"] == pytest.approx(math.sqrt(0.9 * 2.0))
    assert aggregates["time_weighted"] == pytest.approx(158.0 / 125.0)
    shapes = {s["name"]: s for s in json.loads(goal.read_text())["shapes"]}
    assert shapes["sum_000000000001[s1]"]["ratio"] == pytest.approx(0.9)
    assert shapes["sum_000000000001[s1]"]["artifact"].endswith("__s1.json")
    text = (run / "report.md").read_text()
    assert "1 of the 2 bench records are aggregates over 3 bench processes" in text
    assert "| 0.900 | 0.800..1.100 (3) |" in text and "| 2.000 | - |" in text
    assert "## Inductor compiles missing" not in text
    # Each compile's picks, with the processes that ended on them.
    assert (
        "- sum_000000000001[s1] inductor_cd (1 of 3 processes): "
        "triton_k: XBLOCK=8 warps=4 stages=1\n"
    ) in text
    # --rebalance recomputes rounds, which only the processes' records hold.
    monkeypatch.setattr(sys, "argv", ["report.py", str(run), "--rebalance"])
    report.main()
    again = json.loads((run / "report.json").read_text())
    assert again["rows"]["cold_us"] == result["rows"]["cold_us"]

    # A port that failed the gate in one process is out of every comparison.
    failed = dict(processes)
    name = "sum_000000000001__s1.r2.json"
    failed[name] = _toy_process(
        run,
        2,
        {"inductor": 90.0, "inductor_cd": 120.0, "helion_cute": 100.0},
        helion_cute={"numerics_pass": False},
    )
    out.write_text(json.dumps(common.aggregate_bench_records(failed, digests)))  # type: ignore[arg-type]
    monkeypatch.setattr(sys, "argv", ["report.py", str(run)])
    report.main()
    result = json.loads((run / "report.json").read_text())
    row = next(row for row in result["rows"]["cold_us"] if row["shape"] == "s1")
    assert row["helion_cute"] is None and row["cute_vs_inductor"] is None
    assert row["numerics"]["helion_cute"] == "FAIL" and row["inductor"] == 90.0

    # A port with overlapping spans that failed in one process has no kernel
    # totals in the aggregate: the report leaves it out and goes on.
    failed[name] = _toy_process(
        run,
        2,
        {"inductor": 90.0, "inductor_cd": 120.0, "helion_cute": 100.0},
        helion_cute=_FAILED,
    )
    for other in (1, 3):
        failed[f"sum_000000000001__s1.r{other}.json"]["impls"]["helion_cute"].update(  # type: ignore[index]
            _OVERLAPPING
        )
    aggregate = common.aggregate_bench_records(failed, digests)  # type: ignore[arg-type]
    # Nor does a record that has the flag without the totals stop it.
    flagged = json.loads(json.dumps(aggregate))
    flagged["impls"]["helion_cute"]["kernel_spans_overlap"] = True
    for record in (aggregate, flagged):
        out.write_text(json.dumps(record))
        report.main()
        result = json.loads((run / "report.json").read_text())
        row = next(row for row in result["rows"]["cold_us"] if row["shape"] == "s1")
        assert row["helion_cute"] is None and row["overlapping_spans"] == {}
        assert "## Dependent launches" not in (run / "report.md").read_text()

    # A repeated bench without a clean process is listed and left out.
    dirty = {name: _toy_process(run, 1, {"inductor": 90.0, "helion_cute": 100.0}, 70)}
    out.write_text(json.dumps(common.aggregate_bench_records(dirty, digests)))  # type: ignore[arg-type]
    report.main()
    result = json.loads((run / "report.json").read_text())
    row = next(row for row in result["rows"]["cold_us"] if row["shape"] == "s1")
    assert row["counted"] == [] and "contaminated" in row["issues"]["bench"]
    assert result["contaminated"]["bench"]["sum_000000000001__s1"][-1] == (
        "all 1 bench processes"
    )


def test_the_report_shows_the_debiased_baseline_beside_the_headline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two processes whose two compiles ended on one code (_toy_process: one
    # pick per process) and read 100 us but for noise that falls the other
    # way for each (98 and 102 us): the headline takes the luckier compile,
    # the de-biased baseline the two together. The port takes 100 us. By
    # rot_us the two compiles read alike, 99 us.
    run = tmp_path / "run"
    lucky = [98.0, 98.0, 100.0, 100.0, 96.0, 96.0]
    unlucky = [200.0 - value for value in lucky]
    rotated = {
        "rot_us": 99.0,
        "rot_order_medians_us": [99.0, 99.0],
        "rot_runs_us": [99.0] * 6,
    }
    processes = {
        f"sum_000000000001__s1.r{repeat}.json": _toy_process(
            run,
            repeat,
            {"inductor": 98.0, "inductor_cd": 102.0, "helion_cute": 100.0},
            inductor={
                "cold_order_medians_us": [98.0, 98.0],
                "cold_runs_us": lucky,
                **rotated,
            },
            inductor_cd={
                "cold_order_medians_us": [102.0, 102.0],
                "cold_runs_us": unlucky,
                **rotated,
            },
            helion_cute={"rot_us": 110.0},
        )
        for repeat in (1, 2)
    }
    digests = {
        name: hashlib.sha256((run / "bench" / name).read_bytes()).hexdigest()
        for name in processes
    }
    aggregate = common.aggregate_bench_records(processes, digests)  # type: ignore[arg-type]
    both = {"headline": 98.0, "debiased": pytest.approx(100.0)}
    same = [["inductor", "inductor_cd"]]
    alike = {"headline": 99.0, "debiased": 99.0}
    assert (
        aggregate["fastest_inductor_per_process"]
        == [{"same_code": same, "cold_us": both, "rot_us": alike}] * 2
    )
    assert "fastest_inductor" not in aggregate
    row = report.summarize(aggregate, "cold_us")
    assert row["inductor"] == 98.0 and row["inductor_debiased"] == pytest.approx(100.0)
    assert row["cute_vs_inductor"] == pytest.approx(0.98)
    assert row["cute_vs_inductor_debiased"] == pytest.approx(1.0)
    # A record of one process made before bench.py stored the figure holds
    # the rounds to take it from; bb_us has none.
    single = dict(processes["sum_000000000001__s1.r1.json"])
    assert "fastest_inductor" not in single
    assert report.summarize(single, "cold_us")["cute_vs_inductor_debiased"] == (
        pytest.approx(1.0)
    )
    assert report.summarize(single, "bb_us")["cute_vs_inductor_debiased"] is None
    # So does an aggregate made before, from its processes' records beside
    # it, while they are the records it was made of: the report shows both
    # speedups and says how many lie apart.
    del aggregate["fastest_inductor_per_process"]
    out = common.bench_record_path(run / "bench", "sum_000000000001", "s1")
    out.write_text(json.dumps(aggregate))
    _toy_tuned(run, "sum_000000000001", "s1", "cute")
    assert report.debiased_baseline(aggregate, "cold_us") == [None, None]
    # A process's record that was written again since holds other readings
    # than the headline's: no figure from it.
    again = run / "bench" / "sum_000000000001__s1.r2.json"
    kept = again.read_bytes()
    again.write_bytes(kept + b"\n")
    report.with_fastest_inductor(aggregate, out)
    assert "fastest_inductor_per_process" not in aggregate
    again.unlink()
    report.with_fastest_inductor(aggregate, out)
    assert report.debiased_baseline(aggregate, "cold_us") == [None, None]
    again.write_bytes(kept)
    report.with_fastest_inductor(aggregate, out)
    assert report.debiased_baseline(aggregate, "cold_us") == [pytest.approx(100.0)] * 2
    monkeypatch.setattr(sys, "argv", ["report.py", str(run)])
    report.main()
    text = (run / "report.md").read_text()
    assert (
        "| inductor | helion-cute | cute/ind | cute/ind de-biased "
        "| cute/ind per process (n) | cute/ind rot | cute/ind rot de-biased |"
    ) in text
    assert (
        "| 98.00 | 100.00 | 0.980 | 1.000 | 0.980..0.980 (2) | 0.900 | 0.900 |"
    ) in text
    assert (
        "CuTe/ind geomean 1.000 over 1 kernels, 0.980 by the headline over the "
        "same; 1 of them differ by more than 0.5%."
    ) in text
    assert (
        "The same by `rot_us`: CuTe/ind geomean 0.900 over 1 kernels, 0.900 by "
        "the headline over the same; 0 of them differ by more than 0.5%."
    ) in text
    # The bb table of the same report has no such column.
    assert text.count("| cute/ind de-biased |") == 1


def test_an_inductor_compile_that_failed_in_one_process_leaves_only_that_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Coordinate descent is the faster compile (80 us against 100 us) and
    # failed in the second of three processes; the port takes 85 us. Each
    # process's baseline is the faster of the compiles usable in it: 80,
    # 100, 80 us, ratios 0.941, 1.176, 0.941, median 0.941. Leaving the
    # compile out of every process would give 1.176, a pass.
    run = tmp_path / "run"
    times = {"inductor": 100.0, "inductor_cd": 80.0, "helion_cute": 85.0}
    goal = tmp_path / "goal.json"
    monkeypatch.setattr(sys, "argv", ["report.py", str(run), "--goal", str(goal)])
    _toy_tuned(run, "sum_000000000001", "s1", "cute")
    out = common.bench_record_path(run / "bench", "sum_000000000001", "s1")
    names = [f"sum_000000000001__s1.r{repeat}.json" for repeat in (1, 2, 3)]

    def reported(**second: dict[str, object]) -> tuple[dict[str, object], str]:
        processes = {
            name: _toy_process(run, repeat, times, **(second if repeat == 2 else {}))
            for repeat, name in enumerate(names, start=1)
        }
        digests = dict.fromkeys(processes, "sha")
        out.write_text(json.dumps(common.aggregate_bench_records(processes, digests)))  # type: ignore[arg-type]
        report.main()
        (row,) = json.loads((run / "report.json").read_text())["rows"]["cold_us"]
        return row, (run / "report.md").read_text()

    for failed in (_FAILED, {"numerics_pass": False}):
        row, text = reported(inductor_cd=failed)
        assert row["cute_vs_inductor_per_process"] == pytest.approx(
            [80 / 85, 100 / 85, 80 / 85]
        )
        assert row["cute_vs_inductor"] == pytest.approx(80 / 85)
        assert row["inductor"] == 80.0 and row["counted"] == ["inductor", "helion_cute"]
        assert row["inductor_missing"] == {"inductor_cd": [names[1]]}
        (shape,) = json.loads(goal.read_text())["shapes"]
        assert shape["ratio"] == pytest.approx(80 / 85)
        assert (
            "- sum_000000000001[s1] inductor_cd: not usable in 1 of 3 bench "
            f"processes ({names[1]})\n"
        ) in text
    # The same for the default compile where it is the faster one.
    times = {"inductor": 80.0, "inductor_cd": 100.0, "helion_cute": 85.0}
    row, _ = reported(inductor=_FAILED)
    assert row["cute_vs_inductor"] == pytest.approx(80 / 85)
    assert row["inductor_missing"] == {"inductor": [names[1]]}
    # A process without either compile has no baseline, and the point no
    # ratio.
    row, _ = reported(inductor=_FAILED, inductor_cd=_FAILED)
    assert row["inductor"] is None and row["cute_vs_inductor"] is None
    assert "inductor" not in row["counted"]


def test_the_report_says_which_processes_name_no_inductor_kernel() -> None:
    # A compile that failed in a process of a repeated bench has no picks
    # there: the section counts that process and says so.
    picks = [
        {"name": "triton_k", "block_sizes": {"XBLOCK": 2}}
        | {"num_warps": 4, "num_stages": 1}
    ]
    info = {"per_process": {"inductor_configs": [picks, None, picks]}}
    row = {
        "repro": "p",
        "shape": "s",
        "inductor_configs": {"inductor_max": report.inductor_picks(info)},
    }
    assert report.inductor_config_lines([row]) == [
        "- p[s] inductor_max (2 of 3 processes): triton_k: XBLOCK=2 warps=4 stages=1",
        "- p[s] inductor_max (1 of 3 processes): no kernel recorded",
    ]
    # One process, as a plain bench records a compile without Triton kernels.
    row["inductor_configs"] = {
        "inductor": report.inductor_picks({"inductor_configs": []})
    }
    assert report.inductor_config_lines([row]) == [
        "- p[s] inductor: no kernel recorded"
    ]


def test_a_campaign_queues_every_repeat_and_aggregates_the_finished_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernels = [{"repro": "sum_000000000001", "shape": "s1"}]
    run = tmp_path / "run"
    _toy_tuned(run, "sum_000000000001", "s1", "cute")
    flags = ("--phases", "bench", "--bench-repeats", "3")
    # Every repeat is a job of its own.
    events = _stopped_campaign(tmp_path, monkeypatch, kernels, "cute", *flags)
    assert events[-2]["jobs"] == 3
    assert not any(event["kind"] == "bench_aggregate" for event in events)
    # Two finished: the third is queued, and the point has no aggregate yet.
    times = {"inductor": 100.0, "helion_cute": 50.0}
    for repeat in (1, 2):
        _toy_process(run, repeat, times)
    events = _stopped_campaign(tmp_path, monkeypatch, kernels, "cute", *flags)
    assert events[-2]["jobs"] == 1
    out = common.bench_record_path(run / "bench", "sum_000000000001", "s1")
    assert not out.exists()
    # All finished (by an earlier start): nothing to run, and the aggregate
    # is written from the records, once.
    _toy_process(run, 3, times)
    events = _stopped_campaign(tmp_path, monkeypatch, kernels, "cute", *flags)
    assert events[-2]["jobs"] == 0
    assert [event["kind"] for event in events[:-2]] == ["bench_aggregate"]
    assert events[0]["processes"] == [
        f"sum_000000000001__s1.r{k}.json" for k in (1, 2, 3)
    ]
    first = json.loads(out.read_text())
    assert first["aggregate"]["processes"] == 3
    events = _stopped_campaign(tmp_path, monkeypatch, kernels, "cute", *flags)
    assert [event["kind"] for event in events] == ["campaign_start", "campaign_end"]
    # Another count of repeats: the aggregate is of those, the earlier one
    # kept aside.
    events = _stopped_campaign(
        tmp_path,
        monkeypatch,
        kernels,
        "cute",
        "--phases",
        "bench",
        "--bench-repeats",
        "2",
    )
    assert events[0]["kind"] == "bench_aggregate" and events[-2]["jobs"] == 0
    assert json.loads(out.read_text())["aggregate"]["processes"] == 2
    assert json.loads(out.with_name(f"{out.name}.prev").read_text()) == first
    # Without the option the aggregate stands for the point's bench.
    events = _stopped_campaign(
        tmp_path, monkeypatch, kernels, "cute", "--phases", "bench"
    )
    assert events[-2]["jobs"] == 0
    with pytest.raises(SystemExit):
        _stopped_campaign(
            tmp_path, monkeypatch, kernels, "cute", "--bench-repeats", "0"
        )


@pytest.mark.slow
@_requires_proc
def test_a_campaign_benches_each_repeat_in_a_process_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point = "sum_000000000001__s1"
    names = [f"{point}.r{k}.json" for k in (1, 2, 3)]
    # The second repeat is contaminated once: it runs again, and the
    # aggregate waits for the clean record.
    run, events = _stub_campaign(
        tmp_path, monkeypatch, {names[1]: [70, 0]}, "--bench-repeats", "3"
    )
    started = [e["job"] for e in events if e["kind"] == "job_start"]
    assert sorted(started) == sorted(
        [f"autotune:{point}__triton", *(f"bench:{point}#r{k}" for k in (1, 2, 2, 3))]
    )
    aggregated = [e for e in events if e["kind"] == "bench_aggregate"]
    assert len(aggregated) == 1 and aggregated[0]["processes"] == names
    assert aggregated[0]["missing"] == [] and events[-2] == aggregated[0]
    out = run / "bench" / f"{point}.json"
    record = json.loads(out.read_text())
    assert record["aggregate"]["processes"] == 3
    assert record["aggregate"]["records"] == {
        name: hashlib.sha256((run / "bench" / name).read_bytes()).hexdigest()
        for name in names
    }
    assert (run / "bench" / f"{names[1]}.attempt1").exists()
    # Each process had a cache directory, and with it an Inductor cache, of
    # its own.
    caches = [
        json.loads((run / "bench" / name).read_text())["env"]["HELION_CACHE_DIR"]
        for name in names
    ]
    assert len(set(caches)) == 3
    # Started again, nothing runs and the aggregate stands.
    _, events = _stub_campaign(tmp_path, monkeypatch, {}, "--bench-repeats", "3")
    assert [e["kind"] for e in events] == ["campaign_start", "campaign_end"]
    assert json.loads(out.read_text()) == record
    # Benched again, with a repeat that fails: while they run the point has
    # no aggregate that passes for finished, and the new one is of the
    # repeats that are ok.
    for name in names:
        (tmp_path / f"{name}.count").unlink()
    _, events = _stub_campaign(
        tmp_path, monkeypatch, {names[2]: [None]}, "--bench-repeats", "3", "--rebench"
    )
    aggregated = [e for e in events if e["kind"] == "bench_aggregate"]
    assert len(aggregated) == 1 and aggregated[0]["processes"] == names[:2]
    assert aggregated[0]["missing"] == [names[2]]
    assert json.loads(out.read_text())["aggregate"]["processes"] == 2
    assert json.loads(out.with_name(f"{out.name}.prev").read_text()) == record


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


def test_the_max_autotune_compile_takes_the_modes_options_without_its_graph_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``inductor_max`` compiles with what ``mode="max-autotune"`` sets, as
    PyTorch lists it, minus the CUDA-graph wrapper, on top of the point's
    compile policy; the report compares against the fastest Inductor compile
    a record holds."""
    import torch._inductor

    mode = torch._inductor.list_mode_options("max-autotune")
    assert mode["max_autotune"] is True
    assert mode["coordinate_descent_tuning"] is True
    options = common.inductor_options("inductor_max")
    assert options == {k: v for k, v in mode.items() if k != "triton.cudagraphs"}
    assert options == torch._inductor.list_mode_options("max-autotune-no-cudagraphs")
    assert common.inductor_options("inductor") == {}
    assert common.inductor_options("inductor_cd") == {"coordinate_descent_tuning": True}
    with pytest.raises(ValueError, match="not an Inductor implementation"):
        common.inductor_options("oracle")

    calls: list[dict[str, object]] = []
    harness = types.SimpleNamespace(
        compile_policy_from_config=lambda config: {"inductor": {"some_option": 1}},
        compile_repro=lambda instance, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "repro_harness", harness)
    monkeypatch.setattr(common, "bb_path", lambda: None)
    monkeypatch.setattr(common, "shape_configs", lambda repro_id: {"s": {}})
    for impl in common.INDUCTOR_IMPLS:
        common.compile_inductor("r", "s", object(), impl=impl)
    policy = {"inductor": {"some_option": 1}}
    assert calls == [
        {"compile_policy": policy},
        {"compile_policy": policy, "options": {"coordinate_descent_tuning": True}},
        {"compile_policy": policy, "options": options},
    ]
    assert common.ALL_IMPLS[:3] == common.INDUCTOR_IMPLS
    assert set(common.INDUCTOR_IMPLS) <= set(bench.GATED_IMPLS)

    record = {
        "repro": "sum_000000000001",
        "shape": "s1",
        "status": "ok",
        "impls": {
            "inductor": {"status": "ok", "cold_us": 100.0},
            "inductor_cd": {"status": "ok", "cold_us": 90.0},
            "inductor_max": {"status": "ok", "cold_us": 80.0},
            "helion_cute": {"status": "ok", "cold_us": 60.0},
        },
    }
    times = [
        report.impl_time(record, impl, "cold_us") for impl in common.INDUCTOR_IMPLS
    ]
    assert times == [100.0, 90.0, 80.0]


def test_the_max_autotune_compile_counts_in_each_process_of_a_repeated_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Three processes: the default compile takes 100 us, coordinate descent
    # 90 us and the port 85 us in each; max-autotune takes 80 and 84 us and
    # failed in the third. Each process's baseline is the fastest compile
    # usable in it (80, 84, 90 us), so the ratios are 0.941, 0.988 and
    # 1.059 and the point's is 0.988. Without max-autotune every process
    # would read 1.059, a pass.
    run = tmp_path / "run"
    names = [f"sum_000000000001__s1.r{repeat}.json" for repeat in (1, 2, 3)]
    processes = {
        name: _toy_process(
            run,
            repeat,
            {
                "inductor": 100.0,
                "inductor_cd": 90.0,
                "inductor_max": fastest,
                "helion_cute": 85.0,
            },
            **({"inductor_max": _FAILED} if repeat == 3 else {}),
        )
        for repeat, (name, fastest) in enumerate(
            zip(names, (80.0, 84.0, 70.0), strict=True), start=1
        )
    }
    digests = dict.fromkeys(processes, "sha")
    aggregate = common.aggregate_bench_records(processes, digests)  # type: ignore[arg-type]
    # The aggregate keeps the compile's picks and timings of every process.
    tuned = aggregate["impls"]["inductor_max"]
    assert tuned["status"] == "error" and "cold_us" not in tuned
    assert tuned["per_process"]["status"] == ["ok", "ok", "error"]
    assert tuned["per_process"]["cold_us"] == [80.0, 84.0, None]
    assert [
        picks[0]["block_sizes"] for picks in tuned["per_process"]["inductor_configs"]
    ] == [{"XBLOCK": 2}, {"XBLOCK": 4}, {"XBLOCK": 8}]
    out = common.bench_record_path(run / "bench", "sum_000000000001", "s1")
    out.write_text(json.dumps(aggregate))
    _toy_tuned(run, "sum_000000000001", "s1", "cute")
    # A single bench beside it, where max-autotune is the fastest compile.
    _toy_bench(
        run,
        "sum_000000000001",
        "s2",
        {
            "inductor": 100.0,
            "inductor_cd": 90.0,
            "inductor_max": 80.0,
            "helion_cute": 60.0,
        },
    )
    _toy_tuned(run, "sum_000000000001", "s2", "cute")
    goal = tmp_path / "goal.json"
    monkeypatch.setattr(sys, "argv", ["report.py", str(run), "--goal", str(goal)])
    report.main()
    result = json.loads((run / "report.json").read_text())
    rows = {row["shape"]: row for row in result["rows"]["cold_us"]}
    row = rows["s1"]
    assert row["cute_vs_inductor_per_process"] == pytest.approx(
        [80 / 85, 84 / 85, 90 / 85]
    )
    assert row["cute_vs_inductor"] == pytest.approx(84 / 85)
    assert row["inductor"] == 84.0 and row["counted"] == ["inductor", "helion_cute"]
    assert row["inductor_missing"] == {"inductor_max": [names[2]]}
    assert list(row["inductor_configs"]) == list(common.INDUCTOR_IMPLS)
    assert len(row["inductor_configs"]["inductor_max"]) == 3
    assert list(row["numerics"])[:3] == list(common.INDUCTOR_IMPLS)
    assert list(row["kernels"])[:3] == list(common.INDUCTOR_IMPLS)
    assert rows["s2"]["inductor"] == 80.0
    assert rows["s2"]["cute_vs_inductor"] == pytest.approx(80 / 60)
    shapes = {s["name"]: s["ratio"] for s in json.loads(goal.read_text())["shapes"]}
    assert shapes["sum_000000000001[s1]"] == pytest.approx(84 / 85)
    assert shapes["sum_000000000001[s2]"] == pytest.approx(80 / 60)
    text = (run / "report.md").read_text()
    assert (
        "- sum_000000000001[s1] inductor_max (1 of 3 processes): "
        "triton_k: XBLOCK=4 warps=4 stages=1\n"
    ) in text
    assert (
        "- sum_000000000001[s1] inductor_max: not usable in 1 of 3 bench "
        f"processes ({names[2]})\n"
    ) in text


def test_a_runs_report_names_the_triton_build_of_its_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    _toy_bench(run, "sum_000000000001", "s1", {"inductor": 2.0, "helion_cute": 1.0})
    _toy_tuned(run, "sum_000000000001", "s1", "cute")
    monkeypatch.setattr(sys, "argv", ["report.py", str(run)])
    report.main()
    # A record made before bench.py named the build: the version alone.
    assert "torch None, triton None, cutlass-dsl" in capsys.readouterr().out
    _add_fields(
        run,
        "sum_000000000001",
        "s1",
        versions={"triton": "3.9.0.dev20261113"},
        triton_backend=_toy_backend("fbtriton+meta_ws"),
    )
    report.main()
    assert (
        "triton 3.9.0.dev20261113 (fbtriton+meta_ws: fbtriton 3.9.0.dev20261113, "
        "TRITON_USE_META_WS=1), cutlass-dsl"
    ) in capsys.readouterr().out
    assert common.triton_label(_toy_backend("tileir")) == (
        "tileir: triton 3.9.0+gitb7d, ENABLE_TILE=1, Inductor's launcher off"
    )
