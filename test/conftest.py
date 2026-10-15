from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import Any
import warnings

from _pytest._io.saferepr import saferepr
import psutil
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# The subtest parameters a worker sent as their repr (see below).
_REPR_KWARGS_KEY = "_helion.subtest_repr_kwargs"
_EXECNET_SCALARS = (type(None), bool, float, complex, str, bytes)


def _execnet_serializable(value: object) -> bool:
    """Whether execnet, which dispatches on the exact type, can send ``value``."""
    if type(value) is int:
        # execnet packs every int up to 2**31 - 1 as a signed 4-byte value, so
        # one below -2**31 raised struct.error (e.g. a subTest(begin=-2**31 - 4)).
        return value >= -(2**31)
    if type(value) in _EXECNET_SCALARS:
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value) in (list, tuple, set, frozenset) and all(
            _execnet_serializable(item) for item in value
        )
    if isinstance(value, dict):
        return type(value) is dict and all(
            _execnet_serializable(key) and _execnet_serializable(item)
            for key, item in value.items()
        )
    return False


class _SubtestParameterRepr(str):
    """A subtest parameter received as its repr, displayed as the object was."""

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


@pytest.hookimpl(wrapper=True)
def pytest_report_to_serializable(
    config: pytest.Config, report: pytest.TestReport
) -> Generator[None, dict[str, Any] | None, dict[str, Any] | None]:
    # A subtest report carries its ``subTest(...)`` keyword arguments as they
    # were passed (a dtype, a Config, a sympy Integer), and a pytest-xdist
    # worker sends the report through execnet, which serializes only builtin
    # types: the send raised DumpError and failed the test.  Send each such
    # parameter as its repr, which is all the report shows of it.
    data = yield
    context = data.get("_subtest.context") if data is not None else None
    if context is not None:
        kwargs = context["kwargs"]
        unsendable = [
            key for key, value in kwargs.items() if not _execnet_serializable(value)
        ]
        if unsendable:
            context["kwargs"] = {
                key: saferepr(value) if key in unsendable else value
                for key, value in kwargs.items()
            }
            data[_REPR_KWARGS_KEY] = unsendable
    return data


@pytest.hookimpl(wrapper=True)
def pytest_report_from_serializable(
    config: pytest.Config, data: dict[str, Any]
) -> Generator[None, pytest.TestReport | None, pytest.TestReport | None]:
    keys = data.pop(_REPR_KWARGS_KEY, ())
    report = yield
    if keys:
        # A subtest report (pytest's own, or pytest-subtests' before pytest 9).
        kwargs = report.context.kwargs  # pyrefly: ignore [missing-attribute]
        for key in keys:
            kwargs[key] = _SubtestParameterRepr(kwargs[key])
    return report


# Budgets for the default number of pytest-xdist workers.  Every worker holds
# its own CUDA context plus Triton/Inductor state: four workers exhausted a
# 22 GiB A10G while two fit.  Most tests are CPU-bound compilation, so CPUs
# bound the count too.  On a B200 the full suite kept speeding up through 32.
_GPU_GIB_PER_WORKER = 8
_CPUS_PER_WORKER = 3
_MAX_WORKERS = 32


def _num_cpus() -> int:
    # The affinity mask honors taskset and container cpusets; macOS lacks it.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


# Prints the free memory (MiB) of the device the tests run on, through the
# vendor-neutral torch.accelerator API (ROCm reports itself as "cuda").
_ACCELERATOR_FREE_MIB = """
import torch
if torch.accelerator.is_available():
    if torch.accelerator.current_accelerator().type in ("cuda", "xpu"):
        print(torch.accelerator.get_memory_info(0)[0] // 2**20)
"""


def _first_visible_gpu() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()


def _free_device_memory_gib() -> float | None:
    """Free memory of the GPU the tests run on, or None without a GPU.

    Asks a subprocess so the controller neither imports torch nor holds a
    device context for the whole session.  nvidia-smi answers in ~0.1s; the
    torch.accelerator fallback (ROCm, XPU) takes ~2s.
    """
    commands = [
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            f"--id={_first_visible_gpu()}",
        ]
    ]
    # The torch wheel's local version names its GPU stack (cu130, rocm7.1,
    # xpu); skip the fallback for the cpu wheel, macOS, and source builds.
    torch_build = importlib.metadata.version("torch").partition("+")[2]
    if torch_build.startswith(("cu", "rocm", "xpu")):
        commands.append([sys.executable, "-c", _ACCELERATOR_FREE_MIB])
    for command in commands:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=60, check=True
            )
            return int(result.stdout) / 1024
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    return None


def _gpu_num_workers() -> int | None:
    """Workers that fit the GPU: 0 (serial) below two, None without a GPU."""
    free_gib = _free_device_memory_gib()
    if free_gib is None:
        return None
    workers = min(
        int(free_gib // _GPU_GIB_PER_WORKER),
        _num_cpus() // _CPUS_PER_WORKER,
        _MAX_WORKERS,
    )
    # A single worker only adds xdist's startup cost to a serial run.
    return workers if workers >= 2 else 0


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    # Size ``-n auto`` to the GPU rather than xdist's default of one worker per
    # CPU, unless PYTEST_XDIST_AUTO_NUM_WORKERS (read by xdist) overrides it.
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None
    return _gpu_num_workers()


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    # Runs before pytest-xdist's own hook, which turns these options into
    # workers.  xdist workers rerun this hook with their options cleared.
    if not config.pluginmanager.hasplugin("xdist") or hasattr(config, "workerinput"):
        return
    option = config.option
    # Without -n, run in parallel on a GPU, except where one process is
    # expected: -s, --pdb, --trace, --collect-only, and a pytest run started
    # from inside a test (PYTEST_CURRENT_TEST is set while one runs), which
    # observes its own reports in order.  -n0 runs serially.
    if (
        option.numprocesses is None
        and not option.tx
        and option.capture != "no"
        and not (option.usepdb or option.trace or option.collectonly)
        and "PYTEST_CURRENT_TEST" not in os.environ
    ):
        option.numprocesses = _gpu_num_workers()
    # worksteal lets idle workers take tests queued behind a long-running one;
    # xdist's default "load" scheduler leaves them stuck on a single worker
    # (e.g. the FLA-backed linear examples serialized into a ~16 min tail).
    if option.numprocesses and option.dist == "no":
        option.dist = "worksteal"


def _worker_status_dir(config: pytest.Config) -> Path | None:
    # xdist gives each worker the basetemp <run's basetemp>/popen-<worker id>.
    basetemp = config.option.basetemp if hasattr(config, "workerinput") else None
    return Path(basetemp).parent if basetemp else None


def _cached_gpu_bytes() -> int:
    # Name the tests' device: resolving the current one asks
    # torch.cuda.is_available(), which CPU-only tests mock to False.
    torch = sys.modules.get("torch")
    return torch.cuda.memory_reserved(0) if torch is not None else 0


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    # Note this worker's test for another worker's out-of-memory report.
    if (status_dir := _worker_status_dir(item.config)) is not None:
        worker = item.config.workerinput["workerid"]  # pyrefly: ignore [missing-attribute]
        (status_dir / f"helion-{worker}.status").write_text(
            f"{os.getpid()}\n{worker} ({_cached_gpu_bytes() / 2**30:.1f} GiB cached"
            f" when it started): {item.nodeid}"
        )


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item) -> None:
    # The workers share one GPU, and a worker keeps a big test's memory cached
    # for the rest of the run unless it is released.  Release whatever cache a
    # test left beyond the per-worker budget, which the other workers need.
    if (
        hasattr(item.config, "workerinput")
        and _cached_gpu_bytes() > _GPU_GIB_PER_WORKER * 2**30
    ):
        sys.modules["torch"].cuda.empty_cache()


def _gpu_memory_report(config: pytest.Config) -> str:
    """The GPU memory of each process, and the test each xdist worker runs."""
    lines = []
    for command in (
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv",
            f"--id={_first_visible_gpu()}",
        ],
        ["amd-smi", "process"],
        ["rocm-smi", "--showpids"],
    ):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=60, check=True
            )
        except (OSError, subprocess.SubprocessError):
            continue
        lines.append(result.stdout)
        break
    if (status_dir := _worker_status_dir(config)) is not None:
        for path in sorted(status_dir.glob("helion-*.status")):
            pid, status = path.read_text().split("\n", 1)
            try:
                children = psutil.Process(int(pid)).children(recursive=True)
            except psutil.Error:
                children = []
            lines.append(
                f"{status} [pid {pid}, child pids {[c.pid for c in children]}]"
            )
    return "\n".join(lines)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if call.excinfo is not None and "out of memory" in str(call.excinfo.value):
        report.sections.append(
            ("GPU memory when it ran out", _gpu_memory_report(item.config))
        )
    return report


def pytest_configure(config: pytest.Config) -> None:
    # xdist workers inherit this environment.  Left alone, OpenMP and OpenBLAS
    # in every worker start a busy-waiting thread per CPU when torch is
    # imported, oversubscribing the machine many times over.
    if workers := len(config.getoption("tx", None) or ()):
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, _num_cpus() // workers)))

    # The final-verification rebench re-times the top configs for a full 5s each
    # by default (HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS=5000), which alone
    # pushes the autotuner tests well past the suite's 60s per-test timeout and
    # gets their xdist worker killed. Clamp it to the floor (200ms) so the step
    # still runs (and stays covered) without dominating the test runtime.
    os.environ.setdefault("HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS", "200")
    # The same pass re-times the top 8 configs (32 for cute); every timed
    # iteration also zeroes the 256 MiB benchmark cache. Two finalists keep
    # the pass covered while it stays cheap on runners that share one GPU
    # between xdist workers. Tests of the default clear this key first.
    os.environ.setdefault("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")
    # At full effort the harness-grade final selection (5 s thermal warmup
    # plus four 0.6 s steady windows per finalist) replaces that pass and
    # ignores the two knobs above. Keep the suite on the single-pass
    # verification they size; tests of the selection enable it themselves.
    os.environ.setdefault("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS", "0")

    # The device-us re-rank needs the TPU profiler plane; under interpret
    # (CPU) it burns ~100 traced calls per candidate just to return inf.
    if os.environ.get("HELION_PALLAS_INTERPRET") == "1":
        os.environ.setdefault("HELION_AUTOTUNE_PALLAS_RANK_BY", "wall_time")

    # TODO(tcombes): remove this once Pallas RNG generation avoids int64.
    # JAX x64 is disabled on TPU, so RNG-generated int64s are truncated and
    # spam Pallas test logs with one warning per generated statement.
    warnings.filterwarnings(
        "ignore",
        message=(
            "Explicitly requested dtype int64 requested in .* is not available, "
            "and will be truncated to dtype int32.*"
        ),
        category=UserWarning,
    )
