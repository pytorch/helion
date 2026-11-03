"""Long-lived spawn subprocess for executing autotune benchmark jobs."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import dataclasses
import functools
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING
from typing import Callable
from typing import TypeVar

from .benchmarking import steady_graph_dispatch
from .logger import _UNRECOVERABLE_RUNTIME_ERROR_RE
from .logger import match_device_out_of_memory
from .process_utils import signal_process_tree
from .process_utils import start_isolated_process_group

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

_T = TypeVar("_T")

_log = logging.getLogger(__name__)

# How long a fresh worker may take to report ready (``PHASE_READY``: its
# interpreter started, the imports done and the device initialized) when the
# provider does not set ``autotune_worker_start_timeout``.
_DEFAULT_START_TIMEOUT_SECONDS = 120.0

# How long a SIGKILLed worker may take to exit before it counts as unreapable.
# A worker killed mid-kernel exits only once the driver has torn down its
# context, which for a hung or minutes-long kernel takes seconds: a 5 second
# window turned the fourth benchmark timeout of a sweep's swiglu search (a
# one-CTA tile of 8M elements) into a fatal BenchmarkWorkerUnkillable that
# discarded the whole autotune.
_REAP_TIMEOUT_SECONDS = 60.0


def _set_pdeathsig() -> None:
    """SIGTERM the child if the parent dies (Linux only, best-effort)."""
    if sys.platform != "linux":
        return
    with contextlib.suppress(Exception):
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)


def _worker_loop(connection: Connection, device: int | None) -> None:
    start_isolated_process_group()
    _set_pdeathsig()
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    # The first job's graphs would otherwise be captured in the fast dispatch
    # state of a fresh process and ranked against later candidates captured
    # in the steady one. The warm-up initializes the device (seconds, more on
    # a loaded host), which like the spawned interpreter's start and imports
    # is the worker's cost, not the first job's: the parent arms no job
    # deadline before the ready report that follows.
    steady_graph_dispatch()
    connection.send(_PhaseReport(PHASE_READY))

    while True:
        try:
            job = connection.recv()
        except EOFError:
            return
        if job is None:
            return
        try:
            # A no-op once the warm-up ran; a warm-up that failed is retried.
            steady_graph_dispatch()
            if isinstance(job, PhasedJob):
                result: object = job(functools.partial(_report_phase, connection))
            else:
                result = job()
        except BaseException as e:
            # Tracebacks pin the job's locals (tensors); strip before pickle.
            e.__traceback__ = None
            result = e
        try:
            connection.send(result)
        except BrokenPipeError:
            return
        except Exception:
            if not isinstance(result, BaseException):
                raise
            fallback = BenchmarkSubprocessError(
                f"worker raised unpickleable "
                f"{type(result).__module__}.{type(result).__qualname__}: {result}"
            )
            try:
                connection.send(fallback)
            except BrokenPipeError:
                return


# The phase a ``PhasedJob`` reports once its candidate is compiled and has
# run once: from here on the parent holds it to the benchmark timeout.
PHASE_COMPILED = "compiled"
# The report a fresh worker sends before its first job: the parent holds the
# start to ``start_timeout`` and arms the job's deadline only afterwards.
PHASE_READY = "ready"


@dataclasses.dataclass(frozen=True)
class _PhaseReport:
    """Progress marker a phased job sends ahead of its result."""

    phase: str


def _report_phase(connection: Connection, phase: str) -> None:
    connection.send(_PhaseReport(phase))


class PhasedJob:
    """A worker job that reports its phases to the parent.

    ``BenchmarkWorker.run`` holds such a job to two deadlines: the
    ``compile_timeout`` until it reports :data:`PHASE_COMPILED` (the candidate
    is loaded, compiled and has run once), then the benchmark ``timeout`` for
    the timing that follows. A job that never reports is held to the whole-job
    ``timeout`` like any other callable.
    """

    def __call__(self, report_phase: Callable[[str], None] | None = None) -> object:
        raise NotImplementedError


class BenchmarkSubprocessError(Exception):
    """Worker-subprocess failure, distinct from exceptions raised by the
    user job (which are re-raised verbatim)."""


class BenchmarkTimeout(BenchmarkSubprocessError):
    pass


class BenchmarkCompileTimeout(BenchmarkSubprocessError):
    """A phased job did not finish compiling (and running once) within the
    compile deadline. Not a :class:`BenchmarkTimeout`: the candidate was never
    timed, so it counts as a compile failure rather than a slow kernel.

    ``cold_start_seconds`` is how long the worker took to start when the job
    was its first (set by ``BenchmarkWorker.run``), None for a job in a warm
    worker. A first job also loads the arguments and the backend's runtime,
    so the search tries a candidate dropped that way once more.
    """

    cold_start_seconds: float | None = None


class BenchmarkWorkerDied(BenchmarkSubprocessError):
    pass


class BenchmarkWorkerUnkillable(BenchmarkSubprocessError):
    """A worker remained alive after the bounded SIGKILL wait.

    ``fn_index`` is the position in the isolated batch of the candidate that
    worker last ran (set by ``benchmark_isolated``), so the search can drop a
    config whose kernel may have hung instead of returning it.
    """

    fn_index: int | None = None


class BenchmarkWorker:
    """Single spawn subprocess. Lazily started on first ``run()``;
    respawned after timeout, sticky CUDA error, device out-of-memory error,
    or unexpected exit.

    A start (the spawned interpreter, its imports and the device
    initialization: seconds on an idle host, far longer on a loaded one) is
    held to ``start_timeout`` and is never charged to the job that caused it;
    ``log`` receives one line per start with the measured time.
    """

    def __init__(
        self,
        device: int | None = None,
        *,
        start_timeout: float = _DEFAULT_START_TIMEOUT_SECONDS,
        log: Callable[[str], None] = _log.debug,
        log_warning: Callable[[str], object] | None = None,
    ) -> None:
        self.device = device
        self.start_timeout = start_timeout
        self.log = log
        # Seconds the most recent start took until the worker reported ready.
        self.last_start_seconds: float | None = None
        self._process: mp.process.BaseProcess | None = None
        self._parent_connection: Connection | None = None
        self._lock = threading.Lock()
        self._fatal_error: str | None = None
        self._log_warning = log_warning
        # Jobs the current process completed without raising, and how many it
        # must have completed for a job that runs out of memory there to be
        # retried in a fresh one (see ``run``).
        self._jobs_succeeded = 0
        self._retry_after_jobs = 1

    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def run(
        self,
        job: Callable[[], _T],
        timeout: float,
        *,
        compile_timeout: float | None = None,
    ) -> _T:
        """Execute ``job`` in the worker.

        Raises ``BenchmarkTimeout`` if the job exceeds ``timeout`` seconds,
        ``BenchmarkWorkerDied`` if the worker crashed, or whatever exception
        the job raised. Sticky CUDA errors and device out-of-memory errors
        additionally kill the worker so the next call respawns it. If a
        killed worker remains alive, raises ``BenchmarkWorkerUnkillable`` and
        permanently disables this instance.

        After an out-of-memory error what the process holds may be what
        filled the device -- its caching allocator's free blocks, which a
        driver allocation (a Triton module load, a kernel's local memory)
        cannot use, the modules it loaded, and the local-memory high-water
        mark, which a spilling kernel raises for the life of the process, the
        failing one included -- and only a new process gives it back. A
        worker that had completed jobs reruns the job once in a fresh one. An
        out-of-memory error there is the job's own: it is raised and doubles
        the successful jobs required before the next rerun, so jobs that run
        out of memory on their own cost few reruns; a rerun that succeeds
        resets that count.

        With ``compile_timeout`` a :class:`PhasedJob` is held to two
        deadlines: ``compile_timeout`` seconds for the compile phase (until
        it reports :data:`PHASE_COMPILED`; past it the worker is killed and
        ``BenchmarkCompileTimeout`` is raised) and then ``timeout`` seconds,
        counted from that report, for the timing. Any other job keeps the
        single whole-job ``timeout``. Both deadlines are armed once the
        worker is ready: a job that starts the worker does not pay for the
        start, and a ``BenchmarkCompileTimeout`` it raises carries the
        start's measured seconds as ``cold_start_seconds``.
        """
        self._raise_if_unusable()
        cold_start_seconds: float | None = None
        if not self.alive():
            cold_start_seconds = self._start()
        connection = self._parent_connection
        process = self._process
        assert connection is not None
        assert process is not None

        phased = compile_timeout is not None and isinstance(job, PhasedJob)
        compiling = phased
        timed_out = threading.Event()
        done = threading.Event()
        completion_lock = threading.Lock()
        # The timer is re-armed when the compile phase ends; ``timer`` is read
        # under ``completion_lock`` so a stale timer never kills the worker
        # after the job completed.
        timer: threading.Timer | None = None

        def timeout_error() -> BenchmarkSubprocessError:
            if compiling:
                assert compile_timeout is not None
                error = BenchmarkCompileTimeout(
                    f"compile and first call exceeded {compile_timeout:.1f}s "
                    "in the benchmark worker"
                )
                error.cold_start_seconds = cold_start_seconds
                return error
            return BenchmarkTimeout(f"benchmark timeout after {timeout:.1f}s")

        def on_timeout(expected: threading.Timer) -> None:
            with completion_lock:
                # A timer the compile phase's end replaced must not fire.
                if done.is_set() or timer is not expected:
                    return
                timed_out.set()
            # Killing the child makes its pipe readable at EOF. Leave the
            # parent's descriptor open until run() leaves poll/recv: closing
            # it from this thread can lose the poll wakeup on macOS.
            self._kill_if_current(
                process, connection, raise_on_failure=False, close_connection=False
            )

        def arm(seconds: float) -> float:
            nonlocal timer
            with completion_lock:
                if timer is not None:
                    timer.cancel()
                armed = threading.Timer(seconds, lambda: on_timeout(armed))
                armed.daemon = True
                timer = armed
                armed.start()
            return time.monotonic() + seconds

        deadline = arm(compile_timeout if phased else timeout)
        try:
            try:
                connection.send(job)
            except (BrokenPipeError, OSError) as e:
                if timed_out.is_set():
                    self._kill_if_current(process, connection)
                    raise timeout_error() from e
                self._kill_if_current(process, connection)
                raise BenchmarkWorkerDied("failed to send job to worker") from e

            while True:
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not connection.poll(remaining):
                        timed_out.set()
                        self._kill_if_current(process, connection)
                        raise timeout_error()
                    message = connection.recv()
                except (EOFError, OSError) as e:
                    if timed_out.is_set():
                        self._kill_if_current(process, connection)
                        raise timeout_error() from e
                    self._kill_if_current(process, connection)
                    raise BenchmarkWorkerDied(
                        "worker pipe closed before sending result"
                        + (" during compile and first call" if compiling else "")
                    ) from e
                if isinstance(message, _PhaseReport):
                    if compiling and message.phase == PHASE_COMPILED:
                        compiling = False
                        deadline = arm(timeout)
                    continue
                result = message
                break
        finally:
            with completion_lock:
                done.set()
                if timer is not None:
                    timer.cancel()

        if timed_out.is_set():
            self._kill_if_current(process, connection)
            raise timeout_error()

        if isinstance(result, BaseException):
            if _UNRECOVERABLE_RUNTIME_ERROR_RE.search(str(result)):
                self._kill_if_current(process, connection)
            elif match_device_out_of_memory(result):
                retry = self._jobs_succeeded >= self._retry_after_jobs
                self._kill_if_current(process, connection)
                if not retry:
                    # spawn the replacement now: its imports overlap the
                    # parent's next steps (it has no CUDA context until a job)
                    self._start()
                    raise result
                if self._log_warning is not None:
                    first_line = str(result).split("\n", 1)[0]
                    self._log_warning(
                        f"Benchmark worker ran out of device memory after "
                        f"{self._jobs_succeeded} successful job(s) "
                        f"({type(result).__name__}: {first_line}); "
                        "retrying the job in a fresh worker"
                    )
                try:
                    # the fresh worker has no successful job: no second retry
                    retried = self.run(job, timeout, compile_timeout=compile_timeout)
                except Exception as error:
                    if match_device_out_of_memory(error):
                        self._retry_after_jobs *= 2
                        if self._log_warning is not None:
                            self._log_warning(
                                "The job ran out of device memory in a fresh "
                                "benchmark worker too; the next retry needs "
                                f"{self._retry_after_jobs} successful job(s) first"
                            )
                    raise
                self._retry_after_jobs = 1
                return retried
            raise result
        self._jobs_succeeded += 1
        return result  # type: ignore[return-value]

    def shutdown(self) -> None:
        process, connection = self._process, self._parent_connection
        if process is not None and process.is_alive() and connection is not None:
            with contextlib.suppress(Exception):
                connection.send(None)
                process.join(timeout=5)
        self._kill()
        self._raise_if_unusable()

    def _start(self) -> float:
        """Spawn the worker and wait for its ready report; returns the
        seconds that took.

        A worker that is not ready within ``start_timeout`` (or exits
        first) is killed and raises ``BenchmarkWorkerDied``.
        """
        self._raise_if_unusable()
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_worker_loop,
            args=(child_connection, self.device),
            daemon=True,
        )
        started = time.monotonic()
        process.start()
        child_connection.close()
        self._process = process
        self._parent_connection = parent_connection
        self._jobs_succeeded = 0
        if not parent_connection.poll(self.start_timeout):
            self._kill()
            self._raise_if_unusable()
            raise BenchmarkWorkerDied(
                f"benchmark worker did not start within {self.start_timeout:g}s "
                "(HELION_AUTOTUNE_WORKER_START_TIMEOUT)"
            )
        try:
            report = parent_connection.recv()
        except (EOFError, OSError) as e:
            self._kill()
            self._raise_if_unusable()
            raise BenchmarkWorkerDied(
                "benchmark worker exited before it was ready"
            ) from e
        assert report == _PhaseReport(PHASE_READY), report
        seconds = time.monotonic() - started
        self.last_start_seconds = seconds
        self.log(f"Benchmark worker ready in {seconds:.1f}s")
        return seconds

    def _kill_if_current(
        self,
        process: mp.process.BaseProcess,
        connection: Connection,
        *,
        raise_on_failure: bool = True,
        close_connection: bool = True,
    ) -> None:
        with self._lock:
            if (
                process is self._process
                and connection is self._parent_connection
                and self._fatal_error is None
            ):
                self._kill(close_connection=False)
            # The watchdog may already have cleared the worker's state. The
            # waiting thread still owns this connection and must close it.
            if close_connection:
                with contextlib.suppress(Exception):
                    connection.close()
            fatal_error = self._fatal_error
        if raise_on_failure and fatal_error is not None:
            raise BenchmarkWorkerUnkillable(fatal_error)

    def _raise_if_unusable(self) -> None:
        fatal_error = self._fatal_error
        if fatal_error is not None:
            raise BenchmarkWorkerUnkillable(fatal_error)

    def _kill(self, *, close_connection: bool = True) -> None:
        process, connection = self._process, self._parent_connection
        if process is not None:
            with contextlib.suppress(Exception):
                signal_process_tree(process, signal.SIGKILL)
            with contextlib.suppress(Exception):
                process.join(timeout=_REAP_TIMEOUT_SECONDS)
        if close_connection and connection is not None:
            with contextlib.suppress(Exception):
                connection.close()
        if process is not None and process.is_alive():
            if self._fatal_error is None:
                self._fatal_error = (
                    f"benchmark worker pid {process.pid} did not exit within "
                    f"{_REAP_TIMEOUT_SECONDS:g} seconds after SIGKILL; refusing to "
                    "launch another worker"
                )
            return
        self._process = None
        self._parent_connection = None
