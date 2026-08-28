"""Long-lived spawn subprocess for executing autotune benchmark jobs."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import multiprocessing as mp
import os
import signal
import sys
import threading
from typing import TYPE_CHECKING
from typing import Callable
from typing import TypeVar

from .logger import _UNRECOVERABLE_RUNTIME_ERROR_RE
from .process_utils import signal_process_tree
from .process_utils import start_isolated_process_group

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

_T = TypeVar("_T")


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

    while True:
        try:
            job = connection.recv()
        except EOFError:
            return
        if job is None:
            return
        try:
            result: object = job()
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


class BenchmarkSubprocessError(Exception):
    """Worker-subprocess failure, distinct from exceptions raised by the
    user job (which are re-raised verbatim)."""


class BenchmarkTimeout(BenchmarkSubprocessError):
    pass


class BenchmarkWorkerDied(BenchmarkSubprocessError):
    pass


class BenchmarkWorkerUnkillable(BenchmarkSubprocessError):
    """A worker remained alive after the bounded SIGKILL wait."""


class BenchmarkWorker:
    """Single spawn subprocess. Lazily started on first ``run()``;
    respawned after timeout, sticky CUDA error, or unexpected exit."""

    def __init__(self, device: int | None = None) -> None:
        self.device = device
        self._process: mp.process.BaseProcess | None = None
        self._parent_connection: Connection | None = None
        self._lock = threading.Lock()
        self._fatal_error: str | None = None

    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def run(self, job: Callable[[], _T], timeout: float) -> _T:
        """Execute ``job`` in the worker.

        Raises ``BenchmarkTimeout`` if the job exceeds ``timeout`` seconds,
        ``BenchmarkWorkerDied`` if the worker crashed, or whatever exception
        the job raised. Sticky CUDA errors additionally kill the worker so the
        next call respawns it. If a killed worker remains alive, raises
        ``BenchmarkWorkerUnkillable`` and permanently disables this instance.
        """
        self._raise_if_unusable()
        if not self.alive():
            self._start()
        connection = self._parent_connection
        process = self._process
        assert connection is not None
        assert process is not None

        timed_out = threading.Event()
        done = threading.Event()
        completion_lock = threading.Lock()

        def on_timeout() -> None:
            with completion_lock:
                if done.is_set():
                    return
                timed_out.set()
            self._kill_if_current(process, connection, raise_on_failure=False)

        timer = threading.Timer(timeout, on_timeout)
        timer.daemon = True
        timer.start()
        try:
            try:
                connection.send(job)
            except (BrokenPipeError, OSError) as e:
                if timed_out.is_set():
                    self._kill_if_current(process, connection)
                    raise BenchmarkTimeout(
                        f"benchmark timeout after {timeout:.1f}s"
                    ) from e
                self._kill_if_current(process, connection)
                raise BenchmarkWorkerDied("failed to send job to worker") from e

            try:
                if not connection.poll(timeout):
                    timed_out.set()
                    self._kill_if_current(process, connection)
                    raise BenchmarkTimeout(f"benchmark timeout after {timeout:.1f}s")
            except (EOFError, OSError) as e:
                if timed_out.is_set():
                    self._kill_if_current(process, connection)
                    raise BenchmarkTimeout(
                        f"benchmark timeout after {timeout:.1f}s"
                    ) from e
                self._kill_if_current(process, connection)
                raise BenchmarkWorkerDied(
                    "worker pipe closed before sending result"
                ) from e

            try:
                result = connection.recv()
            except (EOFError, OSError) as e:
                if timed_out.is_set():
                    self._kill_if_current(process, connection)
                    raise BenchmarkTimeout(
                        f"benchmark timeout after {timeout:.1f}s"
                    ) from e
                self._kill_if_current(process, connection)
                raise BenchmarkWorkerDied(
                    "worker pipe closed before sending result"
                ) from e
        finally:
            with completion_lock:
                done.set()
            timer.cancel()

        if timed_out.is_set():
            self._kill_if_current(process, connection)
            raise BenchmarkTimeout(f"benchmark timeout after {timeout:.1f}s")

        if isinstance(result, BaseException):
            if _UNRECOVERABLE_RUNTIME_ERROR_RE.search(str(result)):
                self._kill_if_current(process, connection)
            raise result
        return result  # type: ignore[return-value]

    def shutdown(self) -> None:
        process, connection = self._process, self._parent_connection
        if process is not None and process.is_alive() and connection is not None:
            with contextlib.suppress(Exception):
                connection.send(None)
                process.join(timeout=5)
        self._kill()
        self._raise_if_unusable()

    def _start(self) -> None:
        self._raise_if_unusable()
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_worker_loop,
            args=(child_connection, self.device),
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._process = process
        self._parent_connection = parent_connection

    def _kill_if_current(
        self,
        process: mp.process.BaseProcess,
        connection: Connection,
        *,
        raise_on_failure: bool = True,
    ) -> None:
        with self._lock:
            if (
                process is self._process
                and connection is self._parent_connection
                and self._fatal_error is None
            ):
                self._kill()
            fatal_error = self._fatal_error
        if raise_on_failure and fatal_error is not None:
            raise BenchmarkWorkerUnkillable(fatal_error)

    def _raise_if_unusable(self) -> None:
        fatal_error = self._fatal_error
        if fatal_error is not None:
            raise BenchmarkWorkerUnkillable(fatal_error)

    def _kill(self) -> None:
        process, connection = self._process, self._parent_connection
        if process is not None:
            with contextlib.suppress(Exception):
                signal_process_tree(process, signal.SIGKILL)
            with contextlib.suppress(Exception):
                process.join(timeout=5)
        if connection is not None:
            with contextlib.suppress(Exception):
                connection.close()
        if process is not None and process.is_alive():
            if self._fatal_error is None:
                self._fatal_error = (
                    f"benchmark worker pid {process.pid} did not exit within 5 seconds "
                    "after SIGKILL; refusing to launch another worker"
                )
            return
        self._process = None
        self._parent_connection = None
