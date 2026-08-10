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


class BenchmarkWorker:
    """Single spawn subprocess. Lazily started on first ``run()``;
    respawned after timeout, sticky CUDA error, or unexpected exit."""

    def __init__(self, device: int | None = None) -> None:
        self.device = device
        self._process: mp.process.BaseProcess | None = None
        self._parent_connection: Connection | None = None
        self._lock = threading.Lock()

    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def run(self, job: Callable[[], _T], timeout: float) -> _T:
        """Execute ``job`` in the worker.

        Raises ``BenchmarkTimeout`` if the job exceeds ``timeout`` seconds,
        ``BenchmarkWorkerDied`` if the worker crashed, or whatever exception
        the job raised. Sticky CUDA errors additionally kill the worker so
        the next call respawns it.
        """
        if not self.alive():
            self._start()
        connection = self._parent_connection
        process = self._process
        assert connection is not None
        assert process is not None

        timed_out = threading.Event()
        done = threading.Event()

        def on_timeout() -> None:
            if done.is_set():
                return
            timed_out.set()
            self._kill_if_current(process, connection)

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

    def _start(self) -> None:
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
    ) -> None:
        with self._lock:
            if process is self._process and connection is self._parent_connection:
                self._kill()

    def _kill(self) -> None:
        process, connection = self._process, self._parent_connection
        if process is not None and process.is_alive():
            with contextlib.suppress(Exception):
                process.kill()
                process.join(timeout=5)
        if connection is not None:
            with contextlib.suppress(Exception):
                connection.close()
        self._process = None
        self._parent_connection = None
