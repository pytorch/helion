"""Long-lived spawn subprocess for executing autotune benchmark jobs."""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import ctypes.util
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple
from typing import TypeVar

from .logger import _UNRECOVERABLE_RUNTIME_ERROR_RE

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

_T = TypeVar("_T")


class WorkerPoolResult(NamedTuple):
    worker_index: int
    elapsed: float
    result: object


def _set_pdeathsig() -> None:
    """SIGTERM the child if the parent dies (Linux only, best-effort)."""
    if sys.platform != "linux":
        return
    with contextlib.suppress(Exception):
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)


def _get_worker_context() -> mp.context.BaseContext:
    """Return the multiprocessing context used to spawn pool/benchmark
    workers. Always ``spawn`` -- ``forkserver`` is not viable because
    PyTorch refuses CUDA re-init in any process forked from one where
    torch was imported (which the forkserver inherits transitively)."""
    return mp.get_context("spawn")


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
        assert connection is not None
        try:
            connection.send(job)
        except (BrokenPipeError, OSError) as e:
            self._kill()
            raise BenchmarkWorkerDied("failed to send job to worker") from e

        if not connection.poll(timeout):
            self._kill()
            raise BenchmarkTimeout(f"benchmark timeout after {timeout:.1f}s")

        try:
            result = connection.recv()
        except EOFError as e:
            self._kill()
            raise BenchmarkWorkerDied("worker pipe closed before sending result") from e

        if isinstance(result, BaseException):
            if _UNRECOVERABLE_RUNTIME_ERROR_RE.search(str(result)):
                self._kill()
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
        context = _get_worker_context()
        parent_connection, child_connection = context.Pipe(duplex=True)
        # ``Process`` is on every concrete ``BaseContext`` subclass at runtime
        # but isn't typed on the ``BaseContext`` ABC.
        process = context.Process(  # pyrefly: ignore[missing-attribute]
            target=_worker_loop,
            args=(child_connection, self.device),
            daemon=True,
        )
        process.start()
        child_connection.close()
        self._process = process
        self._parent_connection = parent_connection

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


class BenchmarkWorkerPool:
    """Pool of long-lived ``BenchmarkWorker`` processes."""

    def __init__(self, num_workers: int) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self.workers = [BenchmarkWorker(device=None) for _ in range(num_workers)]

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    def run_on(self, worker_index: int, job: Callable[[], _T], timeout: float) -> _T:
        return self.workers[worker_index % self.num_workers].run(job, timeout=timeout)

    def map_jobs(
        self, jobs: list[Callable[[], object]], timeout: float
    ) -> list[WorkerPoolResult]:
        """Work-steal across workers and return one result per input job."""
        if not jobs:
            return []
        active_workers = min(self.num_workers, len(jobs))
        results: list[WorkerPoolResult | None] = [None] * len(jobs)
        q: queue.Queue[int] = queue.Queue()
        for i in range(len(jobs)):
            q.put(i)

        def steal(worker_idx: int) -> None:
            worker = self.workers[worker_idx]
            while True:
                try:
                    i = q.get_nowait()
                except queue.Empty:
                    return
                start = time.perf_counter()
                result = _run_capture(worker, jobs[i], timeout)
                results[i] = WorkerPoolResult(
                    worker_index=worker_idx,
                    elapsed=time.perf_counter() - start,
                    result=result,
                )

        _run_in_parallel(steal, active_workers)
        final_results: list[WorkerPoolResult] = []
        for result in results:
            assert result is not None
            final_results.append(result)
        return final_results

    def start_all(self, limit: int | None = None) -> None:
        """Eagerly start worker processes from the calling thread.

        Workers install ``PR_SET_PDEATHSIG`` (Linux) which anchors to the
        thread that spawned them; once that thread terminates, the OS sends
        SIGTERM to the worker. Pinning the spawn to the main thread before
        the first worker-pool phase keeps workers alive for the rest of the
        process."""
        if limit is None:
            limit = self.num_workers
        for worker in self.workers[:limit]:
            if not worker.alive():
                worker._start()

    def shutdown(self) -> None:
        for w in self.workers:
            with contextlib.suppress(Exception):
                w.shutdown()


def _run_capture(
    worker: BenchmarkWorker, job: Callable[[], object], timeout: float
) -> object:
    try:
        return worker.run(job, timeout=timeout)
    except BaseException as e:
        e.__traceback__ = None
        return e


def _run_in_parallel(target: Callable[[int], None], n: int) -> None:
    if n == 1:
        target(0)
        return
    threads = [
        threading.Thread(target=target, args=(i,), daemon=True) for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


_global_pool: BenchmarkWorkerPool | None = None
_global_pool_lock = threading.Lock()
_global_pool_shutdown_registered = False


def get_or_create_pool(num_workers: int) -> BenchmarkWorkerPool:
    """Return the process-level worker pool, creating it on first use.

    Reused across all autotune calls in the same Python process so pool
    cold-start (spawn N interpreters + ``import torch``) is paid once per
    process, not once per ``kernel.autotune()`` call. If a later autotune
    needs a different ``num_workers``, the pool is replaced (rare on a
    single-kernel CI process).
    """
    global _global_pool, _global_pool_shutdown_registered
    with _global_pool_lock:
        if _global_pool is not None and _global_pool.num_workers == num_workers:
            return _global_pool
        if _global_pool is not None:
            _global_pool.shutdown()
        _global_pool = BenchmarkWorkerPool(num_workers=num_workers)
        if not _global_pool_shutdown_registered:
            atexit.register(_shutdown_global_pool)
            _global_pool_shutdown_registered = True
    return _global_pool


def _shutdown_global_pool() -> None:
    global _global_pool
    with _global_pool_lock:
        if _global_pool is not None:
            with contextlib.suppress(Exception):
                _global_pool.shutdown()
            _global_pool = None
