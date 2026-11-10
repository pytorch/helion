from __future__ import annotations

import contextlib
import gc
import multiprocessing as mp
import os
import signal
import sys
import threading


def start_isolated_process_group() -> None:
    """Keep compiler grandchildren killable when their worker times out."""
    with contextlib.suppress(OSError):
        os.setsid()


# Serializes start_without_collection: the collection setting is
# process-wide, and a start overlapping another's could save it off and fork
# after the other turned it back on.
_start_without_collection_lock = threading.Lock()
# While a start is in progress: the starting thread and the collection
# setting it saved.
_start_in_progress: tuple[int, bool] | None = None


def start_without_collection(process: mp.process.BaseProcess) -> None:
    """``process.start()``, with automatic garbage collection off for a
    fork-context process.

    A forked worker inherits the setting, so it cannot collect before its
    target calls :func:`isolate_forked_worker`. A spawned process starts a
    fresh interpreter, so it starts as usual.
    """
    global _start_in_progress
    if sys.platform == "win32" or not isinstance(process, mp.context.ForkProcess):
        process.start()
        return
    with _start_without_collection_lock:
        enabled = gc.isenabled()
        try:
            _start_in_progress = (threading.get_ident(), enabled)
            gc.disable()
            process.start()
        finally:
            if enabled:
                gc.enable()
            _start_in_progress = None


def _reset_start_state_in_child() -> None:
    """After any fork: a child inherits the start lock and the collection
    setting as the fork found them. A fork by another thread during a start
    (a data loader's, a user's pool) would otherwise leave the child's lock
    held and its collection off for good. A worker forked by the start keeps
    collection off until :func:`isolate_forked_worker`."""
    global _start_without_collection_lock, _start_in_progress
    _start_without_collection_lock = threading.Lock()
    if _start_in_progress is not None:
        starter, enabled = _start_in_progress
        _start_in_progress = None
        if enabled and starter != threading.get_ident():
            gc.enable()


if sys.platform != "win32":
    os.register_at_fork(after_in_child=_reset_start_state_in_child)


def isolate_forked_worker() -> None:
    """First thing in a forked worker: detach it from the parent state it
    shares through the fork. Start the worker with
    :func:`start_without_collection` so that nothing collects before this.

    The parent's objects are frozen out of the worker's garbage collection,
    which is then turned on for the objects the worker creates. A collection
    could otherwise free a CUDA object the parent left in a garbage cycle (a
    ``torch.cuda.Event`` from benchmarking), whose destructor switches devices,
    fails with "CUDA error: initialization error" in a forked process and
    aborts the worker.

    tqdm gets a lock of the worker's own. tqdm's default write lock includes a
    ``multiprocessing`` lock, a semaphore the worker shares with the parent,
    and ``torch.fx`` takes it for every graph it interprets during code
    generation. A worker that dies holding it (an abort, or the SIGKILL of a
    timeout) never releases it, and the parent's next code generation blocks
    on it forever. The default lock's thread lock may also have been held by
    another parent thread at the fork, which leaves the worker's copy locked
    for good.
    """
    gc.freeze()
    gc.enable()
    tqdm_std = sys.modules.get("tqdm.std")
    if tqdm_std is not None:
        lock = threading.RLock()
        # The default lock class keeps its locks on the class: a default lock
        # built in the worker gets the worker's lock and no multiprocessing one.
        tqdm_std.TqdmDefaultWriteLock.th_lock = lock
        tqdm_std.TqdmDefaultWriteLock.mp_lock = None
        pending = [tqdm_std.tqdm]
        while pending:
            cls = pending.pop()
            pending.extend(cls.__subclasses__())
            # Including a subclass that set a lock of its own.
            cls.set_lock(lock)


def signal_process_tree(process: mp.process.BaseProcess, sig: signal.Signals) -> None:
    pid = process.pid
    if pid is not None:
        try:
            os.killpg(pid, sig)
            return
        except (PermissionError, ProcessLookupError):
            pass
    with contextlib.suppress(OSError):
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
