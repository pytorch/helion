"""Narrow quiescence for forked source generation, without disabling progress."""

from __future__ import annotations

import atexit
import sys
import threading
from types import MethodType
from typing import Any
from typing import Protocol
from typing import cast
import weakref

_EVENT_METHODS = {
    name: getattr(threading.Event, name) for name in ("set", "wait", "is_set")
}
_THREAD_METHODS = {
    name: getattr(threading.Thread, name) for name in ("join", "is_alive")
}
_MONITOR_METHOD_NAMES = (
    "run",
    "get_instances",
    "report",
    "_atexit_signal",
    "join",
    "is_alive",
)


class _Monitor(Protocol):
    # Optional third-party API, admitted by exact runtime identities below.
    tqdm_cls: Any
    was_killed: threading.Event

    def _atexit_signal(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
    def is_alive(self) -> bool: ...


def _idle_monitor() -> _Monitor | None:
    threads = threading.enumerate()
    main = threading.main_thread()
    if (
        threading.current_thread() is not main
        or len(threads) != 2
        or main not in threads
    ):
        return None
    # tqdm is optional. Never import it just to enable source batching; a real
    # monitor already has both defining modules loaded. Older/unknown APIs decline.
    module = sys.modules.get("tqdm._monitor")
    standard = sys.modules.get("tqdm.std")
    monitor_type = getattr(module, "TMonitor", None)
    owner = getattr(standard, "tqdm", None)
    monitor = next(thread for thread in threads if thread is not main)
    if (
        monitor_type is None
        or owner is None
        or type(monitor) is not monitor_type
        or getattr(monitor, "tqdm_cls", None) is not owner
        or getattr(owner, "monitor", None) is not monitor
        or type(getattr(owner, "_instances", None)) is not weakref.WeakSet
        or len(owner._instances) != 0
        or type(getattr(monitor, "was_killed", None)) is not threading.Event
    ):
        return None
    if any(
        type(bound := getattr(monitor, name, None)) is not MethodType
        or bound.__func__
        is not _THREAD_METHODS.get(name, getattr(monitor_type, name, None))
        or bound.__self__ is not monitor
        for name in _MONITOR_METHOD_NAMES
    ) or any(
        type(bound := getattr(monitor.was_killed, name, None)) is not MethodType
        or bound.__func__ is not method
        or bound.__self__ is not monitor.was_killed
        for name, method in _EVENT_METHODS.items()
    ):
        return None
    return cast("_Monitor", monitor)


def source_threads_supported() -> bool:
    """Read-only admission; do not retire a monitor for an ineligible batch."""
    if threading.current_thread() is not threading.main_thread():
        return False
    return (
        threading.enumerate() == [threading.main_thread()]
        or _idle_monitor() is not None
    )


def prepare_source_fork(*, timeout: float = 1.0) -> bool:
    """Retire only an empty, owned standard monitor and require a lone main thread.

    Only the main thread can create bars here: the sole other admitted thread is
    the monitor, which reads existing instances. No tqdm lock is held while
    joining; the monitor might need that lock to finish an in-progress iteration.
    Preferences are unchanged; normal tqdm.__new__ recreates its monitor lazily.
    """
    if threading.current_thread() is not threading.main_thread():
        return False
    if threading.enumerate() == [threading.main_thread()]:
        return True
    monitor = _idle_monitor()
    if monitor is None:
        return False
    owner = monitor.tqdm_cls
    killed = monitor.was_killed
    interval = owner.monitor_interval
    callback = monitor._atexit_signal
    killed.set()
    monitor.join(timeout=timeout)
    if (
        monitor.is_alive()
        or owner.monitor is not monitor
        or monitor.was_killed is not killed
        or monitor.tqdm_cls is not owner
        or monitor._atexit_signal != callback
        or len(owner._instances) != 0
        or owner.monitor_interval != interval
        or threading.enumerate() != [threading.main_thread()]
    ):
        return False
    # The exact registered bound callback is removed only after a confirmed join.
    atexit.unregister(callback)
    owner.monitor = None
    return True
