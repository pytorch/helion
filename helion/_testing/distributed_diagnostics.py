from __future__ import annotations

from collections import deque
import contextlib
import faulthandler
import json
import os
import select
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


@contextlib.contextmanager
def queued_copy_diagnostics(
    rank: int, *, timeout: float = 15
) -> Iterator[Callable[[str], None]]:
    """Observe one test body in a short-lived, spawned rank process.

    Normal return leaves the daemon and C watchdog armed through process-group
    teardown and process exit. Do not reuse this context in a long-lived process.
    C timeout intervals hint at the rank, but raw stacks can still interleave;
    only the bounded JSON records reliably identify rank and PID. On errors the Python
    reporter is stopped with a bounded join, but the C watchdog stays armed.
    This does not guarantee bounded interpreter shutdown with blocked stderr.
    """
    started = time.monotonic()
    stages = deque([(0.0, "child_test_entry")], maxlen=16)
    stopped = threading.Event()
    pipe_limit = min(4096, select.PIPE_BUF)
    interval = timeout / 2

    def mark(stage: str) -> None:
        stages.append((round(time.monotonic() - started, 6), stage[:96]))

    def emit(kind: str, sequence: int, **details: object) -> None:
        identity = {
            "rank": rank,
            "pid": os.getpid(),
            "sequence": sequence,
            "started": started,
            "elapsed": round(time.monotonic() - started, 6),
        }
        prefix = f"\nREMOTE_COPY_{kind} "
        record = (prefix + json.dumps({**identity, **details}) + "\n").encode()
        if len(record) > pipe_limit:
            # Keep identity parseable even for unusually long paths/labels.
            record = (
                prefix
                + json.dumps(
                    {
                        **identity,
                        "truncated": True,
                        "original_bytes": len(record),
                        "omitted_fields": list(details),
                    }
                )
                + "\n"
            ).encode()
        os.write(2, record)

    def report(sequence: int) -> None:
        history = list(stages)
        frames = []
        for thread_id, frame in sorted(sys._current_frames().items()):
            if thread_id == threading.get_ident():
                continue
            for depth in range(32):
                if frame is None:
                    break
                frames.append(
                    {
                        "thread": thread_id,
                        "depth": depth,
                        "filename": frame.f_code.co_filename[-192:],
                        "function": frame.f_code.co_name[:80],
                        "line": frame.f_lineno,
                    }
                )
                frame = frame.f_back
        # Snapshot frame metadata before any write can release the GIL or block.
        emit("STAGES", sequence, stages=history, history_limit=stages.maxlen)
        for details in frames:
            emit("FRAME", sequence, **details)

    def watch() -> None:
        # Sample at 15/22.5/30/37.5 seconds, leaving room for rank startup in
        # CI's 60-second parent timeout. Startup can still exhaust that budget.
        for sequence in range(4):
            due = started + timeout + sequence * interval
            if stopped.wait(max(0, due - time.monotonic())):
                return
            report(sequence)

    worker = threading.Thread(target=watch, name="remote-copy-diagnostics", daemon=True)
    # Arm once before starting Python: default rank intervals are 20/20.25/20.5/20.75
    # seconds. The header is a rank hint, not reliable attribution of raw frames.
    # Canceling or replacing a C watchdog waits for an active dump,
    # so neither the reporter nor exception cleanup may do that on a full pipe.
    faulthandler.dump_traceback_later(
        timeout * 4 / 3 + rank * timeout / 60, repeat=True, file=2
    )
    try:
        worker.start()
    except BaseException:
        stopped.set()
        raise
    try:
        yield mark
        mark("body_returned")
        # COMPLETE means only that the body returned normally. The diagnostics
        # remain active during process-group teardown and child process exit.
        emit("COMPLETE", -1)
    except BaseException:
        stopped.set()
        # Leave the independent C watchdog armed through error teardown too.
        worker.join(timeout=1)
        raise
