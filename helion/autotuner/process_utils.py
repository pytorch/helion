from __future__ import annotations

import contextlib
import os
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as mp


def start_isolated_process_group() -> None:
    """Keep compiler grandchildren killable when their worker times out."""
    with contextlib.suppress(OSError):
        os.setsid()


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
