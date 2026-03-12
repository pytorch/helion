"""Thread budget validation for CuTe layout planning.

Centralizes the 1024-thread-per-block limit enforcement that was
previously scattered across backend.py and tile_strategy.py.
"""

from __future__ import annotations

from ... import exc

MAX_THREADS_PER_BLOCK = 1024


def check_thread_limit(
    num_threads: int,
    *,
    context: str = "",
) -> None:
    """Raise ``BackendUnsupported`` if *num_threads* exceeds 1024.

    This is the single source of truth for the CuTe thread-per-block limit.
    Both the scattered checks in ``backend.py`` and the layout planner call
    this function.

    Args:
        num_threads: Concrete thread count to validate.
        context: Human-readable description for the error message
                 (e.g. block sizes or node name).
    """
    if num_threads > MAX_THREADS_PER_BLOCK:
        msg = f"thread block too large for cute kernel: {context or num_threads}"
        raise exc.BackendUnsupported("cute", msg)
