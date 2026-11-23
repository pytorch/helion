"""Refusals raised by the lane-loop passes, which CuTe and Metal share."""

from __future__ import annotations

from .... import exc
from ...compile_environment import CompileEnvironment


def lane_loop_unsupported(msg: str) -> exc.BackendUnsupported:
    """A ``BackendUnsupported`` named after the backend compiling the kernel."""
    # The passes also run on bare ASTs outside a compile (CuTe's unit tests),
    # where only CuTe calls them.
    if not CompileEnvironment.has_current():
        return exc.BackendUnsupported("cute", msg)
    return CompileEnvironment.current().backend.unsupported(msg)
