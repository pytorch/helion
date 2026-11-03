"""The CuTe backend's refusals from code that runs without a compile environment.

Core raises ``env.backend.unsupported(detail)``; the lane-loop helpers of
``tile_strategy`` that tests build bare (no ``CompileEnvironment``) name the
backend through this factory instead of a literal in core.
"""

from __future__ import annotations

from ... import exc


def cute_unsupported(detail: str) -> exc.BackendUnsupported:
    """``BackendUnsupported`` of the CuTe backend."""
    return exc.BackendUnsupported("cute", detail)
