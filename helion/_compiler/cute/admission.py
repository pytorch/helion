"""Admission outcomes of the CuTe planners.

A planner (``plan_<family>``) returns its plan record or :class:`Declined`.
A quiet decline (``error is None``) hands the root to the next lowering; a
loud one carries the refusal the caller raises (the exception's class and
message are decided at the declining site, so a ``BackendUnsupported`` /
``InvalidConfig`` text asserted by a test is unchanged by the hand-off).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import exc


@dataclass(frozen=True)
class Declined:
    """The planner did not admit the root; ``error`` is set when the decline is loud."""

    reason: str
    error: exc.BaseError | None = None


def refused(error: exc.BaseError) -> Declined:
    """A loud decline carrying ``error``; its message is the logged reason."""
    return Declined(str(error), error)
