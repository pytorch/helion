"""Programmatic-dependent-launch helpers for standalone Triton pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

import helion.language as hl
from helion.runtime import default_launcher

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")


def signal_dependents() -> None:
    """Make the following PDL grid launchable without waiting for a predecessor."""
    hl.inline_triton(
        "tl.extra.cuda.gdc_launch_dependents()",
        args=(),
        output_like=None,
    )


def wait_and_launch_dependents() -> None:
    """Wait for the preceding grid, then make the following grid launchable."""
    hl.inline_triton(
        """
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
        """,
        args=(),
        output_like=None,
    )


def _pdl_launcher(*args: object, **kwargs: object) -> object:
    kwargs["launch_pdl"] = True
    return default_launcher(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]


def launch_dependent(kernel: Callable[..., _T], *args: object) -> _T:
    """Launch one compiled Helion kernel as a PDL dependent grid."""
    return kernel(*args, _launcher=_pdl_launcher)
