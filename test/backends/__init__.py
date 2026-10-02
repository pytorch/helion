"""Backend registry for the portable test suite.

Each ``test/backends/<name>.py`` file calls ``register_backend()`` at module
level; the conftest imports them all at startup and applies the active
backend's gap markers at collection time.  See ``pallas.py`` for a worked
example.
"""

from __future__ import annotations

import dataclasses

import pytest

_REGISTRY: dict[str, BackendEntry] = {}


@dataclasses.dataclass
class BackendEntry:
    """Everything the test infrastructure knows about one backend."""

    name: str
    gaps: list[PortableGap] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PortableGap:
    """A known gap between a backend and one portable test (or a whole file).

    ``subpath`` is relative to ``test/portable/`` without ``.py``
    (e.g. ``"test_views"``).  ``inner_key`` is either ``""``
    (whole file) or ``"ClassName::method"`` (one test).
    """

    subpath: str
    inner_key: str
    marker: pytest.MarkDecorator


def register_backend(name: str) -> BackendEntry:
    """Register *name* and return its ``BackendEntry`` (idempotent)."""
    if name not in _REGISTRY:
        _REGISTRY[name] = BackendEntry(name=name)
    return _REGISTRY[name]


def get_registered_backends() -> dict[str, BackendEntry]:
    """Return all registered backends, keyed by name."""
    return dict(_REGISTRY)


def xfail_test(
    entry: BackendEntry,
    subpath: str,
    inner_key: str,
    *,
    reason: str,
    condition: bool = True,
) -> None:
    """Declare an expected-failure gap for *entry*'s backend.

    ``inner_key`` is ``""`` (whole file) or ``"ClassName::method"`` (one test).
    ``condition`` is a boolean evaluated at backend-file import time and
    forwarded to ``pytest.mark.xfail(condition=...)`` — when False, pytest
    runs the test normally.  Use this for sub-variants (e.g.
    ``condition=_is_pallas_tpu()``).  The first matching gap per item wins, so
    list more-specific gaps before broader ones.
    """
    entry.gaps.append(
        PortableGap(
            subpath=subpath,
            inner_key=inner_key,
            marker=pytest.mark.xfail(
                condition=condition,
                reason=f"[{entry.name}] {reason}",
                strict=True,
            ),
        )
    )
