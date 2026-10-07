"""Backend gaps for the portable test suite.

See ``test/portable/__init__.py`` for the portable-test policy.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Callable
from typing import Protocol
import unittest

import pytest

from helion._compiler.backend_registry import list_backends

_PORTABLE_DIR = Path(__file__).parents[1] / "portable"


class CollectedItem(Protocol):
    """Pytest item attributes used by portable-gap collection hooks."""

    path: Path
    nodeid: str
    name: str
    stash: pytest.Stash


class PortableItem(CollectedItem, Protocol):
    """Attributes provided by pytest's unittest TestCase items."""

    cls: type[unittest.TestCase]
    instance: unittest.TestCase
    obj: Callable[..., object]


@dataclasses.dataclass
class PortableGap:
    """One backend gap; see ``test/portable/__init__.py`` for semantics."""

    subpath: str
    inner_key: str
    condition: bool
    reason: str
    skip: bool


_GAPS: dict[str, list[PortableGap]] = {name: [] for name in list_backends()}
GAP_STASH_KEY = pytest.StashKey[PortableGap]()


def xfail_test(
    backend: str,
    subpath: str,
    inner_key: str,
    *,
    reason: str,
    condition: bool = True,
    skip: bool = False,
) -> None:
    """Declare a backend gap; see ``test/portable/__init__.py``."""
    if backend not in _GAPS:
        raise KeyError(f"unknown backend {backend!r}")
    _GAPS[backend].append(
        PortableGap(
            subpath=subpath,
            inner_key=inner_key,
            condition=condition,
            reason=f"[{backend}] {reason}",
            skip=skip,
        )
    )


def gaps_for_backend(backend: str) -> list[PortableGap]:
    """Return the gaps declared for *backend*."""
    return _GAPS[backend]


def validate_gaps(collected: dict[str, set[str]]) -> None:
    """Validate registry targets against portable items collected in this run."""
    for backend, gaps in _GAPS.items():
        seen: set[tuple[str, str]] = set()
        for gap in gaps:
            expected = _PORTABLE_DIR / f"{gap.subpath}.py"
            if not expected.exists():
                raise FileNotFoundError(
                    f"[{backend}] gap subpath {gap.subpath!r} does not exist: "
                    f"{expected}"
                )
            if gap.inner_key and gap.subpath in collected:
                if gap.inner_key not in collected[gap.subpath]:
                    raise ValueError(
                        f"[{backend}] gap inner key {gap.inner_key!r} matches no "
                        f"test collected from {expected}"
                    )
            if not gap.condition:
                continue
            target = (gap.subpath, gap.inner_key)
            if target in seen:
                raise ValueError(
                    f"[{backend}] duplicate applicable gap for "
                    f"{gap.inner_key or f'{gap.subpath} (whole file)'}"
                )
            seen.add(target)


def matching_gap(
    gaps: list[PortableGap], subpath: str, class_and_method: str
) -> PortableGap | None:
    """Return an exact applicable gap, otherwise an applicable file gap."""
    applicable = [gap for gap in gaps if gap.condition and gap.subpath == subpath]
    for gap in applicable:
        if gap.inner_key == class_and_method:
            return gap
    return next((gap for gap in applicable if not gap.inner_key), None)


def portable_subpath(item: CollectedItem) -> str | None:
    """Return an item's path below ``test/portable``, without its suffix."""
    if not item.path.is_relative_to(_PORTABLE_DIR):
        return None
    return str(item.path.relative_to(_PORTABLE_DIR).with_suffix(""))


def apply_gap(item: PortableItem, gap: PortableGap) -> None:
    """Apply a gap using a class-local wrapper around the unbound method."""
    cls = item.cls
    original = getattr(cls, item.name)

    @functools.wraps(original)
    def wrapper(self: unittest.TestCase, *args: object, **kwargs: object) -> object:
        return original(self, *args, **kwargs)

    if gap.skip:
        wrapper = unittest.skip(gap.reason)(wrapper)
    else:
        wrapper = unittest.expectedFailure(wrapper)
        item.stash[GAP_STASH_KEY] = gap
    setattr(cls, item.name, wrapper)
    item.obj = getattr(item.instance, item.name)
