"""Backend gaps for the portable test suite.

See ``test/portable/__init__.py`` for the portable-test policy.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Callable
import unittest

import pytest

from helion._compiler.backend_registry import list_backends

if TYPE_CHECKING:
    from types import ModuleType

_BACKENDS_DIR = Path(__file__).parent
_PORTABLE_DIR = _BACKENDS_DIR.parent / "portable"


@dataclasses.dataclass
class PortableGap:
    """One backend gap; see ``test/portable/__init__.py`` for semantics."""

    subpath: str
    inner_key: str
    condition: Callable[[], bool]
    reason: str
    skip: bool


_GAPS: dict[str, list[PortableGap]] = {name: [] for name in list_backends()}
GAP_STASH_KEY = pytest.StashKey[PortableGap]()


def _always_true() -> bool:
    return True


def backend_gap(
    subpath: str,
    inner_key: str,
    *,
    reason: str,
    condition: Callable[[], bool] = _always_true,
    skip: bool = False,
) -> PortableGap:
    """Declare a backend gap; see ``test/portable/__init__.py``."""
    return PortableGap(
        subpath=subpath,
        inner_key=inner_key,
        condition=condition,
        reason=reason,
        skip=skip,
    )


def _import_backend_gaps() -> None:
    """Import gap registries, deriving each backend from its directory."""
    registered = set(list_backends())
    for registry_path in sorted(_BACKENDS_DIR.glob("*/registry.py")):
        backend = registry_path.parent.name
        if backend not in registered:
            raise pytest.UsageError(
                f"gap registry directory {backend!r} is not a registered backend"
            )
        module = importlib.import_module(f"test.backends.{backend}.registry")
        _GAPS[backend].extend(module.GAPS)


def validate_gaps(modules: dict[str, ModuleType]) -> None:
    """Validate all registry targets against collected portable modules."""
    for backend, gaps in _GAPS.items():
        for gap in gaps:
            expected = _PORTABLE_DIR / f"{gap.subpath}.py"
            if not expected.exists():
                raise pytest.UsageError(
                    f"[{backend}] gap subpath {gap.subpath!r} does not exist: "
                    f"{expected}"
                )
            if gap.inner_key and (module := modules.get(gap.subpath)) is not None:
                class_name, separator, method_name = gap.inner_key.partition("::")
                cls = vars(module).get(class_name)
                valid_class = (
                    separator == "::"
                    and isinstance(cls, type)
                    and issubclass(cls, unittest.TestCase)
                    and not inspect.isabstract(cls)
                    and getattr(cls, "__test__", True)
                )
                valid_methods = (
                    unittest.defaultTestLoader.getTestCaseNames(cls)
                    if valid_class
                    else []
                )
                if method_name not in valid_methods or not getattr(
                    getattr(cls, method_name, None), "__test__", True
                ):
                    raise pytest.UsageError(
                        f"[{backend}] gap inner key {gap.inner_key!r} matches no "
                        f"test in {expected}"
                    )


def active_gaps(backend: str) -> list[PortableGap]:
    """Resolve applicable gaps for one backend."""
    applicable: list[PortableGap] = []
    seen: set[tuple[str, str]] = set()
    for gap in _GAPS[backend]:
        if not gap.condition():
            continue
        target = (gap.subpath, gap.inner_key)
        if target in seen:
            raise pytest.UsageError(
                f"[{backend}] duplicate applicable gap for "
                f"{gap.inner_key or f'{gap.subpath} (whole file)'}"
            )
        seen.add(target)
        applicable.append(dataclasses.replace(gap, reason=f"[{backend}] {gap.reason}"))
    return applicable


def matching_gap(
    gaps: list[PortableGap], subpath: str, class_and_method: str
) -> PortableGap | None:
    """Return an exact applicable gap, otherwise an applicable file gap."""
    matching = [gap for gap in gaps if gap.subpath == subpath]
    for gap in matching:
        if gap.inner_key == class_and_method:
            return gap
    return next((gap for gap in matching if not gap.inner_key), None)


def portable_subpath(item: pytest.Item) -> str | None:
    """Return an item's path below ``test/portable``, without its suffix."""
    if not item.path.is_relative_to(_PORTABLE_DIR):
        return None
    return str(item.path.relative_to(_PORTABLE_DIR).with_suffix(""))


def apply_gap(item: pytest.Item, gap: PortableGap) -> None:
    """Apply a gap using a class-local wrapper around the unbound method."""
    if gap.skip:
        item.add_marker(pytest.mark.skip(reason=gap.reason))
        return

    cls = item.cls
    original = getattr(cls, item.name)

    # pytest's strict xfail marker misreports failing unittest subtests as an
    # XPASS, so preserve unittest's expected-failure behavior instead.
    @functools.wraps(original)
    def wrapper(self: unittest.TestCase, *args: object, **kwargs: object) -> object:
        return original(self, *args, **kwargs)

    wrapper = unittest.expectedFailure(wrapper)
    item.stash[GAP_STASH_KEY] = gap
    setattr(cls, item.name, wrapper)
    item.obj = getattr(item.instance, item.name)


_import_backend_gaps()
