"""Root conftest for the Helion test suite.

Tests under ``test/portable/`` are expected to pass on every backend.  Each
backend declares known gaps in ``test/backends/<name>.py`` via
``xfail_test()``; this conftest imports all backend files at startup and
applies the active backend's markers at collection time.

To add a new backend: create ``test/backends/<name>.py``, call
``register_backend("<name>")``, run ``HELION_BACKEND=<name> pytest
test/portable/`` to find failures, then add ``xfail_test(...)`` entries.

Gap subpaths and inner keys are validated at startup.
``HELION_INTERPRET=1`` (eager mode) is orthogonal to ``HELION_BACKEND``; tests
meaningless in eager mode use ``@skipIfRefEager`` directly in the test file.
"""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    import pytest


def pytest_configure() -> None:
    # The final-verification rebench re-times the top configs for a full 5s each
    # by default (HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS=5000), which alone
    # pushes the autotuner tests well past the suite's 60s per-test timeout and
    # gets their xdist worker killed. Clamp it to the floor (200ms) so the step
    # still runs (and stays covered) without dominating the test runtime.
    os.environ.setdefault("HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS", "200")
    # The same pass re-times the top 8 configs (32 for cute); every timed
    # iteration also zeroes the 256 MiB benchmark cache. Two finalists keep
    # the pass covered while it stays cheap on runners that share one GPU
    # between xdist workers. Tests of the default clear this key first.
    os.environ.setdefault("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")

    # The device-us re-rank needs the TPU profiler plane; under interpret
    # (CPU) it burns ~100 traced calls per candidate just to return inf.
    if os.environ.get("HELION_PALLAS_INTERPRET") == "1":
        os.environ.setdefault("HELION_AUTOTUNE_PALLAS_RANK_BY", "wall_time")

    # TODO(tcombes): remove this once Pallas RNG generation avoids int64.
    # JAX x64 is disabled on TPU, so RNG-generated int64s are truncated and
    # spam Pallas test logs with one warning per generated statement.
    warnings.filterwarnings(
        "ignore",
        message=(
            "Explicitly requested dtype int64 requested in .* is not available, "
            "and will be truncated to dtype int32.*"
        ),
        category=UserWarning,
    )


_TEST_DIR = Path(__file__).parent
_PORTABLE_DIR = _TEST_DIR / "portable"
_BACKENDS_DIR = _TEST_DIR / "backends"


def _defined_test_ids(path: Path) -> set[str]:
    """Return the ``"ClassName::method"`` ids defined in a portable test file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    ids = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    ids.add(f"{node.name}::{member.name}")
    return ids


def _validate_gaps() -> None:
    """Validate every declared gap against the filesystem and test sources.

    A missing ``subpath`` file raises ``FileNotFoundError``; an ``inner_key``
    naming no ``Class::method`` defined in that file (typo, renamed or deleted
    test) raises ``ValueError``.  Parametrized id suffixes (``[param0]``) are
    stripped before matching.
    """
    from test.backends import get_registered_backends

    defined: dict[str, set[str]] = {}
    for entry in get_registered_backends().values():
        for gap in entry.gaps:
            expected = _PORTABLE_DIR / f"{gap.subpath}.py"
            if not expected.exists():
                raise FileNotFoundError(
                    f"[{entry.name}] gap subpath {gap.subpath!r} does not exist: "
                    f"{expected}\n"
                    f"Fix the subpath in test/backends/{entry.name}.py or remove the entry."
                )
            if not gap.inner_key:
                continue  # whole-file gap
            if gap.subpath not in defined:
                defined[gap.subpath] = _defined_test_ids(expected)
            if gap.inner_key.split("[", 1)[0] not in defined[gap.subpath]:
                raise ValueError(
                    f"[{entry.name}] gap inner key {gap.inner_key!r} matches no "
                    f"test defined in {expected}\n"
                    f"Fix the inner key in test/backends/{entry.name}.py or remove the entry."
                )


def _import_all_backends() -> None:
    """Import all ``test/backends/<name>.py`` files and validate their gaps.

    Files starting with ``_`` are skipped; see ``_validate_gaps`` for the
    startup checks applied to every declared gap.
    """
    for path in sorted(_BACKENDS_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        importlib.import_module(f"test.backends.{path.stem}")

    _validate_gaps()


_import_all_backends()


def _get_active_backend() -> str:
    """Return the active backend name (reads ``HELION_BACKEND``, no helion import).

    An empty or absent ``HELION_BACKEND`` maps to ``"triton"``, matching
    ``helion.runtime.settings._get_backend()``.  An unrecognised name is
    returned as-is so helion can raise the appropriate error; a warning is
    emitted so that a missing ``test/backends/<name>.py`` is noticed rather
    than silently running with no gap markers applied.
    """
    from test.backends import get_registered_backends

    raw = os.environ.get("HELION_BACKEND", "").strip()
    backend = raw or "triton"
    if backend not in get_registered_backends():
        warnings.warn(
            f"HELION_BACKEND={backend!r} is not registered in test/backends/; "
            f"no gap markers will be applied.  "
            f"Create test/backends/{backend}.py to declare gaps.",
            stacklevel=1,
        )
    return backend


def _portable_subpath(item: pytest.Item) -> str | None:
    """Return the portable subpath for *item*, or None if not under portable/."""
    try:
        rel = item.path.relative_to(_PORTABLE_DIR)
    except ValueError:
        return None
    return str(rel.with_suffix(""))


def _item_class_and_method(item: pytest.Item) -> str:
    """Return the ``ClassName::method`` tail of a collected item's nodeid."""
    parts = item.nodeid.split("::")
    return "::".join(parts[1:])


def _matches_inner_key(class_and_method: str, inner_key: str) -> bool:
    """Return True if *class_and_method* matches *inner_key*.

    ``""`` matches every test.  ``"ClassName::method"`` matches both the bare
    test and all its parametrized variants (``"ClassName::method[param0]"``,
    etc.).  To target a specific variant, include the full suffix in
    ``inner_key`` (e.g. ``"ClassName::method[param0]"``).
    """
    if not inner_key:
        return True
    # Strip the parametrized suffix from the collected id before comparing so
    # that a bare "ClassName::method" entry covers all parameter variants.
    bare = class_and_method.split("[", 1)[0]
    return bare == inner_key or class_and_method == inner_key


def pytest_collection_modifyitems(
    items: list[pytest.Item],
    config: pytest.Config,
) -> None:
    """Apply gap markers to portable test items for the active backend."""
    from test.backends import get_registered_backends

    backend_name = _get_active_backend()
    registry = get_registered_backends()
    entry = registry.get(backend_name)
    if entry is None or not entry.gaps:
        return

    for item in items:
        subpath = _portable_subpath(item)
        if subpath is None:
            continue  # outside test/portable/, not our concern

        class_and_method = _item_class_and_method(item)
        for gap in entry.gaps:
            if gap.subpath == subpath and _matches_inner_key(
                class_and_method, gap.inner_key
            ):
                item.add_marker(gap.marker, append=False)
                break  # first matching gap wins
