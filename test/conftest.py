"""Root conftest for the Helion test suite.

Tests under ``test/portable/`` are expected to pass on every backend.  Each
backend declares known gaps in ``test/backends/<name>.py`` via
``xfail_test()``; this conftest imports all backend files at startup and
applies the active backend's markers at collection time.

To add a new backend: create ``test/backends/<name>.py``, call
``register_backend("<name>")``, run ``HELION_BACKEND=<name> pytest
test/portable/`` to find failures, then add ``xfail_test(...)`` entries.

Gap subpaths are validated against the filesystem at startup.
``HELION_INTERPRET=1`` (eager mode) is orthogonal to ``HELION_BACKEND``; tests
meaningless in eager mode use ``@skipIfRefEager`` directly in the test file.
"""

from __future__ import annotations

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


def _import_all_backends() -> None:
    """Import all ``test/backends/<name>.py`` files and validate gap subpaths.

    Files starting with ``_`` are skipped.  After import, every declared
    ``subpath`` is checked against the filesystem — a missing file raises
    ``FileNotFoundError`` immediately.  Inner-key (method-name) typos are not
    validated; ``strict=True`` on xfail markers surfaces them as plain failures.
    """
    from test.backends import get_registered_backends

    for path in sorted(_BACKENDS_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        importlib.import_module(f"test.backends.{path.stem}")

    # Validate all declared subpaths against the filesystem.
    for entry in get_registered_backends().values():
        for gap in entry.gaps:
            expected = _PORTABLE_DIR / f"{gap.subpath}.py"
            if not expected.exists():
                raise FileNotFoundError(
                    f"[{entry.name}] gap subpath {gap.subpath!r} does not exist: "
                    f"{expected}\n"
                    f"Fix the subpath in test/backends/{entry.name}.py or remove the entry."
                )


_import_all_backends()


def _get_active_backend() -> str:
    """Return the active backend name (reads ``HELION_BACKEND``, no helion import)."""
    return os.environ.get("HELION_BACKEND", "triton").strip()


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

    ``""`` matches every test; ``"ClassName::method"`` is an exact match.
    Parametrized ids (``"TestViews::test_foo[param0]"``) must be spelled out
    explicitly — ``"TestViews::test_foo"`` will not match them.
    """
    if not inner_key:
        return True
    return class_and_method == inner_key


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
