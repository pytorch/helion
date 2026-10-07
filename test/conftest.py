"""Root pytest configuration and portable-gap hooks."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import cast
import unittest
import warnings

import pytest

from test.backends import GAP_STASH_KEY
from test.backends import CollectedItem
from test.backends import PortableItem
from test.backends import apply_gap
from test.backends import gaps_for_backend
from test.backends import matching_gap
from test.backends import portable_subpath
from test.backends import validate_gaps

from helion._compiler.backend_registry import list_backends
from helion.runtime.ref_mode import RefMode
from helion.runtime.settings import _get_backend
from helion.runtime.settings import _get_ref_mode


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


_BACKENDS_DIR = Path(__file__).parent / "backends"


def _import_all_backends() -> None:
    for name in list_backends():
        if (_BACKENDS_DIR / name / "registry.py").exists():
            importlib.import_module(f"test.backends.{name}.registry")


_import_all_backends()


def pytest_collection_modifyitems(items: list[CollectedItem]) -> None:
    """Apply the active backend's gaps to portable TestCase items."""
    portable_items = [
        (item, subpath)
        for item in items
        if (subpath := portable_subpath(item)) is not None
    ]
    if not portable_items:
        return

    checked_items: list[tuple[PortableItem, str]] = []
    for item, subpath in portable_items:
        cls = getattr(item, "cls", None)
        if (
            cls is None
            or not isinstance(cls, type)
            or not issubclass(cls, unittest.TestCase)
        ):
            raise pytest.UsageError(
                f"portable test {item.nodeid} must be a unittest.TestCase method"
            )
        checked_items.append((cast("PortableItem", item), subpath))

    collected: dict[str, set[str]] = {}
    for item, subpath in checked_items:
        collected.setdefault(subpath, set()).add(f"{item.cls.__name__}::{item.name}")
    validate_gaps(collected)

    if _get_ref_mode() == RefMode.EAGER:
        return

    gaps = gaps_for_backend(_get_backend())
    for item, subpath in checked_items:
        class_and_method = f"{item.cls.__name__}::{item.name}"
        if gap := matching_gap(gaps, subpath, class_and_method):
            apply_gap(item, gap)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: CollectedItem):
    """Attach backend-gap reasons to unittest expected-failure reports."""
    report = yield
    gap = item.stash.get(GAP_STASH_KEY, None)
    if gap is None or report.when != "call":
        return report
    if report.skipped and hasattr(report, "wasxfail"):
        report.wasxfail = gap.reason
    elif report.failed:
        report.sections.append(("Portable gap", gap.reason))
    return report
