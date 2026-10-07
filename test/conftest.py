"""Root pytest configuration and portable-gap hooks."""

from __future__ import annotations

import os
import unittest
import warnings

import pytest

from test.backends import GAP_STASH_KEY
from test.backends import active_gaps
from test.backends import apply_gap
from test.backends import matching_gap
from test.backends import portable_subpath
from test.backends import validate_gaps

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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply the active backend's gaps to portable TestCase items."""
    checked_items: list[tuple[pytest.Item, str, str]] = []
    modules = {}
    for item in items:
        if (subpath := portable_subpath(item)) is None:
            continue
        cls = getattr(item, "cls", None)
        if cls is None or not issubclass(cls, unittest.TestCase):
            raise pytest.UsageError(
                f"portable test {item.nodeid} must be a unittest.TestCase method"
            )
        class_and_method = f"{cls.__name__}::{item.name}"
        checked_items.append((item, subpath, class_and_method))
        modules[subpath] = item.module

    if not checked_items:
        return

    validate_gaps(modules)

    if _get_ref_mode() == RefMode.EAGER:
        return

    gaps = active_gaps(_get_backend())
    for item, subpath, class_and_method in checked_items:
        if gap := matching_gap(gaps, subpath, class_and_method):
            if config.option.runxfail and not gap.skip:
                continue
            apply_gap(item, gap)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item):
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
