"""Pytest plugin for per-test Helion performance annotations.

Activated by HELION_PERF_LOG=summary (or any non-zero value).
Appends kernel call counts, autotuning stats, and timing to each test line.
"""

from __future__ import annotations

import os
import time

import pytest


def _perf_enabled() -> bool:
    return os.environ.get("HELION_PERF_LOG", "0").strip().lower() != "0"


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if not _perf_enabled():
        return
    from helion._compile_time import get_tracker

    item.stash["_helion_snap"] = get_tracker().snapshot()  # type: ignore[attr-defined]
    item.stash["_helion_t0"] = time.perf_counter()  # type: ignore[attr-defined]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> object:  # type: ignore[type-arg]
    outcome = yield
    if not _perf_enabled():
        return
    report = outcome.get_result()
    if report.when != "call":
        return

    snap = item.stash.get("_helion_snap", None)  # type: ignore[attr-defined]
    if snap is None:
        return

    from helion._compile_time import get_tracker

    d = get_tracker().diff(snap)
    wall = time.perf_counter() - item.stash.get("_helion_t0", time.perf_counter())  # type: ignore[attr-defined]

    parts = []
    kc = d["kernel_calls"]
    if kc:
        parts.append(f"{kc} calls")
    at = d["autotune_count"]
    if at:
        ac = d["autotune_configs"]
        at_s = d["autotune_time"]
        parts.append(f"{at} tune({ac}cfg {at_s:.1f}s)")
    parts.append(f"{wall:.1f}s")

    suffix = "  <- " + ", ".join(parts)
    # Attach to the report so the terminal writer picks it up
    report._helion_suffix = suffix  # type: ignore[attr-defined]


@pytest.hookimpl(trylast=True)
def pytest_report_teststatus(
    report: pytest.TestReport, config: pytest.Config
) -> tuple[str, str, str] | None:
    if not _perf_enabled():
        return None
    if report.when != "call":
        return None
    suffix = getattr(report, "_helion_suffix", "")
    if not suffix:
        return None

    # Only modify verbose output (when -v is used or default with node names)
    if hasattr(report, "head_line"):
        # For verbose mode: append suffix to the status word
        if report.passed:
            return ("passed", ".", f"PASSED{suffix}")
        elif report.failed:
            return ("failed", "F", f"FAILED{suffix}")
        elif report.skipped:
            return ("skipped", "s", f"SKIPPED{suffix}")
    return None
