"""Preserve pytest's collected identity for run_tests.py isolation reruns."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--helion-serial",
        action="store_true",
        help="run_tests.py internal: request serial execution if xdist is loaded",
    )
    parser.addoption(
        "--helion-basetemp-subdir",
        help="run_tests.py internal: isolate this job beneath a caller's basetemp",
    )
    parser.addoption(
        "--helion-isolated-nodeid",
        help="run_tests.py internal: collect only this isolated failure",
    )


@pytest.hookimpl(wrapper=True)
def pytest_load_initial_conftests(
    early_config: pytest.Config, args: list[str]
) -> Generator[None, None, None]:
    yield
    # Ini, environment and CLI plugin options have all been processed. Keep the
    # marker's position so caller overrides still win for initial serial jobs,
    # while the trailing isolation marker wins over caller worker settings.
    xdist = sys.modules.get("xdist.plugin")
    serial = (
        ["-n0"]
        if xdist is not None and early_config.pluginmanager.is_registered(xdist)
        else []
    )
    args[:] = [
        value
        for arg in args
        for value in (serial if arg == "--helion-serial" else [arg])
    ]


@pytest.hookimpl(wrapper=True)
def pytest_cmdline_main(
    config: pytest.Config,
) -> Generator[None, int | None, int | None]:
    subdir = config.getoption("--helion-basetemp-subdir")
    if (
        subdir is not None
        and config.option.basetemp is not None
        and not hasattr(config, "workerinput")
    ):
        # Run before pytest_configure creates its TempPathFactory. Jobs without
        # an explicit basetemp keep pytest's independently allocated defaults.
        # Workers already have distinct popen-gw* paths beneath this job root.
        basetemp = Path(config.option.basetemp)
        basetemp.mkdir(mode=0o700, parents=True, exist_ok=True)
        config.option.basetemp = str(basetemp / subdir)
    return (yield)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    nodeid = config.getoption("--helion-isolated-nodeid")
    if nodeid is not None:
        # Let pytest parse arbitrary plugin options before replacing selectors.
        # Dropping positional-looking tokens ourselves would also drop opaque
        # plugin values; retaining them here would collect extra test files.
        config.args = [nodeid]
        config.option.file_or_dir = [nodeid]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    _, separator, suffix = report.nodeid.partition("::")
    report.user_properties.extend(
        [
            ("helion_run_tests_path", str(item.path)),
            ("helion_run_tests_selector", separator + suffix),
        ]
    )
    return report
