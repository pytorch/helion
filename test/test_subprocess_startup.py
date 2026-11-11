"""The subprocess test deadlines use explicit readiness, not import duration."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from helion._testing import subprocess_startup as helper

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.cpu_only, pytest.mark.slow]


def _command(body: str) -> list[str]:
    # Load only this stdlib helper in the controlled child, not Helion or Torch.
    return [
        sys.executable,
        "-B",
        "-c",
        f"import runpy; ready = runpy.run_path({helper.__file__!r})['subprocess_ready']; "
        + body,
    ]


def test_ready_subprocess_preserves_output_and_exit_status() -> None:
    result = helper.run_test_subprocess(
        _command(
            "import sys; "
            "sys.stderr.write('x' * 262144); "
            "ready(); print('probe finished'); sys.exit(7)"
        ),
        env=dict(os.environ),
    )
    assert result.returncode == 7
    assert result.stdout == "probe finished\n"
    assert result.stderr == "x" * 262144


@pytest.mark.parametrize("status", [0, 23])
def test_exit_before_readiness_is_a_startup_failure(status: int) -> None:
    with pytest.raises(AssertionError, match=f"before readiness \\(status {status}\\)"):
        helper.run_test_subprocess(
            _command(f"import sys; print('startup output'); sys.exit({status})"),
            env=dict(os.environ),
        )


@pytest.mark.parametrize("phase", ["startup", "execution/shutdown"])
def test_deadline_kills_and_reaps_the_child(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    processes: list[subprocess.Popen[str]] = []
    original = subprocess.Popen

    def record_process(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(helper.subprocess, "Popen", record_process)
    # An event never set makes the timed-out operation deterministic. No sleeps
    # or assumptions about how quickly the interpreter starts are required.
    body = "import threading; "
    if phase == "execution/shutdown":
        body += "ready(); "
    body += "threading.Event().wait()"
    with pytest.raises(AssertionError, match=f"subprocess {phase} exceeded"):
        helper.run_test_subprocess(
            _command(body),
            env=dict(os.environ),
            startup_timeout=0 if phase == "startup" else 120,
            timeout=0,
        )
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_readiness_does_not_modify_the_callers_environment(tmp_path: Path) -> None:
    environment = dict(os.environ, HELION_TEST_SUBPROCESS_READY="parent marker")
    result = helper.run_test_subprocess(
        _command("ready(); print('ok')"), env=environment, cwd=tmp_path
    )
    assert result.returncode == 0
    assert environment["HELION_TEST_SUBPROCESS_READY"] == "parent marker"
