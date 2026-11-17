"""Separate interpreter/import startup from a fresh-process test's deadline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Sequence

_READY_PATH = "HELION_TEST_SUBPROCESS_READY"
_log = logging.getLogger(__name__)


def subprocess_ready() -> None:
    """Call after allowed imports/setup, immediately before the tested operation."""
    Path(os.environ[_READY_PATH]).touch()


def run_test_subprocess(
    args: Sequence[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    startup_timeout: float = 120,
    timeout: float = 30,
) -> subprocess.CompletedProcess[str]:
    """Keep the probe and interpreter shutdown within ``timeout`` after readiness.

    Like the spawned benchmark-worker tests, allow imports a separate startup
    budget. A private readiness file works with an unchanged command/environment
    on every platform; temporary output files cannot block imports on full pipes.
    """
    with (
        tempfile.TemporaryDirectory(prefix="helion-test-startup-") as directory,
        tempfile.TemporaryFile(mode="w+") as stdout,
        tempfile.TemporaryFile(mode="w+") as stderr,
    ):
        ready = Path(directory) / "ready"
        started = time.monotonic()
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env={**env, _READY_PATH: str(ready)},
            stdout=stdout,
            stderr=stderr,
        )
        failure: str | None = None
        try:
            while not ready.is_file():
                remaining = startup_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    failure = f"subprocess startup exceeded {startup_timeout} seconds"
                    break
                try:
                    process.wait(timeout=min(0.05, remaining))
                except subprocess.TimeoutExpired:
                    continue
                if not ready.is_file():
                    failure = (
                        "subprocess exited before readiness "
                        f"(status {process.returncode})"
                    )
                    break
            if failure is None:
                ready_at = time.monotonic()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    failure = (
                        f"subprocess execution/shutdown exceeded {timeout} seconds"
                    )
                _log.info(
                    "Subprocess startup %.3fs; execution/shutdown %.3fs: %s",
                    ready_at - started,
                    time.monotonic() - ready_at,
                    args,
                )
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        stdout.seek(0)
        stderr.seek(0)
        result = subprocess.CompletedProcess(
            args, process.returncode, stdout.read(), stderr.read()
        )
        assert failure is None, f"{failure}\n{result.stdout}\n{result.stderr}"
        return result
