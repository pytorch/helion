from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

SCRIPT = str(
    Path(__file__).parent.parent / "helion" / "experimental" / "crash_recovery.py"
)
HELPER = str(Path(__file__).parent / "data" / "autotune_crash_helper.py")


@pytest.mark.timeout(120)
def test_benchmark_crash_recovery() -> None:
    """End-to-end: first run crashes during real benchmarking via
    monkey-patch, crash recovery script detects pending file, records crashed
    config, re-runs. Second run resumes from checkpoint and succeeds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {
            **os.environ,
            "HELION_AUTOTUNE_CHECKPOINT_DIR": tmpdir,
            "_CRASH_ON_FIRST_BENCHMARK": "1",
            "HELION_AUTOTUNE_MAX_GENERATIONS": "1",
            "HELION_AUTOTUNER": "PatternSearch",
        }
        r = subprocess.run(
            [sys.executable, SCRIPT, "--", sys.executable, HELPER],
            capture_output=True,
            text=True,
            env=env,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "[crash-recovery]" in r.stderr
        assert "Blocked config:" in r.stderr


@pytest.mark.timeout(120)
def test_compile_crash_recovery() -> None:
    """End-to-end: first run crashes during compile_config (simulated via
    os._exit inside monkey-patched _benchmark), crash recovery script detects
    pending file written before compile, records crashed config, re-runs.
    Second run resumes from checkpoint and succeeds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = {
            **os.environ,
            "HELION_AUTOTUNE_CHECKPOINT_DIR": tmpdir,
            "_CRASH_ON_FIRST_COMPILE": "1",
            "HELION_AUTOTUNE_MAX_GENERATIONS": "1",
            "HELION_AUTOTUNER": "PatternSearch",
        }
        r = subprocess.run(
            [sys.executable, SCRIPT, "--", sys.executable, HELPER],
            capture_output=True,
            text=True,
            env=env,
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "[crash-recovery]" in r.stderr
        assert "Blocked config:" in r.stderr
