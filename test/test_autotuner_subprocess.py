from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

import pytest

import helion
from helion.autotuner.logger import match_unrecoverable_runtime_error
from helion.autotuner.subprocess_runner import _append_bad_config
from helion.autotuner.subprocess_runner import cleanup_subprocess_artifacts
from helion.autotuner.subprocess_runner import clear_pending
from helion.autotuner.subprocess_runner import load_bad_configs
from helion.autotuner.subprocess_runner import write_pending


class TestErrorStringMatching:
    """Test match_unrecoverable_runtime_error with bare payload substrings."""

    @pytest.mark.parametrize(
        ("msg", "expected"),
        [
            ("illegal memory access", True),
            ("an illegal memory access was encountered", True),
            ("misaligned address", True),
            ("unspecified launch failure", True),
            ("illegal instruction", True),
            ("ILLEGAL MEMORY ACCESS", True),  # case insensitive
            ("Misaligned Address", True),  # case insensitive
            ("out of memory", False),
            ("CUDA error: out of memory", False),
            ("segfault", False),
            ("", False),
        ],
    )
    def test_match(self, msg: str, expected: bool) -> None:
        err = RuntimeError(msg)
        assert match_unrecoverable_runtime_error(err) == expected


class TestPendingFileIO:
    """Test pending file write/clear lifecycle."""

    def test_write_and_clear(self, tmp_path: Path) -> None:
        config_str = "Config(block_sizes=[32], num_warps=4)"

        write_pending(str(tmp_path), config_str)
        pending_file = tmp_path / "_pending_config.txt"
        assert pending_file.exists()
        assert pending_file.read_text() == config_str

        clear_pending(str(tmp_path))
        assert not pending_file.exists()

    def test_clear_nonexistent(self, tmp_path: Path) -> None:
        # Should not raise
        clear_pending(str(tmp_path))


class TestBadConfigFileIO:
    """Test .bad_configs file read/write."""

    def test_load_empty(self, tmp_path: Path) -> None:
        bad_path = str(tmp_path / "test.bad_configs")
        result = load_bad_configs(bad_path)
        assert result == set()

    def test_append_and_load(self, tmp_path: Path) -> None:
        bad_path = str(tmp_path / "test.bad_configs")
        config1 = "Config(block_sizes=[32], num_warps=4)"
        config2 = "Config(block_sizes=[64], num_warps=8)"

        _append_bad_config(bad_path, config1)
        _append_bad_config(bad_path, config2)

        result = load_bad_configs(bad_path)
        assert config1 in result
        assert config2 in result
        assert len(result) == 2

    def test_config_str_deterministic(self) -> None:
        """Config.__str__() produces sorted, deterministic output."""
        c1 = helion.Config(block_sizes=[32], num_warps=4, num_stages=2)
        c2 = helion.Config(num_stages=2, num_warps=4, block_sizes=[32])
        assert str(c1) == str(c2)


class TestCleanupArtifacts:
    """Test cleanup_subprocess_artifacts removes crash-recovery files."""

    def test_cleanup(self, tmp_path: Path) -> None:
        (tmp_path / "_pending_config.txt").write_text("test")
        (tmp_path / "_bad_configs.txt").write_text("test")

        cleanup_subprocess_artifacts(str(tmp_path))

        assert not (tmp_path / "_pending_config.txt").exists()
        assert not (tmp_path / "_bad_configs.txt").exists()


SCRIPT = str(
    Path(__file__).parent.parent / "scripts" / "autotune_with_crash_recovery.sh"
)
HELPER = str(Path(__file__).parent / "data" / "autotune_crash_helper.py")


class TestBashCrashRecoveryScript:
    """Tests for scripts/autotune_with_crash_recovery.sh.

    These invoke the bash script via subprocess.run(). The crash recovery
    test uses test/data/autotune_crash_helper.py which monkey-patches
    _benchmark_config to crash on the first call, exercising the real
    pending-file and checkpoint code paths.
    """

    def test_normal_exit(self, tmp_path: Path) -> None:
        """Successful command passes through exit 0."""
        r = subprocess.run(
            [
                SCRIPT,
                "--checkpoint-dir",
                str(tmp_path),
                "--",
                "python",
                "-c",
                "pass",
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0

    def test_no_pending_propagates_error(self, tmp_path: Path) -> None:
        """Non-CUDA crash (no pending file) propagates exit code."""
        r = subprocess.run(
            [
                SCRIPT,
                "--checkpoint-dir",
                str(tmp_path),
                "--",
                "python",
                "-c",
                "import sys; sys.exit(42)",
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 42

    def test_crash_with_pending_recovery(self, tmp_path: Path) -> None:
        """Pending file detected by bash script, bad config recorded, re-run succeeds."""
        counter = tmp_path / "_run_counter"
        # First run: write pending via real write_pending() + exit(1)
        # Second run: succeed
        cmd = (
            "import sys, os; "
            "from pathlib import Path; "
            "from helion.autotuner.subprocess_runner import write_pending; "
            f"counter = Path('{counter}'); "
            "run = int(counter.read_text()) if counter.exists() else 0; "
            "counter.write_text(str(run + 1)); "
            f"write_pending('{tmp_path}', 'Config(bad=True)') if run == 0 else None; "
            "sys.exit(1) if run == 0 else None"
        )
        r = subprocess.run(
            [SCRIPT, "--checkpoint-dir", str(tmp_path), "--", "python", "-c", cmd],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        # Bad config was recorded by bash script
        bad = (tmp_path / "_bad_configs.txt").read_text()
        assert "Config(bad=True)" in bad
        # Script ran twice
        assert counter.read_text() == "2"

    def test_max_attempts_exceeded(self, tmp_path: Path) -> None:
        """Script gives up after --max-attempts crashes."""
        # Every run writes pending via real write_pending() + exits 1
        cmd = (
            "from helion.autotuner.subprocess_runner import write_pending; "
            f"write_pending('{tmp_path}', 'Config(always_bad=True)'); "
            "import os; os._exit(1)"
        )
        r = subprocess.run(
            [
                SCRIPT,
                "--checkpoint-dir",
                str(tmp_path),
                "--max-attempts",
                "3",
                "--",
                "python",
                "-c",
                cmd,
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
        bad_lines = (tmp_path / "_bad_configs.txt").read_text().strip().splitlines()
        assert len(bad_lines) == 3

    def test_real_autotune_through_bash(self) -> None:
        """End-to-end: real autotuning succeeds through the bash script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                **os.environ,
                "HELION_AUTOTUNE_MAX_GENERATIONS": "1",
                "HELION_AUTOTUNER": "PatternSearch",
            }
            r = subprocess.run(
                [
                    SCRIPT,
                    "--checkpoint-dir",
                    tmpdir,
                    "--",
                    "python",
                    HELPER,
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_real_crash_recovery_through_bash(self) -> None:
        """End-to-end: first run crashes during real benchmarking via
        monkey-patch, bash script detects pending file, records bad config,
        re-runs. Second run resumes from checkpoint and succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                **os.environ,
                "_CRASH_ON_FIRST_BENCHMARK": "1",
                "HELION_AUTOTUNE_MAX_GENERATIONS": "1",
                "HELION_AUTOTUNER": "PatternSearch",
            }
            r = subprocess.run(
                [
                    SCRIPT,
                    "--checkpoint-dir",
                    tmpdir,
                    "--",
                    "python",
                    HELPER,
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            assert r.returncode == 0, f"stderr: {r.stderr}"
            # Verify crash recovery happened (bad_configs.txt is cleaned
            # up by _cleanup_checkpoint on success, so check stderr)
            assert "[crash-recovery]" in r.stderr
            assert "Blocked config:" in r.stderr
