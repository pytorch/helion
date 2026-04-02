"""File I/O helpers for autotuner crash recovery.

The crash recovery protocol works with an external retry loop
(scripts/autotune_with_crash_recovery.sh). Before benchmarking each
config, the autotuner writes its string representation to a pending
file. If the process crashes (e.g. CUDA illegal memory access), the
pending file survives and the external retry loop records it as a bad
config. On re-run, the autotuner loads the checkpoint + bad configs
and skips the poison config.
"""

from __future__ import annotations

import os
from pathlib import Path

_PENDING_FILENAME = "_pending_config.txt"
_BAD_CONFIGS_FILENAME = "_bad_configs.txt"


def write_pending(checkpoint_dir: str, config_str: str) -> None:
    """Write the config being benchmarked to the pending file."""
    pending_path = Path(checkpoint_dir) / _PENDING_FILENAME
    pending_path.write_text(config_str)


def clear_pending(checkpoint_dir: str) -> None:
    """Remove the pending file after benchmark completes."""
    pending_path = Path(checkpoint_dir) / _PENDING_FILENAME
    if pending_path.exists():
        pending_path.unlink()


def load_bad_configs(bad_configs_path: str) -> set[str]:
    """Load bad config strings from file, one per line."""
    path = Path(bad_configs_path)
    if not path.exists():
        return set()
    lines = path.read_text().splitlines()
    return {line.strip() for line in lines if line.strip()}


def _append_bad_config(bad_configs_path: str, config_str: str) -> None:
    """Append a bad config string to the bad configs file."""
    with open(bad_configs_path, "a") as f:
        f.write(config_str + "\n")
        f.flush()
        os.fsync(f.fileno())


def cleanup_subprocess_artifacts(checkpoint_dir: str) -> None:
    """Remove crash-recovery files in the checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    for name in (
        _PENDING_FILENAME,
        _BAD_CONFIGS_FILENAME,
    ):
        artifact = checkpoint_path / name
        if artifact.exists():
            artifact.unlink()
