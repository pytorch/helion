"""Autotuner crash recovery wrapper.

Runs a command (typically a Python script that calls helion autotuning) in a
retry loop.  When the process crashes due to an unrecoverable CUDA error
(illegal memory access, misaligned address, etc.), the autotuner leaves a
``{hash}.pending_config`` sentinel in the checkpoint directory.  This script
detects that file, records the poison config in ``{hash}.crashed_configs``, and
re-runs the command.  On re-run the autotuner loads its checkpoint and skips
the crashed config.

Progress detection
------------------
Each crash should block a different config (since blocked configs are skipped
on re-run).  If the same config crashes twice, the autotuner is stuck and we
give up.

Requirements
------------
``HELION_AUTOTUNE_CHECKPOINT_DIR`` must be set in the environment.

Usage
-----
::

    HELION_AUTOTUNE_CHECKPOINT_DIR=/tmp/$USER/helion_ckpt \\
        python -m helion.experimental.crash_recovery [--max-retries N] -- COMMAND [ARGS...]

Examples
--------
::

    HELION_AUTOTUNE_CHECKPOINT_DIR=/tmp/$USER/helion_autotune_ckpt \\
        python -m helion.experimental.crash_recovery -- python train.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _log(msg: str) -> None:
    print(f"[crash-recovery] {msg}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autotuner crash recovery wrapper.",
        usage=(
            "HELION_AUTOTUNE_CHECKPOINT_DIR=/path/to/dir\n"
            "           %(prog)s [--max-retries N] -- COMMAND [ARGS...]"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=50,
        help="Maximum number of crash recovery retries (default: 50)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run (after '--' separator)",
    )
    args = parser.parse_args(argv)

    # argparse.REMAINDER absorbs '--' as first element when present.
    command: list[str] = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command specified after --")

    checkpoint_dir_str = os.environ.get("HELION_AUTOTUNE_CHECKPOINT_DIR", "")
    if not checkpoint_dir_str:
        print(
            "Error: HELION_AUTOTUNE_CHECKPOINT_DIR must be set.",
            file=sys.stderr,
        )
        return 1

    checkpoint_dir = Path(checkpoint_dir_str)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    attempt = 0
    all_crashed: set[str] = set()

    while True:
        attempt += 1

        result = subprocess.run(command)
        exit_code = result.returncode

        if exit_code == 0:
            return 0

        # Look for any *.pending_config sentinel left by the autotuner.
        pending_files = sorted(checkpoint_dir.glob("*.pending_config"))

        if pending_files:
            stuck = False
            for pending_path in pending_files:
                hash_prefix = pending_path.stem  # {hash} without .pending_config
                crashed_configs_path = checkpoint_dir / f"{hash_prefix}.crashed_configs"

                config = pending_path.read_text().strip()
                pending_path.unlink()

                with open(crashed_configs_path, "a") as f:
                    f.write(config + "\n")

                _log(f"Blocked config: {config}")

                # If this config was already blocked in a previous attempt,
                # the autotuner is not skipping it -- it's stuck.
                if config in all_crashed:
                    stuck = True
                all_crashed.add(config)

            _log(f"Process crashed (exit code {exit_code}, attempt {attempt}).")

            if stuck:
                _log("Same config crashed twice \u2014 the autotuner appears stuck.")
                _log(
                    "All crashed configs have been recorded. You can re-run "
                    "this script and it will resume from the latest "
                    "checkpoint, skipping all previously recorded crashed "
                    "configs."
                )
                return 1

            if attempt >= args.max_retries:
                _log(f"Reached maximum retry limit ({args.max_retries}). Giving up.")
                return 1

            _log("Restarting from checkpoint...")
        else:
            # No pending file -- not a recoverable CUDA crash.
            return exit_code


if __name__ == "__main__":
    sys.exit(main())
