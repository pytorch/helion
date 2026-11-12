"""Standalone standard-library client for running a command on the hand-off GPU.

Copied into the agent's workspace as ``gpu_run.py``:

    python gpu_run.py [--timeout SECONDS] [--] COMMAND [ARGS...]

The caller runs COMMAND in the current directory, which must be inside the
workspace, on the GPU that measures submissions, and this prints its output.
The caller runs one command at a time and only while it waits for a
submission: a command waits while a submission is measured, and a submission
stops a command that is still running. The exit status is the command's
(128 + N when signal N killed it), or 124 when it did not complete (timed out,
stopped, or could not start). Ending this client, however it ends, cancels its
command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from typing import Any

MAILBOX = ".gpu"
DEFAULT_TIMEOUT = 600.0
MAX_TIMEOUT = 3600.0
INCOMPLETE = 124


def gpu_run(workspace: Path, command: list[str], timeout: float) -> int:
    """Request ``command``, wait for the caller to run it, and print its output."""
    mailbox = workspace.resolve() / MAILBOX
    request_id = uuid.uuid4().hex
    # The caller cancels the command once this lock is free: when this client
    # ends, however it ends.
    lock = os.open(mailbox / f"client_{request_id}.lock", os.O_WRONLY | os.O_CREAT)
    os.lockf(lock, os.F_LOCK, 0)
    temporary = mailbox / f".request_{request_id}.tmp"
    temporary.write_text(
        json.dumps(
            {
                "argv": command,
                "cwd": os.path.relpath(Path.cwd().resolve(), mailbox.parent),
                "timeout": timeout,
            }
        )
    )
    temporary.replace(mailbox / f"request_{request_id}.json")
    response_path = mailbox / f"response_{request_id}.json"
    closed = mailbox / "closed.json"
    response: dict[str, Any]
    while True:
        if response_path.exists():
            response = json.loads(response_path.read_text())
            response_path.unlink()
            break
        if closed.exists() or not mailbox.is_dir():
            # The caller answers before it closes.
            if response_path.exists():
                continue
            response = {"status": "stopped: the session ended", "output": ""}
            break
        time.sleep(0.2)
    sys.stdout.write(response["output"])
    sys.stdout.flush()
    if response["status"] != "completed":
        print(f"\n[gpu_run: {response['status']}]", file=sys.stderr)
        return INCOMPLETE
    returncode = response["returncode"]
    if returncode < 0:
        print(f"\n[gpu_run: killed by signal {-returncode}]", file=sys.stderr)
        return 128 - returncode
    return returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds (default {DEFAULT_TIMEOUT:g}, at most {MAX_TIMEOUT:g})",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("pass the command to run")
    sys.exit(gpu_run(Path(__file__).resolve().parent, command, args.timeout))


if __name__ == "__main__":
    main()
