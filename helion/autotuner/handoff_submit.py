"""Standalone standard-library client for submitting a native source candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
import uuid


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def _source_path(workspace: Path, name: str) -> Path:
    path = (workspace / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(workspace):
        raise ValueError(f"Source path must remain inside the workspace: {name}")
    return path


def submit_candidate(workspace: Path, *, notes: str | None = None) -> dict[str, Any]:
    """Snapshot listed sources, wait for the caller, and print its feedback."""
    workspace = workspace.resolve()
    mailbox = _source_path(workspace, ".submissions")
    session = json.loads((mailbox / "session.json").read_text())
    deadline = session["deadline_monotonic"]

    def terminal() -> dict[str, Any] | None:
        closed = mailbox / "closed.json"
        if closed.exists():
            return json.loads(closed.read_text())
        if deadline is not None and time.monotonic() >= deadline:
            return {"done": True, "reason": "deadline"}
        return None

    response = terminal()
    if response is not None:
        print(json.dumps(response), flush=True)
        return response
    sources = {
        name: _source_path(workspace, name).read_text() for name in session["sources"]
    }
    if notes is None:
        path = _source_path(workspace, "notes.md")
        if path.is_file():
            notes = path.read_text()
    request_id = uuid.uuid4().hex
    _atomic_json(
        mailbox / f"request_{request_id}.json", {"sources": sources, "notes": notes}
    )
    response_path = mailbox / f"response_{request_id}.json"
    while True:
        if response_path.exists():
            response = json.loads(response_path.read_text())
            break
        response = terminal()
        if response is not None:
            break
        time.sleep(0.1)
    print(json.dumps(response), flush=True)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", help="Candidate notes; otherwise read notes.md")
    args = parser.parse_args()
    submit_candidate(Path(__file__).resolve().parent, notes=args.notes)


if __name__ == "__main__":
    main()
