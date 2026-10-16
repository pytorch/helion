from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


@pytest.fixture
def workspace(tmp_path):
    directory = tmp_path / "workspace"
    directory.mkdir()
    shutil.copyfile(
        Path(__file__).parents[1] / "helion/autotuner/handoff_submit.py",
        directory / "submit_candidate.py",
    )
    (directory / "case_0").mkdir()
    (directory / "case_0/kernel.py").write_text("first source\n")
    (directory / "second.py").write_text("second source\n")
    (directory / "notes.md").write_text("notes from file\n")
    mailbox = directory / ".submissions"
    mailbox.mkdir()
    write_json(
        mailbox / "session.json",
        {
            "sources": ["case_0/kernel.py", "second.py"],
            "deadline_monotonic": time.monotonic() + 30,
        },
    )
    return directory


@contextmanager
def client(workspace, *args):
    process = subprocess.Popen(
        [sys.executable, "-I", str(workspace / "submit_candidate.py"), *args],
        cwd=workspace,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def wait_for_request(workspace, process):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        requests = list((workspace / ".submissions").glob("request_*.json"))
        if requests:
            assert len(requests) == 1
            return requests[0]
        if process.poll() is not None:
            pytest.fail(f"Client exited before submitting: {process.communicate()}")
        time.sleep(0.01)
    pytest.fail("Client did not submit within five seconds")


@pytest.mark.parametrize("notes", ["file", "explicit", "absent"])
def test_isolated_client_snapshots_sources_and_prints_feedback(workspace, notes):
    args = ("--notes", "explicit notes") if notes == "explicit" else ()
    if notes == "absent":
        (workspace / "notes.md").unlink()
    with client(workspace, *args) as process:
        request = wait_for_request(workspace, process)
        snapshot = json.loads(request.read_text())
        assert snapshot == {
            "sources": {
                "case_0/kernel.py": "first source\n",
                "second.py": "second source\n",
            },
            "notes": {
                "file": "notes from file\n",
                "explicit": "explicit notes",
                "absent": None,
            }[notes],
        }
        (workspace / "case_0/kernel.py").write_text("later edit\n")
        (workspace / "notes.md").write_text("later notes\n")
        assert json.loads(request.read_text()) == snapshot
        response = {
            "round": 1,
            "accepted": False,
            "reason": "regression_or_noise",
            "current": {"ok": True, "perf": 1.25, "unit": "ms"},
            "done": False,
        }
        write_json(
            request.with_name(request.name.replace("request_", "response_")), response
        )
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert stderr == ""
        assert len(stdout.splitlines()) == 1
        assert json.loads(stdout) == response


@pytest.mark.parametrize("reason", ["closed", "deadline"])
def test_terminal_session_does_not_read_sources_or_create_request(workspace, reason):
    mailbox = workspace / ".submissions"
    write_json(
        mailbox / "session.json",
        {"sources": ["missing.py"], "deadline_monotonic": time.monotonic() - 1},
    )
    expected = {"done": True, "reason": reason}
    if reason == "closed":
        write_json(mailbox / "closed.json", expected)
    with client(workspace) as process:
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert json.loads(stdout) == expected
    assert not list(mailbox.glob("request_*.json"))


@pytest.mark.parametrize("reason", ["closed", "deadline"])
def test_waiting_client_exits_when_session_ends(workspace, reason):
    mailbox = workspace / ".submissions"
    if reason == "deadline":
        session = json.loads((mailbox / "session.json").read_text())
        session["deadline_monotonic"] = time.monotonic() + 2
        write_json(mailbox / "session.json", session)
    expected = {"done": True, "reason": reason}
    with client(workspace) as process:
        wait_for_request(workspace, process)
        if reason == "closed":
            write_json(mailbox / "closed.json", expected)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert json.loads(stdout) == expected


@pytest.mark.parametrize("escape", ["relative", "absolute", "symlink"])
def test_source_paths_cannot_escape_workspace(workspace, escape):
    outside = workspace.parent / "outside.py"
    outside.write_text("must not be submitted\n")
    if escape == "symlink":
        (workspace / "escape.py").symlink_to(outside)
        name = "escape.py"
    else:
        name = "../outside.py" if escape == "relative" else str(outside)
    mailbox = workspace / ".submissions"
    write_json(
        mailbox / "session.json", {"sources": [name], "deadline_monotonic": None}
    )
    with client(workspace) as process:
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0
        assert "Source path must remain inside the workspace" in stderr
        assert stdout == ""
    assert not list(mailbox.glob("request_*.json"))


@pytest.mark.parametrize(
    "option", ["--timeout", "--repetitions", "--device", "--reference"]
)
def test_client_does_not_accept_evaluator_options(workspace, option):
    with client(workspace, option, "arbitrary") as process:
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 2
        assert f"unrecognized arguments: {option} arbitrary" in stderr
        assert stdout == ""
    assert not list((workspace / ".submissions").glob("request_*.json"))
