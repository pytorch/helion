from __future__ import annotations

import hashlib
import json
import operator
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import pytest

from helion import Settings
from helion.autotuner import HandoffBundle
from helion.autotuner import HandoffCaseResult
from helion.autotuner import HandoffEvaluation
from helion.autotuner.handoff_cli import CLISourceAgent
from helion.autotuner.handoff_submit import _atomic_json
from helion.autotuner.logger import AutotuningLogger


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    cases = []
    for number, backend in enumerate(("cute", "triton")):
        source = f"case_{number}/kernel.py"
        path = directory / source
        path.parent.mkdir()
        path.write_text(f"original_{number}\n")
        cases.append({"source": source, "backend": backend})
    (directory / "manifest.json").write_text(
        json.dumps({"cases": cases, "raw_history": ["private manifest payload"]})
    )
    (directory / "prompt.md").write_text("Native contracts and current measurements.")
    return HandoffBundle(directory)


def submit(process, sources=None, *, notes="Source change rationale."):
    if sources is None:
        sources = {
            f"case_{number}/kernel.py": (
                process.workspace / f"case_{number}/kernel.py"
            ).read_text()
            for number in range(2)
        }
    request_id = uuid.uuid4().hex
    _atomic_json(
        process.workspace / ".submissions" / f"request_{request_id}.json",
        {"sources": sources, "notes": notes},
    )
    return request_id


@pytest.fixture(autouse=True)
def sessions(tmp_path, monkeypatch):
    directory = tmp_path / "fake_sessions"
    directory.mkdir()
    monkeypatch.setattr(
        "helion.autotuner.handoff_cli._codex_sessions_directory", lambda: directory
    )
    monkeypatch.setattr(
        "helion.autotuner.handoff_cli._claude_projects_directory", lambda: directory
    )
    for name in (
        "HELION_HANDOFF_AGENT",
        "HELION_HANDOFF_MODEL",
        "HELION_HANDOFF_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)
    return directory


def rollout_bytes(session_id, request, agent="codex"):
    if agent == "claude":
        rows = [
            {"type": "file-history-snapshot", "snapshot": {}},
            {
                "type": "user",
                "sessionId": session_id,
                "message": {"role": "user", "content": request},
            },
            {
                "type": "assistant",
                "sessionId": session_id,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "read_source",
                            "name": "Read",
                            "input": {"file_path": "kernel.py"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": session_id,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "read_source",
                            "content": "Full tool output: α\nNo truncation.",
                        }
                    ],
                },
            },
        ]
        return "".join(
            json.dumps(row, ensure_ascii=False) + "\r\n" for row in rows
        ).encode()
    rows = [
        {
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "base_instructions": {"text": "Initial system instructions."},
                "history_mode": "paginated",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": request}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "read_source",
                "name": "exec_command",
                "arguments": '{"cmd":"cat kernel.py"}',
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "read_source",
                "output": "Full tool output: α\nNo truncation.",
            },
        },
    ]
    return "".join(
        json.dumps(row, ensure_ascii=False) + "\r\n" for row in rows
    ).encode()


def write_rollout(sessions, session_id, request, agent="codex"):
    path = (
        sessions / "-tmp-native-agent" / f"{session_id}.jsonl"
        if agent == "claude"
        else sessions / "2026/10/05" / f"rollout-2026-10-05T00-00-00-{session_id}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rollout_bytes(session_id, request, agent))
    return path


@pytest.fixture(params=("codex",))
def runtime(monkeypatch, sessions, request):
    provider = request.param
    monkeypatch.setenv("HELION_HANDOFF_AGENT", provider)
    state = SimpleNamespace(
        agent=provider,
        processes=[],
        signals=[],
        action=lambda process: submit(process),
        launch_error=None,
        poll_action=lambda process: None,
        clock=100.0,
        resist=False,
    )
    monkeypatch.setattr(
        "helion.autotuner.handoff_cli.shutil.which", lambda name: f"/mock/{name}"
    )
    monkeypatch.setattr(
        "helion.autotuner.handoff_cli.os.killpg",
        lambda pid, sig: state.signals.append((pid, sig)),
    )
    monkeypatch.setattr(
        "helion.autotuner.handoff_cli.time.monotonic", lambda: state.clock
    )
    monkeypatch.setattr(
        "helion.autotuner.handoff_cli.time.sleep",
        lambda seconds: setattr(state, "clock", state.clock + seconds),
    )

    class Process:
        def __init__(self, command, **kwargs):
            if state.launch_error is not None:
                raise state.launch_error
            self.command = command
            self.kwargs = kwargs
            self.workspace = Path(kwargs["cwd"])
            self.request = kwargs["stdin"].read()
            self.pid = 12345
            self.returncode = None
            self.waits = []
            self.session_id = (
                command[command.index("--session-id") + 1]
                if provider == "claude"
                else str(uuid.uuid4())
            )
            self.rollout = write_rollout(
                sessions, self.session_id, self.request, provider
            )
            kwargs["stdout"].write(
                json.dumps(
                    {"type": "system", "subtype": "init", "session_id": self.session_id}
                    if provider == "claude"
                    else {"type": "thread.started", "thread_id": self.session_id}
                )
                + "\n"
            )
            kwargs["stdout"].flush()
            state.processes.append(self)
            state.action(self)

        def poll(self):
            state.poll_action(self)
            return self.returncode

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if state.resist and timeout == 5 and self.returncode is None:
                raise subprocess.TimeoutExpired(self.command, timeout)
            if self.returncode is None:
                self.returncode = -signal.SIGKILL
            return self.returncode

    monkeypatch.setattr("helion.autotuner.handoff_cli.subprocess.Popen", Process)
    return state


def feedback(tmp_path, number=1, *, accepted=False, current=10.0):
    directory = tmp_path / f"round_{number:04d}"
    directory.mkdir(exist_ok=True)
    measured = {
        "ok": True,
        "perf": current,
        "unit": "ms",
        "cases": [
            {
                "index": 0,
                "status": "ok",
                "perf": current,
                "noise": 0.01,
                "error": None,
                "samples": [99.0],
                "private": "full raw provenance",
            }
        ],
        "private": "full evaluation provenance",
    }
    return {
        "round": number + 1,
        "incumbent_evaluation": measured,
        "history": [
            {
                "round": number,
                "accepted": accepted,
                "reason": "accepted" if accepted else "regression_or_noise",
                "candidate": measured,
                "incumbent": measured,
                "error": None,
                "directory": str(directory),
                "improvement_threshold": 0.03,
            }
        ],
        "remaining_seconds": 42.0,
        "deadline_monotonic": 200.0,
    }


def read_response(process, request_id):
    mailbox = process.workspace / ".submissions"
    return json.loads((mailbox / f"response_{request_id}.json").read_text())


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
def test_one_workspace_and_process_across_explicit_submissions(
    bundle, tmp_path, runtime
):
    original = bundle.sources
    manifest = (bundle.directory / "manifest.json").read_bytes()
    old = tmp_path / "calls" / "old_session"
    old.mkdir(parents=True)
    (old / "notes.md").write_text("unrelated prior-session history")
    log = Mock()
    log.record_handoff_chat.return_value = None
    agent = CLISourceAgent(tmp_path / "calls", deadline=200, log=log)
    assert agent(bundle) == original
    process = runtime.processes[0]
    first_id = agent._pending
    assert process.workspace.exists()
    assert not (process.workspace / "manifest.json").exists()
    assert not (process.workspace / "history").exists()
    workspace_text = "\n".join(
        path.read_text() for path in process.workspace.rglob("*") if path.is_file()
    )
    assert "private manifest payload" not in workspace_text
    assert "unrelated prior-session history" not in workspace_text
    assert "manifest" not in process.request
    assert all(
        word not in process.request.lower()
        for word in ("autotune", "config", "tuning", "effort", "model")
    )
    assert "submit_candidate.py" in process.request
    assert "continue in this conversation" in process.request
    assert process.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert process.kwargs["start_new_session"] is True
    assert process.command[0] == f"/mock/{runtime.agent}"
    if runtime.agent == "codex":
        for flag in (
            "--ignore-user-config",
            "memories",
            "external_agent_memory_import",
        ):
            assert flag in process.command
        assert 'model_reasoning_effort="ultra"' in process.command
        assert "gpt-6-astra" in process.command
    else:
        for flag in (
            "--print",
            "--verbose",
            "--include-partial-messages",
            "--safe-mode",
            "--restricted",
            "--strict-mcp-config",
        ):
            assert flag in process.command
        for flag, value in (
            ("--output-format", "stream-json"),
            ("--session-id", process.session_id),
            ("--permission-mode", "dontAsk"),
            ("--tools", "Read,Edit,Write,Glob,Grep,Bash"),
            (
                "--allowedTools",
                f"Read,Edit,Write,Glob,Grep,Bash({sys.executable} submit_candidate.py)",
            ),
            ("--model", "opus"),
            ("--effort", "max"),
        ):
            assert process.command[process.command.index(flag) + 1] == value
        assert str(uuid.UUID(process.session_id)) == process.session_id
        assert process.kwargs["env"]["CLAUDE_CODE_FORCE_SESSION_PERSISTENCE"] == "1"
    assert not {
        "--ephemeral",
        "--no-session-persistence",
        "resume",
        "fork",
        "--resume",
        "--continue",
        "--fork-session",
    }.intersection(process.command)
    with pytest.raises(RuntimeError, match="pending submission"):
        agent(bundle)

    (process.workspace / "case_0/kernel.py").write_text("rejected edit")
    first = feedback(tmp_path)
    agent.complete_round(original, first, done=False)
    assert (process.workspace / "case_0/kernel.py").read_text() == original[
        "case_0/kernel.py"
    ]
    result = read_response(process, first_id)
    assert result["accepted"] is False and result["done"] is False
    assert "samples" not in json.dumps(result) and "private" not in json.dumps(result)
    assert "history" not in result and "directory" not in result
    assert (
        Path(first["history"][-1]["directory"]) / "notes.md"
    ).read_text() == "Source change rationale."

    accepted = {**original, "case_0/kernel.py": "accepted native source\n"}
    (process.workspace / "case_0/kernel.py").write_text(accepted["case_0/kernel.py"])
    second_id = submit(process, accepted, notes="Accepted source notes.")
    assert agent(bundle) == accepted
    runtime.clock = 200.0
    final_feedback = feedback(tmp_path, 2, accepted=True, current=9)
    final_feedback["remaining_seconds"] = 0.0
    agent.complete_round(accepted, final_feedback, done=True)
    assert (process.workspace / "case_0/kernel.py").read_text() == accepted[
        "case_0/kernel.py"
    ]
    result = read_response(process, second_id)
    assert result["accepted"] is True and result["done"] is True
    assert result["current"]["perf"] == 9
    agent.close(reason="deadline")
    agent.close(reason="ignored")
    with pytest.raises(StopIteration):
        agent(bundle)
    assert len(runtime.processes) == 1
    assert not process.workspace.exists()
    assert bundle.sources == original
    assert (bundle.directory / "manifest.json").read_bytes() == manifest
    record = next(path for path in (tmp_path / "calls").iterdir() if path != old)
    assert (record / "prompt.md").read_text() == bundle.prompt
    assert (record / "proposal/case_0/kernel.py").read_text() == accepted[
        "case_0/kernel.py"
    ]
    assert "+accepted native source" in (record / "change.diff").read_text()
    assert len(list((record / "submissions").glob("request_*.json"))) == 2
    outcome = json.loads((record / "outcome.json").read_text())
    assert outcome["agent"] == runtime.agent
    assert (
        json.loads((record / "invocation.json").read_text())["agent"] == runtime.agent
    )
    assert outcome["submissions"] == 2 and outcome["stop_reason"] == "deadline"
    assert log.record_handoff_event.call_args_list[0].kwargs["agent"] == runtime.agent
    assert [call.args[0] for call in log.record_handoff_event.call_args_list] == [
        "handoff_agent_started",
        "handoff_agent_completed",
    ]
    assert runtime.signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
@pytest.mark.parametrize("code,reason", [(0, "agent_completed"), (7, "agent_error")])
def test_exit_never_submits_final_files_or_respawns(
    bundle, tmp_path, runtime, code, reason
):
    def exit_after_edit(process):
        (process.workspace / "case_0/kernel.py").write_text("unsubmitted final edit")
        (process.workspace / "notes.md").write_text("partial notes")
        process.returncode = code

    runtime.action = exit_after_edit
    agent = CLISourceAgent(tmp_path / "calls")
    with pytest.raises(StopIteration):
        agent(bundle)
    with pytest.raises(StopIteration):
        agent(bundle)
    assert agent.stop_reason == reason
    assert len(runtime.processes) == 1
    record = agent._record
    assert (
        record / "proposal/case_0/kernel.py"
    ).read_text() == "unsubmitted final edit"
    assert (record / "notes.md").read_text() == "partial notes"
    assert json.loads((record / "outcome.json").read_text())["submissions"] == 0


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
def test_idle_deadline_reaps_term_resistant_group(bundle, tmp_path, runtime):
    runtime.action = lambda process: None
    runtime.resist = True
    agent = CLISourceAgent(tmp_path / "calls", deadline=100.2)
    with pytest.raises(StopIteration):
        agent(bundle)
    assert agent.stop_reason == "deadline"
    process = runtime.processes[0]
    assert process.waits == [5, None]
    assert runtime.signals[-1] == (process.pid, signal.SIGKILL)
    assert not process.workspace.exists()
    assert json.loads((agent._record / "outcome.json").read_text())["submissions"] == 0


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
def test_deadline_expiring_during_preparation_prevents_launch(
    bundle, tmp_path, runtime, monkeypatch
):
    real_copy = shutil.copyfile

    def copy(*args, **kwargs):
        result = real_copy(*args, **kwargs)
        runtime.clock = 102
        return result

    monkeypatch.setattr("helion.autotuner.handoff_cli.shutil.copyfile", copy)
    agent = CLISourceAgent(tmp_path / "calls", deadline=101)
    with pytest.raises(StopIteration):
        agent(bundle)
    assert agent.stop_reason == "deadline"
    assert runtime.processes == []
    assert not agent._workspace.exists()


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
@pytest.mark.parametrize("failure", ["launch", "interrupt", "poll", "protocol"])
def test_failures_archive_and_never_restart(bundle, tmp_path, runtime, failure):
    expected = OSError
    if failure == "launch":
        runtime.launch_error = OSError("launch failed")
    elif failure == "interrupt":
        expected = KeyboardInterrupt

        def interrupt(process):
            raise KeyboardInterrupt

        runtime.poll_action = interrupt
    elif failure == "poll":

        def poll_error(process):
            raise OSError("poll failed")

        runtime.poll_action = poll_error
    else:
        expected = ValueError

        def invalid(process):
            _atomic_json(
                process.workspace / ".submissions" / f"request_{uuid.uuid4().hex}.json",
                {"sources": bundle.sources, "notes": None, "device": 1},
            )

        runtime.action = invalid
    agent = CLISourceAgent(tmp_path / "calls")
    with pytest.raises(expected):
        agent(bundle)
    with pytest.raises(StopIteration):
        agent(bundle)
    assert not agent._workspace.exists()
    assert (agent._record / "proposal/case_1/kernel.py").read_text() == "original_1\n"
    outcome = json.loads((agent._record / "outcome.json").read_text())
    assert outcome["error"].startswith(expected.__name__)
    assert len(runtime.processes) == (0 if failure == "launch" else 1)
    if failure != "launch":
        assert (
            Path(outcome["chat_path"]).read_bytes()
            == runtime.processes[0].rollout.read_bytes()
        )


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
def test_interrupted_submission_does_not_receive_previous_acceptance(
    bundle, tmp_path, runtime
):
    agent = CLISourceAgent(tmp_path / "calls")
    agent(bundle)
    process = runtime.processes[0]
    first_id = agent._pending
    agent.complete_round(bundle.sources, feedback(tmp_path, accepted=True), done=False)
    read_response(process, first_id)
    second_id = submit(process)
    agent(bundle)
    agent.close(reason="interrupted")
    response = json.loads(
        (agent._record / "submissions" / f"response_{second_id}.json").read_text()
    )
    assert response == {"done": True, "reason": "interrupted"}


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
@pytest.mark.parametrize("failure", ["outcome", "logger", "reap"])
def test_cleanup_survives_artifact_or_logging_failure(
    bundle, tmp_path, runtime, monkeypatch, failure
):
    log = Mock()
    log.record_handoff_chat.return_value = None
    agent = CLISourceAgent(tmp_path / "calls", log=log)
    agent(bundle)
    if failure == "outcome":

        def fail_write(path, value):
            if path.name == "outcome.json":
                raise OSError("disk write failed")
            _atomic_json(path, value)

        monkeypatch.setattr("helion.autotuner.handoff_cli._atomic_json", fail_write)
    elif failure == "logger":
        log.record_handoff_event.side_effect = OSError("logger failed")
    else:
        process = runtime.processes[0]
        original_wait = process.wait

        def fail_reap(timeout=None):
            result = original_wait(timeout)
            if timeout is None:
                raise OSError("reap failed")
            return result

        process.wait = fail_reap
    with pytest.raises(OSError):
        agent.close(reason="interrupted")
    assert not agent._workspace.exists()
    assert runtime.signals[-1][1] == signal.SIGKILL
    if failure == "reap":
        outcome = json.loads((agent._record / "outcome.json").read_text())
        assert outcome["stop_reason"] == "interrupted"
        assert Path(outcome["chat_path"]).read_bytes() == process.rollout.read_bytes()


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
def test_archive_preserves_other_sources_after_deleted_or_nonutf8_edit(
    bundle, tmp_path, runtime
):
    agent = CLISourceAgent(tmp_path / "calls")
    agent(bundle)
    (agent._workspace / "case_0/kernel.py").unlink()
    (agent._workspace / "case_1/kernel.py").write_bytes(b"invalid\xff\n")
    (agent._workspace / "notes.md").write_bytes(b"notes\xff")
    agent.close(reason="interrupted")
    assert (
        agent._record / "proposal/case_1/kernel.py"
    ).read_bytes() == b"invalid\xff\n"
    assert (agent._record / "notes.md").read_bytes() == b"notes\xff"
    outcome = json.loads((agent._record / "outcome.json").read_text())
    assert "case_0/kernel.py" in outcome["archive_errors"][0]


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
def test_symlinked_temporary_directory_is_resolved(
    bundle, tmp_path, runtime, monkeypatch
):
    real_temporary = tempfile.TemporaryDirectory
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        "helion.autotuner.handoff_cli.tempfile.TemporaryDirectory",
        lambda **kwargs: real_temporary(dir=link, **kwargs),
    )
    agent = CLISourceAgent(tmp_path / "calls")
    assert agent(bundle) == bundle.sources
    assert agent._workspace.parent == target
    agent.close(reason="interrupted")


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
def test_model_and_effort_defaults_overrides_and_missing_cli(
    tmp_path, runtime, monkeypatch
):
    agent = CLISourceAgent(tmp_path / "calls")
    assert (agent.model, agent.effort) == (
        ("gpt-6-astra", "ultra") if runtime.agent == "codex" else ("opus", "max")
    )
    monkeypatch.setenv("HELION_HANDOFF_MODEL", "environment-model")
    monkeypatch.setenv("HELION_HANDOFF_EFFORT", "environment-effort")
    agent = CLISourceAgent(tmp_path / "calls")
    assert (agent.model, agent.effort) == ("environment-model", "environment-effort")
    agent = CLISourceAgent(tmp_path / "calls", model="explicit", effort="high")
    assert (agent.model, agent.effort) == ("explicit", "high")
    monkeypatch.setattr("helion.autotuner.handoff_cli.shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError, match=f"(?i){runtime.agent} CLI"):
        CLISourceAgent(tmp_path / "calls")


def test_agent_selection_defaults_environment_and_explicit_override(
    tmp_path, runtime, monkeypatch
):
    monkeypatch.delenv("HELION_HANDOFF_AGENT")
    default = CLISourceAgent(tmp_path / "calls")
    assert default.agent == "codex" and default.executable == "/mock/codex"
    monkeypatch.setenv("HELION_HANDOFF_AGENT", "claude")
    assert CLISourceAgent.validate_available() == "/mock/claude"
    selected = CLISourceAgent(tmp_path / "calls")
    assert selected.agent == "claude" and selected.executable == "/mock/claude"
    explicit = CLISourceAgent(tmp_path / "calls", agent="codex")
    assert explicit.agent == "codex" and explicit.executable == "/mock/codex"
    assert CLISourceAgent.validate_available("codex") == "/mock/codex"


@pytest.mark.parametrize("explicit", [False, True])
def test_unknown_agent_fails_before_start(tmp_path, runtime, monkeypatch, explicit):
    monkeypatch.setenv("HELION_HANDOFF_AGENT", "unsupported")
    with pytest.raises(ValueError, match="(?i)agent"):
        CLISourceAgent.validate_available("unsupported" if explicit else None)
    with pytest.raises(ValueError, match="(?i)agent"):
        CLISourceAgent(
            tmp_path / "calls", **({"agent": "unsupported"} if explicit else {})
        )
    assert runtime.processes == []
    assert not (tmp_path / "calls").exists()


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_default_controller_with_one_real_fake_cli_session(
    bundle, tmp_path, monkeypatch, sessions, provider
):
    """Real subprocess/client/controller; only the GPU evaluation is replaced."""
    monkeypatch.setenv("HELION_HANDOFF_AGENT", provider)
    executable = tmp_path / provider
    trace = tmp_path / "fake_cli_trace.jsonl"
    session_id = str(uuid.uuid4())
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, subprocess, sys\n"
        f"provider = {provider!r}\n"
        f"sessions = pathlib.Path({str(sessions)!r})\n"
        f"trace = pathlib.Path({str(trace)!r})\n"
        f"session_id = {session_id!r}\n"
        "if provider == 'claude': session_id = sys.argv[sys.argv.index('--session-id') + 1]\n"
        "rollout = (sessions / '-tmp-native-agent' / (session_id + '.jsonl')) if provider == 'claude' else (sessions / '2026/10/05' / ('rollout-2026-10-05T00-00-00-' + session_id + '.jsonl'))\n"
        "rollout.parent.mkdir(parents=True, exist_ok=True)\n"
        "print(json.dumps({'type': 'system', 'subtype': 'init', 'session_id': session_id} if provider == 'claude' else {'type': 'thread.started', 'thread_id': session_id}), flush=True)\n"
        "def chat(value):\n"
        "    if provider == 'claude':\n"
        "        payload = value['payload']\n"
        "        if value['type'] == 'session_meta': value = {'type': 'system', 'sessionId': session_id, 'content': payload['base_instructions']['text']}\n"
        "        elif payload['type'] == 'message': value = {'type': 'user', 'sessionId': session_id, 'message': {'role': 'user', 'content': payload['content'][0]['text']}}\n"
        "        elif payload['type'] == 'function_call': value = {'type': 'assistant', 'sessionId': session_id, 'message': {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': payload['call_id'], 'name': 'Bash', 'input': {'command': payload['arguments']}}]}}\n"
        "        else: value = {'type': 'user', 'sessionId': session_id, 'message': {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': payload['call_id'], 'content': payload['output']}]}}\n"
        "    with rollout.open('a') as output: output.write(json.dumps(value) + '\\n')\n"
        "def record(value):\n"
        "    with trace.open('a') as output: output.write(json.dumps(value) + '\\n')\n"
        "record({'event': 'start', 'pid': os.getpid(), 'workspace': os.getcwd(), 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'session_id': session_id, 'rollout': str(rollout)})\n"
        "assert not pathlib.Path('manifest.json').exists()\n"
        "request = sys.stdin.read()\n"
        "rollout.write_text('')\n"
        "chat({'type': 'session_meta', 'payload': {'id': session_id, 'base_instructions': {'text': 'Initial system instructions.'}}})\n"
        "chat({'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': request}]}})\n"
        "record({'event': 'request', 'request': request})\n"
        "source = pathlib.Path('case_0/kernel.py')\n"
        "for candidate in ('slower', 'faster'):\n"
        "    before = source.read_text()\n"
        "    source.write_text(candidate)\n"
        "    pathlib.Path('notes.md').write_text('Notes for ' + candidate)\n"
        "    chat({'type': 'response_item', 'payload': {'type': 'function_call', 'call_id': candidate, 'name': 'exec_command', 'arguments': 'python submit_candidate.py'}})\n"
        "    result = subprocess.run([sys.executable, 'submit_candidate.py'], check=True, capture_output=True, text=True)\n"
        "    chat({'type': 'response_item', 'payload': {'type': 'function_call_output', 'call_id': candidate, 'output': result.stdout}})\n"
        "    response = json.loads(result.stdout)\n"
        "    record({'event': 'feedback', 'candidate': candidate, 'before': before, 'retained': source.read_text(), 'response': response})\n"
        "    print(json.dumps({'event': 'candidate_feedback', 'response': response}), flush=True)\n"
        "    if response['done']: break\n"
        "source.write_text('unsubmitted final edit')\n"
        "record({'event': 'exit'})\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    evaluated = []

    def evaluate(workspace, **kwargs):
        source = workspace.sources["case_0/kernel.py"]
        evaluated.append(source)
        perf = {"original_0\n": 10.0, "slower": 11.0, "faster": 9.0}[source]
        case = HandoffCaseResult(
            0,
            "ok",
            hashlib.sha256(source.encode()).hexdigest(),
            str(workspace.directory / "case_0/kernel.py"),
            "wall_clock",
            (perf - 0.01, perf, perf + 0.01),
            perf,
            0.01,
            None,
        )
        return HandoffEvaluation(
            True,
            perf,
            "ms",
            (case,),
            str(workspace.directory),
            "2026-10-05T00:00:00+00:00",
        )

    monkeypatch.setattr(HandoffBundle, "evaluate", evaluate)
    original_manifest = (bundle.directory / "manifest.json").read_bytes()
    prompt = bundle.prompt
    log_base = tmp_path / "autotune_logs" / "run"
    log = Mock(wraps=AutotuningLogger(Settings(autotune_log=str(log_base))))
    result = bundle.run_agent_rounds(budget_seconds=20, log=log)
    assert result.stop_reason == "agent_completed"
    assert [record.accepted for record in result.rounds] == [False, True]
    assert bundle.sources["case_0/kernel.py"] == "faster"
    assert "unsubmitted final edit" not in evaluated
    assert len(evaluated) == 5
    assert bundle.prompt == prompt
    assert (bundle.directory / "manifest.json").read_bytes() == original_manifest
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    starts = [record for record in records if record["event"] == "start"]
    assert len(starts) == 1 and starts[0]["gpu"] == ""
    session_id = starts[0]["session_id"]
    rollout = Path(starts[0]["rollout"])
    assert not Path(starts[0]["workspace"]).exists()
    sessions = list((Path(result.directory) / "calls").iterdir())
    assert len(sessions) == 1
    saved_prompt = (sessions[0] / "prompt.md").read_text()
    assert saved_prompt.startswith(prompt + "\n\nSession deadline")
    assert "Native baseline:" not in saved_prompt
    assert "Round feedback:" not in saved_prompt
    saved = sessions[0] / "submissions"
    responses = [json.loads(path.read_text()) for path in saved.glob("response_*.json")]
    responses.sort(key=operator.itemgetter("round"))
    assert [response["accepted"] for response in responses] == [False, True]
    assert [response["current"]["perf"] for response in responses] == [10, 9]
    assert not any(response["done"] for response in responses)
    replies = [record for record in records if record["event"] == "feedback"]
    assert [record["before"] for record in replies] == ["original_0\n", "original_0\n"]
    assert [record["retained"] for record in replies] == ["original_0\n", "faster"]
    assert (
        sessions[0] / "proposal/case_0/kernel.py"
    ).read_text() == "unsubmitted final edit"
    for number, candidate in enumerate(("slower", "faster"), 1):
        assert (
            Path(result.directory) / f"round_{number:04d}" / "notes.md"
        ).read_text() == f"Notes for {candidate}"
    outcome = json.loads((sessions[0] / "outcome.json").read_text())
    assert outcome["agent"] == provider
    assert outcome["submissions"] == 2
    assert outcome["session_id"] == session_id
    assert Path(outcome["chat_path"]).read_bytes() == rollout.read_bytes()
    published = log_base.with_suffix(f".handoff.{session_id}.chat.jsonl")
    assert Path(outcome["chat_log_path"]) == published
    assert published.read_bytes() == rollout.read_bytes()
    assert "Initial system instructions." in published.read_text()
    rows = list(map(json.loads, published.read_text().splitlines()))
    tool_outputs = (
        [
            {"call_id": item["tool_use_id"], "output": item["content"]}
            for row in rows
            if row["type"] == "user" and isinstance(row["message"]["content"], list)
            for item in row["message"]["content"]
            if item["type"] == "tool_result"
        ]
        if provider == "claude"
        else [
            row["payload"]
            for row in rows
            if row.get("payload", {}).get("type") == "function_call_output"
        ]
    )
    assert [item["call_id"] for item in tool_outputs] == ["slower", "faster"]
    assert [json.loads(item["output"])["accepted"] for item in tool_outputs] == [
        False,
        True,
    ]
    assert json.loads(tool_outputs[-1]["output"])["done"] is False
    log.record_handoff_chat.assert_called_once_with(
        sessions[0] / "chat.jsonl", session_id
    )
    assert outcome["archive_errors"] == []
    events = [call.args[0] for call in log.record_handoff_event.call_args_list]
    assert (
        events.count("handoff_agent_started")
        == events.count("handoff_agent_completed")
        == 1
    )
    assert events.count("handoff_round") == 2


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
def test_canonical_chat_copies_only_own_exact_bytes_after_reap(
    bundle, tmp_path, runtime, sessions
):
    other_id = str(uuid.uuid4())
    other = write_rollout(
        sessions, other_id, "Unrelated private conversation.", runtime.agent
    )
    other_bytes = other.read_bytes()
    log = Mock()
    log.record_handoff_chat.return_value = None
    agent = CLISourceAgent(tmp_path / "calls", log=log)
    agent(bundle)
    process = runtime.processes[0]
    original_wait = process.wait
    final_line = b'{"type":"event_msg","payload":{"type":"shutdown_complete"}}\r\n'

    def wait(timeout=None):
        result = original_wait(timeout)
        if timeout is None:
            with process.rollout.open("ab") as output:
                output.write(final_line)
        return result

    process.wait = wait
    # An incomplete stdout tail must not obscure the already-recorded thread ID.
    with (agent._record / "events.jsonl").open("a") as output:
        output.write('{"type":"item.started"')
    agent.close(reason="interrupted")
    saved = agent._record / "chat.jsonl"
    expected = (
        rollout_bytes(process.session_id, process.request, runtime.agent) + final_line
    )
    assert saved.read_bytes() == expected
    assert other.read_bytes() == other_bytes
    assert b"Unrelated private conversation" not in saved.read_bytes()
    rows = list(map(json.loads, saved.read_bytes().splitlines()))
    if runtime.agent == "codex":
        assert b"Initial system instructions" in saved.read_bytes()
        tool_output = rows[3]["payload"]["output"]
    else:
        tool_output = rows[3]["message"]["content"][0]["content"]
    assert "Full tool output: α\nNo truncation." in tool_output
    assert not process.workspace.exists()
    log.record_handoff_chat.assert_called_once_with(saved, process.session_id)
    outcome = json.loads((agent._record / "outcome.json").read_text())
    assert outcome["session_id"] == process.session_id
    assert outcome["chat_path"] == str(saved)
    assert outcome["chat_log_path"] is None
    assert outcome["archive_errors"] == []
    completed = log.record_handoff_event.call_args.kwargs
    assert completed["outcome"]["chat_path"] == str(saved)
    assert any(str(saved) in call.args[0] for call in log.call_args_list)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "empty",
        "wrong_id",
        "duplicate",
        "header_array",
        "payload_array",
        "broken_header",
        "broken_tail",
        "nonutf8_tail",
        "untyped_tail",
        "inherited_history",
        "missing_event",
        "event_array",
        "event_payload_missing",
        "invalid_uuid",
    ],
)
def test_chat_archive_failures_are_visible_without_masking_result(
    bundle, tmp_path, runtime, failure
):
    log = Mock()
    log.record_handoff_chat.return_value = None
    agent = CLISourceAgent(tmp_path / "calls", log=log)
    agent(bundle)
    process = runtime.processes[0]
    transcript = process.rollout
    events = agent._record / "events.jsonl"
    if failure == "missing":
        transcript.unlink()
    elif failure == "empty":
        transcript.write_bytes(b"")
    elif failure == "wrong_id":
        transcript.write_bytes(rollout_bytes(str(uuid.uuid4()), "Other conversation"))
    elif failure == "duplicate":
        transcript.with_name(
            f"rollout-duplicate-{process.session_id}.jsonl"
        ).write_bytes(transcript.read_bytes())
    elif failure == "header_array":
        transcript.write_bytes(b"[]\n")
    elif failure == "payload_array":
        transcript.write_bytes(b'{"type":"session_meta","payload":[]}\n')
    elif failure == "broken_header":
        transcript.write_bytes(b'{"type":"session_meta"')
    elif failure in ("broken_tail", "nonutf8_tail", "untyped_tail"):
        with transcript.open("ab") as output:
            output.write(
                {
                    "broken_tail": b'{"type":"response_item"',
                    "nonutf8_tail": b"\xff\n",
                    "untyped_tail": b"[]\n",
                }[failure]
            )
    elif failure == "inherited_history":
        transcript.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": process.session_id,
                        "history_base": {"thread_id": str(uuid.uuid4())},
                    },
                }
            )
            + "\n"
        )
    elif failure == "missing_event":
        events.write_text('{"type":"turn.started"}\n')
    elif failure == "event_array":
        events.write_text("[]\n")
    elif failure == "event_payload_missing":
        events.write_text('{"type":"thread.started"}\n')
    else:
        events.write_text('{"type":"thread.started","thread_id":"../other"}\n')
    agent.close(reason="agent_completed")
    outcome = json.loads((agent._record / "outcome.json").read_text())
    assert outcome["stop_reason"] == "agent_completed"
    assert any(
        "Canonical chat archive:" in error for error in outcome["archive_errors"]
    )
    assert any("archive warning" in call.args[0] for call in log.call_args_list)
    assert outcome["chat_path"] is None and outcome["chat_log_path"] is None
    assert not (agent._record / "chat.jsonl").exists()
    assert not process.workspace.exists()
    log.record_handoff_chat.assert_not_called()
    if failure in ("broken_tail", "nonutf8_tail", "untyped_tail"):
        partial = agent._record / "chat.invalid.jsonl"
        assert partial.read_bytes() == transcript.read_bytes()
        assert str(partial) in outcome["archive_errors"][-1]


@pytest.mark.parametrize("runtime", ["codex", "claude"], indirect=True)
def test_log_export_failure_preserves_local_chat_and_session_result(
    bundle, tmp_path, runtime
):
    log = Mock()
    log.record_handoff_chat.side_effect = OSError("export path is unwritable")
    agent = CLISourceAgent(tmp_path / "calls", log=log)
    agent(bundle)
    process = runtime.processes[0]
    agent.close(reason="deadline")
    outcome = json.loads((agent._record / "outcome.json").read_text())
    assert outcome["stop_reason"] == "deadline"
    assert Path(outcome["chat_path"]).read_bytes() == process.rollout.read_bytes()
    assert outcome["chat_log_path"] is None
    assert "export path is unwritable" in outcome["archive_errors"][-1]
    assert not process.workspace.exists()


@pytest.mark.parametrize("runtime", ["claude"], indirect=True)
@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "empty",
        "wrong_id",
        "mixed_session_ids",
        "missing_session_id",
        "duplicate",
        "broken_header",
        "broken_tail",
        "nonutf8_tail",
        "untyped_tail",
    ],
)
def test_claude_chat_archive_failures_preserve_result_and_invalid_bytes(
    bundle, tmp_path, runtime, failure
):
    log = Mock()
    log.record_handoff_chat.return_value = None
    agent = CLISourceAgent(tmp_path / "calls", log=log)
    agent(bundle)
    process = runtime.processes[0]
    transcript = process.rollout
    if failure == "missing":
        transcript.unlink()
    elif failure == "empty":
        transcript.write_bytes(b"")
    elif failure == "wrong_id":
        transcript.write_bytes(
            rollout_bytes(str(uuid.uuid4()), "Other conversation", "claude")
        )
    elif failure == "mixed_session_ids":
        with transcript.open("a") as output:
            output.write(
                json.dumps({"type": "user", "sessionId": str(uuid.uuid4())}) + "\n"
            )
    elif failure == "missing_session_id":
        transcript.write_bytes(b'{"type":"file-history-snapshot"}\n')
    elif failure == "duplicate":
        duplicate = transcript.parent.with_name("-tmp-other-project") / transcript.name
        duplicate.parent.mkdir()
        duplicate.write_bytes(transcript.read_bytes())
    elif failure == "broken_header":
        transcript.write_bytes(b'{"type":"user"')
    else:
        with transcript.open("ab") as output:
            output.write(
                {
                    "broken_tail": b'{"type":"user"',
                    "nonutf8_tail": b"\xff\n",
                    "untyped_tail": b"[]\n",
                }[failure]
            )
    agent.close(reason="agent_completed")
    outcome = json.loads((agent._record / "outcome.json").read_text())
    assert outcome["stop_reason"] == "agent_completed"
    assert any(
        "Canonical chat archive:" in error for error in outcome["archive_errors"]
    )
    assert outcome["chat_path"] is None and outcome["chat_log_path"] is None
    assert not (agent._record / "chat.jsonl").exists()
    assert not process.workspace.exists()
    log.record_handoff_chat.assert_not_called()
    if failure not in ("missing", "duplicate", "empty"):
        assert (
            agent._record / "chat.invalid.jsonl"
        ).read_bytes() == transcript.read_bytes()


@pytest.mark.parametrize("runtime", ["claude"], indirect=True)
def test_claude_uses_a_new_session_id_for_each_invocation(bundle, tmp_path, runtime):
    ids = []
    for _ in range(2):
        agent = CLISourceAgent(tmp_path / "calls")
        agent(bundle)
        process = runtime.processes[-1]
        ids.append(process.session_id)
        agent.close(reason="interrupted")
    assert len(set(ids)) == 2
    assert len(runtime.processes) == 2
