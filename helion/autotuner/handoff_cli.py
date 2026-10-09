"""Run one continuous agent CLI conversation with explicit native submissions."""

from __future__ import annotations

import contextlib
import datetime
import difflib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING
import uuid

from .handoff_evaluation import _artifact_path
from .handoff_prompt import _evaluation_summary
from .handoff_submit import _atomic_json

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from .handoff_bundle import HandoffBundle
    from .logger import AutotuningLogger


_AGENT_DEFAULTS = {
    "codex": ("gpt-6-astra", "ultra"),
    "claude": ("opus", "max"),
}


def _agent_name(agent: str | None) -> str:
    name = (
        agent if agent is not None else os.environ.get("HELION_HANDOFF_AGENT", "codex")
    )
    if name not in _AGENT_DEFAULTS:
        raise ValueError(f"Unknown handoff agent {name!r}; choose codex or claude")
    return name


def _codex_sessions_directory() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"


def _claude_projects_directory() -> Path:
    return (
        Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "projects"
    )


def _codex_session_id(record: Path) -> str:
    session_id = None
    with (record / "events.jsonl").open() as events:
        for line in events:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Malformed Codex stdout event")
            if event.get("type") == "thread.started":
                session_id = event.get("thread_id")
                break
    if not isinstance(session_id, str) or str(uuid.UUID(session_id)) != session_id:
        raise ValueError("No valid Codex thread.started session ID was recorded")
    return session_id


def _archive_chat(
    record: Path, agent: str, session_id: str | None = None
) -> tuple[str, Path]:
    """Copy this fresh session's canonical chat, never reconstruct stdout."""
    if agent == "codex":
        session_id = _codex_session_id(record)
        paths = list(
            _codex_sessions_directory().glob(f"*/*/*/rollout-*-{session_id}.jsonl")
        )
    else:
        if session_id is None or str(uuid.UUID(session_id)) != session_id:
            raise ValueError("No valid Claude session ID was recorded")
        paths = list(_claude_projects_directory().glob(f"*/{session_id}.jsonl"))
    # Use only the exact session UUID, never a newest-file heuristic.
    if len(paths) != 1:
        raise ValueError(
            f"Expected one canonical rollout for {session_id}; found {len(paths)}"
        )
    raw = paths[0].read_bytes()
    lines = raw.splitlines()
    if not lines:
        raise ValueError(f"Canonical rollout for {session_id} is empty")
    try:
        if agent == "codex":
            first = json.loads(lines[0])
            if (
                not isinstance(first, dict)
                or first.get("type") != "session_meta"
                or not isinstance(first.get("payload"), dict)
                or first["payload"].get("id") != session_id
            ):
                raise ValueError(
                    f"Canonical rollout metadata does not match {session_id}"
                )
            if first["payload"].get("history_base") or first["payload"].get(
                "forked_from_id"
            ):
                raise ValueError(
                    "Canonical rollout refers to another session's history"
                )
        matched_session = False
        for line in lines:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("type"), str):
                raise ValueError("Expected a typed canonical rollout record")
            if agent == "claude" and "sessionId" in row:
                if row["sessionId"] != session_id:
                    raise ValueError(
                        f"Canonical chat sessionId does not match {session_id}"
                    )
                matched_session = True
        if agent == "claude" and not matched_session:
            raise ValueError(f"Canonical chat has no sessionId matching {session_id}")
    except ValueError as error:
        partial = record / "chat.invalid.jsonl"
        partial.write_bytes(raw)
        raise ValueError(
            f"Malformed canonical rollout; raw bytes preserved at {partial}: {error}"
        ) from error
    destination = record / "chat.jsonl"
    destination.write_bytes(raw)
    return session_id, destination


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    """Reap the CLI and its descendants, including an already-exited leader."""
    try:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _archive_proposal(
    workspace: Path, record: Path, sources: Mapping[str, str]
) -> list[str]:
    """Keep partial and non-UTF8 edits without replacing the original failure."""
    errors: list[str] = []
    changes: list[bytes] = []
    for name, original in sources.items():
        try:
            proposed = _artifact_path(workspace, name).read_bytes()
            saved = _artifact_path(record / "proposal", name)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(proposed)
            changes.extend(
                difflib.diff_bytes(
                    difflib.unified_diff,
                    original.encode().splitlines(keepends=True),
                    proposed.splitlines(keepends=True),
                    fromfile=f"incumbent/{name}".encode(),
                    tofile=f"proposal/{name}".encode(),
                )
            )
        except (OSError, ValueError) as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    try:
        notes = _artifact_path(workspace, "notes.md")
        if notes.is_file():
            shutil.copyfile(notes, record / "notes.md")
    except (OSError, ValueError) as error:
        errors.append(f"notes.md: {type(error).__name__}: {error}")
    try:
        (record / "change.diff").write_bytes(b"".join(changes))
    except OSError as error:
        errors.append(f"change.diff: {type(error).__name__}: {error}")
    return errors


class CLISourceAgent:
    """One GPU-hidden agent process, yielding only explicitly submitted sources.

    The standalone ``submit_candidate.py`` command snapshots the allowed files
    and waits for validation feedback. ``complete_round`` restores the retained
    source before releasing that command. ``close`` ends and archives the single
    session; this instance never starts another process. ``deadline`` is an
    absolute ``time.monotonic()`` deadline for the entire session.

    ``agent`` selects ``codex`` (default) or ``claude``. The corresponding
    environment overrides are ``HELION_HANDOFF_AGENT``, ``HELION_HANDOFF_MODEL``,
    and ``HELION_HANDOFF_EFFORT``.
    """

    def __init__(
        self,
        directory: Path,
        *,
        deadline: float | None = None,
        agent: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        log: AutotuningLogger | None = None,
    ) -> None:
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        self.agent = _agent_name(agent)
        self.executable = self.validate_available(self.agent)
        default_model, default_effort = _AGENT_DEFAULTS[self.agent]
        self.directory = Path(directory).resolve()
        self.deadline = deadline
        self.model = (
            model
            if model is not None
            else os.environ.get("HELION_HANDOFF_MODEL", default_model)
        )
        self.effort = (
            effort
            if effort is not None
            else os.environ.get("HELION_HANDOFF_EFFORT", default_effort)
        )
        self.log = log
        self.stop_reason: str | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._workspace: Path | None = None
        self._record: Path | None = None
        self._process: subprocess.Popen[str] | None = None
        self._initial_sources: dict[str, str] = {}
        self._pending: str | None = None
        self._notes: str | None = None
        self._seen: set[str] = set()
        self._started: float | None = None
        self._error: str | None = None
        self._session_id: str | None = None

    @staticmethod
    def validate_available(agent: str | None = None) -> str:
        """Resolve the existing CLI, allowing callers to preflight before search."""
        name = _agent_name(agent)
        executable = shutil.which(name)
        if executable is None:
            raise FileNotFoundError(
                f"Native source optimization requires an authenticated {name.title()} CLI "
                "on PATH, or an explicit source-agent callback"
            )
        return executable

    def _expired(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def _mailbox_path(self, name: str) -> Path:
        assert self._workspace is not None
        return _artifact_path(self._workspace, f".submissions/{name}")

    def _command(self) -> list[str]:
        assert self._workspace is not None and self._record is not None
        if self.agent == "claude":
            assert self._session_id is not None
            return [
                self.executable,
                "--print",
                "--verbose",
                "--output-format",
                "stream-json",
                "--include-partial-messages",
                "--session-id",
                self._session_id,
                # Ignore prior memory and customizations; allow only local edits
                # and the submission command in this fresh conversation.
                "--safe-mode",
                "--restricted",
                "--strict-mcp-config",
                "--permission-mode",
                "dontAsk",
                "--tools",
                "Read,Edit,Write,Glob,Grep,Bash",
                "--allowedTools",
                f"Read,Edit,Write,Glob,Grep,Bash({sys.executable} submit_candidate.py)",
                "--model",
                self.model,
                "--effort",
                self.effort,
            ]
        return [
            self.executable,
            "exec",
            "--ignore-user-config",
            "--disable",
            "memories",
            "--disable",
            "external_agent_memory_import",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "-c",
            'approval_policy="never"',
            "-c",
            f"model_reasoning_effort={json.dumps(self.effort)}",
            "-m",
            self.model,
            "--json",
            "--color",
            "never",
            "--cd",
            str(self._workspace),
            "--output-last-message",
            str(self._record / "response.txt"),
            "-",
        ]

    def _start(self, bundle: HandoffBundle) -> None:
        self._started = time.monotonic()
        self._record = self.directory / f"session_{uuid.uuid4().hex}"
        self._record.mkdir(parents=True)
        self._initial_sources = bundle.sources
        self._temporary = tempfile.TemporaryDirectory(prefix="helion-source-agent-")
        self._workspace = Path(self._temporary.name).resolve()
        for name, source in self._initial_sources.items():
            path = _artifact_path(self._workspace, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        prompt = bundle.prompt
        (self._workspace / "prompt.md").write_text(prompt)
        (self._record / "prompt.md").write_text(prompt)
        shutil.copyfile(
            Path(__file__).with_name("handoff_submit.py"),
            self._workspace / "submit_candidate.py",
        )
        (self._workspace / ".submissions").mkdir()
        _atomic_json(
            self._mailbox_path("session.json"),
            {
                "sources": list(self._initial_sources),
                "deadline_monotonic": self.deadline,
            },
        )
        backends = ", ".join(
            sorted({case["backend"] for case in bundle.manifest["cases"]})
        )
        request = f"""Improve these standalone native {backends} sources: {", ".join(self._initial_sources)}.
Read prompt.md and preserve all contracts. Edit the sources and optional notes.md.
Whenever a candidate is ready, run `{sys.executable} submit_candidate.py`.
The command validates and measures it through the caller, restores the retained
source, and prints feedback. Read the feedback and continue in this conversation
until it says done or the session deadline. Do not run GPU code yourself.
Use only this workspace and conversation for history. Do not fetch external code,
launch subagents, install packages, commit, or push. Only edit the listed sources
and notes.md; leave the submission client and its mailbox unchanged.
"""
        (self._workspace / "AGENTS.md").write_text(request)
        (self._record / "request.txt").write_text(request)
        environment = {"CUDA_VISIBLE_DEVICES": ""}
        if self.agent == "claude":
            self._session_id = str(uuid.uuid4())
            # Claude otherwise skips canonical logs in nested headless sessions.
            environment["CLAUDE_CODE_FORCE_SESSION_PERSISTENCE"] = "1"
        command = self._command()
        _atomic_json(
            self._record / "invocation.json",
            {
                "agent": self.agent,
                "command": command,
                "deadline_monotonic": self.deadline,
                **environment,
            },
        )
        if self._expired():
            self.close(reason="deadline")
            raise StopIteration
        if self.log is not None:
            self.log.record_handoff_event(
                "handoff_agent_started",
                directory=str(self._record),
                prompt_path=str(self._record / "prompt.md"),
                request_path=str(self._record / "request.txt"),
                agent=self.agent,
                model=self.model,
                effort=self.effort,
            )
            self.log(
                f"Native source session started: {self.agent}, {self.model} ({self.effort})"
            )
        with (
            (self._record / "request.txt").open() as stdin,
            (self._record / "events.jsonl").open("w") as stdout,
            (self._record / "stderr.log").open("w") as stderr,
        ):
            self._process = subprocess.Popen(
                command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                text=True,
                cwd=self._workspace,
                start_new_session=True,
                env={**os.environ, **environment},
            )

    def __call__(self, bundle: HandoffBundle) -> Mapping[str, str]:
        if self.stop_reason is not None:
            raise StopIteration
        if self._pending is not None:
            raise RuntimeError(
                "Complete the pending submission before requesting another"
            )
        try:
            if self._expired():
                self.close(reason="deadline")
                raise StopIteration
            if self._started is None:
                self._start(bundle)
            assert self._process is not None and self._workspace is not None
            while True:
                if self._expired():
                    self.close(reason="deadline")
                    raise StopIteration
                returncode = self._process.poll()
                if returncode is not None:
                    self.close(
                        reason="agent_completed" if returncode == 0 else "agent_error"
                    )
                    raise StopIteration
                mailbox = self._mailbox_path("session.json").parent
                for path in sorted(mailbox.glob("request_*.json")):
                    request_id = path.stem.removeprefix("request_")
                    if request_id in self._seen:
                        continue
                    if uuid.UUID(request_id).hex != request_id:
                        raise ValueError("Invalid submission identifier")
                    path = self._mailbox_path(path.name)
                    payload = json.loads(path.read_text())
                    if not isinstance(payload, dict) or set(payload) != {
                        "sources",
                        "notes",
                    }:
                        raise ValueError(
                            "Submission must contain only sources and notes"
                        )
                    sources, notes = payload["sources"], payload["notes"]
                    if (
                        not isinstance(sources, dict)
                        or sources.keys() != self._initial_sources.keys()
                        or any(
                            not isinstance(source, str) for source in sources.values()
                        )
                        or (notes is not None and not isinstance(notes, str))
                    ):
                        raise ValueError(
                            "Submission must snapshot the listed native sources"
                        )
                    self._seen.add(request_id)
                    self._pending, self._notes = request_id, notes
                    return sources
                time.sleep(0.1)
        except StopIteration:
            raise
        except BaseException as error:
            self._error = f"{type(error).__name__}: {error}"
            self.close(
                reason="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "agent_error"
            )
            raise

    def _respond(self, response: dict[str, Any]) -> None:
        assert self._pending is not None
        _atomic_json(self._mailbox_path(f"response_{self._pending}.json"), response)
        self._pending = None
        self._notes = None

    def complete_round(
        self,
        retained_sources: Mapping[str, str],
        feedback: Mapping[str, Any],
        *,
        done: bool,
    ) -> None:
        """Restore retained files before releasing the submitting command."""
        if self.stop_reason is not None or self._pending is None:
            raise RuntimeError("No live submission awaits feedback")
        assert self._workspace is not None
        try:
            if retained_sources.keys() != self._initial_sources.keys():
                raise ValueError("Retained sources must match the session source list")
            latest = feedback["history"][-1]
            if self._notes is not None:
                # This path comes from the controller, never the submit command.
                (Path(latest["directory"]) / "notes.md").write_text(self._notes)
            for name, source in retained_sources.items():
                _artifact_path(self._workspace, name).write_text(source)
            response = {
                **{key: latest[key] for key in ("round", "accepted", "reason")},
                "current": _evaluation_summary(feedback["incumbent_evaluation"]),
                **{
                    key: _evaluation_summary(latest[key])
                    for key in ("candidate", "incumbent")
                    if latest[key] is not None
                },
                **({"error": latest["error"]} if latest["error"] else {}),
                **(
                    {"improvement_threshold": latest["improvement_threshold"]}
                    if latest["improvement_threshold"] is not None
                    else {}
                ),
                "remaining_seconds": feedback["remaining_seconds"],
                "done": done,
            }
            self._respond(response)
        except BaseException as error:
            self._error = f"{type(error).__name__}: {error}"
            self.close(
                reason="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "agent_error"
            )
            raise

    def _archive_session(self, errors: list[str]) -> None:
        assert self._record is not None and self._started is not None
        if self._workspace is not None:
            errors.extend(
                _archive_proposal(self._workspace, self._record, self._initial_sources)
            )
            try:
                shutil.copytree(
                    self._mailbox_path("session.json").parent,
                    self._record / "submissions",
                    symlinks=True,
                )
            except (OSError, ValueError) as error:
                errors.append(f"Submissions archive: {type(error).__name__}: {error}")
        chat: dict[str, str | None] = {
            "session_id": None,
            "chat_path": None,
            "chat_log_path": None,
        }
        try:
            session_id, transcript = _archive_chat(
                self._record, self.agent, self._session_id
            )
            chat.update(session_id=session_id, chat_path=str(transcript))
            if self.log is not None:
                published = self.log.record_handoff_chat(transcript, session_id)
                if published is not None:
                    chat["chat_log_path"] = str(published)
        except (OSError, ValueError) as error:
            errors.append(f"Canonical chat archive: {type(error).__name__}: {error}")
        outcome = {
            "agent": self.agent,
            **chat,
            "stop_reason": self.stop_reason,
            "error": self._error,
            "submissions": len(self._seen),
            "archive_errors": errors,
            "returncode": self._process.returncode
            if self._process is not None
            else None,
            "elapsed_seconds": time.monotonic() - self._started,
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        _atomic_json(self._record / "outcome.json", outcome)
        if self.log is not None:
            self.log.record_handoff_event(
                "handoff_agent_completed",
                directory=str(self._record),
                outcome=outcome,
            )
            self.log(
                f"Native source session completed: {self.stop_reason}; artifacts: {self._record}"
            )
            if chat["chat_path"] is not None:
                self.log(
                    f"Native source chat: {chat['chat_log_path'] or chat['chat_path']}"
                )
            for error in errors:
                self.log(f"Native source archive warning: {error}")

    def close(self, *, reason: str) -> None:
        """Publish terminal feedback, reap the process group, and archive once."""
        if self.stop_reason is not None:
            return
        self.stop_reason = reason
        errors: list[str] = []
        with contextlib.ExitStack() as cleanup:
            # Reverse callback order guarantees: reap, archive, remove workspace,
            # even when a preceding cleanup step raises.
            if self._temporary is not None:
                cleanup.callback(self._temporary.cleanup)
            if self._record is not None:
                cleanup.callback(self._archive_session, errors)
            if self._process is not None:
                cleanup.callback(_stop_process_group, self._process)
            if self._workspace is None:
                return
            terminal = {"done": True, "reason": self.stop_reason}
            if self._error is not None:
                terminal["error"] = self._error
            try:
                if self._pending is not None:
                    self._respond(terminal)
                _atomic_json(self._mailbox_path("closed.json"), terminal)
                # Allow the submit command to deliver terminal feedback and the
                # CLI to persist its tool result before killing the group.
                until = time.monotonic() + 1
                while (
                    self._seen
                    and self._process is not None
                    and self._process.poll() is None
                    and time.monotonic() < until
                ):
                    time.sleep(0.05)
            except (OSError, ValueError) as error:
                errors.append(f"Terminal feedback: {type(error).__name__}: {error}")
