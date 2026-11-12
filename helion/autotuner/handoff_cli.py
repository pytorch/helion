"""Run one continuous agent CLI conversation with explicit native submissions."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import difflib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING
import uuid

from .handoff_evaluation import _artifact_path
from .handoff_gpu_run import MAILBOX as _GPU_MAILBOX
from .handoff_gpu_run import MAX_TIMEOUT as _GPU_MAX_TIMEOUT
from .handoff_prompt import _evaluation_summary
from .handoff_submit import _atomic_json

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import IO
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


def _agent_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(
            f"Native source optimization requires an authenticated {name.title()} CLI "
            "on PATH, or an explicit source-agent callback"
        )
    return executable


def _shell_words(variable: str) -> list[str]:
    """The environment variable ``variable`` split like a shell command."""
    try:
        return shlex.split(os.environ.get(variable, ""))
    except ValueError as e:
        raise ValueError(f"{variable} does not split like a shell command: {e}") from e


def _gpu_wrapper(wrapper: Sequence[str] | None) -> list[str]:
    """``wrapper``, else ``HELION_HANDOFF_GPU_WRAPPER``, with its executable
    resolved here: GPU commands run in the workspace, where a relative path
    would not resolve."""
    command = list(
        wrapper if wrapper is not None else _shell_words("HELION_HANDOFF_GPU_WRAPPER")
    )
    if command:
        executable = shutil.which(command[0])
        if executable is None:
            raise FileNotFoundError(f"Hand-off GPU wrapper {command[0]!r} not found")
        command[0] = os.path.abspath(executable)
    return command


def _agent_args(agent_args: Sequence[str] | None) -> list[str]:
    """``agent_args``, else ``HELION_HANDOFF_AGENT_ARGS``."""
    if agent_args is not None:
        return list(agent_args)
    return _shell_words("HELION_HANDOFF_AGENT_ARGS")


_GPU_OUTPUT_LIMIT = 200_000
# Seconds a stopped command's processes get to exit after SIGTERM (so a
# profiler can finish its report), then after SIGKILL.
_GPU_TERM_GRACE = 3.0
_GPU_STOP_GRACE = 10.0
# Claude's Bash tool times out after 2 minutes by default; let it wait for a
# GPU command (or a submission's measurement) to finish.
_CLAUDE_BASH_TIMEOUT_MS = str(int((_GPU_MAX_TIMEOUT + 600) * 1000))


def _measured_gpu(manifest: Mapping[str, Any]) -> str | None:
    """The ``CUDA_VISIBLE_DEVICES`` value naming the one CUDA device the
    bundle's cases run on, or ``None`` when that is not one CUDA device or
    other variables (ROCm's) also select devices."""
    devices: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "Tensor" and isinstance(value.get("device"), str):
                devices.add(value["device"])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for case in manifest["cases"]:
        visit(case.get("input_metadata"))
    accelerators = {device for device in devices if device != "cpu"}
    if len(accelerators) != 1 or any(
        name in os.environ for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    ):
        return None
    kind, _, index = accelerators.pop().partition(":")
    if kind != "cuda" or not index.isdigit():
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return index
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    return entries[int(index)] if int(index) < len(entries) else None


def _stat_fields(stat: Path) -> list[str]:
    """The fields of a ``/proc/<pid>/stat`` file after the command name."""
    # The command name in parentheses may contain anything.
    return stat.read_text().rsplit(")", 1)[1].split()


def _command_processes(pid: int) -> list[int]:
    """The live processes, other than ``pid``, in its process group or
    descended from one in it, so a stop also reaches those that left the
    group. Found through ``/proc`` (Linux; none elsewhere)."""
    children: dict[int, list[int]] = {}
    pending = [pid]
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, parent, group = _stat_fields(stat)[:3]
            child = int(stat.parent.name)
            children.setdefault(int(parent), []).append(child)
        except (OSError, ValueError):
            continue
        if state != "Z" and int(group) == pid:
            pending.append(child)
    found: set[int] = set()
    while pending:
        process = pending.pop()
        if process not in found:
            found.add(process)
            pending.extend(children.get(process, ()))
    found.discard(pid)
    return sorted(found)


def _alive(pid: int) -> bool:
    """Whether ``pid`` still runs (is not a zombie), as ``/proc`` tells."""
    try:
        return _stat_fields(Path(f"/proc/{pid}/stat"))[0] != "Z"
    except (OSError, IndexError):
        return False


def _arrival(path: Path) -> tuple[int, str]:
    try:
        return path.stat().st_mtime_ns, path.name
    except OSError:
        return 0, path.name


@dataclasses.dataclass
class _GPUCommand:
    """A running ``gpu_run.py`` command and the tail of its output."""

    entry: dict[str, Any]
    process: subprocess.Popen[bytes]
    reader: threading.Thread
    output: bytearray
    total: int
    timeout: float
    started: float
    deadline: float


class _GPUCommands:
    """Run the agent's ``gpu_run.py`` requests one at a time (see
    ``CLISourceAgent``).

    The session calls ``serve`` only while it waits for a submission and
    ``stop`` before returning one, so a running command never overlaps a
    measurement; ``close`` answers every request when the session ends.
    """

    def __init__(
        self,
        workspace: Path,
        record: Path,
        wrapper: Sequence[str],
        gpu: str | None = None,
    ) -> None:
        self._workspace = workspace
        self._wrapper = list(wrapper)
        self._log = record / "gpu_commands.jsonl"
        # A command's own autotuning must not start a nested hand-off.
        self._environment = {"HELION_AUTOTUNE_HANDOFF": "0"}
        if gpu is not None:
            self._environment["CUDA_VISIBLE_DEVICES"] = gpu
        if self._wrapper:
            # Only a wrapper needs the session's paths; a command should not
            # be pointed at the record, beside the bundle that scores it.
            self._environment["HELION_HANDOFF_WORKSPACE"] = str(workspace)
            self._environment["HELION_HANDOFF_RECORD"] = str(record)
        self._path(".").mkdir()
        self._answered: set[str] = set()
        self._running: _GPUCommand | None = None

    def _path(self, name: str) -> Path:
        return _artifact_path(self._workspace, f"{_GPU_MAILBOX}/{name}")

    @property
    def answered(self) -> bool:
        """Whether any request was answered."""
        return bool(self._answered)

    def _client_gone(self, request_id: str) -> bool:
        """Whether the client of ``request_id`` has ended, however it ended:
        it holds a lock on its lock file for as long as it lives."""
        try:
            descriptor = os.open(
                self._path(f"client_{request_id}.lock"), os.O_WRONLY | os.O_NONBLOCK
            )
        except (OSError, ValueError):
            return False
        try:
            os.lockf(descriptor, os.F_TLOCK, 0)
        except OSError:
            return False
        finally:
            # Also releases the lock.
            os.close(descriptor)
        return True

    def _requests(self) -> list[tuple[Path, str]]:
        """The unanswered requests, in arrival order."""
        paths = sorted(self._path(".").glob("request_*.json"), key=_arrival)
        return [
            (path, request_id)
            for path in paths
            if (request_id := path.stem.removeprefix("request_")) not in self._answered
        ]

    def serve(self) -> None:
        """Finish the running command if it is done, else start the next one."""
        running = self._running
        if running is None:
            self._start_next()
        elif running.process.poll() is not None:
            self._finish("completed")
        elif self._client_gone(running.entry["request_id"]):
            self._finish("cancelled: the client ended")
        elif time.monotonic() >= running.deadline:
            self._finish(f"timed out after {running.timeout:g} s")

    def stop(self, status: str) -> None:
        """End the running command, if any, and answer it with ``status``, or
        as completed if it just finished."""
        if self._running is not None:
            done = self._running.process.poll() is not None
            self._finish("completed" if done else status)

    def close(self, status: str) -> None:
        """End the running command and answer every request with ``status``."""
        try:
            self.stop(status)
        finally:
            for _, request_id in self._requests():
                self._respond(
                    {"request_id": request_id, "argv": None}, status, None, ""
                )
            _atomic_json(self._path("closed.json"), {"status": status})

    def _start_next(self) -> None:
        for path, request_id in self._requests():
            entry: dict[str, Any] = {"request_id": request_id, "argv": None}
            if self._client_gone(request_id):
                self._respond(entry, "cancelled: the client ended", None, "")
                continue
            try:
                if uuid.UUID(request_id).hex != request_id:
                    raise ValueError("invalid request identifier")
                request_path = self._path(path.name)
                # A FIFO would block this read, and with it the session.
                if not request_path.is_file():
                    raise ValueError("the request is not a regular file")
                request = json.loads(request_path.read_text())
                argv, timeout = request["argv"], float(request["timeout"])
                if not (
                    isinstance(argv, list)
                    and argv
                    and all(isinstance(word, str) for word in argv)
                ):
                    raise ValueError("argv must be a nonempty list of strings")
                if math.isnan(timeout):
                    raise ValueError("timeout must be a number")
                entry.update(argv=argv, cwd=request["cwd"])
                cwd = _artifact_path(self._workspace, request["cwd"])
                entry["waited_seconds"] = time.time() - path.stat().st_mtime
                self._running = self._launch(entry, argv, cwd, timeout)
                return
            except (OSError, ValueError, KeyError, TypeError) as error:
                self._respond(
                    entry, f"could not start: {type(error).__name__}: {error}", None, ""
                )

    def _launch(
        self, entry: dict[str, Any], argv: list[str], cwd: Path, timeout: float
    ) -> _GPUCommand:
        process = subprocess.Popen(
            [*self._wrapper, *argv],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            # The caller's autotune log (beside the bundle) is not the
            # command's to append to.
            env={
                **{k: v for k, v in os.environ.items() if k != "HELION_AUTOTUNE_LOG"},
                **self._environment,
            },
        )
        assert process.stdout is not None
        output = bytearray()

        def read(stdout: IO[bytes]) -> None:
            # The thread owns the pipe: it may outlive the command when a
            # detached process holds the pipe open.
            with stdout:
                for chunk in iter(lambda: os.read(stdout.fileno(), 65536), b""):
                    command.total += len(chunk)
                    output.extend(chunk)
                    del output[:-_GPU_OUTPUT_LIMIT]

        reader = threading.Thread(target=read, args=(process.stdout,), daemon=True)
        timeout = min(max(timeout, 1.0), _GPU_MAX_TIMEOUT)
        command = _GPUCommand(
            entry,
            process,
            reader,
            output,
            0,
            timeout,
            time.time(),
            time.monotonic() + timeout,
        )
        reader.start()
        return command

    def _finish(self, status: str) -> None:
        # The command stays running here until it is answered, so a stop that
        # is interrupted (Ctrl-C) leaves it for ``close`` to stop again.
        running = self._running
        assert running is not None
        process = running.process
        # End everything the command started, in its process group or not,
        # and wait for it to exit, so none of it overlaps a measurement.
        # Without /proc only the group is reached. A process the command
        # detached that outlived it is beyond reach; a wrapper can contain it.
        others = _command_processes(process.pid)

        def send(signum: int) -> None:
            # macOS refuses to signal a group left with only zombies (EPERM).
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signum)
            for pid in others:
                # One that cannot be killed counts as a survivor below.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signum)

        send(signal.SIGTERM)
        try:
            until = time.monotonic() + _GPU_TERM_GRACE
            while time.monotonic() < until and (
                process.poll() is None or any(map(_alive, others))
            ):
                time.sleep(0.01)
        finally:
            # Even when interrupted: one that ignores SIGTERM must not outlive
            # the session.
            send(signal.SIGKILL)
        until = time.monotonic() + _GPU_STOP_GRACE
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_GPU_STOP_GRACE)
        while time.monotonic() < until and any(map(_alive, others)):
            time.sleep(0.01)
        survivors = [pid for pid in others if _alive(pid)]
        if process.returncode is None:
            survivors.insert(0, process.pid)
        running.reader.join(timeout=1)
        output = bytes(running.output).decode(errors="replace")
        if running.total > _GPU_OUTPUT_LIMIT:
            output = (
                f"[gpu_run: output truncated to its last {_GPU_OUTPUT_LIMIT} bytes]\n"
                + output
            )
        if survivors:
            status = f"{status}; processes {survivors} did not exit"
        self._running = None
        self._respond(
            {**running.entry, "seconds": time.time() - running.started},
            status,
            process.returncode,
            output,
        )
        if survivors:
            raise RuntimeError(
                f"GPU command processes {survivors} did not exit after SIGKILL; "
                "ending the session rather than measuring beside them"
            )

    def _respond(
        self, entry: dict[str, Any], status: str, returncode: int | None, output: str
    ) -> None:
        request_id = entry["request_id"]
        self._answered.add(request_id)
        _atomic_json(
            self._path(f"response_{request_id}.json"),
            {"status": status, "returncode": returncode, "output": output},
        )
        # Keep the mailbox to unanswered requests; the client removes the
        # response once it has read it.
        for name in (f"request_{request_id}.json", f"client_{request_id}.lock"):
            with contextlib.suppress(OSError, ValueError):
                self._path(name).unlink()
        with self._log.open("a") as log:
            log.write(
                json.dumps({**entry, "status": status, "returncode": returncode}) + "\n"
            )


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
    """One agent process, yielding only explicitly submitted sources.

    The standalone ``submit_candidate.py`` command snapshots the allowed files
    and waits for validation feedback. ``complete_round`` restores the retained
    source before releasing that command. ``close`` ends and archives the single
    session; this instance never starts another process. ``deadline`` is an
    absolute ``time.monotonic()`` deadline for the entire session.

    The agent's own shell has no GPU (``CUDA_VISIBLE_DEVICES`` is empty). It
    runs GPU code with the standalone ``gpu_run.py`` command, which this session
    runs with the caller's environment and permissions, outside any sandbox of
    the agent CLI, on the one CUDA device the cases run on
    (``CUDA_VISIBLE_DEVICES``) where it can tell, one command at a time and only
    while it waits for a submission: a command waits while a submission is
    measured, and a submission stops a command that is still running, with its
    process group and descendants. A process a command detached that outlives it
    keeps running. Each command is logged to ``gpu_commands.jsonl`` in the
    session's record directory. ``gpu_wrapper`` (``HELION_HANDOFF_GPU_WRAPPER``,
    split like a shell command) prefixes every such command, for example to
    confine it in a sandbox with its own PID namespace, which also contains
    detached processes. It runs in the command's directory and receives the
    workspace and record directories as ``HELION_HANDOFF_WORKSPACE`` and
    ``HELION_HANDOFF_RECORD``, which it should not pass on to the command.

    ``agent`` selects ``codex`` (default) or ``claude``. The corresponding
    environment overrides are ``HELION_HANDOFF_AGENT``, ``HELION_HANDOFF_MODEL``,
    and ``HELION_HANDOFF_EFFORT``.

    ``agent_args`` (``HELION_HANDOFF_AGENT_ARGS``, split like a shell command)
    adds CLI arguments after Helion's own (before the ``-`` prompt for Codex),
    so they can override them where the CLI allows a repeated option, for
    example ``-c model_auto_compact_token_limit=N`` for a Codex model without a
    default compaction threshold, which ``--ignore-user-config`` otherwise
    rules out. The agent starts in its temporary workspace, where relative
    paths in them resolve.
    """

    def __init__(
        self,
        directory: Path,
        *,
        deadline: float | None = None,
        agent: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        gpu_wrapper: Sequence[str] | None = None,
        agent_args: Sequence[str] | None = None,
        log: AutotuningLogger | None = None,
    ) -> None:
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        self.agent = _agent_name(agent)
        self.executable = _agent_executable(self.agent)
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
        self.gpu_wrapper = _gpu_wrapper(gpu_wrapper)
        self.agent_args = _agent_args(agent_args)
        self.log = log
        self.stop_reason: str | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._workspace: Path | None = None
        self._record: Path | None = None
        self._process: subprocess.Popen[str] | None = None
        self._gpu: _GPUCommands | None = None
        self._initial_sources: dict[str, str] = {}
        self._pending: str | None = None
        self._notes: str | None = None
        self._seen: set[str] = set()
        self._started: float | None = None
        self._error: str | None = None
        self._session_id: str | None = None

    @staticmethod
    def validate_available(agent: str | None = None) -> str:
        """Resolve the existing CLI and check the GPU wrapper and argument
        settings, allowing callers to preflight before search."""
        executable = _agent_executable(_agent_name(agent))
        _gpu_wrapper(None)
        _agent_args(None)
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
                # Ignore prior memory and customizations; allow only local edits,
                # the submission command and GPU commands in this fresh
                # conversation.
                "--safe-mode",
                "--restricted",
                "--strict-mcp-config",
                "--permission-mode",
                "dontAsk",
                "--tools",
                "Read,Edit,Write,Glob,Grep,Bash",
                "--allowedTools",
                (
                    f"Read,Edit,Write,Glob,Grep,Bash({sys.executable} submit_candidate.py),"
                    f"Bash({sys.executable} gpu_run.py *)"
                ),
                "--model",
                self.model,
                "--effort",
                self.effort,
                *self.agent_args,
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
            *self.agent_args,
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
        shutil.copyfile(
            Path(__file__).with_name("handoff_gpu_run.py"),
            self._workspace / "gpu_run.py",
        )
        (self._workspace / ".submissions").mkdir()
        gpu = _measured_gpu(bundle.manifest)
        self._gpu = _GPUCommands(self._workspace, self._record, self.gpu_wrapper, gpu)
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
        # The command sees only the measured GPU, as its device 0.
        device_note = (
            " In a GPU command that GPU is the only one visible, cuda:0, whatever"
            " device prompt.md names."
            if gpu is not None
            else ""
        )
        request = f"""Improve these standalone native {backends} sources: {", ".join(self._initial_sources)}.
Read prompt.md and preserve all contracts. Edit the sources and optional notes.md.
Whenever a candidate is ready, run `{sys.executable} submit_candidate.py`.
The command validates and measures it through the caller, restores the retained
source, and prints feedback. Read the feedback and continue in this conversation
until it says done or the session deadline.
Run GPU code (tests, benchmarks, profilers) only with
`{sys.executable} gpu_run.py [--timeout SECONDS] -- COMMAND...`. It runs COMMAND
in the current directory on the GPU that measures your submissions and prints
its output once it ends. It waits while a submission is measured, and a
submission stops a command that is still running, so give your shell tool a
timeout longer than --timeout (default 600 s) plus that wait.{device_note}
Use only this workspace and conversation for history. Do not fetch external code,
launch subagents, install packages, commit, or push. Only edit the listed sources
and notes.md; leave the submission and GPU clients and their mailboxes unchanged.
"""
        (self._workspace / "AGENTS.md").write_text(request)
        (self._record / "request.txt").write_text(request)
        environment = {"CUDA_VISIBLE_DEVICES": ""}
        if self.agent == "claude":
            self._session_id = str(uuid.uuid4())
            # Claude otherwise skips canonical logs in nested headless sessions.
            environment["CLAUDE_CODE_FORCE_SESSION_PERSISTENCE"] = "1"
            environment["BASH_DEFAULT_TIMEOUT_MS"] = _CLAUDE_BASH_TIMEOUT_MS
            environment["BASH_MAX_TIMEOUT_MS"] = _CLAUDE_BASH_TIMEOUT_MS
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
                f"Native source session started: {self.agent}, {self.model} "
                f"({self.effort}); GPU commands run "
                + (
                    "through HELION_HANDOFF_GPU_WRAPPER"
                    if self.gpu_wrapper
                    else "unconfined"
                )
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
            assert (
                self._process is not None
                and self._workspace is not None
                and self._gpu is not None
            )
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
                    # A FIFO would block this read, and with it the session.
                    if not path.is_file():
                        raise ValueError("Submission is not a regular file")
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
                    self._gpu.stop("stopped: a submission started")
                    self._pending, self._notes = request_id, notes
                    return sources
                self._gpu.serve()
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

    def _close_gpu(self, errors: list[str]) -> None:
        assert self._gpu is not None
        try:
            self._gpu.close("stopped: the session ended")
        except (OSError, ValueError, RuntimeError) as error:
            # The agent controls the mailbox; record and carry on.
            errors.append(f"GPU commands: {type(error).__name__}: {error}")

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
            if self._gpu is not None:
                self._close_gpu(errors)
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
                    (self._seen or (self._gpu is not None and self._gpu.answered))
                    and self._process is not None
                    and self._process.poll() is None
                    and time.monotonic() < until
                ):
                    time.sleep(0.05)
            except (OSError, ValueError) as error:
                errors.append(f"Terminal feedback: {type(error).__name__}: {error}")
