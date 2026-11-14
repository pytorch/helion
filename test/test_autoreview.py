"""Tests of review selection, evidence capture and process cleanup using fake CLIs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextlib import suppress
from functools import partial
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import types

pytestmark = pytest.mark.cpu_only

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "autoreview.py"


def _load() -> types.ModuleType:
    helper = importlib.util.spec_from_file_location(
        "process_supervisor", SCRIPT.with_name("process_supervisor.py")
    )
    assert helper is not None and helper.loader is not None
    supervision = importlib.util.module_from_spec(helper)
    sys.modules["process_supervisor"] = supervision
    helper.loader.exec_module(supervision)
    spec = importlib.util.spec_from_file_location("autoreview", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


autoreview = _load()
supervision = autoreview.process_supervisor

WAITID_CAPABILITIES = ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
GIT = shutil.which("git")


def _git_init(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make path a git repository for the review's working-tree check. No
    user or system git config (commit signing, hooks) applies."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)


@pytest.fixture
def without_waitid(monkeypatch):
    # Model the older macOS Python API without changing os for the test's
    # independent lifetime assertions, subprocess module, or supervisor child.
    proxy = SimpleNamespace(
        **{
            name: getattr(os, name)
            for name in dir(os)
            if name not in WAITID_CAPABILITIES
        }
    )
    monkeypatch.setattr(supervision, "os", proxy)


@pytest.fixture
def replace_supervisor(monkeypatch):
    supervised = autoreview.supervised_command

    def replace(setup):
        bootstrap = (
            f"import runpy, sys\nsys.path.insert(0, {str(SCRIPT.parent)!r})\n"
            f"namespace = runpy.run_path({str(SCRIPT)!r})\n"
            + setup
            + "\nnamespace['process_supervisor']._supervise_command(int(sys.argv[1]), "
            "int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])\n"
        )

        @contextmanager
        def command(cmd):
            with supervised(cmd) as (argv, descriptors, status, control):
                if os.name == "posix":
                    argv = [sys.executable, "-c", bootstrap, *argv[3:]]
                yield argv, descriptors, status, control

        monkeypatch.setattr(autoreview, "supervised_command", command)

    return replace


def _fixture_lifeline(release: Path) -> str:
    # A test can release its own fake process even after the supervisor dies.
    # Unlike a saved PID, this private path cannot identify an unrelated process.
    # Leak assertions must run before finally touches the release file.
    return (
        "import time\nfrom pathlib import Path\n"
        "deadline = time.monotonic() + 30\n"
        f"while not Path({str(release)!r}).exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
    )


def _cleanup_owned_parent(proc: subprocess.Popen) -> None:
    # This test is the sole waiter. After poll returns None, an exiting child
    # stays unreaped and reserves its PID/PGID until our wait below.
    if proc.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)


def _wait_for_file(path: Path, timeout: float = 10) -> Path:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        assert time.monotonic() < deadline, (
            f"Timed out after {timeout:g}s waiting for test process output: {path}"
        )
        time.sleep(0.01)
    return path


@pytest.fixture
def captured_processes(monkeypatch):
    popen = autoreview.subprocess.Popen
    processes = []
    started = threading.Event()

    def capture(*args, **kwargs):
        proc = popen(*args, **kwargs)
        processes.append(proc)
        started.set()
        return proc

    def cleanup():
        # Keep failure paths bounded even when testing a broken parent revision.
        for proc in processes:
            if proc.poll() is None:
                if os.name == "posix":
                    with suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            proc.wait(timeout=10)

    monkeypatch.setattr(autoreview.subprocess, "Popen", capture)
    try:
        yield processes, started, cleanup
    finally:
        cleanup()


@pytest.fixture
def unresponsive_supervisor(monkeypatch):
    supervised = autoreview.supervised_command

    @contextmanager
    def command_without_supervisor(cmd):
        # Exercise parent-side cleanup when no supervisor honors its deadline
        # or termination handler. The caller still owns this process group.
        with supervised(cmd) as (argv, descriptors, status, control):
            # The stand-in inherits the exit writer but ignores stop requests.
            yield cmd, descriptors, None, control

    def enable():
        monkeypatch.setattr(
            autoreview, "supervised_command", command_without_supervisor
        )

    return enable


FAKE_CLAUDE = f"""#!{sys.executable}
import json, os, re, sys, time
args = sys.argv[1:]
prompt = args[args.index("-p") + 1]
tokens = re.findall(r"\\[AUTOREVIEW_COMPLETE_[0-9a-f]{{32}}\\]", prompt)
completion = tokens[-1] if tokens else "parent-has-no-completion-protocol"
history = os.environ["FAKE_CLAUDE_HISTORY"]
previous = [json.loads(line) for line in open(history)] if os.path.exists(history) else []
settings = {{name: os.environ.get(name) for name in ("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", "BASH_DEFAULT_TIMEOUT_MS", "BASH_MAX_TIMEOUT_MS", "HELION_AUTOREVIEW_SESSION")}}
with open(history, "a") as f:
    f.write(json.dumps({{"argv": args, "env": settings, "completion": completion}}) + "\\n")
result = os.environ.get("FAKE_CLAUDE_RESULT", "claude says LGTM")
mode = os.environ.get("FAKE_CLAUDE_COMPLETION", "current")
if mode == "current":
    result += "\\n" + completion
elif mode == "same-line":
    result += " " + completion
elif mode == "indented":
    result += "\\n  " + completion + "  \\n"
elif mode == "previous":
    result += "\\n" + (previous[-1]["completion"] if previous else completion)
elif mode == "trailing":
    result += "\\n" + completion + "\\nstill running"
elif mode == "empty":
    result = "\\n" + completion
payload_key = "FAKE_CLAUDE_FOLLOWUP_PAYLOAD" if "--resume" in sys.argv else "FAKE_CLAUDE_PAYLOAD"
payload = os.environ.get(payload_key)
if payload is not None:
    payload = payload.replace("<CURRENT_COMPLETION>", completion)
print(payload if payload is not None else json.dumps({{"result": result, "session_id": "s1", "is_error": False}}), flush=True)
time.sleep(float(os.environ.get("FAKE_CLAUDE_SLEEP", "0")))
"""
FAKE_CODEX = f"""#!{sys.executable}
import json, os, sys, time
args = sys.argv[1:]
with open(os.environ["FAKE_CODEX_ARGV"], "w") as f:
    json.dump(args, f)
with open(os.environ["FAKE_CODEX_HISTORY"], "a") as f:
    f.write(json.dumps(args) + "\\n")
with open(os.environ["FAKE_CODEX_ENV"], "w") as f:
    json.dump({{name: os.environ.get(name) for name in ("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", "BASH_DEFAULT_TIMEOUT_MS", "BASH_MAX_TIMEOUT_MS")}}, f)
count_path = os.environ["FAKE_CODEX_COUNT"]
count = int(open(count_path).read()) if os.path.exists(count_path) else 0
with open(count_path, "w") as f:
    f.write(str(count + 1))
results = json.loads(os.environ.get("FAKE_CODEX_RESULTS", "null"))
result = results[min(count, len(results) - 1)] if results else os.environ.get("FAKE_CODEX_RESULT", "codex says LGTM")
if "bwrap" in result or os.environ.get("FAKE_CODEX_MOUNT_FAILURE") == "1":
    print("bwrap: Can't bind mount /oldroot/ on /newroot/: Unable to remount recursively with correct flags: Invalid argument", file=sys.stderr)
with open(args[args.index("-o") + 1], "wb") as f:
    encoded = os.environ.get("FAKE_CODEX_RESULT_HEX")
    f.write(bytes.fromhex(encoded) if encoded is not None else result.encode())
time.sleep(float(os.environ.get("FAKE_CODEX_SLEEP", "0")))
sys.exit(int(os.environ.get("FAKE_CODEX_EXIT", "0")))
"""


@pytest.fixture
def review(tmp_path, monkeypatch, capsys):
    """Run autoreview.main with only the named fake reviewers on PATH; returns
    (exit status, stdout, stderr)."""
    monkeypatch.delenv(autoreview.SESSION_MARKER, raising=False)
    monkeypatch.setattr(autoreview, "LOGS_BASE", tmp_path / "logs")
    monkeypatch.setattr(autoreview, "REPO_ROOT", tmp_path)
    # The fakes' bookkeeping files, the logs and the bin directories are
    # ignored, so the working-tree check sees no change.
    _git_init(tmp_path, monkeypatch)
    (tmp_path / ".gitignore").write_text("/*\n!/.gitignore\n")

    def run(*installed: str, args: tuple[str, ...] = ()) -> tuple[int, str, str]:
        bin_dir = tmp_path / ("bin-" + "-".join(installed))
        bin_dir.mkdir(exist_ok=True)
        for name in installed:
            path = bin_dir / name
            path.write_text({"claude": FAKE_CLAUDE, "codex": FAKE_CODEX}[name])
            path.chmod(0o755)
        if not (bin_dir / "git").exists():
            (bin_dir / "git").symlink_to(GIT)
        monkeypatch.setenv("PATH", str(bin_dir))
        monkeypatch.setenv("FAKE_CODEX_ARGV", str(tmp_path / "codex-argv.json"))
        monkeypatch.setenv("FAKE_CODEX_HISTORY", str(tmp_path / "codex-history.jsonl"))
        monkeypatch.setenv("FAKE_CODEX_COUNT", str(tmp_path / "codex-count.txt"))
        monkeypatch.setenv("FAKE_CODEX_ENV", str(tmp_path / "codex-env.json"))
        monkeypatch.setenv(
            "FAKE_CLAUDE_HISTORY", str(tmp_path / "claude-history.jsonl")
        )
        meta = () if "--meta" in args else ("--no-meta",)
        monkeypatch.setattr(sys, "argv", ["autoreview.py", *meta, *args, "x.py"])
        code = autoreview.main()
        out = capsys.readouterr()
        return code, out.out, out.err

    return run


@pytest.mark.slow
def test_both_reviewers(review):
    code, out, err = review("claude", "codex")
    # Codex's review is not "LGTM", so Claude combines the two.
    assert code == 0 and "claude says LGTM" in out and "WARNING" not in err


@pytest.mark.slow
def test_codex_missing_runs_claude_alone_with_a_warning(review):
    code, out, err = review("claude")
    assert code == 0 and out.strip() == "claude says LGTM"
    assert "WARNING: the codex CLI is not installed, so only Claude reviewed" in err
    assert "Tell the user" in err


@pytest.mark.slow
def test_claude_missing_runs_codex_alone_with_a_warning(review):
    code, out, err = review("codex")
    assert code == 0 and out.strip() == "codex says LGTM"
    assert "WARNING: the claude CLI is not installed, so only Codex reviewed" in err
    assert "resume session" not in err


@pytest.mark.slow
def test_no_codex_is_a_choice_not_a_warning(review):
    code, out, err = review("claude", args=("--no-codex",))
    assert code == 0 and "WARNING" not in err


def test_neither_reviewer_fails(review):
    code, _, err = review()
    assert code == 2 and "neither the claude nor the codex CLI is installed" in err
    code, _, err = review("codex", args=("--no-codex",))
    assert code == 2 and "--no-codex was given" in err


def test_fix_needs_claude(review):
    with pytest.raises(SystemExit) as info:
        review("codex", args=("--fix",))
    assert info.value.code == 2


@pytest.mark.parametrize(
    "installed,args,answer,expected,missing",
    [
        (("claude", "codex"), (), "LGTM", ("claude", "codex"), None),
        (("claude",), (), "LGTM", ("claude",), "codex"),
        (("codex",), ("--codex-sandbox", "read-only"), "LGTM", ("codex",), "claude"),
        (("claude", "codex"), ("--no-codex",), "LGTM", ("claude",), None),
        (("claude", "codex"), (), "Fix the reported bug", ("claude", "codex"), None),
    ],
)
def test_main_selection_and_review_record_without_processes(
    tmp_path, monkeypatch, capsys, installed, args, answer, expected, missing
):
    """Keep CLI selection and commit-record coverage in the normal PR tier."""
    calls = []

    def claude_send(self, prompt, *, phase="review", cancelled=None):
        calls.append(("claude", prompt, phase))
        return answer

    def codex_review(prompt, log_path, meta, **kwargs):
        calls.append(("codex", prompt, kwargs["sandbox"]))
        return answer

    def forbidden_process(*args, **kwargs):
        raise AssertionError("fast main coverage must not spawn interpreters")

    monkeypatch.delenv(autoreview.SESSION_MARKER, raising=False)
    monkeypatch.setattr(autoreview, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(autoreview, "LOGS_BASE", tmp_path / "logs")
    monkeypatch.setattr(
        autoreview.shutil, "which", lambda name: name if name in installed else None
    )
    monkeypatch.setattr(autoreview.ClaudeSession, "send", claude_send)
    monkeypatch.setattr(autoreview, "run_codex_review", codex_review)
    monkeypatch.setattr(autoreview.subprocess, "Popen", forbidden_process)
    # The working-tree check runs git; test_tree_changed_during_review_is_not_a_pass
    # covers it.
    monkeypatch.setattr(autoreview, "tree_state", lambda: b"unchanged")
    monkeypatch.setattr(
        sys, "argv", ["autoreview.py", "--no-meta", *args, "origin/main..HEAD"]
    )
    assert autoreview.main() == 0
    output = capsys.readouterr()
    assert output.out.strip() == answer
    initial = [call for call in calls if "origin/main..HEAD" in call[1]]
    assert sorted(call[0] for call in initial) == sorted(expected)
    for name, _prompt, sandbox in initial:
        if name == "codex":
            assert sandbox == (
                "read-only" if "read-only" in args else "workspace-write"
            )
    if missing is None:
        assert "WARNING" not in output.err
    else:
        assert f"WARNING: the {missing} CLI is not installed" in output.err
    if answer == "LGTM":
        assert output.err.rstrip().endswith("autoreview passed: " + ", ".join(expected))
    else:
        assert "autoreview passed:" not in output.err
        assert [phase for name, _, phase in calls if name == "claude"] == [
            "review",
            "combination",
        ]


@pytest.mark.parametrize("change", [None, "write", "commit", "break-git"])
def test_tree_changed_during_review_is_not_a_pass(
    tmp_path, monkeypatch, capsys, change
):
    """codex may write (workspace-write) and claude skips permissions, so a
    review during which the tree changed is shown but exits 3 without the
    commit line; an unchanged tree still passes.  A reviewer that leaves git
    failing changed the tree too: exit 3, not the broken-tool exit 2."""

    def claude_send(self, prompt, *, phase="review", cancelled=None):
        return "LGTM"

    def codex_review(prompt, log_path, meta, **kwargs):
        if change == "write":
            (tmp_path / "written-by-codex.txt").write_text("x")
        elif change == "commit":  # leaves no diff, but moves HEAD
            subprocess.run(
                [
                    *("git", "-c", "user.name=x", "-c", "user.email=x@x"),
                    *("commit", "-q", "--allow-empty", "-m", "x"),
                ],
                cwd=tmp_path,
                check=True,
            )
        elif change == "break-git":
            (tmp_path / ".git" / "HEAD").write_text("garbage\n")
        return "LGTM"

    monkeypatch.delenv(autoreview.SESSION_MARKER, raising=False)
    monkeypatch.setattr(autoreview, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(autoreview, "LOGS_BASE", tmp_path / "logs")
    _git_init(tmp_path, monkeypatch)
    (tmp_path / ".gitignore").write_text("logs/\n")
    monkeypatch.setattr(autoreview.shutil, "which", lambda name: name)
    monkeypatch.setattr(autoreview.ClaudeSession, "send", claude_send)
    monkeypatch.setattr(autoreview, "run_codex_review", codex_review)
    monkeypatch.setattr(sys, "argv", ["autoreview.py", "--no-meta", "x.py"])
    code = autoreview.main()
    output = capsys.readouterr()
    assert output.out.strip() == "LGTM"  # the review itself is still shown
    if change is None:
        assert code == 0
        assert output.err.rstrip().endswith("autoreview passed: claude, codex")
    else:
        assert code == autoreview.TREE_CHANGED_EXIT
        assert "NOT A PASS: the working tree changed during the review" in output.err
        assert "autoreview passed:" not in output.err
        if change == "break-git":
            assert "failed" in output.err


def test_tree_state_covers_untracked_links_and_nested_repos(tmp_path, monkeypatch):
    monkeypatch.setattr(autoreview, "REPO_ROOT", tmp_path)
    _git_init(tmp_path, monkeypatch)
    (tmp_path / "nested").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path / "nested", check=True)
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    (tmp_path / "dirlink").symlink_to(tmp_path / "nested")
    before = autoreview.tree_state()
    assert autoreview.tree_state() == before
    (tmp_path / "dangling").unlink()
    (tmp_path / "dangling").symlink_to(tmp_path / "elsewhere")
    assert autoreview.tree_state() != before
    (tmp_path / "script.sh").write_text("x")
    before = autoreview.tree_state()
    (tmp_path / "script.sh").chmod(0o755)  # same contents, now executable
    assert autoreview.tree_state() != before


def test_tree_state_fails_outside_a_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(autoreview, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    with pytest.raises(autoreview.ReviewAbort, match="git .* failed"):
        autoreview.tree_state()


@pytest.mark.parametrize(
    "meta,no_sandbox,sandbox,launcher_flags,expected_sandbox",
    [
        (False, False, "read-only", [], "read-only"),
        (
            True,
            False,
            "workspace-write",
            ["--dangerously-enable-internet-mode"],
            "workspace-write",
        ),
        (False, True, "read-only", [], "danger-full-access"),
        (
            True,
            True,
            "workspace-write",
            [
                "--dangerously-enable-internet-mode",
                "--dangerously-disable-linux-sandbox",
            ],
            "danger-full-access",
        ),
    ],
)
def test_codex_command_without_processes(
    tmp_path, monkeypatch, meta, no_sandbox, sandbox, launcher_flags, expected_sandbox
):
    prompt = "Review the requested source."
    log_path = tmp_path / "codex.log"
    cancelled = threading.Event()
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert cmd[-2] == "-o"
        Path(cmd[-1]).write_text("LGTM")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def forbidden_process(*args, **kwargs):
        raise AssertionError("fast command coverage must not spawn a process")

    monkeypatch.setattr(autoreview, "run_subprocess", run)
    monkeypatch.setattr(autoreview.subprocess, "Popen", forbidden_process)
    assert (
        autoreview.run_codex_review(
            prompt,
            log_path,
            meta,
            no_sandbox=no_sandbox,
            sandbox=sandbox,
            model="test-codex-model",
            effort="high",
            log=lambda _: None,
            cancelled=cancelled,
        )
        == "LGTM"
    )
    ((cmd, kwargs),) = calls
    assert cmd[:-1] == [
        "codex",
        *launcher_flags,
        "-c",
        'model_reasoning_effort="high"',
        "exec",
        "-s",
        expected_sandbox,
        "-m",
        "test-codex-model",
        prompt,
        "-o",
    ]
    assert kwargs == {
        "label": "codex",
        "log_path": log_path,
        "log_prompt": prompt,
        "cancelled": cancelled,
    }
    assert not Path(cmd[-1]).exists()


@pytest.mark.parametrize(
    "meta,user_cap,expected_cap", [(False, None, "2500"), (True, "23", "23")]
)
def test_claude_command_and_foreground_env_without_processes(
    tmp_path, monkeypatch, meta, user_cap, expected_cap
):
    monkeypatch.setattr(autoreview, "SUBPROCESS_TIMEOUT_SECONDS", 2.5)
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", "0")
    monkeypatch.setenv("BASH_DEFAULT_TIMEOUT_MS", "17")
    if user_cap is None:
        monkeypatch.delenv("BASH_MAX_TIMEOUT_MS", raising=False)
    else:
        monkeypatch.setenv("BASH_MAX_TIMEOUT_MS", user_cap)
    names = (
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS",
        "BASH_DEFAULT_TIMEOUT_MS",
        "BASH_MAX_TIMEOUT_MS",
    )
    original = {name: os.environ.get(name) for name in names}
    cancelled = threading.Event()
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        completion = cmd[cmd.index("-p") + 1].rsplit("\n", 1)[-1]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {"result": "LGTM\n" + completion, "session_id": "session-1"}
            ),
            stderr="",
        )

    def forbidden_process(*args, **kwargs):
        raise AssertionError("fast command coverage must not spawn a process")

    monkeypatch.setattr(autoreview, "run_subprocess", run)
    monkeypatch.setattr(autoreview.subprocess, "Popen", forbidden_process)
    log_path = tmp_path / "claude.log"
    session = autoreview.ClaudeSession(
        log_path, meta, model="test-claude-model", effort="high"
    )
    prompts = ("Review the requested source.", "Combine the findings.")
    for prompt in prompts:
        assert session.send(prompt, cancelled=cancelled) == "LGTM"
    assert len(calls) == 2
    for index, (cmd, kwargs) in enumerate(calls):
        prompt = cmd[cmd.index("-p") + 1]
        assert prompt.startswith(prompts[index] + "\n\n")
        assert cmd == [
            "claude",
            *(["--dangerously-enable-internet-mode"] if meta else []),
            "--dangerously-skip-permissions",
            "--model",
            "test-claude-model",
            "--effort",
            "high",
            "-p",
            prompt,
            "--output-format",
            "json",
            *(["--resume", "session-1"] if index else []),
        ]
        assert kwargs == {
            "label": "claude",
            "timeout": 2.5,
            "log_path": log_path,
            "log_prompt": prompt,
            "cancelled": cancelled,
            "extra_env": {
                "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
                "BASH_MAX_TIMEOUT_MS": expected_cap,
            },
        }
    assert {name: os.environ.get(name) for name in names} == original


@pytest.mark.slow
def test_logs_record_each_reviewer(review, tmp_path):
    review("codex")
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert (log_dir / "codex-review.txt").read_text() == "codex says LGTM"
    assert not (log_dir / "claude-review.txt").exists()


@pytest.mark.slow
def test_codex_no_sandbox_uses_meta_launcher_flags_only_with_meta(review, tmp_path):
    argv_file = tmp_path / "codex-argv.json"
    review("codex", args=("--codex-no-sandbox",))
    argv = json.loads(argv_file.read_text())
    assert argv[argv.index("-s") + 1] == "danger-full-access"
    # Upstream Codex rejects the flags of Meta's launcher.
    assert "--dangerously-disable-linux-sandbox" not in argv
    assert "--dangerously-enable-internet-mode" not in argv
    review("codex", args=("--meta", "--codex-no-sandbox"))
    argv = json.loads(argv_file.read_text())
    assert "--dangerously-disable-linux-sandbox" in argv
    assert "--dangerously-enable-internet-mode" in argv
    review("codex")
    argv = json.loads(argv_file.read_text())
    assert argv[argv.index("-s") + 1] == "workspace-write"


@pytest.mark.slow
def test_a_passing_review_prints_the_commit_line(review, monkeypatch):
    # Findings (not exactly "LGTM"): no line.
    _, _, err = review("claude", "codex")
    assert "autoreview passed:" not in err
    monkeypatch.setenv("FAKE_CLAUDE_RESULT", "LGTM")
    monkeypatch.setenv("FAKE_CODEX_RESULT", "LGTM")
    _, out, err = review("claude", "codex")
    assert out.strip() == "LGTM"
    assert err.rstrip().endswith("\nautoreview passed: claude, codex")
    # The line names only the reviewers that actually ran, and comes last.
    for installed, args, line in [
        (("claude",), (), "autoreview passed: claude"),
        (("codex",), (), "autoreview passed: codex"),
        (("claude", "codex"), ("--no-codex",), "autoreview passed: claude"),
    ]:
        _, _, err = review(*installed, args=args)
        assert err.rstrip().endswith(f"\n{line}"), installed


def _task_notification(result: str) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": "s1",
            "origin": {"kind": "task-notification", "producer": "session-task"},
            "result_index": 1,
            "result": result,
        }
    )


@pytest.mark.slow
@pytest.mark.parametrize("result", ["The review above is unchanged.", "LGTM"])
def test_task_notification_cannot_be_the_review(review, monkeypatch, tmp_path, result):
    payload = _task_notification(result)
    monkeypatch.setenv("FAKE_CLAUDE_PAYLOAD", payload)
    monkeypatch.setenv("FAKE_CODEX_RESULT", "LGTM")
    code, out, err = review("claude", "codex")
    assert code == 2 and not out.strip()
    assert "received a task-notification" in err
    assert "autoreview passed:" not in err
    assert "--resume s1" in err
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert payload in (log_dir / "claude.log").read_text()
    assert not (log_dir / "claude-review.txt").exists()
    assert not (log_dir / "combined-review.txt").exists()


@pytest.mark.slow
@pytest.mark.parametrize("fix", [False, True])
def test_task_notification_aborts_combine_or_fix(review, monkeypatch, tmp_path, fix):
    monkeypatch.setenv("FAKE_CLAUDE_FOLLOWUP_PAYLOAD", _task_notification("LGTM"))
    if fix:
        monkeypatch.setenv("FAKE_CODEX_RESULT", "LGTM")
    code, _, err = review("claude", "codex", args=("--fix",) if fix else ())
    assert code == 2 and "received a task-notification" in err
    assert "autoreview passed:" not in err
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert (log_dir / "claude-review.txt").read_text() == "claude says LGTM"
    assert (log_dir / "claude.log").read_text().count("=== stdout ===") == 2
    assert not (log_dir / "combined-review.txt").exists()
    assert not (log_dir / "fix-result.txt").exists()


@pytest.mark.slow
def test_claude_foreground_contract_is_scoped_and_stripped(
    review, monkeypatch, tmp_path
):
    original = {
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "0",
        "BASH_DEFAULT_TIMEOUT_MS": "17",
        "BASH_MAX_TIMEOUT_MS": "23",
    }
    for name, value in original.items():
        monkeypatch.setenv(name, value)
    code, out, _ = review("claude", "codex")
    assert code == 0 and out.strip() == "claude says LGTM"
    history = [
        json.loads(line)
        for line in (tmp_path / "claude-history.jsonl").read_text().splitlines()
    ]
    assert len(history) == 2
    assert history[0]["completion"] != history[1]["completion"]
    for row in history:
        assert row["env"] == {
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            "BASH_DEFAULT_TIMEOUT_MS": "17",
            "BASH_MAX_TIMEOUT_MS": "23",
            "HELION_AUTOREVIEW_SESSION": "1",
        }
        args = row["argv"]
        assert args[args.index("--model") + 1] == "opus"
        assert args[args.index("--effort") + 1] == "xhigh"
    assert "--resume" not in history[0]["argv"]
    assert history[1]["argv"][-2:] == ["--resume", "s1"]
    assert json.loads((tmp_path / "codex-env.json").read_text()) == original
    assert {name: os.environ[name] for name in original} == original
    (log_dir,) = (tmp_path / "logs").iterdir()
    for name in ("claude-review.txt", "combined-review.txt"):
        assert (log_dir / name).read_text() == "claude says LGTM"
    assert all(
        row["completion"] in (log_dir / "claude.log").read_text() for row in history
    )


@pytest.mark.slow
def test_claude_foreground_tools_do_not_extend_send_deadline(
    review, monkeypatch, tmp_path
):
    monkeypatch.delenv("BASH_DEFAULT_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("BASH_MAX_TIMEOUT_MS", raising=False)
    monkeypatch.setattr(autoreview, "SUBPROCESS_TIMEOUT_SECONDS", 1)
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "30")
    code, out, err = review("claude")
    assert code == 2 and not out.strip() and "timed out after 1s" in err
    row = json.loads((tmp_path / "claude-history.jsonl").read_text())
    assert row["env"]["BASH_DEFAULT_TIMEOUT_MS"] is None
    assert row["env"]["BASH_MAX_TIMEOUT_MS"] == "1000"
    (log_dir,) = (tmp_path / "logs").iterdir()
    raw = (log_dir / "claude.log").read_text()
    assert row["completion"] in raw and "=== stdout ===" in raw
    assert not (log_dir / "claude-review.txt").exists()


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["same-line", "indented"])
def test_completion_accepts_harmless_whitespace(review, monkeypatch, mode):
    monkeypatch.setenv("FAKE_CLAUDE_COMPLETION", mode)
    code, out, _ = review("claude")
    assert code == 0 and out.strip() == "claude says LGTM"


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["missing", "trailing", "empty"])
def test_requested_turn_requires_completion(review, monkeypatch, tmp_path, mode):
    progress = "The review is nearly done. I'll report when the CPU lane finishes."
    monkeypatch.setenv("FAKE_CLAUDE_RESULT", progress)
    monkeypatch.setenv("FAKE_CLAUDE_COMPLETION", mode)
    code, out, err = review("claude")
    assert code == 2 and not out.strip()
    assert "did not confirm completion" in err and "--resume s1" in err
    assert "autoreview passed:" not in err
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert not (log_dir / "claude-review.txt").exists()
    assert "=== stdout ===" in (log_dir / "claude.log").read_text()


@pytest.mark.slow
def test_followup_cannot_reuse_previous_completion(review, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_CLAUDE_COMPLETION", "previous")
    code, out, err = review("claude", "codex")
    assert code == 2 and "did not confirm completion" in err
    assert "autoreview passed:" not in err
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert (log_dir / "claude-review.txt").read_text() == "claude says LGTM"
    assert not (log_dir / "combined-review.txt").exists()


@pytest.mark.slow
def test_incomplete_fix_warns_that_edits_may_be_present(review, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "FAKE_CLAUDE_FOLLOWUP_PAYLOAD",
        json.dumps({"result": "I started fixing the issue.", "session_id": "s1"}),
    )
    code, _, err = review("claude", args=("--fix",))
    assert code == 2 and "fix response did not confirm completion" in err
    assert "working-tree edits may be present" in err
    assert "result is unconfirmed" in err and "--resume s1" in err
    assert "autoreview passed:" not in err
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert (log_dir / "claude-review.txt").read_text() == "claude says LGTM"
    assert not (log_dir / "fix-result.txt").exists()


@pytest.mark.slow
def test_notification_cannot_confirm_completion(review, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "FAKE_CLAUDE_PAYLOAD", _task_notification("LGTM\n<CURRENT_COMPLETION>")
    )
    code, out, err = review("claude")
    assert code == 2 and not out.strip() and "received a task-notification" in err
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert not (log_dir / "claude-review.txt").exists()


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["missing", "current"])
def test_claude_error_remains_an_error_with_or_without_completion(
    review, monkeypatch, mode
):
    monkeypatch.setenv("FAKE_CLAUDE_RESULT", "ERROR: validation could not run")
    monkeypatch.setenv("FAKE_CLAUDE_COMPLETION", mode)
    code, out, err = review("claude")
    assert code == 2 and not out.strip()
    assert "ERROR: validation could not run" in err
    assert "did not confirm completion" not in err
    assert "autoreview passed:" not in err


def test_the_review_prompt_points_at_live_agents_md_sections():
    # The prompt names AGENTS.md sections; renaming one must update the prompt.
    agents = (SCRIPT.parents[1] / "AGENTS.md").read_text()
    prompt = autoreview.build_review_prompt(False)
    for section in ("Compatibility guardrails", "Test Budget Rules"):
        assert f'"{section}"' in prompt
        assert f"### {section}" in agents
    assert "commit message" in autoreview.build_review_prompt(True)


@pytest.mark.slow
def test_nonblocking_output_writer_does_not_lose_stderr():
    # Codex's synchronous stderr writer can inherit nonblocking output. A
    # burst must not turn into EAGAIN merely because the parent is busy.
    size = 8 * 1024 * 1024
    script = f"""
import os
os.set_blocking(2, False)
pending = memoryview(b'x' * {size})
while pending:
    pending = pending[os.write(2, pending):]
print('complete')
"""
    result = autoreview.run_subprocess([sys.executable, "-c", script], label="burst")
    assert result.returncode == 0
    assert result.stdout == "complete\n"
    assert result.stderr == "x" * size


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="observes Linux process lifetime")
@pytest.mark.parametrize("abort", [False, True])
@pytest.mark.parametrize("detached", [False, True])
def test_completion_or_abort_terminates_reviewer_descendants(tmp_path, abort, detached):
    release = tmp_path / "release"
    sleeper = _fixture_lifeline(release)
    pid_path = tmp_path / "descendant.json"
    script = f"""
import json, os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', {sleeper!r}], start_new_session={detached!r})
path = Path({str(pid_path)!r})
path.with_suffix('.tmp').write_text(json.dumps([os.getpid(), child.pid]))
path.with_suffix('.tmp').replace(path)
{sleeper if abort else ""}
"""
    cancelled = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            autoreview.run_subprocess,
            [sys.executable, "-c", script],
            label="descendants",
            timeout=5,
            cancelled=cancelled,
        )
        pids = []
        try:
            pids = json.loads(_wait_for_file(pid_path).read_text())
            if abort:
                cancelled.set()
                with pytest.raises(autoreview.ReviewAbort, match="cancelled"):
                    future.result(timeout=10)
            else:
                assert future.result(timeout=10).returncode == 0
            assert _wait_until_exited([pids[1]])
        finally:
            cancelled.set()
            release.touch()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="Linux child-adoption cleanup")
@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("handles_term", [False, True])
def test_cancellation_gives_descendants_term_grace(tmp_path, detached, handles_term):
    # The CLI ignores TERM. A detached worker only becomes our direct child
    # after the CLI is killed, so it must still get time to release its claim.
    release = tmp_path / "release"
    ready = tmp_path / "ready.json"
    receipt = tmp_path / "released.json"
    log_path = tmp_path / "grace.log"
    worker = f"""
import json, os, signal, sys, time
from pathlib import Path
term_count = 0
def terminate(signum, frame):
    global term_count
    term_count += 1
signal.signal(signal.SIGTERM, terminate if {handles_term!r} else signal.SIG_IGN)
ready = Path({str(ready)!r})
ready.with_suffix('.tmp').write_text(json.dumps([os.getppid(), os.getpid()]))
ready.with_suffix('.tmp').replace(ready)
print('worker ready before cancellation', flush=True)
deadline = time.monotonic() + 30
while not Path({str(release)!r}).exists() and time.monotonic() < deadline:
    if term_count:
        time.sleep(0.2)  # A cleanup handler needs time, not just signal delivery.
        Path({str(receipt)!r}).write_text(json.dumps({{'term_count': term_count}}))
        print('worker cleanup completed', flush=True)
        sys.exit(0)
    time.sleep(0.01)
"""
    script = (
        "import signal, subprocess, sys\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"subprocess.Popen([sys.executable, '-c', {worker!r}], "
        f"start_new_session={detached!r})\n" + _fixture_lifeline(release)
    )
    cancelled = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            autoreview.run_subprocess,
            [sys.executable, "-c", script],
            label="term grace",
            timeout=30,
            log_path=log_path,
            cancelled=cancelled,
        )
        try:
            pids = json.loads(_wait_for_file(ready).read_text())
            started = time.monotonic()
            cancelled.set()
            with pytest.raises(
                autoreview.ReviewCancelled, match=r"^\[term grace\] cancelled$"
            ):
                future.result(timeout=10)
            assert time.monotonic() - started < 9
            assert all(not Path(f"/proc/{pid}").exists() for pid in pids)
            raw = log_path.read_text()
            stdout = raw.split("\n=== stdout ===\n", 1)[1].split(
                "\n=== returncode ===", 1
            )[0]
            assert "worker ready before cancellation" in stdout
            if handles_term:
                assert json.loads(receipt.read_text()) == {"term_count": 1}
                assert "worker cleanup completed" in stdout
            else:
                assert not receipt.exists()
                assert "worker cleanup completed" not in stdout
        finally:
            cancelled.set()
            release.touch()


def _process_running(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    # A killed orphan can briefly remain a zombie until its adopter reaps it.
    return status.rpartition(") ")[2].split()[0] not in {"Z", "X"}


def _launch_parent(tmp_path: Path, launcher: str) -> subprocess.Popen:
    env = {**os.environ, "PATH": str(tmp_path)}
    env.pop(autoreview.SESSION_MARKER, None)
    return subprocess.Popen(
        [sys.executable, "-c", launcher],
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_until_exited(pids: list[int], timeout: float = 2) -> bool:
    deadline = time.monotonic() + timeout
    while any(_process_running(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.01)
    return not any(_process_running(pid) for pid in pids)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="Linux child-adoption cleanup")
def test_review_reaps_orphans_before_the_reviewer_exits(tmp_path):
    pid_path = tmp_path / "orphan.pid"
    release = tmp_path / "release"
    spawner = (
        "import subprocess, sys; from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True); "
        f"p = Path({str(pid_path)!r}); "
        "p.with_suffix('.tmp').write_text(str(child.pid)); "
        "p.with_suffix('.tmp').replace(p)"
    )
    script = (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        f"subprocess.run([sys.executable, '-c', {spawner!r}], check=True)\n"
        f"while not Path({str(release)!r}).exists():\n    time.sleep(0.01)\n"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            autoreview.run_subprocess,
            [sys.executable, "-c", script],
            label="orphan reaping",
            timeout=10,
        )
        try:
            process = Path(f"/proc/{int(_wait_for_file(pid_path).read_text())}")
            deadline = time.monotonic() + 3
            while process.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not future.done(), "the reviewer must still be running"
            assert not process.exists(), "orphan must be reaped, not left as a zombie"
        finally:
            release.touch()
            assert future.result(timeout=12).returncode == 0


@pytest.mark.slow
def test_timeout_preserves_partial_output(tmp_path):
    log_path = tmp_path / "timeout.log"
    script = (
        "import os, time; os.write(1, b'before timeout \\xe2\\x82'); "
        "os.write(2, b'diagnostic \\xe2\\x82'); time.sleep(30)"
    )
    with pytest.raises(autoreview.ReviewAbort, match="timed out"):
        autoreview.run_subprocess(
            [sys.executable, "-c", script],
            label="timeout",
            timeout=3,
            log_path=log_path,
        )
    log = log_path.read_bytes()
    assert b"=== stdout ===\nbefore timeout \xe2\x82" in log
    assert b"=== stderr ===\ndiagnostic \xe2\x82" in log
    assert b"=== returncode ===" in log


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="observes Linux process lifetime")
@pytest.mark.parametrize(
    "signal_name,ignore",
    [("SIGTERM", False), ("SIGHUP", False), ("SIGHUP", True), ("SIGKILL", False)],
)
def test_parent_termination_respects_signal_disposition(tmp_path, signal_name, ignore):
    pid_path = tmp_path / "reviewer.pid"
    release = tmp_path / "release"
    cleanup_release = tmp_path / "cleanup-release"
    cli = tmp_path / "codex"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys, time\nfrom pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {_fixture_lifeline(cleanup_release)!r}], start_new_session=True)\n"
        f"p = Path({str(pid_path)!r})\n"
        "p.with_suffix('.tmp').write_text(json.dumps([os.getpid(), child.pid]))\n"
        "p.with_suffix('.tmp').replace(p)\n"
        "deadline = time.monotonic() + 30\n"
        f"while time.monotonic() < deadline and not Path({str(release)!r}).exists():\n"
        "    time.sleep(0.01)\n"
        "Path(sys.argv[sys.argv.index('-o') + 1]).write_text('LGTM')\n"
    )
    cli.chmod(0o755)
    launcher = f"""
import runpy, signal, sys
from pathlib import Path
if {signal_name!r} != 'SIGKILL':
    signal.signal(
        getattr(signal, {signal_name!r}),
        signal.SIG_IGN if {ignore!r} else signal.SIG_DFL,
    )
import sys
sys.path.insert(0, {str(SCRIPT.parent)!r})
namespace = runpy.run_path({str(SCRIPT)!r})
state = namespace['main'].__globals__
state['REPO_ROOT'] = Path({str(tmp_path)!r})
state['LOGS_BASE'] = state['REPO_ROOT'] / 'logs'
sys.argv = ['autoreview.py', '--no-meta', 'x.py']
raise SystemExit(namespace['main']())
"""
    proc = _launch_parent(tmp_path, launcher)
    pids = []
    try:
        pids = json.loads(_wait_for_file(pid_path).read_text())
        signum = getattr(signal, signal_name)
        os.killpg(proc.pid, signum)
        if ignore:
            release.touch()
            assert proc.wait(timeout=10) == 0
        else:
            expected = -signum if signal_name == "SIGKILL" else 128 + signum
            assert proc.wait(timeout=10) == expected
        assert _wait_until_exited(pids)
    finally:
        cleanup_release.touch()
        release.touch()
        _cleanup_owned_parent(proc)


@pytest.mark.slow
def test_peer_failure_cancels_process_spawned_during_abort(
    monkeypatch, captured_processes, replace_supervisor
):
    supervisors, spawned, cleanup = captured_processes
    popen = autoreview.subprocess.Popen
    cancellation_events = []
    cancellations = []
    replace_supervisor("""
import selectors
# Hold startup before autoreview installs SIGTERM handlers. The stop request
# must survive this interval and reach the supervisor's cleanup protocol.
with selectors.DefaultSelector() as selector:
    selector.register(int(sys.argv[1]), selectors.EVENT_READ)
    assert selector.select(10), 'supervisor did not receive its stop request'
""")

    def delayed_popen(*args, **kwargs):
        proc = popen(*args, **kwargs)
        # Hold the already-created child before run_subprocess receives its
        # handle. Cancellation must still be observed once Popen returns.
        assert cancellation_events[0].wait(10), "peer cancellation was not signaled"
        return proc

    def fail_peer(cancelled):
        assert spawned.wait(10), "process supervisor was not spawned"
        raise autoreview.ReviewAbort("peer failed")

    def late_process(cancelled):
        cancellation_events.append(cancelled)
        with pytest.raises(
            autoreview.ReviewCancelled, match=r"^\[late\] cancelled$"
        ) as error:
            autoreview.run_subprocess(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                label="late",
                timeout=5,
                cancelled=cancelled,
            )
        cancellations.append(str(error.value))
        return "unreachable"

    monkeypatch.setattr(autoreview.subprocess, "Popen", delayed_popen)
    try:
        with pytest.raises(autoreview.ReviewAbort, match="peer failed"):
            autoreview.run_parallel(
                autoreview.Spinner(), {"late": late_process, "peer": fail_peer}
            )
        assert len(cancellations) == 1
        assert len(supervisors) == 1 and supervisors[0].returncode is not None
    finally:
        cleanup()


@pytest.mark.slow
def test_cleanup_failure_does_not_capture_an_outer_exception(monkeypatch):
    finish = autoreview.finish_proc

    def failed_finish(proc, label, stop_fd):
        finish(proc, label, stop_fd)
        raise autoreview.ReviewAbort("injected cleanup failure")

    monkeypatch.setattr(autoreview, "finish_proc", failed_finish)
    try:
        raise ValueError("unrelated outer exception")
    except ValueError:
        with pytest.raises(autoreview.ReviewAbort, match="^injected cleanup failure$"):
            autoreview.run_subprocess(
                [sys.executable, "-c", "pass"], label="outer exception"
            )


_MOUNT_ERROR = (
    "ERROR: bwrap could not initialize the sandbox: "
    "Unable to remount recursively with correct flags: Invalid argument"
)


@pytest.mark.slow
@pytest.mark.parametrize("error", [_MOUNT_ERROR, "ERROR: the sandbox failed to start"])
def test_transient_mount_failure_retries_same_review_settings(
    review, monkeypatch, tmp_path, error
):
    monkeypatch.setenv("FAKE_CODEX_RESULTS", json.dumps([error, "LGTM"]))
    monkeypatch.setenv("FAKE_CODEX_MOUNT_FAILURE", "1")
    code, out, err = review("codex", args=("--codex-sandbox", "workspace-write"))
    assert code == 0 and out.strip() == "LGTM"
    calls = [
        json.loads(line)
        for line in (tmp_path / "codex-history.jsonl").read_text().splitlines()
    ]
    assert len(calls) == 2
    assert calls[0][:-1] == calls[1][:-1]
    assert calls[0][-1] != calls[1][-1]
    assert "danger-full-access" not in calls[1]
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert error in (log_dir / "codex.log").read_text()
    assert "retry" in err.lower()


@pytest.mark.slow
@pytest.mark.parametrize(
    "error,attempts", [(_MOUNT_ERROR, 2), ("ERROR: permission denied", 1)]
)
def test_review_errors_fail_closed_after_bounded_attempts(
    review, monkeypatch, tmp_path, error, attempts
):
    monkeypatch.setenv("FAKE_CODEX_RESULT", error)
    code, out, err = review("codex")
    assert code == 2 and not out.strip()
    assert "autoreview passed:" not in err
    assert int((tmp_path / "codex-count.txt").read_text()) == attempts


@pytest.mark.slow
@pytest.mark.parametrize("returncode", [0, 1, "timeout"])
def test_codex_preserves_truncated_result_and_fails_closed(
    review, monkeypatch, tmp_path, returncode
):
    payload = b"partial \xe2\x82"
    monkeypatch.setenv("FAKE_CODEX_RESULT_HEX", payload.hex())
    if returncode == "timeout":
        monkeypatch.setenv("FAKE_CODEX_SLEEP", "30")
        monkeypatch.setattr(
            autoreview, "run_subprocess", partial(autoreview.run_subprocess, timeout=3)
        )
    else:
        monkeypatch.setenv("FAKE_CODEX_EXIT", str(returncode))
    code, out, err = review("codex")
    assert code == 2 and not out.strip()
    assert "autoreview passed:" not in err
    expected = {0: "invalid UTF-8", 1: "non-zero exit 1", "timeout": "timed out"}
    assert expected[returncode] in err
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert b"=== result ===\n" + payload in (log_dir / "codex.log").read_bytes()


@pytest.mark.slow
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"result": None},
        {"result": ""},
        {"result": "  "},
        {"result": 17},
        {"result": "LGTM", "session_id": 17},
    ],
)
def test_malformed_claude_response_fails_closed(review, monkeypatch, tmp_path, payload):
    monkeypatch.setenv("FAKE_CLAUDE_PAYLOAD", json.dumps(payload))
    code, out, err = review("claude")
    assert code == 2 and not out.strip()
    assert "[claude]" in err and "autoreview passed:" not in err
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert json.dumps(payload) in (log_dir / "claude.log").read_text()
    assert not (log_dir / "claude-review.txt").exists()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="observes Linux process lifetime")
def test_peer_cleanup_error_preserves_primary_and_cleans_other_peer(
    tmp_path, monkeypatch, capsys
):
    release = tmp_path / "release"
    ready = {label: tmp_path / f"{label}.pid" for label in ("denied", "clean")}
    finish = autoreview.finish_proc
    cleaned = {}
    cancellation_events = {}
    primary_error = autoreview.ReviewAbort("original peer failure")

    def finish_with_error(proc, label, stop_fd):
        # Actually retire both peers, then inject one cleanup failure at the
        # worker boundary. It must be reported without hiding the first error
        # or preventing the independent worker from completing its cleanup.
        finish(proc, label, stop_fd)
        cleaned[label] = proc
        if label == "denied":
            raise PermissionError("cleanup permission denied")

    def run_peer(label, cancelled):
        cancellation_events[label] = cancelled
        script = (
            "import os, time; from pathlib import Path; "
            f"p = Path({str(ready[label])!r}); "
            "p.with_suffix('.tmp').write_text(str(os.getpid())); "
            "p.with_suffix('.tmp').replace(p)\n" + _fixture_lifeline(release)
        )
        autoreview.run_subprocess(
            [sys.executable, "-c", script],
            label=label,
            timeout=5,
            cancelled=cancelled,
        )
        return "unreachable"

    def fail_peer(cancelled):
        cancellation_events["failure"] = cancelled
        for path in ready.values():
            _wait_for_file(path)
        raise primary_error

    monkeypatch.setattr(autoreview, "finish_proc", finish_with_error)
    try:
        with pytest.raises(autoreview.ReviewAbort) as error:
            autoreview.run_parallel(
                autoreview.Spinner(),
                {
                    "denied": partial(run_peer, "denied"),
                    "clean": partial(run_peer, "clean"),
                    "failure": fail_peer,
                },
            )
        assert error.value is primary_error
        assert len({id(event) for event in cancellation_events.values()}) == 1
        assert cancellation_events["failure"].is_set()
        assert set(cleaned) == {"denied", "clean"}
        assert all(proc.returncode is not None for proc in cleaned.values())
        assert _wait_until_exited([int(path.read_text()) for path in ready.values()])
        message = "denied failed during cancellation: cleanup permission denied"
        stderr = capsys.readouterr().err
        assert stderr.count(message) == 1
        assert "clean failed during cancellation" not in stderr
    finally:
        release.touch()


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor exit protocol")
@pytest.mark.parametrize(
    "script,expected",
    [
        ("raise SystemExit(7)", 7),
        ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", -signal.SIGTERM),
    ],
)
def test_supervisor_preserves_reviewer_failure_status(script, expected):
    result = autoreview.run_subprocess([sys.executable, "-c", script], label="status")
    assert result.returncode == expected


@pytest.mark.slow
@pytest.mark.parametrize("exists", [False, True])
def test_reviewer_launch_error_is_reported_as_launch_failure(tmp_path, exists):
    command = tmp_path / "reviewer"
    if exists:
        command.write_text("not executable")
    log_path = tmp_path / "launch.log"
    with pytest.raises(autoreview.ReviewAbort, match="launch failed") as error:
        autoreview.run_subprocess([str(command)], label="launch", log_path=log_path)
    assert str(command) in str(error.value)
    assert ("Permission denied" if exists else "No such file") in str(error.value)
    assert str(command) in log_path.read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-name bytes")
@pytest.mark.parametrize("unreadable", [False, True])
def test_cleanup_accepts_non_utf8_process_names(tmp_path, monkeypatch, unreadable):
    # An unrelated process name must not prevent our descendant cleanup.
    # Use saved /proc-format bytes to keep invalid names out of other agents'
    # concurrent process monitors on this shared machine.
    process = tmp_path / "123"
    process.mkdir()
    (process / "stat").write_bytes(b"123 (child\xff) S 0 1 1\n")
    iterdir = Path.iterdir
    read_bytes = Path.read_bytes

    def process_entries(path):
        return iter([process]) if path == Path("/proc") else iterdir(path)

    def process_stat(path):
        if unreadable and path == process / "stat":
            raise PermissionError("hidepid=1")
        return read_bytes(path)

    monkeypatch.setattr(Path, "iterdir", process_entries)
    monkeypatch.setattr(Path, "read_bytes", process_stat)
    assert supervision._reap_children()


@pytest.mark.parametrize("reap_time", [5.0, 5.5])
def test_cleanup_final_reap_at_deadline_succeeds(tmp_path, monkeypatch, reap_time):
    process = tmp_path / "123"
    process.mkdir()
    (process / "stat").write_bytes(b"123 (child) Z 100 123 1\n")
    now = 0.0
    reaped = False
    signals = []

    def process_entries(path):
        nonlocal now
        assert path == Path("/proc")
        now = reap_time
        return iter([process])

    def waitpid(pid, options):
        nonlocal reaped
        assert (pid, options) == (-1, os.WNOHANG)
        if reaped:
            raise ChildProcessError
        reaped = True
        return 123, 0

    def unexpected_sleep(seconds):
        pytest.fail("ECHILD already proves cleanup is complete")

    monkeypatch.setattr(Path, "iterdir", process_entries)
    monkeypatch.setattr(
        supervision,
        "os",
        SimpleNamespace(
            getpid=lambda: 100,
            getpgrp=lambda: 100,
            kill=lambda pid, sig: signals.append((pid, sig)),
            waitpid=waitpid,
            WNOHANG=os.WNOHANG,
        ),
    )
    monkeypatch.setattr(
        supervision,
        "time",
        SimpleNamespace(monotonic=lambda: now, sleep=unexpected_sleep),
    )
    assert supervision._reap_children()
    assert reaped and (123, signal.SIGKILL) in signals


@pytest.mark.parametrize("final_reap_delay", [0.0, 0.1, None])
def test_cleanup_drains_late_adopted_generations(
    tmp_path, monkeypatch, capsys, final_reap_delay
):
    # Four detached generations reach the 2 s, 4 s and 5 s kill boundaries;
    # the last is adopted during the final reap grace and must also be killed.
    # Kernel exit/reap may lag SIGKILL; an unreapable child must still time out.
    processes = []
    for pid in (123, 124, 125, 126):
        process = tmp_path / str(pid)
        process.mkdir()
        (process / "stat").write_bytes(f"{pid} (child) S 100 {pid} 1\n".encode())
        processes.append(process)
    now = 0.0
    generation = 0
    killed = {}
    reaped = []
    signals = []

    def process_entries(path):
        assert path == Path("/proc")
        return iter(processes[generation : generation + 1])

    def kill(pid, sig):
        assert pid == 123 + generation  # Never signal a reaped identity.
        signals.append((pid, sig, now))
        if sig == signal.SIGKILL:
            killed.setdefault(pid, now)

    def waitpid(pid, options):
        nonlocal generation
        assert (pid, options) == (-1, os.WNOHANG)
        if generation == len(processes):
            raise ChildProcessError
        child = 123 + generation
        delay = final_reap_delay if generation == 3 else 0.0
        if child in killed and delay is not None and now >= killed[child] + delay:
            reaped.append(child)
            generation += 1
            return child, 0
        return 0, 0

    def sleep(seconds):
        nonlocal now
        now = round(now + seconds, 6)

    monkeypatch.setattr(Path, "iterdir", process_entries)
    monkeypatch.setattr(
        supervision,
        "os",
        SimpleNamespace(
            getpid=lambda: 100,
            getpgrp=lambda: 100,
            kill=kill,
            waitpid=waitpid,
            WNOHANG=os.WNOHANG,
        ),
    )
    monkeypatch.setattr(
        supervision,
        "time",
        SimpleNamespace(monotonic=lambda: now, sleep=sleep),
    )
    assert supervision._reap_children() is (final_reap_delay is not None)
    assert killed == {123: 2.0, 124: 4.01, 125: 5.0, 126: 5.01}
    assert reaped == (
        [123, 124, 125] if final_reap_delay is None else [123, 124, 125, 126]
    )
    assert [pid for pid, sig, _ in signals if sig == signal.SIGTERM] == [
        123,
        124,
        125,
        126,
    ]
    assert 5.0 <= now <= 5.5 < supervision._SUPERVISOR_CLEANUP_TIMEOUT_SECONDS
    if final_reap_delay is None:
        assert now == 5.5
        assert "could not reap children" in capsys.readouterr().err
    else:
        assert not capsys.readouterr().err


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor cancellation")
def test_unresponsive_supervisor_has_bounded_cleanup_and_retains_output(
    tmp_path, unresponsive_supervisor
):
    unresponsive_supervisor()
    script = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('cleanup evidence', flush=True); time.sleep(30)"
    )
    log_path = tmp_path / "cleanup.log"
    with pytest.raises(
        autoreview.ReviewAbort, match="cleanup exceeded its deadline"
    ) as error:
        autoreview.run_subprocess(
            [sys.executable, "-c", script],
            label="unresponsive",
            timeout=3,
            log_path=log_path,
        )
    assert "timed out after 3s" in str(error.value)
    assert "timed out after 3s" in log_path.read_text()
    assert "cleanup evidence" in log_path.read_text()


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor cancellation")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("stop_again", [False, True])
def test_owned_stop_bounds_polling_without_supervisor_exit(
    tmp_path, monkeypatch, signum, stop_again
):
    supervised = supervision.supervised_command

    @contextmanager
    def unresponsive_command(cmd):
        # Keep the real exit pipe open, but replace its supervisor with a
        # process that never handles the lifeline EOF. Its PID remains owned.
        with supervised(cmd) as (_, descriptors, status, control):
            yield cmd, descriptors, status, control

    monkeypatch.setattr(supervision, "supervised_command", unresponsive_command)
    monkeypatch.setattr(supervision, "_SUPERVISOR_CLEANUP_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(supervision, "_CHILD_TERM_GRACE_SECONDS", 0.1)
    ready = tmp_path / "ready"
    continued = tmp_path / "continued"
    script = f"""
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
def resume(signum, frame):
    Path({str(continued)!r}).touch()
    if {stop_again!r}:
        os.kill(os.getpid(), signal.SIGSTOP)
signal.signal(signal.SIGCONT, resume)
Path({str(ready)!r}).touch()
while True:
    time.sleep(1)
"""
    with (tmp_path / "process.log").open("wb") as output:
        proc = supervision.OwnedProcess(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=os.environ.copy(),
            stdout=output,
            label="unresponsive owned process",
        )
    try:
        _wait_for_file(ready)
        started = time.monotonic()
        with pytest.raises(
            supervision.CleanupError, match="cleanup exceeded its deadline"
        ):
            while time.monotonic() - started < 2:
                # Repeated requests must not postpone the first deadline.
                proc.request_stop(signum)
                assert proc.poll() is None
                time.sleep(0.02)
            pytest.fail("stop request left poll waiting for supervisor exit")
        assert continued.is_file(), "the real process must receive SIGCONT"
        assert proc.proc.returncode == -signal.SIGKILL
    finally:
        with suppress(supervision.ProcessError):
            proc.close()


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX stopped-supervisor cleanup")
def test_timeout_resumes_stopped_supervisor(captured_processes):
    supervisors, started, cleanup = captured_processes
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            autoreview.run_subprocess,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            label="stopped",
            timeout=0.5,
        )
        try:
            assert started.wait(10), "process supervisor was not spawned"
            os.kill(supervisors[0].pid, signal.SIGSTOP)
            with pytest.raises(autoreview.ReviewAbort, match="timed out"):
                future.result(timeout=10)
            assert supervisors[0].returncode is not None
        finally:
            cleanup()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="POSIX supervisor cancellation")
@pytest.mark.parametrize("unresponsive", [False, True])
def test_peer_failure_interrupts_long_supervisor_wait(
    tmp_path, capsys, unresponsive, captured_processes, unresponsive_supervisor
):
    ready = tmp_path / "ready"
    log_path = tmp_path / "cancelled.log"
    supervisors, started, cleanup = captured_processes
    failure_triggered = threading.Event()
    if unresponsive:
        unresponsive_supervisor()
    script = (
        "import signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('cancellation evidence', flush=True); "
        f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )

    def fail_peer(cancelled):
        assert started.wait(10), "process supervisor was not spawned"
        _wait_for_file(ready)
        if not unresponsive:
            os.kill(supervisors[0].pid, signal.SIGSTOP)
        failure_triggered.set()
        raise autoreview.ReviewAbort("peer failed")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            autoreview.run_parallel,
            autoreview.Spinner(),
            {
                "supervisor": lambda cancelled: autoreview.run_subprocess(
                    [sys.executable, "-c", script],
                    label="long wait",
                    timeout=60,
                    log_path=log_path,
                    cancelled=cancelled,
                ),
                "peer": fail_peer,
            },
        )
        try:
            # Startup has its own diagnostic bound. Keep the tested ten-second
            # cancellation bound measured from the actual peer failure.
            assert failure_triggered.wait(25), "peer did not reach its failure trigger"
            with pytest.raises(autoreview.ReviewAbort, match="peer failed"):
                future.result(timeout=10)
            assert all(proc.returncode is not None for proc in supervisors)
            assert "cancellation evidence" in log_path.read_text()
            if unresponsive:
                message = "detached-child cleanup is unconfirmed"
                assert message in capsys.readouterr().err
                assert message in log_path.read_text()
        finally:
            # Retire processes before the outer pool joins on a failing test.
            cleanup()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor cleanup status")
@pytest.mark.parametrize("fault", ["adoption-error", "unexpected-stop"])
def test_supervisor_internal_failure_is_not_reviewer_status(
    tmp_path, replace_supervisor, fault
):
    started = tmp_path / "reviewer-started"
    log_path = tmp_path / "supervisor.log"
    replace_supervisor(f"""
import errno, os, signal
def fail_adoption():
    if {fault!r} == 'adoption-error':
        raise OSError(errno.EPERM, 'injected adoption error')
    os.kill(os.getpid(), signal.SIGTERM)
namespace['process_supervisor']._supervise_command.__globals__['_adopt_orphans'] = fail_adoption
""")
    with pytest.raises(autoreview.ReviewAbort) as error:
        autoreview.run_subprocess(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(started)!r}).touch()",
            ],
            label="internal failure",
            log_path=log_path,
            timeout=10,
        )
    assert not started.exists()
    raw = log_path.read_text()
    if fault == "adoption-error":
        assert "supervisor failed: Operation not permitted; cleanup completed" in str(
            error.value
        )
        assert "injected adoption error" in raw
    else:
        assert (
            "supervisor stopped unexpectedly; command completion is unconfirmed"
            in str(error.value)
        )
    assert str(error.value) in raw


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor cleanup status")
@pytest.mark.parametrize("reason", ["timeout", "cancel"])
@pytest.mark.parametrize("disposition", ["false", "oserror"])
def test_supervisor_cleanup_failure_preserves_pending_reason(
    tmp_path, replace_supervisor, reason, disposition
):
    release = tmp_path / "release"
    pid_path = tmp_path / "reviewer-pids.json"
    log_path = tmp_path / "cleanup-status.log"
    replace_supervisor(f"""
real_reap = namespace['process_supervisor']._reap_children
def failed_reap():
    assert real_reap(), 'real cleanup failed before fault injection'
    print('real descendants reaped; injected cleanup {disposition}', file=sys.stderr, flush=True)
    if {disposition!r} == 'false':
        return False
    raise OSError('injected cleanup OSError')
namespace['process_supervisor']._supervise_command.__globals__['_reap_children'] = failed_reap
""")
    script = (
        "import json, os, subprocess, sys, time\nfrom pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {_fixture_lifeline(release)!r}], start_new_session=True)\n"
        f"p = Path({str(pid_path)!r})\n"
        "p.with_suffix('.tmp').write_text(json.dumps([os.getpid(), child.pid]))\n"
        "print('reviewer evidence before stop', flush=True)\n"
        "p.with_suffix('.tmp').replace(p)\n" + _fixture_lifeline(release)
    )
    cancelled = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            autoreview.run_subprocess,
            [sys.executable, "-c", script],
            label="cleanup status",
            timeout=5 if reason == "timeout" else 30,
            log_path=log_path,
            cancelled=cancelled,
        )
        pids = []
        try:
            pids = json.loads(_wait_for_file(pid_path).read_text())
            if reason == "cancel":
                cancelled.set()
            with pytest.raises(autoreview.ReviewAbort) as error:
                future.result(timeout=15)
            raw = log_path.read_text()
            assert f"real descendants reaped; injected cleanup {disposition}" in raw
            assert "reviewer evidence before stop" in raw
            assert _wait_until_exited(pids)
            assert all(not Path(f"/proc/{pid}").exists() for pid in pids)
            expected = "timed out after 5s" if reason == "timeout" else "cancelled"
            assert expected in str(error.value)
            warning = "detached-child cleanup is unconfirmed"
            assert warning in str(error.value)
            assert expected in raw and warning in raw
        finally:
            cancelled.set()
            release.touch()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="Linux child-adoption cleanup")
def test_signal_at_cleanup_entry_does_not_skip_reaping(tmp_path, replace_supervisor):
    release = tmp_path / "release"
    pid_path = tmp_path / "detached.pid"
    replace_supervisor("""
import os, signal
injected = False
original_signal = signal.signal
def signal_at_cleanup(signum, handler):
    global injected
    if not injected and signum == signal.SIGTERM and handler == signal.SIG_IGN:
        injected = True
        os.kill(os.getpid(), signal.SIGTERM)
        print('injected SIGTERM at cleanup entry', file=sys.stderr, flush=True)
    return original_signal(signum, handler)
signal.signal = signal_at_cleanup
""")
    script = (
        "import subprocess, sys; from pathlib import Path; "
        f"child = subprocess.Popen([sys.executable, '-c', {_fixture_lifeline(release)!r}], start_new_session=True); "
        f"Path({str(pid_path)!r}).write_text(str(child.pid))"
    )
    try:
        result = autoreview.run_subprocess(
            [sys.executable, "-c", script], label="cleanup signal", timeout=5
        )
        assert result.returncode == 0, result.stderr
        assert "injected SIGTERM at cleanup entry" in result.stderr
        assert not _process_running(int(pid_path.read_text()))
    finally:
        release.touch()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="observes Linux process lifetime")
@pytest.mark.parametrize("phase", ["combine", "fix"])
def test_termination_during_followup_spawn_does_not_leak_reviewer(tmp_path, phase):
    release = tmp_path / "release"
    pid_path = tmp_path / "followup.pid"
    claude = tmp_path / "claude"
    claude.write_text(
        f"#!{sys.executable}\nimport json, re, sys, time\n"
        "if '--resume' in sys.argv:\n"
        f"{textwrap.indent(_fixture_lifeline(release), '    ')}"
        "prompt = sys.argv[sys.argv.index('-p') + 1]\n"
        "completion = re.findall(r'\\[AUTOREVIEW_COMPLETE_[0-9a-f]{32}\\]', prompt)[-1]\n"
        "print(json.dumps({'result': 'A finding\\n' + completion, 'session_id': 's1'}))\n"
    )
    claude.chmod(0o755)
    codex = tmp_path / "codex"
    codex.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        "Path(sys.argv[sys.argv.index('-o') + 1]).write_text("
        f"{'Another finding' if phase == 'combine' else 'LGTM'!r})\n"
    )
    codex.chmod(0o755)
    launcher = f"""
import os, runpy, signal, subprocess, sys
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_DFL)
import sys
sys.path.insert(0, {str(SCRIPT.parent)!r})
namespace = runpy.run_path({str(SCRIPT)!r})
state = namespace['main'].__globals__
state['REPO_ROOT'] = Path({str(tmp_path)!r})
state['LOGS_BASE'] = state['REPO_ROOT'] / 'logs'
original = subprocess.Popen
def interrupt_after_spawn(cmd, **kwargs):
    proc = original(cmd, **kwargs)
    if '--resume' in cmd:
        Path({str(pid_path)!r}).write_text(str(proc.pid))
        os.kill(os.getpid(), signal.SIGTERM)
    return proc
subprocess.Popen = interrupt_after_spawn
sys.argv = ['autoreview.py', '--no-meta', *{["--fix"] if phase == "fix" else []!r}, 'x.py']
raise SystemExit(namespace['main']())
"""
    proc = _launch_parent(tmp_path, launcher)
    reviewer = None
    try:
        assert proc.wait(timeout=15) == 128 + signal.SIGTERM
        reviewer = int(_wait_for_file(pid_path).read_text())
        assert _wait_until_exited([reviewer])
    finally:
        release.touch()
        _cleanup_owned_parent(proc)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="Linux orphan adoption and waitid")
@pytest.mark.parametrize("without_waitid", [False, True])
def test_supervisor_death_stops_its_cli_before_reaping_identity(
    tmp_path, without_waitid
):
    # An isolated driver adopts the killed supervisor's child. Even on the
    # broken parent, its finally block can kill/reap that still-owned PID
    # without leaving a sleeper or risking a recycled unrelated process.
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"""
import importlib.util, json, os, signal, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, {str(SCRIPT.parent)!r})
spec = importlib.util.spec_from_file_location('autoreview', {str(SCRIPT)!r})
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
if {without_waitid!r}:
    review.process_supervisor.os = SimpleNamespace(**{{name: getattr(os, name) for name in dir(os)
        if name not in {WAITID_CAPABILITIES!r}}})
review.process_supervisor._adopt_orphans()
directory = Path({str(tmp_path)!r})
pid_file = directory / 'cli.pid'
log = directory / 'review.log'
supervisors = []
original_popen = subprocess.Popen
def capture(*args, **kwargs):
    proc = original_popen(*args, **kwargs)
    supervisors.append(proc)
    return proc
subprocess.Popen = capture
killpg = os.killpg
retained = []
def checked_killpg(pgid, sig):
    if sig == signal.SIGKILL:
        proc = supervisors[0]
        assert pgid == proc.pid and proc.returncode is None
        status = os.waitid(os.P_PID, pgid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        assert status is not None, 'dead supervisor must still reserve its PID'
        retained.append(pgid)
    killpg(pgid, sig)
os.killpg = checked_killpg
if {without_waitid!r}:
    review.process_supervisor.os.killpg = checked_killpg
child = None
script = (
    "import os, signal, sys, time; from pathlib import Path; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "print('partial stdout before supervisor death', flush=True); "
    "print('partial stderr before supervisor death', file=sys.stderr, flush=True); "
    f"p=Path({{str(pid_file)!r}}); p.with_suffix('.tmp').write_text(str(os.getpid())); "
    "p.with_suffix('.tmp').replace(p); time.sleep(30)"
)
try:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(review.run_subprocess, [sys.executable, '-c', script],
                             label='supervisor death', timeout=5, log_path=log)
        deadline = time.monotonic() + 10
        while not pid_file.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        child = int(pid_file.read_text())
        os.kill(supervisors[0].pid, signal.SIGKILL)
        try:
            future.result(timeout=10)
        except review.ReviewAbort as error:
            assert 'supervisor exited without confirming cleanup' in str(error)
            assert 'detached-child cleanup is unconfirmed' in str(error)
        else:
            raise AssertionError('abnormal supervisor death cannot pass')
        # killpg queues SIGKILL; allow bounded delivery without reaping the
        # adopted child or releasing the identity used by finally cleanup.
        deadline = time.monotonic() + 2
        while os.waitid(os.P_PID, child, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            assert time.monotonic() < deadline, 'reviewer remained alive after supervisor death'
            time.sleep(0.01)
        assert retained == [supervisors[0].pid]
        assert supervisors[0].returncode == -signal.SIGKILL
        text = log.read_text()
        assert 'partial stdout before supervisor death' in text
        assert 'partial stderr before supervisor death' in text
        assert 'cleanup is unconfirmed' in text
finally:
    if child is not None:
        # It is our unreaped adopted child; this PID cannot have been reused.
        os.kill(child, signal.SIGKILL)
        os.waitpid(child, 0)
"""
    )
    result = subprocess.run(
        [sys.executable, str(driver)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor identity")
@pytest.mark.parametrize("cached_status", [False, True])
def test_reaped_supervisor_identity_is_never_signaled(monkeypatch, cached_status):
    if not cached_status and not all(hasattr(os, name) for name in WAITID_CAPABILITIES):
        pytest.skip("external reaping detection additionally uses optional waitid")
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    pid, status = os.waitpid(proc.pid, 0)
    assert pid == proc.pid
    if cached_status:
        proc.returncode = os.waitstatus_to_exitcode(status)

    def forbidden_signal(*args):
        raise AssertionError("must not signal a PID/PGID after its identity was reaped")

    with (
        monkeypatch.context() as scoped,
        pytest.raises(autoreview.ReviewAbort, match="identity was already reaped"),
    ):
        scoped.setattr(autoreview.os, "killpg", forbidden_signal)
        scoped.setattr(autoreview.os, "kill", forbidden_signal)
        supervision._check_proc_identity(proc)
    proc.returncode = os.waitstatus_to_exitcode(status)


def test_non_posix_wait_keeps_the_popen_path(monkeypatch):
    calls = []

    class Process:
        returncode = None

        def kill(self):
            calls.append("kill")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            self.returncode = 0

    proc = Process()
    with monkeypatch.context() as scoped:
        scoped.setattr(supervision, "os", SimpleNamespace(name="nt"))
        autoreview.finish_proc(proc, "non-posix")
    assert calls == ["kill", ("wait", 6), ("wait", 1)]


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor exit pipe")
@pytest.mark.parametrize("timeout", [False, True])
def test_exit_pipe_without_waitid(
    tmp_path, without_waitid, captured_processes, timeout
):
    processes, started, cleanup = captured_processes
    log = tmp_path / "exit-pipe.log"
    cmd = [
        sys.executable,
        "-c",
        "import time; print('retained output', flush=True); "
        + ("time.sleep(30)" if timeout else "raise SystemExit(7)"),
    ]
    if timeout:
        with pytest.raises(autoreview.ReviewAbort, match="timed out after 3s"):
            autoreview.run_subprocess(cmd, label="exit pipe", timeout=3, log_path=log)
    else:
        result = autoreview.run_subprocess(
            cmd, label="exit pipe", timeout=10, log_path=log
        )
        assert result.returncode == 7
        assert result.stdout == "retained output\n"
    assert len(processes) == 1
    assert processes[0].returncode == -signal.SIGKILL
    assert "retained output" in log.read_text()
    assert "unconfirmed" not in log.read_text()


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor exit pipe")
@pytest.mark.parametrize("spawned", [False, True])
@pytest.mark.parametrize("endpoint", ["writer", "reader", "stop"])
def test_child_pipe_ends_close_once_without_closing_a_reused_fd(spawned, endpoint):
    replacement = None
    try:
        with autoreview.supervised_command(["unused"]) as (
            argv,
            pass_fds,
            status,
            control,
        ):
            if endpoint == "stop":
                descriptor = control.stop_fd
                close = control.close_stop
            else:
                descriptor = (
                    control.parent_exit_writer
                    if endpoint == "writer"
                    else control.parent_stop_reader
                )
                close = getattr(control, f"close_parent_{endpoint}")
            assert (descriptor in pass_fds) == (endpoint != "stop")
            if spawned:
                close()
                replacement = os.open(os.devnull, os.O_RDONLY)
                if replacement != descriptor:
                    os.dup2(replacement, descriptor)
                    os.close(replacement)
                    replacement = descriptor
                # Includes an explicit repeated close and the context's close.
                close()
                os.fstat(replacement)
        if spawned:
            os.fstat(replacement)
        else:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        if replacement is not None:
            os.close(replacement)


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor exit pipe")
def test_exit_writer_is_not_inherited_by_reviewer(replace_supervisor, without_waitid):
    replace_supervisor("""
import os
fd = int(sys.argv[3])
os.environ['TEST_EXIT_FD'] = str(fd)
os.environ['TEST_EXIT_INODE'] = str(os.fstat(fd).st_ino)
""")
    result = autoreview.run_subprocess(
        [
            sys.executable,
            "-c",
            """
import os
try:
    state = os.fstat(int(os.environ['TEST_EXIT_FD']))
except OSError:
    pass
else:
    assert state.st_ino != int(os.environ['TEST_EXIT_INODE']), 'inherited exit writer'
print('exit writer is private')
""",
        ],
        label="private exit writer",
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exit writer is private\n"


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGPIPE disposition")
@pytest.mark.parametrize("disposition", ["ignored", "default", "custom"])
def test_sigpipe_disposition_is_preserved_during_success_and_timeout(disposition):
    # The subprocess contains SIG_DFL safely; the worker call must never change
    # process-wide handlers, including on the cancellation path.
    driver = f"""
import runpy, signal, sys
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, {str(SCRIPT.parent)!r})
namespace = runpy.run_path({str(SCRIPT)!r})
signals = []
handler = {{'ignored': signal.SIG_IGN, 'default': signal.SIG_DFL,
           'custom': lambda *args: signals.append(args)}}[{disposition!r}]
signal.signal(signal.SIGPIPE, handler)
with ThreadPoolExecutor(max_workers=1) as pool:
    result = pool.submit(namespace['run_subprocess'],
                         [sys.executable, '-c', "print('ok')"],
                         label='success', timeout=5).result(timeout=10)
    assert result.returncode == 0 and result.stdout == 'ok\\n'
    future = pool.submit(namespace['run_subprocess'],
                         [sys.executable, '-c', 'import time; time.sleep(30)'],
                         label='timeout', timeout=0.2)
    try:
        future.result(timeout=10)
    except namespace['ReviewAbort'] as error:
        assert 'timed out' in str(error)
    else:
        raise AssertionError('sleeping reviewer must time out')
assert signal.getsignal(signal.SIGPIPE) == handler
assert not signals
"""
    result = subprocess.run(
        [sys.executable, "-c", driver], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGCHLD disposition")
@pytest.mark.parametrize("disposition", ["ignored", "custom"])
def test_non_default_sigchld_refuses_before_launch(tmp_path, disposition):
    driver = f"""
import runpy, signal
import sys
sys.path.insert(0, {str(SCRIPT.parent)!r})
namespace = runpy.run_path({str(SCRIPT)!r})
handler = signal.SIG_IGN if {disposition!r} == 'ignored' else lambda *_: None
signal.signal(signal.SIGCHLD, handler)
def forbidden(*args, **kwargs):
    raise AssertionError('must refuse before spawning an auto-reaped child')
namespace['subprocess'].Popen = forbidden
try:
    namespace['run_subprocess'](['unused'], label='SIGCHLD')
except namespace['ReviewAbort'] as error:
    assert 'requires default SIGCHLD' in str(error)
else:
    raise AssertionError('non-default SIGCHLD must not be admitted')
"""
    result = subprocess.run(
        [sys.executable, "-c", driver], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGCHLD inheritance")
def test_cli_restores_inherited_ignored_sigchld(tmp_path, monkeypatch):
    # Exec the real CLI in a private checkout so its logs stay in the fixture.
    script = tmp_path / "scripts" / "autoreview.py"
    script.parent.mkdir()
    script.write_bytes(SCRIPT.read_bytes())
    script.with_name("process_supervisor.py").write_bytes(
        SCRIPT.with_name("process_supervisor.py").read_bytes()
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    codex = bin_dir / "codex"
    codex.write_text(
        f"#!{sys.executable}\n"
        "import signal, sys\nfrom pathlib import Path\n"
        "assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL\n"
        "Path(sys.argv[sys.argv.index('-o') + 1]).write_text('LGTM')\n"
    )
    codex.chmod(0o755)
    (bin_dir / "git").symlink_to(GIT)
    # The CLI checks that the review left the working tree unchanged.
    _git_init(tmp_path, monkeypatch)
    (tmp_path / ".gitignore").write_text("/*\n!/.gitignore\n")
    driver = (
        "import os, signal, sys\n"
        "signal.signal(signal.SIGCHLD, signal.SIG_IGN)\n"
        f"os.execv(sys.executable, [sys.executable, {str(script)!r}, "
        "'--no-meta', '--head'])\n"
    )
    env = {**os.environ, "PATH": str(bin_dir)}
    env.pop(autoreview.SESSION_MARKER, None)
    result = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "LGTM"
    assert "autoreview passed: codex" in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor exit pipe")
def test_exit_wait_reuses_selector_and_closes_on_context_error(monkeypatch):
    selector_factory = supervision.selectors.DefaultSelector
    selectors = []

    def make_selector():
        selector = selector_factory()
        selectors.append(selector)
        return selector

    monkeypatch.setattr(supervision.selectors, "DefaultSelector", make_selector)
    # Exercise real pipe readiness without inventing a retained process ID.
    monkeypatch.setattr(
        supervision,
        "_check_proc_identity",
        lambda *args, **kwargs: None,
    )
    proc = SimpleNamespace(args=["unused"])
    with (
        pytest.raises(RuntimeError, match="context teardown"),
        autoreview.supervised_command(proc.args) as (_, _, _, control),
    ):
        for _ in range(2):
            with pytest.raises(subprocess.TimeoutExpired):
                autoreview.wait_proc(proc, 0, control)
        control.close_parent_writer()
        autoreview.wait_proc(proc, 0, control)
        assert len(selectors) == 1
        assert list(selectors[0].get_map()) == [control.exit_fd]
        raise RuntimeError("context teardown")
    assert selectors[0].get_map() is None
