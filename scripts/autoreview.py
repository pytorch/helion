#!/usr/bin/env python3
"""Run claude and codex reviewers in parallel and combine their reviews.

Usage:
    scripts/autoreview.py                  # review staged + uncommitted changes
    scripts/autoreview.py --head           # review last commit (git diff HEAD~ HEAD)
    scripts/autoreview.py FILE_OR_URL      # review the thing described by args
    scripts/autoreview.py --no-codex       # skip codex
    scripts/autoreview.py --fix            # ask claude to fix the reported issues
    scripts/autoreview.py --fix --add      # also `git add` modified files after fixing
    scripts/autoreview.py --meta           # add Meta-internal launcher flags
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_EXCEPTION
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from contextlib import suppress
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_BASE = REPO_ROOT / ".logs" / "autoreview"
META_MARKER = Path("/usr/local/bin/claude_code/api-key-helper")
CLAUDE_EFFORT = "max"
CODEX_REASONING_EFFORT = "xhigh"


def _meta_default() -> bool:
    """Auto-detect Meta launcher unless HELION_AUTOREVIEW_META overrides it."""
    override = os.environ.get("HELION_AUTOREVIEW_META")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return META_MARKER.exists()


def _timeout_default() -> float:
    """Per-reviewer subprocess timeout; HELION_AUTOREVIEW_TIMEOUT overrides."""
    override = os.environ.get("HELION_AUTOREVIEW_TIMEOUT")
    if override:
        return float(override)
    return 60.0 * 60.0  # 1 hour per reviewer call


SUBPROCESS_TIMEOUT_SECONDS = _timeout_default()


PROMPT_DIFF_PREFIX = (
    "Review the staged and uncommitted changes since HEAD, if there are no "
    'meaningful changes print only "ERROR: no changes, use --head to review '
    'last commit" and stop.'
)

PROMPT_HEAD_PREFIX = (
    "Review only the changes introduced by the last commit "
    "(`git diff HEAD~ HEAD`); ignore any uncommitted changes in the worktree."
)

PROMPT_ARGS_PREFIX = """\
Review only the thing described by these command-line arguments. Treat them as
the user's description of what to review, and resolve any referenced commits,
URLs, diffs, files, issues, or pull requests as needed. Do not review
staged/uncommitted changes or the last commit unless the arguments explicitly
ask for that.

Review target:
{review_target}\
"""

PROMPT_REVIEW_BODY = """\
Check for:
- Correctness issues
- Feature regressions (such as disabled/weakened tests)
- Opportunities for simplification or cleanups
- Places that are hacky or overly narrow
- Code duplication
- Backend-specific tests or test files that should use `@onlyBackends([...])`
- Too broad `try/except:` lines that could hide bugs
- Overly defensive getattr/hasattr checks that should be base class schema updates
- Comments that might go stale quickly (e.g. line numbers)
- Other important issues

If you hit a permission error or other system error that makes it hard to conduct the review respond only with "ERROR: <1-line reason>" and stop.

Do *NOT* make any code changes, on success respond only with the review. If there are no issues, respond with only "LGTM".\
"""

PROMPT_COMBINE = """\
Another agent using a different model produced the following review (between the BEGIN/END markers below). Combine your review (above) with the other agent's review into a single review. De-duplicate any common issues and resolve any disagreements/conflicts using your best judgment:

===== BEGIN OTHER REVIEW =====
{codex_review}
===== END OTHER REVIEW =====
"""

PROMPT_FIX = (
    "Fix all the issues in the above review. If you are unable to fix any "
    'of the issues, respond with "ERROR: <1 line reason>" and stop.'
)

PROMPT_FIX_ADD_SUFFIX = (
    " After fixing, run `git add -u` to stage your modifications, and "
    "`git add <path>` for any new files belonging with the changes you "
    "reviewed. Do not stage unrelated files."
)


class ReviewAbort(Exception):
    """Raised by review workers when they hit an unrecoverable condition."""


def format_review_args(review_args: list[str]) -> str:
    return shlex.join(review_args)


def build_review_prompt(head: bool, review_args: list[str] | None = None) -> str:
    review_args = review_args or []
    if review_args:
        prefix = PROMPT_ARGS_PREFIX.format(
            review_target=format_review_args(review_args)
        )
    else:
        prefix = PROMPT_HEAD_PREFIX if head else PROMPT_DIFF_PREFIX
    return f"{prefix}\n\n{PROMPT_REVIEW_BODY}"


def subprocess_env() -> dict[str, str]:
    """Return env stripped of vars that prevent the installed launcher binaries
    from running. The Meta launcher passes META_DANGEROUSLY_DISABLE_LINUX_SANDBOX
    through to the inner binary, which then refuses to start."""
    env = os.environ.copy()
    env.pop("META_CLAUDE_DANGEROUSLY_DISABLE_LINUX_SANDBOX", None)
    env.pop("META_DANGEROUSLY_DISABLE_LINUX_SANDBOX", None)
    return env


def is_error_response(text: str) -> str | None:
    """If the entire response is an ERROR: line (per the prompt contract),
    return the message. Returns None when ERROR: appears only inside the body
    of a normal review."""
    stripped = text.strip()
    if not stripped.startswith("ERROR:"):
        return None
    return stripped.splitlines()[0].strip()


def stderr_tail(stderr: str, *, max_lines: int = 5) -> str:
    """Return up to the last `max_lines` non-empty lines of stderr joined by
    newlines so user-visible error messages preserve the meaningful trailing
    context (e.g. a short traceback) rather than just the final line."""
    tail = stderr.strip().splitlines()[-max_lines:]
    return "\n".join(tail)


_active_procs: list[subprocess.Popen[str]] = []
_active_procs_lock = threading.Lock()


def _register_proc(proc: subprocess.Popen[str]) -> None:
    with _active_procs_lock:
        _active_procs.append(proc)


def _unregister_proc(proc: subprocess.Popen[str]) -> None:
    with _active_procs_lock:
        if proc in _active_procs:
            _active_procs.remove(proc)


def kill_active_procs() -> None:
    """Kill every subprocess currently tracked by run_subprocess so that a
    failure in one parallel worker does not block on the others completing."""
    with _active_procs_lock:
        procs = list(_active_procs)
    for p in procs:
        with suppress(ProcessLookupError, OSError):
            p.kill()


def run_subprocess(
    cmd: list[str], *, label: str, timeout: float = SUBPROCESS_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run cmd, converting launch failures and timeouts into ReviewAbort and
    registering the live process so it can be killed on parallel-worker abort."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=subprocess_env(),
            cwd=REPO_ROOT,
        )
    except FileNotFoundError as e:
        raise ReviewAbort(f"[{label}] binary not found: {e}") from None
    except PermissionError as e:
        raise ReviewAbort(f"[{label}] permission error: {e}") from None
    except OSError as e:
        raise ReviewAbort(f"[{label}] launch failed: {e}") from None
    _register_proc(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise ReviewAbort(f"[{label}] timed out after {timeout:.0f}s") from None
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    finally:
        _unregister_proc(proc)


class Spinner:
    """Spinning progress line on stderr with permanent messages logged above.

    When stderr is not a tty (nohup, redirected output, CI), the rotating
    "waiting for: ..." indicator is suppressed; only `log()` and `emit_stdout()`
    still produce output. This is intentional, not a bug."""

    FRAMES = "|/-\\"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.waiting: list[str] = []
        self.stop_event = threading.Event()
        self.tty = sys.stderr.isatty()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.tty:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def add(self, label: str) -> None:
        with self.lock:
            if label not in self.waiting:
                self.waiting.append(label)

    def remove(self, label: str) -> None:
        with self.lock:
            if label in self.waiting:
                self.waiting.remove(label)

    def log(self, msg: str) -> None:
        with self.lock:
            self._clear_locked()
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

    def emit_stdout(self, text: str) -> None:
        """Write to stdout under the spinner lock so the spinner does not
        redraw on top of a partial line."""
        with self.lock:
            self._clear_locked()
            sys.stderr.flush()
            sys.stdout.write(text)
            if not text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

    def stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        with self.lock:
            self._clear_locked()
            sys.stderr.flush()

    def _clear_locked(self) -> None:
        if self.tty:
            sys.stderr.write("\r\033[K")

    def _run(self) -> None:
        i = 0
        # 0.2s wait keeps stop() responsive while still throttling redraws.
        while not self.stop_event.wait(0.2):
            with self.lock:
                if self.waiting:
                    self._clear_locked()
                    msg = ", ".join(self.waiting)
                    sys.stderr.write(
                        f"{self.FRAMES[i % len(self.FRAMES)]} waiting for: {msg}"
                    )
                    sys.stderr.flush()
            i += 1


class ClaudeSession:
    """Multi-turn Claude session driven by `claude -p` and `--resume <id>`."""

    def __init__(self, log_path: Path, meta: bool) -> None:
        self.log_path = log_path
        self.meta = meta
        self.session_id: str | None = None
        self.log_path.write_text("")

    def _build_cmd(self, prompt: str) -> list[str]:
        cmd = ["claude"]
        if self.meta:
            cmd.append("--dangerously-enable-internet-mode")
        cmd += [
            "--dangerously-skip-permissions",
            "--effort",
            CLAUDE_EFFORT,
            "-p",
            prompt,
            "--output-format",
            "json",
        ]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        return cmd

    def send(self, prompt: str) -> str:
        cmd = self._build_cmd(prompt)
        proc = run_subprocess(cmd, label="claude")
        with self.log_path.open("a") as f:
            f.write(f"\n=== prompt ===\n{prompt}\n")
            f.write(f"\n=== returncode === {proc.returncode}\n")
            f.write(f"\n=== stderr ===\n{proc.stderr}\n")
            f.write(f"\n=== stdout ===\n{proc.stdout}\n")
        if proc.returncode != 0:
            raise ReviewAbort(
                f"[claude] non-zero exit {proc.returncode}: {stderr_tail(proc.stderr)}"
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise ReviewAbort(
                f"[claude] failed to parse JSON ({e}): {proc.stdout[:500]!r}"
            ) from None
        if data.get("is_error"):
            raise ReviewAbort(
                f"[claude] error result: subtype={data.get('subtype')!r} "
                f"api_error_status={data.get('api_error_status')!r}"
            )
        sid = data.get("session_id")
        if sid:
            self.session_id = sid
        result = data.get("result", "") or ""
        err = is_error_response(result)
        if err is not None:
            raise ReviewAbort(f"[claude] {err}")
        return result


def run_codex_review(prompt: str, log_path: Path, meta: bool) -> str:
    """Run codex one-shot, return the last message text. Codex only ever
    reviews — fixes are performed by the claude session — so the sandbox is
    always read-only regardless of how the parent script was invoked."""
    cmd = ["codex"]
    if meta:
        cmd.append("--dangerously-enable-internet-mode")
    cmd += [
        "-c",
        f'model_reasoning_effort="{CODEX_REASONING_EFFORT}"',
        "exec",
        "-s",
        "read-only",
    ]
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        out_path = Path(tf.name)
    try:
        cmd += [prompt, "-o", str(out_path)]
        proc = run_subprocess(cmd, label="codex")
        log_path.write_text(
            f"=== cmd ===\n{cmd}\n\n"
            f"=== returncode === {proc.returncode}\n\n"
            f"=== stderr ===\n{proc.stderr}\n\n"
            f"=== stdout ===\n{proc.stdout}\n"
        )
        if proc.returncode != 0:
            raise ReviewAbort(
                f"[codex] non-zero exit {proc.returncode}: {stderr_tail(proc.stderr)}"
            )
        result = out_path.read_text() if out_path.exists() else ""
        if not result.strip():
            raise ReviewAbort(
                "[codex] produced an empty review (use --no-codex to skip)"
            )
        err = is_error_response(result)
        if err is not None:
            raise ReviewAbort(f"[codex] {err}")
        return result
    finally:
        out_path.unlink(missing_ok=True)


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:.0f}s"


def run_with_timing(label: str, spinner: Spinner, fn: Callable[[], str]) -> str:
    spinner.add(label)
    start = time.monotonic()
    success = False
    try:
        result = fn()
        success = True
        return result
    finally:
        spinner.remove(label)
        elapsed = fmt_seconds(time.monotonic() - start)
        outcome = "finished" if success else "failed"
        spinner.log(f"{label} {outcome} in {elapsed}")


def run_parallel(
    spinner: Spinner, jobs: dict[str, Callable[[], str]]
) -> dict[str, str]:
    """Submit each (label -> callable) to a thread pool, raise the first
    ReviewAbort if any worker raises one, otherwise return a dict of results.

    On first failure, any still-running subprocess registered via run_subprocess
    is killed so the surviving workers unblock immediately."""
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    futs = {
        label: pool.submit(run_with_timing, label, spinner, fn)
        for label, fn in jobs.items()
    }
    try:
        # FIRST_EXCEPTION returns when any future raises (or all complete).
        # Inspect only the `done` set so we never block on a still-running
        # future inside fut.exception() before killing the surviving workers.
        done, not_done = wait(futs.values(), return_when=FIRST_EXCEPTION)
        for fut in done:
            exc = fut.exception()
            if exc is not None:
                kill_active_procs()
                wait(not_done)
                raise exc
        return {label: fut.result() for label, fut in futs.items()}
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "review_args",
        nargs="*",
        metavar="TARGET",
        help=(
            "Review the target described by these arguments instead of "
            "staged/uncommitted changes."
        ),
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="Review last commit (git diff HEAD~ HEAD) instead of staged/uncommitted changes.",
    )
    parser.add_argument(
        "--no-codex",
        action="store_true",
        help="Skip the codex review and just run claude.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="If review is not LGTM, ask claude to fix the issues.",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="With --fix, also run `git add` on modified files after fixing.",
    )
    parser.add_argument(
        "--meta",
        action=argparse.BooleanOptionalAction,
        default=_meta_default(),
        help=(
            "Pass Meta-internal launcher flags to claude/codex. Auto-detected "
            f"from {META_MARKER} (override with HELION_AUTOREVIEW_META=0/1). "
            "Use --no-meta to disable."
        ),
    )
    args = parser.parse_args()

    if args.add and not args.fix:
        parser.error("--add requires --fix")
    if args.head and args.review_args:
        parser.error("--head cannot be used with review target arguments")

    review_prompt = build_review_prompt(args.head, args.review_args)
    # Include PID so two near-simultaneous runs don't share log_dir.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    log_dir = LOGS_BASE / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    spinner = Spinner()
    spinner.start()
    spinner.log(f"logs: {log_dir.relative_to(REPO_ROOT)}")

    claude = ClaudeSession(log_dir / "claude.log", args.meta)
    combined = ""

    try:
        jobs = {"claude review": lambda: claude.send(review_prompt)}
        if not args.no_codex:
            jobs["codex review"] = lambda: run_codex_review(
                review_prompt, log_dir / "codex.log", args.meta
            )
        results = run_parallel(spinner, jobs)
        claude_review = results["claude review"]
        codex_review = results.get("codex review", "")
        codex_stripped = codex_review.strip()

        (log_dir / "claude-review.txt").write_text(claude_review)
        if not args.no_codex:
            (log_dir / "codex-review.txt").write_text(codex_review)

        if codex_stripped and codex_stripped != "LGTM":
            combine_prompt = PROMPT_COMBINE.format(codex_review=codex_review)
            combined = run_with_timing(
                "combining", spinner, lambda: claude.send(combine_prompt)
            )
            # Only written when an actual combine round produced new text.
            (log_dir / "combined-review.txt").write_text(combined)
        else:
            combined = claude_review
        spinner.emit_stdout(combined)

        if args.fix and combined.strip() != "LGTM":
            fix_prompt = PROMPT_FIX + (PROMPT_FIX_ADD_SUFFIX if args.add else "")
            fix_result = run_with_timing(
                "fixing", spinner, lambda: claude.send(fix_prompt)
            )
            (log_dir / "fix-result.txt").write_text(fix_result)
    except ReviewAbort as e:
        spinner.stop()
        sys.stderr.write(f"\n{e}\n")
        return 2
    finally:
        spinner.stop()
        if claude.session_id:
            meta_flag = "--dangerously-enable-internet-mode " if args.meta else ""
            effort_flag = f"--effort {CLAUDE_EFFORT} "
            sys.stderr.write(
                f"\nresume session: claude "
                f"{meta_flag}{effort_flag}--resume {claude.session_id}\n"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
