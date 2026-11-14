"""Owned subprocess groups shared by the test and review runners.

A per-command Linux subreaper owns only that command's descendants. The
parent retains the supervisor PID until bounded cleanup finishes, including
children that create a new session. Other POSIX systems clean the owned
process group; they cannot account for detached descendants.
"""

from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
from contextlib import suppress
import ctypes
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping
    from typing import BinaryIO


class ProcessError(Exception):
    """A supervised command could not complete."""


class LaunchError(ProcessError):
    def __init__(self, label: str, error: OSError) -> None:
        super().__init__(f"[{label}] launch failed: {error}")
        self.error = error


class CleanupError(ProcessError):
    """Descendant cleanup could not be confirmed; retain resource claims."""


# Cleanup handlers (including GPU claim release) get time before forced exit.
# A newly adopted detached child gets its own grace within the shared signal
# budget. The reap grace then covers children killed at that deadline while
# still scanning for adopted descendants. The parent's cleanup margin covers
# this reap grace plus status publication and supervisor exit.
_CHILD_TERM_GRACE_SECONDS = 2.0
_CHILD_CLEANUP_TIMEOUT_SECONDS = 5.0
_CHILD_REAP_GRACE_SECONDS = 0.5
_SUPERVISOR_CLEANUP_MARGIN_SECONDS = 1.0
_SUPERVISOR_CLEANUP_TIMEOUT_SECONDS = (
    _CHILD_CLEANUP_TIMEOUT_SECONDS + _SUPERVISOR_CLEANUP_MARGIN_SECONDS
)

# Status kinds shared by the supervisor writer and the parent decoder.
_STATUS_RETURN_CODE = 0
_STATUS_LAUNCH_ERROR = 1
_STATUS_STOPPED = 2
_STATUS_CLEANUP_FAILED = 3
_STATUS_INTERNAL_ERROR_CLEANED = 4


class _SupervisorControl:
    """Parent stop writer and exit reader, plus temporary child-owned copies."""

    def __init__(
        self, stop_fd: int, exit_fd: int, exit_writer: int, stop_reader: int
    ) -> None:
        self.stop_fd: int | None = stop_fd
        self.exit_fd = exit_fd
        self.parent_exit_writer: int | None = exit_writer
        self.parent_stop_reader: int | None = stop_reader
        self.selector: selectors.BaseSelector | None = None
        self.exit_eof = False

    def close_stop(self) -> None:
        writer = self.stop_fd
        self.stop_fd = None
        if writer is not None:
            os.close(writer)

    def close_parent_writer(self) -> None:
        writer = self.parent_exit_writer
        self.parent_exit_writer = None
        if writer is not None:
            os.close(writer)

    def close_parent_reader(self) -> None:
        reader = self.parent_stop_reader
        self.parent_stop_reader = None
        if reader is not None:
            os.close(reader)


def require_default_sigchld() -> None:
    """Reject dispositions that invalidate the caller's retained child identity."""
    # SIG_IGN can auto-reap children, and a custom handler could reap ours.
    # In either case EOF would no longer prove a retained PID/PGID identity.
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise ProcessError("POSIX process supervision requires default SIGCHLD")


def _check_proc_identity(
    proc: subprocess.Popen[bytes], *, check_waitid: bool = True
) -> None:
    require_default_sigchld()
    if proc.returncode is not None:
        raise CleanupError(
            "supervisor identity was already reaped; group cleanup is unconfirmed"
        )
    # This is an additional stale-handle check where available, not the exit
    # notification mechanism. Older macOS Python lacks these APIs.
    if check_waitid and all(
        hasattr(os, name)
        for name in ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    ):
        try:
            os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            raise CleanupError(
                "supervisor identity was already reaped; group cleanup is unconfirmed"
            ) from None


def wait_proc(
    proc: subprocess.Popen[bytes],
    timeout: float,
    control: _SupervisorControl | None = None,
) -> None:
    """Observe EOF without releasing a POSIX supervisor's process-group ID.

    This caller owns the Popen child and is its sole reaper. Keeping its zombie
    until final group cleanup prevents its PID/PGID being reused between exit
    detection and killpg, including when the whole original group has exited.
    """
    if os.name != "posix":
        proc.wait(timeout=timeout)
        return
    _check_proc_identity(proc, check_waitid=False)
    assert control is not None and control.selector is not None
    if not control.selector.select(timeout):
        raise subprocess.TimeoutExpired(proc.args, timeout)
    if os.read(control.exit_fd, 1):
        raise ProcessError("unexpected data on supervisor exit pipe")
    control.exit_eof = True
    _check_proc_identity(proc)


def _kill_proc(proc: subprocess.Popen[bytes]) -> OSError | None:
    if proc.returncode is not None:
        return None
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except OSError as error:
        return error
    return None


def finish_proc(
    proc: subprocess.Popen[bytes],
    label: str,
    control: _SupervisorControl | None = None,
    *,
    timeout: float = _SUPERVISOR_CLEANUP_TIMEOUT_SECONDS,
) -> None:
    """Stop and reap an owned supervisor within the remaining cleanup budget.

    The caller must retain the child identity, then decode its status with
    read_result before treating command cleanup as confirmed.
    """
    error = None
    forced = False
    if os.name == "posix":
        assert control is not None
        # Refuse stale handles before any PID-based signal. An already-reaped
        # child no longer reserves its numeric process-group identity.
        _check_proc_identity(proc)
        # EOF means the supervisor is already dead. Its unreaped identity still
        # protects the final group KILL, but it needs no stop request or resume.
        needs_wait = not control.exit_eof
        if needs_wait:
            # EOF works before the supervisor installs handlers and after it
            # dies. Closing the sole writer never exposes the caller to SIGPIPE.
            control.close_stop()
            # A stopped supervisor cannot observe the pipe or run its cleanup.
            try:
                os.kill(proc.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            except OSError as resume_error:
                error = resume_error
    else:
        error = _kill_proc(proc)
        needs_wait = True
    if needs_wait:
        try:
            wait_proc(proc, timeout, control)
        except subprocess.TimeoutExpired:
            forced = True
    try:
        if os.name == "posix":
            # The leader is still our unreaped child even after abnormal death.
            # Kill any command left in its group before allowing PID/PGID reuse.
            _check_proc_identity(proc)
            os.killpg(proc.pid, signal.SIGKILL)
        elif forced:
            proc.kill()
    except ProcessLookupError:
        pass
    except OSError as kill_error:
        error = kill_error
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        detail = f": {error}" if error is not None else ""
        raise CleanupError(
            f"[{label}] cleanup could not stop process {proc.pid}{detail}"
        ) from None
    if forced:
        raise CleanupError(
            f"[{label}] supervisor cleanup exceeded its deadline; "
            "forcibly stopped its group, detached-child cleanup is unconfirmed"
        )


@contextmanager
def supervised_command(
    cmd: list[str],
) -> Iterator[
    tuple[list[str], tuple[int, ...], BinaryIO | None, _SupervisorControl | None]
]:
    """Opposing pipes notify each side of an uncatchable death of the other."""
    if os.name != "posix":
        yield cmd, (), None, None
        return
    read_fd, write_fd = os.pipe()
    control: _SupervisorControl | None = None
    try:
        exit_fd, exit_writer = os.pipe()
        control = _SupervisorControl(write_fd, exit_fd, exit_writer, read_fd)
        try:
            with (
                selectors.DefaultSelector() as selector,
                tempfile.TemporaryFile(mode="w+b") as status_file,
            ):
                selector.register(exit_fd, selectors.EVENT_READ)
                control.selector = selector
                yield (
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--supervise-command",
                        str(read_fd),
                        str(status_file.fileno()),
                        str(exit_writer),
                        *cmd,
                    ],
                    (read_fd, status_file.fileno(), exit_writer),
                    status_file,
                    control,
                )
        finally:
            control.close_parent_writer()
            os.close(exit_fd)
    finally:
        if control is None:
            os.close(write_fd)
            os.close(read_fd)
        else:
            control.close_stop()
            control.close_parent_reader()


def _adopt_orphans() -> None:
    # On Linux, adopting orphans keeps descendants under this supervisor even
    # when a command creates children in another session.
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    prctl.restype = ctypes.c_int
    if prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _reap_children() -> bool:
    deadline = time.monotonic() + _CHILD_CLEANUP_TIMEOUT_SECONDS
    reap_deadline = deadline + _CHILD_REAP_GRACE_SECONDS
    term_deadlines: dict[int, float] = {}
    while True:
        children = []
        # The per-task children file depends on CONFIG_CHECKPOINT_RESTORE,
        # which is disabled on some devserver kernels. stat's parent PID is
        # available independently of that kernel option.
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal():
                continue
            try:
                fields = (path / "stat").read_bytes().rpartition(b") ")[2].split()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if len(fields) >= 3 and int(fields[1]) == os.getpid():
                children.append((int(path.name), int(fields[2])))
        if not children:
            return True
        # This single-threaded supervisor has not reaped these direct children,
        # so their PIDs cannot be reused between this read and the signals.
        for pid, group in children:
            now = time.monotonic()
            if pid not in term_deadlines:
                # The supervisor already sent TERM/CONT to its own group.
                # Do not interrupt those handlers a second time. Detached
                # children only become ours when their parent exits; signal
                # each once while its unreaped identity is still reserved.
                if group != os.getpgrp():
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGTERM)
                        os.kill(pid, signal.SIGCONT)
                term_deadlines[pid] = min(deadline, now + _CHILD_TERM_GRACE_SECONDS)
            if now >= term_deadlines[pid]:
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                # The last reap can finish at the deadline. Do not reject it
                # using the pre-reap /proc snapshot below.
                return True
            if pid == 0:
                break
            term_deadlines.pop(pid, None)
        # Killing a child can adopt another generation, including detached
        # processes. Keep draining until all children have been reaped.
        if time.monotonic() >= reap_deadline:
            print(
                f"process supervisor could not reap children: {children}",
                file=sys.stderr,
                flush=True,
            )
            return False
        time.sleep(0.01)


def _supervise_command(
    lifeline_fd: int, status_fd: int, exit_fd: int, cmd: list[str]
) -> None:
    """Lead a command group until its command exits or the parent pipe closes.

    Linux child adoption also covers descendants that create a new session;
    other POSIX systems retain process-group cleanup. The status file carries
    the command's exit or launch error after cleanup, then the supervisor kills
    its own group. It never exits between command completion and group cleanup.
    """
    # Retain this descriptor until process death. The command must never inherit
    # it: EOF tells the sole parent waiter to perform final group cleanup.
    os.set_inheritable(exit_fd, False)
    if os.name != "posix" or os.getpgrp() != os.getpid():
        raise RuntimeError("process supervisor must lead its own process group")
    returncode = None
    launch_errno = None
    supervisor_errno = None
    adopt_orphans = sys.platform == "linux"
    stopping = False
    interrupt_deadline = None

    def terminate(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True

    # A flag avoids asynchronous exceptions during Popen or cleanup. Repeated
    # signals cannot unwind the cleanup; the selector bounds response latency.
    signal.signal(signal.SIGTERM, terminate)
    try:
        if adopt_orphans:
            _adopt_orphans()
        with selectors.DefaultSelector() as selector:
            selector.register(lifeline_fd, selectors.EVENT_READ)
            if stopping or selector.select(0):
                return
            # close_fds keeps both control descriptors out of the command.
            try:
                child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, close_fds=True)
            except OSError as error:
                launch_errno = error.errno
                print(f"command launch failed: {error}", file=sys.stderr, flush=True)
                return
            while True:
                if stopping:
                    return
                if selector.select(0.1):
                    # The first eight bytes are reserved for the final status.
                    # A separate request slot avoids SIGPIPE and works even if
                    # the parent cancels between our pre-spawn check and Popen.
                    request = os.pread(status_fd, 4, 8)
                    if request != struct.pack("!i", signal.SIGINT):
                        return
                    selector.unregister(lifeline_fd)
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    os.killpg(os.getpid(), signal.SIGINT)
                    interrupt_deadline = time.monotonic() + _CHILD_TERM_GRACE_SECONDS
                if (
                    interrupt_deadline is not None
                    and time.monotonic() >= interrupt_deadline
                ):
                    return
                if adopt_orphans:
                    # Drain adopted children while the command runs. Reap it
                    # here too and preserve its status; poll() after waitpid
                    # would otherwise turn ECHILD into a successful exit.
                    while True:
                        pid, status = os.waitpid(-1, os.WNOHANG)
                        if pid == 0:
                            break
                        if pid == child.pid:
                            returncode = os.waitstatus_to_exitcode(status)
                            child.returncode = returncode
                            return
                else:
                    returncode = child.poll()
                if returncode is not None:
                    return
    except BaseException as error:
        supervisor_errno = (error.errno or 0) if isinstance(error, OSError) else 0
        print(f"process supervisor failed: {error}", file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Every exit reports whether cleanup completed, even when cancellation
        # left no command status. A timely supervisor exit alone is not that proof.
        kind, value = _STATUS_CLEANUP_FAILED, 0
        try:
            # The leader stays alive and unreaped throughout this grace, so
            # signaling its group cannot target a reused process-group ID.
            os.killpg(os.getpid(), signal.SIGTERM)
            os.killpg(os.getpid(), signal.SIGCONT)
            if not adopt_orphans:
                # Other POSIX hosts retain group cleanup without Linux child
                # adoption. Let group members finish handlers before the final
                # group KILL; do not infer their lifetime from a reaped command.
                time.sleep(_CHILD_TERM_GRACE_SECONDS)
            cleaned = not adopt_orphans or _reap_children()
            if cleaned:
                if supervisor_errno is not None:
                    kind, value = _STATUS_INTERNAL_ERROR_CLEANED, supervisor_errno
                elif returncode is not None:
                    kind, value = _STATUS_RETURN_CODE, returncode
                elif launch_errno is not None:
                    kind, value = _STATUS_LAUNCH_ERROR, launch_errno
                else:
                    kind, value = _STATUS_STOPPED, 0
        except OSError as error:
            value = error.errno or 0
            print(
                f"process supervisor cleanup failed: {error}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            try:
                os.write(status_fd, struct.pack("!ii", kind, value))
            finally:
                os.killpg(os.getpid(), signal.SIGKILL)


def read_result(
    proc: subprocess.Popen[bytes],
    status_file: BinaryIO | None,
    label: str,
    cmd: list[str],
    *,
    stopped_ok: bool,
) -> int:
    """Decode the one post-cleanup protocol used by both runners."""
    if status_file is None:
        assert proc.returncode is not None
        return proc.returncode
    try:
        status_file.seek(0)
        status = status_file.read(8)
    except Exception as error:
        raise CleanupError(
            f"[{label}] cannot read cleanup confirmation: {error}"
        ) from error
    if len(status) != 8 or proc.returncode != -signal.SIGKILL:
        raise CleanupError(
            f"[{label}] supervisor exited without confirming cleanup; "
            "detached-child cleanup is unconfirmed"
        )
    kind, value = struct.unpack("!ii", status)
    if kind == _STATUS_RETURN_CODE:
        return value
    if kind == _STATUS_LAUNCH_ERROR:
        raise LaunchError(label, OSError(value, os.strerror(value), cmd[0]))
    if kind == _STATUS_INTERNAL_ERROR_CLEANED:
        detail = f": {os.strerror(value)}" if value else ""
        raise ProcessError(f"[{label}] supervisor failed{detail}; cleanup completed")
    if kind == _STATUS_STOPPED:
        if stopped_ok:
            return -signal.SIGTERM
        raise ProcessError(
            f"[{label}] supervisor stopped unexpectedly; command completion is unconfirmed"
        )
    detail = f": {os.strerror(value)}" if value else ""
    raise CleanupError(
        f"[{label}] supervisor cleanup failed; detached-child cleanup is unconfirmed{detail}"
    )


class OwnedProcess:
    """A command whose completion includes bounded descendant cleanup.

    Callers must close every instance before releasing its resources, including
    on exceptions. poll() returns the command's status only after cleanup;
    CleanupError means resources must remain reserved for manual inspection.
    """

    def __init__(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout: BinaryIO,
        label: str,
    ) -> None:
        if os.name == "posix":
            require_default_sigchld()
        self.label = label
        self.command = cmd
        self.returncode: int | None = None
        self.closed = False
        self.stop_deadline: float | None = None
        self.stack = ExitStack()
        spawned = None
        try:
            launch, descriptors, self.status, self.control = self.stack.enter_context(
                supervised_command(cmd)
            )
            spawned = self.proc = subprocess.Popen(
                launch,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                pass_fds=descriptors,
            )
            if self.control is not None:
                self.control.close_parent_writer()
                self.control.close_parent_reader()
        except BaseException:
            try:
                if spawned is not None:
                    self.close()
            finally:
                self.stack.close()
            raise

    def request_stop(self, signum: int = signal.SIGTERM) -> None:
        if self.closed:
            return
        if self.control is not None:
            _check_proc_identity(self.proc)
        if self.stop_deadline is None:
            timeout = _SUPERVISOR_CLEANUP_TIMEOUT_SECONDS
            if signum == signal.SIGINT and self.control is not None:
                assert self.status is not None
                os.pwrite(self.status.fileno(), struct.pack("!i", signum), 8)
                timeout += _CHILD_TERM_GRACE_SECONDS
            self.stop_deadline = time.monotonic() + timeout
        if self.control is not None:
            self.control.close_stop()
            with suppress(ProcessLookupError):
                os.kill(self.proc.pid, signal.SIGCONT)
        else:
            _kill_proc(self.proc)

    def poll(self) -> int | None:
        if self.closed:
            return self.returncode
        try:
            wait_proc(self.proc, 0, self.control)
        except subprocess.TimeoutExpired:
            if self.stop_deadline is None or time.monotonic() < self.stop_deadline:
                return None
            # A stalled or re-stopped supervisor cannot prevent the parent
            # entering bounded cleanup. Repeated requests retain this deadline.
        self.close()
        return self.returncode

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            try:
                timeout = (
                    _SUPERVISOR_CLEANUP_TIMEOUT_SECONDS
                    if self.stop_deadline is None
                    else max(0, self.stop_deadline - time.monotonic())
                )
                finish_proc(self.proc, self.label, self.control, timeout=timeout)
                self.returncode = read_result(
                    self.proc, self.status, self.label, self.command, stopped_ok=True
                )
            finally:
                self.stack.close()
        except ProcessError:
            raise
        except BaseException as error:
            raise CleanupError(
                f"[{self.label}] cleanup unconfirmed: {error}"
            ) from error


if __name__ == "__main__":
    if sys.argv[1:2] != ["--supervise-command"]:
        raise SystemExit(
            "internal process supervisor; use run_tests.py or autoreview.py"
        )
    _supervise_command(
        int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
    )
    raise SystemExit("process supervisor returned without terminating its group")
