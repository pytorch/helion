from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from helion._testing import distributed_diagnostics as diagnostics

pytestmark = pytest.mark.cpu_only


class TestRemoteCopyDiagnostics(unittest.TestCase):
    def test_normal_return_keeps_process_watchdogs(self) -> None:
        worker = Mock()
        writes = Mock()
        handler = Mock()
        with (
            patch.object(diagnostics.threading, "Thread", return_value=worker),
            patch.object(diagnostics.os, "write", writes),
            patch.object(diagnostics, "faulthandler", handler),
            patch.object(diagnostics.time, "monotonic", return_value=100.0),
        ):
            with diagnostics.queued_copy_diagnostics(2) as mark:
                mark("init")
                writes.assert_not_called()
                handler.dump_traceback_later.assert_called_once_with(
                    20.5, repeat=True, file=2
                )
                worker.start.assert_called_once_with()
            # The mocked worker never runs; no process-global watchdog survives
            # this test. A real rank retains both monitors after body return.
            handler.cancel_dump_traceback_later.assert_not_called()
            worker.join.assert_not_called()
            writes.assert_called_once()
            fd, record = writes.call_args.args
            self.assertEqual(fd, 2)
            complete = json.loads(record.strip().removeprefix(b"REMOTE_COPY_COMPLETE "))
            self.assertEqual(complete["rank"], 2)
            self.assertEqual(complete["pid"], os.getpid())
            self.assertEqual(complete["sequence"], -1)
            self.assertEqual(complete["started"], 100.0)
            self.assertEqual(complete["elapsed"], 0.0)

    def test_exception_stops_python_reporter_without_canceling_c_watchdog(self) -> None:
        worker = Mock()
        stopped = Mock()
        writes = Mock()
        handler = Mock()
        with (
            patch.object(diagnostics.threading, "Thread", return_value=worker),
            patch.object(diagnostics.threading, "Event", return_value=stopped),
            patch.object(diagnostics.os, "write", writes),
            patch.object(diagnostics, "faulthandler", handler),
        ):
            with (
                self.assertRaisesRegex(RuntimeError, "body failed"),
                diagnostics.queued_copy_diagnostics(1) as mark,
            ):
                mark("init")
                raise RuntimeError("body failed")
            stopped.set.assert_called_once_with()
            handler.cancel_dump_traceback_later.assert_not_called()
            worker.join.assert_called_once_with(timeout=1)
            writes.assert_not_called()

    def test_thread_start_failure_keeps_independent_watchdog(self) -> None:
        worker = Mock()
        stopped = Mock()
        worker.start.side_effect = RuntimeError("thread failed")
        handler = Mock()
        with (
            patch.object(diagnostics.threading, "Thread", return_value=worker),
            patch.object(diagnostics.threading, "Event", return_value=stopped),
            patch.object(diagnostics, "faulthandler", handler),
        ):
            with (
                self.assertRaisesRegex(RuntimeError, "thread failed"),
                diagnostics.queued_copy_diagnostics(0),
            ):
                self.fail("failed watchdog start entered the test")
            stopped.set.assert_called_once_with()
            handler.cancel_dump_traceback_later.assert_not_called()
            worker.join.assert_not_called()

    def test_bounded_records_snapshot_frames_before_writing(self) -> None:
        worker = Mock()
        stopped = Mock()
        stopped.wait.return_value = False
        frame = SimpleNamespace(
            f_code=SimpleNamespace(co_filename="example.py", co_name="work"),
            f_lineno=17,
            f_back=None,
        )

        def write(fd: int, record: bytes) -> None:
            if record.startswith(b"\nREMOTE_COPY_STAGES "):
                frame.f_lineno = 23

        with (
            patch.object(
                diagnostics.threading, "Thread", return_value=worker
            ) as thread,
            patch.object(diagnostics.threading, "Event", return_value=stopped),
            patch.object(diagnostics.select, "PIPE_BUF", 512),
            patch.object(diagnostics.os, "write", side_effect=write) as writes,
            patch.object(diagnostics.sys, "_current_frames", return_value={-1: frame}),
            patch.object(diagnostics, "faulthandler"),
        ):
            with diagnostics.queued_copy_diagnostics(2) as mark:
                for _ in range(20):
                    mark("\U0001f30d" * 96)
            thread.call_args.kwargs["target"]()
        stage_records = []
        frame_lines = []
        for call in writes.call_args_list:
            fd, record = call.args
            self.assertEqual(fd, 2)
            self.assertLessEqual(len(record), 512)
            label, payload = record.strip().split(b" ", 1)
            data = json.loads(payload)
            self.assertEqual(data["rank"], 2)
            self.assertEqual(data["pid"], os.getpid())
            if label == b"REMOTE_COPY_STAGES":
                self.assertTrue(data["truncated"])
                self.assertGreater(data["original_bytes"], 512)
                self.assertEqual(data["omitted_fields"], ["stages", "history_limit"])
                stage_records.append(data["sequence"])
            elif label == b"REMOTE_COPY_FRAME":
                frame_lines.append(data["line"])
        self.assertEqual(stage_records, [0, 1, 2, 3])
        self.assertEqual(frame_lines, [17, 23, 23, 23])

    @pytest.mark.slow
    @unittest.skipUnless(os.name == "posix", "shared pipe descriptors require POSIX")
    def test_concurrent_reports_remain_attributable_through_teardown(self) -> None:
        script = """
import os
import sys
import threading
from helion._testing import distributed_diagnostics as diagnostics
rank = int(sys.argv[1])
gate = threading.Event()
class GatedThread(threading.Thread):
    def run(self):
        gate.wait()
        super().run()
diagnostics.threading.Thread = GatedThread
with diagnostics.queued_copy_diagnostics(rank, timeout=0.4) as mark:
    mark("body_work")
# Release the reporter only after the body has returned, without requiring the
# parent to acknowledge a sample within a short wall-clock scheduling window.
os.write(1, b"ready\\n")
assert sys.stdin.readline() == "start\\n"
gate.set()
assert sys.stdin.readline() == "exit\\n"
"""
        read_fd, write_fd = os.pipe()
        pipe_limit = os.fpathconf(write_fd, "PC_PIPE_BUF")
        children: list[subprocess.Popen[bytes]] = []
        captured = bytearray()
        pending = bytearray()
        ready: set[int] = set()
        complete: set[int] = set()
        frames: set[int] = set()
        sequences: dict[int, list[int]] = {0: [], 1: []}
        starts: dict[int, float] = {}
        deadline = time.monotonic() + 30
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(read_fd, selectors.EVENT_READ, ("diagnostics", -1))
                for rank in range(2):
                    child = subprocess.Popen(
                        [sys.executable, "-c", script, str(rank)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=write_fd,
                    )
                    children.append(child)
                    assert child.stdout is not None
                    selector.register(
                        child.stdout, selectors.EVENT_READ, ("ready", rank)
                    )
                os.close(write_fd)
                write_fd = -1
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    self.assertGreater(remaining, 0, captured.decode(errors="replace"))
                    events = selector.select(remaining)
                    self.assertTrue(events, captured.decode(errors="replace"))
                    for key, _ in events:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fd)
                            continue
                        kind, rank = key.data
                        if kind == "ready":
                            self.assertEqual(chunk, b"ready\n")
                            ready.add(rank)
                            if len(ready) == 2:
                                for child in children:
                                    assert child.stdin is not None
                                    child.stdin.write(b"start\n")
                                    child.stdin.flush()
                            continue
                        captured.extend(chunk)
                        pending.extend(chunk)
                        while b"\n" in pending:
                            line, _, remainder = pending.partition(b"\n")
                            pending = bytearray(remainder)
                            if not line.startswith(b"REMOTE_COPY_"):
                                continue  # Raw C fallback output has no rank identity.
                            self.assertLessEqual(len(line) + 2, pipe_limit)
                            label, payload = line.split(b" ", 1)
                            record = json.loads(payload)
                            rank = record["rank"]
                            self.assertIn(rank, (0, 1))
                            self.assertEqual(record["pid"], children[rank].pid)
                            self.assertGreater(record["started"], 0)
                            self.assertEqual(
                                starts.setdefault(rank, record["started"]),
                                record["started"],
                            )
                            sequence = record["sequence"]
                            if label == b"REMOTE_COPY_COMPLETE":
                                self.assertEqual(sequence, -1)
                                complete.add(rank)
                            elif label == b"REMOTE_COPY_FRAME":
                                self.assertIn(sequence, range(4))
                                self.assertIn("function", record)
                                self.assertNotIn(
                                    record["function"], ("report", "watch")
                                )
                                frames.add(rank)
                            elif label == b"REMOTE_COPY_STAGES":
                                sequences[rank].append(sequence)
                                child = children[rank]
                                assert child.stdin is not None
                                self.assertIn(rank, complete)
                                self.assertEqual(record["history_limit"], 16)
                                self.assertEqual(
                                    record["stages"][-1][1], "body_returned"
                                )
                                if sequence == 3:
                                    child.stdin.write(b"exit\n")
                                    child.stdin.flush()
                for child in children:
                    self.assertEqual(
                        child.wait(timeout=max(0.01, deadline - time.monotonic())),
                        0,
                        captured.decode(errors="replace"),
                    )
        finally:
            if write_fd != -1:
                os.close(write_fd)
            os.close(read_fd)
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)
                if child.stdin is not None:
                    child.stdin.close()
                if child.stdout is not None:
                    child.stdout.close()
        self.assertEqual(complete, {0, 1})
        self.assertEqual(frames, {0, 1})
        self.assertEqual(sequences, {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]})

    @pytest.mark.slow
    @unittest.skipUnless(sys.platform == "linux", "requires /proc thread wait channels")
    def test_body_exception_propagates_before_full_stderr_is_drained(self) -> None:
        script = """
import json
import os
import sys
from helion._testing import distributed_diagnostics as diagnostics
original_thread = diagnostics.threading.Thread
workers = []
def record_thread(*args, **kwargs):
    worker = original_thread(*args, **kwargs)
    workers.append(worker)
    return worker
diagnostics.threading.Thread = record_thread
baseline_threads = [int(tid) for tid in os.listdir('/proc/self/task')]
os.set_blocking(2, False)
while True:
    try:
        os.write(2, b'x' * 4096)
    except BlockingIOError:
        break
os.set_blocking(2, True)
try:
    with diagnostics.queued_copy_diagnostics(0, timeout=0.1):
        info = {'baseline_threads': baseline_threads,
                'reporter_thread': workers[0].native_id}
        os.write(1, (json.dumps(info) + '\\n').encode())
        assert sys.stdin.readline() == 'raise\\n'
        raise RuntimeError('body failed')
except RuntimeError as exc:
    os.write(1, ('propagated:' + str(exc) + '\\n').encode())
    assert sys.stdin.readline() == 'exit\\n'
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            assert child.stdout is not None
            assert child.stdin is not None
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                self.assertTrue(selector.select(20), "child did not initialize")
                info = json.loads(child.stdout.readline())
                excluded = {*info["baseline_threads"], info["reporter_thread"]}
                deadline = time.monotonic() + 10
                while True:
                    blocked = []
                    observed = {}
                    for task in Path(f"/proc/{child.pid}/task").iterdir():
                        if int(task.name) not in excluded:
                            wait_channel = (task / "wchan").read_text().strip()
                            observed[int(task.name)] = wait_channel
                            # Kernel compiler clones may add a .llvm suffix.
                            if wait_channel.partition(".")[0].endswith("pipe_write"):
                                blocked.append(int(task.name))
                    if blocked:
                        break
                    self.assertLess(
                        time.monotonic(),
                        deadline,
                        f"C watchdog did not block on stderr: {observed}",
                    )
                    time.sleep(0.01)
                # The real C writer is blocked, distinct from the Python
                # reporter and main thread. Keep stderr undrained until the
                # original body exception is observed on this separate pipe.
                child.stdin.write(b"raise\n")
                child.stdin.flush()
                self.assertTrue(
                    selector.select(10), "body exception waited for stderr drainage"
                )
                self.assertEqual(child.stdout.readline(), b"propagated:body failed\n")
        finally:
            try:
                child.communicate(input=b"exit\n", timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=10)
        self.assertEqual(child.returncode, 0)

    @pytest.mark.slow
    @unittest.skipUnless(
        sys.platform == "linux", "requires a libc call holding the GIL"
    )
    def test_independent_fallback_reports_while_gil_is_held(self) -> None:
        script = """
import ctypes
from helion._testing.distributed_diagnostics import queued_copy_diagnostics
hold_gil = ctypes.PyDLL(None).sleep
hold_gil.argtypes = [ctypes.c_uint]
hold_gil.restype = ctypes.c_uint
with queued_copy_diagnostics(3, timeout=0.1) as mark:
    mark("holding_gil")
    hold_gil(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        fallback = result.stderr.find(b"Timeout (")
        complete = result.stderr.find(b"REMOTE_COPY_COMPLETE ")
        self.assertGreaterEqual(fallback, 0, result.stderr.decode(errors="replace"))
        self.assertGreater(complete, fallback, result.stderr.decode(errors="replace"))
