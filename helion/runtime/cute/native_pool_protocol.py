"""Finite spawned worker ownership and plain compile metadata transport."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from contextlib import suppress
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: str | Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze(value: object) -> object:
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return {"float": value.hex()}
    if type(value) in (tuple, list):
        return {
            type(value).__name__: [
                freeze(x) for x in cast("tuple[object, ...] | list[object]", value)
            ]
        }
    if type(value) is dict and all(
        type(k) is str for k in cast("dict[object, object]", value)
    ):
        return {
            "dict": [
                [k, freeze(v)]
                for k, v in sorted(cast("dict[str, object]", value).items())
            ]
        }
    raise TypeError(f"Unsupported compile metadata: {type(value)}")


def thaw(value: object) -> object:
    if not isinstance(value, dict):
        return value
    assert len(value) == 1
    kind, content = next(iter(value.items()))
    if kind == "float":
        return float.fromhex(content)
    if kind == "tuple":
        return tuple(thaw(x) for x in content)
    if kind == "list":
        return [thaw(x) for x in content]
    assert kind == "dict"
    return {k: thaw(v) for k, v in content}


def wrapper_ast(function: Callable[..., Any], kernel: object) -> str:
    import inspect

    source = inspect.getsource(inspect.unwrap(function))
    tree = ast.parse(source)
    tag = f"{cast('Any', kernel).__name__}_{id(kernel):x}"

    class Canonical(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            assert node.name == "_helion_cute_launch_" + tag
            node.name = "canonical_wrapper"
            return self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if node.value == tag:
                return ast.copy_location(ast.Constant("canonical_kernel_tag"), node)
            return node

    return ast.dump(Canonical().visit(tree), include_attributes=False)


@contextmanager
def interruption_cleanup() -> Iterator[None]:

    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"Parent interrupted by signal {signum}")

    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for signum in previous:
            signal.signal(signum, interrupted)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class NativeRequestTimeout(TimeoutError):
    """A supervisor-observed request deadline, distinct from startup/budget."""

    def __init__(self, worker: int, pid: int, request: int, timeout: float) -> None:
        super().__init__(f"Native worker {worker} startup/request exceeded {timeout}s")
        self.worker = worker
        self.pid = pid
        self.request = request


def run_workers(
    commands: list[tuple[list[str], dict[str, str]]],
    output: str | Path,
    *,
    workers: int,
    timeout: float,
    budget_exceeded: Callable[[], bool] = lambda: False,
    request_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    assert 1 <= workers <= 32 and timeout > 0
    output = Path(output)
    live = {}
    progress = {}
    records = []
    highwater = 0
    next_index = 0
    started = time.monotonic()
    with interruption_cleanup():
        try:
            while next_index < len(commands) or live:
                if budget_exceeded():
                    raise TimeoutError("Native batch budget exhausted")
                while next_index < len(commands) and len(live) < workers:
                    argv, env = commands[next_index]
                    handle = (output / f"worker-{next_index}.log").open("w")
                    try:
                        blocked = signal.pthread_sigmask(
                            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}
                        )
                        try:
                            process = subprocess.Popen(
                                argv,
                                env=env,
                                stdout=handle,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            live[process.pid] = (
                                process,
                                handle,
                                time.monotonic(),
                                next_index,
                            )
                        finally:
                            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
                    except BaseException:
                        if not any(handle is entry[1] for entry in live.values()):
                            handle.close()
                        raise
                    highwater = max(highwater, len(live))
                    next_index += 1
                for pid, (process, handle, begin, index) in list(live.items()):
                    code = process.poll()
                    if code is not None:
                        if code != 0:
                            raise RuntimeError(f"Native worker {index} exited {code}")
                        process.wait()
                        with suppress(ProcessLookupError):
                            os.killpg(pid, 0)
                            raise RuntimeError(
                                f"Worker {index} left a live process group"
                            )
                        handle.close()
                        records.append(
                            {
                                "pid": pid,
                                "index": index,
                                "exitcode": code,
                                "seconds": time.monotonic() - begin,
                            }
                        )
                        live.pop(pid)
                    else:
                        now = time.monotonic()
                        active = output / f"process-{index}" / "ACTIVE.json"
                        if request_counts is not None and active.exists():
                            current = json.loads(active.read_text())
                            request_index = current["index"]
                            assert (
                                type(request_index) is int
                                and 0 <= request_index < request_counts[index]
                            )
                            prior, since = progress.get(index, (-1, begin))
                            assert request_index >= prior
                            if request_index != prior:
                                progress[index] = (request_index, now)
                                since = now
                        else:
                            since = begin
                        if now - since > timeout:
                            if index in progress:
                                raise NativeRequestTimeout(
                                    index, pid, progress[index][0], timeout
                                )
                            raise TimeoutError(
                                f"Native worker {index} startup/request exceeded {timeout}s"
                            )
                        limit = 1 + (request_counts[index] if request_counts else 0)
                        if now - begin > timeout * limit:
                            raise TimeoutError(
                                f"Native worker {index} exceeded finite batch timeout"
                            )
                if live:
                    time.sleep(0.05)
        finally:
            blocked = signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}
            )
            try:
                for pid in live:
                    with suppress(ProcessLookupError):
                        os.killpg(pid, signal.SIGTERM)
                for pid, (process, handle, begin, index) in list(live.items()):
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(pid, signal.SIGKILL)
                        process.wait()
                    finally:
                        with suppress(ProcessLookupError):
                            os.killpg(pid, signal.SIGKILL)
                        handle.close()
                    records.append(
                        {
                            "pid": pid,
                            "index": index,
                            "exitcode": process.returncode,
                            "seconds": time.monotonic() - begin,
                            "cleanup": True,
                        }
                    )
                write(
                    output / "REAPS.json",
                    {
                        "records": records,
                        "highwater": highwater,
                        "wall_seconds": time.monotonic() - started,
                        "all_direct_children_reaped": True,
                    },
                )
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
    assert len(records) == len(commands) and all(x["exitcode"] == 0 for x in records)
    return records
