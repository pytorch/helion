"""Global-timer stamps of a megakernel's launches (the ``megakernel_timeline``
setting), which ``helion.runtime.timeline`` writes out.

Each launch writes one section of a launcher-owned buffer, picked by its epoch,
so the last ``EPOCHS`` launches stay and CUDA graph replays need no host work.
A section holds, per worker, ``(epoch << 32 | smid, entry, exit)`` (entry: when
its grid wait released, before it reads the launch state) and, per
task of every root, ``(epoch << 32 | worker + 1, start, ready, end)``: ``ready``
is when the task's last task-level scheduler wait ended. A continuation task
runs inside the task whose arrival completed it, so its window nests there.
Every word is written by a relaxed atomic max, so any thread or partition may
stamp, repeats keep the latest time and a word left by an older launch never
wins.
"""

from __future__ import annotations

import ast
import dataclasses
import os
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import cast

from .ast_extension import statement_from_string
from .cross_loop_report import log_once
from .device_function import SyncRole
from .host_function import HostFunction

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from .device_function import DeviceFunction


# The buffer layout, which ``helion.runtime.timeline`` reads: launches kept,
# words per worker and words per task of a launch's section.
EPOCHS = 16
WORKER_WORDS = 3
TASK_WORDS = 4


class TimelineMeta(TypedDict):
    """What the runtime writes a timeline buffer with."""

    prefix: str
    kernel: str
    workers: int
    dispatch: str
    # Words of one launch's section.
    section: int
    # (root, its first task's word in a section, its task count, its name)
    roots: list[tuple[int, int, int, str]]


_NOW = (
    "tl.inline_asm_elementwise(asm='mov.u64 $0, %globaltimer;', constraints='=l', "
    "args=[], dtype=tl.int64, is_pure=False, pack=1)"
)
_SMID = (
    "tl.cast(tl.inline_asm_elementwise(asm='mov.u32 $0, %smid;', constraints='=r', "
    "args=[], dtype=tl.int32, is_pure=False, pack=1), tl.int64)"
)


@dataclasses.dataclass
class Timeline:
    device_function: DeviceFunction
    # The launcher-owned buffer argument, its persistent-state slot, and the
    # name of this launch's section.
    buffer: str
    slot: int
    base: str
    # The clock read as the grid wait releases.
    released: str
    # root -> (its first task's word in a section, its task count)
    tasks: dict[int, tuple[int, int]]
    meta: TimelineMeta

    @staticmethod
    def create(
        device_function: DeviceFunction,
        *,
        prefix: str | os.PathLike[str],
        workers: int,
        dispatch: str,
        roots: dict[int, tuple[int, str]],
        register: Callable[[str], tuple[str, int]],
    ) -> Timeline:
        """Lay out the buffer of ``workers`` and ``roots`` (root -> task count,
        name) and register it (``register(numel)`` returns its argument and
        persistent-state slot)."""
        tasks: dict[int, tuple[int, int]] = {}
        word = workers * WORKER_WORDS
        for root, (count, _name) in roots.items():
            tasks[root] = (word, count)
            word += count * TASK_WORDS
        buffer, slot = register(str(EPOCHS * word))
        log_once(
            f"megakernel timeline: {workers} workers, "
            f"{sum(count for count, _name in roots.values())} tasks, "
            f"{-(-word * 8 // 1024)} KiB a launch"
        )
        return Timeline(
            device_function,
            buffer=buffer,
            slot=slot,
            base=device_function.new_var("tile_dependency_timeline_base", dce=False),
            released=device_function.new_var(
                "tile_dependency_timeline_released", dce=False
            ),
            tasks=tasks,
            meta={
                # Absolute now: the files are written at exit.
                "prefix": os.path.abspath(prefix),
                "kernel": HostFunction.current().name,
                "workers": workers,
                "dispatch": dispatch,
                "section": word,
                "roots": [
                    (root, tasks[root][0], count, name)
                    for root, (count, name) in roots.items()
                ],
            },
        )

    def clock(self) -> ast.stmt:
        """Read the clock right after the grid wait, ahead of the launch state."""
        self.device_function.sync_roles[self.released] = SyncRole("stamp")
        return statement_from_string(f"{self.released} = {_NOW}")

    def setup(self, epoch: str, worker: str) -> list[ast.stmt]:
        """This launch's section and the worker's entry stamps (after
        ``clock``)."""
        index = f"tl.cast({epoch}, tl.int64)"
        word = f"{worker} * {WORKER_WORDS}"
        return [
            statement_from_string(
                f"{self.base} = {self.buffer} + {index} % {EPOCHS} * {self.meta['section']}"
            ),
            self._max(word, f"({index} << 32) | {_SMID}"),
            self._max(f"{word} + 1", self.released),
        ]

    def exit(self, worker: str) -> list[ast.stmt]:
        return [self._max(f"{worker} * {WORKER_WORDS} + 2", _NOW)]

    def task(
        self, root: int, task: str, epoch: str, worker: str, body: list[ast.stmt]
    ) -> list[ast.stmt]:
        """``body`` (one task of ``root``, ``task`` its index among the root's
        tasks) between its start and end stamps."""
        word = self._task_word(root, task)
        index = f"tl.cast({epoch}, tl.int64)"
        return [
            self._max(word, f"({index} << 32) | (tl.cast({worker}, tl.int64) + 1)"),
            self._max(f"{word} + 1", _NOW),
            *body,
            self._max(f"{word} + 3", _NOW),
        ]

    def ready(self, root: int, task: str, body: list[ast.stmt]) -> None:
        """Stamp the end of each run of scheduler spins in ``body`` and in its
        conditionals (``DeviceFunction.spin_ends``; spins in loops run per
        iteration and are left out): the latest stamp is when the task's last
        wait ended."""
        ends = self.device_function.spin_ends
        waits = self.device_function.sync_names("wait")
        for block in _blocks(body):
            # A spin's end is the last statement of the block assigning its
            # names: a poll's first loads, or a token's constant pre-assignment,
            # assign them before its spin does.
            last = {
                name: index
                for index, statement in enumerate(block)
                for name in _assigns(statement) & ends
            }
            # After a spin's end and the wait statements right after it (its
            # fence, the next spin's first poll), once something else follows.
            stamps: list[int] = []
            pending: int | None = None
            for index, statement in enumerate(block):
                names = _assigns(statement)
                if index in last.values() and isinstance(
                    cast("ast.Assign", statement).value, ast.Call
                ):
                    pending = index
                elif pending is not None and names and names <= waits:
                    if index == pending + 1:
                        pending = index
                elif pending is not None and not isinstance(statement, ast.While):
                    stamps.append(pending)
                    pending = None
            if pending is not None:
                stamps.append(pending)
            for index in reversed(stamps):
                block.insert(
                    index + 1, self._max(f"{self._task_word(root, task)} + 2", _NOW)
                )

    def _task_word(self, root: int, task: str) -> str:
        first, _count = self.tasks[root]
        return f"{first} + ({task}) * {TASK_WORDS}"

    def _max(self, offset: str, value: str) -> ast.stmt:
        name = self.device_function.new_var("tile_dependency_timeline", dce=False)
        self.device_function.sync_roles[name] = SyncRole("stamp")
        return statement_from_string(
            f"{name} = tl.atomic_max({self.base} + ({offset}), {value}, "
            "sem='relaxed', scope='gpu')"
        )


def _blocks(body: list[ast.stmt]) -> Iterator[list[ast.stmt]]:
    yield body
    for statement in body:
        if isinstance(statement, ast.If):
            yield from _blocks(statement.body)
            yield from _blocks(statement.orelse)


def _assigns(statement: ast.stmt) -> set[str]:
    """The names ``statement`` assigns (a tuple target's too)."""
    if not isinstance(statement, ast.Assign):
        return set()
    return {
        name.id
        for target in statement.targets
        for name in (target.elts if isinstance(target, ast.Tuple) else [target])
        if isinstance(name, ast.Name)
    }
