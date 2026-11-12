"""Megakernel timelines (the ``megakernel_timeline`` setting, or
``HELION_LOGS=timeline``): the buffers the launcher hands megakernels, written
out as text at exit (or on ``write()``), read back and drawn.

    python -m helion.runtime.timeline plot <timeline.txt> [-o out.png] [--launch N]

A timeline file has a header, a per-root summary over the kept launches and
then each launch's records, in microseconds from the launch's first worker
entry on its rank (%globaltimer is per GPU, so ranks are not aligned)::

    launch <epoch> <first entry, ns>
    worker <worker> <sm> <entry> <exit>
    task <root> <task> <worker> <start> <ready> <end>

``ready`` is when the task's last task-level scheduler wait ended (its start
when it had none): ``ready - start`` is time waiting, ``end - ready`` working.
What a root waits for ahead of its task loop (root barriers, hoisted waits, a
dry pass) is in no task's window; the summary's ``idle``, the time since the
worker's previous task ended (or it entered) when a task starts, shows it.
Tasks are numbered as their root schedules them.  Under a ring kernel the
stamps are the compute partition's.  Stamps cost time themselves and quantize
to the GPU's global-timer step (in the header), so compare times within one
timeline rather than against unstamped runs.

A launch's records go to the section of its epoch, which the kernel's launch
state counts per stream.  A CUDA graph whose capture first allocated that
state (no launch on the capturing stream before) zeroes it on every replay, so
only its last replay is kept; launch once on that stream before capturing to
keep the last 16 replays.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import dataclasses
import functools
import logging
import os
import pathlib
import statistics
from typing import TYPE_CHECKING

import torch

from .._compiler.cross_loop_timeline import EPOCHS
from .._compiler.cross_loop_timeline import TASK_WORDS
from .._compiler.cross_loop_timeline import WORKER_WORDS
from .._compiler.cross_loop_timeline import TimelineMeta

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

# data_ptr -> (buffer, metadata from the compiler, rank, file name part)
_buffers: dict[int, tuple[torch.Tensor, TimelineMeta, int, str]] = {}


@dataclasses.dataclass
class Task:
    root: int
    task: int
    worker: int
    start: float
    ready: float
    end: float


@dataclasses.dataclass
class Launch:
    epoch: int
    origin_ns: int
    # worker -> (sm, entry, exit)
    workers: dict[int, tuple[int, float, float]]
    tasks: list[Task]

    @property
    def span(self) -> float:
        return max(end for _sm, _start, end in self.workers.values())


@dataclasses.dataclass
class Timeline:
    kernel: str
    rank: int
    workers: int
    dispatch: str
    launches_total: int
    timer_step_ns: int
    # root -> (task count, name)
    roots: dict[int, tuple[int, str]]
    launches: list[Launch]


@functools.cache
def _write_at_exit() -> None:
    atexit.register(write)


def register(buffer: torch.Tensor, meta: TimelineMeta) -> None:
    """Keep a megakernel's timeline buffer to write out, named by this process's
    rank (read now: the process group may be gone at exit), or by its device
    outside a process group.  Buffers stay for the process's life, so a kernel
    dropped before the exit is written too."""
    _write_at_exit()
    if buffer.data_ptr() not in _buffers:
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        rank = torch.distributed.get_rank() if distributed else 0
        where = (
            f"rank{rank}"
            if distributed
            else f"{buffer.device.type}{buffer.device.index}"
        )
        _buffers[buffer.data_ptr()] = (buffer, meta, rank, where)


def write() -> list[str]:
    """Write every megakernel timeline of this process; returns the paths."""
    seen: collections.Counter[str] = collections.Counter()
    paths = []
    for buffer, meta, rank, where in _buffers.values():
        stem = f"{meta['prefix']}.{meta['kernel']}"
        path = f"{stem}.{seen[stem]}.{where}.txt"
        seen[stem] += 1
        torch.cuda.synchronize(buffer.device)
        timeline = decode(buffer.cpu(), meta, rank)
        if timeline.launches:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            pathlib.Path(path).write_text(format_timeline(timeline))
            paths.append(path)
    if paths:
        log.info("megakernel timelines: %s", " ".join(paths))
    return paths


def decode(words: torch.Tensor, meta: TimelineMeta, rank: int = 0) -> Timeline:
    """The complete launches a timeline buffer holds, oldest first."""
    workers, section = meta["workers"], meta["section"]
    roots = {root: (first, count, name) for root, first, count, name in meta["roots"]}
    launches = []
    total = 0
    for index in range(EPOCHS):
        part = words[index * section : (index + 1) * section]
        header = part[: workers * WORKER_WORDS].view(workers, WORKER_WORDS)
        epochs = header[:, 0] >> 32
        epoch = int(epochs.max())
        total = max(total, epoch)
        # A launch some worker has not entered or not exited (its exit still
        # an older launch's, or none) is torn.
        if (
            epoch == 0
            or bool((epochs != epoch).any())
            or bool((header[:, 2] < header[:, 1]).any())
        ):
            continue
        origin = int(header[:, 1].min())

        def us(ns: int, origin: int = origin) -> float:
            return (ns - origin) / 1000

        launch_workers = {
            worker: (int(word) & 0xFFFFFFFF, us(int(entry)), us(int(leave)))
            for worker, (word, entry, leave) in enumerate(header.tolist())
        }
        tasks = []
        for root, (first, count, _name) in roots.items():
            records = part[first : first + TASK_WORDS * count].view(count, TASK_WORDS)
            for task, (word, start, ready, end) in enumerate(records.tolist()):
                if word >> 32 != epoch:
                    continue
                tasks.append(
                    Task(
                        root,
                        task,
                        (word & 0xFFFFFFFF) - 1,
                        us(start),
                        us(max(ready, start)),
                        us(end),
                    )
                )
        launches.append(Launch(epoch, origin, launch_workers, tasks))
    launches.sort(key=lambda launch: launch.epoch)
    stamps = words.view(EPOCHS, section)[:, workers * WORKER_WORDS :]
    stamps = torch.unique(stamps.reshape(-1, TASK_WORDS)[:, 1:])
    steps = torch.diff(stamps[stamps > 0])
    return Timeline(
        kernel=meta["kernel"],
        rank=rank,
        workers=workers,
        dispatch=meta["dispatch"],
        launches_total=total,
        timer_step_ns=int(steps.min()) if steps.numel() else 0,
        roots={root: (count, name) for root, (_f, count, name) in roots.items()},
        launches=launches,
    )


def _idle(launch: Launch) -> dict[tuple[int, int], float]:
    """Per (root, task): how long its worker had run no task when it started
    (since its previous task ended, or since it entered)."""
    by_worker: dict[int, list[Task]] = collections.defaultdict(list)
    for task in launch.tasks:
        by_worker[task.worker].append(task)
    idle = {}
    for worker, tasks in by_worker.items():
        last = launch.workers[worker][1]
        for task in sorted(tasks, key=lambda t: t.start):
            # A continuation task nests in another: no idle time.
            idle[task.root, task.task] = max(0.0, task.start - last)
            last = max(last, task.end)
    return idle


def _median_max(values: Sequence[float]) -> str:
    if not values:
        return "-"
    return f"{statistics.median(values):.2f}/{max(values):.2f}"


def format_timeline(timeline: Timeline) -> str:
    launches = timeline.launches
    spans = [launch.span for launch in launches]
    lines = [
        "helion megakernel timeline",
        (
            f"kernel {timeline.kernel}  rank {timeline.rank}  workers "
            f"{timeline.workers}  dispatch {timeline.dispatch}"
        ),
        (
            f"launches {len(launches)} kept of {timeline.launches_total}  "
            f"globaltimer step {timeline.timer_step_ns} ns"
        ),
        (
            "times in us from each launch's first worker entry; stamping adds "
            "work, so compare times within this timeline"
        ),
        "",
        (
            f"summary: span median {statistics.median(spans):.2f} "
            f"(min {min(spans):.2f}, max {max(spans):.2f})"
        ),
        (
            "  root  tasks  start  end  (medians over launches)  idle med/max  "
            "wait med/max  work med/max  name"
        ),
    ]
    idle = [_idle(launch) for launch in launches]
    for root, (count, name) in timeline.roots.items():
        mine = [[t for t in launch.tasks if t.root == root] for launch in launches]
        starts = [min(t.start for t in tasks) for tasks in mine if tasks]
        ends = [max(t.end for t in tasks) for tasks in mine if tasks]
        idles = [
            gaps[t.root, t.task]
            for tasks, gaps in zip(mine, idle, strict=True)
            for t in tasks
        ]
        waits = [t.ready - t.start for tasks in mine for t in tasks]
        works = [t.end - t.ready for tasks in mine for t in tasks]
        lines.append(
            f"  {root:<4}  {count:<5}  "
            + (
                f"{statistics.median(starts):.2f}  {statistics.median(ends):.2f}"
                if starts
                else "-  -"
            )
            + f"  {_median_max(idles)}  {_median_max(waits)}  {_median_max(works)}"
            f"  {name}"
        )
    lines.append("")
    for launch in launches:
        lines.append(f"launch {launch.epoch} {launch.origin_ns}")
        lines += [
            f"worker {worker} {sm} {entry:.3f} {leave:.3f}"
            for worker, (sm, entry, leave) in launch.workers.items()
        ]
        lines += [
            f"task {t.root} {t.task} {t.worker} {t.start:.3f} {t.ready:.3f} {t.end:.3f}"
            for t in launch.tasks
        ]
    return "\n".join(lines) + "\n"


def load(path: str) -> Timeline:
    """Read a timeline file back."""
    lines = pathlib.Path(path).read_text().splitlines()
    head = lines[1].split()
    counts = lines[2].split()
    timeline = Timeline(
        kernel=head[1],
        rank=int(head[3]),
        workers=int(head[5]),
        dispatch=head[7],
        launches_total=int(counts[4]),
        timer_step_ns=int(counts[7]),
        roots={},
        launches=[],
    )
    summary = lines.index("") + 3
    for line in lines[summary:]:
        if not line:
            break
        root, count, *_columns = line.split(maxsplit=6)
        name = line.split("  ")[-1]
        timeline.roots[int(root)] = (int(count), name)
    for line in lines[summary:]:
        kind, *fields = line.split() or [""]
        if kind == "launch":
            timeline.launches.append(Launch(int(fields[0]), int(fields[1]), {}, []))
        elif kind == "worker":
            worker, sm = int(fields[0]), int(fields[1])
            timeline.launches[-1].workers[worker] = (
                sm,
                float(fields[2]),
                float(fields[3]),
            )
        elif kind == "task":
            root, task, worker = (int(x) for x in fields[:3])
            start, ready, end = (float(x) for x in fields[3:])
            timeline.launches[-1].tasks.append(
                Task(root, task, worker, start, ready, end)
            )
    return timeline


# The categorical palette (fixed order); roots past it repeat it hatched.
_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_HATCHES = ("", "//", "\\\\", "xx")


def plot(timeline: Timeline, output: str, launch: int = -1) -> None:
    """A Gantt chart of one launch: a row per SM lane (the workers resident on an
    SM together), a bar per task (pale while it waits, solid while it works),
    colored by root."""
    # A bare Figure draws without pyplot, leaving the process's backend alone.
    import matplotlib.figure  # pyrefly: ignore [missing-import]
    import matplotlib.patches  # pyrefly: ignore [missing-import]

    chosen = timeline.launches[launch]
    # Workers resident together on an SM take its lanes; one after another, one.
    lanes: dict[int, list[float]] = {}  # SM -> each lane's last exit
    lane_of: dict[int, tuple[int, int]] = {}
    for worker, (sm, entry, leave) in sorted(
        chosen.workers.items(), key=lambda item: (item[1][0], item[1][1])
    ):
        ends = lanes.setdefault(sm, [])
        lane = next((i for i, end in enumerate(ends) if end <= entry), len(ends))
        ends[lane : lane + 1] = [leave]
        lane_of[worker] = (sm, lane)
    rows = {lane: index for index, lane in enumerate(sorted(set(lane_of.values())))}
    row = {worker: rows[lane] for worker, lane in lane_of.items()}
    fig = matplotlib.figure.Figure(figsize=(14, max(3.0, 0.06 * len(rows) + 1.5)))
    ax = fig.subplots()
    for task in chosen.tasks:
        color = _COLORS[task.root % len(_COLORS)]
        hatch = _HATCHES[task.root // len(_COLORS) % len(_HATCHES)]
        y = row[task.worker]
        ax.broken_barh(
            [(task.start, task.ready - task.start)],
            (y - 0.4, 0.8),
            facecolors=color,
            alpha=0.3,
            linewidth=0,
        )
        ax.broken_barh(
            [(task.ready, task.end - task.ready)],
            (y - 0.4, 0.8),
            facecolors=color,
            edgecolor="#fcfcfb",
            linewidth=0.3,
            hatch=hatch,
        )
    for worker, (_sm, entry, leave) in chosen.workers.items():
        y = row[worker]
        ax.plot([entry, leave], [y, y], color="#52514e", linewidth=0.3)
    ax.set_xlabel("us from the first worker entry")
    ax.set_ylabel("SM")
    ax.set_ylim(-1, len(rows))
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{timeline.kernel}  rank {timeline.rank}  launch {chosen.epoch}  "
        f"span {chosen.span:.1f} us  (pale: waiting)",
        loc="left",
        fontsize=10,
        color="#0b0b0b",
    )
    handles = [
        matplotlib.patches.Patch(
            facecolor=_COLORS[root % len(_COLORS)],
            hatch=_HATCHES[root // len(_COLORS) % len(_HATCHES)],
            label=name,
        )
        for root, (_count, name) in timeline.roots.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m helion.runtime.timeline")
    commands = parser.add_subparsers(dest="command", required=True)
    plot_parser = commands.add_parser("plot", help="draw one launch as a Gantt chart")
    plot_parser.add_argument("timeline")
    plot_parser.add_argument("-o", "--output", help="image path (default: .png)")
    plot_parser.add_argument(
        "--launch", type=int, default=-1, help="launch index (default: the last)"
    )
    args = parser.parse_args(argv)
    output = args.output or os.path.splitext(args.timeline)[0] + ".png"
    plot(load(args.timeline), output, args.launch)
    print(output)


if __name__ == "__main__":
    main()
