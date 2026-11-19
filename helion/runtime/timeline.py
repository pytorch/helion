"""Megakernel timelines (the ``megakernel_timeline`` setting, or
``HELION_LOGS=timeline``): the buffers the launcher hands megakernels, written
out as text at exit (or on ``write()``), read back and drawn.

    python -m helion.runtime.timeline plot <timeline.txt> [-o out.png] [--launch N]
        [--view both|overlap|sm]

A timeline file has a header, a per-root summary over the kept launches and
then each launch's records, in microseconds from the launch's first worker
entry on its rank (%globaltimer is per GPU, so ranks are not aligned)::

    launch <epoch> <first entry, ns>
    worker <worker> <sm> <entry> <exit>
    task <root> <task> <worker> <start> <ready> <end>

``ready`` is when the task's last task-level wait ended (its start when it
had none): ``end - ready`` is work, and ``ready - start`` (the summary's
``wait``) is the waiting plus any of the task's work placed ahead of that
wait (a scoped wait sits right before the first statement that needs it, an
in-band poll right before its read).
What a root waits for ahead of its task loop (root barriers, hoisted waits, a
dry pass) is in no task's window; the summary's ``idle``, the time since the
worker's previous task ended (or it entered) when a task starts, shows it.
Tasks are numbered as their root schedules them.  Under a ring kernel the
stamps are the compute partition's.  Stamps cost time themselves and quantize
to the GPU's global-timer step (in the header), so compare times within one
timeline rather than against unstamped runs.

A launch's records go to the section of its epoch, which the kernel's launch
state counts.  A single-process kernel keeps that state per stream: a CUDA
graph whose capture first allocated it (no launch on the capturing stream
before) zeroes it on every replay, so only its last replay is kept; launch
once on that stream before capturing to keep the last 16 replays.  A kernel
in a process group keeps one state per process, which must exist before
capture, so its graph replays are always kept.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import dataclasses
import functools
import logging
import math
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

    from matplotlib.axes import Axes  # pyrefly: ignore [missing-import]

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
    rank (read now: the process group may be gone at exit), or outside a
    process group by its device and pid (processes that each see one GPU all
    see it as device 0).  Buffers stay for the process's life, so a kernel
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
            else f"{buffer.device.type}{buffer.device.index}.pid{os.getpid()}"
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
        root, count, *_columns, name = line.split(maxsplit=7)
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


# Tableau 20, in the order roots take it; roots past it repeat it hatched.
_COLORS = (
    "#4e79a7",
    "#a0cbe8",
    "#f28e2b",
    "#ffbe7d",
    "#e15759",
    "#ff9d9a",
    "#b07aa1",
    "#d4a6c8",
    "#9d7660",
    "#d7b5a6",
    "#bab0ac",
    "#59a14f",
    "#86bcb6",
    "#f1ce63",
    "#79706e",
    "#8cd17d",
    "#499894",
    "#d37295",
    "#fabfd2",
    "#b6992d",
)
_HATCHES = ("", "//", "\\\\", "xx")
VIEWS = ("both", "overlap", "sm")


def _style(root: int) -> tuple[str, str]:
    return (
        _COLORS[root % len(_COLORS)],
        _HATCHES[root // len(_COLORS) % len(_HATCHES)],
    )


def _sm_lanes(chosen: Launch) -> tuple[dict[int, int], list[int]]:
    """Each worker's row, and each row's SM: workers resident on an SM together
    take its lanes, one after another one lane."""
    lanes: dict[int, list[float]] = {}  # SM -> each lane's last exit
    lane_of: dict[int, tuple[int, int]] = {}
    for worker, (sm, entry, leave) in sorted(
        chosen.workers.items(), key=lambda item: (item[1][0], item[1][1])
    ):
        ends = lanes.setdefault(sm, [])
        lane = next((i for i, end in enumerate(ends) if end <= entry), len(ends))
        ends[lane : lane + 1] = [leave]
        lane_of[worker] = (sm, lane)
    ordered = sorted(set(lane_of.values()))
    rows = {lane: index for index, lane in enumerate(ordered)}
    return {worker: rows[lane] for worker, lane in lane_of.items()}, [
        sm for sm, _lane in ordered
    ]


def _plot_overlap(ax: Axes, timeline: Timeline, chosen: Launch) -> None:
    """A bar per root: thin from its first task's start to its first task's
    ready time, thick from there to its last task's end."""
    roots = sorted({task.root for task in chosen.tasks})
    for y, root in enumerate(roots):
        tasks = [task for task in chosen.tasks if task.root == root]
        start = min(task.start for task in tasks)
        ready = min(task.ready for task in tasks)
        color, hatch = _style(root)
        ax.broken_barh(
            [(start, ready - start)],
            (y - 0.06, 0.12),
            facecolors=color,
            alpha=0.6,
            linewidth=0,
        )
        ax.broken_barh(
            [(ready, max(task.end for task in tasks) - ready)],
            (y - 0.2, 0.4),
            facecolors=color,
            edgecolor="#fcfcfb",
            linewidth=0.3,
            hatch=hatch,
        )
    ax.set_yticks(range(len(roots)))
    ax.set_yticklabels([timeline.roots[root][1] for root in roots], fontsize=7)
    ax.set_ylim(len(roots) - 0.5, -0.5)
    ax.set_title("Root windows · thin: waiting, thick: working", loc="left", fontsize=9)


def _sm_ticks(row_sm: list[int]) -> list[int]:
    """The rows to label, about 20: each SM's first lane, for the first SM and
    every SM a multiple of a power-of-two step, and the last SM unless it would
    crowd the label before it."""
    step = 1 << max(0, round(math.log2(max(1, len(set(row_sm))) / 20)))
    firsts = [
        index
        for index, sm in enumerate(row_sm)
        if index == 0 or row_sm[index - 1] != sm
    ]
    ticks = [firsts[0]]
    ticks += [index for index in firsts[1:] if row_sm[index] % step == 0]
    # The last SM too, unless it would crowd the label before it.
    if row_sm[firsts[-1]] - row_sm[ticks[-1]] >= step / 2:
        ticks.append(firsts[-1])
    return ticks


def _plot_sm(ax: Axes, chosen: Launch) -> None:
    """A row per SM lane, a bar per task: pale while it waits, solid while it
    works."""
    row, row_sm = _sm_lanes(chosen)
    for task in chosen.tasks:
        color, hatch = _style(task.root)
        y = row[task.worker]
        ax.broken_barh(
            [(task.start, task.ready - task.start)],
            (y - 0.08, 0.16),
            facecolors=color,
            alpha=0.6,
            linewidth=0,
        )
        ax.broken_barh(
            [(task.ready, task.end - task.ready)],
            (y - 0.35, 0.7),
            facecolors=color,
            edgecolor="#fcfcfb",
            linewidth=0.2,
            hatch=hatch,
        )
    ticks = _sm_ticks(row_sm)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"SM {row_sm[index]}" for index in ticks], fontsize=7)
    ax.set_ylim(len(row_sm) - 0.5, -0.5)
    ax.set_title(
        "Physical SM lanes · concurrent tasks on one SM are stacked · "
        "thin: waiting, thick: working",
        loc="left",
        fontsize=9,
    )


def plot(timeline: Timeline, output: str, launch: int = -1, view: str = "both") -> None:
    """Draw one launch: ``view`` "overlap" gives each root's window (waiting
    until its first task is ready, then working until its last task ends), "sm"
    a row per SM lane with a bar per task, and "both" the two panels on one
    time axis."""
    # A bare Figure draws without pyplot, leaving the process's backend alone.
    import matplotlib.figure  # pyrefly: ignore [missing-import]
    import matplotlib.patches  # pyrefly: ignore [missing-import]

    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
    chosen = timeline.launches[launch]
    roots = len({task.root for task in chosen.tasks})
    window_height = max(1.2, 0.18 * roots)
    heights = {
        "overlap": [window_height],
        "sm": [8.0],
        "both": [window_height, max(4.0, 10.4 - window_height)],
    }[view]
    fig = matplotlib.figure.Figure(figsize=(24, sum(heights) + 1.6))
    axes = fig.subplots(
        len(heights), 1, sharex=True, gridspec_kw={"height_ratios": heights}
    )
    axes = list(axes) if len(heights) > 1 else [axes]
    names = ("overlap", "sm") if view == "both" else (view,)
    panels = dict(zip(names, axes, strict=True))
    if "overlap" in panels:
        _plot_overlap(panels["overlap"], timeline, chosen)
    if "sm" in panels:
        _plot_sm(panels["sm"], chosen)
    for ax in axes:
        ax.set_xlim(0, chosen.span)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#bdbcb8")
    axes[-1].set_xlabel("Time since the launch's first worker entry (us)", fontsize=8)
    if "overlap" not in panels:
        axes[0].legend(
            handles=[
                matplotlib.patches.Patch(
                    facecolor=_style(root)[0], hatch=_style(root)[1], label=name
                )
                for root, (_count, name) in timeline.roots.items()
            ],
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=7,
            frameon=False,
        )
    # Room left of the panels for the longest label (7 pt, about 0.055 in a
    # character), and the header lines a fixed distance from the top.
    labels = [name for _count, name in timeline.roots.values()]
    longest = max(map(len, labels)) if "overlap" in panels else len("SM 000")
    left = min(0.4, (0.055 * longest + 0.3) / fig.get_figwidth())
    height = fig.get_figheight()
    fig.suptitle(
        f"{timeline.kernel} · "
        + {
            "both": "megakernel timeline",
            "overlap": "root windows",
            "sm": "SM-level Gantt",
        }[view],
        x=left,
        y=1 - 0.15 / height,
        ha="left",
        va="top",
        fontsize=14,
    )
    fig.text(
        left,
        1 - 0.5 / height,
        f"rank {timeline.rank} · {timeline.dispatch} dispatch · "
        f"{timeline.workers} workers · {len(chosen.tasks)} tasks in {roots} "
        f"roots · launch {chosen.epoch} · {chosen.span:.3f} us span",
        fontsize=9,
        color="#3a3936",
    )
    fig.subplots_adjust(
        left=left,
        right=0.86 if "overlap" not in panels else 0.98,
        top=1 - 0.85 / height,
        bottom=0.5 / height,
        hspace=0.12,
    )
    fig.savefig(output, dpi=150, facecolor="white")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m helion.runtime.timeline")
    commands = parser.add_subparsers(dest="command", required=True)
    plot_parser = commands.add_parser("plot", help="draw one launch as a Gantt chart")
    plot_parser.add_argument("timeline")
    plot_parser.add_argument("-o", "--output", help="image path (default: .png)")
    plot_parser.add_argument(
        "--launch", type=int, default=-1, help="launch index (default: the last)"
    )
    plot_parser.add_argument(
        "--view",
        choices=VIEWS,
        default="both",
        help="overlap: each root's window; sm: a row per SM lane; both "
        "(default): the two panels on one time axis",
    )
    args = parser.parse_args(argv)
    output = args.output or os.path.splitext(args.timeline)[0] + ".png"
    plot(load(args.timeline), output, args.launch, args.view)
    print(output)


if __name__ == "__main__":
    main()
