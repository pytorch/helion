"""Progress-bar utilities used by the autotuner.

We rely on `rich` to render colored, full-width progress bars that
show the description, percentage complete, and how many items have been
processed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import TextColumn
from rich.text import Text
import torch

from helion._dist_utils import is_master_rank

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

    from rich.progress import Task

T = TypeVar("T")


class SpeedColumn(ProgressColumn):
    """Render the processing speed in configs per second."""

    def render(self, task: Task) -> Text:
        # Fixed-width placeholder before first sample to avoid bar jitter.
        speed = f"{task.speed:.1f}" if task.speed is not None else "..."
        return Text(f"{speed:>5} configs/s", style="bold blue")


def iter_with_progress(
    iterable: Iterable[T], *, total: int, description: str | None = None, enabled: bool
) -> Iterator[T]:
    """Yield items from *iterable*, optionally showing a progress bar.

    Parameters
    ----------
    iterable:
        Any iterable whose items should be yielded.
    total:
        Total number of items expected from the iterable.
    description:
        Text displayed on the left side of the bar.  Defaults to ``"Progress"``.
    enabled:
        When ``False`` the iterable is returned unchanged so there is zero
        overhead; when ``True`` a Rich progress bar is rendered on **stderr**
        (so stdout lines from other libraries stay readable). When Rich detects
        a non-interactive, non-Jupyter console, the bar is skipped entirely.
    """
    if (not enabled) or torch._utils_internal.is_fb_unit_test() or not is_master_rank():
        yield from iterable
        return

    if description is None:
        description = "Progress"

    # Render on stderr so third-party stdout prints (e.g. torch_npu profiler
    # ``print(..., flush=True)``) do not corrupt Rich Live cursor control and
    # produce one-character-per-line spinner junk (| / - \\) on stdout.
    progress_console = Console(stderr=True)
    if not progress_console.is_interactive and not progress_console.is_jupyter:
        yield from iterable
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[bold blue]{task.percentage:>3.0f}%"),
        BarColumn(bar_width=None, complete_style="yellow", finished_style="green"),
        MofNCompleteColumn(),
        SpeedColumn(),
        console=progress_console,
    ) as progress:
        yield from progress.track(iterable, total=total, description=description)
