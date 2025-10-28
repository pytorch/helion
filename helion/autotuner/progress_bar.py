"""Progress-bar utilities used by the autotuner.

We rely on `rich` to render colored, full-width progress bars that
show the description, percentage complete, and how many items have been
processed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import TextColumn
from rich.text import Text
import torch

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

    from rich.progress import Task

T = TypeVar("T")


class SpeedColumn(ProgressColumn):
    """Render the processing speed in configs per second."""

    def render(self, task: Task) -> Text:
        return Text(
            f"{task.speed:.1f} configs/s" if task.speed is not None else "- configs/s",
            style="magenta",
        )


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
        overhead; when ``True`` a Rich progress bar is rendered.
    """
    if (not enabled) or torch._utils_internal.is_fb_unit_test():  # pyright: ignore[reportAttributeAccessIssue]
        yield from iterable
        return

    if description is None:
        description = "Progress"

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(bar_width=None, complete_style="yellow", finished_style="green"),
        MofNCompleteColumn(),
        SpeedColumn(),
    ) as progress:
        yield from progress.track(iterable, total=total, description=description)
