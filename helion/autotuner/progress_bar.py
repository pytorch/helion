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
from rich.progress import TextColumn

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

T = TypeVar("T")


def iter_with_progress(
    iterable: Iterable[T], *, total: int, description: str | None, enabled: bool
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

    if not enabled:
        yield from iterable
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(bar_width=None, complete_style="yellow", finished_style="green"),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task(description or "Progress", total=total)
        for item in iterable:
            # Yield before updating so the consumer sees the item immediately.
            yield item
            progress.update(task, advance=1)
        progress.refresh()
