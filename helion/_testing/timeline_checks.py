"""Checks of recorded megakernel timelines shared by the GPU tests of
``test/test_cross_loop_codegen.py`` and ``test/test_gluon_ring.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from helion.runtime import timeline

if TYPE_CHECKING:
    import unittest


def check_written(
    test: unittest.TestCase, launches: int
) -> list[tuple[str, timeline.Timeline]]:
    """Write every registered timeline and check each file: its last
    ``launches`` launches kept (epochs 1, 2, ...), every task recorded in each,
    every record in order inside its worker's window.  Returns the files and
    their timelines, sorted by path."""
    written = []
    for path in sorted(timeline.write()):
        recorded = timeline.load(path)
        test.assertEqual(
            [launch.epoch for launch in recorded.launches],
            list(range(1, launches + 1)),
        )
        tasks = sum(count for count, _name in recorded.roots.values())
        for launch in recorded.launches:
            test.assertEqual(len(launch.tasks), tasks)
            for task in launch.tasks:
                _sm, entry, leave = launch.workers[task.worker]
                test.assertTrue(
                    entry <= task.start <= task.ready <= task.end <= leave, task
                )
        written.append((path, recorded))
    return written
