from __future__ import annotations

import os
import pathlib
import tempfile
from unittest import mock

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cross_loop_timeline import EPOCHS
from helion._compiler.cross_loop_timeline import TASK_WORDS
from helion._compiler.cross_loop_timeline import WORKER_WORDS
from helion._compiler.cross_loop_timeline import TimelineMeta
from helion._testing import TestCase
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.finite_search import FiniteSearch
import helion.language as hl
from helion.runtime import timeline
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only

_SECTION = 2 * WORKER_WORDS + 3 * TASK_WORDS
_META: TimelineMeta = {
    "prefix": "unused",
    "kernel": "k",
    "workers": 2,
    "dispatch": "static",
    "section": _SECTION,
    "roots": [
        (0, 6, 2, "root 0 · up  projection [lines:3-5]"),
        (1, 14, 1, "root 1 [lines:7-9]"),
    ],
}


def _add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


class _Stop(Exception):
    pass


# A cross_loop_pipeline fragment, and whether it offers a megakernel: none (no
# cross-loop dependencies), root barriers only (CuTe), and a pipeline.
_PIPELINES = (
    (None, False),
    (EnumFragment(("barrier",)), False),
    (EnumFragment(("barrier", "static")), True),
)


def _buffer() -> torch.Tensor:
    """Launch 1, overwritten by launch 17 (complete, but root 1's task record is
    an older launch's); launch 2, overwritten by launch 18, which worker 1 has
    not finished (its exit word still launch 2's); launch 19, which worker 1
    has not finished (no exit word)."""
    words = torch.zeros(EPOCHS * _SECTION, dtype=torch.int64)

    def launch(epoch: int, t0: int, done: bool) -> torch.Tensor:
        section = words[epoch % EPOCHS * _SECTION :][:_SECTION]
        for worker, sm in enumerate((5, 9)):
            section[worker * 3 : worker * 3 + 2] = torch.tensor(
                [(epoch << 32) | sm, t0 + worker * 64]
            )
            if done or worker == 0:
                section[worker * 3 + 2] = t0 + 9000
        # Root 0's task 0 waits 1 us; its task 1 has no waits.
        tasks = [(0, 1000, 2000, 4000), (1, 1000, 0, 3000), (0, 5000, 5000, 8000)]
        for index, (worker, start, ready, end) in enumerate(tasks):
            offset = 6 + index * TASK_WORDS
            section[offset : offset + TASK_WORDS] = torch.tensor(
                [
                    (epoch << 32) | (worker + 1),
                    t0 + start,
                    t0 + ready if ready else 0,
                    t0 + end,
                ]
            )
        return section

    launch(1, 10**12, True)
    launch(2, 10**12, True)
    launch(17, 2 * 10**12, True)[14] = (1 << 32) | 1
    launch(18, 3 * 10**12, False)
    launch(19, 3 * 10**12, False)
    return words


class TestMegakernelTimeline(TestCase):
    def test_decode_keeps_complete_launches_of_this_epoch(self) -> None:
        decoded = timeline.decode(_buffer(), _META)
        # 18 and 19 are torn (a worker has not exited).
        self.assertEqual([launch.epoch for launch in decoded.launches], [17])
        self.assertEqual(decoded.launches_total, 19)
        (launch,) = decoded.launches
        self.assertEqual(launch.workers, {0: (5, 0.0, 9.0), 1: (9, 0.064, 9.0)})
        self.assertEqual(
            [(t.root, t.task, t.worker, t.start, t.ready, t.end) for t in launch.tasks],
            # No wait stamp: ready is the start.
            [(0, 0, 0, 1.0, 2.0, 4.0), (0, 1, 1, 1.0, 1.0, 3.0)],
        )

    def test_text_round_trips(self) -> None:
        decoded = timeline.decode(_buffer(), _META)
        text = timeline.format_timeline(decoded)
        self.assertIn("summary: span median 9.00", text)
        # Per root: tasks, first start, last end, idle (since the worker's
        # entry: 1 and 0.936 us), wait and work median/max.
        self.assertIn(
            "0     2      1.00  4.00  0.97/1.00  0.50/1.00  2.00/2.00  root 0", text
        )
        self.assertIn("1     1      -  -  -  -  -  root 1", text)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "k.txt")
            pathlib.Path(path).write_text(text)
            self.assertEqual(timeline.load(path), decoded)

    def test_plot_writes_an_image(self) -> None:
        pytest.importorskip("matplotlib")
        decoded = timeline.decode(_buffer(), _META)
        with tempfile.TemporaryDirectory() as directory:
            for view in timeline.VIEWS:
                path = os.path.join(directory, f"k.{view}.png")
                timeline.plot(decoded, path, view=view)
                self.assertGreater(os.path.getsize(path), 0)
            with self.assertRaises(ValueError):
                timeline.plot(decoded, os.path.join(directory, "k.png"), view="x")

    def test_sm_lanes(self) -> None:
        # Workers resident on one SM together take its lanes; a worker that
        # starts after another left reuses its lane.
        launch = timeline.Launch(
            1,
            0,
            {0: (3, 0.0, 5.0), 1: (3, 1.0, 4.0), 2: (3, 6.0, 9.0), 3: (0, 0.0, 9.0)},
            [],
        )
        self.assertEqual(
            timeline._sm_lanes(launch), ({3: 0, 0: 1, 2: 1, 1: 2}, [0, 3, 3])
        )

    def test_sm_labels(self) -> None:
        # Every eighth SM, and the last unless it crowds the label before it.
        for count, last in ((148, []), (152, [151])):
            rows = [sm for sm in range(count) for _lane in range(2)]
            labels = [rows[row] for row in timeline._sm_ticks(rows)]
            self.assertEqual(labels, [*range(0, count, 8), *last])
        # Only odd SMs in use: the first still gets a label.
        odd = list(range(1, 80, 2))
        self.assertEqual(timeline._sm_ticks(odd)[0], 0)

    def test_helion_logs_turns_it_on(self) -> None:
        with mock.patch.dict(os.environ, {"HELION_LOGS": "megakernel"}):
            self.assertIsNone(Settings().megakernel_timeline)
        with mock.patch.dict(os.environ, {"HELION_LOGS": "megakernel, timeline"}):
            self.assertEqual(
                Settings().megakernel_timeline,
                os.path.abspath(os.path.join("helion_timeline", "timeline")),
            )

    def test_autotuning_a_megakernel_raises(self) -> None:
        """Stamps change a megakernel's timings, so autotuning a kernel that
        some config compiles as a megakernel raises under the setting. A
        kernel without cross-loop dependencies, or whose backend lowers only
        root barriers, autotunes as usual."""
        kernel = helion.kernel(_add_one, megakernel_timeline="unused")
        args = (torch.empty(8),)
        with cpu_render_target():
            bound = cpu_bind(kernel, args)
        configs = [helion.Config(block_sizes=[8]), helion.Config(block_sizes=[4])]
        for pipeline, megakernel in _PIPELINES:
            search = FiniteSearch(bound, args, configs)
            # (No candidate runs: the search stops where it would set them up.)
            search._benchmark_provider_cls = mock.Mock(side_effect=_Stop)
            error = exc.AutotuningDisallowedInEnvironment if megakernel else _Stop
            with mock.patch.object(search.config_spec, "cross_loop_pipeline", pipeline):
                # Raising leaves the search unprepared: a retry raises again.
                for _attempt in range(2 if megakernel else 1):
                    with self.assertRaises(error):
                        search.autotune()
