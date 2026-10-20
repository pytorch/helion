from __future__ import annotations

from contextlib import contextmanager
import gc
import unittest
from unittest.mock import patch

from helion.runtime.cute import launcher


class TestCuteCaptureGCScope(unittest.TestCase):
    def setUp(self) -> None:
        self.enabled = gc.isenabled()
        gc.enable()

    def tearDown(self) -> None:
        if self.enabled:
            gc.enable()
        else:
            gc.disable()

    @contextmanager
    def fake_capture(self, graph: object, **kwargs: object):
        self.assertFalse(gc.isenabled())
        yield graph

    def run_capture(self, *, fail_body: bool = False) -> None:
        with (
            patch.object(launcher, "_CuteCUDAGraph", return_value=object()),
            patch.object(launcher.torch.cuda, "graph", self.fake_capture),
            launcher.cute_cuda_graph() as graph,
        ):
            self.assertFalse(gc.isenabled())
            self.assertIs(launcher._CUTE_ACTIVE_CUDA_GRAPH.get(), graph)
            if fail_body:
                raise RuntimeError("capture body failed")
        self.assertIsNone(launcher._CUTE_ACTIVE_CUDA_GRAPH.get())

    def test_enabled_gc_is_restored(self) -> None:
        self.run_capture()
        self.assertTrue(gc.isenabled())

    def test_disabled_gc_stays_disabled(self) -> None:
        gc.disable()
        self.run_capture()
        self.assertFalse(gc.isenabled())

    def test_body_failure_restores_gc_and_context(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "capture body failed"):
            self.run_capture(fail_body=True)
        self.assertTrue(gc.isenabled())
        self.assertIsNone(launcher._CUTE_ACTIVE_CUDA_GRAPH.get())

    def test_entry_failure_restores_gc_and_context(self) -> None:
        with (
            patch.object(launcher, "_CuteCUDAGraph", return_value=object()),
            patch.object(
                launcher.torch.cuda, "graph", side_effect=RuntimeError("entry failed")
            ),
            self.assertRaisesRegex(RuntimeError, "entry failed"),
            launcher.cute_cuda_graph(),
        ):
            raise AssertionError("entry should fail")
        self.assertTrue(gc.isenabled())
        self.assertIsNone(launcher._CUTE_ACTIVE_CUDA_GRAPH.get())


if __name__ == "__main__":
    unittest.main()
