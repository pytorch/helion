from __future__ import annotations

from unittest import TestCase

from helion._compiler.pallas.dma import DmaResources


class TestDmaResources(TestCase):
    def test_single_buffer_refs(self) -> None:
        resources = DmaResources("scratch", "semaphore", 1)

        self.assertEqual(resources.scratch_ref(None), "scratch")
        self.assertEqual(resources.semaphore_ref(None), "semaphore")

    def test_double_buffer_refs(self) -> None:
        resources = DmaResources("scratch", "semaphore", 2)

        self.assertEqual(resources.scratch_ref("stage"), "scratch.at[stage]")
        self.assertEqual(resources.semaphore_ref("stage"), "semaphore.at[stage]")
