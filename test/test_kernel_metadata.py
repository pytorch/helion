from __future__ import annotations

import csv
import json
import tempfile
import unittest

import torch

import helion
from helion._testing import TestCase
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuneLogSink
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.metrics import KernelMetadata
import helion.language as hl


@helion.kernel(config=helion.Config(block_sizes=[16]))
def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(config=helion.Config(block_sizes=[16]))
def _other_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * y[tile]
    return out


def _side_table_available() -> bool:
    try:
        from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table
    except Exception:
        return False
    return helion_kernel_side_table is not None


class TestKernelIdentity(TestCase):
    def test_kernel_source_stable_and_distinct(self) -> None:
        first = _add_kernel.kernel_source()
        second = _add_kernel.kernel_source()
        self.assertEqual(first, second)
        self.assertIn("def _add_kernel", first)
        self.assertNotEqual(first, _other_kernel.kernel_source())

    @unittest.skipUnless(_side_table_available(), "kernel side table unavailable")
    def test_kernel_id_idempotent(self) -> None:
        first = _add_kernel.kernel_id()
        second = _add_kernel.kernel_id()
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)

    @unittest.skipUnless(_side_table_available(), "kernel side table unavailable")
    def test_kernel_id_distinct_between_kernels(self) -> None:
        self.assertNotEqual(_add_kernel.kernel_id(), _other_kernel.kernel_id())

    @unittest.skipUnless(_side_table_available(), "kernel side table unavailable")
    def test_kernel_id_matches_side_table(self) -> None:
        from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table

        idx = _add_kernel.kernel_id()
        self.assertIs(helion_kernel_side_table.get_kernel(idx), _add_kernel)


class TestMetadataSchema(TestCase):
    def test_autotune_metrics_to_dict_has_kernel_fields(self) -> None:
        record = AutotuneMetrics(
            kernel_idx=7, kernel_name="k", kernel_source="def k(): ..."
        ).to_dict()
        self.assertEqual(record["kernel_idx"], 7)
        self.assertEqual(record["kernel_name"], "k")
        self.assertEqual(record["kernel_source"], "def k(): ...")
        json.dumps(record)

    def test_kernel_metadata_to_dict_round_trip(self) -> None:
        record = KernelMetadata(
            kernel_idx=3,
            kernel_name="k",
            kernel_source="def k(): ...",
            input_shapes="[(16,)]",
            dtypes="['torch.float32']",
            hardware="TestGPU",
        ).to_dict()
        self.assertEqual(record["kernel_idx"], 3)
        self.assertEqual(record["kernel_source"], "def k(): ...")
        # Must be JSON serializable for the sidecar file.
        json.dumps(record)


class TestAutotuneLogSink(TestCase):
    def _entry(self, perf_ms: float) -> AutotuneLogEntry:
        return AutotuneLogEntry(
            generation=0,
            status="ok",
            perf_ms=perf_ms,
            compile_time=0.5,
            config=helion.Config(block_sizes=[16]),
        )

    def test_sink_writes_metadata_sidecar_and_per_config_rows(self) -> None:
        metadata = KernelMetadata(
            kernel_idx=1,
            kernel_name="_add_kernel",
            kernel_source=_add_kernel.kernel_source(),
            input_shapes="[(64,)]",
            dtypes="['torch.float32']",
            hardware="TestGPU",
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = f"{tmp}/run"
            with AutotuneLogSink(base, metadata) as sink:
                sink.start_run()
                sink.record(self._entry(0.1))
                sink.record(self._entry(0.2))
                sink.end_run()

            # Sidecar holds the kernel identity (stored once).
            sidecar = json.loads(sink.meta_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["kernel_name"], "_add_kernel")
            self.assertIn("def _add_kernel", sidecar["kernel_source"])

            # Per-config CSV holds one row per benchmarked config + its result.
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0][:1], ["timestamp_s"])
            data_rows = rows[1:]
            self.assertEqual(len(data_rows), 2)

    def test_sink_without_metadata_writes_no_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = f"{tmp}/run"
            with AutotuneLogSink(base) as sink:
                sink.start_run()
                sink.record(self._entry(0.1))
                sink.end_run()
            self.assertFalse(sink.meta_path.exists())


if __name__ == "__main__":
    unittest.main()
