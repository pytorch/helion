from __future__ import annotations

import csv
import hashlib
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
    """
    A kernel that adds two tensors.
    """
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(config=helion.Config(block_sizes=[16]))
def _other_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    A kernel that multiplies two tensors.
    """
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * y[tile]
    return out


def _probe_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    A bare kernel function used to build kernels with identical source but
    different settings.
    """
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


# Same source, different code-generation settings -> different kernel_id.
_probe_static = helion.kernel(
    _probe_kernel, config=helion.Config(block_sizes=[16]), static_shapes=True
)
_probe_dynamic = helion.kernel(
    _probe_kernel, config=helion.Config(block_sizes=[16]), static_shapes=False
)


class TestKernelIdentity(TestCase):
    def test_kernel_source_stable_and_distinct(self) -> None:
        """
        Check if the kernel source is stable and distinct between kernels.
        """
        first = _add_kernel.kernel_source()
        second = _add_kernel.kernel_source()
        self.assertEqual(first, second)
        self.assertIn("def _add_kernel", first)
        self.assertNotEqual(first, _other_kernel.kernel_source())

    def test_kernel_id_is_stable_hex_hash(self) -> None:
        """
        kernel_id is a stable 64-character sha256 hex digest.
        """
        first = _add_kernel.kernel_id()
        second = _add_kernel.kernel_id()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)  # raises ValueError if not valid hex

    def test_kernel_id_distinct_between_kernels(self) -> None:
        """
        Different kernel source yields a different kernel_id.
        """
        self.assertNotEqual(_add_kernel.kernel_id(), _other_kernel.kernel_id())

    def test_kernel_id_matches_manual_hash(self) -> None:
        """
        kernel_id equals sha256(source + settings signature).
        """
        payload = _add_kernel.kernel_source() + _add_kernel._settings_signature()
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.assertEqual(_add_kernel.kernel_id(), expected)

    def test_kernel_id_changes_with_settings(self) -> None:
        """
        Identical source under different codegen settings yields different ids.
        """
        self.assertEqual(_probe_static.kernel_source(), _probe_dynamic.kernel_source())
        self.assertNotEqual(_probe_static.kernel_id(), _probe_dynamic.kernel_id())


class TestMetadataSchema(TestCase):
    def test_autotune_metrics_to_dict_has_kernel_fields(self) -> None:
        """
        Check if the AutotuneMetrics.to_dict() method includes the kernel fields.
        """
        record = AutotuneMetrics(
            kernel_id="abc123", kernel_name="k", kernel_source="def k(): ..."
        ).to_dict()
        self.assertEqual(record["kernel_id"], "abc123")
        self.assertEqual(record["kernel_name"], "k")
        self.assertEqual(record["kernel_source"], "def k(): ...")
        json.dumps(record)

    def test_kernel_metadata_to_dict_round_trip(self) -> None:
        """
        Check if the KernelMetadata.to_dict() is JSON serializable.
        """
        record = KernelMetadata(
            kernel_id="abc123",
            kernel_name="k",
            kernel_source="def k(): ...",
            input_shapes="[(16,)]",
            dtypes="['torch.float32']",
            hardware="TestGPU",
        ).to_dict()
        self.assertEqual(record["kernel_id"], "abc123")
        self.assertEqual(record["kernel_source"], "def k(): ...")
        # Must be JSON serializable for the sidecar file.
        json.dumps(record)


class TestAutotuneLogSink(TestCase):
    def _entry(self, perf_ms: float) -> AutotuneLogEntry:
        """
        Create fake log entry for testing.
        """
        return AutotuneLogEntry(
            generation=0,
            status="ok",
            perf_ms=perf_ms,
            compile_time=0.5,
            config=helion.Config(block_sizes=[16]),
        )

    def test_sink_writes_metadata_sidecar_and_per_config_rows(self) -> None:
        metadata = KernelMetadata(
            kernel_id="abc123",
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
