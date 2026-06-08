from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import types
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
        # Same source, different code-generation settings -> different kernel_id.
        _probe_static = helion.kernel(
            _probe_kernel, config=helion.Config(block_sizes=[16]), static_shapes=True
        )
        _probe_dynamic = helion.kernel(
            _probe_kernel, config=helion.Config(block_sizes=[16]), static_shapes=False
        )
        self.assertEqual(_probe_static.kernel_source(), _probe_dynamic.kernel_source())
        self.assertNotEqual(_probe_static.kernel_id(), _probe_dynamic.kernel_id())

    def test_kernel_id_independent_of_config(self) -> None:
        """
        kernel_id must NOT depend on the (autotunable) config: the same source
        and settings under different configs share one id, so it stays constant
        across the configs benchmarked in a run and can group their rows.
        """
        k16 = helion.kernel(_probe_kernel, config=helion.Config(block_sizes=[16]))
        k32 = helion.kernel(_probe_kernel, config=helion.Config(block_sizes=[32]))
        self.assertEqual(k16.kernel_id(), k32.kernel_id())

    def test_dynamic_source_raises_oserror(self) -> None:
        """
        kernel_source/kernel_id raise OSError (and nothing broader) when the
        source cannot be located, e.g. a function defined dynamically. This is
        the contract the narrowed ``except OSError`` in the autotuner relies on.
        """
        namespace: dict[str, object] = {}
        exec(
            compile("def _dynamic(x, y):\n    return x\n", "<dynamic>", "exec"),
            namespace,
        )
        fn = namespace["_dynamic"]
        assert isinstance(fn, types.FunctionType)
        dynamic = helion.kernel(fn, config=helion.Config(block_sizes=[16]))
        with self.assertRaises(OSError):
            dynamic.kernel_source()
        with self.assertRaises(OSError):
            dynamic.kernel_id()


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

    def test_default_metadata_has_empty_identity(self) -> None:
        """
        Default KernelMetadata carries an empty identity and still serializes,
        so a sidecar can be written even when the kernel is unidentifiable.
        """
        record = KernelMetadata().to_dict()
        self.assertEqual(record["kernel_id"], "")
        self.assertEqual(record["kernel_source"], "")
        json.dumps(record)

    def test_log_entry_defaults_to_empty_sample_id(self) -> None:
        """
        sample_id is optional on AutotuneLogEntry and defaults to empty.
        """
        entry = AutotuneLogEntry(
            generation=0,
            status="ok",
            perf_ms=1.0,
            compile_time=0.1,
            config=helion.Config(block_sizes=[16]),
        )
        self.assertEqual(entry.sample_id, "")


class TestAutotuneLogSink(TestCase):
    def _entry(self, perf_ms: float, sample_id: str = "sample-xyz") -> AutotuneLogEntry:
        """
        Build a minimal AutotuneLogEntry for sink tests.

        Only perf_ms and sample_id vary across callers; the remaining fields
        (generation, status, compile_time, config) are fixed placeholders.
        """
        return AutotuneLogEntry(
            generation=0,
            status="ok",
            perf_ms=perf_ms,
            compile_time=0.5,
            config=helion.Config(block_sizes=[16]),
            sample_id=sample_id,
        )

    def test_sink_writes_metadata_sidecar_and_per_config_rows(self) -> None:
        """
        With KernelMetadata, the sink writes both outputs over a run: a JSON
        sidecar holding the kernel identity once, and a CSV with one row per
        recorded config. Each CSV row is stamped with the kernel_id foreign key
        (matching the sidecar) and carries the entry's sample_id.
        """
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
            self.assertEqual(sidecar["kernel_id"], "abc123")
            self.assertEqual(sidecar["kernel_name"], "_add_kernel")
            self.assertIn("def _add_kernel", sidecar["kernel_source"])

            # Per-config CSV holds one row per benchmarked config + its result.
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            header = rows[0]
            self.assertEqual(header[0], "kernel_id")
            self.assertIn("timestamp_s", header)
            data_rows = rows[1:]
            self.assertEqual(len(data_rows), 2)
            # kernel_id foreign key is stamped on every row, matching the sidecar.
            kid_col = header.index("kernel_id")
            self.assertTrue(all(row[kid_col] == "abc123" for row in data_rows))
            # sample_id (per-(kernel, config) key) is carried from the entry.
            sid_col = header.index("sample_id")
            self.assertTrue(all(row[sid_col] == "sample-xyz" for row in data_rows))

    def test_sink_without_metadata_writes_no_sidecar(self) -> None:
        """
        When the sink is created without KernelMetadata, a full run (start,
        record, end) writes no sidecar file, since there is no kernel identity
        to persist.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = f"{tmp}/run"
            with AutotuneLogSink(base) as sink:
                sink.start_run()
                sink.record(self._entry(0.1))
                sink.end_run()
            self.assertFalse(sink.meta_path.exists())

    def test_sink_without_metadata_rows_have_empty_kernel_id(self) -> None:
        """
        Without metadata the CSV still has the kernel_id column, but the
        foreign key is empty on every row (no kernel identity to attach).
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = f"{tmp}/run"
            with AutotuneLogSink(base) as sink:
                sink.start_run()
                sink.record(self._entry(0.1))
                sink.end_run()
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            header = rows[0]
            self.assertEqual(header[0], "kernel_id")
            kid_col = header.index("kernel_id")
            self.assertTrue(all(row[kid_col] == "" for row in rows[1:]))

    def test_record_before_open_is_noop(self) -> None:
        """
        Recording on a sink that was never opened writes nothing and does not
        create the CSV file (guards against lifecycle ordering bugs).
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = f"{tmp}/run"
            sink = AutotuneLogSink(base)  # not opened
            sink.record(self._entry(0.1))
            self.assertFalse(sink.csv_path.exists())


if __name__ == "__main__":
    unittest.main()
