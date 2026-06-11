from __future__ import annotations

import csv
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

# Acceptance suite for the lean kernel-artifact schema (.spec/PRD.md). It pins the
# two-file telemetry contract:
#   .meta.jsonl (per run): run_id, kernel_name, kernel_source, input_shapes,
#                          dtypes, hardware, settings, config_defaults
#   .csv (per entry):      run_id, timestamp_s, config_index, generation, status,
#                          perf_ms, compile_time_s, config
# and the invariants: no kernel_id/sample_id/decorator anywhere; run_id derived
# directly (no kernel_id intermediary); config stored as JSON; full settings
# serialized JSON-safe. Pure-assertion, no CUDA.

_LEAN_CSV_HEADER = [
    "run_id",
    "timestamp_s",
    "config_index",
    "generation",
    "status",
    "perf_ms",
    "compile_time_s",
    "config",
]

_META_KEYS = {
    "run_id",
    "kernel_name",
    "kernel_source",
    "input_shapes",
    "dtypes",
    "hardware",
    "settings",
    "config_defaults",
}


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


class TestKernelSource(TestCase):
    def test_kernel_source_stable_and_distinct(self) -> None:
        """
        Check if the kernel source is stable and distinct between kernels.
        """
        first = _add_kernel.kernel_source()
        second = _add_kernel.kernel_source()
        self.assertEqual(first, second)
        self.assertIn("def _add_kernel", first)
        self.assertNotEqual(first, _other_kernel.kernel_source())

    def test_dynamic_source_raises_oserror(self) -> None:
        """
        kernel_source raises OSError (and nothing broader) when the source
        cannot be located, e.g. a function defined dynamically. This is the
        contract the narrowed ``except OSError`` in the autotuner relies on.
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


class TestMetadataSchema(TestCase):
    def _metadata(self, **overrides: object) -> KernelMetadata:
        fields: dict[str, object] = {
            "kernel_name": "k",
            "kernel_source": "def k(): ...",
            "input_shapes": "[(16,)]",
            "dtypes": "['torch.float32']",
            "hardware": "TestGPU",
            "settings": {"static_shapes": True, "index_dtype": None},
            "config_defaults": {"block_sizes": [1], "num_warps": 4},
        }
        fields.update(overrides)
        return KernelMetadata(**fields)

    def test_to_dict_has_exactly_the_lean_keys(self) -> None:
        """
        KernelMetadata.to_dict() carries exactly the lean per-run fields and, in
        particular, NO kernel_id (it is removed from the collected schema).
        """
        record = self._metadata().to_dict()
        self.assertEqual(set(record), _META_KEYS)
        self.assertNotIn("kernel_id", record)

    def test_settings_are_stored_and_json_safe(self) -> None:
        """
        The full helion.settings are stored under ``settings`` and the record
        serializes JSON-safe even when settings carry non-serializable values
        (callables, torch.dtype) via ``default=str`` (FR3/NFR2).
        """
        record = self._metadata(
            settings={
                "static_shapes": True,
                "index_dtype": torch.int32,
                "autotune_baseline_fn": (lambda *a: None),
            }
        ).to_dict()
        self.assertIn("settings", record)
        self.assertEqual(record["settings"]["static_shapes"], True)
        # Must not raise: non-serializable leaves are stringified.
        line = json.dumps(record, default=str)
        reparsed = json.loads(line)
        self.assertEqual(reparsed["settings"]["index_dtype"], "torch.int32")

    def test_real_settings_to_dict_serializes(self) -> None:
        """
        A real helion.Settings().to_dict() round-trips through the JSON-safe dump.
        """
        record = self._metadata(settings=helion.Settings().to_dict()).to_dict()
        json.dumps(record, default=str)

    def test_run_id_derived_directly_and_distinguishes_identity(self) -> None:
        """
        run_id is derived directly (no kernel_id intermediary) from
        (kernel_source, codegen-settings-signature, input_shapes, dtypes,
        hardware): derived when not supplied, stable for identical invocations,
        and distinct across every identity component (FR7).
        """
        base = self._metadata()
        self.assertTrue(base.run_id)  # derived, non-empty
        self.assertEqual(len(base.run_id), 64)  # sha256 hex
        self.assertEqual(base.run_id, self._metadata().run_id)  # reproducible
        self.assertNotEqual(
            base.run_id, self._metadata(kernel_source="def z(): 1").run_id
        )
        self.assertNotEqual(base.run_id, self._metadata(input_shapes="[(32,)]").run_id)
        self.assertNotEqual(
            base.run_id, self._metadata(dtypes="['torch.float16']").run_id
        )
        self.assertNotEqual(base.run_id, self._metadata(hardware="OtherGPU").run_id)
        # Codegen-affecting settings change the run_id (different generated code).
        self.assertNotEqual(
            base.run_id,
            self._metadata(
                settings={"static_shapes": False, "index_dtype": None}
            ).run_id,
        )
        # An explicitly supplied run_id is preserved (not overwritten).
        self.assertEqual(self._metadata(run_id="pinned").run_id, "pinned")

    def test_default_metadata_serializes_with_empty_identity(self) -> None:
        """
        Default KernelMetadata carries an empty identity, no kernel_id, and still
        serializes, so a sidecar can be written even when the kernel is
        unidentifiable.
        """
        record = KernelMetadata().to_dict()
        self.assertNotIn("kernel_id", record)
        self.assertEqual(record["kernel_source"], "")
        json.dumps(record, default=str)

    def test_autotune_metrics_to_dict_has_no_kernel_id(self) -> None:
        """
        AutotuneMetrics no longer carries kernel_id (removed everywhere).
        """
        record = AutotuneMetrics(
            kernel_name="k", kernel_source="def k(): ..."
        ).to_dict()
        self.assertNotIn("kernel_id", record)
        self.assertEqual(record["kernel_name"], "k")
        json.dumps(record, default=str)

    def test_config_defaults_reconstruct_minimized_row(self) -> None:
        """
        The per-run config_defaults plus a per-config minimized row reconstruct
        the full config as benchmarked: full = {**config_defaults, **row_config}.
        This is what makes the dataset self-contained despite minimized rows.
        """
        record = self._metadata(
            config_defaults={"block_sizes": [1], "num_warps": 4, "num_stages": 2}
        ).to_dict()
        # A minimized row drops keys equal to the default (num_warps), keeps the
        # required block_sizes and any non-default value (num_stages).
        row_config = {"block_sizes": [64], "num_stages": 3}
        full = {**record["config_defaults"], **row_config}
        self.assertEqual(full, {"block_sizes": [64], "num_warps": 4, "num_stages": 3})

    def test_run_id_signature_shares_decorator_source(self) -> None:
        """
        The run_id codegen signature is derived from the SAME helper that
        BoundKernel.format_kernel_decorator uses, so the two cannot drift. A
        parallel reimplementation would let run_id silently stop reflecting a
        newly codegen-affecting setting.
        """
        from helion.autotuner.metrics import _codegen_signature
        from helion.runtime.settings import codegen_decorator_parts

        settings = helion.Settings(static_shapes=False, index_dtype=torch.int32)
        expected = ", ".join(
            codegen_decorator_parts(settings.static_shapes, settings.index_dtype)
        )
        self.assertEqual(_codegen_signature(settings.to_dict()), expected)
        self.assertIn("static_shapes=False", expected)
        self.assertIn("index_dtype=torch.int32", expected)


class TestAutotuneLogEntry(TestCase):
    def test_entry_has_no_sample_id_or_decorator(self) -> None:
        """
        AutotuneLogEntry no longer carries sample_id or decorator fields.
        """
        entry = AutotuneLogEntry(
            generation=0,
            status="ok",
            perf_ms=1.0,
            compile_time=0.1,
            config=helion.Config(block_sizes=[16]),
        )
        self.assertFalse(hasattr(entry, "sample_id"))
        self.assertFalse(hasattr(entry, "decorator"))


class TestAutotuneLogSink(TestCase):
    def _entry(self, perf_ms: float, **config_kwargs: object) -> AutotuneLogEntry:
        """Build a minimal AutotuneLogEntry for sink tests."""
        config = (
            helion.Config(**config_kwargs)
            if config_kwargs
            else helion.Config(block_sizes=[16])
        )
        return AutotuneLogEntry(
            generation=0,
            status="ok",
            perf_ms=perf_ms,
            compile_time=0.5,
            config=config,
        )

    def _metadata(self, **overrides: object) -> KernelMetadata:
        fields: dict[str, object] = {
            "kernel_name": "_add_kernel",
            "kernel_source": _add_kernel.kernel_source(),
            "input_shapes": "[(64,)]",
            "dtypes": "['torch.float32']",
            "hardware": "TestGPU",
            "settings": {"static_shapes": True, "index_dtype": None},
            "config_defaults": {"block_sizes": [1], "num_warps": 4},
        }
        fields.update(overrides)
        return KernelMetadata(**fields)

    def test_csv_header_is_lean(self) -> None:
        """
        The CSV header is exactly the lean column set — no kernel_id, sample_id,
        or decorator (FR4/FR6).
        """
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(f"{tmp}/run", self._metadata()) as sink:
                sink.start_run()
                sink.record(self._entry(0.1))
                sink.end_run()
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                header = next(csv.reader(f))
        self.assertEqual(header, _LEAN_CSV_HEADER)
        for absent in ("kernel_id", "sample_id", "decorator"):
            self.assertNotIn(absent, header)

    def test_sidecar_has_lean_keys_and_settings(self) -> None:
        """
        The .meta.jsonl sidecar holds one record per run with exactly the lean
        keys (including ``settings``) and no kernel_id (FR3/FR6).
        """
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(f"{tmp}/run", self._metadata()) as sink:
                sink.start_run()
                sink.record(self._entry(0.1))
                sink.end_run()
            meta_lines = sink.meta_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(meta_lines), 1)
        sidecar = json.loads(meta_lines[0])
        self.assertEqual(set(sidecar), _META_KEYS)
        self.assertNotIn("kernel_id", sidecar)
        self.assertIn("def _add_kernel", sidecar["kernel_source"])
        self.assertEqual(sidecar["settings"]["static_shapes"], True)

    def test_config_column_is_json_round_trippable(self) -> None:
        """
        The ``config`` column stores config JSON that round-trips via
        Config.from_json (FR4).
        """
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(f"{tmp}/run", self._metadata()) as sink:
                sink.start_run()
                sink.record(self._entry(0.1, block_sizes=[32], num_warps=4))
                sink.end_run()
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
        header, data = rows[0], rows[1:]
        self.assertEqual(len(data), 1)
        cfg = helion.Config.from_json(data[0][header.index("config")])
        self.assertEqual(cfg.block_sizes, [32])

    def test_rows_join_to_sidecar_on_run_id(self) -> None:
        """
        Every CSV row carries the run_id join key matching the sidecar record;
        one row per recorded entry.
        """
        meta = self._metadata()
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(f"{tmp}/run", meta) as sink:
                sink.start_run()
                sink.record(self._entry(0.1))
                sink.record(self._entry(0.2))
                sink.end_run()
            sidecar = json.loads(
                sink.meta_path.read_text(encoding="utf-8").splitlines()[0]
            )
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
        header, data = rows[0], rows[1:]
        self.assertEqual(len(data), 2)
        rid_col = header.index("run_id")
        self.assertTrue(all(r[rid_col] == sidecar["run_id"] for r in data))

    def test_appends_across_runs_at_same_base_path(self) -> None:
        """
        Multiple runs sharing one base path accumulate: single CSV header, one
        data row per run, one sidecar record per run, distinct run_ids.
        """
        meta_a = self._metadata(
            kernel_name="kernel_a", kernel_source="def kernel_a(): ..."
        )
        meta_b = self._metadata(
            kernel_name="kernel_b",
            kernel_source="def kernel_b(): ...",
            input_shapes="[(128,)]",
            dtypes="['torch.float16']",
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = f"{tmp}/run"
            for meta in (meta_a, meta_b):
                with AutotuneLogSink(base, meta) as sink:
                    sink.start_run()
                    sink.record(self._entry(0.1))
                    sink.end_run()
            meta_lines = sink.meta_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(meta_lines), 2)
            rec_a, rec_b = (json.loads(line) for line in meta_lines)
            self.assertNotEqual(rec_a["run_id"], rec_b["run_id"])
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows[0], _LEAN_CSV_HEADER)
        data = rows[1:]
        self.assertEqual(len(data), 2)
        rid_col = rows[0].index("run_id")
        self.assertEqual([r[rid_col] for r in data], [rec_a["run_id"], rec_b["run_id"]])

    def test_without_metadata_writes_no_sidecar(self) -> None:
        """
        Without KernelMetadata a full run writes no sidecar (no identity).
        """
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(f"{tmp}/run") as sink:
                sink.start_run()
                sink.record(self._entry(0.1))
                sink.end_run()
            self.assertFalse(sink.meta_path.exists())

    def test_record_before_open_is_noop(self) -> None:
        """
        Recording on an unopened sink writes nothing and creates no CSV file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sink = AutotuneLogSink(f"{tmp}/run")  # not opened
            sink.record(self._entry(0.1))
            self.assertFalse(sink.csv_path.exists())


if __name__ == "__main__":
    unittest.main()
