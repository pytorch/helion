from __future__ import annotations

import csv
import json
import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

import helion
from helion._hardware import _device_props
from helion._hardware import collect_hardware_info
from helion._testing import TestCase
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuneLogSink
from helion.autotuner.metrics import KernelMetadata
import helion.language as hl

# Acceptance suite (pure-assertion, no CUDA) pinning the two-file contract:
#   .meta.jsonl (per run): run_id, kernel_name, kernel_source, input_shapes,
#                          dtypes, hardware, settings, configs
#   .csv (per entry):      run_id, timestamp_s, config_id, generation, status,
#                          perf_ms, compile_time_s, config
# Rows join to a config via the content-addressed config_id; the configs map
# dedups identical configs. Settings serialized JSON-safe.

_LEAN_CSV_HEADER = [
    "run_id",
    "timestamp_s",
    "config_id",
    "generation",
    "status",
    "perf_ms",
    "compile_time_s",
    "config",
]

# Keys of KernelMetadata.to_dict() (the per-run identity). The sink augments this
# with the run's ``configs`` map when writing the .meta.jsonl record.
_METADATA_DICT_KEYS = {
    "run_id",
    "kernel_name",
    "kernel_source",
    "input_shapes",
    "dtypes",
    "hardware",
    "hardware_info",
    "settings",
}

# Keys of one on-disk .meta.jsonl record (identity + ir_graph + configs).
_SIDECAR_KEYS = _METADATA_DICT_KEYS | {"ir_graph", "configs"}

# Fixed key set of the hardware_info snapshot (collect_hardware_info). The
# per-backend variation is confined to the nested ``versions`` block.
_HARDWARE_INFO_KEYS = {
    "device_id",
    "device_kind",
    "device_name",
    "compute_capability",
    "sm_count",
    "max_threads_per_sm",
    "max_threads_per_block",
    "warp_size",
    "shared_mem_per_block",
    "regs_per_multiprocessor",
    "total_mem",
    "l2_cache_size",
    "cpu_num_threads",
    "versions",
}


def _sample_hardware_info(**overrides: object) -> dict[str, object]:
    """Complete ``HardwareInfoRecord``-shaped dict (asserts full schema), with overrides."""
    record: dict[str, object] = {
        "device_id": "cuda:TestGPU:sm90",
        "device_kind": "cuda",
        "device_name": "TestGPU",
        "compute_capability": "sm90",
        "sm_count": 99,
        "max_threads_per_sm": 2048,
        "max_threads_per_block": 1024,
        "warp_size": 32,
        "shared_mem_per_block": 49152,
        "regs_per_multiprocessor": 65536,
        "total_mem": 1024,
        "l2_cache_size": 512,
        "cpu_num_threads": 8,
        "versions": {"torch": "x", "triton": "y", "helion": "z", "cuda": "12"},
    }
    assert set(record) == _HARDWARE_INFO_KEYS
    record.update(overrides)
    return record


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


class TestAutotuneDatasetSetting(TestCase):
    """The opt-in flag ``autotune_dataset`` (env HELION_AUTOTUNE_DATASET, FR1)."""

    def test_defaults_off(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_AUTOTUNE_DATASET", None)
            self.assertFalse(helion.Settings().autotune_dataset)

    def test_env_opt_in(self) -> None:
        with patch.dict(os.environ, {"HELION_AUTOTUNE_DATASET": "1"}, clear=False):
            self.assertTrue(helion.Settings().autotune_dataset)

    def test_explicit_argument_overrides_env(self) -> None:
        with patch.dict(os.environ, {"HELION_AUTOTUNE_DATASET": "1"}, clear=False):
            self.assertFalse(helion.Settings(autotune_dataset=False).autotune_dataset)


class TestDatasetWithoutLogWarning(TestCase):
    """Opting into the dataset without a log path warns once (FR3)."""

    def test_warns_only_once(self) -> None:
        import helion.autotuner.base_search as base_search

        log = MagicMock()
        with patch.object(base_search, "_DATASET_NO_LOG_WARNED", False):
            base_search._warn_dataset_without_log(log)
            base_search._warn_dataset_without_log(log)
        self.assertEqual(log.warning.call_count, 1)


class TestMetadataSchema(TestCase):
    def _metadata(self, **overrides: object) -> KernelMetadata:
        fields: dict[str, object] = {
            "kernel_name": "k",
            "kernel_source": "def k(): ...",
            "input_shapes": "[(16,)]",
            "dtypes": "['torch.float32']",
            "hardware": "TestGPU",
            "settings": {"static_shapes": True, "index_dtype": None},
        }
        fields.update(overrides)
        return KernelMetadata(**fields)

    def test_to_dict_has_exactly_the_lean_keys(self) -> None:
        """KernelMetadata.to_dict() carries exactly the lean per-run identity fields."""
        record = self._metadata().to_dict()
        self.assertEqual(set(record), _METADATA_DICT_KEYS)

    def test_to_dict_carries_hardware_info(self) -> None:
        """hardware_info (descriptive snapshot) rides on the record verbatim."""
        hw = _sample_hardware_info(device_name="TestGPU", sm_count=99)
        record = self._metadata(hardware_info=hw).to_dict()
        self.assertEqual(record["hardware_info"], hw)

    def test_hardware_info_excluded_from_run_id(self) -> None:
        """hardware_info never changes run_id (descriptive only)."""
        base = self._metadata(hardware_info=_sample_hardware_info(device_name="A"))
        other = self._metadata(
            hardware_info=_sample_hardware_info(device_name="B", sm_count=7)
        )
        self.assertEqual(base.run_id, other.run_id)

    def test_settings_are_stored_and_json_safe(self) -> None:
        """
        The full helion.settings are stored under ``settings`` and the record
        serializes JSON-safe even when settings carry non-serializable values
        (callables, torch.dtype) via ``default=str`` (FR5/NFR2).
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
        run_id is derived directly from (kernel_source, codegen-settings-signature,
        input_shapes, dtypes, hardware): derived when not supplied, stable for
        identical invocations, and distinct across every identity component (FR6).
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

    def test_default_metadata_serializes_with_empty_identity(self) -> None:
        """
        Default KernelMetadata carries an empty identity and still serializes, so a
        sidecar can be written even when the kernel is unidentifiable.
        """
        record = KernelMetadata().to_dict()
        self.assertEqual(record["kernel_source"], "")
        json.dumps(record, default=str)

    def test_codegen_settings_tagged(self) -> None:
        """CODEGEN_AFFECTING_SETTINGS is exactly the 10 codegen-tagged fields, in
        sorted order (the run_id wire format is stable across field reordering)."""
        from helion.runtime.settings import CODEGEN_AFFECTING_SETTINGS

        self.assertEqual(
            CODEGEN_AFFECTING_SETTINGS,
            (
                "allow_warp_specialize",
                "backend",
                "debug_dtype_asserts",
                "dot_precision",
                "fast_math",
                "index_dtype",
                "pallas_interpret",
                "persistent_reserved_sms",
                "static_shapes",
                "triton_do_not_specialize",
            ),
        )

    def test_run_id_signature_covers_codegen_settings(self) -> None:
        """run_id signature includes every codegen field in order, no extras."""
        from helion.autotuner.metrics import _codegen_signature
        from helion.runtime.settings import CODEGEN_AFFECTING_SETTINGS

        sig = _codegen_signature(helion.Settings().to_dict())
        for name in CODEGEN_AFFECTING_SETTINGS:
            self.assertIn(f"{name}=", sig)

    def test_run_id_distinguishes_each_codegen_setting(self) -> None:
        """Flipping any single codegen setting changes run_id, exercised with the
        setting's real value type (bool/int/str/torch.dtype/None)."""
        from helion.runtime.settings import CODEGEN_AFFECTING_SETTINGS

        # (base, changed) pairs with type-accurate values per setting.
        pairs: dict[str, tuple[object, object]] = {
            "backend": ("triton", "cute"),
            "dot_precision": ("tf32", "ieee"),
            "fast_math": (False, True),
            "static_shapes": (True, False),
            "index_dtype": (None, torch.int64),
            "allow_warp_specialize": (True, False),
            "triton_do_not_specialize": (False, True),
            "pallas_interpret": (False, True),
            "debug_dtype_asserts": (False, True),
            "persistent_reserved_sms": (0, 8),
        }
        base_settings = {n: pairs[n][0] for n in CODEGEN_AFFECTING_SETTINGS}
        base = self._metadata(settings=base_settings)
        for name in CODEGEN_AFFECTING_SETTINGS:
            with self.subTest(setting=name):
                changed = {**base_settings, name: pairs[name][1]}
                self.assertNotEqual(
                    base.run_id, self._metadata(settings=changed).run_id, name
                )


class TestCollectHardwareInfo(TestCase):
    """collect_hardware_info is best-effort, fixed-shape, and never raises."""

    def test_has_fixed_keys_and_versions_block(self) -> None:
        """Fixed key set; ``versions`` always carries torch/triton/helion."""
        info = collect_hardware_info()
        self.assertEqual(set(info), _HARDWARE_INFO_KEYS)
        self.assertIsInstance(info["versions"], dict)
        for name in ("torch", "triton", "helion"):
            self.assertIn(name, info["versions"])

    def test_serializes_json_safe(self) -> None:
        """The snapshot round-trips through the record's JSON-safe dump."""
        json.dumps(collect_hardware_info(), default=str)

    def test_returns_independent_copies(self) -> None:
        """Each call returns a fresh object; mutating one cannot poison later callers."""
        first = collect_hardware_info()
        second = collect_hardware_info()
        self.assertIsNot(first, second)
        first["device_name"] = "poisoned"
        first["versions"]["torch"] = "poisoned"
        third = collect_hardware_info()
        self.assertNotEqual(third["device_name"], "poisoned")
        self.assertNotEqual(third["versions"]["torch"], "poisoned")

    def test_cpu_device_not_misreported(self) -> None:
        """Explicit cpu device reports a cpu identity with null props, even on a GPU host."""
        info = collect_hardware_info(torch.device("cpu"))
        self.assertEqual(set(info), _HARDWARE_INFO_KEYS)
        self.assertEqual(info["device_kind"], "cpu")
        self.assertIsNone(info["sm_count"])
        self.assertIsNone(info["total_mem"])
        self.assertIsNone(info["l2_cache_size"])
        self.assertIsInstance(info["versions"], dict)

    def test_never_raises_on_prop_failure(self) -> None:
        """A device-property read failure degrades to null props instead of raising."""
        with patch(
            "torch.cuda.get_device_properties", side_effect=RuntimeError("boom")
        ):
            info = collect_hardware_info(torch.device("cuda"))
        self.assertEqual(set(info), _HARDWARE_INFO_KEYS)
        self.assertIsNone(info["sm_count"])

    def test_xpu_props_use_workgroup_fallbacks(self) -> None:
        """XPU props fill the analogous cuda-named fields; no-analog fields stay null."""
        fake = types.SimpleNamespace(
            max_compute_units=512,
            max_work_group_size=1024,
            local_mem_size=65536,
            total_memory=68719476736,
            # no warp_size / max_threads_per_multi_processor / L2 / regs
        )
        with (
            patch("torch.xpu.is_available", return_value=True),
            patch("torch.xpu.get_device_properties", return_value=fake),
        ):
            props = _device_props(torch.device("xpu"), "xpu")
        self.assertEqual(props["sm_count"], 512)
        self.assertEqual(props["max_threads_per_block"], 1024)
        self.assertEqual(props["shared_mem_per_block"], 65536)
        self.assertEqual(props["total_mem"], 68719476736)
        self.assertIsNone(props["warp_size"])
        self.assertIsNone(props["max_threads_per_sm"])
        self.assertIsNone(props["regs_per_multiprocessor"])
        self.assertIsNone(props["l2_cache_size"])

    def test_tpu_path_reports_xla_version(self) -> None:
        """The tpu branch reports null device props and an ``xla`` runtime version."""
        with (
            patch(
                "helion._hardware._hardware_identity",
                return_value=("tpu", "TPU v5", "TPU v5"),
            ),
            patch("helion._hardware._tpu_runtime_version", return_value="2.7.0"),
        ):
            info = collect_hardware_info()
        self.assertEqual(set(info), _HARDWARE_INFO_KEYS)
        self.assertEqual(info["device_kind"], "tpu")
        self.assertIsNone(info["sm_count"])
        self.assertEqual(info["versions"]["xla"], "2.7.0")


class TestAutotuneLogEntry(TestCase):
    def test_entry_carries_config_id_and_config(self) -> None:
        """
        AutotuneLogEntry references the config by content-addressed config_id (the
        .meta.jsonl join key) and also carries the config to be written to the CSV.
        """
        config = helion.Config(block_sizes=[16])
        entry = AutotuneLogEntry(
            generation=0,
            status="ok",
            perf_ms=1.0,
            compile_time=0.1,
            config_id="deadbeefdeadbeef",
            config=config,
        )
        self.assertEqual(entry.config_id, "deadbeefdeadbeef")
        self.assertIs(entry.config, config)
        self.assertFalse(hasattr(entry, "sample_id"))
        self.assertFalse(hasattr(entry, "decorator"))


class TestAutotuneLogSink(TestCase):
    def _metadata(self, **overrides: object) -> KernelMetadata:
        fields: dict[str, object] = {
            "kernel_name": "_add_kernel",
            "kernel_source": _add_kernel.kernel_source(),
            "input_shapes": "[(64,)]",
            "dtypes": "['torch.float32']",
            "hardware": "TestGPU",
            "settings": {"static_shapes": True, "index_dtype": None},
        }
        fields.update(overrides)
        return KernelMetadata(**fields)

    def _record(
        self,
        sink: AutotuneLogSink,
        config: helion.Config,
        *,
        status: str = "ok",
        perf_ms: float = 0.1,
        generation: int = 0,
    ) -> str:
        """Register ``config`` and record one entry; return its config_id."""
        config_id = sink.register_config(config)
        assert config_id is not None
        sink.record(
            AutotuneLogEntry(
                generation=generation,
                status=status,
                perf_ms=perf_ms,
                compile_time=0.5,
                config_id=config_id,
                config=config,
            )
        )
        return config_id

    def test_csv_header_is_lean(self) -> None:
        """The CSV header is exactly the lean column set (FR4)."""
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                self._record(sink, helion.Config(block_sizes=[16]))
                sink.end_run()
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                header = next(csv.reader(f))
        self.assertEqual(header, _LEAN_CSV_HEADER)

    def test_csv_config_column_carries_config(self) -> None:
        """The trailing ``config`` column is populated inline (compat with existing
        CSV consumers)."""
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                self._record(sink, helion.Config(block_sizes=[32]))
                sink.end_run()
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
        header, data = rows[0], rows[1:]
        self.assertEqual(len(data), 1)
        config_cell = data[0][header.index("config")]
        self.assertTrue(config_cell)
        self.assertIn("32", config_cell)

    def test_sidecar_has_lean_keys_and_configs(self) -> None:
        """
        With dataset collection enabled the .meta.jsonl sidecar holds one record
        per run with exactly the lean keys (including ``settings`` and the
        ``configs`` map) (FR5).
        """
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                self._record(sink, helion.Config(block_sizes=[16]))
                sink.end_run()
            meta_lines = sink.meta_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(meta_lines), 1)
        sidecar = json.loads(meta_lines[0])
        self.assertEqual(set(sidecar), _SIDECAR_KEYS)
        self.assertIn("def _add_kernel", sidecar["kernel_source"])
        self.assertEqual(sidecar["settings"]["static_shapes"], True)
        self.assertIsInstance(sidecar["configs"], dict)
        # configs map values contain perf_stats with 6 subfields
        for cfg_val in sidecar["configs"].values():
            self.assertIn("perf_stats", cfg_val)
            self.assertEqual(
                set(cfg_val["perf_stats"]),
                {"min", "median", "mean", "p90", "std", "n_samples"},
            )

    def test_sidecar_includes_ir_graph_when_provided(self) -> None:
        """The per-run record carries the ir_graph passed to the sink."""
        ir = {
            "schema_version": 1,
            "directed": True,
            "multigraph": False,
            "graph": {
                "num_graphs": 1,
                "root_ids": [0],
                "graphs": [],
                "rolled_reductions": [],
            },
            "nodes": [],
            "links": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True, ir_graph=ir
            ) as sink:
                sink.start_run()
                self._record(sink, helion.Config(block_sizes=[16]))
                sink.end_run()
            sidecar = json.loads(sink.meta_path.read_text().splitlines()[0])
        self.assertEqual(sidecar["ir_graph"], ir)

    def test_ir_graph_defaults_to_null_in_record(self) -> None:
        """With no ir_graph supplied the record still carries ir_graph=null."""
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                self._record(sink, helion.Config(block_sizes=[16]))
                sink.end_run()
            sidecar = json.loads(sink.meta_path.read_text().splitlines()[0])
        self.assertIn("ir_graph", sidecar)
        self.assertIsNone(sidecar["ir_graph"])

    def test_config_resolves_by_id_against_configs_map(self) -> None:
        """
        A CSV row's config_id resolves to its full config via the sidecar's
        ``configs`` map, round-tripping via Config.from_json (FR6).
        """
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                self._record(sink, helion.Config(block_sizes=[32], num_warps=4))
                sink.end_run()
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            sidecar = json.loads(
                sink.meta_path.read_text(encoding="utf-8").splitlines()[0]
            )
        header, data = rows[0], rows[1:]
        self.assertEqual(len(data), 1)
        config_id = data[0][header.index("config_id")]
        stored = sidecar["configs"][config_id]["config"]
        cfg = helion.Config.from_json(json.dumps(stored))
        self.assertEqual(cfg.block_sizes, [32])

    def test_config_id_stable_and_distinct(self) -> None:
        """
        config_id is content-addressed: identical configs share one id, different
        configs differ (FR6).
        """
        with (
            tempfile.TemporaryDirectory() as tmp,
            AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True
            ) as sink,
        ):
            sink.start_run()
            first = sink.register_config(helion.Config(block_sizes=[16]))
            same = sink.register_config(helion.Config(block_sizes=[16]))
            other = sink.register_config(helion.Config(block_sizes=[32]))
            sink.end_run()
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)

    def test_no_duplicate_config_entries(self) -> None:
        """
        Recording a config's started+ok rows and re-benchmarking it in a later
        generation share one config_id and produce exactly one ``configs`` entry
        (FR6 dedup).
        """
        cfg = helion.Config(block_sizes=[16], num_warps=4)
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                ids = [
                    self._record(sink, cfg, status="started", perf_ms=float("nan")),
                    self._record(sink, cfg, status="ok", perf_ms=0.1, generation=0),
                    self._record(sink, cfg, status="ok", perf_ms=0.2, generation=3),
                ]
                sink.end_run()
            sidecar = json.loads(
                sink.meta_path.read_text(encoding="utf-8").splitlines()[0]
            )
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(len(set(ids)), 1)  # all share one id
        self.assertEqual(len(sidecar["configs"]), 1)  # one configs entry
        self.assertEqual(len(rows) - 1, 3)  # three CSV rows

    def test_rows_join_to_sidecar_on_run_id(self) -> None:
        """
        Every CSV row carries the run_id join key matching the sidecar record;
        one row per recorded entry.
        """
        meta = self._metadata()
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(f"{tmp}/run", meta, collect_dataset=True) as sink:
                sink.start_run()
                self._record(sink, helion.Config(block_sizes=[16]))
                self._record(sink, helion.Config(block_sizes=[32]))
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
                with AutotuneLogSink(base, meta, collect_dataset=True) as sink:
                    sink.start_run()
                    self._record(sink, helion.Config(block_sizes=[16]))
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

    def test_without_dataset_flag_writes_csv_but_no_sidecar(self) -> None:
        """
        With dataset collection off (default), a full run writes the CSV (gated by
        the log path) but no .meta.jsonl, even with metadata present (FR1/FR9).
        """
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(f"{tmp}/run", self._metadata()) as sink:
                sink.start_run()
                self._record(sink, helion.Config(block_sizes=[16]))
                sink.end_run()
            self.assertTrue(sink.csv_path.exists())
            self.assertFalse(sink.meta_path.exists())

    def test_without_metadata_writes_no_sidecar(self) -> None:
        """
        Without KernelMetadata a full run writes no sidecar even when dataset
        collection is requested (no identity to write).
        """
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(f"{tmp}/run", collect_dataset=True) as sink:
                sink.start_run()
                self._record(sink, helion.Config(block_sizes=[16]))
                sink.end_run()
            self.assertFalse(sink.meta_path.exists())

    def test_register_config_before_open_returns_none(self) -> None:
        """
        register_config on an unopened sink safely returns None (Gap 1).
        """
        with tempfile.TemporaryDirectory() as tmp:
            sink = AutotuneLogSink(f"{tmp}/run")  # not opened
            self.assertIsNone(sink.register_config(helion.Config(block_sizes=[16])))

    def test_record_before_open_is_noop(self) -> None:
        """
        Recording on an unopened sink writes nothing and creates no CSV file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sink = AutotuneLogSink(f"{tmp}/run")  # not opened
            sink.record(
                AutotuneLogEntry(
                    generation=0,
                    status="ok",
                    perf_ms=0.1,
                    compile_time=0.5,
                    config_id="abc123",
                    config=helion.Config(block_sizes=[16]),
                )
            )
            self.assertFalse(sink.csv_path.exists())


class TestPerfStatsSchema(TestCase):
    def _metadata(self, **overrides: object) -> KernelMetadata:
        fields: dict[str, object] = {
            "kernel_name": "_add_kernel",
            "kernel_source": _add_kernel.kernel_source(),
            "input_shapes": "[(64,)]",
            "dtypes": "['torch.float32']",
            "hardware": "TestGPU",
            "settings": {"static_shapes": True, "index_dtype": None},
        }
        fields.update(overrides)
        return KernelMetadata(**fields)

    def test_sink_records_perf_stats(self) -> None:
        """AutotuneLogSink stores perf_stats dict in configs map on record."""
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                cfg = helion.Config(block_sizes=[16])
                config_id = sink.register_config(cfg)
                assert config_id is not None
                stats = {
                    "min": 0.1,
                    "median": 0.12,
                    "mean": 0.13,
                    "p90": 0.15,
                    "std": 0.01,
                    "n_samples": 50,
                }
                sink.record(
                    AutotuneLogEntry(
                        generation=0,
                        status="ok",
                        perf_ms=0.12,
                        compile_time=0.5,
                        config_id=config_id,
                        config=cfg,
                        perf_stats=stats,
                    )
                )
                sink.end_run()
            sidecar = json.loads(sink.meta_path.read_text().splitlines()[0])
        cfg_val = sidecar["configs"][config_id]
        self.assertIn("perf_stats", cfg_val)
        self.assertEqual(cfg_val["perf_stats"], stats)

    def test_null_template_on_failure(self) -> None:
        """Failure status results in null perf_stats template."""
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", self._metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                cfg = helion.Config(block_sizes=[16])
                config_id = sink.register_config(cfg)
                assert config_id is not None
                sink.record(
                    AutotuneLogEntry(
                        generation=0,
                        status="error",
                        perf_ms=None,
                        compile_time=0.5,
                        config_id=config_id,
                        config=cfg,
                        perf_stats=None,
                    )
                )
                sink.end_run()
            sidecar = json.loads(sink.meta_path.read_text().splitlines()[0])
        perf_stats = sidecar["configs"][config_id]["perf_stats"]
        self.assertEqual(
            perf_stats,
            {
                "min": None,
                "median": None,
                "mean": None,
                "p90": None,
                "std": None,
                "n_samples": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
