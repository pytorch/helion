from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
import tempfile
from typing import TYPE_CHECKING
from typing import cast
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner._metadata.hardware import _DEVICE_PROPS_ATTRS
from helion.autotuner._metadata.hardware import _GPU_BACKENDS
from helion.autotuner._metadata.hardware import HardwareInfoRecord
from helion.autotuner._metadata.hardware import collect_hardware_info
from helion.autotuner._metadata.ir_features import _has_networkx_node_link
from helion.autotuner.base_search import _warn_dataset_without_log
from helion.autotuner.benchmarking import PerfStats
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuneLogSink
from helion.autotuner.metrics import KernelMetadata
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.device_ir import DeviceIR
    from helion.autotuner.logger import ConfigEntry

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

_SIDECAR_KEYS = {
    "run_id",
    "kernel_name",
    "kernel_source",
    "input_shapes",
    "dtypes",
    "hardware_info",
    "settings",
    "ir_graph",
    "configs",
}

# Schema key set from the TypedDict itself (``.keys()`` fails on a TypedDict at runtime).
_HARDWARE_INFO_KEYS = set(HardwareInfoRecord.__annotations__)

# Always-present subset (device_props is GPU-only/optional); TPU/CPU carry exactly these.
_HARDWARE_INFO_REQUIRED_KEYS = set(HardwareInfoRecord.__required_keys__)

# Mirrors the extractor gate: missing networkx should degrade to None, not fail E2E.
_HAS_NETWORKX = _has_networkx_node_link()


@helion.kernel(config=helion.Config(block_sizes=[16]))
def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


def _metadata() -> KernelMetadata:
    return KernelMetadata(
        kernel_name="_add_kernel",
        kernel_source=_add_kernel.kernel_source(),
        input_shapes="[(64,)]",
        dtypes="['torch.float32']",
        settings={"static_shapes": True, "index_dtype": None},
        # cpu device: a device-less hardware_info probe raises on non-accelerator
        # CI lanes (metal/pallas). The real GPU snapshot is covered by the E2E test.
        _device=torch.device("cpu"),
    )


class TestAutotuneLogDetailsSetting(TestCase):
    def test_autotune_log_details_opt_in(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_AUTOTUNE_LOG_DETAILS", None)
            self.assertFalse(helion.Settings().autotune_log_details)
        with patch.dict(os.environ, {"HELION_AUTOTUNE_LOG_DETAILS": "1"}, clear=False):
            self.assertTrue(helion.Settings().autotune_log_details)

    def test_autotune_log_details_without_log_warns(self) -> None:
        _warn_dataset_without_log.cache_clear()
        self.addCleanup(_warn_dataset_without_log.cache_clear)

        log = Mock()
        _warn_dataset_without_log(log)
        _warn_dataset_without_log(log)
        log.warning.assert_called_once_with(
            "HELION_AUTOTUNE_LOG_DETAILS is set but HELION_AUTOTUNE_LOG is not; no "
            "autotune dataset will be collected. Set HELION_AUTOTUNE_LOG to a base "
            "path to enable collection."
        )

        other_log = Mock()
        _warn_dataset_without_log(other_log)
        other_log.warning.assert_called_once()


class TestCodegenSettings(TestCase):
    def test_codegen_settings_pinned_and_valid(self) -> None:
        from helion.autotuner.metrics import _CODEGEN_SETTINGS

        self.assertEqual(
            _CODEGEN_SETTINGS,
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
        # run_id wire format is order-sensitive.
        self.assertEqual(_CODEGEN_SETTINGS, tuple(sorted(_CODEGEN_SETTINGS)))
        settings_keys = helion.Settings().to_dict()
        for name in _CODEGEN_SETTINGS:
            self.assertIn(name, settings_keys)


class TestRunId(TestCase):
    def test_run_id_changes_with_codegen_settings(self) -> None:
        from helion.autotuner.metrics import _CODEGEN_SETTINGS

        # A new _CODEGEN_SETTINGS entry without a pair here raises KeyError.
        pairs: dict[str, tuple[object, object]] = {
            "allow_warp_specialize": (True, False),
            "backend": ("triton", "cute"),
            "debug_dtype_asserts": (False, True),
            "dot_precision": ("tf32", "ieee"),
            "fast_math": (False, True),
            "index_dtype": (None, torch.int64),
            "pallas_interpret": (False, True),
            "persistent_reserved_sms": (0, 8),
            "static_shapes": (True, False),
            "triton_do_not_specialize": (False, True),
        }
        base_settings = {name: pairs[name][0] for name in _CODEGEN_SETTINGS}

        def _run_id(settings: dict[str, object]) -> str:
            return KernelMetadata(
                kernel_name="k",
                kernel_source="def k(): ...",
                input_shapes="[(64,)]",
                dtypes="['torch.float32']",
                settings=settings,
            ).run_id

        base_run_id = _run_id(base_settings)
        for name in _CODEGEN_SETTINGS:
            with self.subTest(setting=name):
                changed = {**base_settings, name: pairs[name][1]}
                self.assertNotEqual(base_run_id, _run_id(changed), name)


class TestCollectHardwareInfo(TestCase):
    """collect_hardware_info has best-effort numeric props and a backend-aware
    versions block; identity/required-version probes may raise."""

    @unittest.skipIf(
        DEVICE.type in {"mps", "mtia"},
        "collect_hardware_info probes cpu/cuda/rocm/xpu/tpu, not this device",
    )
    def test_collect_hardware_info_has_required_content(self) -> None:
        """Real (unpatched) snapshot has the production schema and stable fields.
        Best-effort / backend-specific fields are intentionally not asserted."""
        info = collect_hardware_info(DEVICE)
        # device_props is GPU-only (optional key); the always-present keys are a subset.
        self.assertLessEqual(_HARDWARE_INFO_REQUIRED_KEYS, set(info))
        self.assertLessEqual(set(info), _HARDWARE_INFO_KEYS)
        self.assertIsNotNone(info["device_kind"])
        self.assertIsNotNone(info["device_name"])
        self.assertIsNotNone(info["cpu_num_threads"])
        self.assertEqual(info["versions"]["torch"], torch.__version__)
        self.assertIn("helion", info["versions"])

        if info["device_kind"] in _GPU_BACKENDS:
            self.assertIn("device_props", info)
            self.assertEqual(
                set(info["device_props"]), set(_DEVICE_PROPS_ATTRS[info["device_kind"]])
            )
            # First attr is the backend's compute-unit count (multi_processor_count on
            # cuda/rocm, max_compute_units on xpu); a populated value proves the probe
            # ran, not the all-None miss fallback.
            sm_attr = _DEVICE_PROPS_ATTRS[info["device_kind"]][0]
            self.assertIsNotNone(info["device_props"][sm_attr])
            self.assertIsNotNone(info["versions"]["triton"])

    def test_cpu_device_not_misreported(self) -> None:
        """Explicit cpu device reports a cpu identity with no device_props block (not
        a GPU) and no backend toolkit, even on a GPU host."""
        info = collect_hardware_info(torch.device("cpu"))
        self.assertEqual(set(info), _HARDWARE_INFO_REQUIRED_KEYS)
        self.assertEqual(info["device_kind"], "cpu")
        self.assertNotIn("device_props", info)
        self.assertEqual(set(info["versions"]), {"torch", "helion"})


class TestKernelMetadataHardwareInfo(TestCase):
    """hardware_info is collected lazily in to_dict() and excluded from run_id."""

    def test_collected_lazily_in_to_dict(self) -> None:
        """Constructing the shell must not probe hardware; collection happens only
        when to_dict() materializes the record (the dataset-only write path)."""
        with patch(
            "helion.autotuner._metadata.hardware.collect_hardware_info"
        ) as collect:
            collect.return_value = {"device_kind": "cuda"}
            meta = _metadata()
            collect.assert_not_called()
            record = meta.to_dict()
            collect.assert_called_once()
        self.assertEqual(record["hardware_info"], {"device_kind": "cuda"})

    def test_run_id_uses_device_name(self) -> None:
        """run_id derives its hardware dimension from the device name."""
        with patch("helion.autotuner.metrics.get_device_name", return_value="H100"):
            run_id_h100 = _metadata().run_id
            run_id_h100_again = _metadata().run_id
        self.assertEqual(run_id_h100, run_id_h100_again)


class TestAutotuneLogSink(TestCase):
    def test_dataset_logged_when_enabled(self) -> None:
        config = helion.Config(block_sizes=[32], num_warps=4)
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", _metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                config_id = sink.register_config(config)
                assert config_id is not None
                sink.record(
                    AutotuneLogEntry(
                        generation=5,
                        status="ok",
                        perf_ms=1.234,
                        compile_time=0.5,
                        config_id=config_id,
                        config=config,
                    )
                )
                sink.end_run()
            with sink.csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            sidecar = json.loads(
                sink.meta_path.read_text(encoding="utf-8").splitlines()[0]
            )

        header, data = rows[0], rows[1:]
        self.assertEqual(header, _LEAN_CSV_HEADER)
        self.assertEqual(len(data), 1)

        def cell(name: str) -> str:
            return data[0][header.index(name)]

        self.assertTrue(cell("config"))
        self.assertIn("32", cell("config"))

        self.assertEqual(set(sidecar), _SIDECAR_KEYS)
        self.assertIn("def _add_kernel", sidecar["kernel_source"])

        # The CSV row joins to its config via config_id; each entry nests the
        # tested config alongside its generated_code (None here -- this device-free
        # sink test does not capture source).
        entry = sidecar["configs"][cell("config_id")]
        self.assertIsNone(entry["generated_code"])
        cfg = helion.Config.from_json(json.dumps(entry["config"]))
        self.assertEqual(cfg.block_sizes, [32])
        self.assertEqual(cell("run_id"), sidecar["run_id"])

    def _capture_entry(
        self,
        kernel: Mock,
        config: helion.Config,
        *,
        collect_dataset: bool = True,
        times: int = 1,
    ) -> dict[str, object] | None:
        """Run one config through a sink (register + ``capture_generated_code`` x
        ``times``) and return its written ``configs`` entry (``None`` if not collected)."""
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", _metadata(), collect_dataset=collect_dataset
            ) as sink:
                sink.start_run()
                config_id = sink.register_config(config)
                assert config_id is not None
                for _ in range(times):
                    sink.capture_generated_code(config_id, kernel, config)
                sink.end_run()
            configs = json.loads(
                sink.meta_path.read_text(encoding="utf-8").splitlines()[0]
            )["configs"]
        return configs.get(config_id)

    def test_capture_generated_code_attaches_source_read_once(self) -> None:
        """Source attaches to the config entry; the file is read at most once."""
        config = helion.Config(block_sizes=[32], num_warps=4)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "compiled_kernel.py"
            src.write_text("# generated triton source\n", encoding="utf-8")
            kernel = Mock()
            kernel.get_cached_path.return_value = str(src)
            entry = self._capture_entry(kernel, config, times=2)
        assert entry is not None
        self.assertEqual(entry["generated_code"], "# generated triton source\n")
        self.assertEqual(
            helion.Config.from_json(json.dumps(entry["config"])).block_sizes, [32]
        )
        kernel.get_cached_path.assert_called_once()

    def test_capture_generated_code_none_when_no_cached_path(self) -> None:
        """generated_code stays None when the config has no cached artifact path."""
        config = helion.Config(block_sizes=[32])
        kernel = Mock()
        kernel.get_cached_path.return_value = None
        entry = self._capture_entry(kernel, config)
        assert entry is not None
        self.assertIsNone(entry["generated_code"])

    def test_capture_generated_code_raises_when_artifact_unreadable(self) -> None:
        """A read failure of a known artifact surfaces (beta feature fails loudly)."""
        config = helion.Config(block_sizes=[32])
        kernel = Mock()
        kernel.get_cached_path.return_value = "/nonexistent/dir/compiled_kernel.py"
        with self.assertRaises(OSError):
            self._capture_entry(kernel, config)

    def test_capture_generated_code_noop_when_not_collecting(self) -> None:
        """With collection off there is no configs entry, so the file is never read
        (no .meta.jsonl is written either)."""
        config = helion.Config(block_sizes=[32])
        kernel = Mock()
        with (
            tempfile.TemporaryDirectory() as tmp,
            AutotuneLogSink(f"{tmp}/run", _metadata(), collect_dataset=False) as sink,
        ):
            sink.start_run()
            config_id = sink.register_config(config)
            assert config_id is not None
            sink.capture_generated_code(config_id, kernel, config)
            sink.end_run()
        kernel.get_cached_path.assert_not_called()


_PERF_STATS_KEYS = {"min", "median", "mean", "p90", "std", "n_samples"}
_PERF_STATS = PerfStats(
    min=1.0,
    median=1.1,
    mean=1.2,
    p90=1.3,
    std=0.1,
    n_samples=50,
)


class TestPerfStats(TestCase):
    """perf_stats rides in the meta configs map; last *successful* benchmark wins."""

    def _record(
        self,
        entries: Callable[[str, helion.Config], list[AutotuneLogEntry]],
    ) -> tuple[str, dict[str, object]]:
        config = helion.Config(block_sizes=[32], num_warps=4)
        with tempfile.TemporaryDirectory() as tmp:
            with AutotuneLogSink(
                f"{tmp}/run", _metadata(), collect_dataset=True
            ) as sink:
                sink.start_run()
                config_id = sink.register_config(config)
                assert config_id is not None
                for entry in entries(config_id, config):
                    sink.record(entry)
                sink.end_run()
            record = json.loads(
                sink.meta_path.read_text(encoding="utf-8").splitlines()[0]
            )
        return config_id, record["configs"][config_id]

    def test_perf_stats_recorded_per_config(self) -> None:
        _cid, entry = self._record(
            lambda cid, cfg: [
                AutotuneLogEntry(
                    generation=0,
                    status="ok",
                    perf_ms=1.1,
                    compile_time=0.5,
                    config_id=cid,
                    config=cfg,
                    perf_stats=_PERF_STATS,
                )
            ]
        )
        self.assertEqual(set(entry["perf_stats"]), _PERF_STATS_KEYS)
        self.assertEqual(entry["perf_stats"], _PERF_STATS.to_dict())

    def test_started_only_config_has_null_perf_stats(self) -> None:
        _cid, entry = self._record(
            lambda cid, cfg: [
                AutotuneLogEntry(
                    generation=0,
                    status="started",
                    perf_ms=None,
                    compile_time=None,
                    config_id=cid,
                    config=cfg,
                )
            ]
        )
        self.assertEqual(set(entry["perf_stats"]), _PERF_STATS_KEYS)
        self.assertEqual(entry["perf_stats"]["n_samples"], 0)
        self.assertIsNone(entry["perf_stats"]["median"])

    def test_failed_rebenchmark_keeps_good_perf_stats(self) -> None:
        _cid, entry = self._record(
            lambda cid, cfg: [
                AutotuneLogEntry(
                    generation=0,
                    status="ok",
                    perf_ms=1.1,
                    compile_time=0.5,
                    config_id=cid,
                    config=cfg,
                    perf_stats=_PERF_STATS,
                ),
                AutotuneLogEntry(
                    generation=1,
                    status="error",
                    perf_ms=None,
                    compile_time=None,
                    config_id=cid,
                    config=cfg,
                    perf_stats=None,
                ),
            ]
        )
        self.assertEqual(entry["perf_stats"], _PERF_STATS.to_dict())


@onlyBackends(["triton"])
class TestAutotuneDatasetE2E(TestCase):
    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_autotune_writes_dataset_sidecar(self) -> None:
        configs = [
            helion.Config(block_sizes=[32]),
            helion.Config(block_sizes=[64]),
        ]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        base_path = Path(tmpdir.name) / "autotune_run"

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNE_LOG": str(base_path),
                "HELION_AUTOTUNE_LOG_DETAILS": "1",
                "HELION_AUTOTUNE_LOG_LEVEL": "0",
            },
        ):

            @helion.kernel()
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            args = (
                torch.randn([64], device=DEVICE),
                torch.randn([64], device=DEVICE),
            )
            bound_kernel = add.bind(args)
            random.seed(123)
            search = FiniteSearch(bound_kernel, args, configs=configs)
            search.autotune()

        csv_path = base_path.with_suffix(".csv")
        meta_path = base_path.with_suffix(".meta.jsonl")
        self.assertTrue(csv_path.exists())
        self.assertTrue(meta_path.exists())

        # Union the per-run configs maps and run_ids from the sidecar records, and
        # keep a per-run-id view so CSV rows can be joined to their own record.
        configs_by_id: dict[str, ConfigEntry] = {}
        configs_by_run_id: dict[str, dict[str, ConfigEntry]] = {}
        run_ids: set[str] = set()
        meta_lines = [
            line for line in meta_path.read_text().splitlines() if line.strip()
        ]
        self.assertGreater(len(meta_lines), 0)
        for line in meta_lines:
            record = json.loads(line)
            self.assertIn("kernel_source", record)
            hw = record["hardware_info"]
            # device_props is GPU-only (optional); always-present keys are a subset.
            self.assertLessEqual(_HARDWARE_INFO_REQUIRED_KEYS, set(hw))
            self.assertLessEqual(set(hw), _HARDWARE_INFO_KEYS)
            # torch/helion are always present; per-backend toolkit keys vary by host.
            self.assertLessEqual({"torch", "helion"}, set(hw["versions"]))
            self.assertTrue(hw["device_name"])
            # GPU lanes report a real accelerator + sm_count + triton toolchain;
            # tolerate a triton CPU-fallback or TPU backend.
            if hw["device_kind"] in _GPU_BACKENDS:
                self.assertIn("device_props", hw)
                # Backend's compute-unit count (see collect_hardware_info test above).
                sm_attr = _DEVICE_PROPS_ATTRS[hw["device_kind"]][0]
                self.assertIsNotNone(hw["device_props"][sm_attr])
                self.assertIsNotNone(hw["versions"]["triton"])
            record_configs = cast("dict[str, ConfigEntry]", record["configs"])
            # Each configs entry nests the tested config and its per-config generated
            # source; generated_code is null when the config has no cached artifact
            # path (a config that failed to compile is still registered).
            for cfg_entry in record_configs.values():
                self.assertIn("config", cfg_entry)
                self.assertIsInstance(cfg_entry["generated_code"], (str, type(None)))
                self.assertEqual(set(cfg_entry["perf_stats"]), _PERF_STATS_KEYS)
            configs_by_id.update(record_configs)
            configs_by_run_id.setdefault(record["run_id"], {}).update(record_configs)
            run_ids.add(record["run_id"])
            if _HAS_NETWORKX:
                ir = record["ir_graph"]
                self.assertIsNotNone(ir)
                self.assertTrue(ir["nodes"])
                self.assertIn("edges", ir)
                self.assertTrue(ir["directed"])
            else:
                self.assertIsNone(record["ir_graph"])
        self.assertGreater(len(configs_by_id), 0)
        # generated_code is null only when a config has no cached path, but a real
        # Triton search compiles at least one config, so at least one is captured.
        self.assertTrue(any(e["generated_code"] for e in configs_by_id.values()))

        rows = list(csv.reader(csv_path.read_text().splitlines()))
        header, data = rows[0], rows[1:]
        self.assertGreater(len(data), 0)
        config_id = data[0][header.index("config_id")]
        self.assertIn(config_id, configs_by_id)
        self.assertIn(data[0][header.index("run_id")], run_ids)
        decoded_config = helion.Config.from_json(
            json.dumps(configs_by_id[config_id]["config"])
        )
        self.assertIn(decoded_config.block_sizes, ([32], [64]))

        # Every successful (status="ok") CSV row must join -- within its OWN run_id
        # record -- to a config entry with real benchmark stats. The contract is
        # that a successful benchmark always yields a PerfStats (n_samples >= 1,
        # non-None median), never the null sentinel. This is the e2e proof of
        # return_mode="stats"; it is stronger than "some config has a sample" and
        # the device-free sink tests (which inject _PERF_STATS) cannot prove it.
        # Keying by (run_id, config_id) -- not the unioned map -- ensures a broken
        # per-run join isn't masked by another record carrying the same config.
        status_idx = header.index("status")
        cfg_idx = header.index("config_id")
        run_idx = header.index("run_id")
        ok_rows = [row for row in data if row[status_idx] == "ok"]
        self.assertTrue(ok_rows)
        for row in ok_rows:
            row_run_id, row_cfg_id = row[run_idx], row[cfg_idx]
            self.assertIn(row_run_id, configs_by_run_id)
            self.assertIn(row_cfg_id, configs_by_run_id[row_run_id])
            stats = configs_by_run_id[row_run_id][row_cfg_id]["perf_stats"]
            self.assertGreaterEqual(stats["n_samples"], 1)
            self.assertIsNotNone(stats["median"])


class TestIrGraphDegradation(TestCase):
    def test_ir_graph_none_without_device_ir(self) -> None:
        # cpu device for a host-independent hardware_info probe; see _metadata().
        meta = KernelMetadata(
            kernel_name="k", kernel_source="src", _device=torch.device("cpu")
        )
        record = meta.to_dict()
        self.assertIn("ir_graph", record)
        self.assertIsNone(record["ir_graph"])

    def test_run_id_excludes_device_ir(self) -> None:
        fields = {
            "kernel_name": "k",
            "kernel_source": "src",
            "input_shapes": "[s0]",
            "dtypes": "torch.float32",
            "settings": {},
        }
        self.assertEqual(
            KernelMetadata(**fields).run_id,
            KernelMetadata(**fields, _device_ir=cast("DeviceIR", object())).run_id,
        )


if __name__ == "__main__":
    unittest.main()
