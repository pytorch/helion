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
from helion.autotuner._metadata.hardware import _GPU_BACKENDS
from helion.autotuner._metadata.hardware import DevicePropsRecord
from helion.autotuner._metadata.hardware import HardwareInfoRecord
from helion.autotuner._metadata.hardware import _device_id
from helion.autotuner._metadata.hardware import collect_hardware_info
from helion.autotuner._metadata.ir_features import _has_networkx_node_link
from helion.autotuner.base_search import _warn_dataset_without_log
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuneLogSink
from helion.autotuner.metrics import KernelMetadata
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.device_ir import DeviceIR

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
_DEVICE_PROPS_KEYS = set(DevicePropsRecord.__annotations__)

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
        self.assertIsNotNone(info["device_id"])
        self.assertIsNotNone(info["cpu_num_threads"])
        self.assertEqual(info["versions"]["torch"], torch.__version__)
        self.assertIn("helion", info["versions"])

        if info["device_kind"] in _GPU_BACKENDS:
            self.assertIn("device_props", info)
            self.assertEqual(set(info["device_props"]), _DEVICE_PROPS_KEYS)
            self.assertIsNotNone(info["device_props"]["sm_count"])
            self.assertIsNotNone(info["versions"]["triton"])

    def test_cpu_device_not_misreported(self) -> None:
        """Explicit cpu device reports a cpu identity with no device_props block (not
        a GPU) and no backend toolkit, even on a GPU host."""
        info = collect_hardware_info(torch.device("cpu"))
        self.assertEqual(set(info), _HARDWARE_INFO_REQUIRED_KEYS)
        self.assertEqual(info["device_kind"], "cpu")
        self.assertNotIn("device_props", info)
        self.assertEqual(set(info["versions"]), {"torch", "helion"})

    def test_device_id_sanitizes_name(self) -> None:
        """device_id stays kind:name:cap-parseable: spaces and reserved punctuation
        in the device name collapse to ``_``; a missing name yields None."""
        self.assertEqual(
            _device_id("cuda", "NVIDIA H100", "sm90"), "cuda:NVIDIA_H100:sm90"
        )
        self.assertIsNone(_device_id("cuda", "", "sm90"))
        self.assertIsNone(_device_id("cuda", None, "sm90"))


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

        stored = sidecar["configs"][cell("config_id")]
        cfg = helion.Config.from_json(json.dumps(stored))
        self.assertEqual(cfg.block_sizes, [32])
        self.assertEqual(cell("run_id"), sidecar["run_id"])


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

        configs_by_id: dict[str, object] = {}
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
                self.assertIsNotNone(hw["device_props"]["sm_count"])
                self.assertIsNotNone(hw["versions"]["triton"])
            configs_by_id.update(record["configs"])
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

        rows = list(csv.reader(csv_path.read_text().splitlines()))
        header, data = rows[0], rows[1:]
        self.assertGreater(len(data), 0)
        config_id = data[0][header.index("config_id")]
        self.assertIn(config_id, configs_by_id)
        self.assertIn(data[0][header.index("run_id")], run_ids)
        decoded_config = helion.Config.from_json(json.dumps(configs_by_id[config_id]))
        self.assertIn(decoded_config.block_sizes, ([32], [64]))


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
