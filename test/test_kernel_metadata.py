from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner._metadata.hardware import _device_id
from helion.autotuner._metadata.hardware import _device_props
from helion.autotuner._metadata.hardware import _hardware_versions
from helion.autotuner._metadata.hardware import collect_hardware_info
from helion.autotuner._metadata.ir_features import _has_networkx_node_link
from helion.autotuner.base_search import _warn_dataset_without_log
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuneLogSink
from helion.autotuner.metrics import KernelMetadata
import helion.language as hl

# Telemetry suite. Device-free tests run on all lanes; E2E is Triton-gated.
# CSVs join to .meta.jsonl via config_id.
# Sidecar dataset is written only when enabled.

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

# Keys of one on-disk .meta.jsonl record: the KernelMetadata identity, the
# config-independent device-IR dump, plus the run's config_id -> config map.
_SIDECAR_KEYS = {
    "run_id",
    "kernel_name",
    "kernel_source",
    "input_shapes",
    "dtypes",
    "hardware",
    "hardware_info",
    "settings",
    "ir_graph",
    "configs",
}

# Fixed key set of the hardware_info snapshot (collect_hardware_info); per-backend
# variation is confined to the nested ``versions`` block.
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

# Same capability check the extractor uses: the e2e ir_graph tests run the producer
# (which needs networkx>=3.4) and assert ir_graph is populated, so they skip rather
# than fail when networkx is missing or too old.
_HAS_NETWORKX = _has_networkx_node_link()


@helion.kernel(config=helion.Config(block_sizes=[16]))
def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A kernel that adds two tensors."""
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
        hardware="TestGPU",
        settings={"static_shapes": True, "index_dtype": None},
    )


class TestAutotuneLogDetailsSetting(TestCase):
    def test_autotune_log_details_opt_in(self) -> None:
        """``autotune_log_details`` is off by default and enabled via the env flag."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_AUTOTUNE_LOG_DETAILS", None)
            self.assertFalse(helion.Settings().autotune_log_details)
        with patch.dict(os.environ, {"HELION_AUTOTUNE_LOG_DETAILS": "1"}, clear=False):
            self.assertTrue(helion.Settings().autotune_log_details)

    def test_autotune_log_details_without_log_warns(self) -> None:
        """Opt-in without a log path warns once per logger and writes nothing.
        Tested via ``_warn_dataset_without_log`` to stay independent of the
        autotune flow. Uses ``@functools.cache`` to ensure it fires only once.
        """
        _warn_dataset_without_log.cache_clear()
        self.addCleanup(_warn_dataset_without_log.cache_clear)

        log = Mock()
        _warn_dataset_without_log(log)
        _warn_dataset_without_log(log)  # same logger -> cached, no second warning
        log.warning.assert_called_once_with(
            "HELION_AUTOTUNE_LOG_DETAILS is set but HELION_AUTOTUNE_LOG is not; no "
            "autotune dataset will be collected. Set HELION_AUTOTUNE_LOG to a base "
            "path to enable collection."
        )

        # A distinct logger is a distinct cache key -> the warning fires again.
        other_log = Mock()
        _warn_dataset_without_log(other_log)
        other_log.warning.assert_called_once()


class TestCodegenSettings(TestCase):
    def test_codegen_settings_pinned_and_valid(self) -> None:
        """Pins and validates run_id codegen fields against ``Settings``.
        Catches typos/renames and forces conscious updates for new codegen settings.
        Note: Cannot detect new codegen settings omitted from this list.
        """
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
        # Sorted -> stable run_id wire format across edits.
        self.assertEqual(_CODEGEN_SETTINGS, tuple(sorted(_CODEGEN_SETTINGS)))
        settings_keys = helion.Settings().to_dict()
        for name in _CODEGEN_SETTINGS:
            self.assertIn(name, settings_keys)


class TestRunId(TestCase):
    def test_run_id_changes_with_codegen_settings(self) -> None:
        """Verifies that flipping any codegen setting changes the run_id.
        Ensures runs with different generated code never collide; tested with real types.
        """
        from helion.autotuner.metrics import _CODEGEN_SETTINGS

        # Type-accurate (base, changed) value per codegen setting. A new entry in
        # _CODEGEN_SETTINGS without a pair here raises KeyError.
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
                hardware="TestGPU",
                settings=settings,
            ).run_id

        base_run_id = _run_id(base_settings)
        for name in _CODEGEN_SETTINGS:
            with self.subTest(setting=name):
                changed = {**base_settings, name: pairs[name][1]}
                self.assertNotEqual(base_run_id, _run_id(changed), name)


class TestCollectHardwareInfo(TestCase):
    """collect_hardware_info is best-effort, fixed-shape, and never raises."""

    def test_has_fixed_keys_and_versions_block(self) -> None:
        """Fixed key set; ``versions`` carries torch/triton/helion; each call returns
        a fresh object (no shared mutable state to poison)."""
        info = collect_hardware_info()
        self.assertEqual(set(info), _HARDWARE_INFO_KEYS)
        self.assertIsInstance(info["versions"], dict)
        for name in ("torch", "triton", "helion"):
            self.assertIn(name, info["versions"])
        # Built fresh per call: mutating one result cannot poison the next.
        info["versions"]["torch"] = "poisoned"
        self.assertNotEqual(collect_hardware_info()["versions"]["torch"], "poisoned")

    def test_cpu_device_not_misreported(self) -> None:
        """Explicit cpu device reports a cpu identity with null props, even on a GPU host."""
        info = collect_hardware_info(torch.device("cpu"))
        self.assertEqual(set(info), _HARDWARE_INFO_KEYS)
        self.assertEqual(info["device_kind"], "cpu")
        self.assertIsNone(info["sm_count"])
        self.assertIsNone(info["total_mem"])
        self.assertIsNone(info["l2_cache_size"])
        self.assertIsInstance(info["versions"], dict)

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

    def test_cuda_props_partial_attrs_fall_back_to_none(self) -> None:
        """A props object missing some attrs yields per-field None, never an error."""
        fake = types.SimpleNamespace(
            multi_processor_count=132,
            max_threads_per_block=1024,
            # no warp_size / shared_memory_per_block / regs / total / L2 / per-sm
        )
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=fake),
        ):
            props = _device_props(torch.device("cuda"), "cuda")
        self.assertEqual(props["sm_count"], 132)
        self.assertEqual(props["max_threads_per_block"], 1024)
        self.assertIsNone(props["warp_size"])
        self.assertIsNone(props["max_threads_per_sm"])
        self.assertIsNone(props["shared_mem_per_block"])
        self.assertIsNone(props["regs_per_multiprocessor"])
        self.assertIsNone(props["total_mem"])
        self.assertIsNone(props["l2_cache_size"])

    def test_versions_has_exactly_one_backend_key_by_kind(self) -> None:
        """``versions`` adds exactly one backend key named by device_kind (value may
        be None), and none for cpu/unknown -- the count is consistent across backends."""
        base = {"torch", "triton", "helion"}
        cases = (("cuda", "cuda"), ("rocm", "hip"), ("xpu", "xpu"), ("tpu", "xla"))
        for kind, key in cases:
            with self.subTest(kind=kind):
                self.assertEqual(set(_hardware_versions(kind)) - base, {key})
        self.assertEqual(set(_hardware_versions("cpu")), base)
        self.assertEqual(set(_hardware_versions(None)), base)

    def test_device_id_sanitizes_name(self) -> None:
        """device_id stays kind:name:cap-parseable: spaces and reserved punctuation
        in the device name collapse to ``_``; a missing name yields None."""
        self.assertEqual(
            _device_id("cuda", "NVIDIA H100", "sm90"), "cuda:NVIDIA_H100:sm90"
        )
        # ':' , ',' , '(' , ')' would break the kind:name:cap split -> sanitized.
        self.assertEqual(
            _device_id("xpu", "Mystery:X1, Rev(2)", "none"),
            "xpu:Mystery_X1_Rev_2_:none",
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

    def test_excluded_from_run_id(self) -> None:
        """Identical identity fields with different hardware_info share one run_id --
        hardware_info is descriptive, not identity."""
        with patch(
            "helion.autotuner._metadata.hardware.collect_hardware_info"
        ) as collect:
            collect.return_value = {"device_name": "GPU-A"}
            meta_a = _metadata()
            run_id_a, hw_a = meta_a.run_id, meta_a.to_dict()["hardware_info"]
            collect.return_value = {"device_name": "GPU-B"}
            meta_b = _metadata()
            run_id_b, hw_b = meta_b.run_id, meta_b.to_dict()["hardware_info"]
        self.assertEqual(run_id_a, run_id_b)
        self.assertNotEqual(hw_a, hw_b)


class TestAutotuneLogSink(TestCase):
    def test_dataset_logged_when_enabled(self) -> None:
        """Device-free schema check verifying that the sink writes valid CSV and JSONL records.
        Ensures rows join via config_id and run_id. Flag integration is Triton/GPU-gated.
        """
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

        # The full config is written inline (compat with existing CSV consumers).
        self.assertTrue(cell("config"))
        self.assertIn("32", cell("config"))

        # One sidecar record with exactly the lean keys, including the configs map.
        self.assertEqual(set(sidecar), _SIDECAR_KEYS)
        self.assertIn("def _add_kernel", sidecar["kernel_source"])

        # The CSV row joins to its config via config_id ...
        stored = sidecar["configs"][cell("config_id")]
        cfg = helion.Config.from_json(json.dumps(stored))
        self.assertEqual(cfg.block_sizes, [32])
        # ... and to its run via run_id.
        self.assertEqual(cell("run_id"), sidecar["run_id"])


@onlyBackends(["triton"])
class TestAutotuneDatasetE2E(TestCase):
    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_autotune_writes_dataset_sidecar(self) -> None:
        """E2E test verifying ``HELION_AUTOTUNE_LOG_DETAILS=1`` writes the ``.meta.jsonl`` sidecar.
        Gated to Triton (NVIDIA/AMD/XPU/TileIR); excludes cute, pallas, and metal.
        Schema correctness is independently verified on all lanes via device-free sink tests.
        """
        # block_sizes is backend-agnostic; num_warps is triton-only, so omit it to
        # keep these configs valid on the cute lane too.
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

            @helion.kernel()  # unpinned -> a real (non-restricted) search
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

        # Flag on + non-restricted -> dataset sidecar written alongside the CSV.
        csv_path = base_path.with_suffix(".csv")
        meta_path = base_path.with_suffix(".meta.jsonl")
        self.assertTrue(csv_path.exists())
        self.assertTrue(meta_path.exists())

        # Union the per-run configs maps and run_ids from the sidecar records.
        configs_by_id: dict[str, object] = {}
        run_ids: set[str] = set()
        meta_lines = [
            line for line in meta_path.read_text().splitlines() if line.strip()
        ]
        self.assertGreater(len(meta_lines), 0)
        for line in meta_lines:
            record = json.loads(line)
            self.assertIn("kernel_source", record)
            # Structured hardware/runtime snapshot, collected lazily in to_dict().
            hw = record["hardware_info"]
            self.assertEqual(set(hw), _HARDWARE_INFO_KEYS)
            self.assertTrue(hw["device_name"])
            # GPU lanes report a real accelerator + sm_count; tolerate a triton
            # CPU-fallback / TPU / future backend without failing the schema check.
            if hw["device_kind"] in ("cuda", "rocm", "xpu"):
                self.assertIsNotNone(hw["sm_count"])
            self.assertIn("torch", hw["versions"])
            configs_by_id.update(record["configs"])
            run_ids.add(record["run_id"])
        self.assertGreater(len(configs_by_id), 0)

        # A CSV row joins to a stored config via config_id and to its run via run_id.
        rows = list(csv.reader(csv_path.read_text().splitlines()))
        header, data = rows[0], rows[1:]
        self.assertGreater(len(data), 0)
        config_id = data[0][header.index("config_id")]
        self.assertIn(config_id, configs_by_id)
        self.assertIn(data[0][header.index("run_id")], run_ids)
        decoded_config = helion.Config.from_json(json.dumps(configs_by_id[config_id]))
        self.assertIn(decoded_config.block_sizes, ([32], [64]))


@onlyBackends(["triton"])
@unittest.skipUnless(_HAS_NETWORKX, "networkx>=3.4 required")
class TestIrGraphDatasetE2E(TestCase):
    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_autotune_writes_ir_graph(self) -> None:
        """E2E: a non-restricted autotune with the dataset on writes a
        config-independent ir_graph into every .meta.jsonl line."""
        configs = [helion.Config(block_sizes=[32]), helion.Config(block_sizes=[64])]
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
            bound = add.bind(args)
            random.seed(123)
            FiniteSearch(bound, args, configs=configs).autotune()

        lines = [
            ln
            for ln in base_path.with_suffix(".meta.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        self.assertGreater(len(lines), 0)
        by_run: dict[str, str] = {}
        for line in lines:
            rec = json.loads(line)
            ir = rec["ir_graph"]
            self.assertIsNotNone(ir)
            self.assertTrue(ir["nodes"])
            self.assertIn("edges", ir)
            self.assertTrue(ir["directed"])
            blob = json.dumps(ir, sort_keys=True)
            self.assertEqual(by_run.setdefault(rec["run_id"], blob), blob)

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_ir_graph_omitted_for_restricted_search(self) -> None:
        """``ir_graph`` rides on the ``.meta.jsonl`` record, gated by
        ``collect_dataset``. A restricted (user-pinned) search writes the CSV but no
        dataset record, so no ``ir_graph`` is emitted even with the opt-in on --
        extraction is lazy and the sink is now the only gate."""
        args = (
            torch.randn([64], device=DEVICE),
            torch.randn([64], device=DEVICE),
        )
        cfgs = [helion.Config(block_sizes=[32]), helion.Config(block_sizes=[64])]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "run"
            with patch.dict(
                os.environ,
                {"HELION_AUTOTUNE_LOG": str(base), "HELION_AUTOTUNE_LOG_DETAILS": "1"},
            ):

                @helion.kernel(configs=[helion.Config(block_sizes=[32])])
                def pinned(a, b):
                    out = torch.empty_like(a)
                    for tile in hl.tile(out.size()):
                        out[tile] = a[tile] + b[tile]
                    return out

                random.seed(123)
                FiniteSearch(pinned.bind(args), args, configs=cfgs).autotune()
            # Restricted -> CSV written, but no dataset record (hence no ir_graph).
            self.assertTrue(base.with_suffix(".csv").exists())
            self.assertFalse(base.with_suffix(".meta.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
