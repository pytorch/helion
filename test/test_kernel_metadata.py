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
from helion.autotuner._metadata.hardware import _device_props
from helion.autotuner._metadata.hardware import collect_hardware_info
from helion.autotuner._metadata.hardware import describe_device
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


class TestKernelMetadataIrGraph(TestCase):
    """ir_graph rides on KernelMetadata but is excluded from run_id."""

    def test_run_id_excludes_ir_graph(self) -> None:
        base = _metadata()
        with_empty = _metadata()
        with_empty.ir_graph = {}
        with_populated = _metadata()
        with_populated.ir_graph = {"schema_version": 1, "nodes": [{"id": "g0:a"}]}
        # ir_graph (None vs {} vs populated) never changes run_id.
        self.assertEqual(base.run_id, with_empty.run_id)
        self.assertEqual(base.run_id, with_populated.run_id)
        # Sanity: run_id IS sensitive to an identity field, so the test can fail.
        changed = _metadata()
        changed.kernel_source = base.kernel_source + " # edit"
        self.assertNotEqual(base.run_id, changed.run_id)

    def test_to_dict_includes_ir_graph_none_by_default(self) -> None:
        d = _metadata().to_dict()
        self.assertIn("ir_graph", d)
        self.assertIsNone(d["ir_graph"])


class TestKernelMetadataHardwareInfo(TestCase):
    """hardware_info rides on KernelMetadata but is excluded from run_id."""

    def test_to_dict_carries_hardware_info(self) -> None:
        """hardware_info (descriptive snapshot) rides on the record verbatim."""
        hw = _sample_hardware_info(device_name="TestGPU", sm_count=99)
        meta = _metadata()
        meta.hardware_info = hw
        self.assertEqual(meta.to_dict()["hardware_info"], hw)

    def test_run_id_excludes_hardware_info(self) -> None:
        """Differing hardware_info snapshots never change run_id (descriptive only)."""
        base = _metadata()
        base.hardware_info = _sample_hardware_info(device_name="A")
        other = _metadata()
        other.hardware_info = _sample_hardware_info(device_name="B", sm_count=7)
        self.assertEqual(base.run_id, other.run_id)


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

    def test_describe_device_pairs_name_and_optional_snapshot(self) -> None:
        """describe_device derives (hardware, hardware_info) from one device; the
        snapshot is collected only when asked, and the hardware string is identical
        whether or not it is — so run_id never depends on the dataset flag."""
        device = torch.device("cpu")
        name_off, info_off = describe_device(device, collect_info=False)
        name_on, info_on = describe_device(device, collect_info=True)
        self.assertIsNone(info_off)
        self.assertIsNotNone(info_on)
        self.assertEqual(set(info_on), _HARDWARE_INFO_KEYS)
        self.assertEqual(name_off, name_on)


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
            # Structured hardware/runtime snapshot is recorded per run.
            hw = record["hardware_info"]
            self.assertIsNotNone(hw)
            self.assertTrue(hw["device_name"])
            self.assertTrue(hw["device_id"])
            self.assertIn(hw["device_kind"], ("cuda", "rocm", "xpu"))
            self.assertIsNotNone(hw["sm_count"])
            # warp_size is cuda/rocm-only; XPU has no single-value analog (null).
            if hw["device_kind"] in ("cuda", "rocm"):
                self.assertIsNotNone(hw["warp_size"])
            self.assertIsInstance(hw["versions"], dict)
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


class TestIrGraphSidecar(TestCase):
    def test_ir_graph_in_sidecar_schema(self) -> None:
        """Device-free: a provided ir_graph round-trips into the .meta.jsonl record."""
        config = helion.Config(block_sizes=[32], num_warps=4)
        ir_graph = {
            "schema_version": 1,
            "directed": True,
            "multigraph": False,
            "graph": {"num_graphs": 0},
            "nodes": [],
            "edges": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            metadata = _metadata()
            metadata.ir_graph = ir_graph
            with AutotuneLogSink(f"{tmp}/run", metadata, collect_dataset=True) as sink:
                sink.start_run()
                config_id = sink.register_config(config)
                assert config_id is not None
                sink.record(
                    AutotuneLogEntry(
                        generation=0,
                        status="ok",
                        perf_ms=1.0,
                        compile_time=0.5,
                        config_id=config_id,
                        config=config,
                    )
                )
                sink.end_run()
            sidecar = json.loads(
                sink.meta_path.read_text(encoding="utf-8").splitlines()[0]
            )
        self.assertIn("ir_graph", sidecar)
        self.assertEqual(sidecar["ir_graph"], ir_graph)


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
            self.assertEqual(ir["schema_version"], 1)
            self.assertTrue(ir["nodes"])
            self.assertIn("edges", ir)
            self.assertTrue(ir["directed"])
            blob = json.dumps(ir, sort_keys=True)
            self.assertEqual(by_run.setdefault(rec["run_id"], blob), blob)

    @skipIfRefEager("Autotuning not supported in ref eager mode")
    def test_ir_graph_gated_on_details(self) -> None:
        """``ir_graph`` is built only when a log path is set, the dataset opt-in is
        on, and the search is not restricted to user-pinned configs."""
        args = (
            torch.randn([64], device=DEVICE),
            torch.randn([64], device=DEVICE),
        )
        cfgs = [helion.Config(block_sizes=[32]), helion.Config(block_sizes=[64])]

        def make_add():
            @helion.kernel()
            def add(a, b):
                out = torch.empty_like(a)
                for tile in hl.tile(out.size()):
                    out[tile] = a[tile] + b[tile]
                return out

            return add

        with tempfile.TemporaryDirectory() as tmp:
            base = f"{tmp}/run"
            # (a) log path + details on, unrestricted -> built.
            with patch.dict(
                os.environ,
                {"HELION_AUTOTUNE_LOG": base, "HELION_AUTOTUNE_LOG_DETAILS": "1"},
            ):
                search = FiniteSearch(make_add().bind(args), args, configs=cfgs)
                search._prepare()
                self.assertIsNotNone(search._kernel_metadata.ir_graph)
                self.assertTrue(search._kernel_metadata.ir_graph["nodes"])
            # (b) details on, NO log path -> None (guards the gate's log-path term).
            with patch.dict(
                os.environ, {"HELION_AUTOTUNE_LOG_DETAILS": "1"}, clear=False
            ):
                os.environ.pop("HELION_AUTOTUNE_LOG", None)
                search = FiniteSearch(make_add().bind(args), args, configs=cfgs)
                search._prepare()
                self.assertIsNone(search._kernel_metadata.ir_graph)
            # (c) restricted (user-pinned) search -> None.
            with patch.dict(
                os.environ,
                {"HELION_AUTOTUNE_LOG": base, "HELION_AUTOTUNE_LOG_DETAILS": "1"},
            ):

                @helion.kernel(configs=[helion.Config(block_sizes=[32])])
                def pinned(a, b):
                    out = torch.empty_like(a)
                    for tile in hl.tile(out.size()):
                        out[tile] = a[tile] + b[tile]
                    return out

                search = FiniteSearch(pinned.bind(args), args, configs=cfgs)
                search._prepare()
                self.assertIsNone(search._kernel_metadata.ir_graph)


if __name__ == "__main__":
    unittest.main()
