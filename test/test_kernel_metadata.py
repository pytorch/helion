from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner.base_search import _warn_dataset_without_log
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuneLogSink
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.metrics import KernelMetadata
import helion.language as hl

# Acceptance suite for the opt-in cost-model telemetry. Most tests are device-free;
# the end-to-end autotune -> .meta.jsonl test (TestAutotuneDatasetE2E) is gated to
# the triton lanes.

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

# Keys of one on-disk .meta.jsonl record.
_SIDECAR_KEYS = {
    "run_id",
    "kernel_name",
    "kernel_source",
    "input_shapes",
    "dtypes",
    "hardware",
    "settings",
    "configs",
}


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
        """The opt-in flag with no log path warns once per logger and writes
        nothing."""
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
        """Pin the run_id codegen set and confirm every name is a real Settings
        field."""
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
        # Sorted -> stable run_id wire format.
        self.assertEqual(_CODEGEN_SETTINGS, tuple(sorted(_CODEGEN_SETTINGS)))
        settings_keys = helion.Settings().to_dict()
        for name in _CODEGEN_SETTINGS:
            self.assertIn(name, settings_keys)


class TestAutotuneMetrics(TestCase):
    def test_accepts_all_identity_fields(self) -> None:
        """``AutotuneMetrics`` accepts every kernel-identity field the call site
        passes and round-trips them through ``to_dict()`` (the original failure
        was a missing ``dtypes`` field)."""
        metrics = AutotuneMetrics(
            kernel_name="k",
            kernel_source="def k(): ...",
            input_shapes="[(64,)]",
            dtypes="['torch.float32']",
            hardware="TestGPU",
            random_seed=123,
            search_algorithm="FiniteSearch",
        )
        self.assertEqual(metrics.dtypes, "['torch.float32']")
        self.assertEqual(metrics.to_dict()["dtypes"], "['torch.float32']")

    def test_construction_kwargs_are_all_fields(self) -> None:
        """Every kwarg passed to ``AutotuneMetrics(...)`` and ``KernelMetadata(...)``
        in ``base_search.py`` must be a real dataclass field, so a call site can't
        drift ahead of the dataclass again."""
        import ast
        import dataclasses
        import inspect

        from helion.autotuner import base_search

        source = inspect.getsource(base_search)
        tree = ast.parse(source)
        targets = {
            "AutotuneMetrics": {
                f.name for f in dataclasses.fields(AutotuneMetrics)
            },
            "KernelMetadata": {f.name for f in dataclasses.fields(KernelMetadata)},
        }
        seen_calls = {name: 0 for name in targets}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in targets
            ):
                seen_calls[node.func.id] += 1
                fields = targets[node.func.id]
                for kw in node.keywords:
                    if kw.arg is None:
                        continue  # **kwargs splat
                    self.assertIn(
                        kw.arg,
                        fields,
                        f"{node.func.id}(...) in base_search.py passes "
                        f"'{kw.arg}=', which is not a dataclass field",
                    )
        # Make sure the parser found the call sites it is guarding.
        for name, count in seen_calls.items():
            self.assertGreaterEqual(count, 1, f"no {name}(...) call found to check")


class TestRunId(TestCase):
    def test_run_id_changes_with_codegen_settings(self) -> None:
        """Flipping any codegen setting changes run_id."""
        from helion.autotuner.metrics import _CODEGEN_SETTINGS

        # (base, changed) value per codegen setting. A new entry in
        # _CODEGEN_SETTINGS without a pair here raises KeyError (keeps this in sync).
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


class TestAutotuneLogSink(TestCase):
    def test_dataset_logged_when_enabled(self) -> None:
        """Device-free schema check: driving the sink with collection enabled
        writes the lean CSV header and one .meta.jsonl record, and a CSV row joins
        to its config via config_id and to its run via run_id."""
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

        # The full config is written inline.
        self.assertTrue(cell("config"))
        self.assertIn("32", cell("config"))

        # One sidecar record with exactly the lean keys.
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
        """End-to-end: with ``HELION_AUTOTUNE_LOG_DETAILS=1`` a non-restricted
        autotune writes the ``.meta.jsonl`` sidecar, and a CSV ``config_id``/
        ``run_id`` resolve in a record."""
        # block_sizes is backend-agnostic; num_warps is triton-only, so omit it.
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

            @helion.kernel()  # unpinned -> a real search
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


if __name__ == "__main__":
    unittest.main()
