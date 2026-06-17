from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

import helion
from helion._testing import TestCase
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuneLogSink
from helion.autotuner.metrics import KernelMetadata
import helion.language as hl

# Lean acceptance suite (pure-assertion, no CUDA) for the opt-in cost-model
# telemetry. The per-config <base>.csv joins to the per-run <base>.meta.jsonl via
# the content-addressed config_id; the dataset sidecar is written only when
# collection is explicitly enabled.

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

# Keys of one on-disk .meta.jsonl record: the KernelMetadata identity plus the
# run's config_id -> config map.
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


class TestAutotuneDatasetSetting(TestCase):
    def test_autotune_dataset_opt_in(self) -> None:
        """``autotune_dataset`` is off by default and enabled via the env flag."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HELION_AUTOTUNE_DATASET", None)
            self.assertFalse(helion.Settings().autotune_dataset)
        with patch.dict(os.environ, {"HELION_AUTOTUNE_DATASET": "1"}, clear=False):
            self.assertTrue(helion.Settings().autotune_dataset)


class TestDatasetWithoutLogWarning(TestCase):
    def test_warns_without_log_once(self) -> None:
        """Opting into the dataset without a log path warns exactly once."""
        import helion.autotuner.base_search as base_search

        base_search._warn_dataset_without_log.cache_clear()
        log = MagicMock()
        base_search._warn_dataset_without_log(log)
        base_search._warn_dataset_without_log(log)
        self.assertEqual(log.warning.call_count, 1)


class TestAutotuneLogSink(TestCase):
    def test_dataset_logged_when_enabled(self) -> None:
        """Enabling collection writes the lean CSV and the .meta.jsonl sidecar; a
        CSV row joins to its full config via config_id and to its run via run_id."""
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


if __name__ == "__main__":
    unittest.main()
