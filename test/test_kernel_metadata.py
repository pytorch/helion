from __future__ import annotations

import json
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfRefEager
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.metrics import KernelMetadata
from helion.autotuner.metrics import register_kernel_metadata_hook
from helion.autotuner.metrics import register_post_autotune_hook
from helion.autotuner.metrics import remove_kernel_metadata_hook
from helion.autotuner.metrics import remove_post_autotune_hook
import helion.language as hl


@helion.kernel(config=helion.Config(block_sizes=[16]))
def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(config=helion.Config(block_sizes=[16]))
def _other_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * y[tile]
    return out


def _side_table_available() -> bool:
    try:
        from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table
    except Exception:
        return False
    return helion_kernel_side_table is not None


class TestKernelId(TestCase):
    @unittest.skipUnless(_side_table_available(), "kernel side table unavailable")
    def test_kernel_id_idempotent(self) -> None:
        first = _add_kernel.kernel_id()
        second = _add_kernel.kernel_id()
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)

    @unittest.skipUnless(_side_table_available(), "kernel side table unavailable")
    def test_kernel_id_distinct_between_kernels(self) -> None:
        self.assertNotEqual(_add_kernel.kernel_id(), _other_kernel.kernel_id())

    @unittest.skipUnless(_side_table_available(), "kernel side table unavailable")
    def test_kernel_id_matches_side_table(self) -> None:
        from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table

        idx = _add_kernel.kernel_id()
        self.assertIs(helion_kernel_side_table.get_kernel(idx), _add_kernel)

    def test_kernel_source_hash_stable_and_hex(self) -> None:
        first = _add_kernel.kernel_source_hash()
        second = _add_kernel.kernel_source_hash()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)  # raises ValueError if not hex
        self.assertNotEqual(first, _other_kernel.kernel_source_hash())


class TestMetadataSchema(TestCase):
    def test_autotune_metrics_to_dict_has_kernel_fields(self) -> None:
        record = AutotuneMetrics(
            kernel_idx=7, kernel_name="k", kernel_source_hash="abc"
        ).to_dict()
        self.assertEqual(record["kernel_idx"], 7)
        self.assertEqual(record["kernel_name"], "k")
        self.assertEqual(record["kernel_source_hash"], "abc")

    def test_kernel_metadata_to_dict_round_trip(self) -> None:
        record = KernelMetadata(
            kernel_idx=3,
            kernel_name="k",
            kernel_source_hash="abc",
            config="helion.Config(...)",
            input_shapes="[(16,)]",
            dtypes="['torch.float32']",
            hardware="TestGPU",
            path="default",
        ).to_dict()
        self.assertEqual(record["kernel_idx"], 3)
        self.assertEqual(record["path"], "default")
        # Must be JSON serializable for any downstream sink.
        json.dumps(record)


class TestMetadataHooks(TestCase):
    def test_kernel_metadata_hook_fires(self) -> None:
        seen: list[KernelMetadata] = []
        hook = seen.append
        register_kernel_metadata_hook(hook)
        try:
            from helion.autotuner.metrics import _run_kernel_metadata_hooks

            metadata = KernelMetadata(kernel_idx=1, path="default")
            _run_kernel_metadata_hooks(metadata)
        finally:
            remove_kernel_metadata_hook(hook)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].kernel_idx, 1)
        # Hook is removed: a second dispatch must not reach it.
        from helion.autotuner.metrics import _run_kernel_metadata_hooks

        _run_kernel_metadata_hooks(KernelMetadata(kernel_idx=2))
        self.assertEqual(len(seen), 1)

    def test_post_autotune_hook_fires(self) -> None:
        seen: list[AutotuneMetrics] = []
        hook = seen.append
        register_post_autotune_hook(hook)
        try:
            from helion.autotuner.metrics import _run_post_autotune_hooks

            _run_post_autotune_hooks(AutotuneMetrics(kernel_idx=5))
        finally:
            remove_post_autotune_hook(hook)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].kernel_idx, 5)


class TestEndToEndEmission(TestCase):
    @skipIfRefEager(
        "Ref eager mode runs kernels via run_ref(), which bypasses "
        "set_config()/metadata emission"
    )
    def test_set_config_emits_metadata_on_default_path(self) -> None:
        seen: list[KernelMetadata] = []
        hook = seen.append
        register_kernel_metadata_hook(hook)
        try:
            x = torch.randn(64, device=DEVICE)
            y = torch.randn(64, device=DEVICE)
            result = _add_kernel(x, y)
            torch.testing.assert_close(result, x + y)
        finally:
            remove_kernel_metadata_hook(hook)

        records = [m for m in seen if m.kernel_name == "_add_kernel"]
        self.assertTrue(records, "expected a KernelMetadata record for _add_kernel")
        record = records[-1]
        self.assertEqual(record.path, "default")
        self.assertIn("64", record.input_shapes)
        if _side_table_available():
            self.assertEqual(record.kernel_idx, _add_kernel.kernel_id())


if __name__ == "__main__":
    unittest.main()
