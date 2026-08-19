"""Tests for Pallas kernels under ``pretuned_kernels/``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pytest
import torch

try:
    import jax

    HAS_TPU = any(device.platform == "tpu" for device in jax.devices())
except Exception:  # pragma: no cover - JAX is optional or TPU is busy
    HAS_TPU = False


_PRETUNED_KERNELS_DIR = Path(__file__).resolve().parents[1] / "pretuned_kernels"

# Median latencies measured on an otherwise-idle 8-device TPU v7x.  The gate
# allows a deliberately generous 60% slowdown because it uses absolute device
# latency rather than a simultaneously measured baseline.  It is intended to
# catch substantial regressions, not ordinary run-to-run noise.
_HSTU_EXPECTED_LATENCY_MS = {
    "s4096_b65": {"fwd": 7.5, "bwd": 22.5},
    "s3072_b5": {"fwd": 1.1, "bwd": 2.7},
    "s16384_b3": {"fwd": 3.3, "bwd": 8.8},
}
_HSTU_MAX_SLOWDOWN = 1.60


def _import_pretuned_module(name: str) -> object:
    module_name = f"_helion_pretuned_pallas_test_{name}"
    if module_name not in sys.modules:
        file_path = _PRETUNED_KERNELS_DIR / name / f"{name}.py"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


@unittest.skipUnless(HAS_TPU, "requires a real JAX TPU device")
class TestPretunedPallasMetadata(unittest.TestCase):
    """Metadata checks for pretuned Pallas kernels on a real TPU."""

    def test_jagged_hstu_fixed_config(self) -> None:
        module = _import_pretuned_module("jagged_hstu_attention")
        self.assertEqual(
            module.HSTU_CONFIG.config,
            {
                "block_sizes": [512, 512],
                "pallas_loop_type": "emit_pipeline",
                "pallas_worklist_grouping": 2,
            },
        )

    def test_jagged_hstu_static_shape_matrix(self) -> None:
        module = _import_pretuned_module("jagged_hstu_attention")
        self.assertEqual((module.NUM_HEADS, module.HEAD_DIM), (4, 128))
        actual = {
            name: (
                case.local_batch_size,
                case.declared_max_seq_len,
                case.local_tokens,
            )
            for name, case in module.SHAPE_CASES.items()
        }
        self.assertEqual(
            actual,
            {
                "s4096_b65": (65, 4096, 245760),
                "s3072_b5": (5, 4096, 18432),
                "s16384_b3": (3, 16384, 32768),
            },
        )
        for case in module.SHAPE_CASES.values():
            self.assertEqual(case.declared_max_seq_len.bit_count(), 1)
            self.assertEqual(case.local_tokens % 1024, 0)
            self.assertLessEqual(
                case.local_tokens,
                case.local_batch_size * case.declared_max_seq_len,
            )

    def test_jagged_hstu_uniform_distributions_hit_rounded_bounds(self) -> None:
        module = _import_pretuned_module("jagged_hstu_attention")
        for case in module.SHAPE_CASES.values():
            with self.subTest(case=case.name):
                lengths, targets = module.make_distribution(case, 8, "uniform")
                self.assertTrue(bool((lengths.sum(1) == case.local_tokens).all()))
                self.assertEqual(int(lengths.max()), case.declared_max_seq_len)
                self.assertTrue(bool((targets == 10).all()))

    def test_jagged_hstu_distributions_preserve_static_shapes(self) -> None:
        module = _import_pretuned_module("jagged_hstu_attention")
        for case in module.SHAPE_CASES.values():
            with self.subTest(case=case.name):
                lengths, targets = module.make_distribution(case, 8, "jagged")
                repeated_lengths, repeated_targets = module.make_distribution(
                    case, 8, "jagged"
                )
                self.assertEqual(lengths.shape, (8, case.local_batch_size))
                self.assertTrue(bool((lengths.sum(1) == case.local_tokens).all()))
                self.assertTrue(bool((targets <= lengths).all()))
                self.assertEqual(int(lengths.max()), case.declared_max_seq_len)
                self.assertGreater(int(lengths.max() - lengths.min()), 0)
                np.testing.assert_array_equal(lengths, repeated_lengths)
                np.testing.assert_array_equal(targets, repeated_targets)


@unittest.skipUnless(HAS_TPU, "requires a real JAX TPU device")
class TestPretunedPallasNumerics(unittest.TestCase):
    """Numerical checks for pretuned Pallas kernels on a real TPU."""

    @staticmethod
    def _relative_l2(actual: object, expected: torch.Tensor) -> float:
        actual_tensor = torch.from_numpy(np.asarray(actual).astype(np.float32))
        numerator = (actual_tensor - expected.float()).norm()
        denominator = expected.float().norm().clamp_min(1e-12)
        return float(numerator / denominator)

    @pytest.mark.timeout(300)
    def test_jagged_hstu_forward_and_backward(self) -> None:
        import jax
        import jax.numpy as jnp

        module = _import_pretuned_module("jagged_hstu_attention")
        torch.manual_seed(0)
        lengths = torch.tensor([17, 31, 9], dtype=torch.int32)
        offsets = torch.cat(
            (torch.zeros(1, dtype=torch.int32), torch.cumsum(lengths, dim=0))
        )
        targets = torch.tensor([2, 5, 1], dtype=torch.int32)
        shape = (int(offsets[-1]), module.NUM_HEADS, module.HEAD_DIM)
        q_cpu = torch.randn(shape, dtype=torch.float32).mul_(0.05).to(torch.bfloat16)
        k_cpu = torch.randn(shape, dtype=torch.float32).mul_(0.05).to(torch.bfloat16)
        v_cpu = torch.randn(shape, dtype=torch.float32).mul_(0.05).to(torch.bfloat16)
        grad_out_cpu = (
            torch.randn(shape, dtype=torch.float32).mul_(0.05).to(torch.bfloat16)
        )

        expected_fwd = module.reference_jagged_hstu_attention_fwd(
            q_cpu, k_cpu, v_cpu, offsets, targets
        )
        expected_bwd = module.reference_jagged_hstu_attention_bwd(
            q_cpu, k_cpu, v_cpu, grad_out_cpu, offsets, targets
        )

        q = jnp.asarray(q_cpu.float().numpy()).astype(jnp.bfloat16)
        k = jnp.asarray(k_cpu.float().numpy()).astype(jnp.bfloat16)
        v = jnp.asarray(v_cpu.float().numpy()).astype(jnp.bfloat16)
        grad_out = jnp.asarray(grad_out_cpu.float().numpy()).astype(jnp.bfloat16)
        offsets_device = jnp.asarray(offsets.numpy())
        targets_device = jnp.asarray(targets.numpy())

        actual_fwd = jax.block_until_ready(
            jax.jit(module.jagged_hstu_attention_fwd.jax_fn)(
                q, k, v, offsets_device, targets_device
            )
        )
        actual_dq = jax.block_until_ready(
            jax.jit(module.jagged_hstu_attention_bwd_dq.jax_fn)(
                q, k, v, grad_out, offsets_device, targets_device
            )
        )
        actual_dk_dv = jax.block_until_ready(
            jax.jit(module.jagged_hstu_attention_bwd_dk_dv.jax_fn)(
                q, k, v, grad_out, offsets_device, targets_device
            )
        )
        actual_bwd = (actual_dq, *actual_dk_dv)

        self.assertLess(self._relative_l2(actual_fwd, expected_fwd), 1e-2)
        for actual, expected in zip(actual_bwd, expected_bwd, strict=True):
            self.assertLess(self._relative_l2(actual, expected), 1e-2)


@unittest.skipUnless(HAS_TPU, "requires a real JAX TPU device")
class TestPretunedPallasPerformance(unittest.TestCase):
    """Performance regression checks for TPU-specific pretuned kernels."""

    @pytest.mark.timeout(300)
    def test_jagged_hstu_latency(self) -> None:
        import jax

        devices = jax.devices()
        if len(devices) != 8 or any(
            not device.device_kind.startswith("TPU v7") for device in devices
        ):
            self.skipTest("HSTU latency targets are calibrated for an 8-device TPU v7x")

        module = _import_pretuned_module("jagged_hstu_attention")
        for case_name, expected in _HSTU_EXPECTED_LATENCY_MS.items():
            with self.subTest(case=case_name):
                results = module._run_case(
                    module.SHAPE_CASES[case_name],
                    distribution="jagged",
                    kernel_names=("fwd", "dq", "dk_dv"),
                    warm_repetitions=11,
                )
                for kernel_name, result in results.items():
                    self.assertTrue(
                        bool(np.isfinite(result.checksum)),
                        f"{case_name} {kernel_name} produced a non-finite checksum",
                    )

                actual = {
                    "fwd": results["fwd"].warm_median_ms,
                    # The public backward path launches these two kernels.
                    "bwd": (
                        results["dq"].warm_median_ms
                        + results["dk_dv"].warm_median_ms
                    ),
                }
                for pass_name, actual_ms in actual.items():
                    ceiling_ms = expected[pass_name] * _HSTU_MAX_SLOWDOWN
                    self.assertLessEqual(
                        actual_ms,
                        ceiling_ms,
                        f"{case_name} {pass_name} latency {actual_ms:.3f} ms "
                        f"exceeds {ceiling_ms:.3f} ms "
                        f"(expected ~{expected[pass_name]:.3f} ms, "
                        f"slowdown allowance {_HSTU_MAX_SLOWDOWN - 1:.0%})",
                    )


if __name__ == "__main__":
    unittest.main()
