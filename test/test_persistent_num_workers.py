from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion._compiler.compile_environment import ConfigValueExpression
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import is_xpu
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner import EnumFragment
from helion.autotuner.config_spec import GRF_MODE_KEY
import helion.language as hl
from helion.language.persistent_ops import computes_persistent_num_workers
from helion.runtime.triton.launcher import get_xpu_eu_resources
from helion.runtime.triton.launcher import get_xpu_grf_modes


@helion.kernel(static_shapes=True)
def strided_sum(x: torch.Tensor) -> torch.Tensor:
    """Each worker sums every ``num_workers``-th element."""
    num_workers = hl.persistent_num_workers(x.device)
    out = torch.zeros(num_workers, device=x.device, dtype=x.dtype)
    for worker_id in hl.grid(num_workers):
        acc = hl.zeros([], dtype=x.dtype)
        for i in hl.grid(x.size(0)):
            if i % num_workers == worker_id:
                acc += x[i]
        out[worker_id] = acc
    return out


@helion.kernel(static_shapes=True)
def strided_sum_reserved(x: torch.Tensor) -> torch.Tensor:
    num_workers = hl.persistent_num_workers(x.device, reserved_sms=2)
    out = torch.zeros(num_workers, device=x.device, dtype=x.dtype)
    for worker_id in hl.grid(num_workers):
        out[worker_id] = x[worker_id]
    return out


def _strided_sum_reference(x: torch.Tensor, num_workers: int) -> torch.Tensor:
    return torch.stack([x[w::num_workers].sum() for w in range(num_workers)])


class TestPersistentNumWorkers(TestCase):
    def test_computes_persistent_num_workers(self) -> None:
        self.assertTrue(computes_persistent_num_workers(torch.device("xpu", 0)))
        self.assertFalse(computes_persistent_num_workers(torch.device("cuda", 0)))
        self.assertFalse(computes_persistent_num_workers(torch.device("cpu")))

    def test_config_value_expressions(self) -> None:
        m = ConfigValueExpression("config", ("m",))
        config = helion.Config(m=3)
        self.assertEqual(ConfigValueExpression("mul", (32, m)).evaluate(config), 96)
        self.assertEqual(
            ConfigValueExpression("floordiv", (32, m)).evaluate(config), 10
        )
        self.assertEqual(ConfigValueExpression("max", (1, m)).evaluate(config), 3)
        self.assertEqual(ConfigValueExpression("min", (1, m)).evaluate(config), 1)

    def test_programs_per_xecore_formula(self) -> None:
        # An EU with 10 threads and 1280 GRF registers, split in 32-register steps.
        with patch(
            "helion.runtime.triton.launcher.get_xpu_eu_resources",
            return_value=(8, 10, 1280),
        ):
            programs = helion.runtime.get_xpu_programs_per_xecore
            device = torch.device("xpu", 0)
            # Thread-limited: 1280 // 96 = 13 threads fit, the EU holds 10.
            self.assertEqual(programs(device, num_warps=8, grf_mode=96), 10)
            # Register-limited: 1280 // 160 = 8 threads per EU.
            self.assertEqual(programs(device, num_warps=8, grf_mode=160), 8)
            self.assertEqual(programs(device, num_warps=16, grf_mode=512), 1)

    @unittest.skipIf(is_xpu(), "XPU computes workers per Xe-core")
    @skipIfRefEager("inspects compiled code")
    @onlyBackends(["triton"])
    def test_other_devices_one_worker_per_sm(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        bound = strided_sum.bind((x,))
        self.assertNotIn(GRF_MODE_KEY, bound.config_spec.backend_tunable_fragments)
        code = bound.to_triton_code(helion.Config())
        self.assertIn(
            "num_workers = helion.runtime.get_num_sm(x.device, reserved_sms=0)\n",
            code,
        )
        self.assertNotIn("grf_mode", code)
        num_sm = helion.runtime.get_num_sm(x.device)
        self.assertIn(f"({num_sm},)", code)
        kernel = helion.kernel(
            strided_sum.fn, config=helion.Config(), static_shapes=True
        )
        torch.testing.assert_close(kernel(x), _strided_sum_reference(x, num_sm))

    @unittest.skipUnless(is_xpu(), "Xe-core residency is XPU only")
    def test_programs_per_xecore(self) -> None:
        eus, max_threads, grf_per_eu = get_xpu_eu_resources(DEVICE)
        threads = eus * max_threads
        programs = helion.runtime.get_xpu_programs_per_xecore
        self.assertEqual(grf_per_eu, max_threads * 128)
        self.assertEqual(programs(DEVICE, num_warps=8, grf_mode=128), threads // 8)
        self.assertEqual(programs(DEVICE, num_warps=8, grf_mode=256), threads // 16)
        self.assertEqual(programs(DEVICE, num_warps=threads, grf_mode=256), 1)

    @skipIfRefEager("inspects the compile environment")
    @unittest.skipUnless(is_xpu(), "Xe-core residency is XPU only")
    def test_config_value_expression_matches_runtime(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        (expression,) = strided_sum.bind((x,)).env.config_value_expressions.values()
        num_sm = helion.runtime.get_num_sm(x.device)
        for grf_mode in get_xpu_grf_modes(DEVICE):
            for num_warps in (4, 8, 16, 32):
                config = helion.Config(num_warps=num_warps, grf_mode=grf_mode)
                self.assertEqual(
                    expression.evaluate(config),
                    num_sm
                    * helion.runtime.get_xpu_programs_per_xecore(
                        DEVICE, num_warps=num_warps, grf_mode=grf_mode
                    ),
                )

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "grf_mode is XPU only")
    def test_narrows_grf_mode(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        fragment = strided_sum.bind((x,)).config_spec.backend_tunable_fragments[
            GRF_MODE_KEY
        ]
        self.assertEqual(fragment, EnumFragment(get_xpu_grf_modes(DEVICE)))

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "grf_mode is XPU only")
    def test_codegen_computes_workers(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        bound = strided_sum.bind((x,))
        code = bound.to_triton_code(helion.Config(num_warps=8, grf_mode=128))
        self.assertIn(
            "num_workers = helion.runtime.get_num_sm(x.device, reserved_sms=0) * "
            "helion.runtime.get_xpu_programs_per_xecore(x.device, num_warps=8, "
            "grf_mode=128)",
            code,
        )
        self.assertIn("num_workers: tl.constexpr", code)
        self.assertIn("(num_workers,)", code)
        self.assertIn("grf_mode='128'", code)

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "grf_mode is XPU only")
    def test_missing_key_defaults_to_256(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        code = strided_sum.bind((x,)).to_triton_code(helion.Config(num_warps=4))
        self.assertIn("num_warps=4, grf_mode=256)", code)
        self.assertIn("grf_mode='256'", code)

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "grf_mode is XPU only")
    def test_automatic_grf_mode_rejected(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        with self.assertRaisesRegex(helion.exc.InvalidConfig, "non-zero grf_mode"):
            strided_sum.bind((x,)).to_triton_code(helion.Config(grf_mode=0))

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "grf_mode is XPU only")
    def test_reserved_sms(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        code = strided_sum_reserved.bind((x,)).to_triton_code(
            helion.Config(num_warps=4, grf_mode=256)
        )
        self.assertIn("get_num_sm(x.device, reserved_sms=2) * ", code)

    @unittest.skipUnless(is_xpu(), "grf_mode is XPU only")
    def test_results(self) -> None:
        x = torch.randn(4096, device=DEVICE)
        num_sm = helion.runtime.get_num_sm(x.device)
        for num_warps, grf_mode in ((4, 128), (4, 256), (16, 256)):
            num_workers = num_sm * helion.runtime.get_xpu_programs_per_xecore(
                x.device, num_warps=num_warps, grf_mode=grf_mode
            )
            kernel = helion.kernel(
                strided_sum.fn,
                config=helion.Config(num_warps=num_warps, grf_mode=grf_mode),
                static_shapes=True,
            )
            out = kernel(x)
            self.assertEqual(out.numel(), num_workers)
            torch.testing.assert_close(out, _strided_sum_reference(x, num_workers))


if __name__ == "__main__":
    unittest.main()
