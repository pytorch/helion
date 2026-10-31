from __future__ import annotations

import unittest

import torch

import helion
from helion._compiler.compile_environment import ConfigValueExpression
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import is_xpu
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion.autotuner import PowerOfTwoFragment
from helion.autotuner.config_spec import PERSISTENT_NUM_WORKERS_MULTIPLIER_KEY
import helion.language as hl
from helion.language.tunable_ops import tunes_persistent_num_workers


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
    def test_tunes_persistent_num_workers(self) -> None:
        self.assertTrue(tunes_persistent_num_workers(torch.device("xpu", 0)))
        self.assertFalse(tunes_persistent_num_workers(torch.device("cuda", 0)))
        self.assertFalse(tunes_persistent_num_workers(torch.device("cpu")))

    def test_mul_config_value_expression(self) -> None:
        expression = ConfigValueExpression(
            "mul", (32, ConfigValueExpression("config", ("m",)))
        )
        self.assertEqual(expression.evaluate(helion.Config(m=3)), 96)

    @unittest.skipIf(is_xpu(), "XPU tunes the worker multiplier")
    @skipIfRefEager("inspects compiled code")
    @onlyBackends(["triton"])
    def test_other_devices_one_worker_per_sm(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        bound = strided_sum.bind((x,))
        self.assertNotIn(
            PERSISTENT_NUM_WORKERS_MULTIPLIER_KEY,
            bound.config_spec.user_defined_tunables,
        )
        code = bound.to_triton_code(helion.Config())
        self.assertIn(
            "num_workers = helion.runtime.get_num_sm(x.device, reserved_sms=0)\n",
            code,
        )
        num_sm = helion.runtime.get_num_sm(x.device)
        self.assertIn(f"({num_sm},)", code)
        kernel = helion.kernel(
            strided_sum.fn, config=helion.Config(), static_shapes=True
        )
        torch.testing.assert_close(kernel(x), _strided_sum_reference(x, num_sm))

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "worker multiplier is XPU only")
    def test_registers_tunable(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        fragment = strided_sum.bind((x,)).config_spec.user_defined_tunables[
            PERSISTENT_NUM_WORKERS_MULTIPLIER_KEY
        ]
        self.assertEqual(fragment, PowerOfTwoFragment(1, 16, 1))

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "worker multiplier is XPU only")
    def test_codegen_scales_workers(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        bound = strided_sum.bind((x,))
        code = bound.to_triton_code(helion.Config(persistent_num_workers_multiplier=4))
        self.assertIn(
            "num_workers = helion.runtime.get_num_sm(x.device, reserved_sms=0) * 4",
            code,
        )
        self.assertIn("num_workers: tl.constexpr", code)
        self.assertIn("(num_workers,)", code)

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "worker multiplier is XPU only")
    def test_missing_key_defaults_to_one(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        code = strided_sum.bind((x,)).to_triton_code(helion.Config())
        self.assertIn("get_num_sm(x.device, reserved_sms=0) * 1", code)

    @skipIfRefEager("inspects compiled code")
    @unittest.skipUnless(is_xpu(), "worker multiplier is XPU only")
    def test_reserved_sms(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        code = strided_sum_reserved.bind((x,)).to_triton_code(
            helion.Config(persistent_num_workers_multiplier=2)
        )
        self.assertIn("get_num_sm(x.device, reserved_sms=2) * 2", code)

    @unittest.skipUnless(is_xpu(), "worker multiplier is XPU only")
    def test_results(self) -> None:
        x = torch.randn(4096, device=DEVICE)
        num_sm = helion.runtime.get_num_sm(x.device)
        for multiplier in (1, 2, 4):
            kernel = helion.kernel(
                strided_sum.fn,
                config=helion.Config(persistent_num_workers_multiplier=multiplier),
                static_shapes=True,
            )
            out = kernel(x)
            self.assertEqual(out.numel(), num_sm * multiplier)
            torch.testing.assert_close(
                out, _strided_sum_reference(x, num_sm * multiplier)
            )


if __name__ == "__main__":
    unittest.main()
