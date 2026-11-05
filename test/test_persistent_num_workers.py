from __future__ import annotations

import functools
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import torch
import triton

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


@helion.kernel(static_shapes=True)
def column_sums(x: torch.Tensor) -> torch.Tensor:
    """The column reduction across sub-groups uses shared local memory."""
    num_workers = hl.persistent_num_workers(x.device)
    out = torch.zeros(num_workers, x.size(1), device=x.device, dtype=x.dtype)
    for worker_id in hl.grid(num_workers):
        for tile_n in hl.tile(x.size(1)):
            out[worker_id, tile_n] = x[:, tile_n].sum(0) + worker_id
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
        # An Xe2 Xe-core: 8 EUs of 8 threads at 128 GRF, 1024 GRF per EU.
        with patch(
            "helion.runtime.triton.launcher.get_xpu_eu_resources",
            return_value=(8, 8, 1024),
        ):
            programs = functools.partial(
                helion.runtime.get_xpu_programs_per_xecore, torch.device("xpu", 0)
            )
            self.assertEqual(programs(num_warps=8, grf_mode=128), 8)
            self.assertEqual(programs(num_warps=8, grf_mode=256), 4)
            self.assertEqual(programs(num_warps=128, grf_mode=256), 1)
            # 4 warps fit 16 times by threads; 16 KB of SLM fits 8 times in 128 KB.
            self.assertEqual(programs(num_warps=4, grf_mode=128, shared=16384), 8)
            # 20 KB is allocated as 24 KB.
            self.assertEqual(programs(num_warps=4, grf_mode=128, shared=20480), 5)
            self.assertEqual(programs(num_warps=4, grf_mode=128, shared=1), 16)
            self.assertEqual(programs(num_warps=1, grf_mode=128, shared=131072), 1)

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
    def test_kernel_shared_before_compile(self) -> None:
        @triton.jit
        def never_compiled(x):
            pass

        self.assertEqual(
            helion.runtime.get_xpu_kernel_shared(never_compiled, DEVICE), 0
        )

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
            "grf_mode=128, shared=helion.runtime.get_xpu_kernel_shared("
            "_helion_strided_sum, x.device))",
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
        self.assertIn("num_warps=4, grf_mode=256, shared=", code)
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

    @skipIfRefEager("shared local memory comes from the compiled kernel")
    @unittest.skipUnless(is_xpu(), "shared local memory is counted on XPU only")
    def test_shared_local_memory_limits_workers(self) -> None:
        x = torch.randn(256, 256, device=DEVICE)
        num_sm = helion.runtime.get_num_sm(x.device)
        kernel = helion.kernel(
            column_sums.fn,
            config=helion.Config(block_sizes=[256], num_warps=4, grf_mode=128),
            static_shapes=True,
        )
        threads_limit = num_sm * helion.runtime.get_xpu_programs_per_xecore(
            x.device, num_warps=4, grf_mode=128
        )
        # Shrink the Xe-core's SLM so that it, not threads, limits residency.
        with patch("helion.runtime.triton.launcher.XPU_SLM_PER_XECORE", 4096):
            first = kernel(x)
            second = kernel(x)
        # The first launch compiles the kernel, so only then is its SLM known.
        self.assertEqual(first.size(0), threads_limit)
        self.assertLess(second.size(0), threads_limit)
        self.assertEqual(second.size(0) % num_sm, 0)
        for out in (first, second):
            workers = torch.arange(out.size(0), device=DEVICE, dtype=x.dtype)
            torch.testing.assert_close(out, x.sum(0) + workers[:, None])

    @skipIfRefEager("exports generated code")
    @unittest.skipUnless(is_xpu(), "the XPU worker count needs runtime helpers")
    def test_standalone_runs_without_helion(self) -> None:
        x = torch.randn(1024, device=DEVICE)
        code = strided_sum.bind((x,)).to_code(
            helion.Config(num_warps=8, grf_mode=256),
            options=helion.OutputCodeOptions(allow_helion_deps=False),
        )
        self.assertNotIn("import helion", code)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "standalone.py"
            path.write_text(code)
            script = textwrap.dedent(f"""
                import importlib.util
                import sys
                import torch

                sys.modules["helion"] = None
                spec = importlib.util.spec_from_file_location("standalone", {str(path)!r})
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                x = torch.arange(1024, device="{DEVICE}", dtype=torch.float32)
                for _ in range(2):
                    print(mod.strided_sum(x).numel())
            """)
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        num_workers = helion.runtime.get_num_sm(x.device) * (
            helion.runtime.get_xpu_programs_per_xecore(
                x.device, num_warps=8, grf_mode=256
            )
        )
        self.assertEqual(result.stdout.split(), [str(num_workers)] * 2)


if __name__ == "__main__":
    unittest.main()
