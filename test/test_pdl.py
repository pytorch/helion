from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_launch_dependents()
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_exit_trigger(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_after_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    hl.pdl_wait()
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_on_device(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        hl.pdl_wait()
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_under_if(x: torch.Tensor, wait: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    if wait:
        hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


def _kernel_statements(code: str, kernel: str) -> list[str]:
    body = code[code.index("def _helion_") : code.index(f"\ndef {kernel}(")]
    return [
        line.strip()
        for line in body.splitlines()[1:]
        if line.strip() and not line.strip().startswith("#")
    ]


@onlyBackends(["triton"])
class TestPdl(RefEagerTestBase, TestCase):
    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    def test_entry_ops_run_in_source_order(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(pdl_add, (x,), block_sizes=[64])

        torch.testing.assert_close(out, x + 1)
        self.assertIn("launch_pdl=True", code)
        statements = _kernel_statements(code, "pdl_add")
        self.assertEqual(
            statements[:2],
            [
                "tl.extra.cuda.gdc_launch_dependents()",
                "tl.extra.cuda.gdc_wait()",
            ],
        )
        self.assertEqual(code.count("gdc_"), 2)

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    def test_trigger_after_loops_runs_at_exit(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(pdl_exit_trigger, (x,), block_sizes=[64])

        torch.testing.assert_close(out, x + 1)
        statements = _kernel_statements(code, "pdl_exit_trigger")
        self.assertEqual(statements[0], "tl.extra.cuda.gdc_wait()")
        self.assertEqual(statements[-1], "tl.extra.cuda.gdc_launch_dependents()")

    @skipIfRefEager("PDL placement is checked at compile time")
    def test_misplaced_ops_raise(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        for kernel, args in (
            (pdl_wait_after_loop, (x,)),
            (pdl_wait_on_device, (x,)),
            (pdl_wait_under_if, (x, True)),
        ):
            with (
                self.subTest(kernel=kernel.name),
                self.assertRaises(helion.exc.PdlPlacement),
            ):
                code_and_output(kernel, args, block_sizes=[64])
