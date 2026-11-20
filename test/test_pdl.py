from __future__ import annotations

from unittest import mock

import pytest
import torch

import helion
from helion._compiler import compile_environment
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.triton_codegen import cpu_codegen as cpu_codegen
import helion.language as hl
from helion.runtime.settings import _get_backend


@helion.kernel(static_shapes=True, autotune_effort="none")
def prefetch_code_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.prefetch_code()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


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


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_in_host_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for _ in range(1):
        hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_trigger_between_roots(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    for tile in hl.tile(x.size(0)):
        # y[0] keeps the roots apart (no task of the second continues one of
        # the first).
        out[tile] = y[tile] * 2 + y[0]
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_trigger_between_loops(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * 2
    return y, out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_two_triggers_between_loops(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    hl.pdl_launch_dependents()
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * 2
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_wait_between_loops(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_wait()
    for tile in hl.tile(x.size(0)):
        y[tile] = y[tile] * 2
    return y


@helion.kernel(static_shapes=True, autotune_effort="none")
def prefetch_code_after_loops(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    for tile in hl.tile(x.size(0)):
        y[tile] = y[tile] * 2
    hl.prefetch_code()
    return y


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_trigger_after_host_loop(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    hl.pdl_wait()
    for _ in range(2):
        pass
    for tile in hl.tile(x.size(0)):
        y[tile] = x[tile] + 1
    hl.pdl_launch_dependents()
    for tile in hl.tile(x.size(0)):
        out[tile] = y[tile] * 2 + y[0]
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def pdl_trigger_after_host_only_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for _ in range(2):
        pass
    hl.pdl_launch_dependents()
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


_X = torch.arange(256, dtype=torch.float32)


@pytest.mark.cpu_only
@skipUnlessBackends(["triton"])
@skipIfTileIR("TileIR emits no PDL ops")
@skipIfRefEager("PDL placement is checked at compile time")
def test_trigger_after_a_host_only_loop_runs_at_entry(cpu_codegen: None) -> None:
    """A host loop is no root: a trigger before the first grid loop runs at
    entry, as the pdl_launch_dependents docstring says."""
    bound = pdl_trigger_after_host_only_loop.bind((_X,))
    assert bound.env.pdl_entry == ["launch_dependents"]
    assert bound.env.pdl_exit == []


@onlyBackends(["triton"])
class TestPdl(RefEagerTestBase, TestCase):
    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
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
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_trigger_after_loops_runs_at_exit(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(pdl_exit_trigger, (x,), block_sizes=[64])

        torch.testing.assert_close(out, x + 1)
        statements = _kernel_statements(code, "pdl_exit_trigger")
        self.assertEqual(statements[0], "tl.extra.cuda.gdc_wait()")
        self.assertEqual(statements[-1], "tl.extra.cuda.gdc_launch_dependents()")

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_trigger_between_roots_runs_before_the_later_root(self) -> None:
        """A persistent megakernel triggers once per program before the second
        root's tasks (after its grid wait), not at exit."""
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            pdl_trigger_between_roots,
            (x,),
            block_sizes=[64, 64],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
        )

        torch.testing.assert_close(out, (x + 1) * 2 + 1)
        statements = _kernel_statements(code, "pdl_trigger_between_roots")
        trigger = statements.index("tl.extra.cuda.gdc_launch_dependents()")
        self.assertEqual(statements[0], "tl.extra.cuda.gdc_wait()")
        loops = [
            i for i, s in enumerate(statements) if s.startswith("for virtual_pid in")
        ]
        self.assertEqual(len(loops), 2)
        self.assertTrue(loops[0] < trigger < loops[1], statements)
        self.assertEqual(code.count("gdc_launch_dependents()"), 1)
        # A host loop is not a root: the trigger still precedes the second.
        code, out = code_and_output(
            pdl_trigger_after_host_loop,
            (x,),
            block_sizes=[64, 64],
            pid_type="persistent_blocked",
            cross_loop_pipeline="static",
        )
        torch.testing.assert_close(out, (x + 1) * 2 + 1)
        statements = _kernel_statements(code, "pdl_trigger_after_host_loop")
        trigger = statements.index("tl.extra.cuda.gdc_launch_dependents()")
        loops = [
            i for i, s in enumerate(statements) if s.startswith("for virtual_pid in")
        ]
        self.assertTrue(loops[0] < trigger < loops[1], statements)

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_trigger_between_roots_runs_at_exit_without_a_static_schedule(
        self,
    ) -> None:
        """Packet dispatch and the barrier pipeline place no trigger between
        roots: it runs once, at exit."""
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        for pipeline in ("dynamic", "barrier"):
            with self.subTest(pipeline=pipeline):
                code, out = code_and_output(
                    pdl_trigger_between_roots,
                    (x,),
                    block_sizes=[64, 64],
                    pid_type="persistent_blocked",
                    cross_loop_pipeline=pipeline,
                )
                torch.testing.assert_close(out, (x + 1) * 2 + 1)
                statements = _kernel_statements(code, "pdl_trigger_between_roots")
                self.assertEqual(
                    statements[-1], "tl.extra.cuda.gdc_launch_dependents()"
                )
                self.assertEqual(code.count("gdc_launch_dependents()"), 1)

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    @skipIfCudaCapabilityLessThan((9, 0), reason="PDL needs sm90")
    @skipIfTileIR("checks PTX that TileIR does not emit")
    def test_trigger_between_independent_loops_runs_at_exit(self) -> None:
        """Without a megakernel to place it in, a trigger between loops runs at
        exit."""
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, (y, out) = code_and_output(
            pdl_trigger_between_loops, (x,), block_sizes=[64, 64]
        )

        torch.testing.assert_close(y, x + 1)
        torch.testing.assert_close(out, x * 2)
        statements = _kernel_statements(code, "pdl_trigger_between_loops")
        self.assertEqual(statements[-1], "tl.extra.cuda.gdc_launch_dependents()")
        self.assertEqual(code.count("gdc_launch_dependents()"), 1)

    @skipIfNotCUDA()
    @skipIfRefEager("PDL codegen is unavailable in ref mode")
    def test_ops_do_nothing_before_sm90(self) -> None:
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def add_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            hl.pdl_launch_dependents()
            hl.pdl_wait()
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        with mock.patch.object(
            compile_environment, "target_device_capability", lambda device: (8, 0)
        ):
            code, out = code_and_output(add_one, (x,), block_sizes=[64])

        torch.testing.assert_close(out, x + 1)
        self.assertNotIn("gdc_", code)
        self.assertNotIn("launch_pdl", code)

    @skipIfRefEager("PDL placement is checked at compile time")
    def test_misplaced_ops_raise(self) -> None:
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        for kernel, args in (
            (pdl_wait_after_loop, (x,)),
            (pdl_wait_on_device, (x,)),
            (pdl_wait_under_if, (x, True)),
            (pdl_wait_in_host_loop, (x,)),
            (pdl_two_triggers_between_loops, (x,)),
            # Between two loops (a megakernel's) a wait raises too.
            (pdl_wait_between_loops, (x,)),
            # After the last loop no code is left to prefetch.
            (prefetch_code_after_loops, (x,)),
        ):
            with (
                self.subTest(kernel=kernel.name),
                self.assertRaises(helion.exc.PdlPlacement),
            ):
                code_and_output(kernel, args, block_sizes=[64])


class TestPrefetchCodeBackends(RefEagerTestBase, TestCase):
    def test_prefetch_code_runs_on_every_backend(self) -> None:
        # Only Triton and Gluon on sm_90+ emit the prefetch; elsewhere the call
        # compiles to nothing.
        x = torch.arange(256, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(prefetch_code_add, (x,), block_sizes=[64])
        torch.testing.assert_close(out, x + 1)
        if _get_backend() not in ("triton", "gluon"):
            # The source comments quote hl.prefetch_code(); check the emitted call.
            self.assertNotIn("helion_prefetch_code.prefetch_code()", code)
