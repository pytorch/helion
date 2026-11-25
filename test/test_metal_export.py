"""A Metal kernel's launches as data (``metal/export.py``): the MSL, grid,
threadgroup and arguments :func:`helion.runtime.capture_metal_launches`
records instead of compiling and dispatching them.  Binds CPU tensors
only."""

from __future__ import annotations

import unittest
from unittest import mock

import pytest
import torch

import helion
from helion import exc
from helion._testing import skipUnlessBackends
from helion._testing.metal import scale_by_0dim
from helion._testing.metal_codegen import launch_dims
from helion._testing.metal_export import add_one
from helion._testing.metal_export import shift
import helion.language as hl
from helion.runtime import capture_metal_launches

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def double_in_place(x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] * 2
    return x


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def add_into(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        out[tile] = out[tile] + x[tile]
    return out


@helion.kernel(
    backend="metal",
    static_shapes=True,
    config=helion.Config(block_sizes=[8], reduction_loops=[32]),
)
def rolled_row_sum(x: torch.Tensor) -> torch.Tensor:
    """A rolled reduction, whose chunk the host passes the kernel as a
    value baked into its source."""
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[tm] = x[tm, :].sum(-1)
    return out


@helion.kernel(
    backend="metal",
    static_shapes=True,
    autotune_effort="none",
    persistent_reserved_sms=2,
)
def add_one_reserving(x: torch.Tensor) -> torch.Tensor:
    """``add_one``, reserving 2 cores from a persistent schedule."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1
    return out


def _bindings(launch: helion.runtime.MetalLaunch) -> list[tuple[str, str, bool]]:
    return [(a.name, a.binding, a.written) for a in launch.arguments]


def _argument(
    launch: helion.runtime.MetalLaunch, name: str
) -> helion.runtime.MetalArgument:
    (arg,) = [a for a in launch.arguments if a.name == name]
    return arg


class TestCaptureMetalLaunches(unittest.TestCase):
    def test_launch_matches_the_host_code(self) -> None:
        """One launch, named in its source, with the threadgroup the host
        code launches and enough threadgroups to cover every element."""
        x = torch.randn(40, 72)
        (launch,) = capture_metal_launches(add_one, (x,))
        code = add_one.bind((x,)).to_code()
        self.assertIn(f"kernel void {launch.name}(", launch.source)
        self.assertEqual(launch.threadgroup, launch_dims(code))
        self.assertIn(f"_launcher({launch.name}, {launch.grid[:1]}", code)
        self.assertEqual(
            _bindings(launch), [("x", "buffer", False), ("out", "buffer", True)]
        )
        (x_arg, out_arg) = (a.value for a in launch.arguments)
        self.assertIs(x_arg, x)
        assert isinstance(out_arg, torch.Tensor)
        self.assertEqual(out_arg.shape, x.shape)

    def test_scalar_binds_by_value(self) -> None:
        """A float the host passes as a runtime scalar binds by value, in the
        buffer slot after the tensors."""
        (launch,) = capture_metal_launches(shift, (torch.randn(256), 2.0))
        bound = [b for b in _bindings(launch) if b[1] != "baked"]
        self.assertEqual(
            bound,
            [("x", "buffer", False), ("out", "buffer", True), ("f", "scalar", False)],
        )
        self.assertIn("constant float& f [[buffer(2)]]", launch.source)
        self.assertEqual(_argument(launch, "f").value, 2.0)

    def test_0dim_tensor_is_a_buffer(self) -> None:
        args = (torch.randn(64), torch.tensor(2.0))
        (launch,) = capture_metal_launches(scale_by_0dim, args)
        s = _argument(launch, "s")
        self.assertEqual((s.binding, s.read, s.written), ("buffer", True, False))

    def test_host_constant_is_baked(self) -> None:
        """The host values a launch passes besides its tensors and runtime
        scalars (a rolled reduction's chunk; a dynamic kernel's sizes and
        strides) are baked into the source, not bound, and not written."""
        for kernel, args in (
            (rolled_row_sum, (torch.randn(16, 100),)),
            (shift, (torch.randn(256), 2.0)),
        ):
            with self.subTest(kernel=kernel.name):
                (launch,) = capture_metal_launches(kernel, args)
                constants = [a for a in launch.arguments if a.binding == "baked"]
                self.assertTrue(constants)
                for a in constants:
                    self.assertIn(f"{a.name} = {a.value};", launch.source)
                    self.assertNotIn(f" {a.name} [[buffer", launch.source)
                    self.assertFalse(a.written)

    def test_reads_writes_and_aliases(self) -> None:
        """Which buffers the kernel reads and writes, and which share
        storage with another argument."""
        x = torch.randn(64)
        (launch,) = capture_metal_launches(add_one, (x,))
        self.assertEqual(
            [(a.name, a.read, a.written) for a in launch.arguments],
            [("x", True, False), ("out", False, True)],
        )
        (launch,) = capture_metal_launches(double_in_place, (x,))
        self.assertEqual(
            [(a.name, a.read, a.written) for a in launch.arguments],
            [("x", True, True)],
        )
        (launch,) = capture_metal_launches(add_into, (x, x))
        self.assertTrue(all(a.shares_storage for a in launch.arguments))
        (launch,) = capture_metal_launches(add_into, (torch.randn(64), x))
        self.assertFalse(any(a.shares_storage for a in launch.arguments))

    def test_launches_compare_by_identity(self) -> None:
        """A launch holds tensors, so it compares as an object, not by value."""
        x = torch.randn(64)
        (a,) = capture_metal_launches(add_one, (x,))
        (b,) = capture_metal_launches(add_one, (x,))
        self.assertEqual(a, a)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a.arguments[0], b.arguments[0])

    def test_persistent_kernel_takes_a_core_count(self) -> None:
        """A persistent kernel bakes in the GPU's core count, which CPU
        tensors do not have: ``num_sm`` gives it, and the capture refuses
        one without it.  The count is the capture's own: none is left set
        after it."""
        x = torch.ones(64, 32)
        for pid_type in ("persistent_blocked", "persistent_interleaved"):
            config = helion.Config(block_sizes=[16, 16], pid_type=pid_type)
            with self.subTest(pid_type=pid_type):
                with self.assertRaisesRegex(exc.BackendUnsupported, "num_sm"):
                    capture_metal_launches(add_one, (x,), config)
                (a,) = capture_metal_launches(add_one, (x,), config, num_sm=3)
                (b,) = capture_metal_launches(add_one, (x,), config, num_sm=6)
                self.assertIn("constexpr int _NUM_SM = 3;", a.source)
                self.assertIn("constexpr int _NUM_SM = 6;", b.source)
                self.assertEqual(2 * a.grid[0], b.grid[0])
        self.assertIsNone(helion.runtime._num_sm_override.get())

    def test_core_count_less_reserved_cores(self) -> None:
        """The cores the kernel reserves come off ``num_sm``, leaving at
        least 1 as on a GPU; a count below 1 is refused."""
        x = torch.ones(64, 32)
        config = helion.Config(block_sizes=[16, 16], pid_type="persistent_blocked")
        for num_sm, count in ((5, 3), (2, 1), (1, 1)):
            with self.subTest(num_sm=num_sm):
                (launch,) = capture_metal_launches(
                    add_one_reserving, (x,), config, num_sm=num_sm
                )
                self.assertIn(f"constexpr int _NUM_SM = {count};", launch.source)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            capture_metal_launches(add_one, (x,), config, num_sm=0)

    def test_nothing_compiles(self) -> None:
        """Capturing compiles no shader, so it runs without an Apple GPU."""
        compiled = []

        def compile_shader(source: str) -> object:
            compiled.append(source)
            raise AssertionError("capture compiled a shader")

        with mock.patch.object(
            torch.mps, "compile_shader", compile_shader, create=True
        ):
            capture_metal_launches(add_one, (torch.randn(32),))
        self.assertEqual(compiled, [])


if __name__ == "__main__":
    unittest.main()
