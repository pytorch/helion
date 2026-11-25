"""A Metal kernel's launches as data (``metal/export.py``): the MSL, grid,
threadgroup and arguments :func:`helion.runtime.capture_metal_launches`
records instead of compiling and dispatching them, and their form for MLX's
``mx.fast.metal_kernel`` (``MetalLaunch.to_mlx``).  Binds CPU tensors only;
``test_metal_export_mlx.py`` runs the kernels under MLX."""

from __future__ import annotations

import unittest
from unittest import mock

import pytest
import torch

import helion
from helion import exc
from helion._compiler.metal._constants import DTYPE_TO_METAL
from helion._testing import skipUnlessBackends
from helion._testing.metal import addmm_k_loop
from helion._testing.metal import mpp_only
from helion._testing.metal import scale_by_0dim
from helion._testing.metal_codegen import launch_dims
from helion._testing.metal_export import add_one
from helion._testing.metal_export import position_named
from helion._testing.metal_export import row_sum
from helion._testing.metal_export import scatter_even
from helion._testing.metal_export import shift
from helion._testing.metal_preamble import erf_of
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


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def copy_into(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        out[tile] = x[tile]
    return out


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def add_into_fresh(x: torch.Tensor) -> torch.Tensor:
    """Writes every element of an output the host fills with ``x``."""
    out = x.clone()
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def scatter_even_into(
    x: torch.Tensor, evens: torch.Tensor, copy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Writes the even positions of ``evens`` and all of ``copy``."""
    (n,) = x.shape
    bm = hl.register_block_size(16, 64)
    for g in hl.grid((n + bm - 1) // bm):
        src = g * bm + hl.arange(bm)
        v = hl.load(x, [src], extra_mask=src < n)
        hl.store(evens, [2 * src], v, extra_mask=src < n)
        hl.store(copy, [src], v, extra_mask=src < n)
    return evens, copy


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


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def add_shaped(
    x: torch.Tensor, x_shape: torch.Tensor, x_shape__shape: torch.Tensor
) -> torch.Tensor:
    """``x_shape`` is named like MLX's parameter for ``x``'s shape, and
    ``x_shape__shape`` like the one for ``x_shape``'s, once renamed."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + x_shape[tile] + x_shape__shape[tile]
    return out


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def add_grid(grid_size: torch.Tensor, _helion_T0: torch.Tensor) -> torch.Tensor:
    """An input named like MLX's ``grid_size`` attribute, and one like the
    body's first template parameter."""
    out = torch.empty_like(grid_size)
    for tile in hl.tile(grid_size.size()):
        out[tile] = grid_size[tile] + _helion_T0[tile]
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


class TestToMLX(unittest.TestCase):
    def test_buffers_become_inputs_and_outputs(self) -> None:
        x = torch.randn(40, 72)
        (launch,) = capture_metal_launches(add_one, (x,))
        k = launch.to_mlx(init_value=None)
        self.assertEqual((k.input_names, k.output_names), (("x",), ("out",)))
        self.assertIs(k.inputs[0], x)
        self.assertEqual(k.output_shapes, ((40, 72),))
        self.assertEqual(k.output_dtypes, (torch.float32,))
        gx, gy, gz = launch.grid
        bx, by, bz = launch.threadgroup
        self.assertEqual(k.grid, (gx * bx, gy * by, gz * bz))
        self.assertEqual(k.threadgroup, launch.threadgroup)

    def test_header_is_self_contained(self) -> None:
        """MLX compiles the header without torch's include path, so it holds
        the c10 headers a launch includes (for erf, which Helion carries no
        copy of), and the kernel body as a function the source calls with
        MLX's attributes."""
        x = torch.randn(1024)
        config = erf_of.bind((x,)).config_spec.default_config()
        (launch,) = capture_metal_launches(erf_of, (x,), config)
        k = launch.to_mlx(init_value=None)
        self.assertIn("#include <c10/metal/special_math.h>", launch.source)
        self.assertNotIn("#include <c10/", k.header)
        self.assertIn("namespace c10 {", k.header)
        self.assertIn(f"inline void {launch.name}_body(", k.header)
        self.assertIn(f"{launch.name}_body(_helion_buffer<float>(x)", k.source)
        self.assertIn("threadgroup_position_in_grid", k.source)
        self.assertNotIn("kernel void", k.header + k.source)

    def test_name_follows_what_mlx_compiles(self) -> None:
        """MLX caches kernels by name: two shapes, or two values of a scalar
        it bakes in, get two names."""
        x = torch.randn(64)
        (a,) = capture_metal_launches(add_one, (x,))
        (b,) = capture_metal_launches(add_one, (torch.randn(128),))
        self.assertNotEqual(
            a.to_mlx(init_value=None).name, b.to_mlx(init_value=None).name
        )
        (c,) = capture_metal_launches(shift, (x, 2.0))
        (d,) = capture_metal_launches(shift, (x, 3.0))
        self.assertEqual(c.source, d.source)
        self.assertNotEqual(
            c.to_mlx(init_value=None).name, d.to_mlx(init_value=None).name
        )
        self.assertEqual(a.to_mlx(name="mine", init_value=None).name, "mine")

    def test_scalars_are_baked(self) -> None:
        """MLX binds arrays only, so a runtime scalar is a constant of the
        header, and not an input."""
        x = torch.randn(256)
        (launch,) = capture_metal_launches(shift, (x, 2.0))
        k = launch.to_mlx(init_value=None)
        self.assertEqual(k.input_names, ("x",))
        self.assertIn("const float f = 2.0;", k.header)
        (launch,) = capture_metal_launches(shift, (x, float("inf")))
        self.assertIn(
            "const float f = INFINITY;", launch.to_mlx(init_value=None).header
        )

    def test_init_value(self) -> None:
        """The caller states MLX's one init_value, or takes it from the
        outputs as captured when they all hold one value: here the zeros of
        an output the kernel writes only in part."""
        x = torch.randn(40)
        (launch,) = capture_metal_launches(scatter_even, (x,))
        self.assertEqual(launch.to_mlx(init_value="captured").init_value, 0.0)
        args = (x, torch.full((80,), -1.0), torch.full((40,), -1.0))
        (launch,) = capture_metal_launches(scatter_even_into, args)
        self.assertEqual(launch.to_mlx(init_value="captured").init_value, -1.0)
        self.assertIsNone(launch.to_mlx(init_value=None).init_value)
        self.assertEqual(launch.to_mlx(init_value=7.0).init_value, 7.0)

    @unittest.skipUnless(
        torch.uint32 in DTYPE_TO_METAL, "no uint32 under this PyTorch's Metal"
    )
    def test_captured_unsigned_fill(self) -> None:
        """An unsigned output's fill, whose sign bit torch.signbit reads too."""
        n = torch.arange(40, dtype=torch.int32).to(torch.uint32)
        fill = (torch.full((80,), 5), torch.full((40,), 5))
        args = (n, *(t.to(torch.uint32) for t in fill))
        (launch,) = capture_metal_launches(scatter_even_into, args)
        self.assertEqual(launch.to_mlx(init_value="captured").init_value, 5.0)

    def test_init_value_refusals(self) -> None:
        """ "captured" refuses outputs that one float32 init_value cannot
        reproduce: an output that holds several values (data the kernel
        keeps, or what torch.empty held: a capture cannot tell them apart;
        zeros of both signs count as two values), two fills, or an integer
        float32 does not hold.  A caller that knows the fill states it."""
        x = torch.randn(40)
        n = torch.arange(40, dtype=torch.int32)
        big = 2**24 + 1
        for kernel, args, reason in (
            (
                scatter_even_into,
                (x, torch.arange(80, dtype=torch.float32), torch.zeros(40)),
                "several values",
            ),
            (add_into_fresh, (torch.randn(64),), "several values"),
            (
                scatter_even_into,
                (x, torch.tensor([0.0, -0.0] * 40), torch.zeros(40)),
                "several values",
            ),
            (
                scatter_even_into,
                (x, torch.zeros(80), torch.ones(40)),
                "one for all outputs",
            ),
            (
                scatter_even_into,
                (x, torch.zeros(80), torch.full((40,), -0.0)),
                "one for all outputs",
            ),
            (
                scatter_even_into,
                (
                    n,
                    torch.full((80,), big, dtype=torch.int32),
                    torch.full((40,), big, dtype=torch.int32),
                ),
                "float32",
            ),
        ):
            with self.subTest(kernel=kernel.name, reason=reason):
                (launch,) = capture_metal_launches(kernel, args)
                with self.assertRaisesRegex(exc.BackendUnsupported, reason):
                    launch.to_mlx(init_value="captured")
                self.assertIsNone(launch.to_mlx(init_value=None).init_value)
                self.assertEqual(launch.to_mlx(init_value=7.0).init_value, 7.0)

    def test_read_only_inputs_may_share_storage(self) -> None:
        """MLX passes each array as its own buffer, which only matters for
        a buffer the kernel writes."""
        x = torch.randn(64)
        (launch,) = capture_metal_launches(add, (x, x))
        self.assertEqual(launch.to_mlx(init_value=None).input_names, ("x", "y"))

    def test_0dim_input(self) -> None:
        """MLX binds a 0-dim input by reference, which the header's buffer
        cast takes too."""
        args = (torch.randn(64), torch.tensor(2.0))
        (launch,) = capture_metal_launches(scale_by_0dim, args)
        k = launch.to_mlx(init_value=None)
        self.assertEqual(k.input_names, ("x", "s"))
        self.assertEqual(k.inputs[1].shape, ())
        self.assertIn("_helion_buffer<float>(s)", k.source)
        self.assertIn("_helion_buffer(const constant P& p)", k.header)

    def test_threadgroup_arrays_stay_in_kernel_scope(self) -> None:
        """Threadgroup arrays (here a reduction's scratch) are declared in the
        source, where MLX's kernel scope is, and passed to the body by
        reference."""
        (launch,) = capture_metal_launches(row_sum, (torch.randn(16, 256),))
        k = launch.to_mlx(init_value=None)
        self.assertTrue(launch.msl.scratch)
        for dtype, name, slots in launch.msl.scratch:
            self.assertIn(f"threadgroup {dtype} {name}[{slots}];", k.source)
            self.assertIn(f"threadgroup {dtype} (&{name})[{slots}]", k.header)

    def test_names_mlx_declares_take_a_suffix(self) -> None:
        """MLX declares a position attribute, and an input's shape, strides
        and rank, as parameters of their own names: a buffer of such a name
        takes a ``_`` suffix in MLX's signature, and keeps its own in the
        body, renamed again when a renamed buffer's own shape takes a name.
        The body's template parameters begin as no buffer's name does."""
        (launch,) = capture_metal_launches(position_named, (torch.arange(64.0),))
        k = launch.to_mlx(init_value=None)
        self.assertEqual(k.input_names, ("threadgroup_position_in_grid_",))
        self.assertIn("_helion_buffer<float>(threadgroup_position_in_grid_)", k.source)
        self.assertIn(", threadgroup_position_in_grid,", k.source)
        self.assertIn("_helion_T0 threadgroup_position_in_grid,", k.header)
        (launch,) = capture_metal_launches(add_shaped, (torch.ones(64),) * 3)
        k = launch.to_mlx(init_value=None)
        self.assertEqual(sorted(k.input_names), ["x", "x_shape_", "x_shape__shape_"])
        (launch,) = capture_metal_launches(add_grid, (torch.ones(64),) * 2)
        k = launch.to_mlx(init_value=None)
        self.assertEqual(sorted(k.input_names), ["_helion_T0", "grid_size_"])
        self.assertNotIn("_helion_T0 _helion_T0", k.header)
        self.assertIn("typename _helion_T_0", k.header)

    def test_refusals(self) -> None:
        """What MLX's arrays cannot carry: a buffer read and written (MLX's
        outputs are fresh), buffers that share storage, and strides."""
        x = torch.randn(64)
        for kernel, args, reason in (
            (double_in_place, (x,), "read and written"),
            (add_into, (torch.randn(64), x), "read and written"),
            (copy_into, (x, x), "shares storage"),
            (add_one, (torch.randn(16, 32).t(),), "strides"),
        ):
            with self.subTest(kernel=kernel.name, reason=reason):
                (launch,) = capture_metal_launches(kernel, args)
                with self.assertRaisesRegex(exc.BackendUnsupported, reason):
                    launch.to_mlx(init_value=None)
        (launch,) = capture_metal_launches(add_one, (torch.randn(64, 2)[:, 0],))
        with self.assertRaisesRegex(exc.BackendUnsupported, "strides"):
            launch.to_mlx(init_value=None)

    def test_small_mpp_operand_is_refused(self) -> None:
        """MLX binds an input of fewer than 8 elements in constant memory,
        which an MPP device view cannot read: a 2x2 matmul's operands are
        refused, a 16x16 one's are not."""
        for size, refused in ((2, True), (16, False)):
            args = (torch.randn(size, size), torch.randn(size, size))
            config = mpp_only(addmm_k_loop, args, None)
            (launch,) = capture_metal_launches(addmm_k_loop, args, config)
            with self.subTest(size=size):
                self.assertEqual(launch.msl.device_views, {"x", "y"})
                if refused:
                    with self.assertRaisesRegex(exc.BackendUnsupported, "constant"):
                        launch.to_mlx(init_value=None)
                else:
                    launch.to_mlx(init_value=None)


if __name__ == "__main__":
    unittest.main()
