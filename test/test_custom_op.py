from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfPallas
from helion._testing import skipIfRefEager
import helion.language as hl


@onlyBackends(["triton", "cute"])
class TestCustomOp(RefEagerTestBase, TestCase):
    def test_custom_op(self):
        """Test directly registering a helion kernel as PyTorch custom op"""

        @torch.library.custom_op("testlib::sub_one", mutates_args=())
        @helion.kernel(autotune_effort="none")
        def sub_one(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] - 1.0
            return out

        @sub_one.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        x = torch.randn([64, 64], device=DEVICE)
        expected = x - 1.0

        # Test calling via torch.ops.testlib.sub_one (registry access)
        result_registry = torch.ops.testlib.sub_one(x)
        torch.testing.assert_close(result_registry, expected)

        # Verify the direct call also works (backwards compatibility)
        result_direct = sub_one(x)
        torch.testing.assert_close(result_direct, expected)


def _k_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _k_add_scale(
    x: torch.Tensor, y: torch.Tensor, scale: float
) -> tuple[torch.Tensor, int, torch.Tensor]:
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out_x[tile] = x[tile] + y[tile] * scale
        out_y[tile] = out_x[tile] * 2.0
    return out_x, 42, out_y


def _make_fake_tensors(shape):
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    with mode:
        sizes = []
        for _ in shape:
            s = shape_env.create_unbacked_symint()
            torch._check(s >= 2)
            torch._check(s <= 4096)
            sizes.append(s)
        x = torch.empty(sizes, dtype=torch.float16, device=torch.device("cpu"))
        y = torch.empty(sizes, dtype=torch.float16, device=torch.device("cpu"))
    return x, y, mode


def _bind_and_run_fake(kernel, x, y):
    bound = kernel.bind((x, y))
    cfg = bound.config_spec.default_config()
    compiled = bound.compile_config(cfg)
    return compiled(x, y, _launcher=lambda *a, **kw: None)


def _make_backed_fake_tensors(*shapes):
    """Create FakeTensors with backed symbolic sizes from an outer ShapeEnv.

    Uses DimDynamic.DYNAMIC to force all dimensions to be symbolic,
    matching the behavior of make_fx(tracing_mode="symbolic").
    """
    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)
    tensors = []
    with mode:
        for shape in shapes:
            sym_sizes = []
            for j, val in enumerate(shape):
                source = torch._dynamo.source.TensorPropertySource(
                    torch._dynamo.source.ConstantSource(f"t{len(tensors)}"),
                    torch._dynamo.source.TensorProperty.SIZE,
                    j,
                )
                sym = shape_env.create_symbol(
                    val, source, dynamic_dim=DimDynamic.DYNAMIC
                )
                sym_sizes.append(
                    shape_env.create_symintnode(sym, hint=val, source=source)
                )
            tensors.append(
                torch.empty(sym_sizes, dtype=torch.float16, device=torch.device("cpu"))
            )
    return tensors, mode


class TestInferFakeImpl(TestCase):
    @skipIfRefEager("compile_config requires host_function")
    @skipIfPallas("fake launcher incompatible with output-only return capture")
    def test_static_shapes(self):
        k = helion.kernel(static_shapes=True, autotune_effort="none")(_k_add)
        x, y, mode = _make_fake_tensors((4, 8))
        with mode:
            result = _bind_and_run_fake(k, x, y)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.dtype, x.dtype)

    @skipIfRefEager("compile_config requires host_function")
    @skipIfPallas("fake launcher incompatible with output-only return capture")
    def test_unbacked_symints(self):
        k = helion.kernel(static_shapes=False, autotune_effort="none")(_k_add)
        x, y, mode = _make_fake_tensors((4, 8))
        with mode:
            result = _bind_and_run_fake(k, x, y)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.dtype, x.dtype)

    @skipIfRefEager("compile_config requires host_function")
    @skipIfPallas("fake launcher incompatible with output-only return capture")
    def test_backed_symints(self):
        k = helion.kernel(static_shapes=False, autotune_effort="none")(_k_add)
        (x, y), mode = _make_backed_fake_tensors((7, 13), (7, 13))
        with mode:
            result = _bind_and_run_fake(k, x, y)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.dtype, x.dtype)

    @skipIfRefEager("compile_config requires host_function")
    @skipIfPallas("fake launcher incompatible with output-only return capture")
    def test_backed_symints_shared_dim(self):
        @helion.kernel(static_shapes=False, autotune_effort="none")
        def k_square(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + 1.0
            return out

        (x,), mode = _make_backed_fake_tensors((8, 8))
        with mode:
            bound = k_square.bind((x,))
            cfg = bound.config_spec.default_config()
            compiled = bound.compile_config(cfg)
            result = compiled(x, _launcher=lambda *a, **kw: None)
            self.assertEqual(result.shape, x.shape)

    @skipIfRefEager("infer_output_spec requires host_function")
    def test_infer_output_spec_remaps_symbols(self):
        from helion._compiler._dynamo.variables import infer_output_spec

        k = helion.kernel(static_shapes=False, autotune_effort="none")(_k_add)
        (x, y), mode = _make_backed_fake_tensors((7, 13), (7, 13))
        with mode:
            spec = infer_output_spec(k, (x, y))
        leaf_specs = spec["leaf_specs"]
        self.assertEqual(len(leaf_specs), 1)
        out_shape = leaf_specs[0]["shape"]
        # Output shape should reference the outer (caller's) SymInts,
        # not helion's internal ones.
        for out_s, inp_s in zip(out_shape, x.shape, strict=True):
            if isinstance(inp_s, torch.SymInt):
                self.assertIsInstance(out_s, torch.SymInt)
                self.assertIs(out_s.node.shape_env, inp_s.node.shape_env)

    @skipIfRefEager("infer_output_spec requires host_function")
    def test_infer_output_spec_multi_output(self):
        from helion._compiler._dynamo.variables import infer_output_spec

        k = helion.kernel(static_shapes=False, autotune_effort="none")(_k_add_scale)
        (x, y), mode = _make_backed_fake_tensors((7, 13), (7, 13))
        with mode:
            spec = infer_output_spec(k, (x, y, 0.5))
        leaf_specs = spec["leaf_specs"]
        # _k_add_scale returns (tensor, int, tensor) -> 3 leaves
        self.assertEqual(len(leaf_specs), 3)
        self.assertEqual(leaf_specs[0]["type"], "tensor")
        self.assertEqual(leaf_specs[1]["type"], "scalar")
        self.assertEqual(leaf_specs[1]["scalar_value"], 42)
        self.assertEqual(leaf_specs[2]["type"], "tensor")
        # Both output tensors should have shapes remapped to outer symbols
        for spec_entry in [leaf_specs[0], leaf_specs[2]]:
            for out_s, inp_s in zip(spec_entry["shape"], x.shape, strict=True):
                if isinstance(inp_s, torch.SymInt):
                    self.assertIsInstance(out_s, torch.SymInt)
                    self.assertIs(out_s.node.shape_env, inp_s.node.shape_env)
