from __future__ import annotations

import unittest

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import count_triton_kernels
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
import helion.language as hl

def assert_no_graph_breaks(test_case):
    """Assert that no graph breaks occurred during compilation."""
    graph_breaks = torch._dynamo.utils.counters["graph_break"]
    test_case.assertEqual(
        len(graph_breaks),
        0,
        f"Graph breaks detected: {dict(graph_breaks)}",
    )

class TestTorchCompile(RefEagerTestDisabled, TestCase):
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_add_kernel(self):
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = add(x, y)
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_elementwise_kernel(self):
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            a = x * 2.0
            b = y + z
            result = add(a, b)
            result = result * 0.5
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), z.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), z.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inplace_mutation_with_return(self):
        """Test: kernel that mutates an input tensor in-place and returns it."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add y to x in-place and return x."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = add_inplace(x, scaled_y)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs(self):
        """Test: kernel with multiple outputs."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_and_mul(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return both add and mul results."""
            add_out = torch.empty_like(x)
            mul_out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                add_out[tile] = x[tile] + y[tile]
                mul_out[tile] = x[tile] * y[tile]
            return add_out, mul_out

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            a = x.abs()
            b = y.neg()
            add_result, mul_result = add_and_mul(a, b)
            add_result = add_result * 2.0
            mul_result = mul_result + 1.0
            add_result = torch.relu(add_result) + 1.0
            mul_result = torch.relu(mul_result) + 1.0
            return add_result, mul_result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        # Run f eagerly to get expected
        expected_add, expected_mul = f(x.clone(), y.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_add, actual_mul = compiled_f(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual_add, expected_add, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(actual_mul, expected_mul, atol=1e-3, rtol=1e-3)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs_with_input_alias(self):
        """Test: kernel with multiple outputs where one is an input alias."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def inplace_and_new(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Mutate x in-place and also return a new tensor."""
            new_out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
                new_out[tile] = x[tile] * 2.0
            return x, new_out

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            out_x, out_new = inplace_and_new(x, scaled_y)
            out_x = out_x + 1.0
            out_new = out_new - 1.0
            out_x = torch.relu(out_x) + 1.0
            out_new = torch.relu(out_new) + 1.0
            return out_x, out_new

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected_x, expected_new = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_x, actual_new = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual_x, expected_x)
        torch.testing.assert_close(actual_new, expected_new)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_tensor_and_scalar_output(self):
        """Test: kernel returning both a tensor and a scalar constant."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_with_count(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, int]:
            """Return sum tensor and a constant count."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out, 42

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, int]:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            result, count = add_with_count(scaled_x, y)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result, count

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected_sum, expected_count = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_sum, actual_count = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual_sum, expected_sum)
        self.assertEqual(actual_count, expected_count)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("arg_style", ("all_keyword", "mixed"))
    def test_keyword_arg_styles(self, arg_style):
        """Test: keyword and mixed positional/keyword argument passing."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_three(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile] + z[tile]
            return out

        def f_all_kw(x, y, z):
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = add_three(z=z, y=y, x=x) * 0.5
            result = torch.relu(result) + 1.0
            return result

        def f_mixed(x, y, z):
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = add_three(x, z=z, y=y) - 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        if arg_style == "all_keyword":
            f = f_all_kw
        else:
            f = f_mixed

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), z.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), z.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_default_params(self):
        """Test: kernel with optional parameter using default value."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def scale_add(
            x: torch.Tensor, y: torch.Tensor, scale: float = 2.0
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile] * scale
            return out

        def f_with_default(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            biased_x = x + bias
            result = scale_add(biased_x, y)
            result = result * 0.5
            result = torch.relu(result) + 1.0
            return result

        def f_with_override(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            biased_x = x + bias
            result = scale_add(biased_x, y, scale=3.0)
            result = result * 0.5
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        # Test with default - run eagerly to get expected
        expected_default = f_with_default(x.clone(), y.clone(), bias.clone())

        torch._dynamo.reset()
        compiled_default = torch.compile(
            f_with_default, fullgraph=True, backend="inductor"
        )
        actual_default = compiled_default(x.clone(), y.clone(), bias.clone())
        assert_no_graph_breaks(self)
        torch.testing.assert_close(actual_default, expected_default, rtol=1e-3, atol=1e-3)

        # Test with override - run eagerly to get expected
        expected_override = f_with_override(x.clone(), y.clone(), bias.clone())

        torch._dynamo.reset()
        compiled_override = torch.compile(
            f_with_override, fullgraph=True, backend="inductor"
        )
        actual_override = compiled_override(x.clone(), y.clone(), bias.clone())
        assert_no_graph_breaks(self)
        torch.testing.assert_close(actual_override, expected_override, rtol=1e-3, atol=1e-3)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_constant_scalar_args(self):
        """Test: kernel with scalar constant arguments."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def scale_and_shift(
            x: torch.Tensor, scale: float, shift: float
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] * scale + shift
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            combined = x + y
            result = scale_and_shift(combined, 2.5, 1.0)
            result = result - 0.5
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transposed_input(self):
        """Test: transposed (non-contiguous) tensor input."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_transposed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            a = x.T * scale
            b = y.T
            result = add_transposed(a, b)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        y = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_called_twice(self):
        """Test: same kernel called twice with different inputs."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, scale: torch.Tensor
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            a = add_k(scaled_x, y)
            b = add_k(a, z)
            result = b + 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), z.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), z.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_same_tensor_twice(self):
        """Test: same tensor passed as two different arguments."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_self(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            scaled = x * 2.0 + bias
            result = add_self(scaled, scaled)
            result = result.mean(dim=-1)
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), bias.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), bias.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_all_outputs_alias_inputs(self):
        """Test: multi-output where all outputs are input aliases."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def swap_inplace(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Swap values between x and y, return both (which alias inputs)."""
            for tile in hl.tile(x.size()):
                tmp = x[tile].clone()
                x[tile] = y[tile]
                y[tile] = tmp
            return x, y

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            scaled_y = y * scale
            out_x, out_y = swap_inplace(scaled_x, scaled_y)
            out_x = out_x + 1.0
            out_y = out_y - 1.0
            out_x = torch.relu(out_x) + 1.0
            out_y = torch.relu(out_y) + 1.0
            return out_x, out_y

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected_x, expected_y = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_x, actual_y = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual_x, expected_x)
        torch.testing.assert_close(actual_y, expected_y)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic_add_mutation(self):
        """Test: mutation via atomic operations."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def atomic_add_kernel(
            x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            """Atomically add x and y to out."""
            for tile in hl.tile(x.size()):
                hl.atomic_add(out, tile, x[tile])
                hl.atomic_add(out, tile, y[tile])
            return out

        def f(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x * 0.5
            b = y.abs()
            result = atomic_add_kernel(a, b, out)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        # Run f eagerly to get expected (need fresh out tensor)
        out_eager = torch.zeros_like(x)
        expected = f(x.clone(), y.clone(), out_eager)

        # Run compiled f
        torch._dynamo.reset()
        out_compiled = torch.zeros_like(x)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), out_compiled)
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_not_returned(self):
        """Test: kernel mutates input but returns different output."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def mutate_and_return_new(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Mutate x in-place, but return a new tensor with different computation."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
                out[tile] = x[tile] * 2.0
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = mutate_and_return_new(x, scaled_y)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_indirect_output_alias(self):
        """Test: output is a slice/view of input (indirect alias with different shape)."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def slice_and_mutate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Mutate a slice of x and return that slice (indirect alias)."""
            # Get a slice view of x
            x_slice = x[:2, :4]
            for tile in hl.tile(x_slice.size()):
                x_slice[tile] = x_slice[tile] + y[tile]
            return x_slice

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = slice_and_mutate(x, scaled_y)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_empty_tensor(self):
        """Test: tensors with zero-size dimensions."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_empty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            result = add_empty(scaled_x, y)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        # Test with zero-size first dimension
        x = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_reduction_sum(self):
        """Test: kernel with reduction dimension (sum along axis)."""

        @helion.kernel(
            config=helion.Config(block_sizes=[4]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].to(torch.float32).sum(-1).to(x.dtype)
            return out

        def f(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            weight = weight * 2.0
            scaled = x * weight
            # Helion kernel with reduction
            row_sums = row_sum(scaled)
            result = row_sums.softmax(dim=0)
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)

        # Run f eagerly to get expected
        expected = f(x.clone(), weight.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), weight.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inline_triton_mutation(self):
        """Test: kernel using inline_triton marks all inputs as potentially mutated."""

        @helion.kernel(
            config=helion.Config(block_sizes=[16]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def inline_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                x_val = x[tile]
                y_val = y[tile]
                # inline_triton could potentially mutate any input
                result = hl.inline_triton(
                    """
                    tmp = {lhs} + {rhs}
                    tmp
                    """,
                    args={"lhs": x_val, "rhs": y_val},
                    output_like=x_val,
                )
                out[tile] = result
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x.exp()
            b = y.log1p()
            # Helion kernel with inline_triton
            result = inline_add_kernel(a, b)
            result = result * 2.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()
        y = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_single_argument_kernel_mutation(self):
        """Test: kernel with only one tensor argument that is mutated (tests single-input edge case)."""

        @helion.kernel(
            config=helion.Config(block_sizes=[16]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def increment_kernel(x: torch.Tensor) -> torch.Tensor:
            """Increment each element by 1 in-place."""
            for tile in hl.tile(x.shape):
                x[tile] = x[tile] + 1.0
            return x

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            scaled = (x * 2.0 + bias).contiguous()
            # Helion kernel with single tensor mutation
            result = increment_kernel(scaled)
            result = result * 0.5
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(64, device=DEVICE, dtype=torch.float32)

        # Run f eagerly to get expected
        expected = f(x.clone(), bias.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), bias.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    def test_signal_mutation(self):
        """Test: kernel using hl.signal correctly tracks mutation of signal_pad tensor."""

        @helion.kernel(
            autotune_effort="none",
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def signal_kernel(
            signal_pad: torch.Tensor, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = torch.empty_like(x)
            (n,) = x.shape
            for i in hl.grid(n):
                hl.signal(signal_pad, [i], signal=2)
                out[i] = x[i] * 2
            return out, signal_pad

        def f(
            signal_pad: torch.Tensor, x: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Only apply prologue to float tensors, not signal_pad (int32)
            x = x * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            # Helion kernel with signal
            out, sig = signal_kernel(signal_pad, scaled_x)
            out = out + 1.0
            # Only apply epilogue to float output, not signal_pad
            out = torch.relu(out) + 1.0
            return out, sig

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(4, device=DEVICE, dtype=torch.float32)

        # Run f eagerly to get expected
        expected_out, expected_sig = f(signal_pad.clone(), x.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_out, actual_sig = compiled_f(signal_pad.clone(), x.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual_out, expected_out)
        # signal_pad should have been mutated (set to 2)
        self.assertTrue(torch.all(actual_sig == 2))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    def test_wait_with_update_mutation(self):
        """Test: kernel using hl.wait with update parameter tracks mutation."""

        @helion.kernel(
            autotune_effort="none",
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def wait_update_kernel(
            signal_pad: torch.Tensor, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = torch.empty_like(x)
            (n,) = x.shape
            for i in hl.grid(n):
                hl.wait(signal_pad, [i], signal=1, update=2)
                out[i] = x[i] * 2
            return out, signal_pad

        def f(
            signal_pad: torch.Tensor, x: torch.Tensor, bias: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Only apply prologue to float tensors, not signal_pad (int32)
            x = x * 2.0
            bias = bias * 2.0
            biased_x = x + bias
            # Helion kernel with wait/update
            out, sig = wait_update_kernel(signal_pad, biased_x)
            out = out - 0.5
            # Only apply epilogue to float output, not signal_pad
            out = torch.relu(out) + 1.0
            return out, sig

        signal_pad = torch.ones(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, device=DEVICE, dtype=torch.float32)

        # Run f eagerly to get expected
        expected_out, expected_sig = f(signal_pad.clone(), x.clone(), bias.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_out, actual_sig = compiled_f(signal_pad.clone(), x.clone(), bias.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual_out, expected_out)
        # signal_pad should have been mutated (updated to 2)
        self.assertTrue(torch.all(actual_sig == 2))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_view_input_mutate_different_view(self):
        """Test: input is a slice, mutate and return a different slice of the same base."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def process_slice_return_other(
            x_slice: torch.Tensor, y: torch.Tensor, x_full: torch.Tensor
        ) -> torch.Tensor:
            """Process one slice but return a different slice of the same tensor."""
            for tile in hl.tile(x_slice.size()):
                x_slice[tile] = x_slice[tile] + y[tile]
            # Return a different slice of x_full (shares storage with x_slice)
            return x_full[2:4, 4:8]

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            # Pass slice as first arg, but also pass full tensor
            result = process_slice_return_other(x[:2, :4], scaled_y, x)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_permute_view(self):
        """Test: output is permuted view of input."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def mutate_return_permuted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Mutate 3D tensor and return permuted view."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x.permute(2, 0, 1)  # (B, H, W) -> (W, B, H)

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = mutate_return_permuted(x, scaled_y)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_alias_view_as_two_args(self):
        """Test: passing x and x.view(-1) as two different arguments."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_views(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add y to x where y may be a view of x."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(a: torch.Tensor) -> torch.Tensor:
            a = a * 2.0
            x = a * 2
            y = x.view(-1)
            result = add_views(x, y)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        a = torch.randn(64, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(a.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(a.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliasing_inputs_used_after(self):
        """Test: view of input used after kernel mutation."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def inplace_add_one(x: torch.Tensor) -> torch.Tensor:
            """Add 1 to x in-place."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
            return x

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = x.view(-1)  # View before kernel
            _ = inplace_add_one(x)  # Mutate x
            result = y + 1  # Use view after - should see mutation
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(64, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_through_internal_view(self):
        """Test: kernel that creates a view inside the kernel and mutates through it."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def view_and_mutate(x: torch.Tensor) -> torch.Tensor:
            """Create view internally and mutate through it."""
            y = x.view(x.size())  # Same-shape view
            for tile in hl.tile(y.size()):
                y[tile] = y[tile] + 1
            return x  # Return original (same storage)

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            x = x - 1  # Prologue
            result = view_and_mutate(x)
            result = result * 2  # Epilogue
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)

        # Run f eagerly to get expected
        expected = f(x.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_mutated_inputs(self):
        """Test: kernel that mutates multiple input tensors independently."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def mutate_two_inputs(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            """Mutate both x and y, return a new tensor computed from z."""
            out = torch.empty_like(z)
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1.0
                y[tile] = y[tile] * 2.0
                out[tile] = z[tile] + x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = mutate_two_inputs(x, y, z)
            result = result - 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), z.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), z.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_detached_input(self):
        """Test: input is detached from grad-tracking tensor."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_tensors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y = y * 2.0
            # Detach x before passing to kernel
            x_detached = x.detach()
            result = add_tensors(x_detached, y)
            result = result * 2.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        _ = add_tensors(x.detach(), y)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_module_forward_with_kernel(self):
        """Test: Helion kernel called inside nn.Module.forward()."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def linear_add(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            """Simple element-wise linear: x * weight + bias."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * weight[tile] + bias[tile]
            return out

        class SimpleModule(torch.nn.Module):
            def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.bias = torch.nn.Parameter(bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return linear_add(x, self.weight, self.bias)

        def f(module, x):
            x = x * 2.0
            result = module(x)
            result = torch.relu(result) + 1.0
            return result

        weight = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        module = SimpleModule(weight.clone(), bias.clone())

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        # Warmup
        _ = linear_add(x, weight, bias)

        # Run f eagerly to get expected
        expected = f(module, x.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(module, x.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_chained_inplace_mutations(self):
        """Test: multiple successive in-place mutations on same tensor in one loop."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def chained_mutate(x: torch.Tensor) -> torch.Tensor:
            """Apply chained in-place mutations: x = ((x + 1) * 2) - 3."""
            for tile in hl.tile(x.size()):
                # Chain of mutations in single loop body
                val = x[tile] + 1.0
                val = val * 2.0
                val = val - 3.0
                x[tile] = val
            return x

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            biased = x + bias
            result = chained_mutate(biased)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        x_warmup = (x + bias).clone()
        _ = chained_mutate(x_warmup)

        # Run f eagerly to get expected
        expected = f(x.clone(), bias.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), bias.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate(self):
        """Test: clone tensor, mutate clone, verify original unchanged."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add y to x in-place."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = add_inplace(x_clone, y)
            # Apply epilogue only to result, not to x (which we're verifying stayed unchanged)
            result = torch.relu(result) + 1.0
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = add_inplace(x_warmup, y)

        # Run f eagerly to get expected
        expected_result, expected_original = f(x.clone(), y.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_result, actual_original = compiled_f(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual_result, expected_result)
        torch.testing.assert_close(actual_original, expected_original)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_preallocated_output(self):
        """Test: kernel fills pre-allocated output tensor passed as argument."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_into_out(
            x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
        ) -> torch.Tensor:
            """Add x and y, store result in pre-allocated out tensor."""
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x * 2.0
            b = y + 1.0
            result = add_into_out(a, b, out)
            result = result * 0.5
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        out_warmup = torch.empty_like(x)
        _ = add_into_out(x * 2.0, y + 1.0, out_warmup)

        # Run f eagerly to get expected
        out_expected = torch.empty_like(x)
        expected = f(x.clone(), y.clone(), out_expected)

        # Run compiled f
        torch._dynamo.reset()
        out_test = torch.empty_like(x)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), out_test)
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs_same_storage(self):
        """Test: multiple outputs that share the same underlying storage."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def mutate_return_views(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Mutate x and return both x and a view of x."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            # Return x and a reshaped view of x (same storage)
            return x, x.view(-1)

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            out_2d, out_1d = mutate_return_views(x, scaled_y)
            out_2d = out_2d + 1.0
            out_1d = out_1d * 2.0
            out_2d = torch.relu(out_2d) + 1.0
            out_1d = torch.relu(out_1d) + 1.0
            return out_2d, out_1d

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = mutate_return_views(x_warmup, y * scale)

        # Run f eagerly to get expected
        expected_2d, expected_1d = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_2d, actual_1d = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual_2d, expected_2d)
        torch.testing.assert_close(actual_1d, expected_1d)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliased_storage_different_shape(self):
        """Test: inputs share storage but have different shapes."""

        @helion.kernel(
            config=helion.Config(block_sizes=[4]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add two 1D tensors."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(base: torch.Tensor) -> torch.Tensor:
            base = base * 2.0
            # Create two views of base with different strides
            x = base[::2]  # Every other element: shape [16]
            y = base[1::2]  # Every other element offset by 1: shape [16]
            result = add_1d(x, y)
            result = result + 1.0
            result = torch.relu(result) + 1.0
            return result

        base = torch.randn(32, device=DEVICE, dtype=torch.float16)
        _ = add_1d(base[::2], base[1::2])

        # Run f eagerly to get expected
        expected = f(base.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(base.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_partial_tensor_mutation(self):
        """Test: mutate only a slice of tensor, rest remains unchanged."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_inplace_slice(x_slice: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add y to x_slice in-place."""
            for tile in hl.tile(x_slice.size()):
                x_slice[tile] = x_slice[tile] + y[tile]
            return x_slice

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Take a slice, mutate it, return both slice result and full tensor
            x_slice = x[:2, :4]  # First 2 rows, first 4 cols
            result = add_inplace_slice(x_slice, y)
            # Apply epilogue only to result, not to x (which shows mutation pattern)
            result = torch.relu(result) + 1.0
            return result, x  # x should have mutation in slice region

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = add_inplace_slice(x_warmup[:2, :4], y)

        # Run f eagerly to get expected
        expected_slice, expected_full = f(x.clone(), y.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_slice, actual_full = compiled_f(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual_slice, expected_slice)
        torch.testing.assert_close(actual_full, expected_full)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_output_aliases_intermediate(self):
        """Test: output aliases tensor created inside the kernel."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def create_and_return_view(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Create intermediate tensor, mutate it, return a view."""
            intermediate = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                intermediate[tile] = x[tile] + y[tile]
            # Return a view of the intermediate
            return intermediate.view(-1)

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            a = x * scale
            b = y + 1.0
            result = create_and_return_view(a, b)
            result = result * 2.0
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = create_and_return_view(x * scale, y + 1.0)

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone(), scale.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone(), scale.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inference_mode(self):
        """Test: kernel works correctly inside inference_mode context."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(x, y):
            x = x * 2.0
            y = y * 2.0
            z = x + 0.5  # prologue
            result = add_kernel(z, y)
            result = result * 2  # epilogue
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_kernel(x.clone(), y)  # warmup

        # Run f eagerly to get expected
        with torch.inference_mode():
            expected = f(x.clone(), y.clone())

        # Run compiled f
        torch._dynamo.reset()
        with torch.inference_mode():
            compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
            actual = compiled_f(x.clone(), y.clone())

        torch.testing.assert_close(actual, expected)
        assert_no_graph_breaks(self)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_identical_aliased_inputs(self):
        """Test: same tensor passed twice as different mutated arguments."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_both(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add 1 to x and 2 to y (which may alias)."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
                y[tile] = y[tile] + 2
            return x

        def f(z):
            z = z * 2.0
            # Pass same tensor as both x and y
            a = z.clone()
            result = add_both(a, a)
            # Apply epilogue only to result, not to z (which we're verifying stayed unchanged)
            result = torch.relu(result) + 1.0
            return result, z  # original should be unchanged

        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_both(z.clone(), z.clone())  # warmup

        # Run f eagerly to get expected
        expected_result, expected_orig = f(z.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual_result, actual_orig = compiled_f(z.clone())

        torch.testing.assert_close(actual_result, expected_result)
        torch.testing.assert_close(actual_orig, expected_orig)
        assert_no_graph_breaks(self)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_graph_input_is_view_with_kernel(self):
        """Test: graph input is a view, kernel operates on derived view."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(x, y):
            x = x * 2.0
            y = y * 2.0
            # x is already a view (passed from outside), take a 2D slice
            a = x[:2]  # 2D slice of view
            result = add_inplace(a.clone(), y[:2])
            result = torch.relu(result) + 1.0
            return result

        base = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x = base[1:]  # view with shape [3, 8]
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_inplace(x[:2].clone(), y[:2])  # warmup

        # Run f eagerly to get expected
        expected = f(x.clone(), y.clone())

        # Run compiled f
        torch._dynamo.reset()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(x.clone(), y.clone())

        torch.testing.assert_close(actual, expected)
        assert_no_graph_breaks(self)

    # =========================================================================
    # Mutation Tests (moved from TestMutation)
    # =========================================================================

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inplace(self):
        k_inplace.settings._wip_experimental_allow_torch_compile_fusion = True

        def fn(x):
            x = x * 2.0
            x = x * 2
            x = k_inplace(x)
            result = x + 1
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(64, device=DEVICE)
        # Warmup
        _ = k_inplace(torch.randn(64, device=DEVICE))

        # Run fn eagerly to get expected
        expected = fn(x.clone())

        # Run compiled fn
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        actual = compiled_fn(x.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_two_mutated(self):
        k_two_mutated.settings._wip_experimental_allow_torch_compile_fusion = True

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x + 1, y - 1
            x, y = k_two_mutated(x, y)
            rx, ry = x * 2, y * 2
            rx = torch.relu(rx) + 1.0
            ry = torch.relu(ry) + 1.0
            return rx, ry

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        # Warmup
        _ = k_two_mutated(
            torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        )

        # Run fn eagerly to get expected
        exp_x, exp_y = fn(x.clone(), y.clone())

        # Run compiled fn
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        rx, ry = compiled_fn(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(rx, exp_x)
        torch.testing.assert_close(ry, exp_y)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_one_mutated(self):
        k_one_mutated.settings._wip_experimental_allow_torch_compile_fusion = True

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y * 2
            x = k_one_mutated(x, y)
            result = x - 1
            result = torch.relu(result) + 1.0
            return result

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        # Warmup
        _ = k_one_mutated(
            torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        )

        # Run fn eagerly to get expected
        expected = fn(x.clone(), y.clone())

        # Run compiled fn
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        actual = compiled_fn(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mut_and_out(self):
        k_mut_and_out.settings._wip_experimental_allow_torch_compile_fusion = True

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x + 1, y + 1
            x, out = k_mut_and_out(x, y)
            x = torch.relu(x) + 1.0
            out = torch.relu(out) + 1.0
            return x, out

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        # Warmup
        _ = k_mut_and_out(
            torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        )

        # Run fn eagerly to get expected
        exp_x, exp_out = fn(x.clone(), y.clone())

        # Run compiled fn
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        rx, rout = compiled_fn(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(rx, exp_x)
        torch.testing.assert_close(rout, exp_out)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_not_returned_input(self):
        k_mut_no_return.settings._wip_experimental_allow_torch_compile_fusion = True

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x = x + 1
            out = k_mut_no_return(x, y)
            rx, rout = x + 1, out  # use mutated input after kernel
            rx = torch.relu(rx) + 1.0
            rout = torch.relu(rout) + 1.0
            return rx, rout

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        # Warmup
        _ = k_mut_no_return(
            torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        )

        # Run fn eagerly to get expected
        exp_x, exp_out = fn(x.clone(), y.clone())

        # Run compiled fn
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        rx, rout = compiled_fn(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(rx, exp_x)
        torch.testing.assert_close(rout, exp_out)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inplace_ignored_return(self):
        k_inplace.settings._wip_experimental_allow_torch_compile_fusion = True

        def fn(x):
            x = x * 2.0
            k_inplace(x)  # return ignored; still mutates x
            result = x + 1
            result = torch.relu(result) + 1.0
            return result

        x = torch.randn(64, device=DEVICE)
        # Warmup
        _ = k_inplace(torch.randn(64, device=DEVICE))

        # Run fn eagerly to get expected
        expected = fn(x.clone())

        # Run compiled fn
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        actual = compiled_fn(x.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_store(self):
        k_store.settings._wip_experimental_allow_torch_compile_fusion = True

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y + 1
            x = k_store(x, y)
            result = x - 1
            result = torch.relu(result) + 1.0
            return result

        x, y = torch.zeros(64, device=DEVICE), torch.randn(64, device=DEVICE)
        # Warmup
        _ = k_store(torch.zeros(64, device=DEVICE), torch.randn(64, device=DEVICE))

        # Run fn eagerly to get expected
        expected = fn(x.clone(), y.clone())

        # Run compiled fn
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        actual = compiled_fn(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic(self):
        """Atomic ops should prevent epilogue fusion, resulting in >1 kernels."""
        k_atomic.settings._wip_experimental_allow_torch_compile_fusion = True

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y * 2
            x = k_atomic(x, y)
            result = x + 1
            result = torch.relu(result) + 1.0
            return result

        x, y = torch.zeros(64, device=DEVICE), torch.ones(64, device=DEVICE)
        # Warmup
        _ = k_atomic(torch.zeros(64, device=DEVICE), torch.ones(64, device=DEVICE))

        # Run fn eagerly to get expected
        expected = fn(x.clone(), y.clone())

        # Run compiled fn
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        actual = compiled_fn(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_no_mutation(self):
        k_no_mut.settings._wip_experimental_allow_torch_compile_fusion = True

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x * 2, y * 2
            out = k_no_mut(x, y)
            result = out + 1
            result = torch.relu(result) + 1.0
            return result

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        # Warmup
        _ = k_no_mut(torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE))

        # Run fn eagerly to get expected
        expected = fn(x.clone(), y.clone())

        # Run compiled fn
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
        actual = compiled_fn(x.clone(), y.clone())
        assert_no_graph_breaks(self)

        torch.testing.assert_close(actual, expected)

    # =========================================================================
    # Epilogue Fusion Tests
    # =========================================================================

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_prologue_epilogue_fusion(self):
        """Prologue + epilogue: input -> sigmoid -> kernel -> relu -> add bias."""
        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, out_bias):
            # Prologue: ops before kernel
            x_processed = torch.sigmoid(x) * 1.5
            out, info = elementwise_2d_fusion(x_processed, kernel_scale)
            # Epilogue: ops after kernel
            out_processed = torch.relu(out) + out_bias
            return out_processed, info

        # Warmup
        _ = elementwise_2d_fusion(torch.sigmoid(x) * 1.5, kernel_scale)

        # Eager reference
        out_eager, info_eager = f(x, out_bias)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, out_bias)
        assert_no_graph_breaks(self)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)

        # Count kernels - with fusion enabled this should be 1
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}"
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_prologue_epilogue_chained_ops(self):
        """Prologue + epilogue with chained ops on both sides."""
        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, out_bias, out_scale):
            # Prologue: chained ops before kernel
            x_processed = torch.relu(x) + 0.1
            out, info = elementwise_2d_fusion(x_processed, kernel_scale)
            # Epilogue: chained ops after kernel
            out_relu = torch.relu(out)
            out_tanh = torch.tanh(out_relu)
            out_biased = out_tanh + out_bias
            out_scaled = out_biased * out_scale
            return out_scaled, info

        # Warmup
        _ = elementwise_2d_fusion(torch.relu(x) + 0.1, kernel_scale)

        # Eager reference
        out_eager, info_eager = f(x, out_bias, out_scale)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, out_bias, out_scale)
        assert_no_graph_breaks(self)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)

        # Count kernels - with fusion enabled this should be 1
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}"
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_single_output_prologue_epilogue(self):
        """Prologue + epilogue with single tensor output (no scalar)."""
        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0

        def f(x, out_bias):
            # Prologue: ops before kernel
            x_processed = torch.tanh(x) * 2.0
            out = elementwise_2d_single_output_fusion(x_processed, kernel_scale)
            # Epilogue: ops after kernel
            out_processed = torch.relu(out) + out_bias
            return out_processed

        # Warmup
        _ = elementwise_2d_single_output_fusion(torch.tanh(x) * 2.0, kernel_scale)

        # Eager reference
        out_eager = f(x, out_bias)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, out_bias)
        assert_no_graph_breaks(self)

        torch.testing.assert_close(result, out_eager, rtol=1e-3, atol=1e-3)

        # Count kernels - with fusion enabled this should be 1
        kernel_count, all_code = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}"
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_rms_norm_prologue_epilogue(self):
        """Prologue + epilogue fusion with multi-output RMS norm kernel."""
        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, weight, out_bias, res_bias):
            # Prologue: ops before kernel
            x_processed = torch.relu(x) + 0.5
            out, residual, info = rms_norm_multi_output(x_processed, weight)
            # Epilogue: ops after kernel (different epilogue per output)
            return torch.relu(out) + out_bias, torch.sigmoid(residual) + res_bias, info

        # Warmup
        _ = rms_norm_multi_output(torch.relu(x) + 0.5, weight)

        inputs = (x, weight, out_bias, res_bias)
        out_eager, res_eager, info_eager = f(*inputs)
        self.assertEqual(info_eager, 42)

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, *inputs)
        assert_no_graph_breaks(self)

        out_compiled, res_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(res_compiled, res_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)

        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(kernel_count, 1, f"Expected 1 kernel, got {kernel_count}")

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_autotune_no_fusion_final_has_fusion(self):
        """Test that fusion is applied AFTER autotuning, not during."""
        from unittest.mock import patch

        from helion.runtime.kernel import BoundKernel

        x = torch.randn(128, 256, device=DEVICE)
        y = torch.randn(128, 256, device=DEVICE)
        bias = torch.randn(128, 256, device=DEVICE)

        autotune_codes = []
        orig = BoundKernel.compile_config

        def track(self, config=None, **kw):
            if config:
                autotune_codes.append(self.to_triton_code(config))
            return orig(self, config, **kw)

        def f(x, y, bias):
            out, _ = elementwise_two_inputs(x, y)
            return torch.relu(out) + bias

        with patch.object(BoundKernel, "compile_config", track):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(f), x, y, bias)

        torch.testing.assert_close(result, torch.relu(x + y) + bias)
        self.assertGreater(len(autotune_codes), 0)
        # Autotune code should NOT have fusion
        for c in autotune_codes:
            self.assertNotIn("relu", c.lower())
            self.assertNotIn("epilogue_input", c)
        # Final code SHOULD have fusion
        self.assertIn("relu", code.lower())
        self.assertIn("epilogue", code.lower())

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transpose_then_view_to_3d_epilogue(self):
        """Epilogue: transpose then 2D->3D view out.T.reshape(D1, D2, D3) -> ops."""
        d1, d2, d3 = 8, 16, 32
        m, n = d1 * d2, d3
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        kernel_scale = 2.0
        epilogue_bias = torch.randn(d3, d1, d2, device=DEVICE, dtype=torch.float32)

        def f(x, epilogue_bias):
            out, info = elementwise_2d_fusion(x, kernel_scale)
            out_t = out.T
            out_3d = out_t.reshape(d3, d1, d2)
            out_processed = torch.relu(out_3d) + epilogue_bias
            return out_processed, info

        # Warmup
        _ = elementwise_2d_fusion(x, kernel_scale)

        # Eager reference
        out_eager, info_eager = f(x, epilogue_bias)
        self.assertEqual(info_eager, 42)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, epilogue_bias)
        assert_no_graph_breaks(self)

        out_compiled, info_compiled = result
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)
        self.assertEqual(info_compiled, 42)

        # Count kernels - view ops prevent fusion, expect >1 kernel
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertGreater(kernel_count, 1, f"Expected >1 kernels, got {kernel_count}")

    # =========================================================================
    # Epilogue Fusion Dtype Tests
    # =========================================================================

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_dtype(self):
        """Test fp16 prologue + kernel + epilogue with fp32 bias - dtype promotion."""
        m, n = 64, 128
        kernel_scale = 2.0

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, bias):
            # Prologue: ops before kernel (stays fp16)
            x_processed = torch.relu(x) * 1.2
            out = elementwise_2d_single_output_fusion(x_processed, kernel_scale)
            # Epilogue: ops after kernel with dtype promotion
            out_sigmoid = torch.sigmoid(out)
            return out_sigmoid + bias  # fp16 + fp32 -> fp32

        # Warmup
        _ = elementwise_2d_single_output_fusion(torch.relu(x) * 1.2, kernel_scale)

        # Eager reference
        out_eager = f(x, bias)
        self.assertEqual(out_eager.dtype, torch.float32)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, bias)
        assert_no_graph_breaks(self)

        self.assertEqual(result.dtype, out_eager.dtype)
        torch.testing.assert_close(result, out_eager, rtol=1e-3, atol=1e-3)

        # Count kernels - expect 1 kernel if fusion works
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}"
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_chained_ops(self):
        """Test fp16 prologue + chained epilogue ops with fp32 mul."""
        m, n = 64, 128
        kernel_scale = 2.0

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(n, device=DEVICE, dtype=torch.float32)

        def f(x, scale):
            # Prologue: ops before kernel (stays fp16)
            x_processed = torch.sigmoid(x) + 0.1
            out = elementwise_2d_single_output_fusion(x_processed, kernel_scale)
            # Epilogue: chained ops after kernel
            out = torch.sigmoid(out)
            out = torch.relu(out)
            out = torch.tanh(out)
            return out * scale  # fp16 * fp32 -> fp32

        # Warmup
        _ = elementwise_2d_single_output_fusion(torch.sigmoid(x) + 0.1, kernel_scale)

        # Eager reference
        out_eager = f(x, scale)
        self.assertEqual(out_eager.dtype, torch.float32)

        # Compiled
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, source_codes = run_and_get_code(compiled_f, x, scale)
        assert_no_graph_breaks(self)

        self.assertEqual(result.dtype, out_eager.dtype)
        torch.testing.assert_close(result, out_eager, rtol=1e-3, atol=1e-3)

        # Count kernels - expect 1 kernel if fusion works
        kernel_count, _ = count_triton_kernels(source_codes)
        self.assertEqual(
            kernel_count, 1, f"Expected 1 kernel (fusion), got {kernel_count}"
        )


instantiate_parametrized_tests(TestTorchCompile)

# ============================================================================
# Epilogue Fusion Helper Kernels
# ============================================================================

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def elementwise_2d_fusion(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, int]:
    """2D elementwise kernel: returns x * scale and scalar 42."""
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out, 42

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def elementwise_2d_single_output_fusion(x: torch.Tensor, scale: float) -> torch.Tensor:
    """2D elementwise kernel: returns x * scale (single output, no scalar)."""
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def elementwise_3d_fusion(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, int]:
    """3D elementwise kernel: returns x * scale and scalar 42."""
    d1, d2, d3 = x.size()
    out = torch.empty_like(x)

    for tile_d1 in hl.tile(d1):
        for tile_d2 in hl.tile(d2):
            x_tile = x[tile_d1, tile_d2, :]
            out[tile_d1, tile_d2, :] = x_tile * scale

    return out, 42

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def elementwise_with_different_shapes_same_elements(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Elementwise returning outputs with DIFFERENT shapes but SAME element count."""
    m, n = x.size()
    out1 = torch.empty([m * n], dtype=x.dtype, device=x.device)
    out2 = torch.empty([m, n], dtype=x.dtype, device=x.device)
    out1_view = out1.view(m, n)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out1_view[tile_m, :] = x_tile * 2
        out2[tile_m, :] = x_tile * 3
    return out1, out2, 42

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def elementwise_2d_explicit_dtype(x: torch.Tensor, scale: float) -> torch.Tensor:
    """2D elementwise kernel using explicit dtype=x.dtype for output allocation."""
    m, n = x.size()
    # Use explicit dtype=x.dtype instead of empty_like to test dtype propagation
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def rms_norm_multi_output(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """RMS normalization: returns (out, residual, 42)."""
    m, n = x.size()
    assert weight.size(0) == n
    out = torch.empty_like(x)
    residual = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_squared = torch.mean(x_squared, dim=-1)
        inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
        normalized = x_tile * inv_rms_tile[:, None]
        result = normalized * weight[:].to(torch.float32)
        out[tile_m, :] = result.to(out.dtype)
        residual[tile_m, :] = normalized.to(out.dtype)

    return out, residual, 42

@helion.kernel(
    static_shapes=True,
    autotune_effort="quick",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def elementwise_two_inputs(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Elementwise kernel with two inputs: returns x + y and scalar 42."""
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        y_tile = y[tile_m, :]
        out[tile_m, :] = x_tile + y_tile

    return out, 42

# ============================================================================
# Mutation Test Kernels
# ============================================================================

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def k_inplace(x: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1
    return x

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def k_two_mutated(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    for tile in hl.tile(x.size(0)):
        x[tile], y[tile] = x[tile] + 1, y[tile] * 2
    return x, y

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def k_one_mutated(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + y[tile]
    return x

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def k_mut_and_out(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + 1
        out[tile] = x[tile] + y[tile]
    return x, out

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def k_mut_no_return(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        x[tile] = x[tile] + 1
        out[tile] = y[tile] * 2
    return out

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def k_store(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.store(x, [tile], y[tile] * 2)
    return x

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def k_atomic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(x, [tile], y[tile])
    return x

@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    _wip_experimental_allow_torch_compile_fusion=True,
)
def k_no_mut(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


if __name__ == "__main__":
    unittest.main()
