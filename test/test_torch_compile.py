from __future__ import annotations

import unittest

import pytest
import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
import helion.language as hl


class TestTorchCompile(RefEagerTestBase, TestCase):
    def assert_no_graph_breaks(self):
        """Assert that no graph breaks occurred during compilation."""
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(
            len(graph_breaks),
            0,
            f"Graph breaks detected: {dict(graph_breaks)}",
        )
    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_add_kernel(self, allow_fusion):
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            a = x * 2.0
            b = y + z
            result = add(a, b)
            return result * 0.5

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add(x * 2.0, y + z)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, z)
        self.assert_no_graph_breaks()

        expected = ((x * 2.0) + (y + z)) * 0.5
        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_mutation_input(self, allow_fusion):
        """Test: kernel that mutates an input tensor."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add y to x in-place and return x."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_y = y * scale
            result = add_inplace(x, scaled_y)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = add_inplace(x_warmup, y * scale)

        x_test = x.clone()
        expected = (x + y * scale) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)
        torch.testing.assert_close(x_test, x + y * scale)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_multiple_outputs(self, allow_fusion):
        """Test: kernel with multiple outputs."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
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
            a = x.abs()
            b = y.neg()
            add_result, mul_result = add_and_mul(a, b)
            return add_result * 2.0, mul_result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_and_mul(x.abs(), y.neg())

        a, b = x.abs(), y.neg()
        expected_add = (a + b) * 2.0
        expected_mul = (a * b) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_add, compiled_mul = compiled_f(x, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_add, expected_add)
        torch.testing.assert_close(compiled_mul, expected_mul)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_output_alias_input(self, allow_fusion):
        """Test: kernel where output is directly the input (aliasing)."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_inplace_return(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add y to x in-place and return x (same tensor)."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_y = y * scale
            out = add_inplace_return(x, scaled_y)
            return out * 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = add_inplace_return(x_warmup, y * scale)

        x_test = x.clone()
        expected = (x + y * scale) * 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_multiple_outputs_with_input_alias(self, allow_fusion):
        """Test: kernel with multiple outputs where one is an input alias."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
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
            scaled_y = y * scale
            out_x, out_new = inplace_and_new(x, scaled_y)
            return out_x + 1.0, out_new - 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = inplace_and_new(x_warmup, y * scale)

        x_test = x.clone()
        expected_x = (x + y * scale) + 1.0
        expected_new = ((x + y * scale) * 2.0) - 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_x, compiled_new = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(x_test, x + y * scale)
        torch.testing.assert_close(compiled_x, expected_x)
        torch.testing.assert_close(compiled_new, expected_new)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_tensor_and_scalar_output(self, allow_fusion):
        """Test: kernel returning both a tensor and a scalar constant."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
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
            scaled_x = x * scale
            result, count = add_with_count(scaled_x, y)
            return result + 1.0, count

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_with_count(x * scale, y)

        expected_sum = (x * scale + y) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_sum, compiled_count = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_sum, expected_sum)
        self.assertEqual(compiled_count, 42)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_keyword_args(self, allow_fusion):
        """Test: passing tensor args by keyword name."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            a = x.square()
            b = y.sqrt()
            # Helion kernel with keyword args
            result = add(y=b, x=a)  # Pass by keyword, different order
            return result * 0.5

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16).abs()
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16).abs()
        _ = add(x.square(), y.sqrt())

        expected = (x.square() + y.sqrt()) * 0.5
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected, rtol=1e-2, atol=1e-2)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_default_params(self, allow_fusion):
        """Test: kernel with optional parameter using default value."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def scale_add(
            x: torch.Tensor, y: torch.Tensor, scale: float = 2.0
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile] * scale
            return out

        def f_with_default(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            biased_x = x + bias
            # Helion kernel with default scale=2.0
            result = scale_add(biased_x, y)
            return result * 0.5

        def f_with_override(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            biased_x = x + bias
            # Helion kernel with override scale=3.0
            result = scale_add(biased_x, y, scale=3.0)
            return result * 0.5

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        # Warmup both specializations
        _ = scale_add(x + bias, y)
        _ = scale_add(x + bias, y, scale=3.0)

        # Test with default
        expected_default = ((x + bias) + y * 2.0) * 0.5
        compiled_default = torch.compile(f_with_default, fullgraph=True, backend="inductor")
        torch.testing.assert_close(
            compiled_default(x, y, bias), expected_default, rtol=1e-3, atol=1e-3
        )
        self.assert_no_graph_breaks()

        # Test with override
        torch._dynamo.reset()
        expected_override = ((x + bias) + y * 3.0) * 0.5
        compiled_override = torch.compile(f_with_override, fullgraph=True, backend="inductor")
        torch.testing.assert_close(
            compiled_override(x, y, bias), expected_override, rtol=1e-3, atol=1e-3
        )
        self.assert_no_graph_breaks()

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_constant_scalar_args(self, allow_fusion):
        """Test: kernel with scalar constant arguments."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def scale_and_shift(
            x: torch.Tensor, scale: float, shift: float
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] * scale + shift
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            combined = x + y
            # Helion kernel with scalar args
            result = scale_and_shift(combined, 2.5, 1.0)
            return result - 0.5

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        _ = scale_and_shift(x + y, 2.5, 1.0)

        expected = ((x + y) * 2.5 + 1.0) - 0.5
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected, rtol=1e-3, atol=1e-3)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_1d_tensor(self, allow_fusion):
        """Test: 1D tensor input."""
        @helion.kernel(
            config=helion.Config(block_sizes=[4]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            a = x * 3.0
            b = y - 1.0
            result = add_1d(a, b)
            return result + 0.5

        x = torch.randn(32, device=DEVICE, dtype=torch.float16)
        y = torch.randn(32, device=DEVICE, dtype=torch.float16)
        _ = add_1d(x * 3.0, y - 1.0)

        expected = ((x * 3.0) + (y - 1.0)) + 0.5
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_3d_tensor(self, allow_fusion):
        """Test: 3D tensor input (batch, height, width)."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            a = x * scale.unsqueeze(1).unsqueeze(2)
            b = y.tanh()
            result = add_3d(a, b)
            return result.sum(dim=-1)

        x = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, device=DEVICE, dtype=torch.float16)
        _ = add_3d(x * scale.unsqueeze(1).unsqueeze(2), y.tanh())

        a = x * scale.unsqueeze(1).unsqueeze(2)
        b = y.tanh()
        expected = (a + b).sum(dim=-1)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected, rtol=1e-2, atol=1e-2)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_float32_dtype(self, allow_fusion):
        """Test: float32 dtype instead of float16."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_f32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            biased_x = x + bias
            result = add_f32(biased_x, y)
            return result * 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        _ = add_f32(x + bias, y)

        expected = ((x + bias) + y) * 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, bias)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_bfloat16_dtype(self, allow_fusion):
        """Test: bfloat16 dtype."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_bf16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_x = x * scale
            result = add_bf16(scaled_x, y)
            return result - 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.bfloat16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.bfloat16)
        _ = add_bf16(x * scale, y)

        expected = ((x * scale) + y) - 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_transposed_input(self, allow_fusion):
        """Test: transposed (non-contiguous) tensor input."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_transposed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # PyTorch ops before kernel (transpose is also an op)
            a = x.T * scale
            b = y.T
            result = add_transposed(a, b)
            return result + 1.0

        x = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        y = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_transposed(x.T * scale, y.T)

        expected = (x.T * scale + y.T) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_sliced_input(self, allow_fusion):
        """Test: sliced tensor input (view of larger tensor)."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_sliced(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # PyTorch ops before kernel (slicing + scaling)
            a = x[:2, :4] * scale
            b = y[:2, :4]
            result = add_sliced(a, b)
            return result * 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        _ = add_sliced(x[:2, :4] * scale, y[:2, :4])

        expected = ((x[:2, :4] * scale) + y[:2, :4]) * 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_kernel_after_pytorch_ops(self, allow_fusion):
        """Test: Helion kernel consuming output of PyTorch ops."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def double(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] * 2.0
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            z = x + y
            result = double(z)
            return result - 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = double(x + y)

        expected = ((x + y) * 2.0) - 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_chained_kernels(self, allow_fusion):
        """Test: two Helion kernels in sequence with PyTorch ops."""
        if not allow_fusion:
            pytest.xfail("Known bug: multiple kernels cause Dynamo MappingProxy error without fusion")

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def mul_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] * y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            # PyTorch ops before first kernel
            a = x * 2.0
            b = y + bias
            # First Helion kernel
            z = add_k(a, b)
            # Second Helion kernel
            result = mul_k(z, a)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        # Warmup both kernels
        a = x * 2.0
        b = y + bias
        z = add_k(a, b)
        _ = mul_k(z, a)

        expected = ((a + b) * a) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, bias)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_kernel_called_twice(self, allow_fusion):
        """Test: same kernel called twice with different inputs."""
        if not allow_fusion:
            pytest.xfail("Known bug: multiple kernel calls cause Dynamo MappingProxy error without fusion")

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_x = x * scale
            # First kernel call
            a = add_k(scaled_x, y)
            # Second kernel call
            b = add_k(a, z)
            return b + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_k(x * scale, y)

        expected = (((x * scale) + y) + z) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, z, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_same_tensor_twice(self, allow_fusion):
        """Test: same tensor passed as two different arguments."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_self(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            scaled = x * 2.0 + bias
            # Helion kernel - same tensor as both args
            result = add_self(scaled, scaled)
            return result.mean(dim=-1)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scaled = x * 2.0 + bias
        _ = add_self(scaled, scaled)

        expected = (scaled + scaled).mean(dim=-1)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, bias)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected, rtol=1e-2, atol=1e-2)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_all_outputs_alias_inputs(self, allow_fusion):
        """Test: multi-output where all outputs are input aliases."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
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
            scaled_x = x * scale
            scaled_y = y * scale
            out_x, out_y = swap_inplace(scaled_x, scaled_y)
            return out_x + 1.0, out_y - 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup, y_warmup = (x * scale).clone(), (y * scale).clone()
        _ = swap_inplace(x_warmup, y_warmup)

        x_test, y_test = x.clone(), y.clone()
        x_orig_scaled, y_orig_scaled = x * scale, y * scale

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        out_x, out_y = compiled_f(x_test, y_test, scale)
        self.assert_no_graph_breaks()

        # After swap: out_x has y*scale values, out_y has x*scale values
        # Then +1.0 and -1.0 respectively
        torch.testing.assert_close(out_x, y_orig_scaled + 1.0)
        torch.testing.assert_close(out_y, x_orig_scaled - 1.0)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_atomic_add_mutation(self, allow_fusion):
        """Test: mutation via atomic operations."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
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
            a = x * 0.5
            b = y.abs()
            # Helion kernel with atomic mutation
            result = atomic_add_kernel(a, b, out)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        out_warmup = torch.zeros_like(x)
        _ = atomic_add_kernel(x * 0.5, y.abs(), out_warmup)

        expected = ((x * 0.5) + y.abs()) + 1.0
        out_test = torch.zeros_like(x)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, out_test)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_mutation_not_returned(self, allow_fusion):
        """Test: kernel mutates input but returns different output."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def mutate_and_return_new(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            """Mutate x in-place, but return a new tensor with different computation."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
                out[tile] = x[tile] * 2.0
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_y = y * scale
            result = mutate_and_return_new(x, scaled_y)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = mutate_and_return_new(x_warmup, y * scale)

        x_test = x.clone()
        expected_x = x + y * scale
        expected_out = (expected_x * 2.0) + 1.0

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(x_test, expected_x)
        torch.testing.assert_close(compiled_out, expected_out)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_indirect_output_alias(self, allow_fusion):
        """Test: output is a slice/view of input (indirect alias with different shape)."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def slice_and_mutate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Mutate a slice of x and return that slice (indirect alias)."""
            # Get a slice view of x
            x_slice = x[:2, :4]
            for tile in hl.tile(x_slice.size()):
                x_slice[tile] = x_slice[tile] + y[tile]
            return x_slice

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_y = y * scale
            result = slice_and_mutate(x, scaled_y)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = slice_and_mutate(x_warmup, y * scale)

        x_test = x.clone()
        expected_slice = (x[:2, :4] + y * scale) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected_slice)
        torch.testing.assert_close(x_test[:2, :4], x[:2, :4] + y * scale)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_mixed_positional_and_keyword_args(self, allow_fusion):
        """Test: mixed positional and keyword arguments."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_three(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile] + z[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_x = x * scale
            # Mix positional and keyword args
            result = add_three(scaled_x, z=z, y=y)
            return result - 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_three(x * scale, y, z)

        expected = ((x * scale) + y + z) - 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, z, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_empty_tensor(self, allow_fusion):
        """Test: tensors with zero-size dimensions."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_empty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_x = x * scale
            result = add_empty(scaled_x, y)
            return result + 1.0

        # Test with zero-size first dimension
        x = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        _ = add_empty(x * scale, y)

        expected = ((x * scale) + y) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        self.assertEqual(compiled_out.shape, expected.shape)
        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_int32_dtype(self, allow_fusion):
        """Test: int32 integer dtype."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_int32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            biased_x = x + bias
            result = add_int32(biased_x, y)
            return result * 2

        x = torch.randint(-100, 100, (4, 8), device=DEVICE, dtype=torch.int32)
        y = torch.randint(-100, 100, (4, 8), device=DEVICE, dtype=torch.int32)
        bias = torch.randint(-10, 10, (4, 8), device=DEVICE, dtype=torch.int32)
        _ = add_int32(x + bias, y)

        expected = ((x + bias) + y) * 2
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, bias)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_int64_dtype(self, allow_fusion):
        """Test: int64 integer dtype."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_int64(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            biased_y = y + bias
            result = add_int64(x, biased_y)
            return result - 1

        x = torch.randint(-1000, 1000, (4, 8), device=DEVICE, dtype=torch.int64)
        y = torch.randint(-1000, 1000, (4, 8), device=DEVICE, dtype=torch.int64)
        bias = torch.randint(-100, 100, (4, 8), device=DEVICE, dtype=torch.int64)
        _ = add_int64(x, y + bias)

        expected = (x + (y + bias)) - 1
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, bias)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_reduction_sum(self, allow_fusion):
        """Test: kernel with reduction dimension (sum along axis)."""
        @helion.kernel(
            config=helion.Config(block_sizes=[4]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def row_sum(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], device=x.device, dtype=x.dtype)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].to(torch.float32).sum(-1).to(x.dtype)
            return out

        def f(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            scaled = x * weight
            # Helion kernel with reduction
            row_sums = row_sum(scaled)
            return row_sums.softmax(dim=0)

        x = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        _ = row_sum(x * weight)

        expected = (x * weight).sum(dim=1).softmax(dim=0)
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, weight)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected, rtol=1e-3, atol=1e-3)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_inline_triton_mutation(self, allow_fusion):
        """Test: kernel using inline_triton marks all inputs as potentially mutated."""
        if not allow_fusion:
            pytest.xfail("Known bug: inline_triton uses asm that requires fusion enabled")

        @helion.kernel(
            config=helion.Config(block_sizes=[16]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
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
            a = x.exp()
            b = y.log1p()
            # Helion kernel with inline_triton
            result = inline_add_kernel(a, b)
            return result * 2.0

        x = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()
        y = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()
        _ = inline_add_kernel(x.exp(), y.log1p())

        expected = (x.exp() + y.log1p()) * 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_single_tensor_mutation(self, allow_fusion):
        """Test: kernel with single tensor that is mutated (fewer than 2 tensors path)."""
        @helion.kernel(
            config=helion.Config(block_sizes=[16]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def increment_kernel(x: torch.Tensor) -> torch.Tensor:
            """Increment each element by 1 in-place."""
            for tile in hl.tile(x.shape):
                x[tile] = x[tile] + 1.0
            return x

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            scaled = (x * 2.0 + bias).contiguous()
            # Helion kernel with single tensor mutation
            result = increment_kernel(scaled)
            return result * 0.5

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(64, device=DEVICE, dtype=torch.float32)
        x_warmup = (x * 2.0 + bias).contiguous()
        _ = increment_kernel(x_warmup)

        expected = ((x * 2.0 + bias) + 1.0) * 0.5
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, bias)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    @parametrize("allow_fusion", (True, False))
    def test_signal_mutation(self, allow_fusion):
        """Test: kernel using hl.signal correctly tracks mutation of signal_pad tensor."""
        if not allow_fusion:
            pytest.xfail("Known bug: signal/wait uses asm that requires fusion enabled")

        @helion.kernel(
            autotune_effort="none",
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
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
            scaled_x = x * scale
            # Helion kernel with signal
            out, sig = signal_kernel(signal_pad, scaled_x)
            return out + 1.0, sig

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        scale = torch.randn(4, device=DEVICE, dtype=torch.float32)
        signal_pad_warmup = signal_pad.clone()
        _ = signal_kernel(signal_pad_warmup, x * scale)

        signal_pad_test = signal_pad.clone()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out, compiled_signal = compiled_f(signal_pad_test, x, scale)
        self.assert_no_graph_breaks()

        expected = ((x * scale) * 2) + 1.0
        torch.testing.assert_close(compiled_out, expected)
        # signal_pad should have been mutated (set to 2)
        self.assertTrue(torch.all(compiled_signal == 2))

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    @parametrize("allow_fusion", (True, False))
    def test_wait_with_update_mutation(self, allow_fusion):
        """Test: kernel using hl.wait with update parameter tracks mutation."""
        if not allow_fusion:
            pytest.xfail("Known bug: signal/wait uses asm that requires fusion enabled")

        @helion.kernel(
            autotune_effort="none",
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
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
            biased_x = x + bias
            # Helion kernel with wait/update
            out, sig = wait_update_kernel(signal_pad, biased_x)
            return out - 0.5, sig

        signal_pad = torch.ones(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, device=DEVICE, dtype=torch.float32)
        signal_pad_warmup = signal_pad.clone()
        _ = wait_update_kernel(signal_pad_warmup, x + bias)

        signal_pad_test = signal_pad.clone()
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out, compiled_signal = compiled_f(signal_pad_test, x, bias)
        self.assert_no_graph_breaks()

        expected = ((x + bias) * 2) - 0.5
        torch.testing.assert_close(compiled_out, expected)
        # signal_pad should have been mutated (updated to 2)
        self.assertTrue(torch.all(compiled_signal == 2))

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_output_transpose_of_input(self, allow_fusion):
        """Test: output is transpose of input (indirect alias with swapped strides)."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def mutate_and_return_transpose(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Mutate x and return x.T (indirect alias with different strides)."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x.T

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_y = y * scale
            result = mutate_and_return_transpose(x, scaled_y)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = mutate_and_return_transpose(x_warmup, y * scale)

        x_test = x.clone()
        expected = (x + y * scale).T + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)
        # Verify x was mutated
        torch.testing.assert_close(x_test, x + y * scale)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_slice_with_storage_offset(self, allow_fusion):
        """Test: output is slice with non-zero storage offset."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def mutate_middle_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Mutate middle portion of x and return that slice."""
            x_slice = x[1:3, 2:6]  # Non-zero storage offset
            for tile in hl.tile(x_slice.size()):
                x_slice[tile] = x_slice[tile] + y[tile]
            return x_slice

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_y = y * scale
            result = mutate_middle_slice(x, scaled_y)
            return result * 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = mutate_middle_slice(x_warmup, y * scale)

        x_test = x.clone()
        expected_slice = (x[1:3, 2:6] + y * scale) * 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected_slice)
        # Verify the slice in x was mutated
        torch.testing.assert_close(x_test[1:3, 2:6], x[1:3, 2:6] + y * scale)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_strided_slice_input(self, allow_fusion):
        """Test: input with stride > 1 (e.g., x[::2, ::2])."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_strided(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # Strided slices - every other row and column
            a = x[::2, ::2] * scale
            b = y[::2, ::2]
            result = add_strided(a, b)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        _ = add_strided(x[::2, ::2] * scale, y[::2, ::2])

        expected = (x[::2, ::2] * scale + y[::2, ::2]) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_multiple_indirect_aliases(self, allow_fusion):
        """Test: multiple outputs that are different views of the same input."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def mutate_return_two_slices(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Mutate x and return two different slices."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x[:2, :4], x[2:4, 4:8]

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            scaled_y = y * scale
            slice1, slice2 = mutate_return_two_slices(x, scaled_y)
            return slice1 + 1.0, slice2 * 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = mutate_return_two_slices(x_warmup, y * scale)

        x_test = x.clone()
        x_expected = x + y * scale
        expected1 = x_expected[:2, :4] + 1.0
        expected2 = x_expected[2:4, 4:8] * 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        out1, out2 = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(out1, expected1)
        torch.testing.assert_close(out2, expected2)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_view_input_mutate_different_view(self, allow_fusion):
        """Test: input is a slice, mutate and return a different slice of the same base."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
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
            scaled_y = y * scale
            # Pass slice as first arg, but also pass full tensor
            result = process_slice_return_other(x[:2, :4], scaled_y, x)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = process_slice_return_other(x_warmup[:2, :4], y * scale, x_warmup)

        x_test = x.clone()
        # x[:2, :4] gets mutated, but we return x[2:4, 4:8] (unmodified)
        expected = x[2:4, 4:8] + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)
        # Verify the first slice was mutated
        torch.testing.assert_close(x_test[:2, :4], x[:2, :4] + y * scale)
        # Verify the second slice was NOT mutated
        torch.testing.assert_close(x_test[2:4, 4:8], x[2:4, 4:8])

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_unsqueeze_view_input(self, allow_fusion):
        """Test: input with unsqueeze (adds dimension of size 1)."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # Unsqueeze to add batch dimension
            a = x.unsqueeze(0) * scale
            b = y.unsqueeze(0)
            result = add_3d(a, b)
            return result.squeeze(0) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(1, 4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_3d(x.unsqueeze(0) * scale, y.unsqueeze(0))

        expected = ((x.unsqueeze(0) * scale + y.unsqueeze(0)).squeeze(0)) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_contiguous_after_transpose(self, allow_fusion):
        """Test: input that was transposed then made contiguous."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_contiguous(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # Transpose then contiguous - creates a copy with contiguous strides
            a = x.T.contiguous() * scale
            b = y.T.contiguous()
            result = add_contiguous(a, b)
            return result + 1.0

        x = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        y = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_contiguous(x.T.contiguous() * scale, y.T.contiguous())

        expected = (x.T.contiguous() * scale + y.T.contiguous()) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_indirect_alias_1d_from_2d(self, allow_fusion):
        """Test: return 1D view of 2D input (reshape/flatten)."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def mutate_return_row(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Mutate 2D tensor and return first row (1D view)."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x[0]  # Returns 1D tensor

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_y = y * scale
            result = mutate_return_row(x, scaled_y)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = mutate_return_row(x_warmup, y * scale)

        x_test = x.clone()
        expected = (x + y * scale)[0] + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)
        torch.testing.assert_close(x_test, x + y * scale)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_permute_view(self, allow_fusion):
        """Test: output is permuted view of input."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def mutate_return_permuted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Mutate 3D tensor and return permuted view."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x.permute(2, 0, 1)  # (B, H, W) -> (W, B, H)

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            scaled_y = y * scale
            result = mutate_return_permuted(x, scaled_y)
            return result + 1.0

        x = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = mutate_return_permuted(x_warmup, y * scale)

        x_test = x.clone()
        expected = (x + y * scale).permute(2, 0, 1) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)
        torch.testing.assert_close(x_test, x + y * scale)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_alias_view_as_two_args(self, allow_fusion):
        """Test: passing x and x.view(-1) as two different arguments."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_views(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add y to x where y may be a view of x."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(a: torch.Tensor) -> torch.Tensor:
            x = a * 2
            y = x.view(-1)
            result = add_views(x, y)
            return result + 1.0

        a = torch.randn(64, device=DEVICE, dtype=torch.float16)
        a_warmup = a.clone()
        x_warmup = a_warmup * 2
        _ = add_views(x_warmup, x_warmup.view(-1))

        expected_x = a * 2
        # After add_views: x = x + x = 2*x = 4*a
        expected = expected_x * 2 + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(a)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_aliasing_inputs_used_after(self, allow_fusion):
        """Test: view of input used after kernel mutation."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def inplace_add_one(x: torch.Tensor) -> torch.Tensor:
            """Add 1 to x in-place."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
            return x

        def f(x: torch.Tensor) -> torch.Tensor:
            y = x.view(-1)  # View before kernel
            _ = inplace_add_one(x)  # Mutate x
            return y + 1  # Use view after - should see mutation

        x = torch.randn(64, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = inplace_add_one(x_warmup)

        x_test = x.clone()
        # x gets +1, y is view of x, so y also has +1, then +1 more
        expected = x + 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_mutation_after_kernel(self, allow_fusion):
        """Test: mutation after kernel preserves output correctness."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def add_tensors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add x and y, return new tensor."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = add_tensors(x, y)  # out = x + y
            x.add_(10)  # Mutate x AFTER kernel
            return out  # Should still be original x + y

        x = torch.randn(64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = add_tensors(x_warmup, y)

        x_test = x.clone()
        expected = x + y  # Output computed before x mutation
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)
        # x_test should have been mutated
        torch.testing.assert_close(x_test, x + 10)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_view_internal(self, allow_fusion):
        """Test: kernel that uses view internally."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def view_and_mutate(x: torch.Tensor) -> torch.Tensor:
            """Create view internally and mutate through it."""
            y = x.view(x.size())  # Same-shape view
            for tile in hl.tile(y.size()):
                y[tile] = y[tile] + 1
            return x  # Return original (same storage)

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x - 1  # Prologue
            result = view_and_mutate(x)
            return result * 2  # Epilogue

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        x_warmup = (x - 1).clone()
        _ = view_and_mutate(x_warmup)

        # x_pre = x - 1
        # After view_and_mutate: x_pre + 1 = x
        # Epilogue: x * 2
        expected = x * 2
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x.clone())
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_narrow_view_input(self, allow_fusion):
        """Test: narrow (slice with offset) input tensor."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def scale_tensor(x: torch.Tensor, scale: float) -> torch.Tensor:
            """Scale a tensor."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        def f(x: torch.Tensor) -> torch.Tensor:
            # narrow: select a portion along dimension 1
            x_narrow = x.narrow(1, 2, 4)  # dim=1, start=2, length=4
            result = scale_tensor(x_narrow, 3.0)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = scale_tensor(x.narrow(1, 2, 4), 3.0)

        expected = x.narrow(1, 2, 4) * 3.0 + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_expand_view_input(self, allow_fusion):
        """Test: expanded (broadcast) input tensor."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 1, 2]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def scale_3d(x: torch.Tensor, scale: float) -> torch.Tensor:
            """Scale a 3D tensor."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        batch_size = 4

        def f(x: torch.Tensor) -> torch.Tensor:
            # Expand: broadcast 2D to 3D
            x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)
            x_contig = x_expanded.contiguous()
            result = scale_3d(x_contig, 2.0)
            return result.sum(dim=0)  # Sum over batch

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_exp = x.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        _ = scale_3d(x_exp, 2.0)

        expected = (x * 2.0) * batch_size  # Each batch is x*2, sum gives batch*x*2
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRefEager("does not work with ref eager")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("allow_fusion", (True, False))
    def test_diagonal_slice_input(self, allow_fusion):
        """Test: diagonal view of input tensor."""
        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=allow_fusion,
        )
        def scale_1d(x: torch.Tensor, scale: float) -> torch.Tensor:
            """Scale a 1D tensor."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        def f(x: torch.Tensor) -> torch.Tensor:
            # diagonal: 1D view of main diagonal
            diag = x.diagonal()
            result = scale_1d(diag, 2.0)
            return result + 1.0

        n = 8
        x = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
        _ = scale_1d(x.diagonal(), 2.0)

        expected = x.diagonal() * 2.0 + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)


instantiate_parametrized_tests(TestTorchCompile)


if __name__ == "__main__":
    unittest.main()
