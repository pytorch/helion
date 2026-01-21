from __future__ import annotations

import unittest

import pytest
import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
import helion.language as hl


class TestTorchCompile(RefEagerTestDisabled, TestCase):
    def assert_no_graph_breaks(self):
        """Assert that no graph breaks occurred during compilation."""
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(
            len(graph_breaks),
            0,
            f"Graph breaks detected: {dict(graph_breaks)}",
        )

    @skipIfRocm("torch.compile add kernel missing kernel metadata fields on ROCm")
    @skipIfTileIR("torch.compile add kernel missing kernel metadata fields on tileir")
    def test_add_kernel_fusion_disabled(self):
        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=False,
        )
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return add(x, y)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        out = add(x, y)
        compiled_add = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_add(x, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(out, x + y)
        torch.testing.assert_close(compiled_out, x + y)

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
            return add_three(z=z, y=y, x=x) * 0.5

        def f_mixed(x, y, z):
            return add_three(x, z=z, y=y) - 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_three(x, y, z)

        if arg_style == "all_keyword":
            f = f_all_kw
            expected = (x + y + z) * 0.5
        else:
            f = f_mixed
            expected = (x + y + z) - 1.0

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, z)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

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
            biased_x = x + bias
            # Helion kernel with default scale=2.0
            result = scale_add(biased_x, y)
            return result * 0.5

        def f_with_override(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
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
        compiled_default = torch.compile(
            f_with_default, fullgraph=True, backend="inductor"
        )
        torch.testing.assert_close(
            compiled_default(x, y, bias), expected_default, rtol=1e-3, atol=1e-3
        )
        self.assert_no_graph_breaks()

        # Test with override
        torch._dynamo.reset()
        expected_override = ((x + bias) + y * 3.0) * 0.5
        compiled_override = torch.compile(
            f_with_override, fullgraph=True, backend="inductor"
        )
        torch.testing.assert_close(
            compiled_override(x, y, bias), expected_override, rtol=1e-3, atol=1e-3
        )
        self.assert_no_graph_breaks()

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

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @parametrize("dtype", (torch.float32, torch.bfloat16, torch.int32, torch.int64))
    def test_dtype_variations(self, dtype):
        """Test: various dtypes work correctly."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_typed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            biased_x = x + bias
            result = add_typed(biased_x, y)
            return result * 2

        is_int = dtype in (torch.int32, torch.int64)
        if is_int:
            x = torch.randint(-100, 100, (4, 8), device=DEVICE, dtype=dtype)
            y = torch.randint(-100, 100, (4, 8), device=DEVICE, dtype=dtype)
            bias = torch.randint(-10, 10, (4, 8), device=DEVICE, dtype=dtype)
        else:
            x = torch.randn(4, 8, device=DEVICE, dtype=dtype)
            y = torch.randn(4, 8, device=DEVICE, dtype=dtype)
            bias = torch.randn(4, 8, device=DEVICE, dtype=dtype)
        _ = add_typed(x + bias, y)

        expected = ((x + bias) + y) * 2
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, bias)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

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

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_after_pytorch_ops(self):
        """Test: Helion kernel consuming output of PyTorch ops."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
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

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_after_kernel(self):
        """Test: mutation after kernel preserves output correctness."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=True,
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
            result = mutate_two_inputs(x, y, z)
            return result - 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup, y_warmup = x.clone(), y.clone()
        _ = mutate_two_inputs(x_warmup, y_warmup, z)

        x_test, y_test = x.clone(), y.clone()
        x_mutated = x + 1.0
        y_mutated = y * 2.0
        expected = (z + x_mutated + y_mutated) - 1.0

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x_test, y_test, z)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)
        torch.testing.assert_close(x_test, x_mutated)
        torch.testing.assert_close(y_test, y_mutated)

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
            # Detach x before passing to kernel
            x_detached = x.detach()
            result = add_tensors(x_detached, y)
            return result * 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        _ = add_tensors(x.detach(), y)

        expected = (x.detach() + y) * 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

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

        weight = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        module = SimpleModule(weight.clone(), bias.clone())

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        # Warmup
        _ = linear_add(x, weight, bias)

        expected = x * weight + bias
        compiled_module = torch.compile(module, fullgraph=True, backend="inductor")
        compiled_out = compiled_module(x)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_concat_input_mutation(self):
        """Test: mutate tensor that was created by concatenation."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add y to x in-place and return x."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(a: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Concatenate a and b, then mutate the result
            x = torch.cat([a, b], dim=0)
            result = add_inplace(x, y)
            return result * 2.0

        a = torch.randn(2, 8, device=DEVICE, dtype=torch.float16)
        b = torch.randn(2, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = torch.cat([a, b], dim=0)
        _ = add_inplace(x_warmup, y)

        x_concat = torch.cat([a, b], dim=0)
        expected = (x_concat + y) * 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(a, b, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

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
            biased = x + bias
            result = chained_mutate(biased)
            return result + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        x_warmup = (x + bias).clone()
        _ = chained_mutate(x_warmup)

        # Expected: ((x + bias + 1) * 2 - 3) + 1
        expected = ((x + bias + 1.0) * 2.0 - 3.0) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, bias)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

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
            x_clone = x.clone()
            result = add_inplace(x_clone, y)
            # Return both: mutated clone and original (should be unchanged)
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = add_inplace(x_warmup, y)

        x_test = x.clone()
        expected_result = x + y
        expected_original = x.clone()  # Original should be unchanged

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_result, compiled_original = compiled_f(x_test, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_result, expected_result)
        torch.testing.assert_close(compiled_original, expected_original)

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
            a = x * 2.0
            b = y + 1.0
            result = add_into_out(a, b, out)
            return result * 0.5

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        out_warmup = torch.empty_like(x)
        _ = add_into_out(x * 2.0, y + 1.0, out_warmup)

        out_test = torch.empty_like(x)
        expected = ((x * 2.0) + (y + 1.0)) * 0.5
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, out_test)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)
        # Verify out_test was also filled correctly
        torch.testing.assert_close(out_test, (x * 2.0) + (y + 1.0))

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
            scaled_y = y * scale
            out_2d, out_1d = mutate_return_views(x, scaled_y)
            return out_2d + 1.0, out_1d * 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = mutate_return_views(x_warmup, y * scale)

        x_test = x.clone()
        x_mutated = x + y * scale
        expected_2d = x_mutated + 1.0
        expected_1d = x_mutated.view(-1) * 2.0

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_2d, compiled_1d = compiled_f(x_test, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_2d, expected_2d)
        torch.testing.assert_close(compiled_1d, expected_1d)

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
            # Create two views of base with different strides
            x = base[::2]  # Every other element: shape [16]
            y = base[1::2]  # Every other element offset by 1: shape [16]
            result = add_1d(x, y)
            return result + 1.0

        base = torch.randn(32, device=DEVICE, dtype=torch.float16)
        _ = add_1d(base[::2], base[1::2])

        expected = (base[::2] + base[1::2]) + 1.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(base)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

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
            # Take a slice, mutate it, return both slice result and full tensor
            x_slice = x[:2, :4]  # First 2 rows, first 4 cols
            result = add_inplace_slice(x_slice, y)
            return result, x  # x should have mutation in slice region

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        x_warmup = x.clone()
        _ = add_inplace_slice(x_warmup[:2, :4], y)

        x_test = x.clone()
        expected_slice = x[:2, :4] + y
        expected_full = x.clone()
        expected_full[:2, :4] = expected_slice

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_slice, compiled_full = compiled_f(x_test, y)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_slice, expected_slice)
        torch.testing.assert_close(compiled_full, expected_full)
        # Verify the rest of x_test is unchanged
        torch.testing.assert_close(x_test[2:, :], x[2:, :])
        torch.testing.assert_close(x_test[:, 4:], x[:, 4:])

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
            a = x * scale
            b = y + 1.0
            result = create_and_return_view(a, b)
            return result * 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = create_and_return_view(x * scale, y + 1.0)

        expected = ((x * scale) + (y + 1.0)).view(-1) * 2.0
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_f(x, y, scale)
        self.assert_no_graph_breaks()

        torch.testing.assert_close(compiled_out, expected)

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
            z = x + 0.5  # prologue
            result = add_kernel(z, y)
            return result * 2  # epilogue

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_kernel(x.clone(), y)  # warmup

        with torch.inference_mode():
            compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
            result = compiled_f(x, y)

        expected = ((x + 0.5) + y) * 2
        torch.testing.assert_close(result, expected)
        self.assert_no_graph_breaks()

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
            # Pass same tensor as both x and y
            a = z.clone()
            result = add_both(a, a)
            return result, z  # original should be unchanged

        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_both(z.clone(), z.clone())  # warmup

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, orig = compiled_f(z)

        # Both operations happen on same tensor, so result = z + 1 + 2 = z + 3
        expected = z + 3
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(orig, z)  # original unchanged
        self.assert_no_graph_breaks()

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
            # x is already a view (passed from outside), take a 2D slice
            a = x[:2]  # 2D slice of view
            return add_inplace(a.clone(), y[:2])

        base = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x = base[1:]  # view with shape [3, 8]
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_inplace(x[:2].clone(), y[:2])  # warmup

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result = compiled_f(x, y)

        expected = x[:2] + y[:2]
        torch.testing.assert_close(result, expected)
        self.assert_no_graph_breaks()

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_unbind_view_mutation(self):
        """Test: unbind creates views, kernel mutates one of them."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_one(x: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
            return x

        def f(input_tensor):
            # Unbind creates multiple views along dim 0
            views = input_tensor.unbind(0)
            # Mutate just the first view (via clone to not affect original)
            first = views[0].clone()
            result = add_one(first)
            return result, input_tensor  # original should be unchanged

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_one(x[0].clone())  # warmup

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result, orig = compiled_f(x)

        expected = x[0] + 1
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(orig, x)
        self.assert_no_graph_breaks()

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_copy_preserve_strides(self):
        """Test: kernel output copied to tensor with different strides."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 2]),
            _wip_experimental_allow_torch_compile_fusion=True,
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(x, y):
            # Create output with different strides
            result = add_kernel(x.clone(), y)
            # Copy to strided tensor
            strided_out = torch.empty_strided(
                (4, 8), (16, 1), device=DEVICE, dtype=torch.float16
            )
            strided_out.copy_(result)
            return strided_out

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = add_kernel(x.clone(), y)  # warmup

        # Compute expected before calling compiled
        expected = x + y

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        result = compiled_f(x, y)

        torch.testing.assert_close(result, expected)
        self.assertEqual(result.stride(), (16, 1))
        self.assert_no_graph_breaks()


instantiate_parametrized_tests(TestTorchCompile)


if __name__ == "__main__":
    unittest.main()
