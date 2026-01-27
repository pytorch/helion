from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
import helion.language as hl

# -----------------------------------------------------------------------------
# Basic Operations (no mutation, return new tensor)
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(autotune_effort="none")
def k_add_mul(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both add and mul results."""
    add_out = torch.empty_like(x)
    mul_out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        add_out[tile] = x[tile] + y[tile]
        mul_out[tile] = x[tile] * y[tile]
    return add_out, mul_out


@helion.kernel(autotune_effort="none")
def k_scale_with_scalar_output(
    x: torch.Tensor, scale: float
) -> tuple[torch.Tensor, int]:
    """2D elementwise kernel: returns x * scale and scalar 42."""
    m, n = x.size()
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :]
        out[tile_m, :] = x_tile * scale

    return out, 42


@helion.kernel(autotune_effort="none")
def k_sum_rows(x: torch.Tensor) -> torch.Tensor:
    """Sum each row of x."""
    m, _ = x.size()
    out = torch.empty([m], device=x.device, dtype=x.dtype)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].to(torch.float32).sum(-1).to(x.dtype)
    return out


@helion.kernel(autotune_effort="none")
def k_rms_norm(
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


@helion.kernel(autotune_effort="none")
def k_inline_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add using inline_triton."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        x_val = x[tile]
        y_val = y[tile]
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


# -----------------------------------------------------------------------------
# Mutations
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_add_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]
    return x


@helion.kernel(autotune_effort="none")
def k_mutate_both(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    for tile in hl.tile(x.size(0)):
        x[tile], y[tile] = x[tile] + 1, y[tile] * 2
    return x, y


@helion.kernel(autotune_effort="none")
def k_mutate_via_view(x: torch.Tensor) -> torch.Tensor:
    """Create view internally and mutate through it."""
    y = x.view(x.size())
    for tile in hl.tile(y.size()):
        y[tile] = y[tile] + 1
    return x


@helion.kernel(autotune_effort="none")
def k_add_to_both(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add 1 to x and 2 to y (which may alias)."""
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1
        y[tile] = y[tile] + 2
    return x


@helion.kernel(autotune_effort="none")
def k_store(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.store(x, [tile], y[tile] * 2)
    return x


@helion.kernel(autotune_effort="none")
def k_atomic_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(x.size(0)):
        hl.atomic_add(x, [tile], y[tile])
    return x


@helion.kernel(autotune_effort="none")
def k_mutate_with_out(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1
        out[tile] = x[tile] + y[tile]
    return x, out


@helion.kernel(autotune_effort="none")
def k_mutate_return_new(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1
        out[tile] = y[tile] * 2
    return out


@helion.kernel(autotune_effort="none")
def k_mutate_two_return_new(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Mutate both x and y, return a new tensor computed from z."""
    out = torch.empty_like(z)
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + 1.0
        y[tile] = y[tile] * 2.0
        out[tile] = z[tile] + x[tile] + y[tile]
    return out


# -----------------------------------------------------------------------------
# Pre-allocated/External Output
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_add_into_out(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Add x and y, store result in pre-allocated out tensor."""
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(autotune_effort="none")
def k_atomic_add_to_out(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Atomically add x and y to out."""
    for tile in hl.tile(x.size()):
        hl.atomic_add(out, tile, x[tile])
        hl.atomic_add(out, tile, y[tile])
    return out


# -----------------------------------------------------------------------------
# View/Slice Operations
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_slice_mutate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mutate a slice of x and return that slice (indirect alias)."""
    x_slice = x[:2, :4]
    for tile in hl.tile(x_slice.size()):
        x_slice[tile] = x_slice[tile] + y[tile]
    return x_slice


@helion.kernel(autotune_effort="none")
def k_slice_return_other(
    x_slice: torch.Tensor, y: torch.Tensor, x_full: torch.Tensor
) -> torch.Tensor:
    """Process one slice but return a different slice of the same tensor."""
    for tile in hl.tile(x_slice.size()):
        x_slice[tile] = x_slice[tile] + y[tile]
    return x_full[2:4, 4:8]


@helion.kernel(autotune_effort="none")
def k_mutate_permuted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mutate 3D tensor and return permuted view."""
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]
    return x.permute(2, 0, 1)


@helion.kernel(autotune_effort="none")
def k_mutate_return_view(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutate x and return both x and a view of x."""
    for tile in hl.tile(x.size()):
        x[tile] = x[tile] + y[tile]
    return x, x.view(-1)


@helion.kernel(autotune_effort="none")
def k_create_return_view(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Create intermediate tensor, mutate it, return a view."""
    intermediate = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        intermediate[tile] = x[tile] + y[tile]
    return intermediate.view(-1)


# -----------------------------------------------------------------------------
# Special Operations (signal/wait)
# -----------------------------------------------------------------------------


@helion.kernel(autotune_effort="none")
def k_signal(
    signal_pad: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signal kernel using hl.signal."""
    out = torch.empty_like(x)
    (n,) = x.shape
    for i in hl.grid(n):
        hl.signal(signal_pad, [i], signal=2)
        out[i] = x[i] * 2
    return out, signal_pad


@helion.kernel(autotune_effort="none")
def k_wait_update(
    signal_pad: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wait kernel using hl.wait with update."""
    out = torch.empty_like(x)
    (n,) = x.shape
    for i in hl.grid(n):
        hl.wait(signal_pad, [i], signal=1, update=2)
        out[i] = x[i] * 2
    return out, signal_pad


# =============================================================================
# Test Class
# =============================================================================


class TestTorchCompile(RefEagerTestDisabled, TestCase):
    def _run_compile_test(
        self,
        f,
        kernel,
        test_args: tuple,
        warmup_args: tuple | None = None,
        rtol: float | None = None,
        atol: float | None = None,
    ):
        """Run torch.compile test comparing eager vs compiled execution."""
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        kernel.reset()
        _ = kernel(*(warmup_args or test_args))

        # Clone tensor args for expected computation
        expected_args = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
        )
        expected = f(*expected_args)

        # Clone tensor args for compiled computation
        compiled_args = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
        )
        compiled_f = torch.compile(f, fullgraph=True, backend="inductor")
        actual = compiled_f(*compiled_args)

        # Verify no graph breaks occurred during compilation
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(len(graph_breaks), 0, f"Graph breaks: {dict(graph_breaks)}")

        # Compare results
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_add_kernel(self):
        """Test: basic addition kernel with prologue/epilogue ops."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = k_add(x, y)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_elementwise_kernel(self):
        """Test: multi-input elementwise ops with complex prologue/epilogue."""

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            a = x * 2.0
            b = y + z
            result = k_add(a, b)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y, z), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_multi_input_return_used(self):
        """Test: kernel with multiple inputs that mutates and returns one."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = k_add_inplace(x, scaled_y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y, scale), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs(self):
        """Test: kernel with multiple outputs."""

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            a = x.abs()
            b = y.neg()
            add_result, mul_result = k_add_mul(a, b)
            add_result = add_result * 2.0
            mul_result = mul_result + 1.0
            add_result = torch.relu(add_result) + 1.0
            mul_result = torch.relu(mul_result) + 1.0
            return add_result, mul_result

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(f, k_add_mul, (x, y), atol=1e-3, rtol=1e-3)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_keyword_arg_styles_all_keyword(self):
        """Test: all keyword argument passing."""

        def f(x, y, z):
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = k_add(y=y + z, x=x) * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y, z), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_keyword_arg_styles_mixed(self):
        """Test: mixed positional/keyword argument passing."""

        def f(x, y, z):
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = k_add(x, y=y + z) - 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y, z), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_default_params(self):
        """Test: kernel with default vs custom parameter values."""

        def f_with_default_scale(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            biased_x = x + bias
            # Inline scaling (default scale=2.0) before kernel
            result = k_add(biased_x, y * 2.0)
            result = result * 0.5
            return torch.relu(result) + 1.0

        def f_with_custom_scale(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            biased_x = x + bias
            # Inline scaling (custom scale=3.0) before kernel
            result = k_add(biased_x, y * 3.0)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        # Test with default scale
        self._run_compile_test(
            f_with_default_scale,
            k_add,
            (x, y, bias),
            warmup_args=(x, y),
            rtol=1e-3,
            atol=1e-3,
        )

        # Test with custom scale
        self._run_compile_test(
            f_with_custom_scale,
            k_add,
            (x, y, bias),
            warmup_args=(x, y),
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_constant_scalar_args(self):
        """Test: scalar constants in prologue/epilogue operations."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            # Apply scale and shift as prologue operations
            a = x * 2.5 + 1.0
            b = y * 2.5 + 1.0
            result = k_add(a, b)
            result = result - 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f, k_add, (x, y), warmup_args=(x, y), rtol=1e-3, atol=1e-3
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transposed_input(self):
        """Test: transposed (non-contiguous) tensor input."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            a = x.T * scale
            b = y.T
            result = k_add(a, b)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        y = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add, (x, y, scale), warmup_args=warmup)

    @unittest.expectedFailure
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_called_twice(self):
        """Test: same kernel called twice with different inputs.

        Expected failure: torch.dynamo raises 'Unsupported: mapping proxy
        affected by dictionary mutation' when the same Helion kernel is called
        twice in one function.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, scale: torch.Tensor
        ) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            a = k_add(scaled_x, y)
            b = k_add(a, z)
            result = b + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, y, z, scale), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_same_tensor_as_two_different_args(self):
        """Test: same tensor passed as two different arguments."""

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            scaled = x * 2.0 + bias
            result = k_add(scaled, scaled)
            result = result.mean(dim=-1)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add, (x, bias), rtol=1e-2, atol=1e-2)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic_add_mutation(self):
        """Test: mutation via atomic operations."""

        def f(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x * 0.5
            b = y.abs()
            result = k_atomic_add_to_out(a, b, out)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        out = torch.zeros(4, 8, device=DEVICE, dtype=torch.float32)
        warmup = (x, y, torch.zeros_like(x))
        self._run_compile_test(f, k_atomic_add_to_out, (x, y, out), warmup_args=warmup)

    @unittest.expectedFailure
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_indirect_output_alias(self):
        """Test: output is a slice/view of input (indirect alias with different shape).

        Expected failure: Indirect output aliasing (returning a slice of an
        input tensor) is not correctly tracked during torch.compile integration,
        causing incorrect results (AssertionError: Tensor-likes are not close).
        """

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = k_slice_mutate(x, scaled_y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(2, 4, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_slice_mutate, (x, y, scale), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_empty_tensor(self):
        """Test: tensors with zero-size dimensions."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_x = x * scale
            result = k_add(scaled_x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        # Test with zero-size first dimension
        x = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(0, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add, (x, y, scale), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_reduction_sum(self):
        """Test: kernel with reduction dimension (sum along axis)."""

        def f(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            weight = weight * 2.0
            scaled = x * weight
            # Helion kernel with reduction
            row_sums = k_sum_rows(scaled)
            result = row_sums.softmax(dim=0)
            return torch.relu(result) + 1.0

        x = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f, k_sum_rows, (x, weight), warmup_args=(x,), rtol=1e-3, atol=1e-3
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inline_triton_mutation(self):
        """Test: kernel using inline_triton marks all inputs as potentially mutated."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x.exp()
            b = y.log1p()
            # Helion kernel with inline_triton
            result = k_inline_add(a, b)
            result = result * 2.0
            return torch.relu(result) + 1.0

        x = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()
        y = torch.randn(32, device=DEVICE, dtype=torch.float32).abs()
        self._run_compile_test(f, k_inline_add, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_single_argument_kernel_mutation(self):
        """Test: kernel with mutation on first argument (tests mutation pattern)."""

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            scaled = (x * 2.0 + bias).contiguous()
            ones = torch.ones_like(scaled)
            # Helion kernel with mutation
            result = k_add_inplace(scaled, ones)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        warmup_x = torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(f, k_add_inplace, (x, bias), warmup_args=warmup)

    @unittest.expectedFailure
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    def test_signal_mutation(self):
        """Test: kernel using hl.signal correctly tracks mutation.

        Expected failure: hl.signal() generates code referencing
        'helion.runtime.triton_*_signal' but the generated Triton code does
        not import 'helion', causing NameError('helion is not defined').
        """

        def f(
            signal_pad: torch.Tensor, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            processed_x = x * y
            out, sig = k_signal(signal_pad, processed_x)
            out = out + 1.0
            out = torch.relu(out) + 1.0
            return out, sig

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        warmup_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, device=DEVICE, dtype=torch.float32)
        warmup = (warmup_pad, torch.randn(4, device=DEVICE, dtype=torch.float32))
        self._run_compile_test(f, k_signal, (signal_pad, x, y), warmup_args=warmup)

    @unittest.expectedFailure
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    def test_wait_mutation(self):
        """Test: kernel using hl.wait correctly tracks mutation.

        Expected failure: hl.wait() generates code referencing
        'helion.runtime.triton_*_signal' but the generated Triton code does
        not import 'helion', causing NameError('helion is not defined').
        """

        def f(
            signal_pad: torch.Tensor, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            processed_x = x + y
            out, sig = k_wait_update(signal_pad, processed_x)
            out = out - 0.5
            out = torch.relu(out) + 1.0
            return out, sig

        signal_pad = torch.ones(4, device=DEVICE, dtype=torch.int32)
        warmup_pad = torch.ones(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, device=DEVICE, dtype=torch.float32)
        warmup = (warmup_pad, torch.randn(4, device=DEVICE, dtype=torch.float32))
        self._run_compile_test(f, k_wait_update, (signal_pad, x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_view_input_mutate_different_view(self):
        """Test: input is a slice, mutate and return a different slice of the same base."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            # Pass slice as first arg, but also pass full tensor
            result = k_slice_return_other(x[:2, :4], scaled_y, x)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        warmup_x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (warmup_x[:2, :4], y, warmup_x)
        self._run_compile_test(
            f, k_slice_return_other, (x, y, scale), warmup_args=warmup
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_permute_view(self):
        """Test: output is permuted view of input."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = k_mutate_permuted(x, scaled_y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_mutate_permuted, (x, y, scale), warmup_args=(x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_alias_view_as_two_args(self):
        """Test: passing x and aliased view of x as two different arguments."""

        def f(a: torch.Tensor) -> torch.Tensor:
            a = a * 2.0
            x = a * 2
            y = x.view(-1).view(8, 8)  # Reshape to maintain 2D, aliased with x
            result = k_add_inplace(x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        a = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(8, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(8, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_inplace, (a,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliasing_inputs_used_after(self):
        """Test: view of input used after kernel mutation."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = x.view(-1)  # View before kernel
            ones = torch.ones_like(x)
            _ = k_add_inplace(x, ones)  # Mutate x
            result = y + 1  # Use view after - should see mutation
            return torch.relu(result) + 1.0

        x = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        warmup_x = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(f, k_add_inplace, (x,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_through_internal_view(self):
        """Test: kernel that creates a view inside the kernel and mutates through it."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            x = x - 1  # Prologue
            result = k_mutate_via_view(x)
            result = result * 2  # Epilogue
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(f, k_mutate_via_view, (x,))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_mutated_inputs(self):
        """Test: kernel that mutates multiple input tensors independently."""

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            z = z * 2.0
            result = k_mutate_two_return_new(x, y, z)
            result = result - 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_mutate_two_return_new, (x, y, z))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_detached_input(self):
        """Test: input is detached from grad-tracking tensor."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y = y * 2.0
            # Detach x before passing to kernel
            x_detached = x.detach()
            result = k_add(x_detached, y)
            result = result * 2.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32, requires_grad=True)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(f, k_add, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_module_forward_with_kernel(self):
        """Test: Helion kernel called inside nn.Module.forward()."""

        class SimpleModule(torch.nn.Module):
            def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.bias = torch.nn.Parameter(bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return k_add(x * self.weight, self.bias)

        def f(module, x):
            x = x * 2.0
            result = module(x)
            return torch.relu(result) + 1.0

        weight = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        module = SimpleModule(weight.clone(), bias.clone())
        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        warmup = (x, x)
        self._run_compile_test(f, k_add, (module, x), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_with_chained_prologue_epilogue_ops(self):
        """Test: mutation with prologue/epilogue operations."""

        def f(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            bias = bias * 2.0
            biased = x + bias
            ones = torch.ones_like(biased)
            result = k_add_inplace(biased, ones)
            result = result + 1.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        warmup_x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(f, k_add_inplace, (x, bias), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate(self):
        """Test: clone tensor, mutate clone, verify original unchanged."""

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            # Apply epilogue only to result, not to x (which we're verifying stayed unchanged)
            result = torch.relu(result) + 1.0
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_preallocated_output(self):
        """Test: kernel fills pre-allocated output tensor passed as argument."""

        def f(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            a = x * 2.0
            b = y + 1.0
            result = k_add_into_out(a, b, out)
            result = result * 0.5
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        out = torch.empty(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(f, k_add_into_out, (x, y, out))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs_same_storage(self):
        """Test: multiple outputs that share the same underlying storage."""

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            out_2d, out_1d = k_mutate_return_view(x, scaled_y)
            out_2d = out_2d + 1.0
            out_1d = out_1d * 2.0
            out_2d = torch.relu(out_2d) + 1.0
            out_1d = torch.relu(out_1d) + 1.0
            return out_2d, out_1d

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f, k_mutate_return_view, (x, y, scale), warmup_args=(x, y)
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliased_storage_different_shape(self):
        """Test: inputs share storage but have different shapes."""

        def f(base: torch.Tensor) -> torch.Tensor:
            base = base * 2.0
            # Create two views of base with different strides
            x = base[::2]  # Every other element: shape [16]
            y = base[1::2]  # Every other element offset by 1: shape [16]
            result = k_add(x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        base = torch.randn(32, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(16, device=DEVICE, dtype=torch.float16),
            torch.randn(16, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add, (base,), warmup_args=warmup)

    @unittest.expectedFailure
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_partial_tensor_mutation(self):
        """Test: mutate only a slice of tensor, rest remains unchanged.

        Expected failure: Partial tensor mutation through a slice is not
        correctly tracked, causing the mutated region to have incorrect values
        (AssertionError: Tensor-likes are not close).
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Take a slice, mutate it, return both slice result and full tensor
            x_slice = x[:2, :4]  # First 2 rows, first 4 cols
            result = k_add_inplace(x_slice, y)
            # Apply epilogue only to result, not to x (which shows mutation pattern)
            result = torch.relu(result) + 1.0
            return result, x  # x should have mutation in slice region

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(2, 4, device=DEVICE, dtype=torch.float16),
            torch.randn(2, 4, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_inplace, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_output_aliases_intermediate(self):
        """Test: output aliases tensor created inside the kernel."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            a = x * scale
            b = y + 1.0
            result = k_create_return_view(a, b)
            result = result * 2.0
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f, k_create_return_view, (x, y, scale), warmup_args=(x, y)
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inference_mode(self):
        """Test: kernel works correctly inside inference_mode context."""

        def f(x, y):
            x = x * 2.0
            y = y * 2.0
            z = x + 0.5  # prologue
            result = k_add_inplace(z, y)
            result = result * 2  # epilogue
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)

        with torch.inference_mode():
            self._run_compile_test(f, k_add_inplace, (x, y))

    @unittest.expectedFailure
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_identical_aliased_inputs(self):
        """Test: same tensor passed twice as different mutated arguments.

        Expected failure: Passing the same tensor as two different mutated
        arguments causes incorrectness - the aliasing is not detected and
        both mutations are applied incorrectly (AssertionError: Tensor-likes
        are not close, ~94% mismatch).
        """

        def f(z):
            z = z * 2.0
            # Pass same tensor as both x and y
            a = z.clone()
            result = k_add_to_both(a, a)
            # Apply epilogue only to result, not to z (which we're verifying stayed unchanged)
            result = torch.relu(result) + 1.0
            return result, z  # original should be unchanged

        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(4, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_to_both, (z,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_graph_input_is_view_with_kernel(self):
        """Test: graph input is a view, kernel operates on derived view."""

        def f(x, y):
            x = x * 2.0
            y = y * 2.0
            # x is already a view (passed from outside), take a 2D slice
            a = x[:2]  # 2D slice of view
            result = k_add_inplace(a.clone(), y[:2])
            return torch.relu(result) + 1.0

        base = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        x = base[1:]  # view with shape [3, 8]
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup = (
            torch.randn(2, 8, device=DEVICE, dtype=torch.float16),
            torch.randn(2, 8, device=DEVICE, dtype=torch.float16),
        )
        self._run_compile_test(f, k_add_inplace, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_return_assigned(self):
        """Test: mutation where return value is assigned to a variable."""

        def fn(x):
            x = x * 2.0
            x = x * 2
            ones = torch.ones_like(x)
            x = k_add_inplace(x, ones)
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE)
        warmup_x = torch.randn(64, device=DEVICE)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(fn, k_add_inplace, (x,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_return_discarded(self):
        """Test: mutation where return value is discarded (not assigned)."""

        def fn(x):
            x = x * 2.0
            x = x * 2
            ones = torch.ones_like(x)
            k_add_inplace(x, ones)  # return ignored; still mutates x
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE)
        warmup_x = torch.randn(64, device=DEVICE)
        warmup = (warmup_x, torch.ones_like(warmup_x))
        self._run_compile_test(fn, k_add_inplace, (x,), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_two_mutated(self):
        """Test: kernel that mutates two inputs and returns both."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x + 1, y - 1
            x, y = k_mutate_both(x, y)
            rx, ry = x * 2, y * 2
            rx = torch.relu(rx) + 1.0
            ry = torch.relu(ry) + 1.0
            return rx, ry

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_mutate_both, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_one_mutated(self):
        """Test: kernel that mutates one input."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y * 2
            x = k_add_inplace(x, y)
            result = x - 1
            return torch.relu(result) + 1.0

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_add_inplace, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mut_and_out(self):
        """Test: kernel that mutates input and also returns new tensor."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x + 1, y + 1
            x, out = k_mutate_with_out(x, y)
            x = torch.relu(x) + 1.0
            out = torch.relu(out) + 1.0
            return x, out

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_mutate_with_out, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_input_reused_after_call(self):
        """Test: mutated input is used after kernel call, but kernel returns a different tensor."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x = x + 1
            out = k_mutate_return_new(x, y)
            rx, rout = x + 1, out  # use mutated input after kernel
            rx = torch.relu(rx) + 1.0
            rout = torch.relu(rout) + 1.0
            return rx, rout

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_mutate_return_new, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_store_operation(self):
        """Test hl.store write operation."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y + 1
            x = k_store(x, y)
            result = x - 1
            return torch.relu(result) + 1.0

        x = torch.zeros(64, device=DEVICE)
        y = torch.randn(64, device=DEVICE)
        warmup_y = torch.randn(64, device=DEVICE)
        warmup = (torch.zeros(64, device=DEVICE), warmup_y)
        self._run_compile_test(fn, k_store, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic_add_operation(self):
        """Test hl.atomic_add write operation."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y * 2
            x = k_atomic_add(x, y)
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.zeros(64, device=DEVICE)
        y = torch.ones(64, device=DEVICE)
        warmup_y = torch.ones(64, device=DEVICE)
        warmup = (torch.zeros(64, device=DEVICE), warmup_y)
        self._run_compile_test(fn, k_atomic_add, (x, y), warmup_args=warmup)

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_no_mutation(self):
        """Test: pure function kernel with no input mutations."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x * 2, y * 2
            out = k_add(x, y)
            result = out + 1
            return torch.relu(result) + 1.0

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(fn, k_add, (x, y))

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_prologue_epilogue_tuple(self):
        """Test: prologue/epilogue with tuple (tensor, scalar) output."""
        kernel_scale = 2.0

        def f(x, out_bias):
            # Prologue: ops before kernel
            x_processed = torch.sigmoid(x) * 1.5
            out, info = k_scale_with_scalar_output(x_processed, kernel_scale)
            # Epilogue: ops after kernel
            out_processed = torch.relu(out) + out_bias
            return out_processed, info

        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        warmup = (torch.sigmoid(x) * 1.5, kernel_scale)
        self._run_compile_test(
            f,
            k_scale_with_scalar_output,
            (x, out_bias),
            warmup_args=warmup,
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_prologue_epilogue_single(self):
        """Test: prologue/epilogue with single tensor output."""

        def f(x, out_bias):
            # Prologue: ops before kernel
            x_processed = torch.tanh(x) * 2.0
            # Use k_add with processed input added to itself (equivalent to *2)
            out = k_add(x_processed, x_processed)
            # Epilogue: ops after kernel
            return torch.relu(out) + out_bias

        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        warmup = (torch.tanh(x) * 2.0, torch.tanh(x) * 2.0)
        self._run_compile_test(
            f, k_add, (x, out_bias), warmup_args=warmup, rtol=1e-3, atol=1e-3
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_prologue_epilogue_chained_ops(self):
        """Test: prologue/epilogue with chained ops on both sides."""
        kernel_scale = 2.0

        def f(x, out_bias, out_scale):
            # Prologue: chained ops before kernel
            x_processed = torch.relu(x) + 0.1
            out, info = k_scale_with_scalar_output(x_processed, kernel_scale)
            # Epilogue: chained ops after kernel
            out_relu = torch.relu(out)
            out_tanh = torch.tanh(out_relu)
            out_biased = out_tanh + out_bias
            out_scaled = out_biased * out_scale
            return out_scaled, info

        m, n = 64, 128
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out_scale = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        warmup = (torch.relu(x) + 0.1, kernel_scale)
        self._run_compile_test(
            f,
            k_scale_with_scalar_output,
            (x, out_bias, out_scale),
            warmup_args=warmup,
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_rms_norm_prologue_epilogue(self):
        """Test: prologue/epilogue with multi-output RMS norm kernel."""

        def f(x, weight, out_bias, res_bias):
            # Prologue: ops before kernel
            x_processed = torch.relu(x) + 0.5
            out, residual, info = k_rms_norm(x_processed, weight, 1e-5)
            # Epilogue: ops after kernel (different epilogue per output)
            return torch.relu(out) + out_bias, torch.sigmoid(residual) + res_bias, info

        m, n = 128, 256
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(n, device=DEVICE, dtype=torch.float32)
        out_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        res_bias = torch.randn(n, device=DEVICE, dtype=torch.float32)
        warmup = (torch.relu(x) + 0.5, weight, 1e-5)
        self._run_compile_test(
            f,
            k_rms_norm,
            (x, weight, out_bias, res_bias),
            warmup_args=warmup,
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transpose_then_view_to_3d_epilogue(self):
        """Test: prologue/epilogue with transpose and reshape view ops."""
        d1, d2, d3 = 8, 16, 32
        kernel_scale = 2.0

        def f(x, epilogue_bias):
            # Prologue view ops: mirror of epilogue (3D->2D then transpose)
            x_3d = x.T.reshape(d3, d1, d2)  # (m, n) -> (n, m) -> (d3, d1, d2)
            x_2d = x_3d.reshape(
                d3, d1 * d2
            ).T  # (d3, d1, d2) -> (d3, m) -> (m, d3) = (m, n)
            out, info = k_scale_with_scalar_output(x_2d, kernel_scale)
            # Epilogue view ops: transpose then 2D->3D
            out_t = out.T
            out_3d = out_t.reshape(d3, d1, d2)
            out_processed = torch.relu(out_3d) + epilogue_bias
            return out_processed, info

        m, n = d1 * d2, d3
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        epilogue_bias = torch.randn(d3, d1, d2, device=DEVICE, dtype=torch.float32)
        warmup = (x, kernel_scale)
        self._run_compile_test(
            f,
            k_scale_with_scalar_output,
            (x, epilogue_bias),
            warmup_args=warmup,
            rtol=1e-3,
            atol=1e-3,
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_dtype_promotion_simple(self):
        """Test: fp16 kernel with fp32 epilogue (simple dtype promotion)."""
        m, n = 64, 128

        def f(x, y):
            # Prologue: ops before kernel (stays fp16)
            x_processed = torch.relu(x) * 1.2
            # Use k_add with x_processed added to itself (equivalent to *2)
            out = k_add(x_processed, x_processed)
            # Epilogue: simple ops with dtype promotion
            out_sigmoid = torch.sigmoid(out)
            return out_sigmoid + y  # fp16 + fp32 -> fp32

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        y = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_proc = torch.relu(x) * 1.2
        warmup = (x_proc, x_proc)
        self._run_compile_test(
            f, k_add, (x, y), warmup_args=warmup, rtol=1e-3, atol=1e-3
        )

    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_dtype_promotion_chained(self):
        """Test: fp16 kernel with fp32 epilogue (chained dtype promotion)."""
        m, n = 64, 128

        def f(x, y):
            # Prologue: ops before kernel (stays fp16)
            x_processed = torch.sigmoid(x) + 0.1
            # Use k_add with x_processed added to itself (equivalent to *2)
            out = k_add(x_processed, x_processed)
            # Epilogue: chained ops after kernel
            out = torch.sigmoid(out)
            out = torch.relu(out)
            out = torch.tanh(out)
            return out * y  # fp16 * fp32 -> fp32

        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        y = torch.randn(n, device=DEVICE, dtype=torch.float32)
        x_proc = torch.sigmoid(x) + 0.1
        warmup = (x_proc, x_proc)
        self._run_compile_test(
            f, k_add, (x, y), warmup_args=warmup, rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
