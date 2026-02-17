from __future__ import annotations

import math
import os
import unittest

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize

import helion
from helion._compat import requires_torch_version
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import is_cpu
from helion._testing import onlyBackends
from helion._testing import skipIfCpu
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
def k_scale_two(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale two tensors of potentially different shapes."""
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    for tile in hl.tile(x.size()):
        out_x[tile] = x[tile] * 2.0
    for tile in hl.tile(y.size()):
        out_y[tile] = y[tile] * 3.0
    return out_x, out_y


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
def k_tensor_scalar_tensor(
    x: torch.Tensor, y: torch.Tensor, scale: float
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Return (tensor, scalar, tensor) - exercises multi-output with interspersed scalar."""
    out_x = torch.empty_like(x)
    out_y = torch.empty_like(y)
    for tile in hl.tile(x.size()):
        out_x[tile] = x[tile] * scale
    for tile in hl.tile(y.size()):
        out_y[tile] = y[tile] * scale
    return out_x, 7, out_y


@helion.kernel(autotune_effort="none")
def k_single_element_tuple(x: torch.Tensor) -> tuple[torch.Tensor]:
    """Return a single-element tuple."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * 2.0
    return (out,)


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


GLOBAL_SCALE_FACTOR = 2.5


@helion.kernel(autotune_effort="none")
def k_scale_with_global_var(x: torch.Tensor) -> torch.Tensor:
    """Scale x by a captured global variable."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * GLOBAL_SCALE_FACTOR
    return out


# =============================================================================
# Test Class
# =============================================================================


@onlyBackends(["triton"])
class TestTorchCompile(RefEagerTestDisabled, TestCase):
    def _run_compile_test(
        self,
        f,
        test_args: tuple,
        kernels: list,
        rtol: float | None = None,
        atol: float | None = None,
        expected_error: tuple[type[Exception], str] | None = None,
        dynamic: bool = False,
        allow_torch_compile_fusion: bool = False,
        compare_fn=None,
    ):
        """Run torch.compile test comparing eager vs compiled execution."""
        # Skip fusion tests on PyTorch < 2.11 or CPU backend
        if allow_torch_compile_fusion:
            if not requires_torch_version("2.11"):
                self.skipTest("torch.compile fusion requires PyTorch >= 2.11")
            if is_cpu():
                self.skipTest(
                    "torch.compile fusion not supported yet on Triton CPU backend"
                )

        # Reset specific kernels and configure fusion setting via env var
        if allow_torch_compile_fusion:
            os.environ["_WIP_DEV_ONLY_HELION_TORCH_COMPILE_FUSION"] = "1"
        else:
            os.environ.pop("_WIP_DEV_ONLY_HELION_TORCH_COMPILE_FUSION", None)
        for kernel in kernels:
            kernel.reset()

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()

        # Warmup by calling f (this warms up any kernels inside f)
        warmup_args = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
        )
        _ = f(*warmup_args)

        # Compile
        compiled_f = torch.compile(
            f, fullgraph=True, backend="inductor", dynamic=dynamic
        )

        if expected_error is not None:
            error_type, error_pattern = expected_error
            compiled_args = tuple(
                a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
            )
            with self.assertRaisesRegex(error_type, error_pattern):
                compiled_f(*compiled_args)
            return

        # Get expected result
        expected_args = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
        )
        expected = f(*expected_args)

        # Get actual result
        compiled_args = tuple(
            a.clone() if isinstance(a, torch.Tensor) else a for a in test_args
        )
        actual = compiled_f(*compiled_args)

        # Verify no graph breaks
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(len(graph_breaks), 0, f"Graph breaks: {dict(graph_breaks)}")

        # Compare results
        if compare_fn is not None:
            compare_fn(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_add_kernel(self, allow_torch_compile_fusion):
        """Test: basic addition kernel with prologue/epilogue ops."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = k_add(x, y)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_elementwise_kernel(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, z),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_multi_input_return_used(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs(self, allow_torch_compile_fusion):
        """Test: kernel with multiple differently-shaped outputs."""

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            scaled_x, scaled_y = k_scale_two(x, y)
            scaled_x = torch.relu(scaled_x) + 1.0
            scaled_y = torch.relu(scaled_y) + 1.0
            return scaled_x, scaled_y

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 16, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_scale_two],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_keyword_arg_styles_all_keyword(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, z),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_keyword_arg_styles_mixed(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, z),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_default_params(self, allow_torch_compile_fusion):
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
            (x, y, bias),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

        # Test with custom scale
        self._run_compile_test(
            f_with_custom_scale,
            (x, y, bias),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_constant_scalar_args(self, allow_torch_compile_fusion):
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
            f,
            (x, y),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transposed_input(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_called_twice(self, allow_torch_compile_fusion):
        """Test: same kernel called twice with different inputs."""

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
        self._run_compile_test(
            f,
            (x, y, z, scale),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_same_tensor_as_two_different_args(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, bias),
            kernels=[k_add],
            rtol=1e-2,
            atol=1e-2,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic_add_mutation(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, out),
            kernels=[k_atomic_add_to_out],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @unittest.skip("Correctness bug with indirect output aliasing")
    def test_indirect_output_alias(self, allow_torch_compile_fusion):
        """Test: output is a slice/view of input (indirect alias with different shape)."""

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
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_slice_mutate],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_empty_tensor(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_reduction_sum(self, allow_torch_compile_fusion):
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
            f,
            (x, weight),
            kernels=[k_sum_rows],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inline_triton_mutation(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_inline_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_single_argument_kernel_mutation(self, allow_torch_compile_fusion):
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
        torch.randn(8, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, bias),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    @unittest.skip("Generated code missing helion import for signal ops")
    def test_signal_mutation(self, allow_torch_compile_fusion):
        """Test: kernel using hl.signal correctly tracks mutation."""

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
        torch.zeros(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (signal_pad, x, y),
            kernels=[k_signal],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @skipIfNotCUDA()
    @unittest.skip("Generated code missing helion import for wait ops")
    def test_wait_mutation(self, allow_torch_compile_fusion):
        """Test: kernel using hl.wait correctly tracks mutation."""

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
        torch.ones(4, device=DEVICE, dtype=torch.int32)
        x = torch.randn(4, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (signal_pad, x, y),
            kernels=[k_wait_update],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_view_input_mutate_different_view(self, allow_torch_compile_fusion):
        """Test: passing both view and base as separate args raises error."""

        def f(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            result = k_slice_return_other(x[:2, :4], scaled_y, x)
            return torch.relu(result + 1.0) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(2, 4, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_slice_return_other],
            expected_error=(
                torch._dynamo.exc.InternalTorchDynamoError,
                "does not support multiple mutated arguments that share storage",
            )
            if allow_torch_compile_fusion
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_permute_view(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_mutate_permuted],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_alias_view_as_two_args(self, allow_torch_compile_fusion):
        """Test: passing x and aliased view of x as two different arguments."""

        def f(a: torch.Tensor) -> torch.Tensor:
            a = a * 2.0
            x = a * 2
            y = x.view(-1).view(8, 8)  # Reshape to maintain 2D, aliased with x
            result = k_add_inplace(x, y)
            result = result + 1.0
            return torch.relu(result) + 1.0

        a = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (a,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliasing_inputs_used_after(self, allow_torch_compile_fusion):
        """Test: view of input used after kernel mutation."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = x.view(-1)  # View before kernel
            ones = torch.ones_like(x)
            _ = k_add_inplace(x, ones)  # Mutate x
            result = y + 1  # Use view after - should see mutation
            return torch.relu(result) + 1.0

        x = torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        torch.randn(8, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_through_internal_view(self, allow_torch_compile_fusion):
        """Test: kernel that creates a view inside the kernel and mutates through it."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            x = x - 1  # Prologue
            result = k_mutate_via_view(x)
            result = result * 2  # Epilogue
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_mutate_via_view],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_mutated_inputs(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, z),
            kernels=[k_mutate_two_return_new],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_detached_input(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_module_forward_with_kernel(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (module, x),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_with_chained_prologue_epilogue_ops(
        self, allow_torch_compile_fusion
    ):
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
        torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, bias),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_preallocated_output(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y, out),
            kernels=[k_add_into_out],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_outputs_same_storage(self, allow_torch_compile_fusion):
        """Test: multiple outputs that share the same underlying storage raises error."""

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            scale = scale * 2.0
            scaled_y = y * scale
            out_2d, out_1d = k_mutate_return_view(x, scaled_y)
            out_2d = torch.relu(out_2d + 1.0) + 1.0
            out_1d = torch.relu(out_1d * 2.0) + 1.0
            return out_2d, out_1d

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        scale = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y, scale),
            kernels=[k_mutate_return_view],
            expected_error=(
                RuntimeError,
                r"Returning multiple outputs that share storage.*not yet supported",
            )
            if allow_torch_compile_fusion
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_aliased_storage_different_shape(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (base,),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @unittest.skip("Correctness bug with partial tensor mutation")
    def test_partial_tensor_mutation(self, allow_torch_compile_fusion):
        """Test: mutate only a slice of tensor, rest remains unchanged."""

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
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_output_aliases_intermediate(self, allow_torch_compile_fusion):
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
            f,
            (x, y, scale),
            kernels=[k_create_return_view],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_inference_mode(self, allow_torch_compile_fusion):
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
            self._run_compile_test(
                f,
                (x, y),
                kernels=[k_add_inplace],
                allow_torch_compile_fusion=allow_torch_compile_fusion,
            )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_identical_aliased_inputs(self, allow_torch_compile_fusion):
        """Test: same tensor passed twice as different mutated arguments raises error."""
        if not allow_torch_compile_fusion:
            self.skipTest(
                "Aliased mutation only detected with torch.compile fusion enabled"
            )

        def f(z):
            z = z * 2.0
            a = z.clone()
            result = k_add_to_both(a, a)
            return torch.relu(result) + 1.0, z

        z = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (z,),
            kernels=[k_add_to_both],
            expected_error=(
                torch._dynamo.exc.InternalTorchDynamoError,
                "does not support multiple mutated arguments that share storage",
            )
            if allow_torch_compile_fusion
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_graph_input_is_view_with_kernel(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_return_assigned(self, allow_torch_compile_fusion):
        """Test: mutation where return value is assigned to a variable."""

        def fn(x):
            x = x * 2.0
            x = x * 2
            ones = torch.ones_like(x)
            x = k_add_inplace(x, ones)
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE)
        torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_return_discarded(self, allow_torch_compile_fusion):
        """Test: mutation where return value is discarded (not assigned)."""

        def fn(x):
            x = x * 2.0
            x = x * 2
            ones = torch.ones_like(x)
            k_add_inplace(x, ones)  # return ignored; still mutates x
            result = x + 1
            return torch.relu(result) + 1.0

        x = torch.randn(64, device=DEVICE)
        torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_two_mutated(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_mutate_both],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_one_mutated(self, allow_torch_compile_fusion):
        """Test: kernel that mutates one input."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            y = y * 2
            x = k_add_inplace(x, y)
            result = x - 1
            return torch.relu(result) + 1.0

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mut_and_out(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_mutate_with_out],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_mutation_input_reused_after_call(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_mutate_return_new],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_store_operation(self, allow_torch_compile_fusion):
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
        torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_store],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_atomic_add_operation(self, allow_torch_compile_fusion):
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
        torch.ones(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_atomic_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_no_mutation(self, allow_torch_compile_fusion):
        """Test: pure function kernel with no input mutations."""

        def fn(x, y):
            x = x * 2.0
            y = y * 2.0
            x, y = x * 2, y * 2
            out = k_add(x, y)
            result = out + 1
            return torch.relu(result) + 1.0

        x, y = torch.randn(64, device=DEVICE), torch.randn(64, device=DEVICE)
        self._run_compile_test(
            fn,
            (x, y),
            kernels=[k_add],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_prologue_epilogue_tuple(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, out_bias),
            kernels=[k_scale_with_scalar_output],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_basic_prologue_epilogue_single(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, out_bias),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_prologue_epilogue_chained_ops(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, out_bias, out_scale),
            kernels=[k_scale_with_scalar_output],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_rms_norm_prologue_epilogue(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, weight, out_bias, res_bias),
            kernels=[k_rms_norm],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transpose_then_view_to_3d_epilogue(self, allow_torch_compile_fusion):
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
        self._run_compile_test(
            f,
            (x, epilogue_bias),
            kernels=[k_scale_with_scalar_output],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_dtype_promotion_simple(
        self, allow_torch_compile_fusion
    ):
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
        torch.relu(x) * 1.2
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_fp16_prologue_epilogue_dtype_promotion_chained(
        self, allow_torch_compile_fusion
    ):
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
        torch.sigmoid(x) + 0.1
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_original_twice_in_output(
        self, allow_torch_compile_fusion
    ):
        """Test: same unmutated original appears twice in output tuple.

        This tests that when the same FX node appears multiple times as a graph
        output, all references correctly read from the preserved original buffer.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # Return x twice - both should be unchanged
            return result, x, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_view_of_original_as_output(
        self, allow_torch_compile_fusion
    ):
        """Test: view of original is output alongside clone-then-mutate.

        This tests that views derived from the original also see the preserved
        pre-mutation value.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_view = x.view(-1)  # view of original
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # Both x and view of x should be unchanged
            return result, x, x_view

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_transform_original(self, allow_torch_compile_fusion):
        """Test: original undergoes computation before being output.

        This tests that computations on the original (like x + 1) use the
        pre-mutation value, not the mutated value.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # x + 1 should use pre-mutation value of x
            return result, x + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_chained_kernels(self, allow_torch_compile_fusion):
        """Test: two kernel calls, each with clone-then-mutate pattern.

        This tests complex graphs with multiple HOPs where each needs
        independent cloning to preserve originals.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # First kernel: mutate clone of x
            x_clone1 = x.clone()
            result1 = k_add_inplace(x_clone1, y)
            # Second kernel: mutate the result of first kernel
            ones = torch.ones_like(result1)
            result2 = k_add_inplace(result1, ones)
            result2 = torch.relu(result2) + 1.0
            # Both x and y should be unchanged
            return result2, x, y

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_of_view_then_mutate(self, allow_torch_compile_fusion):
        """Test: clone a view, mutate the clone, original unchanged.

        This tests that cloning a view and mutating the clone doesn't affect
        the original base tensor.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Create a view, clone it, mutate the clone
            x_view = x.view(-1)
            x_view_clone = x_view.clone()
            result = k_add_inplace(x_view_clone, y.view(-1))
            result = torch.relu(result) + 1.0
            # Original x should be unchanged
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_multiple_clones_same_tensor(self, allow_torch_compile_fusion):
        """Test: multiple clones of same tensor, each mutated independently.

        This tests that when the same original is cloned multiple times and
        each clone is mutated, the original remains unchanged.
        """

        def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            # Create two clones
            x_clone1 = x.clone()
            x_clone2 = x.clone()
            # Mutate first clone
            ones = torch.ones_like(x_clone1)
            result1 = k_add_inplace(x_clone1, ones)
            # Mutate second clone
            twos = ones * 2
            result2 = k_add_inplace(x_clone2, twos)
            result1 = torch.relu(result1) + 1.0
            result2 = torch.relu(result2) + 1.0
            # x should be unchanged
            return result1, result2, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_transposed(self, allow_torch_compile_fusion):
        """Test: clone transposed tensor, mutate clone, original unchanged.

        This tests non-contiguous tensor handling in the clone-then-mutate pattern.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Transpose x, clone the transpose, mutate
            x_t = x.T
            x_t_clone = x_t.clone()
            y_t = y.T
            result = k_add_inplace(x_t_clone, y_t)
            result = torch.relu(result) + 1.0
            # Original x should be unchanged (not transposed)
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_with_inplace_epilogue(self, allow_torch_compile_fusion):
        """Test: in-place PyTorch op on mutated result.

        This tests interaction between Helion mutation and PyTorch in-place ops.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = result.mul_(2.0)  # in-place PyTorch op
            result = torch.relu(result) + 1.0
            return result, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_result_used_twice(self, allow_torch_compile_fusion):
        """Test: mutated result is used in multiple computations.

        This tests that the mutated clone can be used multiple times while
        the original remains unchanged.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            # Use result in two different computations
            out1 = torch.relu(result) + 1.0
            out2 = result.sum()
            return out1, out2, x

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_identical_aliased_three_args(self, allow_torch_compile_fusion):
        """Test: same tensor passed as three different mutated arguments raises error."""
        if not allow_torch_compile_fusion:
            self.skipTest(
                "Aliased mutation only detected with torch.compile fusion enabled"
            )

        @helion.kernel(autotune_effort="none")
        def k_add_to_three(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            """Add 1 to x, 2 to y, 3 to z (which may all alias)."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
                y[tile] = y[tile] + 2
                z[tile] = z[tile] + 3
            return x

        def f(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            w = w * 2.0
            a = w.clone()
            result = k_add_to_three(a, a, a)
            return torch.relu(result) + 1.0, w

        w = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (w,),
            kernels=[k_add_to_three],
            expected_error=(
                torch._dynamo.exc.InternalTorchDynamoError,
                "does not support multiple mutated arguments that share storage",
            )
            if allow_torch_compile_fusion
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_original_reduction_as_output(
        self, allow_torch_compile_fusion
    ):
        """Test: reduction of original as output alongside mutation.

        This tests that reductions (like sum) on the original use the
        pre-mutation value.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # sum of x should use pre-mutation value
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_mutate_both_inputs_as_outputs(self, allow_torch_compile_fusion):
        """Test: clone x, mutate clone, return result along with both x and y unchanged.

        This tests that non-mutated inputs (y) are also correctly handled
        when mutated inputs have clone-then-mutate pattern.
        """

        def f(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            x_clone = x.clone()
            result = k_add_inplace(x_clone, y)
            result = torch.relu(result) + 1.0
            # Both x and y should be unchanged
            return result, x, y

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_multiple_chained_views_mutate(self, allow_torch_compile_fusion):
        """Test: clone then many chained view ops, mutate, original unchanged.

        This tests that clone detection correctly traces through multiple
        consecutive view operations: clone -> t -> contiguous -> view -> flatten -> mutate
        """

        @helion.kernel(autotune_effort="none")
        def k_add_inplace_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """1D in-place add."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Clone then multiple chained views
            x_clone = x.clone()
            x_t = x_clone.t()  # (8, 4)
            x_contig = x_t.contiguous()  # Makes a copy since t() is non-contiguous!
            x_view = x_contig.view(32)  # (32,)
            y_flat = y.flatten()
            result = k_add_inplace_1d(x_view, y_flat)
            result = torch.relu(result) + 1.0
            # x.sum() should use pre-mutation value of x
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace_1d],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_with_multiple_views_one_mutated(self, allow_torch_compile_fusion):
        """Test: clone with multiple views, only one is mutated.

        This tests that when a clone has multiple views and only one is mutated,
        the other view correctly sees the mutation (since both views share the
        same clone's storage in eager mode).

        The fix works by detecting whether the original tensor (before clone) has
        direct (non-view) uses in the output. If all uses of the original go through
        views (sibling views of the mutated input), we don't clone at Inductor level,
        allowing the mutation to propagate correctly to sibling views.
        """

        @helion.kernel(autotune_effort="none")
        def k_add_inplace_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """1D in-place add."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            return x

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Clone then create two different views
            x_clone = x.clone()
            x_flat = x_clone.flatten()  # view 1 - will be mutated
            x_transposed = x_clone.t()  # view 2 - not mutated, used in output
            y_flat = y.flatten()
            result = k_add_inplace_1d(x_flat, y_flat)
            result = torch.relu(result) + 1.0
            # x_transposed should use pre-mutation value (same as x.t() since clone was made)
            return result, x_transposed.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace_1d],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_two_clones_of_same_tensor_both_mutated(self, allow_torch_compile_fusion):
        """Test: create two clones of same tensor, pass both to kernel, both mutated.

        This tests that when two independent clones are made from the same tensor
        and both are passed to the kernel as different arguments, the original
        tensor remains unchanged and both clones receive independent mutations.
        """

        @helion.kernel(autotune_effort="none")
        def k_add_two_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add 1 to x and 2 to y (mutates both)."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
                y[tile] = y[tile] + 2
            return x + y

        def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            # Create two independent clones of x
            clone1 = x.clone()
            clone2 = x.clone()
            # Both clones are mutated
            result = k_add_two_inplace(clone1, clone2)
            result = torch.relu(result) + 1.0
            # x should be unchanged (both mutations happened to clones)
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_two_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_passed_to_two_kernels(self, allow_torch_compile_fusion):
        """Test: same clone passed to two different kernels in sequence.

        The first kernel mutates the clone, then a second kernel uses it.
        The original tensor should remain unchanged.

        Uses graph-level clone cache to share clones across kernels.
        """

        @helion.kernel(autotune_effort="none")
        def k_add_one(x: torch.Tensor) -> torch.Tensor:
            """Add 1 to x."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + 1
            return x

        @helion.kernel(autotune_effort="none")
        def k_mul_two(x: torch.Tensor) -> torch.Tensor:
            """Multiply x by 2."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] * 2
            return x

        def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            x_clone = x.clone()
            # First kernel mutates clone
            _ = k_add_one(x_clone)
            # Second kernel mutates same clone
            result = k_mul_two(x_clone)
            result = torch.relu(result) + 1.0
            # x.sum() should use pre-mutation value of x
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup1 = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        warmup2 = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        k_add_one.reset()
        k_mul_two.reset()
        _ = k_add_one(warmup1)
        _ = k_mul_two(warmup2)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_one, k_mul_two],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_clone_then_repeat_mutate(self, allow_torch_compile_fusion):
        """Test: clone then repeat (expansion), mutate.

        repeat(1,1) is a no-op that may be optimized away. Clone detection
        traces through no-op repeats to find the underlying clone.
        """

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            # Clone then repeat (repeat creates a new tensor, not a view)
            x_clone = x.clone()
            x_repeated = x_clone.repeat(1, 1)  # Same shape, but new tensor
            result = k_add_inplace(x_repeated, y)
            result = torch.relu(result) + 1.0
            # x.sum() should use pre-mutation value of x
            return result, x.sum()

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (False,))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_dynamic_shapes_basic(self, allow_torch_compile_fusion):
        """Test: kernel with dynamic shapes enabled."""

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = k_add(x, y)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_add],
            dynamic=True,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_no_return(self, allow_torch_compile_fusion):
        """Test: kernel with no return statement (pure mutation, returns None).

        This tests that when a kernel only mutates inputs and has no explicit
        return statement, the compilation handles it correctly.
        """

        @helion.kernel(autotune_effort="none")
        def k_mutate_no_return(x: torch.Tensor, y: torch.Tensor) -> None:
            """Mutate x in-place with no return."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            k_mutate_no_return(x, y)
            # Use x after mutation
            return torch.relu(x) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_mutate_no_return],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_optional_tensor_parameter(self, allow_torch_compile_fusion):
        """Test: kernel with Optional[torch.Tensor] parameter.

        Verifies that kernels with Optional[torch.Tensor] parameters work correctly
        with torch.compile. The typing import is added dynamically when Optional
        is detected in the generated code.
        """

        @helion.kernel(autotune_effort="none")
        def k_add_optional(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            """Add x + y, optionally adding bias."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                result = x[tile] + y[tile]
                if bias is not None:
                    result = result + bias[tile]
                out[tile] = result
            return out

        def f(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            bias = bias * 2.0
            result = k_add_optional(x, y, bias)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y, bias),
            kernels=[k_add_optional],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_same_kernel_different_shapes(self, allow_torch_compile_fusion):
        """Test: same kernel called twice with different input shapes.

        This tests that when the same Helion kernel is called multiple times with
        different input shapes, each instance gets unique inner Triton kernel names.
        Without proper name uniquification, the second inner kernel would overwrite
        the first in the generated code, causing incorrect results.
        """

        @helion.kernel(autotune_effort="none")
        def k_scale(x: torch.Tensor) -> torch.Tensor:
            """Scale x by 2."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Apply same kernel to two tensors of different shapes
            scaled1 = k_scale(x)  # 4x8
            scaled2 = k_scale(y)  # 2x4
            return scaled1.sum() + scaled2.sum()

        def warmup():
            # Warmup both shapes separately (kernel takes single tensor)
            k_scale.reset()
            k_scale(torch.randn(4, 8, device=DEVICE, dtype=torch.float32))
            k_scale(torch.randn(2, 4, device=DEVICE, dtype=torch.float32))

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 4, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_scale],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_captured_global_variable(self, allow_torch_compile_fusion):
        """Test: kernel using captured global variable from module scope.

        This tests that when a Helion kernel references a global variable defined
        in the module scope (GLOBAL_SCALE_FACTOR), the generated Inductor code
        correctly imports _source_module to resolve the captured variable.
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            return k_scale_with_global_var(x)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_scale_with_global_var],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    @unittest.skip("Correctness bug with overlapping views mutation")
    def test_overlapping_views_both_mutated(self, allow_torch_compile_fusion):
        """Test: two overlapping views of the same tensor, both mutated."""

        def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            view1 = x[:3, :]  # First 3 rows
            view2 = x[1:4, :]  # Rows 1-3 (overlaps with view1)
            ones = torch.ones_like(view1)
            result1 = k_add_inplace(view1, ones)
            twos = torch.ones_like(view2) * 2
            result2 = k_add_inplace(view2, twos)
            return result1, result2

        x = torch.randn(5, 4, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_add_inplace],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_none_in_tuple(self, allow_torch_compile_fusion):
        """Test: kernel that returns None as part of a tuple."""

        @helion.kernel(autotune_effort="none")
        def k_compute_with_none(
            x: torch.Tensor, flag: int
        ) -> tuple[torch.Tensor, None, int]:
            """Return (tensor, None, scalar)."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return out, None, flag * 2

        def f(x: torch.Tensor) -> tuple[torch.Tensor, None, int]:
            x = x * 2.0
            result, none_val, scalar = k_compute_with_none(x, 21)
            result = torch.relu(result) + 1.0
            return result, none_val, scalar

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_compute_with_none],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_none_first_in_tuple(self, allow_torch_compile_fusion):
        """Test: kernel that returns None as first element of tuple."""

        @helion.kernel(autotune_effort="none")
        def k_compute_none_first(
            x: torch.Tensor, flag: int
        ) -> tuple[None, torch.Tensor, int]:
            """Return (None, tensor, scalar)."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return None, out, flag * 2

        def f(x: torch.Tensor) -> tuple[None, torch.Tensor, int]:
            x = x * 2.0
            none_val, result, scalar = k_compute_none_first(x, 21)
            result = torch.relu(result) + 1.0
            return none_val, result, scalar

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_compute_none_first],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_tuple_of_scalars(self, allow_torch_compile_fusion):
        """Test: kernel returning only scalars works with fusion (constants inlined)."""

        @helion.kernel(autotune_effort="none")
        def k_two_scalars(x: torch.Tensor, a: int, b: int) -> tuple[int, int]:
            """Return two scalars based on input args."""
            for tile in hl.tile(x.size()):
                _ = x[tile]
            return a * 2, b * 3

        def f(x: torch.Tensor) -> tuple[int, int]:
            x = x * 2.0
            s1, s2 = k_two_scalars(x, 10, 20)
            return s1, s2

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_two_scalars],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_same_tensor_twice(self, allow_torch_compile_fusion):
        """Test: kernel returns the same tensor as multiple outputs raises error."""

        @helion.kernel(autotune_effort="none")
        def k_return_same_twice(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Add x+y and return the result twice."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] + y[tile]
            return out, out

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            result1, result2 = k_return_same_twice(x, y)
            return torch.relu(result1) + 1.0, torch.relu(result2) + 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_return_same_twice],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_same_local_twice_alias_input(
        self, allow_torch_compile_fusion
    ):
        """Test: kernel assigns input to local and returns local twice raises error."""

        @helion.kernel(autotune_effort="none")
        def k_alias_return_twice(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Mutate x, assign to local, return local twice."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            result = x
            return result, result

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            result1, result2 = k_alias_return_twice(x, y)
            return torch.relu(result1) + 1.0, torch.relu(result2) + 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_alias_return_twice],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_input_via_local_alias(self, allow_torch_compile_fusion):
        """Test: kernel assigns input to local variable and returns it."""

        @helion.kernel(autotune_effort="none")
        def k_local_alias(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Mutate x, assign to local, return the local."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] + y[tile]
            result = x
            return result  # noqa: RET504

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            out = k_local_alias(x, y)
            return torch.relu(out) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_local_alias],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_list_return(self, allow_torch_compile_fusion):
        """Test: kernel returns a list of tensors works with fusion."""

        @helion.kernel(autotune_effort="none")
        def k_return_list(x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
            """Add x+y and return both results in a list."""
            out1 = torch.empty_like(x)
            out2 = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out1[tile] = x[tile] + y[tile]
                out2[tile] = x[tile] - y[tile]
            return [out1, out2]

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            result_list = k_return_list(x, y)
            return torch.relu(result_list[0]) + 1.0, torch.relu(result_list[1]) + 2.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_return_list],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_nested_tuple_return(self, allow_torch_compile_fusion):
        """Test: kernel returns a nested tuple works with fusion."""

        @helion.kernel(autotune_effort="none")
        def k_nested(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """Return nested structure."""
            out1 = torch.empty_like(x)
            out2 = torch.empty_like(x)
            out3 = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out1[tile] = x[tile] + y[tile]
                out2[tile] = x[tile] - y[tile]
                out3[tile] = x[tile] * y[tile]
            return out1, (out2, out3)

        def f(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            a, (b, c) = k_nested(x, y)
            return torch.relu(a) + 1.0, torch.relu(b) + 2.0, torch.relu(c) + 3.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_nested],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_float_scalar(self, allow_torch_compile_fusion):
        """Test: kernel returns a float scalar (not int)."""

        @helion.kernel(autotune_effort="none")
        def k_float_scalar(x: torch.Tensor) -> tuple[torch.Tensor, float]:
            """Return tensor and a float scalar."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return out, math.pi

        def f(x: torch.Tensor) -> tuple[torch.Tensor, float]:
            x = x * 2.0
            result, scalar = k_float_scalar(x)
            result = torch.relu(result) + 1.0
            return result, scalar

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_float_scalar],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_scalar_output_used_in_computation(self, allow_torch_compile_fusion):
        """Test: scalar return value is correctly used in downstream tensor ops."""

        def f(x: torch.Tensor, scale: float) -> torch.Tensor:
            x = x * 2.0
            result, scalar_val = k_scale_with_scalar_output(x, scale)
            # Use the scalar output in downstream tensor computation
            return result + scalar_val

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, 2.0),
            kernels=[k_scale_with_scalar_output],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_tensor_scalar_tensor_output(self, allow_torch_compile_fusion):
        """Test: kernel returning (tensor, scalar, tensor) exercises multi-output with interspersed scalar."""

        def f(
            x: torch.Tensor, y: torch.Tensor, scale: float
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = x * 2.0
            y = y * 2.0
            result_x, scalar_val, result_y = k_tensor_scalar_tensor(x, y, scale)
            # Use all outputs: both tensors and the scalar
            return torch.relu(result_x) + scalar_val, torch.relu(result_y) + scalar_val

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 16, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y, 3.0),
            kernels=[k_tensor_scalar_tensor],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_single_element_tuple_return(self, allow_torch_compile_fusion):
        """Test: kernel returning (out,) works with fusion."""

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            (result,) = k_single_element_tuple(x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_single_element_tuple],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_list(self, allow_torch_compile_fusion):
        """Test: kernel returning a list works with fusion."""

        @helion.kernel(autotune_effort="none")
        def k_list_return(x: torch.Tensor) -> list[torch.Tensor]:
            """Return a list of tensors."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return [out]  # type: ignore[return-value]

        def f(x: torch.Tensor) -> torch.Tensor:
            [result] = k_list_return(x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_list_return],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_nested_tuple(self, allow_torch_compile_fusion):
        """Test: kernel returning nested tuple works with fusion."""

        @helion.kernel(autotune_effort="none")
        def k_nested_return(
            x: torch.Tensor,
        ) -> tuple[tuple[torch.Tensor], torch.Tensor]:
            """Return nested tuple."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return (out,), out  # type: ignore[return-value]

        def f(x: torch.Tensor) -> torch.Tensor:
            (inner,), outer = k_nested_return(x)
            return torch.relu(inner) + torch.relu(outer)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_nested_return],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_deep_nested_structure(self, allow_torch_compile_fusion):
        """Test: kernel returning ((tensor_a, tensor_b), scalar, tensor_c) exercises multi-level access paths."""

        @helion.kernel(autotune_effort="none")
        def k_deep_nested(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[tuple[torch.Tensor, torch.Tensor], int, torch.Tensor]:
            """Return a deeply nested structure with tensors and scalar."""
            out_a = torch.empty_like(x)
            out_b = torch.empty_like(x)
            out_c = torch.empty_like(y)
            for tile in hl.tile(x.size()):
                out_a[tile] = x[tile] * 2.0
                out_b[tile] = x[tile] * 3.0
            for tile in hl.tile(y.size()):
                out_c[tile] = y[tile] * 4.0
            return (out_a, out_b), 7, out_c  # type: ignore[return-value]

        def f(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            (a, b), scalar_val, c = k_deep_nested(x, y)
            return torch.relu(a) + torch.relu(b) + scalar_val, torch.relu(
                c
            ) + scalar_val

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        y = torch.randn(2, 16, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_deep_nested],
            atol=1e-3,
            rtol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_only_scalars(self, allow_torch_compile_fusion):
        """Test: kernel returning only scalars works with fusion (constants inlined)."""

        @helion.kernel(autotune_effort="none")
        def k_scalar_only(x: torch.Tensor) -> tuple[int, float]:
            """Return only scalars."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return 42, math.pi

        def f(x: torch.Tensor) -> tuple[int, float]:
            return k_scalar_only(x)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_scalar_only],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_scalar_return_depends_on_parameter(self, allow_torch_compile_fusion):
        """Test: scalar return that references a kernel parameter raises error with fusion."""

        @helion.kernel(autotune_effort="none")
        def k_param_scalar(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, float]:
            """Return tensor and a parameter-dependent scalar."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out, scale

        def f(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, float]:
            return k_param_scalar(x, scale)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, 2.0),
            kernels=[k_param_scalar],
            expected_error=(
                RuntimeError,
                r"Returning SymFloat values from a Helion kernel is not supported",
            )
            if allow_torch_compile_fusion
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_reassigns_parameter_to_new_tensor(self, allow_torch_compile_fusion):
        """Test: kernel reassigns parameter to new tensor and returns it."""

        @helion.kernel(autotune_effort="none")
        def k_reassign(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Reassign x to a new tensor and return it."""
            x = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                x[tile] = y[tile] * 2.0
            return x

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = k_reassign(x, y)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_reassign],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_local_variable_from_expression(
        self, allow_torch_compile_fusion
    ):
        """Test: kernel returns local variable assigned from expression."""

        @helion.kernel(
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def k_local_return(x: torch.Tensor) -> torch.Tensor:
            """Assign expression to local, return local."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return out.sum(dim=1)

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            result = k_local_return(x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_local_return],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_local_variable_from_control_flow(
        self, allow_torch_compile_fusion
    ):
        """Test: kernel returns local variable assigned in if-else control flow."""

        @helion.kernel(
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def k_control_flow(x: torch.Tensor, use_sum: bool) -> torch.Tensor:
            """Assign expression to local based on condition, return local."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            if use_sum:
                result = out.sum(dim=1)
            else:
                result = out.mean(dim=1)
            return result

        def f(x: torch.Tensor, use_sum: bool) -> torch.Tensor:
            x = x * 2.0
            result = k_control_flow(x, use_sum)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, True),
            kernels=[k_control_flow],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_multiple_return_statements_in_branches(
        self, allow_torch_compile_fusion
    ):
        """Test: kernel has multiple return statements in if-else branches raises error."""

        @helion.kernel(
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def k_multi_return(x: torch.Tensor, use_sum: bool) -> torch.Tensor:
            """Multiple return statements in branches."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            if use_sum:
                return out.sum(dim=1)
            return out.mean(dim=1)

        def f(x: torch.Tensor, use_sum: bool) -> torch.Tensor:
            x = x * 2.0
            result = k_multi_return(x, use_sum)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, True),
            kernels=[k_multi_return],
            expected_error=(
                RuntimeError,
                r"Return statements inside control flow.*not supported",
            )
            if allow_torch_compile_fusion
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_augmented_assignment_in_return(
        self, allow_torch_compile_fusion
    ):
        """Test: kernel uses augmented assignment and returns that variable.

        When a variable is defined with augmented assignment (e.g., result += 1)
        and then returned, the _find_variable_definitions function only handles
        ast.Assign, not ast.AugAssign. This could cause issues if the variable
        is not found in var_defs.
        """

        @helion.kernel(
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def k_augassign(x: torch.Tensor) -> torch.Tensor:
            """Use augmented assignment before return."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            # Start with initial value, then modify with augmented assignment
            result = out.sum(dim=1)  # 1D tensor
            return result + 1.0  # NOT augmented assignment, regular assignment

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            result = k_augassign(x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_augassign],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_annotated_assignment(self, allow_torch_compile_fusion):
        """Test: kernel uses annotated assignment (result: Tensor = ...).

        When a variable is defined with annotated assignment (PEP 526 style),
        the _find_variable_definitions function must handle ast.AnnAssign
        in addition to ast.Assign.
        """

        @helion.kernel(
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def k_annotated(x: torch.Tensor) -> torch.Tensor:
            """Use annotated assignment."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            # Annotated assignment
            result: torch.Tensor = out.sum(dim=1)  # 1D tensor
            return result

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            result = k_annotated(x)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_annotated],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_adjacent_non_overlapping_slices(
        self, allow_torch_compile_fusion
    ) -> None:
        """Test: multiple views of same base raises error."""
        if not allow_torch_compile_fusion:
            self.skipTest(
                "Overlapping view mutation only detected with torch.compile fusion enabled"
            )

        @helion.kernel(
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Mutate a and b (adjacent slices of same base), return a+b."""
            for tile in hl.tile(a.size()):
                a[tile] = a[tile] * 2.0
            for tile in hl.tile(b.size()):
                b[tile] = b[tile] * 3.0
            return a.sum() + b.sum()

        def f(base: torch.Tensor) -> torch.Tensor:
            base = base * 2.0
            slice_a = base[:2, :]
            slice_b = base[2:4, :]
            result = k(slice_a, slice_b)
            return result + base.sum()

        base = torch.randn(4, 4, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (base,),
            kernels=[k],
            expected_error=(
                torch._dynamo.exc.InternalTorchDynamoError,
                "does not support multiple mutated arguments that share storage",
            )
            if allow_torch_compile_fusion
            else None,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_transposed_input_through_kernel_chain(self, allow_torch_compile_fusion):
        """Test: chain of kernels with transposed intermediate tensors.

        This test verifies that transposed (non-contiguous) tensor inputs are
        handled correctly when compiled through torch.compile. The kernel must
        preserve the input strides in the output tensor layout.
        """

        @helion.kernel(autotune_effort="none")
        def k_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # First kernel on transposed input
            x_t = x.T
            result = k_add(x_t, y.T)
            # Second kernel on result
            result = k_scale(result, 2.0)
            return result.T

        def warmup():
            k_add.reset()
            k_scale.reset()
            warmup_x = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
            warmup_y = torch.randn(8, 4, device=DEVICE, dtype=torch.float16)
            k_add(warmup_x, warmup_y)
            k_scale(warmup_x, 2.0)

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_scale, k_add],
            rtol=1e-2,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_scalar_first_then_aliased_tensor_output(self, allow_torch_compile_fusion):
        """Test: kernel returns (scalar, aliased_tensor).

        This tests the edge case in multi-output handling where the first output
        is a scalar (make_layout returns None) and the second is an aliased tensor.
        The fallback layout logic might have issues here.
        """

        @helion.kernel(autotune_effort="none")
        def k_scalar_and_mutate(
            x: torch.Tensor, scale: float
        ) -> tuple[int, torch.Tensor]:
            """Return a scalar and the mutated input tensor."""
            for tile in hl.tile(x.size()):
                x[tile] = x[tile] * scale
            return 99, x  # Scalar first, aliased tensor second

        def f(x: torch.Tensor) -> tuple[int, torch.Tensor]:
            scalar, result = k_scalar_and_mutate(x, 2.0)
            return scalar, result + 0.5

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_scalar_and_mutate],
            rtol=1e-3,
            atol=1e-3,
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_tuple_input(self, allow_torch_compile_fusion):
        """Test: kernel with tuple of tensors as input."""

        @helion.kernel(autotune_effort="none")
        def k_sum_tuple(tensors: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            """Sum two tensors from a tuple."""
            out = torch.empty_like(tensors[0])
            for tile in hl.tile(tensors[0].size()):
                out[tile] = tensors[0][tile] + tensors[1][tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = k_sum_tuple((x, y))
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_sum_tuple],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_constexpr_parameter(self, allow_torch_compile_fusion):
        """Test: kernel with hl.constexpr parameter.

        Tests that kernels with compile-time constant parameters are
        correctly handled by torch.compile.
        """

        @helion.kernel(autotune_effort="none")
        def k_scale_constexpr(x: torch.Tensor, scale: hl.constexpr) -> torch.Tensor:
            """Scale x by a compile-time constant."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        def f(x: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            # Pass 3.0 as constexpr parameter
            result = k_scale_constexpr(x, 3.0)
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_scale_constexpr],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_with_dict_input(self, allow_torch_compile_fusion):
        """Test: kernel with dict of tensors as input."""

        @helion.kernel(autotune_effort="none")
        def k_sum_dict(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
            """Sum two tensors from a dict."""
            out = torch.empty_like(tensors["a"])
            for tile in hl.tile(tensors["a"].size()):
                out[tile] = tensors["a"][tile] + tensors["b"][tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x * 2.0
            y = y * 2.0
            result = k_sum_dict({"a": x, "b": y})
            return torch.relu(result) + 1.0

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x, y),
            kernels=[k_sum_dict],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_kernel_returns_string(self, allow_torch_compile_fusion):
        """Test: kernel that returns a string as part of a tuple."""
        if not allow_torch_compile_fusion:
            self.skipTest(
                "String return type only detected with torch.compile fusion enabled"
            )

        @helion.kernel(
            autotune_effort="none",
            ignore_warnings=[helion.exc.TensorOperationInWrapper],
        )
        def k_returns_string(x: torch.Tensor) -> tuple[torch.Tensor, str]:
            """Return tensor and string."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return out, "hello"

        def f(x: torch.Tensor) -> tuple[torch.Tensor, str]:
            return k_returns_string(x)

        # Custom compare needed because torch.testing.assert_close doesn't support str values.
        def compare(actual, expected):
            self.assertEqual(len(actual), len(expected))
            torch.testing.assert_close(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        self._run_compile_test(
            f,
            (x,),
            kernels=[k_returns_string],
            allow_torch_compile_fusion=allow_torch_compile_fusion,
            compare_fn=compare,
        )

    @parametrize("allow_torch_compile_fusion", (True, False))
    @skipIfCpu("torch.compile fusion not supported on Triton CPU backend")
    @skipIfRocm("torch.compile missing kernel metadata on ROCm")
    @skipIfTileIR("torch.compile missing kernel metadata on tileir")
    def test_symint_return_from_tensor_shape(self, allow_torch_compile_fusion):
        """Test: kernel returning SymInt (tensor shape) with dynamic shapes."""
        if not allow_torch_compile_fusion:
            self.skipTest("Only testing with torch.compile fusion enabled")
        if not requires_torch_version("2.11"):
            self.skipTest("torch.compile fusion requires PyTorch >= 2.11")
        os.environ["_WIP_DEV_ONLY_HELION_TORCH_COMPILE_FUSION"] = "1"

        @helion.kernel(autotune_effort="none", static_shapes=False)
        def k_return_size(x: torch.Tensor) -> tuple[torch.Tensor, int]:
            """Return a computed tensor and x.size(0) as a SymInt scalar."""
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * 2.0
            return out, x.size(0)

        def f(x: torch.Tensor) -> torch.Tensor:
            out, n = k_return_size(x)
            return out + n

        k_return_size.reset()
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()

        # Warmup
        x0 = torch.randn(4, 8, device=DEVICE, dtype=torch.float16)
        _ = f(x0.clone())

        compiled_f = torch.compile(f, fullgraph=True, backend="inductor", dynamic=True)

        # Test with multiple shapes to exercise dynamic SymInt return values
        for nrows in (4, 16, 32):
            x = torch.randn(nrows, 8, device=DEVICE, dtype=torch.float16)
            expected = f(x.clone())
            actual = compiled_f(x.clone())
            torch.testing.assert_close(actual, expected)


instantiate_parametrized_tests(TestTorchCompile)


@onlyBackends(["triton"])
class TestMakeFxSymbolicTracing(RefEagerTestDisabled, TestCase):
    def test_hop_preserves_symbolic_shapes(self):
        """Verify _trace_hop_proxy preserves symbolic shapes as FX Node references.

        When helion_kernel_wrapper_mutation is called inside make_fx with
        tracing_mode="symbolic", the output_spec may contain SymInts from
        FakeTensor shapes. _trace_hop_proxy must convert these to FX Node
        references so downstream passes see correct symbolic relationships.
        """
        if not requires_torch_version("2.11"):
            self.skipTest("HOP infrastructure requires PyTorch >= 2.11")

        from torch.fx import Node as FxNode
        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
        from torch.fx.experimental.proxy_tensor import make_fx

        from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table
        from helion._compiler._dynamo.higher_order_ops import (
            helion_kernel_wrapper_mutation,
        )

        helion_kernel_side_table.reset_table()
        kernel_idx = helion_kernel_side_table.add_kernel(k_add)

        def call_hop(x, y):
            with disable_proxy_modes_tracing():
                fake_out = torch.empty_like(x)
            output_spec = {
                "leaf_specs": [
                    {
                        "type": "tensor",
                        "shape": list(fake_out.shape),
                        "stride": list(fake_out.stride()),
                        "dtype": fake_out.dtype,
                        "device": str(fake_out.device),
                    },
                    {"type": "scalar", "scalar_value": x.size(0)},
                    {"type": "scalar", "scalar_value": 42},
                ],
                "tree_spec_str": "",
            }
            return helion_kernel_wrapper_mutation(
                kernel_idx=kernel_idx,
                constant_args={},
                tensor_args={"x": x, "y": y},
                output_spec=output_spec,
            )

        x = torch.randn(4, 8, device=DEVICE)
        y = torch.randn(4, 8, device=DEVICE)
        gm = make_fx(call_hop, tracing_mode="symbolic")(x, y)

        hop_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is helion_kernel_wrapper_mutation
        ]
        self.assertEqual(len(hop_nodes), 1)
        node = hop_nodes[0]

        specs = node.kwargs["output_spec"]["leaf_specs"]
        tensor_spec = specs[0]

        self.assertTrue(
            any(isinstance(s, FxNode) for s in tensor_spec["shape"]),
            "Expected symbolic dimensions as FX Node references in shape",
        )
        self.assertIsInstance(specs[1]["scalar_value"], FxNode)
        self.assertEqual(specs[2]["scalar_value"], 42)

        hop_val = node.meta["val"]
        self.assertTrue(
            all(isinstance(s, torch.SymInt) for s in hop_val[0].shape),
        )


if __name__ == "__main__":
    unittest.main()
